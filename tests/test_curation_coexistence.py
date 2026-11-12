"""Task 8：system 作者修订历史可见 + 人工编辑与自动再沉淀共存。

验证知识治理决策"自动发布、靠修订历史回滚"：
1. 自动沉淀页面以 system 保留账号写入，历史接口可查；
2. 人工编辑沉淀页后再次自动沉淀，页面始终合法、lint 无 error，
   人工修订作为历史版本保留（可回滚）。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from src.api.wiki import WikiPageUpdate, page_history, update_page
from src.curation import service as svc
from src.curation.llm import CurationExtraction, EntityExtraction
from src.persistence import database as db
from src.rag import knowledge_contract as kc
from src.rag.lint import lint_tenant_knowledge
from src.rag.wiki_schema import parse_frontmatter, validate_frontmatter

pytestmark = pytest.mark.usefixtures("isolated_db")

TENANT = db.DEFAULT_TENANT_ID


@pytest_asyncio.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    settings = SimpleNamespace(
        knowledge_dir=tmp_path / "knowledge",
        data_dir=tmp_path / "data",
        wiki_curation_enabled=True,
        wiki_curation_timeout_seconds=30.0,
        wiki_curation_max_chars=60000,
    )
    from src.api import curation as curation_api

    monkeypatch.setattr(curation_api, "get_settings", lambda: settings)
    monkeypatch.setattr(svc, "get_settings", lambda: settings)
    monkeypatch.setattr(kc, "get_settings", lambda: settings)
    kc.tenant_root(TENANT).mkdir(parents=True, exist_ok=True)
    return settings


def _fake_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(text, source_id, existing):
        return CurationExtraction(
            summary="摘要",
            key_points=["要点"],
            entities=[
                EntityExtraction(
                    name="张三", type="person", aliases=["老张"],
                    summary="简介", points=["事实"],
                )
            ],
        )

    monkeypatch.setattr(svc, "extract_curated", fake)


async def _slugs() -> set[str]:
    return {p["slug"] for p in await db.list_wiki_pages(TENANT)}


# ── TR-8.1 system 作者修订可经历史接口查看 ─────────────────────────


async def test_system_author_revisions_visible(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    (kc.tenant_root(TENANT) / "report.md").write_text("# 报告\n正文", encoding="utf-8")
    _fake_extraction(monkeypatch)

    await svc.curate_document(TENANT, "report.md")

    viewer = {"id": "user-1", "email": "u@example.com", "tenant_id": TENANT}
    doc_revs = await page_history("doc-report", viewer)
    assert doc_revs, "文档摘要页应有修订记录"
    assert all(rev["author_id"] == "system" for rev in doc_revs)

    entity_slug = next(s for s in await _slugs() if s.startswith("entity-"))
    ent_revs = await page_history(entity_slug, viewer)
    assert ent_revs and ent_revs[0]["author_id"] == "system"


# ── TR-8.2 人工编辑后自动再沉淀：页面合法、lint 无 error、历史可回滚 ─


async def test_human_edit_then_re_curate_keeps_valid_page_and_history(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    (kc.tenant_root(TENANT) / "doc.md").write_text("# 文档\n正文", encoding="utf-8")
    _fake_extraction(monkeypatch)

    await svc.curate_document(TENANT, "doc.md")
    entity_slug = next(s for s in await _slugs() if s.startswith("entity-"))

    # 人工编辑该实体页：保留系统 frontmatter，正文追加人工核对备注
    human_id = await db.create_user(
        email="editor@example.com", password_hash="h", display_name="编辑员"
    )
    human = {"id": human_id, "email": "editor@example.com", "tenant_id": TENANT}
    before = await db.get_wiki_page(TENANT, entity_slug)
    assert before is not None
    human_content = before["content"].rstrip() + "\n\n## 人工核对\n已人工确认补充备注。\n"
    await update_page(
        entity_slug,
        WikiPageUpdate(title=before["title"], content=human_content,
                       summary="人工核对补充"),
        human,
    )

    # 同一原始文档再次自动沉淀（重新上传/手动触发场景）
    await svc.curate_document(TENANT, "doc.md")
    after = await db.get_wiki_page(TENANT, entity_slug)
    assert after is not None
    fm, _ = parse_frontmatter(after["content"])
    assert validate_frontmatter(fm) == []
    assert fm.get("kind") == "entity"

    # 人工修订与系统修订都在历史中 → 可回滚到人工版本
    revs = await db.list_wiki_revisions(TENANT, entity_slug)
    authors = {rev["author_id"] for rev in revs}
    assert human_id in authors and "system" in authors

    # 全部镜像通过 lint：无 error（允许 warning/info）
    report = lint_tenant_knowledge(kc.tenant_root(TENANT), await _slugs())
    errors = [i for i in report.issues if i["severity"] == "error"]
    assert errors == []
