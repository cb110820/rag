"""手动沉淀 API 测试：单篇校验/统计、批量幂等、开关 409、跨租户隔离。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import HTTPException

from src.api.curation import (
    CurateRequest,
    curate_all_knowledge,
    curate_knowledge,
)
from src.curation import scheduler as sch
from src.curation import service as svc
from src.curation.llm import (
    CurationExtractError,
    CurationExtraction,
    EntityExtraction,
)
from src.persistence import database as db
from src.rag import knowledge_contract as kc

pytestmark = pytest.mark.usefixtures("isolated_db")

TENANT = db.DEFAULT_TENANT_ID
OTHER_TENANT = "tenant-b"

_USER = {"id": "user-1", "email": "u@example.com", "tenant_id": TENANT}


@pytest_asyncio.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    settings = SimpleNamespace(
        knowledge_dir=tmp_path / "knowledge",
        data_dir=tmp_path / "data",
        wiki_curation_enabled=True,
        wiki_curation_timeout_seconds=30.0,
        wiki_curation_max_chars=60000,
    )
    # API 端点、service、scheduler、契约层各自持有 get_settings 名字
    from src.api import curation as curation_api

    monkeypatch.setattr(curation_api, "get_settings", lambda: settings)
    monkeypatch.setattr(svc, "get_settings", lambda: settings)
    monkeypatch.setattr(sch, "get_settings", lambda: settings)
    monkeypatch.setattr(kc, "get_settings", lambda: settings)
    kc.tenant_root(TENANT).mkdir(parents=True, exist_ok=True)
    kc.tenant_root(OTHER_TENANT).mkdir(parents=True, exist_ok=True)
    return settings


def _fake_extract(monkeypatch: pytest.MonkeyPatch, *, entity: str = "张三") -> None:
    async def fake(text, source_id, existing):
        return CurationExtraction(
            summary=f"{source_id} 的摘要",
            key_points=["要点一"],
            entities=[
                EntityExtraction(
                    name=entity, type="person", aliases=[],
                    summary="人物简介", points=["事实一"],
                )
            ],
        )

    monkeypatch.setattr(svc, "extract_curated", fake)


def _write(rel: str, content: str = "# 标题\n正文内容") -> None:
    path = kc.tenant_root(TENANT) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def _slugs() -> set[str]:
    return {p["slug"] for p in await db.list_wiki_pages(TENANT)}


# ── TR-6.1 单篇：200 统计 / 404 / 400 / 跨租户 ─────────────────────


async def test_curate_single_ok(env, monkeypatch: pytest.MonkeyPatch) -> None:
    _write("report.md")
    _fake_extract(monkeypatch)

    stats = await curate_knowledge(CurateRequest(source_id="report.md"), _USER)

    assert stats.source_id == "report.md"
    assert stats.doc_slug == "doc-report"
    assert stats.created >= 2  # 摘要页 + 实体页
    assert stats.entities == 1
    slugs = await _slugs()
    assert "doc-report" in slugs
    assert any(s.startswith("entity-") for s in slugs)  # 中文实体名 → sha1 后缀 slug
    # 系统页存在但不进业务统计
    assert "index" in slugs and "log" in slugs


async def test_curate_missing_returns_404(env) -> None:
    with pytest.raises(HTTPException) as exc:
        await curate_knowledge(CurateRequest(source_id="nope.md"), _USER)
    assert exc.value.status_code == 404


async def test_curate_wiki_source_returns_400(env) -> None:
    with pytest.raises(HTTPException) as exc:
        await curate_knowledge(
            CurateRequest(source_id="wiki/doc-report.md"), _USER
        )
    assert exc.value.status_code == 400
    assert "wiki" in exc.value.detail


async def test_curate_cross_tenant_returns_404(env) -> None:
    # 在另一租户根放文件，尝试用 ../ 穿越访问
    (kc.tenant_root(OTHER_TENANT) / "secret.md").write_text("机密", encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        await curate_knowledge(
            CurateRequest(source_id="../tenant-b/secret.md"), _USER
        )
    assert exc.value.status_code == 404
    # 未在当前租户沉淀任何页面
    assert await _slugs() == set()


async def test_curate_unsupported_suffix_returns_400(env) -> None:
    _write("data.bin", "binary")
    with pytest.raises(HTTPException) as exc:
        await curate_knowledge(CurateRequest(source_id="data.bin"), _USER)
    assert exc.value.status_code == 400


# ── TR-6.2 批量：两篇成功 + 重复执行幂等 ────────────────────────────


async def test_curate_all_and_idempotent_rerun(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write("a.md")
    _write("sub/b.md")
    _fake_extract(monkeypatch)

    report = await curate_all_knowledge(_USER)
    assert report.total == 2
    assert report.succeeded == 2
    assert report.failed == 0
    assert {r.source_id for r in report.results} == {"a.md", "sub/b.md"}
    assert {r.status for r in report.results} == {"ok"}

    slugs_after_first = await _slugs()
    assert "doc-a" in slugs_after_first and "doc-b" in slugs_after_first
    # wiki/ 镜像不计入批量枚举：index/log 镜像虽为 .md 也不能被当作原始文档
    assert not any(
        r.source_id.startswith("wiki/") for r in report.results
    )

    # 第二次批量：页面集合不增（幂等），统计走 updated
    report2 = await curate_all_knowledge(_USER)
    assert report2.total == 2
    assert report2.succeeded == 2
    assert await _slugs() == slugs_after_first
    assert all(r.stats is not None and r.stats.updated >= 1 for r in report2.results)


async def test_curate_all_continues_after_extraction_failure(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write("good.md")
    _write("bad.md")

    async def fake(text, source_id, existing):
        if source_id == "bad.md":
            raise CurationExtractError("LLM 不可用")
        return CurationExtraction(summary="ok", key_points=[], entities=[])

    monkeypatch.setattr(svc, "extract_curated", fake)

    report = await curate_all_knowledge(_USER)
    assert report.total == 2
    assert report.succeeded == 1
    assert report.failed == 1
    by_source = {r.source_id: r for r in report.results}
    assert by_source["good.md"].status == "ok"
    assert by_source["bad.md"].status == "failed"
    assert "LLM 不可用" in by_source["bad.md"].detail
    # 失败篇无摘要页，成功篇正常沉淀
    slugs = await _slugs()
    assert "doc-good" in slugs and "doc-bad" not in slugs


async def test_curate_empty_document_skipped(env, monkeypatch) -> None:
    _write("empty.md", "   \n  ")

    async def fake(text, source_id, existing):
        raise AssertionError("空正文不应调用提取器")

    monkeypatch.setattr(svc, "extract_curated", fake)

    report = await curate_all_knowledge(_USER)
    assert report.total == 1
    assert report.skipped == 1
    assert report.results[0].status == "skipped"


# ── TR-6.3 开关关闭：两端口 409 + 零变更 ───────────────────────────


async def test_disabled_returns_409(env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.wiki_curation_enabled = False
    _write("report.md")
    _fake_extract(monkeypatch)

    with pytest.raises(HTTPException) as exc_single:
        await curate_knowledge(CurateRequest(source_id="report.md"), _USER)
    assert exc_single.value.status_code == 409

    with pytest.raises(HTTPException) as exc_batch:
        await curate_all_knowledge(_USER)
    assert exc_batch.value.status_code == 409

    assert await _slugs() == set()
