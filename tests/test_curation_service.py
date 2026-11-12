"""curation.service 沉淀编排测试（fake 提取器 + 临时库/临时知识库）。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from src.curation import markdown as cm
from src.curation import service as svc
from src.curation.llm import (
    CurationExtractError,
    CurationExtraction,
    EntityExtraction,
)
from src.persistence import database as db
from src.rag import knowledge_contract as kc
from src.rag.lint import lint_tenant_knowledge
from src.rag.wiki_schema import parse_frontmatter

pytestmark = pytest.mark.usefixtures("isolated_db")

TENANT = db.DEFAULT_TENANT_ID


# ── fixtures ─────────────────────────────────────────────────────────


class _FakeExtractor:
    """按 source_id 返回脚本化提取结果，可抛异常。"""

    def __init__(self, results: dict[str, CurationExtraction | Exception]) -> None:
        self.results = results
        self.calls: list[str] = []

    async def __call__(self, document_text, source_id, existing_entities):
        self.calls.append(source_id)
        result = self.results[source_id]
        if isinstance(result, Exception):
            raise result
        return result


@pytest_asyncio.fixture
async def knowledge_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    fake = SimpleNamespace(
        knowledge_dir=tmp_path / "knowledge",
        data_dir=tmp_path / "data",
        wiki_curation_max_chars=60000,
        wiki_curation_timeout_seconds=30.0,
    )
    monkeypatch.setattr(kc, "get_settings", lambda: fake)
    monkeypatch.setattr(svc, "get_settings", lambda: fake)
    root = kc.tenant_root(TENANT)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_doc(root: Path, source_id: str, text: str) -> None:
    path = root / source_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _extraction(*entities: EntityExtraction) -> CurationExtraction:
    return CurationExtraction(
        summary="文档摘要", key_points=["要点一", "要点二"], entities=list(entities)
    )


def _entity(name: str, *, etype: str = "concept", aliases=None,
            contradiction: str | None = None) -> EntityExtraction:
    return EntityExtraction(
        name=name, type=etype, aliases=aliases or [],
        summary=f"{name}概述", points=[f"{name}事实"], contradiction=contradiction,
    )


async def _slugs() -> set[str]:
    return {p["slug"] for p in await db.list_wiki_pages(TENANT)}


async def _fm(slug: str) -> dict:
    page = await db.get_wiki_page(TENANT, slug)
    assert page is not None
    fm, _ = parse_frontmatter(page["content"])
    return fm


# ── TR-4.1 首篇沉淀：摘要页 + 实体页 ────────────────────────────────


async def test_first_curation_creates_pages_and_mirrors(knowledge_root: Path) -> None:
    _write_doc(knowledge_root, "policies.md", "退款期限 7 天。")
    extractor = _FakeExtractor({
        "policies.md": _extraction(_entity("退款政策", aliases=["退款"]),
                                   _entity("财务部", etype="org")),
    })

    stats = await svc.curate_document(TENANT, "policies.md", extractor=extractor)

    slugs = await _slugs()
    assert stats.doc_slug == "doc-policies"
    assert "doc-policies" in slugs
    entity_slugs = {s for s in slugs if s.startswith("entity-")}
    assert len(entity_slugs) == 2
    # index/log 存在
    assert {"index", "log"} <= slugs
    # 镜像齐全
    mirror_dir = knowledge_root / "wiki"
    assert (mirror_dir / "doc-policies.md").exists()
    assert len(list(mirror_dir.glob("entity-*.md"))) == 2
    assert (mirror_dir / "index.md").exists() and (mirror_dir / "log.md").exists()
    # frontmatter
    fm = await _fm("doc-policies")
    assert fm["kind"] == "doc" and fm["sources"] == ["policies.md"]
    assert stats.created == 3 and stats.entities == 2
    # KB 未初始化，不刷索引
    assert stats.refreshed is False


# ── TR-4.2 幂等 ──────────────────────────────────────────────────────


async def test_repeat_curation_idempotent(knowledge_root: Path) -> None:
    _write_doc(knowledge_root, "policies.md", "正文")
    extractor = _FakeExtractor({"policies.md": _extraction(_entity("退款政策"))})

    await svc.curate_document(TENANT, "policies.md", extractor=extractor)
    slugs_after_first = await _slugs()
    revisions_after_first = len(await db.list_wiki_revisions(TENANT, "doc-policies"))

    await svc.curate_document(TENANT, "policies.md", extractor=extractor)
    slugs_after_second = await _slugs()

    assert slugs_after_first == slugs_after_second
    revisions_after_second = len(await db.list_wiki_revisions(TENANT, "doc-policies"))
    assert revisions_after_second == revisions_after_first + 1
    assert extractor.calls == ["policies.md", "policies.md"]


# ── TR-4.3 跨文档同实体别名合并 ─────────────────────────────────────


async def test_same_entity_merged_across_documents(knowledge_root: Path) -> None:
    _write_doc(knowledge_root, "a.md", "文档A")
    _write_doc(knowledge_root, "b.md", "文档B")

    extractor = _FakeExtractor({
        "a.md": _extraction(_entity("退款政策", aliases=["退款"])),
        "b.md": _extraction(_entity("退款新规", aliases=["退款", "退款政策"])),
    })

    await svc.curate_document(TENANT, "a.md", extractor=extractor)
    await svc.curate_document(TENANT, "b.md", extractor=extractor)

    # 只应有一个实体页（命中别名"退款"）
    all_pages = await db.list_wiki_pages(TENANT)
    entity_slugs = []
    for row in all_pages:
        fm = await _fm(row["slug"])
        if fm.get("kind") == "entity":
            entity_slugs.append(row["slug"])
    assert len(entity_slugs) == 1
    fm = await _fm(entity_slugs[0])
    assert set(fm["sources"]) == {"a.md", "b.md"}


# ── TR-4.4 矛盾标记不覆盖旧结论 + lint 检出 ─────────────────────────


async def test_contradiction_marked_and_linted(knowledge_root: Path) -> None:
    _write_doc(knowledge_root, "old.md", "旧制度：7 天")
    _write_doc(knowledge_root, "new.md", "新制度：30 天")

    extractor = _FakeExtractor({
        "old.md": _extraction(_entity("退款期限")),
        "new.md": _extraction(
            _entity("退款期限", contradiction="退款期限为 30 天")
        ),
    })
    await svc.curate_document(TENANT, "old.md", extractor=extractor)
    await svc.curate_document(TENANT, "new.md", extractor=extractor)

    pages = await db.list_wiki_pages(TENANT)
    entity_slug = next(s for s in (p["slug"] for p in pages) if s.startswith("entity-"))
    page = await db.get_wiki_page(TENANT, entity_slug)
    assert "退款期限为 30 天" in page["content"]
    assert cm.CONTRADICTION_OPEN in page["content"]
    assert cm.CONTRADICTION_CLOSE in page["content"]
    assert "`new.md`" in page["content"] and "`old.md`" in page["content"]
    assert cm.has_open_contradiction(page["content"])

    wiki_slugs = {p["slug"] for p in pages}
    report = lint_tenant_knowledge(knowledge_root, wiki_slugs)
    codes = [issue["code"] for issue in report.issues]
    assert "contradiction_open" in codes


# ── TR-4.5 凭据扫描拦截 ─────────────────────────────────────────────


async def test_secret_content_blocked(knowledge_root: Path) -> None:
    _write_doc(knowledge_root, "leak.md", "含密钥文档")
    extractor = _FakeExtractor({
        "leak.md": CurationExtraction(
            summary="AWS Access Key ID AKIAIOSFODNN7EXAMPLE 泄露",
            key_points=[], entities=[],
        ),
    })
    stats = await svc.curate_document(TENANT, "leak.md", extractor=extractor)
    assert stats.blocked == 1
    assert "doc-leak" not in await _slugs()
    # index/log 仍写入
    assert {"index", "log"} <= await _slugs()


# ── TR-4.6 提取失败零半成品 + log 记录 ──────────────────────────────


async def test_extract_failure_leaves_nothing(knowledge_root: Path) -> None:
    _write_doc(knowledge_root, "bad.md", "正文")
    extractor = _FakeExtractor({"bad.md": CurationExtractError("LLM 超时")})

    with pytest.raises(CurationExtractError):
        await svc.curate_document(TENANT, "bad.md", extractor=extractor)

    slugs = await _slugs()
    assert not any(s.startswith(("doc-", "entity-")) for s in slugs)
    # 仅 log 记录失败
    log_page = await db.get_wiki_page(TENANT, "log")
    assert log_page is not None
    assert "bad.md` 沉淀失败" in log_page["content"]
    assert "LLM 超时" in log_page["content"]


async def test_missing_and_wiki_source_rejected(knowledge_root: Path) -> None:
    extractor = _FakeExtractor({})
    with pytest.raises(FileNotFoundError):
        await svc.curate_document(TENANT, "nope.md", extractor=extractor)
    _write_doc(knowledge_root, "wiki/x.md", "镜像")
    with pytest.raises(ValueError):
        await svc.curate_document(TENANT, "wiki/x.md", extractor=extractor)


# ── TR-4.7 index/log 内容 ───────────────────────────────────────────


async def test_index_and_log_content(knowledge_root: Path) -> None:
    _write_doc(knowledge_root, "policies.md", "正文")
    extractor = _FakeExtractor({"policies.md": _extraction(_entity("退款"))})
    stats = await svc.curate_document(TENANT, "policies.md", extractor=extractor)

    index_page = await db.get_wiki_page(TENANT, "index")
    assert "(doc-policies.md)" in index_page["content"]
    assert "(index.md)" not in index_page["content"]
    assert "(log.md)" not in index_page["content"]
    ifm, _ = parse_frontmatter(index_page["content"])
    assert ifm["kind"] == "index"

    log_page = await db.get_wiki_page(TENANT, "log")
    assert "policies.md` 沉淀成功" in log_page["content"]
    assert f"实体 {stats.entities}" in log_page["content"]


# ── Review 整改回归：D-1 未裁决矛盾跨再沉淀迁移 ─────────────────────


async def test_open_contradiction_survives_re_curation_without_new_conflict(
    knowledge_root: Path,
) -> None:
    """三文档序列：矛盾产生后，后续再沉淀（LLM 不再报冲突）不得静默清除
    未裁决区块——lint 的 contradiction_open 必须持续报出，直至人工裁决。"""

    from src.rag.lint import lint_tenant_knowledge

    _write_doc(knowledge_root, "a.md", "文档A")
    _write_doc(knowledge_root, "b.md", "文档B")
    _write_doc(knowledge_root, "c.md", "文档C")

    # 第一篇建页；第二篇报矛盾（matched 路径）→ 实体页出现 open 区块
    ext = _FakeExtractor({
        "a.md": _extraction(_entity("退款期限")),
        "b.md": _extraction(_entity("退款期限", contradiction="B 文档称 30 天")),
        "c.md": _extraction(_entity("退款期限")),  # 无矛盾信号
    })
    await svc.curate_document(TENANT, "a.md", extractor=ext)
    await svc.curate_document(TENANT, "b.md", extractor=ext)
    entity_slug = next(s for s in await _slugs() if s.startswith("entity-"))
    page_after_b = await db.get_wiki_page(TENANT, entity_slug)
    assert cm.CONTRADICTION_OPEN in page_after_b["content"]

    # 第三篇（同名实体、无新矛盾）：open 区块必须被迁移保留
    await svc.curate_document(TENANT, "c.md", extractor=ext)
    page_after_c = await db.get_wiki_page(TENANT, entity_slug)
    assert cm.CONTRADICTION_OPEN in page_after_c["content"]
    assert "30 天" in page_after_c["content"]  # 旧矛盾主张可检索
    assert cm.has_open_contradiction(page_after_c["content"])

    # lint 仍报 contradiction_open（warning），未经人工裁决不得自动清除
    slugs = await _slugs()
    report = lint_tenant_knowledge(knowledge_root, slugs)
    assert "contradiction_open" in {
        i["code"] for i in report.issues if i["severity"] == "warning"
    }


# ── Review 整改回归：D-2 跨目录同名文档 slug 消歧 ───────────────────


async def test_same_stem_docs_get_distinct_slugs(knowledge_root: Path) -> None:
    """a/report.md 与 b/report.md 不得共享 doc-report 摘要页。"""

    _write_doc(knowledge_root, "a/report.md", "A 部门报告")
    _write_doc(knowledge_root, "b/report.md", "B 部门报告")
    ext = _FakeExtractor({
        "a/report.md": _extraction(_entity("实体甲")),
        "b/report.md": _extraction(_entity("实体乙")),
    })

    s1 = await svc.curate_document(TENANT, "a/report.md", extractor=ext)
    s2 = await svc.curate_document(TENANT, "b/report.md", extractor=ext)

    assert s1.doc_slug != s2.doc_slug
    slugs = await _slugs()
    assert {s1.doc_slug, s2.doc_slug} <= slugs
    # 各自来源独立，第一篇不被第二篇覆盖
    fm1 = await _fm(s1.doc_slug)
    assert "a/report.md" in fm1["sources"]
    assert "b/report.md" not in fm1["sources"]


# ── 回归：无 frontmatter 页面（fm=None）不得阻断沉淀 ────────────────


def test_entity_index_and_find_doc_page_skip_plain_body_pages() -> None:
    """parse_frontmatter 对纯正文页面返回 (None, body)，两个索引函数必须
    跳过而非 fm.get 崩溃（线上事故：人工无 fm 页面导致全部沉淀失败）。"""

    plain_page = {"slug": "onboarding", "title": "入职指南",
                  "content": "# 入职指南\n\n纯正文，没有 frontmatter。"}
    entity_page = {
        "slug": "entity-refund-policy", "title": "退款政策",
        "content": (
            "---\nkind: entity\naliases:\n  - 退款\n---\n"
            "# 退款政策\n\n7 个工作日内无理由退款。"
        ),
    }
    doc_page = {
        "slug": "doc-policy", "title": "政策文档",
        "content": "---\nkind: doc\nsources:\n  - policies/refund.md\n---\n# 政策\n",
    }
    pages = [plain_page, entity_page, doc_page]

    index = svc._entity_index(pages)
    assert set(index) == {"退款政策", "退款"}
    assert "onboarding" not in str(index)

    assert svc._find_doc_page(pages, "policies/refund.md") == doc_page
    # 纯正文页面不得被误判为 doc 页
    assert svc._find_doc_page([plain_page], "onboarding") is None


async def test_curation_succeeds_with_frontmatterless_page(
    knowledge_root: Path,
) -> None:
    """端到端：租户存在人工创建的无 frontmatter 页面时，自动沉淀仍成功，
    且该页面以 kind=None 进入 index 目录的"其他页面"分区。"""

    await db.create_wiki_page(
        tenant_id=TENANT, slug="onboarding", title="入职指南",
        content="# 入职指南\n\n这是人工编写的纯正文页面。",
        author_id=db.SYSTEM_USER_ID,
    )

    _write_doc(knowledge_root, "spec.md", "保修期 3 年。")
    extractor = _FakeExtractor({"spec.md": _extraction(_entity("保修政策"))})

    # 修复前这里抛 AttributeError: 'NoneType' object has no attribute 'get'
    stats = await svc.curate_document(TENANT, "spec.md", extractor=extractor)
    assert stats.doc_slug == "doc-spec"
    assert "doc-spec" in await _slugs()

    index_page = await db.get_wiki_page(TENANT, "index")
    assert index_page is not None
    assert "入职指南" in index_page["content"]
    ifm, _ = parse_frontmatter(index_page["content"])
    assert ifm["kind"] == "index"  # 系统页 frontmatter 未被无 fm 页面污染


async def test_append_log_rebuilds_frontmatter_when_stripped(
    knowledge_root: Path,
) -> None:
    """log 页 frontmatter 被人工剥离后，再沉淀应重建系统 fm 且保留旧正文。"""

    _write_doc(knowledge_root, "a.md", "文档A")
    ext = _FakeExtractor({"a.md": _extraction(_entity("实体甲"))})
    await svc.curate_document(TENANT, "a.md", extractor=ext)

    # 模拟人工把 log 页改成无 frontmatter 的纯正文
    log_before = await db.get_wiki_page(TENANT, "log")
    assert log_before is not None
    _, old_body = parse_frontmatter(log_before["content"])
    await db.update_wiki_page(
        tenant_id=TENANT, slug="log", title=log_before["title"],
        content=old_body, author_id=db.SYSTEM_USER_ID,
    )

    _write_doc(knowledge_root, "b.md", "文档B")
    ext.results["b.md"] = _extraction(_entity("实体乙"))
    await svc.curate_document(TENANT, "b.md", extractor=ext)

    log_after = await db.get_wiki_page(TENANT, "log")
    assert log_after is not None
    fm, body = parse_frontmatter(log_after["content"])
    assert fm is not None and fm["kind"] == "log"
    assert "a.md` 沉淀成功" in body  # 旧日志条目保留
    assert "b.md` 沉淀成功" in body  # 新条目正常追加
