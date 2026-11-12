"""curation.markdown 纯函数测试：slug、页面模板、矛盾区块、index/log。"""

from __future__ import annotations

from src.curation import markdown as cm
from src.rag import knowledge_contract as kc
from src.rag.wiki_schema import validate_frontmatter

# ── slug ────────────────────────────────────────────────────────────


def test_doc_slug_basic_and_deterministic() -> None:
    slug = cm.doc_slug("policies.md")
    assert slug == "doc-policies"
    assert kc.is_valid_wiki_slug(slug)
    assert cm.doc_slug("policies.md") == slug  # 确定性


def test_doc_slug_subdir_and_sanitize() -> None:
    assert cm.doc_slug("sub dir/Report V2.md") == "doc-report-v2"


def test_doc_slug_non_ascii_uses_hash() -> None:
    slug = cm.doc_slug("规章制度.md")
    assert kc.is_valid_wiki_slug(slug)
    assert slug.startswith(cm.DOC_PREFIX)
    assert len(slug) <= 80


def test_entity_slug_chinese_and_collision_stable() -> None:
    s1 = cm.entity_slug("退款政策")
    s2 = cm.entity_slug("退款政策")
    assert s1 == s2 and s1.startswith(cm.ENTITY_PREFIX)
    assert kc.is_valid_wiki_slug(s1)
    assert cm.entity_slug("Refund Policy") == "entity-refund-policy"


def test_entity_slug_long_ascii_truncated() -> None:
    slug = cm.entity_slug("a" * 200)
    assert len(slug) <= 80
    assert kc.is_valid_wiki_slug(slug)


def test_normalize_entity_name_matching() -> None:
    assert cm.normalize_entity_name("X 概念") == cm.normalize_entity_name("x概念")
    assert cm.normalize_entity_name("ABC") == cm.normalize_entity_name(" a b c ")


def test_entity_aliases_dedup() -> None:
    aliases = cm.entity_aliases("X", ["X", " Y ", "", "Y"])
    assert aliases == ["X", "Y"]


# ── 页面模板 ─────────────────────────────────────────────────────────


def test_build_doc_page_valid_and_sourced() -> None:
    fm, body = cm.build_doc_page(
        "policies.md", "制度摘要", ["要点一", "", "要点二"], [("退款", "entity-refund")]
    )
    assert fm == {"kind": "doc", "sources": ["policies.md"], "summary": "制度摘要"}
    assert validate_frontmatter(fm) == []
    assert "要点一" in body and "要点二" in body
    assert "[退款](entity-refund.md)" in body
    assert "`policies.md`" in body


def test_build_entity_page_valid_and_sourced() -> None:
    fm, body = cm.build_entity_page(
        "X", "concept", ["别名X", "X"], "概述", ["p1"], ["a.md", "a.md", "b.md"]
    )
    assert fm["kind"] == "entity"
    assert fm["sources"] == ["a.md", "b.md"]
    assert fm["aliases"] == ["别名X"]  # 规范名不进 aliases
    assert validate_frontmatter(fm) == []
    assert "类型：概念" in body
    assert cm.entity_type_from_body(body) == "概念"


# ── 矛盾区块 ─────────────────────────────────────────────────────────


def test_contradiction_block_markers_and_idempotent() -> None:
    block = cm.wrap_contradiction("2026-11-09", "30 天", "new.md", "7 天", "old.md")
    assert cm.CONTRADICTION_OPEN in block and cm.CONTRADICTION_CLOSE in block
    assert "30 天" in block and "7 天" in block and "new.md" in block
    body = "旧结论：7 天"
    once = cm.append_contradiction(body, block)
    twice = cm.append_contradiction(once, block)
    assert once == twice  # 幂等
    assert cm.has_open_contradiction(once)
    resolved = once + "\n" + cm.CONTRADICTION_RESOLVED + "\n"
    assert not cm.has_open_contradiction(resolved)


# ── index / log ──────────────────────────────────────────────────────


def test_render_index_partitions() -> None:
    entries = [
        {"slug": "index", "title": "目录", "kind": "index"},
        {"slug": "log", "title": "日志", "kind": "log"},
        {"slug": "doc-a", "title": "文档A", "kind": "doc"},
        {"slug": "entity-x", "title": "X", "kind": "entity", "entity_type": "概念"},
        {"slug": "manual", "title": "人工页", "kind": None},
    ]
    fm, body = cm.render_index(entries)
    assert fm == {"kind": "index"}
    assert validate_frontmatter(fm) == []
    assert "(doc-a.md)" in body and "(entity-x.md)" in body
    assert "(index.md)" not in body and "(log.md)" not in body
    assert "### 概念" in body
    # 人工页归入其他
    assert "(manual.md)" in body


def test_append_log_history_preserved() -> None:
    _, body = cm.log_page_initial()
    e1 = cm.log_entry("2026-11-09 10:00:00", "a.md", True,
                      {"created": 2, "updated": 0, "entities": 1, "contradictions": 0})
    e2 = cm.log_entry("2026-11-09 10:05:00", "b.md", False, reason="LLM 超时")
    body = cm.append_log(cm.append_log(body, e1), e2)
    assert "a.md` 沉淀成功" in body
    assert "实体 1 / 矛盾 0" in body
    assert "b.md` 沉淀失败：LLM 超时" in body
    assert body.count("## 沉淀记录") == 1
