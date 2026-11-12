"""Wiki frontmatter Schema 模块单元测试（知识治理 · Schema 约束纪律）。

覆盖：容错解析（含水平分隔线误伤）、严格校验、序列化往返、合并规则。
"""

from __future__ import annotations

from src.rag import wiki_schema as ws

# ── has_frontmatter_block / parse_frontmatter ────────────────────


def test_parse_no_frontmatter() -> None:
    text = "# 标题\n\n普通正文"
    fm, body = ws.parse_frontmatter(text)
    assert fm is None
    assert body == text
    assert not ws.has_frontmatter_block(text)


def test_parse_valid_frontmatter() -> None:
    text = "---\ntitle: 指南\ntags:\n  - 部署\n  - 运维\n---\n\n# 指南\n\n正文"
    fm, body = ws.parse_frontmatter(text)
    assert fm == {"title": "指南", "tags": ["部署", "运维"]}
    assert body.startswith("# 指南")
    assert ws.has_frontmatter_block(text)


def test_parse_crlf_frontmatter() -> None:
    text = "---\r\ntitle: 指南\r\n---\r\n\r\n正文"
    fm, body = ws.parse_frontmatter(text)
    assert fm == {"title": "指南"}
    assert body == "正文"


def test_parse_horizontal_rule_not_frontmatter() -> None:
    """正文以水平分隔线开头是合法 Markdown，不得误判为 frontmatter。"""

    cases = [
        # 闭合线之间的内容不是键值 dict（纯文本 / 列表）
        "---\n\n正文段落\n\n---\n\n更多正文",
        "---\n- 列表项一\n- 列表项二\n---\n正文",
        # 只有起始线没有闭合线
        "---\n普通正文没有闭合线",
        # YAML 语法错误
        "---\n: :\n---\n正文",
    ]
    for text in cases:
        fm, body = ws.parse_frontmatter(text)
        assert fm is None, text
        assert body == text, text


def test_parse_frontmatter_non_dict_yaml() -> None:
    fm, _ = ws.parse_frontmatter("---\n只是一个字符串\n---\n正文")
    assert fm is None


def test_parse_empty_frontmatter_block() -> None:
    """空 frontmatter 块（safe_load 为 None）视为无 frontmatter。"""

    fm, body = ws.parse_frontmatter("---\n---\n正文")
    assert fm is None
    assert body == "---\n---\n正文"


# ── validate_frontmatter ─────────────────────────────────────────


def test_validate_full_valid() -> None:
    meta = {
        "title": "部署指南",
        "summary": "一句话摘要",
        "tags": ["部署", "运维"],
        "version": "v2",
        "effective_date": "2026-01-01",
        "expiry_date": "2027-01-01",
    }
    assert ws.validate_frontmatter(meta) == []


def test_validate_unknown_keys_rejected() -> None:
    errors = ws.validate_frontmatter({"title": "t", "owner": "张三", "level": 3})
    assert any("owner" in e and "level" in e for e in errors)


def test_validate_field_violations() -> None:
    errors = ws.validate_frontmatter(
        {
            "title": "",  # 空标题
            "summary": "x" * 201,  # 超长
            "tags": ["ok", "x" * 31],  # 单项超长
            "version": "v" * 21,  # 超长
            "effective_date": "2026/01/01",  # 非 ISO
            "expiry_date": "2026-13-01",  # 非法日期
        }
    )
    assert len(errors) == 6


def test_validate_wrong_types() -> None:
    errors = ws.validate_frontmatter(
        {"tags": "not-a-list", "summary": 123, "title": 456}
    )
    assert len(errors) == 3


# ── normalize / dump / merge ─────────────────────────────────────


def test_normalize_strips_and_drops_empty() -> None:
    clean = ws.normalize_frontmatter(
        {"title": "  指南  ", "tags": [" a ", "", None, "b "], "summary": "   ",
         "unknown_key": "x"}
    )
    assert clean == {"title": "指南", "tags": ["a", "b"]}


def test_dump_parse_roundtrip() -> None:
    meta = {"title": "部署指南", "tags": ["部署"], "effective_date": "2026-01-01"}
    text = ws.dump_frontmatter(meta)
    assert text.startswith("---\n")
    fm, body = ws.parse_frontmatter(text + "\n正文")
    assert fm == meta
    assert body == "正文"


def test_merge_override_wins_and_filters_unknown() -> None:
    merged = ws.merge_frontmatter(
        {"title": "旧标题", "tags": ["a"], "junk": 1},
        {"title": "新标题", "summary": None},
    )
    assert merged == {"title": "新标题", "tags": ["a"]}


def test_merge_none_base() -> None:
    assert ws.merge_frontmatter(None, {"title": "t"}) == {"title": "t"}


# ── 自动沉淀层新键：kind / sources / aliases ─────────────────────


def test_validate_curation_keys_valid() -> None:
    meta = {
        "kind": "entity",
        "sources": ["policies.md", "wiki/x.md"],
        "aliases": ["退款政策", "Refund Policy"],
    }
    assert ws.validate_frontmatter(meta) == []


def test_validate_kind_violations() -> None:
    errors = ws.validate_frontmatter({"kind": "unknown"})
    assert any("kind" in e for e in errors)
    errors = ws.validate_frontmatter({"kind": 123})
    assert any("kind" in e for e in errors)


def test_validate_sources_aliases_violations() -> None:
    errors = ws.validate_frontmatter({"sources": "not-a-list"})
    assert any("sources" in e for e in errors)
    errors = ws.validate_frontmatter({"sources": ["a", "", "b"]})
    assert any("sources" in e and "空白" in e for e in errors)
    errors = ws.validate_frontmatter({"aliases": ["x" * 101]})
    assert any("aliases" in e for e in errors)
    errors = ws.validate_frontmatter({"sources": [f"s{i}" for i in range(31)]})
    assert any("sources" in e and "30" in e for e in errors)


def test_dump_parse_roundtrip_curation_keys() -> None:
    meta = {
        "kind": "doc",
        "sources": ["a.md"],
        "aliases": ["甲"],
        "title": "摘要页",
    }
    text = ws.dump_frontmatter(meta)
    fm, _ = ws.parse_frontmatter(text + "\n正文")
    assert fm == meta


def test_merge_sources_aliases_union_dedup() -> None:
    merged = ws.merge_frontmatter(
        {"sources": ["a.md", "b.md"], "aliases": ["X"]},
        {"sources": ["b.md", " c.md "], "aliases": ["Y"]},
    )
    assert merged["sources"] == ["a.md", "b.md", "c.md"]
    assert merged["aliases"] == ["X", "Y"]
