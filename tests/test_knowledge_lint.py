"""知识健康检查单元测试（知识治理 · Lint 保证健康）。

lint_tenant_knowledge 为纯函数：租户根目录直接传入临时目录，
无需 monkeypatch settings / 数据库。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from src.rag.lint import lint_tenant_knowledge

TODAY = date(2026, 6, 1)


def _make_mirror(root: Path, name: str, content: str) -> Path:
    mirror_dir = root / "wiki"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    path = mirror_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def _codes(result) -> list[str]:
    return [issue["code"] for issue in result.issues]


def test_clean_mirror_no_issues(tmp_path: Path) -> None:
    _make_mirror(
        tmp_path,
        "guide.md",
        "---\ntitle: 指南\nsummary: 摘要\n---\n\n# 指南\n\n正文",
    )
    result = lint_tenant_knowledge(tmp_path, {"guide"}, today=TODAY)
    assert result.issues == []
    assert result.checked_files == 1


def test_orphan_mirror(tmp_path: Path) -> None:
    _make_mirror(tmp_path, "orphan.md", "---\ntitle: 孤儿\n---\n\n正文")
    result = lint_tenant_knowledge(tmp_path, set(), today=TODAY)
    assert _codes(result) == ["mirror_orphan"]
    assert result.issues[0]["severity"] == "error"


def test_missing_mirror(tmp_path: Path) -> None:
    result = lint_tenant_knowledge(tmp_path, {"ghost"}, today=TODAY)
    assert _codes(result) == ["mirror_missing"]
    assert result.issues[0]["source_id"] == "wiki/ghost.md"


def test_frontmatter_missing_is_warning(tmp_path: Path) -> None:
    _make_mirror(tmp_path, "legacy.md", "# 旧版镜像无 frontmatter\n\n正文")
    result = lint_tenant_knowledge(tmp_path, {"legacy"}, today=TODAY)
    assert _codes(result) == ["frontmatter_missing"]
    assert result.issues[0]["severity"] == "warning"


def test_frontmatter_invalid_is_error(tmp_path: Path) -> None:
    _make_mirror(
        tmp_path, "bad.md", "---\ntitle: 页面\nowner: 张三\n---\n\n正文"
    )
    result = lint_tenant_knowledge(tmp_path, {"bad"}, today=TODAY)
    assert _codes(result) == ["frontmatter_invalid"]
    assert result.issues[0]["severity"] == "error"
    assert "owner" in result.issues[0]["message"]


def test_expired_and_future_content(tmp_path: Path) -> None:
    _make_mirror(
        tmp_path,
        "expired.md",
        "---\ntitle: 过期\nexpiry_date: '2020-01-01'\n---\n\n正文",
    )
    _make_mirror(
        tmp_path,
        "future.md",
        "---\ntitle: 未生效\neffective_date: '2999-01-01'\n---\n\n正文",
    )
    result = lint_tenant_knowledge(tmp_path, {"expired", "future"}, today=TODAY)
    assert sorted(_codes(result)) == ["content_expired", "content_future"]
    severities = {i["code"]: i["severity"] for i in result.issues}
    assert severities["content_expired"] == "warning"
    assert severities["content_future"] == "info"


def test_empty_body(tmp_path: Path) -> None:
    _make_mirror(tmp_path, "empty.md", "---\ntitle: 空页面\n---\n\n   \n")
    result = lint_tenant_knowledge(tmp_path, {"empty"}, today=TODAY)
    assert _codes(result) == ["content_empty"]


def test_invalid_naming_is_warning(tmp_path: Path) -> None:
    _make_mirror(tmp_path, "Bad Slug.md", "---\ntitle: x\n---\n\n正文")
    result = lint_tenant_knowledge(tmp_path, set(), today=TODAY)
    assert _codes(result) == ["mirror_naming"]
    assert result.issues[0]["severity"] == "warning"


def test_open_contradiction_is_warning(tmp_path: Path) -> None:
    """含未裁决矛盾区块（open 多于 resolved）→ contradiction_open warning。"""

    from src.curation import markdown as cm

    block = cm.wrap_contradiction(
        "2026-06-01", "新结论", "new.md", "旧结论", "old.md"
    )
    content = f"---\ntitle: 实体页\nkind: entity\n---\n\n# 实体页\n\n{block}"
    _make_mirror(tmp_path, "entity-x.md", content)
    result = lint_tenant_knowledge(tmp_path, {"entity-x"}, today=TODAY)
    assert _codes(result) == ["contradiction_open"]
    issue = result.issues[0]
    assert issue["severity"] == "warning"
    assert issue["source_id"] == "wiki/entity-x.md"


def test_resolved_contradiction_not_reported(tmp_path: Path) -> None:
    """矛盾经人工裁决（open 数 == resolved 数）后不再报告。"""

    from src.curation import markdown as cm

    content = (
        "---\ntitle: 实体页\nkind: entity\n---\n\n# 实体页\n\n"
        f"{cm.CONTRADICTION_OPEN}\n双方说法\n{cm.CONTRADICTION_RESOLVED}\n"
    )
    _make_mirror(tmp_path, "entity-y.md", content)
    result = lint_tenant_knowledge(tmp_path, {"entity-y"}, today=TODAY)
    assert "contradiction_open" not in _codes(result)


def test_multiple_issues_sorted_by_file(tmp_path: Path) -> None:
    _make_mirror(tmp_path, "orphan.md", "---\ntitle: a\n---\n\n正文")
    _make_mirror(tmp_path, "legacy.md", "无 frontmatter")
    result = lint_tenant_knowledge(tmp_path, {"legacy", "missing"}, today=TODAY)
    codes = set(_codes(result))
    assert codes == {"mirror_orphan", "frontmatter_missing", "mirror_missing"}
    # 受托管的合法 slug：orphan + legacy（orphan 属业务库缺失侧）
    assert result.checked_files == 2
