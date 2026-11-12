"""Wiki↔RAG 文件系统契约层单元测试。

只测路径/物化/守卫/重建逻辑，不触发嵌入模型加载；
request_refresh 用未初始化租户验证懒跳过分支。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag import knowledge_contract as kc


@pytest.fixture()
def kb_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把知识库根目录重定向到临时目录。"""

    root = tmp_path / "knowledge"
    root.mkdir()

    def _fake_settings() -> SimpleNamespace:
        return SimpleNamespace(knowledge_dir=root)

    monkeypatch.setattr(kc, "get_settings", _fake_settings)
    return root


# ── 路径约定 ──────────────────────────────────────────────────────


def test_slug_rules() -> None:
    assert kc.is_valid_wiki_slug("refund-policy_v2")
    assert not kc.is_valid_wiki_slug("Refund")  # 大写
    assert not kc.is_valid_wiki_slug("-abc")  # 连字符开头
    assert not kc.is_valid_wiki_slug("a/b")  # 路径分隔
    assert not kc.is_valid_wiki_slug("")


def test_wiki_source_id_roundtrip() -> None:
    assert kc.wiki_source_id("onboarding") == "wiki/onboarding.md"
    assert kc.slug_from_wiki_source("wiki/onboarding.md") == "onboarding"
    # Windows 分隔符与前导斜杠应被归一化
    assert kc.slug_from_wiki_source("\\wiki\\onboarding.md") == "onboarding"
    assert kc.slug_from_wiki_source("/wiki/onboarding.md") == "onboarding"
    # 非本契约镜像
    assert kc.slug_from_wiki_source("wiki/sub/onboarding.md") is None
    assert kc.slug_from_wiki_source("wiki/onboarding.txt") is None
    assert kc.slug_from_wiki_source("wiki/Bad Slug.md") is None
    assert kc.slug_from_wiki_source("policies.md") is None


def test_is_wiki_source() -> None:
    assert kc.is_wiki_source("wiki/onboarding.md")
    assert kc.is_wiki_source("wiki")
    assert kc.is_wiki_source("wiki/sub/x.md")
    assert not kc.is_wiki_source("wikix.md")
    assert not kc.is_wiki_source("docs/wiki/x.md")


def test_tenant_root_and_mirror_path(kb_root: Path) -> None:
    assert kc.tenant_root("acme") == (kb_root / "acme").resolve()
    mirror = kc.wiki_mirror_path("acme", "onboarding")
    assert mirror == (kb_root / "acme" / "wiki" / "onboarding.md").resolve()
    with pytest.raises(kc.KnowledgePathError):
        kc.tenant_root("")


# ── 镜像物化 ─────────────────────────────────────────────────────


def test_materialize_and_remove(kb_root: Path) -> None:
    path = kc.materialize_wiki_page("acme", "onboarding", "新人指南", "正文内容")
    text = path.read_text(encoding="utf-8")
    # 新契约：frontmatter（自动补 title）+ 一级标题 + 正文
    assert text.startswith("---\n")
    assert "title: 新人指南" in text
    assert "# 新人指南" in text
    assert "正文内容" in text

    # 幂等覆盖
    kc.materialize_wiki_page("acme", "onboarding", "新人指南", "修订正文")
    assert "修订正文" in path.read_text(encoding="utf-8")

    # 正文自带 frontmatter 时保留其字段，title 以页面标题为准
    path = kc.materialize_wiki_page(
        "acme", "onboarding", "新人指南", "---\ntitle: 旧标题\ntags:\n  - 入职\n---\n\n正文"
    )
    text = path.read_text(encoding="utf-8")
    assert "title: 新人指南" in text
    assert "tags:" in text
    assert "旧标题" not in text
    assert text.count("---") == 2  # frontmatter 只出现在文件头

    assert kc.remove_wiki_mirror("acme", "onboarding") is True
    assert not path.exists()
    assert kc.remove_wiki_mirror("acme", "onboarding") is False


def test_materialize_rejects_bad_slug(kb_root: Path) -> None:
    with pytest.raises(kc.KnowledgePathError):
        kc.materialize_wiki_page("acme", "../evil", "标题", "正文")


def test_async_wrappers(kb_root: Path) -> None:
    path = asyncio.run(
        kc.amaterialize_wiki_page("acme", "p1", "标题", "异步正文")
    )
    assert "异步正文" in path.read_text(encoding="utf-8")
    assert asyncio.run(kc.aremove_wiki_mirror("acme", "p1")) is True
    # 未初始化租户：通知刷新必须安全跳过（不触发模型加载）
    assert asyncio.run(kc.arequest_refresh("tenant-never-initialized")) is False


# ── 上传/删除路径安全 ────────────────────────────────────────────


def test_sanitize_filename() -> None:
    assert kc.sanitize_filename("../etc/passwd") == "passwd"
    assert kc.sanitize_filename(r"a\b\c.md") == "c.md"
    assert kc.sanitize_filename("a:b?.md") == "a_b_.md"
    assert kc.sanitize_filename("  report.md  ") == "report.md"


def test_resolve_tenant_file(kb_root: Path) -> None:
    root = (kb_root / "acme").resolve()
    ok = kc.resolve_tenant_file("acme", "docs/a.md")
    assert ok == root / "docs" / "a.md"

    for bad in ("", ".", "..", "../escape.md", "docs/../../escape.md"):
        with pytest.raises(kc.KnowledgePathError):
            kc.resolve_tenant_file("acme", bad)


def test_wiki_prefix_reserved(kb_root: Path) -> None:
    with pytest.raises(kc.KnowledgePathError):
        kc.assert_not_wiki_source("wiki/onboarding.md")
    kc.assert_not_wiki_source("docs/a.md")  # 不抛异常

    assert kc.resolve_upload_dir("acme", "") == (kb_root / "acme").resolve()
    assert (
        kc.resolve_upload_dir("acme", "docs/sub")
        == (kb_root / "acme" / "docs" / "sub").resolve()
    )
    with pytest.raises(kc.KnowledgePathError):
        kc.resolve_upload_dir("acme", "wiki")
    with pytest.raises(kc.KnowledgePathError):
        kc.resolve_upload_dir("acme", "wiki/seed")
    with pytest.raises(kc.KnowledgePathError):
        kc.resolve_upload_dir("acme", "../escape")
    # 显式放行（仅 Wiki 模块内部场景）
    assert (
        kc.resolve_upload_dir("acme", "wiki", allow_wiki=True)
        == (kb_root / "acme" / "wiki").resolve()
    )


# ── 镜像重建（一致性修复） ───────────────────────────────────────


def test_rebuild_wiki_mirrors_prunes_orphans(kb_root: Path) -> None:
    mirror_dir = kb_root / "acme" / "wiki"
    mirror_dir.mkdir(parents=True)
    # 业务库里已不存在的孤儿镜像
    (mirror_dir / "deleted-page.md").write_text("旧", encoding="utf-8")
    # 即将被覆盖的有效镜像
    (mirror_dir / "keep.md").write_text("旧版本", encoding="utf-8")

    result = kc.rebuild_wiki_mirrors(
        "acme",
        [
            ("keep", "保留页", "最新正文"),
            ("new-page", "新页", "新正文"),
        ],
    )

    assert set(result["written"]) == {"wiki/keep.md", "wiki/new-page.md"}
    assert result["removed_orphans"] == ["wiki/deleted-page.md"]
    assert not (mirror_dir / "deleted-page.md").exists()
    assert "最新正文" in (mirror_dir / "keep.md").read_text(encoding="utf-8")
    assert (mirror_dir / "new-page.md").exists()
