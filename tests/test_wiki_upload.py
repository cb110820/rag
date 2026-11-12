"""Wiki 文件上传（文档导入 + 页面附件）相关单元测试。

覆盖三块：
1. parser.extract_text：各受支持后缀抽取纯文本；
2. 契约层附件路径：目录约定、越界/非法输入拒绝、目录幂等删除；
3. 持久化层附件函数：增删查 + 跨租户越权防护。

DB 用例依赖 conftest 的 isolated_db 夹具（临时 SQLite，# foreign key 已开启）。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from docx import Document

from src.persistence import database as db
from src.rag import knowledge_contract as kc
from src.rag import parser

# ── parser.extract_text ───────────────────────────────────────────


def test_extract_text_txt_and_md(tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("纯文本内容\n第二行", encoding="utf-8")
    assert parser.extract_text(txt) == "纯文本内容\n第二行"

    md = tmp_path / "b.md"
    md.write_text("# 标题\n\n正文", encoding="utf-8")
    assert "# 标题" in parser.extract_text(md)


def test_extract_text_html(tmp_path: Path) -> None:
    html = tmp_path / "c.html"
    html.write_text(
        "<html><body><h1>标题</h1><p>段落</p></body></html>", encoding="utf-8"
    )
    text = parser.extract_text(html)
    assert "标题" in text
    assert "段落" in text


def test_extract_text_docx(tmp_path: Path) -> None:
    path = tmp_path / "d.docx"
    doc = Document()
    doc.add_paragraph("Word 文档正文")
    doc.save(path)
    assert "Word 文档正文" in parser.extract_text(path)


def test_extract_text_unsupported_raises(tmp_path: Path) -> None:
    csv = tmp_path / "e.csv"
    csv.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError):
        parser.extract_text(csv)


# ── parser frontmatter 与来源层级（知识治理） ────────────────────


def test_load_document_frontmatter_authority(tmp_path: Path) -> None:
    """frontmatter 权威覆盖正文正则；frontmatter 块从正文剥离。"""

    path = tmp_path / "doc.md"
    path.write_text(
        "---\n"
        "title: 报告\n"
        "version: v3\n"
        "effective_date: '2026-01-01'\n"
        "tags:\n  - 运维\n"
        "---\n"
        "\n# 报告\n\n生效日期: 2025-06-01\n版本: v1\n",
        encoding="utf-8",
    )
    doc = parser.load_document(tmp_path, "doc.md")
    assert doc is not None
    assert doc.metadata["version"] == "v3"
    assert doc.metadata["effective_date"] == "2026-01-01"
    assert doc.metadata["tags"] == ["运维"]
    assert doc.metadata["tier"] == "raw"
    # YAML 噪声不进索引文本
    assert "---" not in doc.text
    assert "title:" not in doc.text


def test_load_document_wiki_tier(tmp_path: Path) -> None:
    """wiki/ 保留前缀下的镜像文档 tier=wiki，其余 tier=raw。"""

    mirror_dir = tmp_path / "wiki"
    mirror_dir.mkdir()
    (mirror_dir / "guide.md").write_text(
        "---\ntitle: 指南\n---\n\n# 指南\n\n正文", encoding="utf-8"
    )
    (tmp_path / "guide.md").write_text("# 指南\n\n正文", encoding="utf-8")

    wiki_doc = parser.load_document(tmp_path, "wiki/guide.md")
    assert wiki_doc is not None
    assert wiki_doc.metadata["tier"] == "wiki"

    raw_doc = parser.load_document(tmp_path, "guide.md")
    assert raw_doc is not None
    assert raw_doc.metadata["tier"] == "raw"


def test_load_document_frontmatter_only_body_skipped(tmp_path: Path) -> None:
    """仅剩 frontmatter、正文为空的文档按空文档处理。"""

    (tmp_path / "hollow.md").write_text("---\ntitle: 空壳\n---\n\n   \n", encoding="utf-8")
    assert parser.load_document(tmp_path, "hollow.md") is None


# ── 契约层附件路径 ───────────────────────────────────────────────


@pytest.fixture()
def attach_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把附件存储根目录与知识库根目录都重定向到临时目录。"""

    data = tmp_path / "data"

    def _fake_settings() -> SimpleNamespace:
        return SimpleNamespace(data_dir=data, knowledge_dir=tmp_path / "knowledge")

    monkeypatch.setattr(kc, "get_settings", _fake_settings)
    return data


def test_attachment_dir_layout(attach_root: Path) -> None:
    assert kc.attachment_dir("acme", "guide") == (
        attach_root.resolve() / "wiki_attachments" / "acme" / "guide"
    )
    with pytest.raises(kc.KnowledgePathError):
        kc.attachment_dir("", "guide")  # tenant_id 为空
    with pytest.raises(kc.KnowledgePathError):
        kc.attachment_dir("acme", "../bad")  # 非法 slug


def test_resolve_attachment_path_rejects_invalid(attach_root: Path) -> None:
    with pytest.raises(kc.KnowledgePathError):
        kc.resolve_attachment_path("acme", "guide", "")  # 空存储名
    with pytest.raises(kc.KnowledgePathError):
        kc.resolve_attachment_path("acme", "../evil", "x.txt")  # 非法 slug


def test_resolve_attachment_path_sanitizes_traversal(attach_root: Path) -> None:
    # 目录成分被 sanitize_filename 剥离，越界转为普通文件名落在附件目录内
    target = kc.resolve_attachment_path("acme", "guide", "../etc/passwd")
    base = kc.attachment_dir("acme", "guide").resolve()
    assert target == base / "passwd"
    assert target.parent == base


def test_remove_attachment_dir_idempotent(attach_root: Path) -> None:
    directory = kc.attachment_dir("acme", "guide")
    directory.mkdir(parents=True)
    (directory / "a.bin").write_bytes(b"x")

    assert kc.remove_attachment_dir("acme", "guide") is True
    assert not directory.exists()
    assert kc.remove_attachment_dir("acme", "guide") is False


# ── 持久化层附件函数 ─────────────────────────────────────────────


async def test_attachment_db_crud(isolated_db) -> None:
    tenant_id = await db.create_tenant(name="acme", slug="acme")
    user_id = await db.create_user(
        email="owner@acme.com",
        password_hash="h",
        display_name="Owner",
        tenant_id=tenant_id,
    )

    page = await db.create_wiki_page(
        tenant_id=tenant_id,
        slug="guide",
        title="Guide",
        content="# Guide\n\n正文",
        author_id=user_id,
    )
    page_id = page["id"]

    att = await db.add_wiki_attachment(
        tenant_id=tenant_id,
        page_id=page_id,
        filename="报告.pdf",
        stored_name="abc123.pdf",
        size_bytes=1024,
        content_type="application/pdf",
        uploaded_by=user_id,
    )
    assert att["id"]
    assert att["stored_name"] == "abc123.pdf"

    rows = await db.list_wiki_attachments(tenant_id, page_id)
    assert len(rows) == 1
    assert rows[0]["filename"] == "报告.pdf"

    # 读取附带 JOIN 出的 slug；跨租户（或错误 id）越权返回 None
    assert (await db.get_wiki_attachment(tenant_id, att["id"]))["slug"] == "guide"
    assert await db.get_wiki_attachment("other", att["id"]) is None
    assert await db.list_wiki_attachments("other", page_id) == []

    deleted = await db.delete_wiki_attachment(tenant_id, att["id"])
    assert deleted is not None
    assert deleted["stored_name"] == "abc123.pdf"
    assert await db.list_wiki_attachments(tenant_id, page_id) == []
    assert await db.delete_wiki_attachment(tenant_id, att["id"]) is None