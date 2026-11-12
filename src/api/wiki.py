"""轻量 Wiki API。

Wiki 页面以 Markdown 存储，同时：
- 写入业务表 wiki_pages / wiki_revisions（版本历史）；
- 通过 Wiki↔RAG 契约层（src.rag.knowledge_contract）物化为
  ``KNOWLEDGE_DIR/<tenant>/wiki/<slug>.md`` 镜像供 RAG 检索；
- 写操作（创建/更新/删除）后经契约层通知知识库增量 refresh。

所有接口按租户隔离，当前用户只能操作所属租户的 Wiki 页面。
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api.auth import CurrentUser
from src.persistence import database as db
from src.rag import knowledge_contract as kc
from src.rag import parser, wiki_schema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wiki", tags=["wiki"])

_TITLE_MAX = 200
_CONTENT_MAX = 200_000  # 200KB
_SUMMARY_MAX = 200
_MAX_IMPORT_BYTES = 10 * 1024 * 1024  # 导入文档源文件上限
_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # 单个附件上限


def _validate_slug(slug: str) -> str:
    """校验 slug 格式（命名规则以契约层为唯一来源）。"""

    if not kc.is_valid_wiki_slug(slug):
        raise HTTPException(
            status_code=400,
            detail="slug 只能包含小写字母、数字、连字符和下划线，且以字母或数字开头",
        )
    return slug


def _slugify_from_name(stem: str) -> str:
    """从文件名 stem 生成合法 slug（小写、非法字符转连字符、压缩、截断）。"""

    slug = re.sub(r"[^a-z0-9_-]+", "-", stem.lower()).strip("-_")
    slug = re.sub(r"-{2,}", "-", slug)[:80]
    if not slug or not kc.is_valid_wiki_slug(slug):
        slug = f"import-{uuid.uuid4().hex[:8]}"
    return slug


def _validate_content_frontmatter(content: str) -> None:
    """Schema 纪律：正文携带 frontmatter 时必须严格合法，缺省合法。

    - 结构存在但 YAML 无法解析 → 400；
    - 解析成功但违反 Schema（未知键/类型/超长）→ 400 带违例列表；
    - 无 frontmatter → 直接放行（materializer 会自动补 title）。
    """

    if not wiki_schema.has_frontmatter_block(content):
        return
    fm, _ = wiki_schema.parse_frontmatter(content)
    if fm is None:
        raise HTTPException(
            status_code=400,
            detail="frontmatter 无法解析为有效的 YAML 键值块，请修正后重试",
        )
    errors = wiki_schema.validate_frontmatter(fm)
    if errors:
        raise HTTPException(
            status_code=400,
            detail="frontmatter 违反 Schema 约束: " + "；".join(errors),
        )


def _title_from_content(content: str, fallback: str) -> str:
    """取正文首个一级标题作为页面标题；无则用文件名 stem 兜底。

    先剥离 frontmatter 再扫描，避免 YAML 注释行被误判为标题。
    """

    _, body = wiki_schema.parse_frontmatter(content)
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title[:_TITLE_MAX]
    return (fallback or "未命名")[:_TITLE_MAX]




# ── 请求/响应模型 ────────────────────────────────────────────────────


class WikiPageCreate(BaseModel):
    slug: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=_TITLE_MAX)
    content: str = Field(min_length=1, max_length=_CONTENT_MAX)
    summary: str = Field(default="", max_length=_SUMMARY_MAX)


class WikiPageUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=_TITLE_MAX)
    content: str = Field(min_length=1, max_length=_CONTENT_MAX)
    summary: str = Field(default="", max_length=_SUMMARY_MAX)


class WikiPageResponse(BaseModel):
    id: str
    slug: str
    title: str
    content: str
    author_id: str
    summary: str
    created_at: float
    updated_at: float


class WikiPageSummary(BaseModel):
    id: str
    slug: str
    title: str
    created_at: float
    updated_at: float


class WikiRevisionResponse(BaseModel):
    id: str
    author_id: str
    summary: str
    created_at: float


class WikiRevisionDetailResponse(WikiRevisionResponse):
    content: str


class WikiRevertRequest(BaseModel):
    revision_id: str
    summary: str = ""


class WikiSearchHit(BaseModel):
    slug: str
    title: str
    updated_at: float
    snippet: str
    title_match: bool


class WikiAttachmentResponse(BaseModel):
    id: str
    filename: str
    size_bytes: int
    content_type: str
    created_at: float


# ── 端点 ──────────────────────────────────────────────────────────────


@router.get("/search", response_model=list[WikiSearchHit])
async def search_pages(q: str, user: CurrentUser) -> list[dict[str, Any]]:
    """全文搜索当前租户 Wiki 页面（标题 + 最新版本正文）。

    返回匹配页面及正文匹配位置附近的摘要片段；关键词为空时返回空列表。
    """

    keyword = q.strip()
    if not keyword:
        return []
    rows = await db.search_wiki_pages(user["tenant_id"], keyword, limit=20)
    hits: list[dict[str, Any]] = []
    for row in rows:
        title = row.get("title") or ""
        content = row.get("content") or ""
        title_match = keyword.lower() in title.lower()
        hits.append(
            {
                "slug": row["slug"],
                "title": title,
                "updated_at": row["updated_at"],
                "snippet": _build_snippet(content, keyword),
                "title_match": title_match,
            }
        )
    return hits


def _build_snippet(content: str, keyword: str, radius: int = 60) -> str:
    """从正文中截取关键词首次出现位置前后的摘要片段。"""

    idx = content.lower().find(keyword.lower())
    if idx < 0:
        # 标题命中但正文不含关键词：返回正文开头
        return content[: radius * 2].strip() + ("…" if len(content) > radius * 2 else "")
    start = max(0, idx - radius)
    end = min(len(content), idx + len(keyword) + radius)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""
    return prefix + content[start:end].strip().replace("\n", " ") + suffix


@router.get("/pages", response_model=list[WikiPageSummary])
async def list_pages(user: CurrentUser) -> list[dict[str, Any]]:
    """列出当前租户的全部 Wiki 页面（不含正文）。"""

    return await db.list_wiki_pages(user["tenant_id"])


@router.get("/pages/{slug}", response_model=WikiPageResponse)
async def get_page(slug: str, user: CurrentUser) -> dict[str, Any]:
    """读取 Wiki 页面当前版本（含正文）。"""

    _validate_slug(slug)
    page = await db.get_wiki_page(user["tenant_id"], slug)
    if page is None:
        raise HTTPException(status_code=404, detail="Wiki 页面不存在")
    return page


@router.post("/pages", response_model=WikiPageResponse, status_code=201)
async def create_page(req: WikiPageCreate, user: CurrentUser) -> dict[str, Any]:
    """创建 Wiki 页面并镜像到知识库。"""

    slug = _validate_slug(req.slug)
    _validate_content_frontmatter(req.content)
    existing = await db.get_wiki_page(user["tenant_id"], slug)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"slug '{slug}' 已存在")
    page = await db.create_wiki_page(
        tenant_id=user["tenant_id"],
        slug=slug,
        title=req.title,
        content=req.content,
        author_id=user["id"],
        summary=req.summary,
    )
    await kc.amaterialize_wiki_page(user["tenant_id"], slug, req.title, req.content)
    refreshed = await kc.arequest_refresh(user["tenant_id"])
    logger.info(
        "Wiki 页面已创建: tenant=%s, slug=%s, title=%r, kb_refreshed=%s",
        user["tenant_id"], slug, req.title, refreshed,
    )
    return {**page, "content": req.content, "author_id": user["id"], "summary": req.summary}


@router.put("/pages/{slug}", response_model=WikiPageResponse)
async def update_page(slug: str, req: WikiPageUpdate, user: CurrentUser) -> dict[str, Any]:
    """更新 Wiki 页面（写入新版本 + 刷新知识库镜像）。"""

    slug = _validate_slug(slug)
    _validate_content_frontmatter(req.content)
    updated = await db.update_wiki_page(
        tenant_id=user["tenant_id"],
        slug=slug,
        title=req.title,
        content=req.content,
        author_id=user["id"],
        summary=req.summary,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Wiki 页面不存在")
    await kc.amaterialize_wiki_page(user["tenant_id"], slug, req.title, req.content)
    refreshed = await kc.arequest_refresh(user["tenant_id"])
    logger.info(
        "Wiki 页面已更新: tenant=%s, slug=%s, kb_refreshed=%s",
        user["tenant_id"], slug, refreshed,
    )
    page = await db.get_wiki_page(user["tenant_id"], slug)
    assert page is not None
    return page


@router.delete("/pages/{slug}", status_code=204)
async def delete_page(slug: str, user: CurrentUser) -> None:
    """删除 Wiki 页面（含全部版本历史、知识库镜像与附件目录）。"""

    slug = _validate_slug(slug)
    deleted = await db.delete_wiki_page(user["tenant_id"], slug)
    if not deleted:
        raise HTTPException(status_code=404, detail="Wiki 页面不存在")
    await kc.aremove_wiki_mirror(user["tenant_id"], slug)
    await kc.aremove_attachment_dir(user["tenant_id"], slug)
    refreshed = await kc.arequest_refresh(user["tenant_id"])
    logger.info(
        "Wiki 页面已删除: tenant=%s, slug=%s, kb_refreshed=%s",
        user["tenant_id"], slug, refreshed,
    )


@router.get("/pages/{slug}/history", response_model=list[WikiRevisionResponse])
async def page_history(slug: str, user: CurrentUser) -> list[dict[str, Any]]:
    """查看 Wiki 页面的版本历史（不含正文）。"""

    _validate_slug(slug)
    return await db.list_wiki_revisions(user["tenant_id"], slug)


@router.get(
    "/pages/{slug}/revisions/{revision_id}",
    response_model=WikiRevisionDetailResponse,
)
async def get_revision(
    slug: str, revision_id: str, user: CurrentUser
) -> dict[str, Any]:
    """读取某个历史版本的完整正文（供查看与 diff 对比）。"""

    _validate_slug(slug)
    revision = await db.get_wiki_revision(user["tenant_id"], slug, revision_id)
    if revision is None:
        raise HTTPException(status_code=404, detail="版本不存在")
    return revision


@router.post("/pages/{slug}/revert", response_model=WikiPageResponse)
async def revert_page(
    slug: str, req: WikiRevertRequest, user: CurrentUser
) -> dict[str, Any]:
    """回滚到指定历史版本。

    采用 append-only 语义：不删除任何历史，而是把目标版本的正文作为
    一个新版本写入；页面标题保持当前值不变。
    """

    slug = _validate_slug(slug)
    target = await db.get_wiki_revision(user["tenant_id"], slug, req.revision_id)
    if target is None:
        raise HTTPException(status_code=404, detail="目标版本不存在")
    _validate_content_frontmatter(target["content"])
    current = await db.get_wiki_page(user["tenant_id"], slug)
    assert current is not None  # 能取到 revision 说明页面必然存在
    summary = req.summary.strip() or f"回滚到版本 {req.revision_id[:8]}"
    updated = await db.update_wiki_page(
        tenant_id=user["tenant_id"],
        slug=slug,
        title=current["title"],
        content=target["content"],
        author_id=user["id"],
        summary=summary,
    )
    if updated is None:  # pragma: no cover - 极端并发删除场景
        raise HTTPException(status_code=404, detail="Wiki 页面不存在")
    await kc.amaterialize_wiki_page(user["tenant_id"], slug, current["title"], target["content"])
    await kc.arequest_refresh(user["tenant_id"])
    logger.info(
        "Wiki 页面已回滚: tenant=%s, slug=%s, revision=%s",
        user["tenant_id"], slug, req.revision_id[:8],
    )
    page = await db.get_wiki_page(user["tenant_id"], slug)
    assert page is not None
    return page


# ── 文档导入 ────────────────────────────────────────────────────────


@router.post("/import", response_model=WikiPageResponse, status_code=201)
async def import_page(
    user: CurrentUser,
    file: Annotated[UploadFile, File()],
    slug: Annotated[str, Form()] = "",
    title: Annotated[str, Form()] = "",
    summary: Annotated[str, Form()] = "导入文档",
) -> dict[str, Any]:
    """上传文档并解析正文，自动创建为 Wiki 页面。

    - 支持 md/txt/html/pdf/docx（同 RAG 解析器白名单）；
    - slug 缺省由文件名生成，title 缺省取正文首个一级标题；
    - 复用既有「创建页面 → 镜像 → 刷新索引」三联动，导入即入库可检索。
    """

    raw_name = kc.sanitize_filename(file.filename or "")
    suffix = Path(raw_name).suffix.lower()
    if suffix not in parser.SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="仅支持 md/txt/html/pdf/docx 文档",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(data) > _MAX_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail="文档超过 10MB 上限")

    # 落临时文件后统一抽取纯文本（parse 器按后缀选择解析策略）
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        try:
            content = parser.extract_text(tmp_path).strip()
        except Exception as exc:  # 解析库异常（pdf/docx 损坏等）
            logger.warning("导入文档解析失败: %s", exc)
            raise HTTPException(
                status_code=400, detail="文档解析失败，请确认文件格式"
            ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    if not content:
        raise HTTPException(status_code=400, detail="文档解析后无有效正文")
    if len(content) > _CONTENT_MAX:
        raise HTTPException(status_code=413, detail=f"正文超过 {_CONTENT_MAX} 字符上限")
    _validate_content_frontmatter(content)

    stem = Path(raw_name).stem
    new_slug = _validate_slug(slug.strip()) if slug.strip() else _slugify_from_name(stem)
    new_title = title.strip() if title.strip() else _title_from_content(content, stem)
    new_summary = summary.strip() or "导入文档"

    existing = await db.get_wiki_page(user["tenant_id"], new_slug)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"slug '{new_slug}' 已存在")

    page = await db.create_wiki_page(
        tenant_id=user["tenant_id"],
        slug=new_slug,
        title=new_title,
        content=content,
        author_id=user["id"],
        summary=new_summary,
    )
    await kc.amaterialize_wiki_page(user["tenant_id"], new_slug, new_title, content)
    refreshed = await kc.arequest_refresh(user["tenant_id"])
    logger.info(
        "Wiki 页面已导入: tenant=%s, slug=%s, source=%r, kb_refreshed=%s",
        user["tenant_id"], new_slug, raw_name, refreshed,
    )
    return {**page, "content": content, "author_id": user["id"], "summary": new_summary}


# ── 页面附件 ────────────────────────────────────────────────────────


def _attachment_payload(row: dict[str, Any]) -> dict[str, Any]:
    """把附件 DB 行映射为对外响应负载（不暴露 stored_name/tenant 等内部字段）。"""

    return {
        "id": row["id"],
        "filename": row["filename"],
        "size_bytes": row["size_bytes"],
        "content_type": row["content_type"],
        "created_at": row["created_at"],
    }


@router.post(
    "/pages/{slug}/attachments",
    response_model=WikiAttachmentResponse,
    status_code=201,
)
async def upload_attachment(
    slug: str, user: CurrentUser, file: Annotated[UploadFile, File()]
) -> dict[str, Any]:
    """给 Wiki 页面挂载附件（原样存储，不解析、不进版本、不进 RAG 索引）。"""

    slug = _validate_slug(slug)
    page = await db.get_wiki_page(user["tenant_id"], slug)
    if page is None:
        raise HTTPException(status_code=404, detail="Wiki 页面不存在")

    raw_name = kc.sanitize_filename(file.filename or "")
    if not raw_name:
        raise HTTPException(status_code=400, detail="文件名无效")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(data) > _MAX_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail="附件超过 20MB 上限")

    ext = Path(raw_name).suffix.lower()
    stored_name = f"{uuid.uuid4().hex}{ext}"
    await kc.awrite_attachment(user["tenant_id"], slug, stored_name, data)

    row = await db.add_wiki_attachment(
        tenant_id=user["tenant_id"],
        page_id=page["id"],
        filename=raw_name,
        stored_name=stored_name,
        size_bytes=len(data),
        content_type=file.content_type or "",
        uploaded_by=user["id"],
    )
    return _attachment_payload(row)


@router.get(
    "/pages/{slug}/attachments",
    response_model=list[WikiAttachmentResponse],
)
async def list_attachments(slug: str, user: CurrentUser) -> list[dict[str, Any]]:
    """列出某个 Wiki 页面的全部附件。"""

    slug = _validate_slug(slug)
    page = await db.get_wiki_page(user["tenant_id"], slug)
    if page is None:
        raise HTTPException(status_code=404, detail="Wiki 页面不存在")
    rows = await db.list_wiki_attachments(user["tenant_id"], page["id"])
    return [_attachment_payload(row) for row in rows]


@router.get("/attachments/{attachment_id}")
async def download_attachment(attachment_id: str, user: CurrentUser) -> FileResponse:
    """下载附件（租户归属校验后返回文件流，Content-Disposition 用原始文件名）。"""

    row = await db.get_wiki_attachment(user["tenant_id"], attachment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="附件不存在")
    path = kc.resolve_attachment_path(user["tenant_id"], row["slug"], row["stored_name"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="附件文件缺失")
    return FileResponse(
        path,
        filename=row["filename"],
        media_type=row["content_type"] or "application/octet-stream",
    )


@router.delete("/attachments/{attachment_id}", status_code=204)
async def delete_attachment(attachment_id: str, user: CurrentUser) -> None:
    """删除附件（DB 记录 + 落盘文件一并清理）。"""

    row = await db.delete_wiki_attachment(user["tenant_id"], attachment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="附件不存在")
    await kc.aremove_attachment_file(user["tenant_id"], row["slug"], row["stored_name"])
    logger.info(
        "Wiki 附件已删除: tenant=%s, slug=%s, attachment=%s",
        user["tenant_id"], row["slug"], attachment_id,
    )
