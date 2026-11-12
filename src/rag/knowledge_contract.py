"""Wiki ↔ RAG 文件系统契约层。

两个子系统通过"共享目录 + 刷新通知"解耦，契约只有三条：

1. **目录布局**：所有知识文件位于 ``KNOWLEDGE_DIR/<tenant_id>/`` 下；
   Wiki 页面镜像独占 ``wiki/`` 子目录，文件名固定为 ``wiki/<slug>.md``。
   Wiki 模块是该目录的唯一写入方，上传/删除端点禁止写入该保留前缀。
2. **物化当前版**：Wiki 只把页面的当前版本物化为镜像文件（历史版本
   只存在业务库，不进检索索引）；物化是幂等覆盖。
3. **通知刷新**：文件增删改后调用 :func:`request_refresh`；知识库尚未
   初始化（嵌入模型未加载）时跳过，等首次检索时自然全量重建。

RAG 索引在这个契约里被视为**可丢弃的派生数据**：事实源是业务库与
原始文件，镜像可随时由业务库重新物化（:func:`rebuild_wiki_mirrors`），
索引可随时全量重建。本模块不包含任何数据库操作。
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import anyio

from src.rag.config import get_settings

if TYPE_CHECKING:
    from src.rag.pipeline import KnowledgeBase

logger = logging.getLogger(__name__)

# Wiki 镜像在租户目录中的保留子目录名
WIKI_DIR_NAME = "wiki"
# slug 规则：小写字母、数字、连字符、下划线；长度 1-80
WIKI_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
# 上传文件名中的非法字符（与 Windows 文件名保留字集对齐）
_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class KnowledgePathError(ValueError):
    """知识库路径违反契约（路径穿越、保留目录、非法文件名等）。"""


# ── 路径契约 ─────────────────────────────────────────────────────────


def knowledge_root() -> Path:
    """返回已解析的知识库根目录。"""

    return get_settings().knowledge_dir.resolve()


def tenant_root(tenant_id: str) -> Path:
    """返回指定租户的知识库根目录 ``KNOWLEDGE_DIR/<tenant_id>/``。

    调用方在做越界判断时应使用 ``.resolve()`` 后再比较
    （:func:`resolve_tenant_file` 已内置该处理）。
    """

    if not tenant_id:
        raise KnowledgePathError("tenant_id 不能为空")
    return knowledge_root() / tenant_id


def is_valid_wiki_slug(slug: str) -> bool:
    """slug 是否符合 Wiki 命名契约。"""

    return bool(WIKI_SLUG_RE.match(slug))


def wiki_source_id(slug: str) -> str:
    """slug → 检索来源标识（租户根下的 POSIX 相对路径）。"""

    if not is_valid_wiki_slug(slug):
        raise KnowledgePathError(f"非法的 Wiki slug: {slug!r}")
    return f"{WIKI_DIR_NAME}/{slug}.md"


def wiki_mirror_path(tenant_id: str, slug: str) -> Path:
    """返回 Wiki 页面镜像文件的绝对路径（不要求文件已存在）。"""

    if not is_valid_wiki_slug(slug):
        raise KnowledgePathError(f"非法的 Wiki slug: {slug!r}")
    return tenant_root(tenant_id) / WIKI_DIR_NAME / f"{slug}.md"


def is_wiki_source(source_id: str) -> bool:
    """判断来源标识是否指向 Wiki 镜像保留区。"""

    normalized = (source_id or "").replace("\\", "/").lstrip("/")
    return normalized == WIKI_DIR_NAME or normalized.startswith(f"{WIKI_DIR_NAME}/")


def slug_from_wiki_source(source_id: str) -> str | None:
    """Wiki 镜像来源 → slug；非本契约镜像（含子目录/其他后缀）返回 None。"""

    normalized = (source_id or "").replace("\\", "/").lstrip("/")
    prefix = f"{WIKI_DIR_NAME}/"
    if not normalized.startswith(prefix) or "/" in normalized[len(prefix):]:
        return None
    name = normalized[len(prefix):]
    if not name.endswith(".md"):
        return None
    slug = name[:-3]
    return slug if is_valid_wiki_slug(slug) else None


# ── 上传/删除路径安全 ───────────────────────────────────────────────


def sanitize_filename(raw: str) -> str:
    """剥离目录成分并替换非法字符，防止路径穿越。

    统一正反斜杠后只取最后一级，任何 ``../`` 或绝对路径成分都会被丢弃。
    """

    name = PurePosixPath(raw.replace("\\", "/")).name
    return _INVALID_NAME_CHARS.sub("_", name).strip().strip(". ")


def assert_not_wiki_source(source_id: str) -> None:
    """拒绝直接操作 Wiki 保留目录，避免出现孤儿镜像。"""

    if is_wiki_source(source_id):
        raise KnowledgePathError(
            "Wiki 镜像文件不可直接操作，请在 Wiki 页面管理中处理对应页面"
        )


def resolve_tenant_file(tenant_id: str, source_id: str) -> Path:
    """把租户相对路径安全解析为绝对路径。

    - 接受 POSIX 风格相对路径（支持子目录）；
    - 拒绝空路径、``.``/``..`` 与解析后越出租户根的路径；
    - 仅做路径级校验，文件是否存在、后缀是否受支持由调用方判断。
    """

    root = tenant_root(tenant_id).resolve()
    normalized = (source_id or "").replace("\\", "/").lstrip("/")
    if not normalized or normalized in {".", ".."}:
        raise KnowledgePathError("非法的文件标识")
    target = (root / Path(normalized)).resolve()
    if not target.is_relative_to(root):
        raise KnowledgePathError("文件路径越界")
    return target


def resolve_upload_dir(tenant_id: str, subdir: str, *, allow_wiki: bool = False) -> Path:
    """校验上传子目录：逐段清洗文件名并拒绝越界；默认拒绝 wiki/ 保留区。"""

    root = tenant_root(tenant_id)
    raw = (subdir or "").replace("\\", "/").strip("/")
    parts: list[str] = []
    for raw_part in PurePosixPath(raw).parts:
        part = sanitize_filename(raw_part)
        if not part or part in {".", ".."}:
            raise KnowledgePathError(f"非法的子目录片段: {raw_part!r}")
        parts.append(part)
    if not allow_wiki and parts and parts[0] == WIKI_DIR_NAME:
        raise KnowledgePathError(
            "wiki/ 是 Wiki 页面的托管目录，请通过新建 Wiki 页面写入文档"
        )
    target_dir = root.joinpath(*parts) if parts else root
    if not target_dir.resolve().is_relative_to(root.resolve()):
        raise KnowledgePathError("子目录路径越界")
    return target_dir


# ── 附件路径契约 ───────────────────────────────────────────────
# 附件不进 RAG 索引：落盘在知识库目录之外的 ``<data_dir>/wiki_attachments/``
# 下，因此不会被 scan_document_stats 的 rglob("*") 扫入检索。


def attachment_storage_root() -> Path:
    """返回附件存储根目录 ``<data_dir>/wiki_attachments/``（位于知识库之外）。"""

    return get_settings().data_dir.resolve() / "wiki_attachments"


def attachment_dir(tenant_id: str, slug: str) -> Path:
    """返回指定页面附件的存储目录 ``<root>/<tenant>/<slug>/``。"""

    if not tenant_id:
        raise KnowledgePathError("tenant_id 不能为空")
    if not is_valid_wiki_slug(slug):
        raise KnowledgePathError(f"非法的 Wiki slug: {slug!r}")
    return attachment_storage_root() / tenant_id / slug


def resolve_attachment_path(tenant_id: str, slug: str, stored_name: str) -> Path:
    """把附件存储名安全解析为绝对路径（防路径越界）。"""

    directory = attachment_dir(tenant_id, slug).resolve()
    name = sanitize_filename(stored_name)
    if not name:
        raise KnowledgePathError("非法的附件存储名")
    target = (directory / name).resolve()
    if not target.is_relative_to(directory):
        raise KnowledgePathError("附件路径越界")
    return target


def remove_attachment_dir(tenant_id: str, slug: str) -> bool:
    """删除指定页面的整个附件目录（幂等），用于删除页面时清理附件文件。

    仅递归删除 ``<root>/<tenant>/<slug>/`` 这一层；上级目录保留，
    避免误删其他页面的附件。
    """

    directory = attachment_dir(tenant_id, slug)
    if not directory.exists():
        return False
    shutil.rmtree(directory)
    return True


# ── Wiki 生产者：镜像物化（同步原语） ────────────────────────────


def materialize_wiki_page(tenant_id: str, slug: str, title: str, content: str) -> Path:
    """把 Wiki 页面当前版本幂等写入镜像文件，返回镜像路径。

    物化纪律（Schema 约束）：
    - 解析正文自带的 frontmatter（可缺省），以页面 title 为准覆盖合并；
    - 镜像始终输出 ``frontmatter + # 一级标题 + 正文`` 结构，保证 RAG
      解析时元数据（版本/时效）与文档标题可被识别；
    - 物化是幂等覆盖：相同输入产出字节相同的镜像。
    """

    from src.rag.wiki_schema import dump_frontmatter, merge_frontmatter, parse_frontmatter

    target = wiki_mirror_path(tenant_id, slug)
    target.parent.mkdir(parents=True, exist_ok=True)
    fm, body = parse_frontmatter(content)
    merged = merge_frontmatter(fm, {"title": title.strip()})
    body = body.strip() or f"# {title.strip()}"
    head = dump_frontmatter(merged)
    heading = "" if body.startswith("# ") else f"# {title.strip()}\n\n"
    target.write_text(f"{head}\n{heading}{body}\n", encoding="utf-8")
    return target


def remove_wiki_mirror(tenant_id: str, slug: str) -> bool:
    """删除 Wiki 页面镜像；文件不存在时返回 False（幂等）。"""

    target = wiki_mirror_path(tenant_id, slug)
    if not target.exists():
        return False
    target.unlink()
    return True


def rebuild_wiki_mirrors(
    tenant_id: str, pages: Iterable[tuple[str, str, str]]
) -> dict[str, list[str]]:
    """以业务库为准全量重建镜像（一致性修复入口）。

    :param pages: ``(slug, title, content)`` 三元组可迭代对象，通常
        直接传 ``SELECT slug,title,最新版正文`` 的结果；
    :returns: ``{"written": [source_id...], "removed_orphans": [source_id...]}``，
        ``removed_orphans`` 为镜像目录中业务库已不存在对应页面的孤儿文件。
    """

    mirror_dir = tenant_root(tenant_id) / WIKI_DIR_NAME
    mirror_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    valid_slugs: set[str] = set()
    for slug, title, content in pages:
        materialize_wiki_page(tenant_id, slug, title, content)
        valid_slugs.add(slug)
        written.append(wiki_source_id(slug))

    orphans: list[str] = []
    for path in mirror_dir.glob("*.md"):
        # 非契约命名（非法 slug）的文件不由本模块托管，保留不动
        if not is_valid_wiki_slug(path.stem) or path.stem in valid_slugs:
            continue
        path.unlink()
        orphans.append(wiki_source_id(path.stem))
    if orphans:
        logger.warning(
            "重建 Wiki 镜像时清理了 %d 个孤儿镜像: tenant=%s, orphans=%s",
            len(orphans), tenant_id, orphans,
        )
    return {"written": written, "removed_orphans": orphans}


# ── 索引通知（RAG 侧） ────────────────────────────────────────────


def refresh_knowledge_base(tenant_id: str) -> KnowledgeBase:
    """强制刷新指定租户的知识库索引（同步，在线程池中调用）。"""

    # 延迟导入：契约层被 wiki/app 引用，避免与 tools 的初始化链互相拖入
    from src.rag.tools import get_knowledge_base

    kb = get_knowledge_base(tenant_id)
    kb.refresh()
    return kb


def is_initialized(tenant_id: str) -> bool:
    """知识库是否已初始化（决定写操作后是否需要同步刷索引）。"""

    from src.rag.tools import is_knowledge_base_initialized

    return is_knowledge_base_initialized(tenant_id)


def request_refresh(tenant_id: str) -> bool:
    """知识库已初始化则增量刷新并返回 True；否则跳过（首次检索时自然重建）。"""

    if not is_initialized(tenant_id):
        return False
    refresh_knowledge_base(tenant_id)
    return True


# ── async 包装（供 FastAPI 端点使用，文件 IO 全部丢到线程池） ──────


async def amaterialize_wiki_page(
    tenant_id: str, slug: str, title: str, content: str
) -> Path:
    return await anyio.to_thread.run_sync(
        materialize_wiki_page, tenant_id, slug, title, content
    )


async def aremove_wiki_mirror(tenant_id: str, slug: str) -> bool:
    return await anyio.to_thread.run_sync(remove_wiki_mirror, tenant_id, slug)


async def arequest_refresh(tenant_id: str) -> bool:
    return await anyio.to_thread.run_sync(request_refresh, tenant_id)


async def awrite_attachment(
    tenant_id: str, slug: str, stored_name: str, data: bytes
) -> Path:
    """把附件字节幂等写入附件目录（文件 IO 走线程池），返回落盘路径。"""

    def _write() -> Path:
        target = resolve_attachment_path(tenant_id, slug, stored_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    return await anyio.to_thread.run_sync(_write)


async def aremove_attachment_file(tenant_id: str, slug: str, stored_name: str) -> bool:
    """删除单个附件文件（幂等）。"""

    def _remove() -> bool:
        target = resolve_attachment_path(tenant_id, slug, stored_name)
        if not target.exists():
            return False
        target.unlink()
        return True

    return await anyio.to_thread.run_sync(_remove)


async def aremove_attachment_dir(tenant_id: str, slug: str) -> bool:
    """删除指定页面的整个附件目录（幂等）。"""

    return await anyio.to_thread.run_sync(remove_attachment_dir, tenant_id, slug)
