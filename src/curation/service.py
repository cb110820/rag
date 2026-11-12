"""沉淀编排服务：原始文档 → LLM 提取 → Wiki 页面 upsert → index/log。

职责边界：
- 只读原始文档、只写 Wiki 业务库与镜像，绝不修改/移动原始文档；
- 全部操作限定在传入 tenant_id 内（实体匹配也按租户）；
- LLM 提取在任何页面写入之前完成，失败不留半成品；
- 同租户任务由模块级 per-tenant 锁串行（进程内 asyncio.Lock + postgres
  advisory lock 跨进程），避免 index/log 丢失更新与多副本重复沉淀。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

import anyio

from src.curation import markdown as cm
from src.curation.llm import (
    CurationExtractError,
    ExistingEntity,
    Extractor,
    extract_curated,
)
from src.memory.guard import scan_secret
from src.persistence import database as db
from src.rag import knowledge_contract as kc
from src.rag.config import get_settings
from src.rag.parser import SUPPORTED_SUFFIXES, extract_text
from src.rag.wiki_schema import dump_frontmatter, parse_frontmatter

logger = logging.getLogger(__name__)

INDEX_TITLE = "知识库总目录"
LOG_TITLE = "沉淀操作日志"
# 传给 LLM 的既有实体结论截断（与 llm.py 内限制双保险）
_CONCLUSION_LIMIT = 600

# 同租户沉淀串行锁（进程内；手动/自动共用）
_tenant_locks: dict[str, asyncio.Lock] = {}


class CurationBusyError(RuntimeError):
    """同租户已有沉淀任务在另一进程执行中（跨进程 advisory 锁未获取）。

    上传端点会静默跳过（下次上传/手动触发重试）；手动端点转 409 提示。
    """


def _tenant_lock(tenant_id: str) -> asyncio.Lock:
    lock = _tenant_locks.get(tenant_id)
    if lock is None:
        lock = asyncio.Lock()
        _tenant_locks[tenant_id] = lock
    return lock


@dataclass
class CurationStats:
    source_id: str
    doc_slug: str = ""
    created: int = 0
    updated: int = 0
    entities: int = 0
    contradictions: int = 0
    blocked: int = 0
    refreshed: bool = False
    skipped: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── 内部工具 ─────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now()


def _page_content(fm: dict[str, Any], body: str) -> str:
    """系统页统一以 frontmatter + 正文入库（frontmatter 已保证合法）。"""

    cm.assert_valid_frontmatter(fm)
    return f"{dump_frontmatter(fm)}\n{body.strip()}\n"


async def _upsert_page(
    tenant_id: str, slug: str, title: str, content: str, author_id: str,
    stats: CurationStats, *, count: bool = True,
) -> None:
    """upsert Wiki 页面并物化镜像；count=True 时统计新建/更新数。

    index/log 系统页传 count=False，不污染业务沉淀统计。
    """

    existing = await db.get_wiki_page(tenant_id, slug)
    if existing is None:
        await db.create_wiki_page(
            tenant_id=tenant_id, slug=slug, title=title,
            content=content, author_id=author_id,
        )
        if count:
            stats.created += 1
    else:
        await db.update_wiki_page(
            tenant_id=tenant_id, slug=slug, title=title,
            content=content, author_id=author_id,
        )
        if count:
            stats.updated += 1
    await kc.amaterialize_wiki_page(tenant_id, slug, title, content)


async def _load_pages_full(tenant_id: str) -> list[dict[str, Any]]:
    """读取租户全部页面（含正文）。页面量级小，N+1 读取可接受。"""

    rows = await db.list_wiki_pages(tenant_id)
    pages: list[dict[str, Any]] = []
    for row in rows:
        full = await db.get_wiki_page(tenant_id, row["slug"])
        if full is not None:
            pages.append(full)
    return pages


def _entity_index(pages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """归一化名/别名 → 实体页信息（slug/title/fm/body）。"""

    index: dict[str, dict[str, Any]] = {}
    for page in pages:
        fm, body = parse_frontmatter(page["content"])
        if fm is None or fm.get("kind") != "entity":
            continue
        keys = {cm.normalize_entity_name(page["title"])}
        keys.update(cm.normalize_entity_name(a) for a in fm.get("aliases", []))
        for key in keys:
            if key:
                index[key] = {"page": page, "fm": fm, "body": body}
    return index


def _find_doc_page(pages: list[dict[str, Any]], source_id: str) -> dict[str, Any] | None:
    """按 sources 回链查找文档摘要页（比 slug 更抗同名文件碰撞）。"""

    for page in pages:
        fm, _ = parse_frontmatter(page["content"])
        if fm is not None and fm.get("kind") == "doc" and source_id in (fm.get("sources") or []):
            return page
    return None


# ── log / index ──────────────────────────────────────────────────────


async def _append_log(
    tenant_id: str, entry: str, author_id: str, stats: CurationStats | None = None,
) -> None:
    """append-only 更新 log 页；log 自身故障不再抛出（避免吞掉原始异常）。"""

    try:
        existing = await db.get_wiki_page(tenant_id, cm.LOG_SLUG)
        if existing is None:
            fm, body = cm.log_page_initial()
            title = LOG_TITLE
        else:
            fm, body = parse_frontmatter(existing["content"])
            if fm is None:  # log 页 frontmatter 被人工剥离：重建系统 fm，正文保留
                fm, _ = cm.log_page_initial()
            title = existing["title"]
        content = _page_content(fm, cm.append_log(body, entry))
        await _upsert_page(tenant_id, cm.LOG_SLUG, title, content, author_id,
                          stats or CurationStats(source_id=""), count=False)
    except Exception:  # noqa: BLE001 - 日志写入失败不能掩盖主流程结果
        logger.exception("写入沉淀 log 页失败 tenant=%s", tenant_id)


async def _rebuild_index(
    tenant_id: str, pages: list[dict[str, Any]], author_id: str, stats: CurationStats,
) -> None:
    """按全量页面重新渲染 index（排除 index/log 自身）。"""

    entries: list[dict[str, Any]] = []
    for page in pages:
        fm, body = parse_frontmatter(page["content"])
        kind = fm.get("kind") if fm else None
        entry = {"slug": page["slug"], "title": page["title"], "kind": kind}
        if kind == "entity":
            entry["entity_type"] = cm.entity_type_from_body(body)
        entries.append(entry)
    fm, body = cm.render_index(entries)
    content = _page_content(fm, body)
    await _upsert_page(tenant_id, cm.INDEX_SLUG, INDEX_TITLE, content, author_id,
                       stats, count=False)


# ── 主编排 ───────────────────────────────────────────────────────────

# 沉淀页面正文大小上限（与人工 Wiki 写路径 _CONTENT_MAX 对齐）
_CURATION_PAGE_MAX = 200_000
# source_id 会落进 frontmatter sources，单项长度受 wiki_schema _SOURCE_MAX 约束
_SOURCE_ID_MAX = 300


async def curate_document(
    tenant_id: str,
    source_id: str,
    *,
    author_id: str = cm.SYSTEM_AUTHOR,
    extractor: Extractor | None = None,
    timeout: float | None = None,
) -> CurationStats:
    """对单篇原始文档执行 LLM 沉淀。调用方负责开关判断。

    :param timeout: 解析+LLM 提取阶段的秒级超时；None 时取配置。
        超时只可能取消提取阶段；其后的页面写入在 shield 段内完成，
        不会因超时留下半成品。
    :raises CurationExtractError: 解析或提取失败（不产生任何页面）。
    :raises FileNotFoundError: source_id 在租户目录下不存在。
    :raises ValueError: source_id 指向 wiki/ 区、超长或后缀不受支持。
    """

    extractor = extractor or extract_curated
    stats = CurationStats(source_id=source_id)
    lock = _tenant_lock(tenant_id)
    async with lock:
        # 跨进程互斥：postgres 用 advisory lock（多副本部署下同租户串行）；
        # sqlite 单机恒成功。未获取说明另一进程正在沉淀同租户文档。
        async with db.try_named_lock(f"curation:{tenant_id}") as acquired:
            if not acquired:
                raise CurationBusyError(
                    f"租户 {tenant_id} 已有沉淀任务在另一进程执行中"
                )
            return await _curate_locked(
                tenant_id, source_id, author_id, extractor, stats, timeout=timeout
            )


def _page_rejected(
    content: str, stats: CurationStats, *, kind: str, where: str
) -> bool:
    """沉淀页写入前的内容守卫：200KB 上限 + 凭据扫描。"""

    if len(content) > _CURATION_PAGE_MAX:
        stats.blocked += 1
        logger.warning(
            "%s 超过 %d 字符上限，已拒绝写入: where=%s size=%d",
            kind, _CURATION_PAGE_MAX, where, len(content),
        )
        return True
    if scan_secret(content):
        stats.blocked += 1
        logger.warning("%s 命中凭据扫描，已拒绝写入: where=%s", kind, where)
        return True
    return False


async def _curate_locked(
    tenant_id: str, source_id: str, author_id: str,
    extractor: Extractor, stats: CurationStats, *, timeout: float | None,
) -> CurationStats:
    settings = get_settings()
    timeout = settings.wiki_curation_timeout_seconds if timeout is None else timeout
    started = time.monotonic()
    timestamp = _now().strftime("%Y-%m-%d %H:%M:%S")
    today = _now().date().isoformat()

    # 1. 入参校验（纯本地快速操作，立即失败，不写任何页面）
    if kc.is_wiki_source(source_id):
        raise ValueError("wiki/ 镜像区不是原始文档，不能沉淀")
    if len(source_id) > _SOURCE_ID_MAX:
        raise ValueError(f"source_id 超过 {_SOURCE_ID_MAX} 字符上限")
    path = kc.resolve_tenant_file(tenant_id, source_id)
    if not path.is_file():
        raise FileNotFoundError(f"原始文档不存在: {source_id}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文档类型: {path.suffix}")

    # 2. 解析 + 页面加载 + LLM 提取：唯一可长耗时阶段，超时在此生效
    try:
        with anyio.fail_after(timeout):
            try:
                text = (await anyio.to_thread.run_sync(extract_text, path)).strip()
            except Exception as exc:  # noqa: BLE001
                raise CurationExtractError(f"文档解析失败: {exc}") from exc

            if not text:
                stats.skipped = "正文为空"
                await _append_log(
                    tenant_id,
                    cm.log_entry(timestamp, source_id, False, reason="正文为空，跳过沉淀"),
                    author_id,
                )
                logger.info(
                    "Wiki 沉淀跳过 tenant=%s source=%s reason=empty elapsed=%.2fs",
                    tenant_id, source_id, time.monotonic() - started,
                )
                return stats

            truncated = len(text) > settings.wiki_curation_max_chars
            text = text[: settings.wiki_curation_max_chars]

            pages = await _load_pages_full(tenant_id)
            ent_index = _entity_index(pages)
            existing_entities = [
                ExistingEntity(
                    slug=info["page"]["slug"],
                    name=info["page"]["title"],
                    aliases=info["fm"].get("aliases", []),
                    conclusion=info["body"][:_CONCLUSION_LIMIT],
                )
                for info in ent_index.values()
            ]

            extraction = await extractor(text, source_id, existing_entities)
    except TimeoutError:
        reason = f"沉淀超时（>{timeout:g}s）"
        with anyio.CancelScope(shield=True):
            await _append_log(
                tenant_id, cm.log_entry(timestamp, source_id, False, reason=reason),
                author_id,
            )
        logger.error(
            "Wiki 沉淀超时 tenant=%s source=%s elapsed=%.2fs",
            tenant_id, source_id, time.monotonic() - started,
        )
        raise
    except Exception as exc:
        reason = str(exc) if isinstance(exc, CurationExtractError) else f"提取异常: {exc}"
        with anyio.CancelScope(shield=True):
            await _append_log(
                tenant_id,
                cm.log_entry(timestamp, source_id, False, reason=reason),
                author_id,
            )
        logger.exception(
            "Wiki 沉淀提取失败 tenant=%s source=%s elapsed=%.2fs",
            tenant_id, source_id, time.monotonic() - started,
        )
        raise

    # 3. 写入阶段：全部为本地快速操作；shield 防止外部取消留下残页
    with anyio.CancelScope(shield=True):
        return await _write_curated_pages(
            tenant_id=tenant_id,
            source_id=source_id,
            author_id=author_id,
            stats=stats,
            pages=pages,
            ent_index=ent_index,
            extraction=extraction,
            truncated=truncated,
            timestamp=timestamp,
            today=today,
            started=started,
        )


async def _write_curated_pages(
    *,
    tenant_id: str,
    source_id: str,
    author_id: str,
    stats: CurationStats,
    pages: list[dict[str, Any]],
    ent_index: dict[str, dict[str, Any]],
    extraction: Any,
    truncated: bool,
    timestamp: str,
    today: str,
    started: float,
) -> CurationStats:
    """提取成功后的全部写入动作（在不可取消的 shield 段内执行）。"""

    # 4. 文档摘要页 slug：先按 sources 回链复用（抗同名碰撞），占用冲突再消歧
    doc_title = PurePosixPath(source_id.replace("\\", "/")).stem
    existing_doc = _find_doc_page(pages, source_id)
    if existing_doc is not None:
        stats.doc_slug = existing_doc["slug"]
    else:
        stats.doc_slug = cm.doc_slug(source_id)
        if any(p["slug"] == stats.doc_slug for p in pages):
            stats.doc_slug = cm.doc_slug_disambiguated(source_id)

    # 先决定每个抽取实体的目标 slug（用于摘要页链接与后续 upsert）
    targets: list[dict[str, Any]] = []
    for ent in extraction.entities:
        keys = {cm.normalize_entity_name(ent.name)}
        keys.update(cm.normalize_entity_name(a) for a in ent.aliases)
        matched = next((ent_index[k] for k in keys if k in ent_index), None)
        targets.append(
            {
                "extraction": ent,
                "matched": matched,
                "slug": matched["page"]["slug"] if matched else cm.entity_slug(ent.name),
            }
        )

    doc_fm, doc_body = cm.build_doc_page(
        source_id,
        extraction.summary,
        extraction.key_points,
        [(t["extraction"].name, t["slug"]) for t in targets],
    )
    doc_content = _page_content(doc_fm, doc_body)
    if not _page_rejected(doc_content, stats, kind="摘要页",
                          where=f"{tenant_id}/{source_id}"):
        await _upsert_page(tenant_id, stats.doc_slug, doc_title, doc_content,
                           author_id, stats)

    # 5. 实体页 upsert（合并 sources/aliases；未裁决矛盾区块随再沉淀迁移）
    for target in targets:
        ent = target["extraction"]
        matched = target["matched"]
        slug = target["slug"]
        if matched is None:
            sources = [source_id]
            aliases = cm.entity_aliases(ent.name, ent.aliases)
            old_sources: list[str] = []
        else:
            sources = list(dict.fromkeys(
                [*matched["fm"].get("sources", []), source_id]
            ))[:30]
            aliases = cm.entity_aliases(
                ent.name, [*matched["fm"].get("aliases", []), *ent.aliases]
            )[:20]
            old_sources = matched["fm"].get("sources", [])

        fm, body = cm.build_entity_page(
            ent.name, ent.type, aliases, ent.summary, ent.points, sources
        )
        if matched is not None:
            # 安全不变量：旧页未裁决区块必须随重建迁移，未经裁决不得消失
            body = cm.carry_open_contradictions(body, matched["body"])
        if ent.contradiction and matched is not None:
            old_claim = matched["body"][:_CONCLUSION_LIMIT].strip()
            old_source = "、".join(old_sources) if old_sources else matched["page"]["slug"]
            block = cm.wrap_contradiction(
                today, ent.contradiction, source_id, old_claim, old_source
            )
            if block.strip() not in body:
                body = cm.append_contradiction(body, block)
                stats.contradictions += 1

        content = _page_content(fm, body)
        if _page_rejected(content, stats, kind="实体页",
                          where=f"{tenant_id}/{ent.name}"):
            continue
        await _upsert_page(tenant_id, slug, ent.name, content, author_id, stats)
        stats.entities += 1

    # 6. 重建 index、追加 log
    pages_after = await _load_pages_full(tenant_id)
    pages_after = [p for p in pages_after if p["slug"] != cm.INDEX_SLUG]
    await _rebuild_index(tenant_id, pages_after, author_id, stats)

    log_detail = stats.to_dict()
    suffix = "（正文超长已截断）" if truncated else ""
    await _append_log(
        tenant_id,
        cm.log_entry(timestamp, source_id, True, log_detail) + suffix,
        author_id,
    )

    # 7. 统一刷新索引（未初始化则跳过）
    stats.refreshed = await kc.arequest_refresh(tenant_id)
    logger.info(
        "Wiki 沉淀完成 tenant=%s source=%s created=%d updated=%d entities=%d "
        "contradictions=%d blocked=%d elapsed=%.2fs",
        tenant_id, source_id, stats.created, stats.updated, stats.entities,
        stats.contradictions, stats.blocked, time.monotonic() - started,
    )
    return stats
