"""LLM Wiki 沉淀的手动触发 API。

- ``POST /knowledge/curate``：对单篇原始文档同步执行沉淀，返回结构化统计；
- ``POST /knowledge/curate-all``：枚举租户根下全部受支持原始文档（排除
  ``wiki/`` 镜像保留区），逐篇串行沉淀，返回每篇结果明细。

与上传后 fire-and-forget 自动调度共用同一 ``service.curate_document``
（内含 per-tenant 锁、超时由本端点以 anyio 围栏兜底），所有操作限定在
当前用户所属租户内。
"""

from __future__ import annotations

import logging

import anyio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.auth import CurrentUser
from src.curation.llm import CurationExtractError
from src.curation.service import CurationBusyError, CurationStats, curate_document
from src.rag import knowledge_contract as kc
from src.rag.config import get_settings
from src.rag.parser import SUPPORTED_SUFFIXES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge-curation"])


# ── 请求/响应模型 ───────────────────────────────────────────────────


class CurateRequest(BaseModel):
    """单篇沉淀请求：source_id 为相对租户根的 POSIX 相对路径。"""

    source_id: str = Field(min_length=1, max_length=300)


class CurationStatsResponse(BaseModel):
    """单篇沉淀统计（字段对齐 service.CurationStats）。"""

    source_id: str
    doc_slug: str
    created: int
    updated: int
    entities: int
    contradictions: int
    blocked: int
    refreshed: bool
    skipped: str


class CurationResultItem(BaseModel):
    """批量沉淀中单篇结果：status ∈ ok / skipped / failed。"""

    source_id: str
    status: str
    detail: str = ""
    stats: CurationStatsResponse | None = None


class CurationBatchResponse(BaseModel):
    """批量沉淀汇总 + 每篇明细。"""

    total: int
    succeeded: int
    skipped: int
    failed: int
    results: list[CurationResultItem]


# ── 内部工具 ─────────────────────────────────────────────────────────


def _ensure_enabled() -> None:
    if not get_settings().wiki_curation_enabled:
        raise HTTPException(status_code=409, detail="Wiki 自动沉淀功能未开启")


def _resolve_raw_document(tenant_id: str, source_id: str) -> str:
    """校验 source_id 指向租户内受支持的原始文档，返回规范化标识。

    - wiki/ 保留区 → 400；
    - 路径越界（含跨租户 ``../``）/不存在 → 404（不暴露目录结构）；
    - 后缀不受支持 → 400。
    """

    normalized = (source_id or "").replace("\\", "/").lstrip("/")
    if not normalized or normalized in {".", ".."}:
        raise HTTPException(status_code=400, detail="非法的文件标识")
    if len(normalized) > 300:
        raise HTTPException(status_code=400, detail="source_id 超过 300 字符上限")
    if kc.is_wiki_source(normalized):
        raise HTTPException(
            status_code=400,
            detail="wiki/ 镜像区不是原始文档，不能沉淀",
        )
    try:
        path = kc.resolve_tenant_file(tenant_id, normalized)
    except kc.KnowledgePathError:
        # 越界（含尝试穿越到其他租户）一律按不可见处理
        raise HTTPException(status_code=404, detail="文件不存在") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    return normalized


def _list_raw_documents(tenant_id: str) -> list[str]:
    """枚举租户根下全部受支持原始文档（排除 wiki/ 镜像区）。"""

    root = kc.tenant_root(tenant_id)
    if not root.exists():
        return []
    source_ids: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        if kc.is_wiki_source(rel):
            continue
        source_ids.append(rel)
    return source_ids


def _stats_response(stats: CurationStats) -> CurationStatsResponse:
    return CurationStatsResponse(**stats.to_dict())


# ── 端点 ─────────────────────────────────────────────────────────────


@router.post("/curate", response_model=CurationStatsResponse)
async def curate_knowledge(
    body: CurateRequest, user: CurrentUser
) -> CurationStatsResponse:
    """对单篇原始文档同步执行 LLM Wiki 沉淀。"""

    _ensure_enabled()
    tenant_id = user["tenant_id"]
    source_id = _resolve_raw_document(tenant_id, body.source_id)

    settings = get_settings()
    try:
        # service 内对"解析+提取"精确超时；外层仅作整体兜底（写页 shield 不被打断）
        with anyio.fail_after(settings.wiki_curation_timeout_seconds + 30.0):
            stats = await curate_document(
                tenant_id, source_id,
                timeout=settings.wiki_curation_timeout_seconds,
            )
    except TimeoutError:
        logger.warning("手动沉淀超时: tenant=%s source=%s", tenant_id, source_id)
        raise HTTPException(status_code=504, detail="沉淀超时，请稍后重试") from None
    except CurationBusyError as exc:
        logger.warning("手动沉淀冲突（另一进程执行中）: tenant=%s source=%s",
                       tenant_id, source_id)
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="文件不存在") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except CurationExtractError as exc:
        logger.warning("手动沉淀提取失败: tenant=%s source=%s: %s",
                       tenant_id, source_id, exc)
        raise HTTPException(status_code=502, detail=f"LLM 提取失败: {exc}") from None

    logger.info(
        "手动沉淀完成: tenant=%s source=%s by=%s created=%d updated=%d",
        tenant_id, source_id, user.get("email"), stats.created, stats.updated,
    )
    return _stats_response(stats)


@router.post("/curate-all", response_model=CurationBatchResponse)
async def curate_all_knowledge(user: CurrentUser) -> CurationBatchResponse:
    """对租户内全部原始文档逐篇串行沉淀（与自动任务共用 per-tenant 锁）。"""

    _ensure_enabled()
    tenant_id = user["tenant_id"]
    source_ids = _list_raw_documents(tenant_id)
    timeout = get_settings().wiki_curation_timeout_seconds

    results: list[CurationResultItem] = []
    succeeded = skipped = failed = 0
    for source_id in source_ids:
        try:
            # 逐篇超时围栏（提取阶段精确超时 + 30s 整体兜底），单篇挂起不拖垮整批
            with anyio.fail_after(timeout + 30.0):
                stats = await curate_document(tenant_id, source_id, timeout=timeout)
        except TimeoutError:
            failed += 1
            results.append(CurationResultItem(
                source_id=source_id, status="failed",
                detail=f"沉淀超时（>{timeout:g}s），已跳过",
            ))
        except CurationBusyError as exc:
            failed += 1
            results.append(CurationResultItem(
                source_id=source_id, status="failed", detail=str(exc),
            ))
        except CurationExtractError as exc:
            failed += 1
            results.append(CurationResultItem(
                source_id=source_id, status="failed", detail=str(exc),
            ))
        except (FileNotFoundError, ValueError) as exc:
            # 枚举后文件被删/类型异常：记为失败但不中断整批
            failed += 1
            results.append(CurationResultItem(
                source_id=source_id, status="failed", detail=str(exc),
            ))
        else:
            if stats.skipped:
                skipped += 1
                results.append(CurationResultItem(
                    source_id=source_id, status="skipped",
                    detail=stats.skipped, stats=_stats_response(stats),
                ))
            else:
                succeeded += 1
                results.append(CurationResultItem(
                    source_id=source_id, status="ok",
                    stats=_stats_response(stats),
                ))

    return CurationBatchResponse(
        total=len(source_ids),
        succeeded=succeeded,
        skipped=skipped,
        failed=failed,
        results=results,
    )
