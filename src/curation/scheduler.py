"""上传后 fire-and-forget 调度 LLM Wiki 沉淀。

范式对齐 src.memory.reflection：
- 开关关闭 / 同一 (tenant, source_id) 已有任务在跑 → 不调度；
- 去重持久化在 DB（wiki_curation_tasks 唯一约束 + lease），多副本部署下
  各进程共享槽位：同一文档全局只有一个运行中任务；worker 崩溃后残留
  行超过 lease 期限可被重新抢占，任务不会因重启而永久丢失；
- anyio 总超时围栏；任何异常只记日志，绝不冒泡到上传端点；
- 任务体执行 service.curate_document，其内部 per-tenant 锁保证
  同租户沉淀串行（进程内 asyncio.Lock + postgres advisory lock）。
"""

from __future__ import annotations

import asyncio
import logging
import time

import anyio

from src.curation.service import CurationBusyError, curate_document
from src.persistence import database as db
from src.rag.config import get_settings

logger = logging.getLogger(__name__)

# 写页阶段在 service 内受 shield 保护；这里额外留出一个后台兜底宽限
_BACKSTOP_MARGIN_SECONDS = 30.0
# 任务槽位 lease：显著大于单次沉淀最大耗时（超时 + 兜底宽限），
# 只用于回收崩溃 worker 的残留行，正常运行的任务总会先自行释放
_LEASE_SLACK_SECONDS = 120.0

# 进程内运行中的沉淀任务句柄（同进程快速去重 + 测试/停机回收用）
_running: dict[tuple[str, str], asyncio.Task] = {}


def _lease_seconds() -> float:
    timeout = get_settings().wiki_curation_timeout_seconds
    return timeout + _BACKSTOP_MARGIN_SECONDS + _LEASE_SLACK_SECONDS


async def _curate_guarded(key: tuple[str, str], tenant_id: str, source_id: str) -> None:
    """带超时与异常围栏的后台任务体；结束时释放并发槽位。"""

    timeout = get_settings().wiki_curation_timeout_seconds
    started = time.monotonic()
    try:
        # service 内部对"解析+提取"精确超时；外层仅作整体兜底（写页 shield 不被打断）
        with anyio.fail_after(timeout + _BACKSTOP_MARGIN_SECONDS):
            await curate_document(tenant_id, source_id, timeout=timeout)
    except CurationBusyError:
        logger.info(
            "Wiki 沉淀跳过: 另一进程正在执行同租户沉淀 tenant=%s source=%s",
            tenant_id, source_id,
        )
    except TimeoutError:
        logger.error(
            "Wiki 沉淀超时: tenant=%s source=%s elapsed=%.2fs",
            tenant_id, source_id, time.monotonic() - started,
        )
    except Exception:  # noqa: BLE001 - 后台任务失败不影响上传主链路
        logger.exception(
            "Wiki 沉淀失败（不影响上传）: tenant=%s source=%s elapsed=%.2fs",
            tenant_id, source_id, time.monotonic() - started,
        )
    finally:
        _running.pop(key, None)
        try:
            await db.release_curation_task(tenant_id, source_id)
        except Exception:  # noqa: BLE001 - 残留行由 lease 过期兜底回收
            logger.exception("释放沉淀任务槽位失败: source=%s", source_id)


async def schedule_curation(tenant_id: str, source_id: str) -> bool:
    """上传成功后调度后台沉淀，返回是否成功调度。

    - 开关关闭返回 False；
    - 同文档已有任务在跑（进程内句柄或 DB 槽位被任一 worker 持有）返回
      False，不排队——原始文档已落盘，下次上传/手动触发可重试；
    - 任务内异常全部就地消化，调用方无需等待任务完成。
    """

    if not get_settings().wiki_curation_enabled:
        logger.info("Wiki 沉淀已关闭，跳过调度: source=%s", source_id)
        return False
    key = (tenant_id, source_id)
    if key in _running:
        logger.info("Wiki 沉淀跳过: 该文档已有任务在运行 source=%s", source_id)
        return False
    # 跨进程去重：DB 槽位被其他 worker 持有（含崩溃残留未过 lease）时拒绝
    if not await db.claim_curation_task(tenant_id, source_id, _lease_seconds()):
        logger.info(
            "Wiki 沉淀跳过: 该文档槽位已被其他 worker 持有 source=%s", source_id
        )
        return False
    task = asyncio.create_task(_curate_guarded(key, tenant_id, source_id))
    _running[key] = task
    logger.info("Wiki 沉淀已调度: tenant=%s source=%s", tenant_id, source_id)
    return True


def is_curation_running(tenant_id: str, source_id: str) -> bool:
    """该文档沉淀任务是否仍在运行（测试与手动入口用；仅本进程可见）。"""

    return (tenant_id, source_id) in _running
