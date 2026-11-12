"""暴露给 LLM 调用的 LangChain Tool。

retrieve_knowledge 被注册为 LangGraph 子代理的工具，
LLM 在回答问题时会决定是否调用它来检索知识库。
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from functools import partial
from typing import Any

import anyio
from langchain_core.tools import tool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.rag.config import get_settings
from src.rag.pipeline import KnowledgeBase
from src.rag.query_enhancer import QueryEnhancer

logger = logging.getLogger(__name__)

# 检索来源发射器：API 流式端点在自己的 context 树中注册回调，
# retrieve_knowledge 命中后调用。deepagents 的 task 子代理在工具内部
# ainvoke，其内部消息/custom 事件不会冒泡到外层 astream，因此用
# contextvar 沿 async 调用链（含 anyio 线程池复制的 context）传递。
# 回调可能在工作线程中被调用，实现方需自行保证线程安全。
SourceEmitter = Callable[[list[dict[str, Any]]], None]
_source_emitters: ContextVar[list[SourceEmitter] | None] = ContextVar(
    "rag_source_emitters", default=None
)


@contextlib.contextmanager
def source_emitter(callback: SourceEmitter) -> Iterator[None]:
    """在当前 context 树内注册一个检索来源回调（退出时自动注销）。"""

    current = _source_emitters.get() or []
    token = _source_emitters.set([*current, callback])
    try:
        yield
    finally:
        _source_emitters.reset(token)

# 按租户缓存知识库实例：每个租户独立目录与索引，互不干扰
_KB_CACHE: dict[str, KnowledgeBase] = {}

# 查询增强器全局单例（LLM 懒加载，失败降级为直通）
_enhancer: QueryEnhancer | None = None


def _get_enhancer() -> QueryEnhancer:
    """懒加载查询增强器单例。"""

    global _enhancer
    if _enhancer is None:
        _enhancer = QueryEnhancer()
    return _enhancer


def get_knowledge_base(tenant_id: str) -> KnowledgeBase:
    """懒加载指定租户的 KnowledgeBase，首次调用时加载该租户目录并构建索引。"""

    kb = _KB_CACHE.get(tenant_id)
    if kb is not None:
        return kb
    settings = get_settings()
    tenant_root = settings.knowledge_dir / tenant_id
    logger.info("首次初始化租户知识库单例（懒加载）: tenant=%s", tenant_id)
    kb = KnowledgeBase(
        tenant_root,
        embedding=settings.rag_embedding,
        reranker=settings.rag_reranker,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )
    _KB_CACHE[tenant_id] = kb
    return kb


def is_knowledge_base_initialized(tenant_id: str) -> bool:
    """判断指定租户的知识库是否已初始化（用于上传后决定是否同步刷新）。"""

    return tenant_id in _KB_CACHE


async def _search_with_timeout(tenant_id: str, query: str, top_k: int) -> str:
    """在线程池中执行同步检索，加上超时控制避免知识库卡死主流程。

    检索前先做查询增强（HyDE 假想答案 + 查询扩展），并把增强查询作为
    CRAG 自我纠正的备选查询传入。

    命中后通过 ``get_stream_writer()`` 发出 custom 事件
    ``{"type": "rag_sources", ...}``，使来源能穿透 task 子图边界
    供 /chat/stream 捕获（子图内部 ToolMessage 不会冒泡到主流）；
    非流式上下文（如 /chat）获取 writer 失败时静默跳过。
    """

    settings = get_settings()
    logger.info(
        "检索工具执行: tenant=%s, query=%r, top_k=%d",
        tenant_id, query[:80], top_k,
    )

    # 查询理解 + 增强（QUERY_UNDERSTAND）：在线程池中执行 LLM 调用。
    # 开关关闭（默认）时零 LLM 调用；开启时套独立短超时，
    # 超时/失败一律降级为原始查询，绝不阻断检索主流程。
    def _enhance() -> tuple[str, str | None, str | None, list[str]]:
        if not settings.query_understand_enabled:
            return query, None, None, []
        try:
            enhancer = _get_enhancer()
            rewritten = enhancer.rewrite(query)
            hyde = enhancer.generate_hypothetical(query)
            expanded = enhancer.expand_query(query)
            main_query = rewritten or query
            # CRAG 备选查询：原查询保底 + 扩展 + 假想答案，覆盖不同语义角度
            enhanced = [
                q for q in (query, expanded, hyde) if q and q != main_query
            ][:2]
            return main_query, hyde, expanded, enhanced
        except Exception as exc:  # noqa: BLE001
            logger.warning("查询理解失败，降级为原查询: %s", exc)
            return query, None, None, []

    async def _enhance_guarded() -> tuple[str, str | None, str | None, list[str]]:
        if not settings.query_understand_enabled:
            return query, None, None, []
        try:
            with anyio.fail_after(settings.query_understand_timeout_seconds):
                # abandon_on_cancel=True：超时立即放弃等待（线程自灭、结果忽略），
                # 否则 anyio 会等线程跑完导致超时围栏失效
                return await anyio.to_thread.run_sync(
                    _enhance, abandon_on_cancel=True
                )
        except TimeoutError:
            logger.warning(
                "查询理解超时（%.1fs），降级为原查询: tenant=%s, query=%r",
                settings.query_understand_timeout_seconds, tenant_id, query[:80],
            )
            return query, None, None, []
        except Exception as exc:  # noqa: BLE001
            logger.warning("查询理解异常，降级为原查询: %s", exc)
            return query, None, None, []

    try:
        with anyio.fail_after(settings.rag_timeout_seconds):
            search_query, hyde, expanded, enhanced = await _enhance_guarded()
            kb = get_knowledge_base(tenant_id)
            results = await anyio.to_thread.run_sync(
                partial(
                    kb.search,
                    search_query,
                    top_k,
                    dense_query=hyde,
                    sparse_query=expanded,
                    enhanced_queries=enhanced,
                )
            )
    except TimeoutError:
        logger.error(
            "知识库检索超时（%.0fs）: tenant=%s, query=%r",
            settings.rag_timeout_seconds, tenant_id, query[:80],
        )
        raise
    if not results:
        logger.warning("检索无命中: tenant=%s, query=%r", tenant_id, query[:80])
        return json.dumps(
            {"status": "no_results", "query": query, "results": []}, ensure_ascii=False
        )
    logger.info("检索工具返回 %d 个片段", len(results))

    # 向流式调用方透传来源（非流式上下文 get_stream_writer 会抛异常，静默跳过）
    seen: set[str] = set()
    slim_sources: list[dict[str, Any]] = []
    for item in results:
        d = item.to_dict()
        source = str(d.get("source") or "")
        if not source or source in seen:
            continue
        seen.add(source)
        md = d.get("metadata") or {}
        slim_sources.append(
            {
                "source": source,
                "section": md.get("section_title") or md.get("section_path") or "",
                "score": d.get("score"),
                "tier": d.get("tier", "raw"),
            }
        )
    # 向流式调用方透传来源（无注册回调时为空列表，静默跳过；
    # 回调可能运行在 anyio 工作线程，由回调实现保证线程安全）
    for emitter in list(_source_emitters.get() or []):
        try:
            emitter(slim_sources)
        except Exception:  # noqa: BLE001 - 来源推送失败不影响检索主流程
            logger.debug("检索来源回调执行失败", exc_info=True)

    return json.dumps(
        {"status": "ok", "query": query, "results": [item.to_dict() for item in results]},
        ensure_ascii=False,
    )


def _log_retry(retry_state) -> None:
    """tenacity 重试前回调：记录第几次重试与异常原因。"""

    logger.warning(
        "知识库检索调用失败，准备第 %d 次重试: %s",
        retry_state.attempt_number,
        retry_state.outcome.exception() if retry_state.outcome else "unknown",
    )


@tool
@retry(
    retry=retry_if_exception_type((TimeoutError, OSError, RuntimeError)),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    stop=stop_after_attempt(get_settings().rag_max_retries),
    before_sleep=_log_retry,
    reraise=True,
)
async def retrieve_knowledge(query: str, top_k: int = 6) -> str:
    """Search the internal knowledge base and return JSON snippets with explicit sources.

    供子代理调用的检索工具，返回 JSON 字符串，每条带 source_id 方便主代理标注来源。
    内置 tenacity 重试和 anyio 超时，保证高可用。
    检索范围严格限定为当前登录用户所属租户的知识库。

    流式调用（/chat/stream）时，命中来源会经 get_stream_writer() 以
    rag_sources 自定义事件额外透传给外层，无需调用方传入任何参数。
    """

    from langgraph.runtime import get_runtime

    tenant_id = get_runtime().context.tenant_id
    return await _search_with_timeout(tenant_id, query, min(max(top_k, 1), 20))