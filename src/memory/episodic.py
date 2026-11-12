"""跨会话情景记忆工具（episodic memory）。

与语义事实记忆（:mod:`src.memory.tools`）的分工：

- **事实记忆**：从对话提炼后的单句事实，结构化条目、向量召回，回答"是什么"；
- **情景记忆（本模块）**：历史会话本体，按用户提问检索过去的对话线程，
  回答"上次我们是怎么解决这个问题的"，保留原始对话上下文而非提炼结论。

数据来源与隔离：

- 会话元数据（标题/更新时间/归属）来自业务库 ``threads`` 表，
  经 :func:`db.search_user_threads` 按 ``(tenant_id, user_id)`` 显式过滤，
  工具不接受外部传入的 thread_id 列表，跨租户/跨用户线程在 SQL 层即被拦截；
- 消息本体来自 LangGraph checkpoint，通过代理图的 ``aget_state_history``
  读取每个候选线程的最新快照；代理图在应用启动时由
  :func:`register_agent_graph` 注册（依赖倒置，本模块不反向 import agents）；
- 默认排除当前 ``thread_id``，避免把正在进行的对话当作"历史"召回。

可靠性：全流程带超时（``memory_episodic_timeout_seconds``）与 tenacity 重试；
功能关闭或图未注册时返回可读提示，不向主对话抛异常。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anyio
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.memory import retrieval
from src.memory.reflection import _coerce_messages
from src.memory.tools import _current_identity, _memory_retry
from src.persistence import database as db
from src.rag.config import get_settings

logger = logging.getLogger("src.memory")

# 每个候选会话最多回溯的消息条数（末尾 N 条，与反思提取口径一致量级）
_MAX_MESSAGES_PER_THREAD = 20

# 应用启动时注册的代理图（编译后的 LangGraph 图，具备 aget_state_history）
_agent_graph: Any = None


def register_agent_graph(graph: Any) -> None:
    """注册代理图，供 checkpoint 历史读取（build_agent 完成后调用一次）。"""

    global _agent_graph
    _agent_graph = graph
    logger.info("情景记忆已绑定代理图: %s", getattr(graph, "name", graph))


def _get_graph() -> Any:
    return _agent_graph


async def _latest_snapshot(graph: Any, thread_id: str) -> Any:
    """读取某线程最新一个检查点快照；线程无 checkpoint 返回 None。"""

    config = {"configurable": {"thread_id": thread_id}}
    async for snapshot in graph.aget_state_history(config):
        return snapshot
    return None


def _build_excerpt(conversation: list[dict[str, str]], max_chars: int) -> str:
    """把 role/content 对话渲染为带角色标签的摘录，并按字符上限截断。"""

    lines = [
        f"{'用户' if item['role'] == 'user' else '助手'}：{item['content']}"
        for item in conversation
    ]
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "……（摘录已截断）"
    return text


async def _search_past_conversations(
    tenant_id: str,
    user_id: str,
    query: str,
    k: int | None = None,
    current_thread_id: str | None = None,
) -> str:
    """跨会话检索当前用户的历史对话（显式传身份，便于单测直接调用）。"""

    settings = get_settings()
    if not settings.memory_episodic_enabled:
        return "历史会话检索功能未开启。"
    if _get_graph() is None:
        # 图尚未注册（如脱离应用上下文的脚本环境），不抛异常打断主对话
        return "历史会话检索尚未就绪（代理图未注册）。"

    query = (query or "").strip()
    if not query:
        return "检索失败：查询内容为空。"
    k = min(max(int(k or settings.memory_episodic_top_k), 1), 10)
    if not retrieval.tokenize(query):
        return "检索失败：查询内容为空。"

    async def _work() -> list[dict[str, Any]]:
        updated_after = (
            time.time() - settings.memory_episodic_since_days * 24 * 3600
        )
        # 候选线程全部来自归属查询：tenant_id 显式过滤，杜绝跨租户读取
        threads = await db.search_user_threads(
            tenant_id,
            user_id,
            limit=settings.memory_episodic_max_threads,
            updated_after=updated_after,
        )
        graph = _get_graph()
        candidates: list[dict[str, Any]] = []
        for thread in threads:
            if thread["id"] == current_thread_id:
                continue
            snapshot = await _latest_snapshot(graph, thread["id"])
            if snapshot is None or not snapshot.values:
                continue
            conversation = _coerce_messages(
                snapshot.values.get("messages", []), _MAX_MESSAGES_PER_THREAD
            )
            if not conversation:
                continue
            excerpt = _build_excerpt(
                conversation, settings.memory_episodic_max_chars
            )
            candidates.append(
                {
                    "key": thread["id"],
                    # BM25 语料 = 标题 + 对话摘录；IDF 在整个候选集上计算，
                    # 稀有词的区分度高于旧的逐篇 token 覆盖度
                    "text": thread["title"] + "\n" + excerpt,
                    "thread": thread,
                    "excerpt": excerpt,
                }
            )
        ranked = retrieval.bm25_rank(query, candidates, k)
        by_key = {candidate["key"]: candidate for candidate in candidates}
        max_score = ranked[0][1] if ranked else 0.0
        matched = [
            {
                # min-max 归一化到 0~1；单命中时直接给满分，
                # 避免 BM25 平移基准为 0 时展示成 0.00
                "score": (
                    1.0 if len(ranked) == 1
                    else (score / max_score if max_score > 0 else 0.0)
                ),
                "thread": by_key[key]["thread"],
                "excerpt": by_key[key]["excerpt"],
            }
            for key, score in ranked
        ]
        # 归一化后平分时，最近更新的会话优先（确定性次序）
        matched.sort(
            key=lambda item: (
                item["score"],
                item["thread"]["updated_at"],
            ),
            reverse=True,
        )
        return matched

    try:
        with anyio.fail_after(settings.memory_episodic_timeout_seconds):
            matched = await _memory_retry()(_work)()
    except TimeoutError:
        logger.error("search_past_conversations 超时: user_id=%s", user_id)
        return "检索失败：历史会话检索超时，请稍后重试。"
    except Exception as exc:
        logger.exception("search_past_conversations 失败: user_id=%s", user_id)
        return f"检索失败：历史会话服务暂时不可用（{type(exc).__name__}）。"

    logger.info(
        "search_past_conversations 完成: tenant=%s user_id=%s 命中=%d",
        tenant_id, user_id, len(matched),
    )
    if not matched:
        return (
            f"未在最近 {settings.memory_episodic_since_days} 天的历史会话中"
            "检索到相关内容。"
        )

    lines = [
        f"[历史会话检索] 共 {len(matched)} 个相关会话"
        f"（扫描窗口 {settings.memory_episodic_since_days} 天）："
    ]
    for i, item in enumerate(matched, 1):
        thread = item["thread"]
        updated = db._epoch_to_iso(thread["updated_at"]) or "未知时间"
        lines.append(
            f"\n{i}. {thread['title']}（会话 id={thread['id']}，"
            f"更新于 {updated}，相关度={item['score']:.2f}）\n{item['excerpt']}"
        )
    return "\n".join(lines)


@tool
async def search_past_conversations(
    query: str, runtime: ToolRuntime, k: int = 3
) -> str:
    """检索当前用户的历史会话记录，回答"上次/之前我们聊过/怎么解决的"类问题。

    跨线程扫描最近的历史对话（不含当前会话），返回相关会话的标题、
    更新时间与对话摘录；适合还原过去的排查过程与讨论上下文。
    事实型结论优先用 recall_memories，本工具只在需要原始对话上下文时使用。

    Args:
        query: 检索语句（自然语言问题或关键主题词）。
        k: 返回会话条数（1~10，默认 3）。
    """

    tenant_id, user_id = _current_identity()
    # thread_id 来自运行时 config（ToolNode 自动注入），不接受模型传参
    config = getattr(runtime, "config", None) or {}
    current_thread_id = (config.get("configurable") or {}).get("thread_id")
    return await _search_past_conversations(
        tenant_id, user_id, query, k, current_thread_id
    )


EPISODIC_TOOLS = [search_past_conversations]
