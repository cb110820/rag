"""跨会话情景记忆（episodic memory）测试。

覆盖：
1. 数据层：search_user_threads 的租户/用户显式过滤、时间窗裁剪；
2. 工具核心：跨线程主题命中、当前线程排除、跨租户隔离、开关/空查询兜底；
3. 运行时注入：ToolNode + ToolRuntime.config.thread_id 端到端。

测试通过真实 AsyncSqliteSaver 承载检查点，无外部模型下载。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from src.memory import episodic
from src.memory.episodic import search_past_conversations
from src.persistence import database as db

pytestmark = pytest.mark.usefixtures("isolated_db")

TENANT = db.DEFAULT_TENANT_ID


# ── 夹具：带真实 saver 的回声图 + 情景记忆注册 ──────────────────────


def _echo_graph():
    g = StateGraph(MessagesState)

    async def echo(state: dict) -> dict:
        return {"messages": [AIMessage(content="答复：" + state["messages"][-1].content)]}

    g.add_node("echo", echo)
    g.add_edge(START, "echo")
    g.add_edge("echo", END)
    return g.compile(checkpointer=db.get_saver())


@pytest_asyncio.fixture
async def registered_graph() -> AsyncIterator[object]:
    graph = _echo_graph()
    episodic.register_agent_graph(graph)
    try:
        yield graph
    finally:
        # 避免图实例跨用例泄漏
        episodic.register_agent_graph(None)


async def _make_user(email: str) -> str:
    return await db.create_user(
        email=email, password_hash="x", display_name=email.split("@")[0],
        tenant_id=TENANT,
    )


async def _seed_thread(
    graph: object, user_id: str, thread_id: str, title: str, user_text: str
) -> None:
    await db.create_thread(
        thread_id=thread_id, user_id=user_id, tenant_id=TENANT, title=title
    )
    await graph.ainvoke(
        {"messages": [HumanMessage(content=user_text)]},
        config={"configurable": {"thread_id": thread_id}},
    )


# ── 1. 数据层：线程归属查询 ─────────────────────────────────────────


async def test_search_user_threads_filters_tenant_user_and_window(
    registered_graph: object,
) -> None:
    uid = await _make_user("epi-1@example.com")
    await _seed_thread(
        registered_graph, uid, "thread-a", "Aurora 排障记录", "讨论 Aurora 连接超时"
    )
    await _seed_thread(
        registered_graph, uid, "thread-b", "咖啡机维修", "茶水间咖啡机故障"
    )

    rows = await db.search_user_threads(TENANT, uid, limit=10)
    assert {r["id"] for r in rows} == {"thread-a", "thread-b"}
    assert all(r["tenant_id"] == TENANT and r["user_id"] == uid for r in rows)

    # 显式租户过滤：错误租户查不到
    assert await db.search_user_threads("other-tenant", uid, limit=10) == []

    # 时间窗：把 thread-a 回退到 30 天前，1 天窗口内只剩 thread-b
    import time as _time

    await db._execute(
        "UPDATE threads SET updated_at = ? WHERE id = ?",
        (_time.time() - 30 * 86400, "thread-a"),
    )
    recent = await db.search_user_threads(
        TENANT, uid, limit=10, updated_after=_time.time() - 86400
    )
    assert [r["id"] for r in recent] == ["thread-b"]


# ── 2. 工具核心：跨线程命中 / 排除当前线程 / 隔离 / 兜底 ────────────


async def test_cross_thread_hit_ranks_by_topic(registered_graph: object) -> None:
    uid = await _make_user("epi-2@example.com")
    await _seed_thread(
        registered_graph, uid, "th-aurora", "Aurora 项目周会",
        "排查 Aurora 数据库连接池耗尽的根因",
    )
    await _seed_thread(
        registered_graph, uid, "th-coffee", "行政事务",
        "茶水间咖啡机已经修好可以正常使用",
    )

    out = await episodic._search_past_conversations(
        TENANT, uid, "Aurora 连接池问题怎么排查", k=5
    )
    assert "th-aurora" in out
    assert "Aurora 项目周会" in out
    # 无关会话不得混入
    assert "咖啡机" not in out


async def test_excludes_current_thread(registered_graph: object) -> None:
    uid = await _make_user("epi-3@example.com")
    await _seed_thread(
        registered_graph, uid, "th-current", "当前会话主题",
        "正在讨论 Aurora 发布窗口安排",
    )
    await _seed_thread(
        registered_graph, uid, "th-history", "历史会话主题",
        "上次讨论过 Aurora 发布窗口的备选方案",
    )

    out = await episodic._search_past_conversations(
        TENANT, uid, "Aurora 发布窗口", k=5, current_thread_id="th-current"
    )
    assert "th-history" in out
    assert "当前会话主题" not in out


async def test_tenant_isolation_between_users(registered_graph: object) -> None:
    uid_a = await _make_user("epi-4a@example.com")
    uid_b = await _make_user("epi-4b@example.com")
    await _seed_thread(
        registered_graph, uid_a, "th-a", "A 的 Aurora 笔记",
        "Aurora 灰度发布步骤记录",
    )
    await _seed_thread(
        registered_graph, uid_b, "th-b", "B 的 Aurora 笔记",
        "Aurora 回滚预案演练",
    )

    out_a = await episodic._search_past_conversations(
        TENANT, uid_a, "Aurora", k=5
    )
    assert "th-a" in out_a
    assert "th-b" not in out_a


async def test_disabled_switch_and_empty_query(
    registered_graph: object, isolated_db
) -> None:
    uid = await _make_user("epi-5@example.com")
    await _seed_thread(
        registered_graph, uid, "th-x", "标题", "Aurora 相关讨论"
    )

    isolated_db.memory_episodic_enabled = False
    assert "未开启" in await episodic._search_past_conversations(
        TENANT, uid, "Aurora"
    )

    isolated_db.memory_episodic_enabled = True
    assert "为空" in await episodic._search_past_conversations(
        TENANT, uid, "   "
    )


async def test_unregistered_graph_returns_readable_message() -> None:
    episodic.register_agent_graph(None)
    out = await episodic._search_past_conversations(
        TENANT, "any-user", "Aurora"
    )
    assert "尚未就绪" in out


# ── 3. 运行时注入：ToolNode + ToolRuntime 端到端 ───────────────────


class _RuntimeContext(BaseModel):
    user_id: str
    tenant_id: str = TENANT


def _tool_graph():
    node = ToolNode([search_past_conversations])
    g = StateGraph(MessagesState, context_schema=_RuntimeContext)
    g.add_node("tools", node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


async def test_tool_resolves_identity_and_thread_from_runtime(
    registered_graph: object,
) -> None:
    uid = await _make_user("epi-6@example.com")
    await _seed_thread(
        registered_graph, uid, "th-live", "进行中的 Aurora 讨论",
        "当前会话里分析 Aurora 慢查询",
    )
    await _seed_thread(
        registered_graph, uid, "th-old", "过去的 Aurora 讨论",
        "历史会话中记录的 Aurora 索引优化方案",
    )

    app = _tool_graph()
    result = await app.ainvoke(
        {"messages": [AIMessage(content="", tool_calls=[{
            "name": "search_past_conversations",
            "args": {"query": "Aurora 优化方案", "k": 3},
            "id": "call-1",
        }])]},
        config={"configurable": {"thread_id": "th-live"}},
        context={"user_id": uid, "tenant_id": TENANT},
    )
    content = result["messages"][-1].content
    assert "th-old" in content
    # 当前线程经 runtime.config.thread_id 自动排除
    assert "th-live" not in content
