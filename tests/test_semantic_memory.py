"""语义事实记忆测试（P2）。

覆盖：
1. 数据层：facts CRUD、跨用户 namespace 隔离；
2. 工具层：save 语义去重（更新/新建）、凭据/超长/配额护栏、
   recall 语义命中与无索引关键词兜底、forget；
3. 运行时注入：ToolNode + runtime.context.user_id 的端到端隔离；
4. TTL：条目过期 + sweep_ttl 物理清除；
5. HTTP：/memory/facts 列表、/memory/fact 删除、清空联动。

测试嵌入固定 hash（conftest），无外部模型下载。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from src.api import auth
from src.api.memory import router as memory_router
from src.memory.tools import (
    _forget_memory,
    _parse_as_of,
    _recall_memories,
    _recall_memories_as_of,
    _save_memory,
    forget_memory,
    recall_memories,
    save_memory,
)
from src.persistence import database as db

pytestmark = pytest.mark.usefixtures("isolated_db")

U1 = "11111111111111111111111111111111"
U2 = "22222222222222222222222222222222"
TENANT = db.DEFAULT_TENANT_ID


# ── 1. 数据层 ──────────────────────────────────────────────────────


async def test_facts_crud_and_namespace_isolation() -> None:
    r1 = await _save_memory(TENANT, U1, "我是数据团队负责人，偏好列表式回答", "identity")
    assert "已保存" in r1
    r2 = await _save_memory(TENANT, U2, "我在做支付风控项目", "project")
    assert "已保存" in r2

    facts_u1 = await db.list_user_facts(TENANT, U1)
    facts_u2 = await db.list_user_facts(TENANT, U2)
    assert len(facts_u1) == 1
    assert len(facts_u2) == 1
    assert facts_u1[0]["category"] == "identity"
    assert facts_u1[0]["size_bytes"] > 0

    key1 = facts_u1[0]["key"]
    fetched = await db.get_user_fact(TENANT, U1, key1)
    assert fetched is not None and fetched["text"].startswith("我是数据团队")
    # 跨用户取不到
    assert await db.get_user_fact(TENANT, U2, key1) is None

    assert await db.delete_user_fact(TENANT, U1, key1) is True
    assert await db.delete_user_fact(TENANT, U1, key1) is False  # 幂等
    assert len(await db.list_user_facts(TENANT, U1)) == 0
    # U2 不受影响
    assert len(await db.list_user_facts(TENANT, U2)) == 1
    assert await db.purge_user_facts(TENANT, U2) == 1


# ── 2. 工具层：去重 / 护栏 / 召回 / 遗忘 ──────────────────────────


async def test_save_dedup_supersedes_old_version_with_chain() -> None:
    text = "团队周报固定在每周五下午三点提交"
    r1 = await _save_memory(TENANT, U1, text, "project")
    assert "created" in r1
    old_key = r1.split("memory_id=")[1].split("。")[0]

    r2 = await _save_memory(TENANT, U1, text, "decision")  # 完全相同 → 命中取代
    assert "superseded" in r2
    new_key = r2.split("memory_id=")[1].split("。")[0]
    assert new_key != old_key

    # 管理列表保留两条：新 active + 旧 superseded 审计副本
    facts = await db.list_user_facts(TENANT, U1)
    assert len(facts) == 2
    by_key = {f["key"]: f for f in facts}
    assert by_key[new_key]["status"] == "active"
    assert by_key[new_key]["category"] == "decision"
    assert by_key[new_key]["supersedes"] == old_key
    assert by_key[old_key]["status"] == "superseded"
    assert by_key[old_key]["superseded_by"] == new_key

    # active 口径只有 1 条（配额/召回口径）
    active = await db.list_active_user_facts(TENANT, U1)
    assert len(active) == 1
    assert active[0]["key"] == new_key


async def test_semantically_different_facts_create_new_items() -> None:
    await _save_memory(TENANT, U1, "团队周报固定在每周五下午三点提交", "project")
    await _save_memory(TENANT, U1, "我喜欢喝美式咖啡，不加糖", "preference")
    facts = await db.list_user_facts(TENANT, U1)
    assert len(facts) == 2


async def test_recall_semantic_hit_and_cross_user_miss() -> None:
    await _save_memory(TENANT, U1, "我偏好简洁的列表式回答，不要长篇大论", "preference")
    await _save_memory(TENANT, U2, "我偏好详尽的分步骤解释，带示例代码", "preference")

    hit = await _recall_memories(TENANT, U1, "回答风格希望简短一点", k=3)
    # hash 模式下 dense 与 BM25 两路同时命中 → 混合管线标注 hybrid
    assert "mode=hybrid" in hit
    assert "列表式回答" in hit
    assert "示例代码" not in hit  # U2 的偏好不能串到 U1

    empty = await _recall_memories(TENANT, U2, "zzzz-no-overlap-token-量子隧穿", k=3)
    # hash 兜底下完全无重叠即未命中
    assert "未检索到" in empty


async def test_save_rejects_empty_secret_and_oversize() -> None:
    assert "为空" in await _save_memory(TENANT, U1, "   ")
    secret = await _save_memory(TENANT, U1, "aws_access_key_id = AKIAIOSFODNN7EXAMPLE")
    assert "凭据" in secret
    assert await db.list_user_facts(TENANT, U1) == []  # 拒绝内容不落盘

    big = await _save_memory(TENANT, U1, "字" * 1001)
    assert "最多" in big
    assert await db.list_user_facts(TENANT, U1) == []


async def test_save_enforces_item_quota(isolated_db) -> None:
    isolated_db.memory_fact_max_items = 3
    # 用词/用字尽量不相交，避免被语义去重判定为同一条
    distinct_facts = [
        "咖啡机摆放在茶水间第二层柜子里",
        "季度财报的截止日期是每月最后一天",
        "生产环境发布窗口固定在每周二上午",
    ]
    for text in distinct_facts:
        assert "已保存" in await _save_memory(TENANT, U1, text)

    # 第四条全新事实应被配额拒绝
    rejected = await _save_memory(TENANT, U1, "新员工入职第一天需要领取门禁卡")
    assert "上限" in rejected
    assert len(await db.list_user_facts(TENANT, U1)) == 3


async def test_supersede_preserves_absolute_ttl() -> None:
    import time as _time

    text = "季度 OKR 评审固定在每月最后一个周五"
    r1 = await _save_memory(TENANT, U1, text, "project", ttl_days=2)
    assert "created" in r1
    old_key = r1.split("memory_id=")[1].split("。")[0]

    r2 = await _save_memory(TENANT, U1, text, "project")  # 命中取代
    assert "superseded" in r2
    new_key = r2.split("memory_id=")[1].split("。")[0]
    ns = db.user_facts_namespace(TENANT, U1)
    store = db.get_store()

    # 新 active 继承原绝对过期时刻（约 48 小时），临时事实不得被刷成永久
    new_item = await store.aget(ns, new_key)
    new_expires = (new_item.value or {}).get("expires_at")
    assert new_expires is not None
    assert 47 < (new_expires - _time.time()) / 3600 <= 48

    # 旧副本标记 superseded；原 TTL（2 天）短于 14 天审计窗口，
    # 收尾 TTL 取较小者，即沿用原过期时刻
    old_item = await store.aget(ns, old_key)
    old_value = old_item.value or {}
    assert old_value["status"] == "superseded"
    assert old_value["superseded_by"] == new_key
    old_remaining_hours = (old_value["expires_at"] - _time.time()) / 3600
    assert 47 < old_remaining_hours <= 48


async def test_save_enforces_tenant_level_quota(isolated_db) -> None:
    # 全局配额充足（默认 200），租户级 max_facts 收紧到 2
    assert await db.update_tenant(TENANT, max_facts=2) is True
    assert "已保存" in await _save_memory(TENANT, U1, "咖啡机摆放在茶水间第二层柜子里")
    assert "已保存" in await _save_memory(TENANT, U1, "季度财报的截止日期是每月最后一天")
    rejected = await _save_memory(TENANT, U1, "生产环境发布窗口固定在每周二上午")
    assert "上限 2 条" in rejected
    assert len(await db.list_user_facts(TENANT, U1)) == 2


async def test_hash_mode_dedup_does_not_merge_similar_wording() -> None:
    # hash 嵌入下两条措辞相近但语义不同的事实不得被去重误合并
    await _save_memory(TENANT, U1, "我喜欢在周五下午打篮球放松", "preference")
    await _save_memory(TENANT, U1, "我喜欢在周五下午开项目评审会", "project")
    facts = await db.list_user_facts(TENANT, U1)
    assert len(facts) == 2
    texts = {f["text"] for f in facts}
    assert "我喜欢在周五下午打篮球放松" in texts
    assert "我喜欢在周五下午开项目评审会" in texts


async def test_forget_memory_accepts_id_with_and_without_prefix() -> None:
    r = await _save_memory(TENANT, U1, "生产环境发布窗口是每周二上午十点")
    key = r.split("memory_id=")[1].split("。")[0]
    assert key.startswith("fact-")

    # 带不带前缀都能删（模型可能只抄 uuid 部分）
    assert "已删除" in await _forget_memory(TENANT, U1, key)
    assert "未找到" in await _forget_memory(TENANT, U1, key)


async def test_ttl_expired_entry_swept() -> None:
    store = db.get_store()
    ns = db.user_facts_namespace(TENANT, U1)
    key = "fact-" + db.new_id()
    # 直接写一条 0.05 分钟（3 秒）后过期的条目
    # （Store 的 expires_at 为秒级精度，TTL 留足跨秒余量）
    await store.aput(
        ns, key, {"text": "临时事实，马上过期", "category": "general"},
        index=["text"], ttl=0.05,
    )
    assert await db.get_user_fact(TENANT, U1, key) is not None
    await asyncio.sleep(4.0)

    # 该版本 sqlite Store 的过期以 sweep 物理删除为准（GET 不过滤、
    # 不做读取续期），清扫后条目彻底不可见
    swept = await store.sweep_ttl()
    assert swept >= 1
    assert await db.get_user_fact(TENANT, U1, key) is None
    expired = await store.asearch(ns, query="临时事实", limit=5)
    assert all(item.key != key for item in expired)


# ── 2b. 取代链软失效契约（阶段 1） ─────────────────────────────────


async def test_recall_excludes_superseded_and_annotates_replacement() -> None:
    text = "生产环境发布窗口固定在每周二上午十点"
    r1 = await _save_memory(TENANT, U1, text, "project")
    old_key = r1.split("memory_id=")[1].split("。")[0]
    r2 = await _save_memory(TENANT, U1, text, "project")
    new_key = r2.split("memory_id=")[1].split("。")[0]

    out = await _recall_memories(TENANT, U1, "生产环境什么时候发布", k=5)
    # 旧版本不进召回：old_key 只允许出现在"已取代旧版本 id=..."标注里，
    # 不允许作为独立命中条目（命中渲染为 [id=xxx, category=...]）
    assert out.count("发布窗口") == 1
    assert new_key in out
    assert f"[id={old_key}," not in out
    assert f"已取代旧版本 id={old_key}" in out


async def test_quota_counts_only_active_facts(isolated_db) -> None:
    isolated_db.memory_fact_max_items = 2
    distinct = [
        "咖啡机摆放在茶水间第二层柜子里",
        "季度财报的截止日期是每月最后一天",
    ]
    for text in distinct:
        assert "已保存" in await _save_memory(TENANT, U1, text)

    # 取代不额外占用 active 配额（active 仍为 2，总数变 3）
    assert "superseded" in await _save_memory(TENANT, U1, distinct[0])
    assert len(await db.list_active_user_facts(TENANT, U1)) == 2
    assert len(await db.list_user_facts(TENANT, U1)) == 3

    # active 满员，新事实仍被拒绝
    rejected = await _save_memory(TENANT, U1, "新员工入职第一天需要领取门禁卡")
    assert "上限" in rejected

    # 删掉一个 active 后配额释放，可再写入（superseded 副本不占名额）
    active_keys = [f["key"] for f in await db.list_active_user_facts(TENANT, U1)]
    assert "已删除" in await _forget_memory(TENANT, U1, active_keys[0])
    assert len(await db.list_active_user_facts(TENANT, U1)) == 1
    assert "已保存" in await _save_memory(TENANT, U1, "新员工入职第一天需要领取门禁卡")


async def test_legacy_fact_without_status_defaults_active() -> None:
    # 模拟阶段 1 上线前的旧格式条目（无 status/source 等字段）
    store = db.get_store()
    ns = db.user_facts_namespace(TENANT, U1)
    legacy_key = "fact-" + db.new_id()
    await store.aput(
        ns, legacy_key, {"text": "旧系统写入的事实：工位在三层 305", "category": "general"},
        index=["text"],
    )

    view = await db.get_user_fact(TENANT, U1, legacy_key)
    assert view is not None and view["status"] == "active"
    assert view["source"] == "manual"  # 缺省值
    active = await db.list_active_user_facts(TENANT, U1)
    assert any(f["key"] == legacy_key for f in active)

    # 旧条目照常参与召回
    out = await _recall_memories(TENANT, U1, "工位在哪里", k=3)
    assert "305" in out


async def test_superseded_copy_uses_configured_audit_window(isolated_db) -> None:
    import time as _time

    # 永久事实被取代后，旧副本收尾 TTL 取配置的审计窗口（14 天）
    isolated_db.memory_superseded_ttl_days = 14
    text = "我在团队里负责检索质量评估"
    r1 = await _save_memory(TENANT, U1, text, "identity")
    old_key = r1.split("memory_id=")[1].split("。")[0]
    r2 = await _save_memory(TENANT, U1, text, "identity")
    new_key = r2.split("memory_id=")[1].split("。")[0]

    ns = db.user_facts_namespace(TENANT, U1)
    old_item = await db.get_store().aget(ns, old_key)
    old_value = old_item.value or {}
    assert old_value["status"] == "superseded"
    remaining_days = (old_value["expires_at"] - _time.time()) / 86400
    assert 13.9 < remaining_days <= 14

    # 新 active 继承永久性：没有 expires_at
    new_item = await db.get_store().aget(ns, new_key)
    assert (new_item.value or {}).get("expires_at") is None


async def test_save_marks_reflection_source() -> None:
    out = await _save_memory(
        TENANT, U1, "用户偏好用表格展示对比结论", "preference", source="reflection"
    )
    key = out.split("memory_id=")[1].split("。")[0]
    view = await db.get_user_fact(TENANT, U1, key)
    assert view is not None
    assert view["source"] == "reflection"
    assert view["status"] == "active"


# ── 2c. 时序字段（valid_from/invalid_at）与 as-of 时点召回 ─────────


async def test_temporal_fields_set_on_create_and_supersede() -> None:
    text = "团队周报固定在每周五下午三点提交"
    r1 = await _save_memory(TENANT, U1, text, "project")
    old_key = r1.split("memory_id=")[1].split("。")[0]
    view_old = await db.get_user_fact(TENANT, U1, old_key)
    assert view_old is not None
    # 新建事实：生效时刻存在、失效时刻为空
    assert view_old["valid_from"] is not None
    assert view_old["invalid_at"] is None

    await asyncio.sleep(1.1)
    r2 = await _save_memory(TENANT, U1, text, "decision")
    new_key = r2.split("memory_id=")[1].split("。")[0]

    view_superseded = await db.get_user_fact(TENANT, U1, old_key)
    view_new = await db.get_user_fact(TENANT, U1, new_key)
    assert view_superseded is not None and view_new is not None
    # 旧版本打上失效时刻；新版本生效时刻不早于旧版本失效时刻（无缝交接）
    assert view_superseded["invalid_at"] is not None
    assert view_new["invalid_at"] is None
    assert view_new["valid_from"] is not None
    assert view_new["valid_from"] >= view_superseded["invalid_at"]


async def test_recall_as_of_reconstructs_historical_version() -> None:
    import datetime as _dt

    text = "团队周报固定在每周五下午三点提交"
    r1 = await _save_memory(TENANT, U1, text, "project")
    old_key = r1.split("memory_id=")[1].split("。")[0]

    await asyncio.sleep(1.1)
    r2 = await _save_memory(TENANT, U1, text, "project")
    new_key = r2.split("memory_id=")[1].split("。")[0]

    ns = db.user_facts_namespace(TENANT, U1)
    store = db.get_store()
    old_value = (await store.aget(ns, old_key)).value
    mid_ts = (old_value["valid_from"] + old_value["invalid_at"]) / 2
    mid_iso = _dt.datetime.fromtimestamp(mid_ts, tz=_dt.UTC).isoformat()

    # 时点位于两版本之间：只命中旧版本，新版本尚未生效
    mid_out = await _recall_memories_as_of(TENANT, U1, "周报什么时候", mid_iso, k=5)
    assert old_key in mid_out
    assert f"[id={new_key}," not in mid_out
    assert "as_of=" in mid_out

    # 未来时点：只命中当前新版本，旧版本已失效
    future_out = await _recall_memories_as_of(
        TENANT, U1, "周报什么时候", "2030-01-01", k=5
    )
    assert new_key in future_out
    assert f"[id={old_key}," not in future_out

    # 早于任何事实的时点：无命中
    before_iso = _dt.datetime.fromtimestamp(
        old_value["valid_from"] - 100, tz=_dt.UTC
    ).isoformat()
    before_out = await _recall_memories_as_of(
        TENANT, U1, "周报什么时候", before_iso, k=5
    )
    assert "未检索到" in before_out


async def test_recall_as_of_rejects_invalid_date_and_empty_query() -> None:
    bad = await _recall_memories_as_of(TENANT, U1, "周报", "not-a-date")
    assert "时间格式无法识别" in bad
    empty = await _recall_memories_as_of(TENANT, U1, "  ", "2030-01-01")
    assert "为空" in empty
    # 解析器单测：日期粒度与完整 ISO 均可识别，非法输入返回 None
    assert isinstance(_parse_as_of("2026-09-01"), float)
    assert isinstance(_parse_as_of("2026-09-01T12:00:00+00:00"), float)
    assert _parse_as_of("bad") is None


async def test_legacy_fact_as_of_falls_back_to_created_at() -> None:
    import datetime as _dt
    import time as _time

    # 旧格式条目：无 valid_from/status，只有 created_at
    store = db.get_store()
    ns = db.user_facts_namespace(TENANT, U1)
    legacy_key = "fact-" + db.new_id()
    created = _time.time()
    await store.aput(
        ns, legacy_key,
        {"text": "旧系统事实：工位在三层 305", "category": "general",
         "created_at": created},
        index=["text"],
    )

    future = await _recall_memories_as_of(TENANT, U1, "工位在哪里", "2030-01-01", k=3)
    assert legacy_key in future

    before_iso = _dt.datetime.fromtimestamp(created - 100, tz=_dt.UTC).isoformat()
    past = await _recall_memories_as_of(TENANT, U1, "工位在哪里", before_iso, k=3)
    assert "未检索到" in past


# ── 3. 无向量索引时的关键词兜底 ────────────────────────────────────


async def test_keyword_fallback_when_vector_disabled(
    isolated_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 关闭向量索引后重建 Store
    isolated_db.memory_embedding = "none"
    assert isolated_db.memory_vector_enabled is False
    await db.close_db()
    await db.init_db()

    await _save_memory(TENANT, U1, "项目代号 Aurora，每周三同步进度")
    await _save_memory(TENANT, U1, "另一条无关内容：咖啡机在茶水间")

    out = await _recall_memories(TENANT, U1, "Aurora 项目什么时候同步？", k=2)
    assert "mode=keyword" in out
    assert "Aurora" in out
    assert "咖啡机" not in out


# ── 4. 运行时注入：真实 @tool + ToolNode 端到端隔离 ────────────────


class _RuntimeContext(BaseModel):
    user_id: str
    tenant_id: str = db.DEFAULT_TENANT_ID


def _tool_graph():
    node = ToolNode([save_memory, recall_memories, forget_memory])
    g = StateGraph(MessagesState, context_schema=_RuntimeContext)
    g.add_node("tools", node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


async def test_tools_resolve_user_id_from_runtime_context() -> None:
    app = _tool_graph()

    async def _call(user_id: str, fn: str, args: dict, call_id: str) -> str:
        out = await app.ainvoke(
            {"messages": [AIMessage(content="", tool_calls=[{
                "name": fn, "args": args, "id": call_id,
            }])]},
            context={"user_id": user_id, "tenant_id": TENANT},
        )
        return out["messages"][-1].content

    saved = await _call(U1, "save_memory", {
        "text": "我的主要技术栈是 Python 和 LangGraph",
        "category": "project",
    }, "c1")
    assert "已保存" in saved

    mine = await _call(U1, "recall_memories", {"query": "技术栈是什么", "k": 3}, "c2")
    assert "LangGraph" in mine

    other = await _call(U2, "recall_memories", {"query": "技术栈是什么", "k": 3}, "c3")
    assert "未检索到" in other


# ── 5. HTTP 接口 ───────────────────────────────────────────────────


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(memory_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client: AsyncClient, email: str) -> tuple[str, str]:
    resp = await client.post(
        "/auth/register", json={"email": email, "password": "Passw0rd!"}
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return body["user"]["id"], body["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_facts_api_list_delete_and_isolation(client: AsyncClient) -> None:
    uid_a, token_a = await _register(client, "facts-a@example.com")
    _, token_b = await _register(client, "facts-b@example.com")

    await _save_memory(TENANT, uid_a, "我负责 RAG 系统的检索质量评估", "project")
    await _save_memory(TENANT, uid_a, "回答必须标注来源", "preference")

    resp = await client.get("/memory/facts", headers=_auth(token_a))
    assert resp.status_code == 200
    body = resp.json()
    assert body["item_count"] == 2
    assert body["max_items"] == 200
    assert all("text" in item for item in body["items"])

    # B 看不到 A 的事实
    bob = await client.get("/memory/facts", headers=_auth(token_b))
    assert bob.json()["item_count"] == 0

    key = body["items"][0]["key"]
    deleted = await client.request(
        "DELETE", "/memory/fact", params={"key": key}, headers=_auth(token_a)
    )
    assert deleted.status_code == 204
    again = await client.request(
        "DELETE", "/memory/fact", params={"key": key}, headers=_auth(token_a)
    )
    assert again.status_code == 404

    # 非法 id 格式
    bad = await client.request(
        "DELETE", "/memory/fact", params={"key": "../etc/passwd"},
        headers=_auth(token_a),
    )
    assert bad.status_code == 400

    # B 不能删 A 的事实
    cross = await client.request(
        "DELETE", "/memory/fact",
        params={"key": (await client.get(
            "/memory/facts", headers=_auth(token_a)
        )).json()["items"][0]["key"]},
        headers=_auth(token_b),
    )
    assert cross.status_code == 404

    assert (await client.get("/memory/facts")).status_code == 401


async def test_clear_memory_covers_facts(client: AsyncClient) -> None:
    from src.agents.graph import GuardedStoreBackend

    uid, token = await _register(client, "facts-clear@example.com")
    await _save_memory(TENANT, uid, "一条待清空的事实")
    backend = GuardedStoreBackend(
        store=db.get_store(),
        namespace=lambda rt: ("tenant", TENANT, "user_memory", uid),
        max_file_bytes=8192,
        max_items=20,
    )
    await backend.awrite("/AGENTS.md", "画像")

    resp = await client.request("DELETE", "/memory", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1, "facts_deleted": 1}
    assert await db.list_user_facts(TENANT, uid) == []
