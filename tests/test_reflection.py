"""LangMem 后台反思提取测试。

覆盖：
1. 消息裁剪：role 映射、tool/system 过滤、块列表拼接、末尾 N 条截断；
2. 提取适配：langmem ExtractedMemory → 候选文本（用假 manager，不调 LLM）；
3. 反思主流程：候选事实经 save 护栏落库（含凭据拦截）、与既有记忆去重合并、
   提取失败就地吞掉、短对话跳过；
4. 调度：开关关闭不调度、同人并发去重、端到端 schedule→后台任务→落库。

全程不触真实 LLM：提取层一律 monkeypatch 掉。
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langmem.knowledge.extraction import ExtractedMemory, Memory

from src.memory import reflection
from src.memory.tools import _save_memory
from src.persistence import database as db

pytestmark = pytest.mark.usefixtures("isolated_db")

U1 = "11111111111111111111111111111111"
U2 = "22222222222222222222222222222222"
TENANT = db.DEFAULT_TENANT_ID


def _fake_extract(facts: list[str]):
    """返回一个替换 extract_candidate_facts 的假实现。"""

    async def _impl(conversation, model=None):  # noqa: ANN001
        return facts

    return _impl


# ── 1. 消息裁剪 ────────────────────────────────────────────────────


def test_coerce_messages_filters_and_maps_roles() -> None:
    tool_msg = ToolMessage(content="检索结果", tool_call_id="t1")
    messages = [
        SystemMessage(content="系统提示"),
        HumanMessage(content="我是数据团队负责人"),
        AIMessage(content="好的，已记住"),
        tool_msg,
        {"role": "user", "content": "偏好列表式回答"},
        {"role": "tool", "content": "字典 tool 应被过滤"},
        AIMessage(content=[{"type": "text", "text": "块列表文本"}]),
        HumanMessage(content="   "),  # 纯空白过滤
    ]
    conversation = reflection._coerce_messages(messages, max_messages=50)
    assert conversation == [
        {"role": "user", "content": "我是数据团队负责人"},
        {"role": "assistant", "content": "好的，已记住"},
        {"role": "user", "content": "偏好列表式回答"},
        {"role": "assistant", "content": "块列表文本"},
    ]


def test_coerce_messages_truncates_to_max() -> None:
    messages = [HumanMessage(content=f"第{i}条") for i in range(10)]
    conversation = reflection._coerce_messages(messages, max_messages=3)
    assert [m["content"] for m in conversation] == ["第7条", "第8条", "第9条"]


# ── 2. 提取适配（假 manager，不调 LLM） ─────────────────────────────


async def test_extract_candidate_facts_from_langmem_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeManager:
        async def ainvoke(self, input, config=None):  # noqa: ANN001, ANN202
            assert input["messages"] == [{"role": "user", "content": "我喜欢简洁回答"}]
            assert input["max_steps"] == 1
            return [
                ExtractedMemory(id="a", content=Memory(content="用户偏好简洁回答")),
                ExtractedMemory(id="b", content=Memory(content="  ")),  # 空白被丢弃
            ]

    import langmem

    monkeypatch.setattr(langmem, "create_memory_manager", lambda *a, **k: _FakeManager())
    facts = await reflection.extract_candidate_facts(
        [{"role": "user", "content": "我喜欢简洁回答"}], model=object()
    )
    assert facts == ["用户偏好简洁回答"]


# ── 3. 反思主流程 ───────────────────────────────────────────────────


async def test_reflect_saves_facts_through_guardrails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = [
        "用户是数据团队负责人",
        "password=supersecret123",  # 命中明文凭据赋值 → 护栏拒绝
        "用户偏好列表式回答",
    ]
    monkeypatch.setattr(reflection, "extract_candidate_facts", _fake_extract(facts))

    stats = await reflection.reflect_and_store(
        TENANT, U1, [HumanMessage(content="随便聊了不少内容")]
    )
    assert stats == {"extracted": 3, "saved": 2, "skipped": 1, "failed": 0}

    saved = await db.list_user_facts(TENANT, U1)
    texts = [f["text"] for f in saved]
    assert len(saved) == 2
    assert "用户是数据团队负责人" in texts
    assert "用户偏好列表式回答" in texts
    assert not any("supersecret" in t for t in texts)


async def test_reflect_dedups_against_existing_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _save_memory(TENANT, U1, "用户是数据团队负责人")
    # 反思提炼出同一事实的近似表述 → 写入层去重建立取代链：
    # 新 active（source=reflection）+ 旧版本 superseded 审计副本
    monkeypatch.setattr(
        reflection,
        "extract_candidate_facts",
        _fake_extract(["用户是数据团队的负责人"]),
    )
    stats = await reflection.reflect_and_store(
        TENANT, U1, [HumanMessage(content="聊一下我的角色和团队分工")]
    )
    assert stats["saved"] == 1  # superseded 也算 saved
    all_facts = await db.list_user_facts(TENANT, U1)
    assert len(all_facts) == 2
    active = await db.list_active_user_facts(TENANT, U1)
    assert len(active) == 1
    assert active[0]["text"] == "用户是数据团队的负责人"
    assert active[0]["source"] == "reflection"


async def test_reflect_extract_failure_is_contained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(conversation, model=None):  # noqa: ANN001
        raise RuntimeError("LLM 欠费")

    monkeypatch.setattr(reflection, "extract_candidate_facts", _boom)
    stats = await reflection.reflect_and_store(
        TENANT, U1, [HumanMessage(content="内容足够长可以触发提取")]
    )
    assert stats == {"extracted": 0, "saved": 0, "skipped": 0, "failed": 0}
    assert await db.list_user_facts(TENANT, U1) == []


async def test_reflect_skips_short_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def _spy(conversation, model=None):  # noqa: ANN001
        nonlocal called
        called = True
        return ["不该出现"]

    monkeypatch.setattr(reflection, "extract_candidate_facts", _spy)
    stats = await reflection.reflect_and_store(TENANT, U1, [HumanMessage(content="嗯")])
    assert called is False  # 低于 min_chars，不触 LLM
    assert stats["extracted"] == 0


async def test_reflect_namespace_isolated_per_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reflection, "extract_candidate_facts", _fake_extract(["用户在做风控项目"])
    )
    await reflection.reflect_and_store(TENANT, U2, [HumanMessage(content="介绍一下项目背景")] )
    assert len(await db.list_user_facts(TENANT, U2)) == 1
    assert await db.list_user_facts(TENANT, U1) == []


# ── 4. 调度 ─────────────────────────────────────────────────────────


def test_schedule_respects_disabled_switch() -> None:
    # conftest 默认 memory_reflection_enabled=False
    ok = reflection.schedule_reflection(TENANT, U1, [HumanMessage(content="hi")])
    assert ok is False


async def test_schedule_skips_when_task_running(
    monkeypatch: pytest.MonkeyPatch, isolated_db
) -> None:
    isolated_db.memory_reflection_enabled = True
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow(conversation, model=None):  # noqa: ANN001
        started.set()
        await release.wait()
        return ["用户在等待中提取的事实"]

    monkeypatch.setattr(reflection, "extract_candidate_facts", _slow)
    messages = [HumanMessage(content="这是一段足够长的对话内容")]
    assert reflection.schedule_reflection(TENANT, U1, messages) is True
    await started.wait()
    # 同一用户已有任务在跑 → 第二次调度被去重
    assert reflection.schedule_reflection(TENANT, U1, messages) is False
    # 不同用户不受影响
    assert reflection.schedule_reflection(TENANT, U2, messages) is True

    release.set()
    await asyncio.gather(*list(reflection._running.values()))
    assert reflection._running == {}
    assert len(await db.list_user_facts(TENANT, U1)) == 1
    assert len(await db.list_user_facts(TENANT, U2)) == 1


async def test_schedule_e2e_writes_facts_in_background(
    monkeypatch: pytest.MonkeyPatch, isolated_db
) -> None:
    isolated_db.memory_reflection_enabled = True
    monkeypatch.setattr(
        reflection,
        "extract_candidate_facts",
        _fake_extract(["用户明确要求每步打印执行日志"]),
    )
    ok = reflection.schedule_reflection(
        TENANT, U1, [HumanMessage(content="请记住我的日志偏好")]
    )
    assert ok is True
    await asyncio.gather(*list(reflection._running.values()))
    facts = await db.list_user_facts(TENANT, U1)
    assert [f["text"] for f in facts] == ["用户明确要求每步打印执行日志"]
