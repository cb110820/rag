"""deepagents 非破坏式上下文压缩中间件契约测试。

锁定三条生产契约：

1. ``src.agents.graph`` 装配的必须是 deepagents 版 ``SummarizationMiddleware``，
   且其 ``.name`` 与框架自动挂载的默认中间件同名——依赖
   ``create_deep_agent`` 的按名替换语义，保证栈内只有一个压缩中间件；
2. 项目自定义参数（trigger/keep/summary_prompt）与 backend 必须原样透传；
3. 压缩是"请求侧覆盖"而非"删 state"：原始消息保留、原文 offload 到
   ``/conversation_history/``、模型实际收到缩短后的摘要视图，且随
   checkpoint 跨图实例可恢复。

全程使用 fake 模型与 InMemorySaver，不发起任何网络请求。
"""

from __future__ import annotations

import itertools

import pytest
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.summarization import (
    SummarizationMiddleware as DASummarizationMiddleware,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import graph as graph_module
from src.agents.prompts import SUMMARY_PROMPT

# 记录每次模型调用实际收到的消息（pydantic 模型不允许动态属性，故用模块级收集器）
_MODEL_CALLS: list[list] = []


class _RecordingFakeChatModel(GenericFakeChatModel):
    """记录模型入参的 fake 模型，用于断言压缩后的"模型视图"。"""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _MODEL_CALLS.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        _MODEL_CALLS.append(list(messages))
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


def _make_fake_model() -> _RecordingFakeChatModel:
    def _infinite_answers():
        for i in itertools.count():
            yield AIMessage(content=f"fake-answer-{i}")

    return _RecordingFakeChatModel(messages=_infinite_answers())


def test_graph_assembles_deepagents_summarization_middleware() -> None:
    """graph 模块必须引用 deepagents 版中间件（防止 import 回退到 LangChain 有损版）。"""

    assert graph_module.SummarizationMiddleware is DASummarizationMiddleware


def test_middleware_name_and_project_params_contract() -> None:
    """中间件名称与框架默认版同名（按名替换不双重压缩），项目参数全部透传。"""

    backend = CompositeBackend(default=StateBackend(), routes={})
    model = _make_fake_model()
    middleware = DASummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=[("tokens", 6000)],
        keep=("messages", 20),
        summary_prompt=SUMMARY_PROMPT,
    )

    # 与 deepagents 框架自动挂载的默认中间件同名 → _apply_custom_middleware 原位替换
    assert middleware.name == "SummarizationMiddleware"
    # 内部 LangChain helper 承接全部压缩参数，自定义摘要 prompt 不被默认值覆盖
    helper = middleware._lc_helper
    assert helper.summary_prompt is SUMMARY_PROMPT
    assert helper.trigger == [("tokens", 6000)]
    assert helper.keep == ("messages", 20)
    # 无损 offload 依赖的 backend（0.7 版为私有属性；落盘路径前缀由 e2e 用例验证）
    assert middleware._backend is backend


def _view_contains_summary(messages: list) -> bool:
    for message in messages:
        if message.additional_kwargs.get("lc_source") == "summarization":
            return True
        content = str(getattr(message, "content", ""))
        if "has been summarized" in content:
            return True
    return False


async def test_compaction_keeps_state_and_offloads_original_history() -> None:
    """触发压缩后：state 原文不删、conversation_history 落盘、模型视图缩短且可恢复。"""

    _MODEL_CALLS.clear()
    backend = CompositeBackend(default=StateBackend(), routes={})
    model = _make_fake_model()
    # 小阈值强制快速触发：第 3 轮模型调用时消息数已达 4 条
    middleware = DASummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=[("messages", 4)],
        keep=("messages", 2),
        summary_prompt=SUMMARY_PROMPT,
    )
    checkpointer = InMemorySaver()
    common_kwargs = dict(
        model=model,
        tools=[],
        system_prompt="contract-probe",
        middleware=[middleware, FilesystemMiddleware(backend=backend)],
        backend=backend,
        checkpointer=checkpointer,
    )
    agent = create_deep_agent(**common_kwargs, name="summarization_contract")

    cfg = {"configurable": {"thread_id": "contract-thread-1"}}
    user_texts = [
        "规则：金额大于 5000 元需要人工审核，请记住这个阈值",
        "第二个问题：Aurora X200 的规格是什么",
        "第三个问题：退款政策是什么",
    ]
    for text in user_texts:
        await agent.ainvoke(
            {"messages": [{"role": "user", "content": text}]}, cfg
        )

    state = (await agent.aget_state(cfg)).values
    raw_messages = state.get("messages", [])

    # 契约 1：非破坏式——原始用户消息一条不少地留在 state
    human_texts = [
        m.content for m in raw_messages if getattr(m, "type", "") == "human"
    ]
    for text in user_texts:
        assert text in human_texts

    # 契约 2：原文无损 offload 到 backend 的 /conversation_history/（files 通道）
    files = state.get("files", {}) or {}
    history_files = [
        path for path in files if path.startswith("/conversation_history/")
    ]
    assert history_files, "压缩后未生成 conversation_history offload 文件"
    offload_text = "".join(str(files[path].get("content", "")) for path in history_files)
    assert "5000" in offload_text and "人工审核" in offload_text

    # 契约 3：模型实际收到的是缩短后的摘要视图，而非全量原文
    compacted_views = [call for call in _MODEL_CALLS if _view_contains_summary(call)]
    assert compacted_views, "模型从未收到带摘要的压缩视图"
    assert min(len(view) for view in compacted_views) < len(raw_messages)

    # 契约 4：checkpoint 持久化——新图实例同 thread 可恢复消息与 offload 文件
    agent_restarted = create_deep_agent(**common_kwargs, name="summarization_contract_2")
    restarted_state = (await agent_restarted.aget_state(cfg)).values
    assert len(restarted_state.get("messages", [])) == len(raw_messages)
    restarted_files = restarted_state.get("files", {}) or {}
    assert any(
        path.startswith("/conversation_history/") for path in restarted_files
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
