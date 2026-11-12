"""ChatController 状态机迁移 + summarize_args 移植逻辑测试（离屏 Qt）。"""

import os
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from desktop.api.errors import ApiError
from desktop.api.models import ChatResult, InterruptPayload
from desktop.state import chat_controller as cc
from desktop.state.chat_controller import (
    STATE_AWAITING,
    STATE_IDLE,
    STATE_STREAMING,
    ChatController,
)
from desktop.ui.chat.approval_card import summarize_args


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


class FakeStreamWorker(QObject):
    """同步执行的假流式 worker：start() 内按脚本立即发信号。"""

    thread_created = Signal(str)
    token = Signal(str)
    sources = Signal(list)
    interrupt = Signal(list)
    done = Signal()
    error = Signal(str, int)
    finished = Signal()

    scripted: list[tuple[str, object]] = []

    def __init__(self, client, message, thread_id, parent=None) -> None:
        super().__init__(parent)

    def start(self) -> None:
        for kind, payload in self.scripted:
            if kind == "token":
                self.token.emit(payload)
            elif kind == "interrupt":
                self.interrupt.emit(payload)
            elif kind == "error":
                self.error.emit(str(payload), 0)
        self.done.emit()

    def isRunning(self) -> bool:
        return False

    def cancel(self) -> None:
        pass

    def deleteLater(self) -> None:
        pass


def _run_sync(fn, on_result, on_error=None):
    """同步版 start_worker：直接执行 fn 并分发回调。"""

    try:
        on_result(fn())
    except ApiError as exc:
        if on_error is not None:
            on_error(exc.detail, exc.status_code)


@pytest.fixture
def controller(qapp, monkeypatch):
    monkeypatch.setattr(cc, "ChatStreamWorker", FakeStreamWorker)
    monkeypatch.setattr(cc, "start_worker", _run_sync)
    client = MagicMock()
    return ChatController(client)


def test_send_stream_then_done_returns_idle(controller) -> None:
    FakeStreamWorker.scripted = [("token", "你"), ("token", "好")]
    tokens: list[str] = []
    finished: list[bool] = []
    controller.token.connect(tokens.append)
    controller.finished.connect(lambda: finished.append(True))

    assert controller.send("hi") is True
    assert tokens == ["你", "好"]
    assert controller.state == STATE_IDLE
    assert finished == [True]


def test_send_rejected_when_busy(controller) -> None:
    FakeStreamWorker.scripted = []
    controller._set_state(STATE_STREAMING)
    assert controller.send("hi") is False


def test_interrupt_enters_awaiting_and_done_does_not_finish(controller) -> None:
    FakeStreamWorker.scripted = [
        ("token", "准备删除"),
        ("interrupt", [{"id": "i1", "tool": "forget_memory", "args": {"fact_id": "f1"}}]),
    ]
    interrupts: list[list] = []
    finished: list[bool] = []
    controller.interrupts.connect(interrupts.append)
    controller.finished.connect(lambda: finished.append(True))

    controller.send("忘掉我的偏好")
    assert controller.state == STATE_AWAITING
    assert finished == []
    assert interrupts[0][0].tool == "forget_memory"
    # done 已在 interrupt 之后到达，但不得错误结束本轮
    assert controller.state == STATE_AWAITING


def test_resume_approve_returns_answer_and_idle(controller) -> None:
    controller._thread_id = "t1"
    controller._set_state(STATE_AWAITING)
    controller._client.resume_chat.return_value = ChatResult(status="ok", answer="已删除")

    answers: list[str] = []
    finished: list[bool] = []
    controller.answer.connect(answers.append)
    controller.finished.connect(lambda: finished.append(True))

    assert controller.resume([{"type": "approve"}]) is True
    assert controller._client.resume_chat.call_args[0][0] == "t1"
    assert controller.state == STATE_IDLE
    assert answers == ["已删除"]
    assert finished == [True]


def test_resume_with_new_interrupts_loops_back_to_awaiting(controller) -> None:
    controller._thread_id = "t1"
    controller._set_state(STATE_AWAITING)
    controller._client.resume_chat.return_value = ChatResult(
        status="pending_approval",
        answer="",
        interrupts=[
            InterruptPayload(id="i2", tool="write_file", args={"path": "/memory/AGENTS.md"})
        ],
    )
    interrupts: list[list] = []
    controller.interrupts.connect(interrupts.append)

    controller.resume([{"type": "approve"}])
    # 多工具连续挂起：回到 awaiting 并推送新一批审批项
    assert controller.state == STATE_AWAITING
    assert interrupts[0][0].tool == "write_file"


def test_resume_rejected_in_wrong_state(controller) -> None:
    controller._set_state(STATE_IDLE)
    assert controller.resume([{"type": "approve"}]) is False


def test_cancel_during_streaming_returns_idle(controller) -> None:
    controller._set_state(STATE_STREAMING)
    controller.cancel()
    assert controller.state == STATE_IDLE


# ── summarize_args（移植 web/app.js）───────────────────────────────────


def test_summarize_args_prefers_path() -> None:
    assert summarize_args({"path": "/memory/AGENTS.md", "fact_id": "x"}) == "/memory/AGENTS.md"


def test_summarize_args_fact_id() -> None:
    assert summarize_args({"fact_id": "fact-abc"}) == "fact_id: fact-abc"


def test_summarize_args_content_truncated_at_60() -> None:
    long_text = "记" * 100
    assert summarize_args({"content": long_text}) == "记" * 60 + "…"
    assert summarize_args({"content": "短文本"}) == "短文本"


def test_summarize_args_json_fallback_truncated_at_80() -> None:
    args = {"key": "v" * 200}
    result = summarize_args(args)
    assert result.endswith("…")
    assert len(result) == 81


def test_summarize_args_non_dict() -> None:
    assert summarize_args("not-a-dict") == ""
    assert summarize_args(None) == ""
