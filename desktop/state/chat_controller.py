"""对话状态机：驱动 ChatStreamWorker 与 HITL 审批恢复（计划 P2 状态机）。

状态迁移：
    idle --send--> streaming
    streaming --interrupt--> awaiting_approval
    streaming --done/error--> idle
    awaiting_approval --resume--> resuming
    resuming --pending_approval--> awaiting_approval（多工具连续挂起循环）
    resuming --ok--> idle
    任意态 --cancel--> idle
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal

from desktop.api.client import RagApiClient
from desktop.api.models import ChatResult, InterruptPayload
from desktop.workers.base import start_worker
from desktop.workers.chat_stream import ChatStreamWorker

STATE_IDLE = "idle"
STATE_STREAMING = "streaming"
STATE_AWAITING = "awaiting_approval"
STATE_RESUMING = "resuming"


class ChatController(QObject):
    """一轮对话的生命周期编排；UI 层只订阅信号，不直接碰 worker。"""

    state_changed = Signal(str)
    thread_created = Signal(str)
    token = Signal(str)
    sources = Signal(list)
    interrupts = Signal(list)  # list[InterruptPayload]
    answer = Signal(str)       # resume 后的最终文本
    finished = Signal()        # 一轮完整结束（正常 done 或 resume 后 ok）
    failed = Signal(str)

    def __init__(self, client: RagApiClient, parent: object = None) -> None:
        super().__init__(parent)
        self._client = client
        self._state = STATE_IDLE
        self._worker: ChatStreamWorker | None = None
        self._thread_id: str | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def thread_id(self) -> str | None:
        return self._thread_id

    def set_thread(self, thread_id: str | None) -> None:
        """切换会话（由侧栏选择驱动）；不影响进行中的轮次状态。"""

        self._thread_id = thread_id

    # ── 发送 ──────────────────────────────────────────────────────────

    def send(self, text: str) -> bool:
        """idle 态发起一轮流式对话；非 idle 返回 False。"""

        if self._state != STATE_IDLE:
            return False
        self._set_state(STATE_STREAMING)
        worker = ChatStreamWorker(self._client, text, self._thread_id)
        worker.thread_created.connect(self._on_thread_created)
        worker.token.connect(self.token)
        worker.sources.connect(self.sources)
        worker.interrupt.connect(self._on_interrupt)
        worker.done.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()
        return True

    def _on_thread_created(self, thread_id: str) -> None:
        self._thread_id = thread_id
        self.thread_created.emit(thread_id)

    def _on_interrupt(self, raw: list[dict[str, Any]]) -> None:
        payloads = [InterruptPayload(**it) for it in raw]
        self._set_state(STATE_AWAITING)
        self.interrupts.emit(payloads)

    def _on_done(self) -> None:
        # interrupt 先于 done 到达并已切换状态；仍为 streaming 说明正常结束
        if self._state == STATE_STREAMING:
            self._set_state(STATE_IDLE)
            self.finished.emit()

    def _on_error(self, detail: str, _status: int) -> None:
        self._set_state(STATE_IDLE)
        self.failed.emit(detail)

    # ── 审批恢复 ──────────────────────────────────────────────────────

    def resume(self, decisions: list[dict[str, Any]]) -> bool:
        """awaiting_approval 态提交审批决策；decisions 与 interrupts 按下标对齐。"""

        if self._state != STATE_AWAITING or self._thread_id is None:
            return False
        self._set_state(STATE_RESUMING)
        start_worker(
            lambda: self._client.resume_chat(self._thread_id or "", decisions),
            self._on_resume_result,
            self._on_error,
        )
        return True

    def _on_resume_result(self, result: ChatResult) -> None:
        if result.status == "pending_approval" and result.interrupts:
            self._set_state(STATE_AWAITING)
            self.interrupts.emit(result.interrupts)
            return
        self._set_state(STATE_IDLE)
        if result.answer:
            self.answer.emit(result.answer)
        self.finished.emit()

    # ── 取消 ──────────────────────────────────────────────────────────

    def cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        self._set_state(STATE_IDLE)

    def shutdown(self) -> None:
        """窗口关闭前调用：取消并等待流式 worker 结束，避免 QThread 运行中被销毁。"""

        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(3000)

    def _set_state(self, state: str) -> None:
        self._state = state
        self.state_changed.emit(state)
