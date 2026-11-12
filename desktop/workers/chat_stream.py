"""SSE 流式对话 worker：在 QThread 中逐事件读取 /chat/stream 并转发 Signal。"""

from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QThread, Signal

from desktop.api.client import RagApiClient
from desktop.api.errors import ApiError


class ChatStreamWorker(QThread):
    """流式对话：thread_created/token/sources/interrupt/done/error 信号。

    thread_id 为 None 时先在 worker 线程内建会话再发流（信号带回新会话 id）。
    cancel() 置位取消标志，流在下个事件到达时中断。
    """

    thread_created = Signal(str)
    token = Signal(str)
    sources = Signal(list)
    interrupt = Signal(list)
    done = Signal()
    error = Signal(str, int)

    def __init__(
        self,
        client: RagApiClient,
        message: str,
        thread_id: str | None,
        parent: object = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._message = message
        self._thread_id = thread_id
        self._cancel = threading.Event()

    def run(self) -> None:
        try:
            thread_id = self._thread_id
            if thread_id is None:
                thread_id = self._client.create_thread().id
                self.thread_created.emit(thread_id)
            self._client.chat_stream(
                self._message, thread_id, self._dispatch, self._cancel
            )
            self.done.emit()
        except ApiError as exc:
            self.error.emit(exc.detail, exc.status_code)
        except Exception as exc:  # 兜底：不让线程静默死亡
            self.error.emit(f"对话失败：{exc}", -1)

    def _dispatch(self, event: str, data: dict[str, Any]) -> None:
        if event == "token":
            text = data.get("text", "")
            if text:
                self.token.emit(str(text))
        elif event == "sources":
            self.sources.emit(list(data.get("sources", [])))
        elif event == "interrupt":
            self.interrupt.emit(list(data.get("interrupts", [])))
        elif event == "error":
            self.error.emit(str(data.get("detail", "执行失败")), 0)

    def cancel(self) -> None:
        self._cancel.set()
