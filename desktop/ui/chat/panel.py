"""对话面板：会话侧栏 + 消息视图 + 输入行（P2 SSE 流式 + HITL 审批）。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from desktop.api.models import InterruptPayload, ThreadDetail, ThreadInfo
from desktop.state.chat_controller import (
    STATE_IDLE,
    STATE_RESUMING,
    STATE_STREAMING,
    ChatController,
)
from desktop.state.session import SessionManager
from desktop.ui.chat.approval_card import ApprovalCard
from desktop.ui.chat.message_view import MessageView
from desktop.ui.chat.sources_view import SourcesView
from desktop.ui.widgets.toast import show_toast
from desktop.workers.base import start_worker


class ChatPanel(QWidget):
    """对话主面板：流式渲染 token、来源 chips、HITL 审批卡片与恢复。"""

    def __init__(self, session: SessionManager, parent: object = None) -> None:
        super().__init__(parent)
        self._session = session
        self._threads: list[ThreadInfo] = []
        self._controller = ChatController(self._require_client(), self)
        self._stream_started = False
        self._seen_sources: set[str] = set()
        self._active_card: ApprovalCard | None = None

        # ── 会话侧栏 ──
        self._thread_list = QListWidget()
        self._thread_list.currentRowChanged.connect(self._on_thread_selected)
        new_btn = QPushButton("新建")
        new_btn.clicked.connect(self._on_new_thread)
        rename_btn = QPushButton("重命名")
        rename_btn.clicked.connect(self._on_rename_thread)
        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(self._on_delete_thread)

        side_buttons = QHBoxLayout()
        side_buttons.addWidget(new_btn)
        side_buttons.addWidget(rename_btn)
        side_buttons.addWidget(delete_btn)
        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self._thread_list, 1)
        side_layout.addLayout(side_buttons)

        # ── 聊天区 ──
        self._view = MessageView()
        self._input = QPlainTextEdit()
        self._input.setPlaceholderText("输入消息，Enter 发送，Shift+Enter 换行")
        self._input.setMaximumHeight(90)
        self._send_btn = QPushButton("发送")
        self._send_btn.setProperty("primary", True)
        self._send_btn.clicked.connect(self._on_send)
        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)

        input_row = QHBoxLayout()
        input_row.addWidget(self._input, 1)
        input_row.addWidget(self._cancel_btn)
        input_row.addWidget(self._send_btn)

        chat_area = QWidget()
        chat_layout = QVBoxLayout(chat_area)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.addWidget(self._view, 1)
        chat_layout.addLayout(input_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(chat_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 700])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._input.installEventFilter(self)
        self._wire_controller()
        self.refresh_threads()

    # ── 控制器信号 ─────────────────────────────────────────────────────

    def _wire_controller(self) -> None:
        c = self._controller
        c.state_changed.connect(self._on_state_changed)
        c.thread_created.connect(self._on_thread_created)
        c.token.connect(self._on_token)
        c.sources.connect(self._on_sources)
        c.interrupts.connect(self._on_interrupts)
        c.answer.connect(lambda text: self._view.add_message("assistant", text))
        c.finished.connect(self._on_round_finished)
        c.failed.connect(self._on_failed)

    def _on_state_changed(self, state: str) -> None:
        busy = state in {STATE_STREAMING, STATE_RESUMING}
        self._send_btn.setEnabled(state == STATE_IDLE)
        self._cancel_btn.setVisible(busy)
        # 计划状态机约定：AwaitingApproval 期间输入同样禁用，审批完成才解锁
        self._input.setEnabled(state == STATE_IDLE)

    def _on_thread_created(self, thread_id: str) -> None:
        self.refresh_threads()

    def _on_token(self, text: str) -> None:
        if not self._stream_started:
            self._view.begin_assistant_stream()
            self._stream_started = True
        self._view.append_stream_text(text)

    def _on_sources(self, sources: list) -> None:
        fresh = []
        for src in sources:
            key = str(src.get("source", ""))
            if key and key not in self._seen_sources:
                self._seen_sources.add(key)
                fresh.append(src)
        if fresh:
            self._view.add_widget(SourcesView(fresh))

    def _on_interrupts(self, payloads: list[InterruptPayload]) -> None:
        card = ApprovalCard(payloads)
        card.decided.connect(lambda dtype, c=card, p=payloads: self._on_decide(c, p, dtype))
        self._active_card = card
        self._view.add_widget(card)

    def _on_decide(
        self, card: ApprovalCard, payloads: list[InterruptPayload], dtype: str
    ) -> None:
        card.set_busy()
        decisions = [
            {"type": "approve"}
            if dtype == "approve"
            else {"type": "reject", "message": "用户拒绝了该操作"}
            for _ in payloads
        ]
        self._pending_card = card
        self._controller.resume(decisions)

    def _on_round_finished(self) -> None:
        card = getattr(self, "_pending_card", None)
        if card is not None:
            card.show_result("✓ 已提交审批，执行完成")
            self._pending_card = None
        self.refresh_threads()

    def _on_failed(self, detail: str) -> None:
        card = getattr(self, "_pending_card", None)
        if card is not None:
            card.show_result(f"⚠ {detail}")
            self._pending_card = None
        else:
            self._view.add_message("notice", f"执行失败：{detail}")

    # ── 会话列表 ─────────────────────────────────────────────────────

    def refresh_threads(self) -> None:
        start_worker(
            self._require_client().list_threads,
            self._on_threads_loaded,
            lambda detail, _s: show_toast(self, f"加载会话失败：{detail}", error=True),
        )

    def _on_threads_loaded(self, threads: list[ThreadInfo]) -> None:
        current_id = self._controller.thread_id
        self._threads = threads
        self._thread_list.blockSignals(True)
        self._thread_list.clear()
        for thread in threads:
            QListWidgetItem(thread.title or "新对话", self._thread_list)
        self._thread_list.blockSignals(False)
        # 保持当前选中项（新建会话后选中刚创建的）
        if threads:
            row = next((i for i, t in enumerate(threads) if t.id == current_id), 0)
            self._thread_list.setCurrentRow(row)
            if self._controller.thread_id is None:
                self._on_thread_selected(row)
        else:
            self._controller.set_thread(None)
            self._view.clear()

    def _selected_thread(self) -> ThreadInfo | None:
        row = self._thread_list.currentRow()
        if 0 <= row < len(self._threads):
            return self._threads[row]
        return None

    def _on_thread_selected(self, row: int) -> None:
        thread = self._threads[row] if 0 <= row < len(self._threads) else None
        if thread is None or thread.id == self._controller.thread_id:
            return
        self._controller.set_thread(thread.id)
        self._seen_sources.clear()
        self._view.clear()
        start_worker(
            lambda: self._require_client().get_thread(thread.id),
            self._on_thread_detail,
            lambda detail, _s: show_toast(self, f"加载消息失败：{detail}", error=True),
        )

    def _on_thread_detail(self, detail: ThreadDetail) -> None:
        self._view.render(detail.messages)
        self._view.scroll_to_bottom()

    def _on_new_thread(self) -> None:
        start_worker(
            lambda: self._require_client().create_thread(),
            lambda _t: self.refresh_threads(),
            lambda detail, _s: show_toast(self, f"新建会话失败：{detail}", error=True),
        )

    def _on_rename_thread(self) -> None:
        thread = self._selected_thread()
        if thread is None:
            return
        title, ok = QInputDialog.getText(self, "重命名会话", "新标题：", text=thread.title)
        if not ok or not title.strip():
            return
        start_worker(
            lambda: self._require_client().rename_thread(thread.id, title.strip()),
            lambda _t: self.refresh_threads(),
            lambda detail, _s: show_toast(self, f"重命名失败：{detail}", error=True),
        )

    def _on_delete_thread(self) -> None:
        thread = self._selected_thread()
        if thread is None:
            return
        confirm = QMessageBox.question(self, "删除会话", f"确定删除「{thread.title}」？")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._controller.set_thread(None)
        start_worker(
            lambda: self._require_client().delete_thread(thread.id),
            lambda _r: self.refresh_threads(),
            lambda detail, _s: show_toast(self, f"删除失败：{detail}", error=True),
        )

    # ── 发送与取消 ─────────────────────────────────────────────────────

    def eventFilter(self, obj: object, event: object) -> bool:  # noqa: N802
        if obj is self._input and event.type() == event.Type.KeyPress:
            if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and not (
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            ):
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._stream_started = False
        self._seen_sources.clear()
        if self._controller.send(text):
            self._input.clear()
            self._view.add_message("user", text)

    def _on_cancel(self) -> None:
        self._controller.cancel()
        self._view.add_message("notice", "已取消本轮对话")

    # ── 工具 ─────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """窗口关闭前调用：等待进行中的流式对话结束。"""

        self._controller.shutdown()

    def _require_client(self):
        assert self._session.client is not None, "会话未建立"
        return self._session.client
