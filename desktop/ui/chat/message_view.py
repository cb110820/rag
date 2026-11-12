"""聊天气泡消息视图：QListWidget + 左右对齐气泡。

气泡文本用 QTextBrowser 承载：内容高度 = 文档自身排版高度（测量与渲染
同一引擎，emoji/中英混排/Markdown 均不会少算），彻底避免裁切问题。
assistant 消息按 Markdown 渲染（与 Web 端一致），user/notice 纯文本。
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QTextBrowser,
    QWidget,
)

from desktop.api.models import Message

# 气泡水平方向预留：QSS padding(13px×2) + 边框(1px×2) + 余量
_BUBBLE_HPAD = 30
# 气泡垂直方向预留：QSS padding(9px×2) + 边框 + 余量
_BUBBLE_VPAD = 24
# QListWidget 视口为 item 预留的垂直内边距（实测恒定 10px）
_ITEM_VINSET = 10


class _BubbleText(QTextBrowser):
    """气泡文本组件：按文档实际排版高度定高（测量即渲染）。

    assistant 按 Markdown 渲染；user/notice 纯文本。流式追加经 ``append_text``。
    """

    def __init__(self, role: str, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setProperty("bubbleRole", role)
        self._role = role
        self._raw = ""
        self.setReadOnly(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setOpenLinks(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_text(self, text: str) -> None:
        self._raw = text
        if self._role == "assistant":
            self.setMarkdown(text)
        else:
            self.setPlainText(text)

    def append_text(self, text: str) -> None:
        self.set_text(self._raw + text)

    def sync_geometry(self, max_width: int) -> None:
        """按内容自然宽度（封顶 max_width）定宽，按文档排版高度定高。"""

        doc = self.document()
        # 内容自然宽度：渲染后纯文本最长逻辑行 + 水平预留。
        # （QTextDocument.idealWidth 对 CJK 按"可断词"计算，长文也返回
        # 极小值，会把气泡压成一条竖条）
        plain = self.toPlainText() if self._role == "assistant" else self._raw
        fm = self.fontMetrics()
        longest = max((fm.horizontalAdvance(line) for line in plain.splitlines()), default=0)
        width = min(max(longest + _BUBBLE_HPAD, 120), max_width)
        self.setFixedWidth(width)
        # 文档排版高度：setFixedWidth 后 viewport 宽度即内容宽，
        # 显式对齐 doc 文本宽度再取高度，与最终渲染逐像素一致
        doc.setTextWidth(max(width - _BUBBLE_HPAD, 50))
        self.setFixedHeight(max(int(doc.size().height()) + _BUBBLE_VPAD, 32))


class MessageView(QListWidget):
    """消息列表：user 右对齐蓝气泡，assistant 左对齐白气泡，notice 居中提示条。"""

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)

    def render(self, messages: list[Message]) -> None:
        self.clear()
        for message in messages:
            if message.role in {"user", "assistant"} and message.content.strip():
                self.add_message(message.role, message.content)
        self.scroll_to_bottom()

    def add_message(self, role: str, text: str) -> None:
        bubble_role = role if role in {"user", "notice"} else "assistant"
        align = {
            "user": Qt.AlignmentFlag.AlignRight,
            "notice": Qt.AlignmentFlag.AlignHCenter,
        }.get(bubble_role, Qt.AlignmentFlag.AlignLeft)
        bubble = self._add_bubble(bubble_role, align)
        bubble.set_text(text)
        bubble.sync_geometry(self._bubble_max_width())
        self._sync_hint(self.item(self.count() - 1), self.itemWidget(self.item(self.count() - 1)))
        self.scroll_to_bottom()

    # ── 流式气泡与嵌入组件 ────────────────────────────────────────────

    def begin_assistant_stream(self) -> None:
        """开启本轮流式气泡；后续 append_stream_text 增量填充。"""

        self._stream_bubble = self._add_bubble("assistant", Qt.AlignmentFlag.AlignLeft)
        self._stream_bubble.set_text("")
        self._stream_bubble.sync_geometry(self._bubble_max_width())
        self._sync_hint(
            self.item(self.count() - 1), self.itemWidget(self.item(self.count() - 1))
        )

    def append_stream_text(self, text: str) -> None:
        bubble = getattr(self, "_stream_bubble", None)
        if bubble is None:
            self.begin_assistant_stream()
            bubble = self._stream_bubble
        bubble.append_text(text)
        bubble.sync_geometry(self._bubble_max_width())
        self._refresh_last_item()

    def add_widget(self, widget: QWidget) -> None:
        """嵌入任意组件（审批卡片、来源 chips 等）作为列表项。"""

        item = QListWidgetItem(self)
        self._sync_hint(item, widget)
        self.addItem(item)
        self.setItemWidget(item, widget)
        self.scroll_to_bottom()

    def _refresh_last_item(self) -> None:
        if not self.count():
            return
        item = self.item(self.count() - 1)
        row = self.itemWidget(item)
        if row is not None:
            row_layout = row.layout()
            if row_layout is not None:
                row_layout.activate()
            self._sync_hint(item, row)
        self.scroll_to_bottom()

    def _sync_hint(self, item: QListWidgetItem, row: QWidget) -> None:
        """按 row 的实际需求 + 视口内边距补偿设置 item 高度。"""

        hint = row.sizeHint()
        item.setSizeHint(QSize(hint.width(), hint.height() + _ITEM_VINSET))

    def _add_bubble(self, bubble_role: str, align: Qt.AlignmentFlag) -> _BubbleText:
        bubble = _BubbleText(bubble_role)

        row = QWidget()
        row.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(8, 4, 8, 4)
        if align == Qt.AlignmentFlag.AlignRight:
            layout.addStretch(1)
            layout.addWidget(bubble)
        elif align == Qt.AlignmentFlag.AlignLeft:
            layout.addWidget(bubble)
            layout.addStretch(1)
        else:
            layout.addStretch(1)
            layout.addWidget(bubble)
            layout.addStretch(1)

        item = QListWidgetItem(self)
        self.addItem(item)
        self.setItemWidget(item, row)
        return bubble

    def _bubble_max_width(self) -> int:
        return max(int(self.viewport().width() * 0.72), 200)

    def scroll_to_bottom(self) -> None:
        if self.count():
            self.scrollToItem(
                self.item(self.count() - 1), QListWidget.ScrollHint.PositionAtBottom
            )

    def resizeEvent(self, event: object) -> None:  # noqa: N802 — Qt 命名约定
        super().resizeEvent(event)
        # 宽度变化时重排气泡：重定宽度与文档高度、item 高度
        for i in range(self.count()):
            item = self.item(i)
            row = self.itemWidget(item)
            if row is None:
                continue
            bubble = row.findChild(_BubbleText)
            if bubble is not None:
                bubble.sync_geometry(self._bubble_max_width())
            row_layout = row.layout()
            if row_layout is not None:  # QScrollArea 等嵌入组件无顶层 layout
                row_layout.activate()
            self._sync_hint(item, row)
