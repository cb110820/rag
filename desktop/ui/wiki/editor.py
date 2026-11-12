"""Markdown 编辑器：工具栏 + QPlainTextEdit + 分屏预览（三视图模式）。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QSplitter, QToolBar, QVBoxLayout, QWidget

from desktop.ui.wiki.preview import MarkdownPreview

_MODE_SPLIT, _MODE_EDIT, _MODE_PREVIEW = 0, 1, 2


class MarkdownEditor(QWidget):
    """编辑 + 预览组合件；content() / set_content() 与面板交互。"""

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self.edit = QPlainTextEdit()
        self.edit.setPlaceholderText("在此编写 Markdown 正文…")
        self.preview = MarkdownPreview()

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self.edit)
        self._splitter.addWidget(self.preview)
        self._splitter.setSizes([500, 500])

        toolbar = QToolBar()
        toolbar.setIconSize(toolbar.iconSize() * 0.9)
        self._add_action(toolbar, "粗体", lambda: self._wrap("**", "**"))
        self._add_action(toolbar, "斜体", lambda: self._wrap("*", "*"))
        self._add_action(toolbar, "标题", lambda: self._line_prefix("## "))
        self._add_action(toolbar, "链接", lambda: self._wrap("[", "](https://)"))
        self._add_action(toolbar, "行内码", lambda: self._wrap("`", "`"))
        self._add_action(toolbar, "代码块", lambda: self._wrap("```\n", "\n```"))
        self._add_action(toolbar, "列表", lambda: self._line_prefix("- "))
        self._add_action(toolbar, "引用", lambda: self._line_prefix("> "))
        self._add_action(toolbar, "分割线", lambda: self._insert("\n---\n"))
        toolbar.addSeparator()
        self._add_action(
            toolbar, "分屏", lambda: self._set_mode(_MODE_SPLIT), checkable=True, checked=True
        )
        self._add_action(toolbar, "仅编辑", lambda: self._set_mode(_MODE_EDIT))
        self._add_action(toolbar, "仅预览", lambda: self._set_mode(_MODE_PREVIEW))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toolbar)
        layout.addWidget(self._splitter, 1)

        self.edit.textChanged.connect(lambda: self.preview.schedule_render(self.content()))

    def _add_action(
        self,
        toolbar: QToolBar,
        text: str,
        slot,
        checkable: bool = False,
        checked: bool = False,
    ) -> QAction:
        action = QAction(text, self)
        action.setCheckable(checkable)
        action.setChecked(checked)
        action.triggered.connect(slot)
        toolbar.addAction(action)
        return action

    def _set_mode(self, mode: int) -> None:
        self.edit.setVisible(mode != _MODE_PREVIEW)
        self.preview.setVisible(mode != _MODE_EDIT)
        if mode == _MODE_PREVIEW:
            self.preview.schedule_render(self.content())

    # ── 光标操作 ──────────────────────────────────────────────────────

    def _wrap(self, prefix: str, suffix: str) -> None:
        cursor = self.edit.textCursor()
        selected = cursor.selectedText()
        cursor.insertText(f"{prefix}{selected}{suffix}")
        self.edit.setTextCursor(cursor)

    def _line_prefix(self, prefix: str) -> None:
        cursor = self.edit.textCursor()
        cursor.beginEditBlock()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        cursor.insertText(prefix)
        cursor.endEditBlock()
        self.edit.setTextCursor(cursor)

    def _insert(self, text: str) -> None:
        self.edit.textCursor().insertText(text)

    # ── 内容 ──────────────────────────────────────────────────────────

    def content(self) -> str:
        return self.edit.toPlainText()

    def set_content(self, text: str) -> None:
        self.edit.setPlainText(text)
        self.preview.schedule_render(text)
