"""Wiki 面板：搜索（防抖）+ 页面列表 + 编辑器 + 保存/新建/导入/历史/附件。"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from desktop.api.models import WikiPage, WikiPageSummary
from desktop.state.session import SessionManager
from desktop.ui.widgets.toast import show_toast
from desktop.ui.wiki.attachments import AttachmentsDialog
from desktop.ui.wiki.editor import MarkdownEditor
from desktop.ui.wiki.history import HistoryDialog
from desktop.workers.base import start_worker

# 与服务端契约一致（wiki.py _validate_slug 的错误文案对应的规则）
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

_IMPORT_FILTER = "可导入文档 (*.md *.txt *.html *.pdf *.docx)"


class WikiPanel(QWidget):
    def __init__(self, session: SessionManager, parent: object = None) -> None:
        super().__init__(parent)
        self._session = session
        self._pages: list[WikiPageSummary] = []
        self._current_slug: str | None = None

        # ── 左侧：搜索 + 列表 + 操作 ──
        self._search = QLineEdit()
        self._search.setPlaceholderText("搜索 Wiki 页面…")
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self._on_search)
        self._search.textChanged.connect(lambda _t: self._search_timer.start())

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_selected)

        new_btn = QPushButton("新建")
        new_btn.clicked.connect(self._on_new)
        import_btn = QPushButton("导入")
        import_btn.clicked.connect(self._on_import)
        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(self._on_delete)
        side_buttons = QHBoxLayout()
        side_buttons.addWidget(new_btn)
        side_buttons.addWidget(import_btn)
        side_buttons.addWidget(delete_btn)

        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self._search)
        side_layout.addWidget(self._list, 1)
        side_layout.addLayout(side_buttons)

        # ── 右侧：元信息 + 编辑器 + 操作 ──
        self._title = QLineEdit()
        self._title.setPlaceholderText("页面标题")
        self._summary = QLineEdit()
        self._summary.setPlaceholderText("本次修改摘要（可选，写入版本历史）")
        self._editor = MarkdownEditor()

        save_btn = QPushButton("保存")
        save_btn.setProperty("primary", True)
        save_btn.clicked.connect(self._on_save)
        history_btn = QPushButton("历史")
        history_btn.clicked.connect(self._on_history)
        attach_btn = QPushButton("附件")
        attach_btn.clicked.connect(self._on_attachments)
        actions = QHBoxLayout()
        actions.addWidget(save_btn)
        actions.addWidget(history_btn)
        actions.addWidget(attach_btn)
        actions.addStretch(1)

        editor_area = QWidget()
        editor_layout = QVBoxLayout(editor_area)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.addRow("标题", self._title)
        form.addRow("摘要", self._summary)
        editor_layout.addLayout(form)
        editor_layout.addWidget(self._editor, 1)
        editor_layout.addLayout(actions)

        splitter = QSplitter()
        splitter.addWidget(sidebar)
        splitter.addWidget(editor_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 700])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._set_editor_enabled(False)
        self.refresh()

    # ── 列表与搜索 ─────────────────────────────────────────────────────

    def refresh(self) -> None:
        start_worker(
            self._require_client().list_wiki_pages,
            self._on_pages_loaded,
            lambda detail, _s: show_toast(self, f"加载 Wiki 失败：{detail}", error=True),
        )

    def _on_pages_loaded(self, pages: list[WikiPageSummary]) -> None:
        self._fill_list(pages)

    def _fill_list(self, pages: list[WikiPageSummary]) -> None:
        self._pages = pages
        self._list.blockSignals(True)
        self._list.clear()
        for page in pages:
            item = QListWidgetItem(page.title or page.slug, self._list)
            item.setToolTip(page.slug)
        self._list.blockSignals(False)
        if pages and self._current_slug:
            row = next((i for i, p in enumerate(pages) if p.slug == self._current_slug), -1)
            if row >= 0:
                self._list.setCurrentRow(row)

    def _on_search(self) -> None:
        query = self._search.text().strip()
        if not query:
            self.refresh()
            return
        start_worker(
            lambda: self._require_client().search_wiki(query),
            lambda hits: self._fill_list(
                [WikiPageSummary(id="", slug=h.slug, title=h.title,
                                 created_at=0.0, updated_at=h.updated_at) for h in hits]
            ),
            lambda detail, _s: show_toast(self, f"搜索失败：{detail}", error=True),
        )

    # ── 页面选择与编辑 ─────────────────────────────────────────────────

    def _on_selected(self, row: int) -> None:
        page = self._pages[row] if 0 <= row < len(self._pages) else None
        if page is None or page.slug == self._current_slug:
            return
        start_worker(
            lambda: self._require_client().get_wiki_page(page.slug),
            self._on_page_loaded,
            lambda detail, _s: show_toast(self, f"加载页面失败：{detail}", error=True),
        )

    def _on_page_loaded(self, page: WikiPage) -> None:
        self._current_slug = page.slug
        self._title.setText(page.title)
        self._summary.clear()
        self._editor.set_content(page.content)
        self._set_editor_enabled(True)

    def _set_editor_enabled(self, enabled: bool) -> None:
        for widget in (self._title, self._summary, self._editor):
            widget.setEnabled(enabled)

    def _on_save(self) -> None:
        if self._current_slug is None:
            return
        title = self._title.text().strip()
        content = self._editor.content().strip()
        if not title or not content:
            show_toast(self, "标题与正文不能为空", error=True)
            return
        slug = self._current_slug
        summary = self._summary.text().strip()
        start_worker(
            lambda: self._require_client().update_wiki_page(slug, title, content, summary),
            lambda _page: (show_toast(self, "已保存"), self.refresh()),
            lambda detail, _s: show_toast(self, f"保存失败：{detail}", error=True),
        )

    # ── 新建 / 导入 / 删除 ─────────────────────────────────────────────

    def _on_new(self) -> None:
        slug, ok = QInputDialog.getText(
            self, "新建 Wiki 页面", "slug（小写字母/数字/连字符/下划线）："
        )
        if not ok:
            return
        slug = slug.strip()
        if not _SLUG_RE.match(slug):
            show_toast(
                self, "slug 只能含小写字母、数字、连字符、下划线，且以字母或数字开头",
                error=True,
            )
            return
        title, ok = QInputDialog.getText(self, "新建 Wiki 页面", "标题：")
        if not ok or not title.strip():
            return
        start_worker(
            lambda: self._require_client().create_wiki_page(
                slug, title.strip(), f"# {title.strip()}\n", "新建页面"
            ),
            lambda page: (show_toast(self, "已创建"), self._open_page(page.slug)),
            lambda detail, _s: show_toast(self, f"创建失败：{detail}", error=True),
        )

    def _open_page(self, slug: str) -> None:
        self._current_slug = slug
        self.refresh()
        start_worker(
            lambda: self._require_client().get_wiki_page(slug),
            self._on_page_loaded,
            lambda detail, _s: show_toast(self, f"加载页面失败：{detail}", error=True),
        )

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入文档为 Wiki 页面", "", _IMPORT_FILTER)
        if not path:
            return
        start_worker(
            lambda: self._require_client().import_wiki_document(Path(path)),
            lambda page: (
                show_toast(self, f"已导入为 {page.slug}"),
                self._open_page(page.slug),
            ),
            lambda detail, _s: show_toast(self, f"导入失败：{detail}", error=True),
        )

    def _on_delete(self) -> None:
        row = self._list.currentRow()
        page = self._pages[row] if 0 <= row < len(self._pages) else None
        if page is None:
            show_toast(self, "请先选择页面")
            return
        confirm = QMessageBox.question(
            self, "删除页面", f"确定删除「{page.title}」（{page.slug}）？历史版本将一并删除。"
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        start_worker(
            lambda: self._require_client().delete_wiki_page(page.slug),
            lambda _r: self._on_deleted(page.slug),
            lambda detail, _s: show_toast(self, f"删除失败：{detail}", error=True),
        )

    def _on_deleted(self, slug: str) -> None:
        show_toast(self, "已删除")
        if self._current_slug == slug:
            self._current_slug = None
            self._title.clear()
            self._summary.clear()
            self._editor.set_content("")
            self._set_editor_enabled(False)
        self.refresh()

    # ── 历史 / 附件 ─────────────────────────────────────────────────────

    def _on_history(self) -> None:
        if self._current_slug is None:
            show_toast(self, "请先选择页面")
            return
        HistoryDialog(
            self._require_client(),
            self._current_slug,
            self._editor.content(),
            on_reverted=lambda: self._open_page(self._current_slug or ""),
            parent=self,
        ).exec()

    def _on_attachments(self) -> None:
        if self._current_slug is None:
            show_toast(self, "请先选择页面")
            return
        AttachmentsDialog(self._require_client(), self._current_slug, self).exec()

    def _require_client(self):
        assert self._session.client is not None, "会话未建立"
        return self._session.client
