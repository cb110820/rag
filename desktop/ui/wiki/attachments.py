"""Wiki 附件管理对话框：列表 / 上传 / 下载 / 删除。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from desktop.api.client import RagApiClient
from desktop.api.models import WikiAttachment
from desktop.ui.widgets.toast import show_toast
from desktop.workers.base import start_worker


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


class AttachmentsDialog(QDialog):
    """某页面的附件管理（原样存储，不进版本与 RAG 索引）。"""

    def __init__(self, client: RagApiClient, slug: str, parent: object = None) -> None:
        super().__init__(parent)
        self._client = client
        self._slug = slug
        self._items: list[WikiAttachment] = []
        self.setWindowTitle(f"附件管理 · {slug}")
        self.resize(560, 400)

        hint = QLabel("附件原样存储，单文件 ≤ 20MB，不参与版本历史与检索")
        hint.setProperty("hint", True)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["文件名", "大小", "上传时间"])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        upload_btn = QPushButton("上传")
        upload_btn.clicked.connect(self._on_upload)
        download_btn = QPushButton("下载")
        download_btn.clicked.connect(self._on_download)
        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(self._on_delete)
        buttons = QHBoxLayout()
        buttons.addWidget(upload_btn)
        buttons.addWidget(download_btn)
        buttons.addWidget(delete_btn)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self._table, 1)
        layout.addLayout(buttons)

        self.refresh()

    def refresh(self) -> None:
        start_worker(
            lambda: self._client.list_wiki_attachments(self._slug),
            self._on_loaded,
            lambda detail, _s: show_toast(self, f"加载附件失败：{detail}", error=True),
        )

    def _on_loaded(self, items: list[WikiAttachment]) -> None:
        self._items = items
        self._table.setRowCount(len(items))
        for row, att in enumerate(items):
            self._table.setItem(row, 0, QTableWidgetItem(att.filename))
            self._table.setItem(row, 1, QTableWidgetItem(_fmt_size(att.size_bytes)))
            self._table.setItem(row, 2, QTableWidgetItem(_fmt_time(att.created_at)))

    def _selected(self) -> WikiAttachment | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def _on_upload(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择附件")
        if not path:
            return
        start_worker(
            lambda: self._client.upload_wiki_attachment(self._slug, Path(path)),
            lambda _att: (show_toast(self, "附件已上传"), self.refresh()),
            lambda detail, _s: show_toast(self, f"上传失败：{detail}", error=True),
        )

    def _on_download(self) -> None:
        att = self._selected()
        if att is None:
            show_toast(self, "请先选择附件")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存附件", att.filename)
        if not save_path:
            return
        start_worker(
            lambda: self._client.download_wiki_attachment(att.id),
            lambda result: self._save_download(result, save_path),
            lambda detail, _s: show_toast(self, f"下载失败：{detail}", error=True),
        )

    def _save_download(self, result: tuple[str, bytes], save_path: str) -> None:
        _filename, content = result
        try:
            Path(save_path).write_bytes(content)
        except OSError as exc:
            show_toast(self, f"保存失败：{exc}", error=True)
            return
        show_toast(self, "附件已保存")

    def _on_delete(self) -> None:
        att = self._selected()
        if att is None:
            show_toast(self, "请先选择附件")
            return
        confirm = QMessageBox.question(self, "删除附件", f"确定删除「{att.filename}」？")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        start_worker(
            lambda: self._client.delete_wiki_attachment(att.id),
            lambda _r: (show_toast(self, "附件已删除"), self.refresh()),
            lambda detail, _s: show_toast(self, f"删除失败：{detail}", error=True),
        )
