"""知识库面板：拖拽/按钮上传 + 文件列表 + 删除。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.api.models import KnowledgeFile, UploadResult
from desktop.state.session import SessionManager
from desktop.ui.widgets.toast import show_toast
from desktop.workers.base import start_worker

_SUPPORTED_FILTER = "知识库文档 (*.txt *.md *.html *.pdf *.docx)"


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


class KnowledgePanel(QWidget):
    def __init__(self, session: SessionManager, parent: object = None) -> None:
        super().__init__(parent)
        self._session = session
        self._files: list[KnowledgeFile] = []
        self.setAcceptDrops(True)

        hint = QLabel("支持 txt / md / html / pdf / docx，单文件 ≤ 50MB；可直接拖拽文件到列表")
        hint.setProperty("hint", True)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["文件名", "大小", "修改时间"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        upload_btn = QPushButton("上传文件")
        upload_btn.setProperty("primary", True)
        upload_btn.clicked.connect(self._on_pick_upload)
        delete_btn = QPushButton("删除选中")
        delete_btn.clicked.connect(self._on_delete)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh)
        toolbar = QHBoxLayout()
        toolbar.addWidget(upload_btn)
        toolbar.addWidget(delete_btn)
        toolbar.addWidget(refresh_btn)
        toolbar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self._table, 1)
        layout.addLayout(toolbar)

        self.refresh()

    # ── 列表 ──────────────────────────────────────────────────────────

    def refresh(self) -> None:
        start_worker(
            self._require_client().list_knowledge_files,
            self._on_loaded,
            lambda detail, _s: show_toast(self, f"加载知识库失败：{detail}", error=True),
        )

    def _on_loaded(self, files: list[KnowledgeFile]) -> None:
        self._files = files
        self._table.setRowCount(len(files))
        for row, f in enumerate(files):
            self._table.setItem(row, 0, QTableWidgetItem(f.name))
            self._table.setItem(row, 1, QTableWidgetItem(_fmt_size(f.size_bytes)))
            self._table.setItem(row, 2, QTableWidgetItem(_fmt_time(f.modified_at)))

    # ── 上传 ──────────────────────────────────────────────────────────

    def _on_pick_upload(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择知识库文档", "", _SUPPORTED_FILTER)
        if paths:
            self._upload_paths([Path(p) for p in paths])

    def _upload_paths(self, paths: list[Path]) -> None:
        for path in paths:
            start_worker(
                lambda p=path: self._require_client().upload_knowledge(p),
                self._make_upload_handler(path),
                lambda detail, _s, p=path: show_toast(
                    self, f"{p.name} 上传失败：{detail}", error=True
                ),
            )

    def _make_upload_handler(self, path: Path):
        def _done(result: UploadResult) -> None:
            action = "覆盖" if result.overwritten else "上传"
            show_toast(self, f"{path.name} {action}成功")
            self.refresh()

        return _done

    # ── 拖拽 ──────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: object) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: object) -> None:  # noqa: N802
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self._upload_paths(paths)

    # ── 删除 ──────────────────────────────────────────────────────────

    def _on_delete(self) -> None:
        row = self._table.currentRow()
        if not (0 <= row < len(self._files)):
            show_toast(self, "请先选择要删除的文件")
            return
        target = self._files[row]
        confirm = QMessageBox.question(self, "删除文件", f"确定从知识库删除「{target.name}」？")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        start_worker(
            lambda: self._require_client().delete_knowledge_file(target.source_id),
            lambda _r: (show_toast(self, f"已删除 {target.name}"), self.refresh()),
            lambda detail, _s: show_toast(self, f"删除失败：{detail}", error=True),
        )

    def _require_client(self):
        assert self._session.client is not None, "会话未建立"
        return self._session.client
