"""租户技能面板：技能列表 / zip 上传（409 冲突可覆盖重试）/ 删除。"""

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

from desktop.api.models import SkillInfo
from desktop.state.session import SessionManager
from desktop.ui.widgets.toast import show_toast
from desktop.workers.base import start_worker


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


class SkillsPanel(QWidget):
    def __init__(self, session: SessionManager, parent: object = None) -> None:
        super().__init__(parent)
        self._session = session
        self._skills: list[SkillInfo] = []

        hint = QLabel(
            "zip 技能包：根目录 SKILL.md 或唯一顶层目录 <name>/SKILL.md；"
            "上传后下一条对话自动生效"
        )
        hint.setProperty("hint", True)
        hint.setWordWrap(True)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["名称", "描述", "文件数", "更新时间", "状态"])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        upload_btn = QPushButton("上传技能包")
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

    def refresh(self) -> None:
        start_worker(
            self._require_client().list_skills,
            self._on_loaded,
            lambda detail, _s: show_toast(self, f"加载技能失败：{detail}", error=True),
        )

    def _on_loaded(self, skills: list[SkillInfo]) -> None:
        self._skills = skills
        self._table.setRowCount(len(skills))
        for row, s in enumerate(skills):
            self._table.setItem(row, 0, QTableWidgetItem(s.name))
            self._table.setItem(row, 1, QTableWidgetItem(s.description))
            self._table.setItem(row, 2, QTableWidgetItem(str(s.file_count)))
            self._table.setItem(row, 3, QTableWidgetItem(_fmt_time(s.updated_at)))
            status = QTableWidgetItem("有效" if s.valid else f"无效：{s.error}")
            if not s.valid:
                status.setForeground(self.palette().color(self.palette().ColorRole.PlaceholderText))
            self._table.setItem(row, 4, status)

    # ── 上传 ──────────────────────────────────────────────────────────

    def _on_pick_upload(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 zip 技能包", "", "Zip 压缩包 (*.zip)")
        if path:
            self._upload(Path(path), overwrite=False)

    def _upload(self, path: Path, overwrite: bool) -> None:
        start_worker(
            lambda: self._require_client().upload_skill(path, overwrite=overwrite),
            lambda info: (show_toast(self, f"技能 {info.name} 已安装"), self.refresh()),
            lambda detail, status: self._on_upload_error(path, detail, status),
        )

    def _on_upload_error(self, path: Path, detail: str, status: int) -> None:
        # 409 同名冲突：询问用户后覆盖重试
        if status == 409:
            confirm = QMessageBox.question(
                self, "技能已存在", f"{detail}\n\n是否覆盖安装？"
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self._upload(path, overwrite=True)
            return
        show_toast(self, f"{path.name} 上传失败：{detail}", error=True)

    # ── 删除 ──────────────────────────────────────────────────────────

    def _on_delete(self) -> None:
        row = self._table.currentRow()
        if not (0 <= row < len(self._skills)):
            show_toast(self, "请先选择要删除的技能")
            return
        target = self._skills[row]
        confirm = QMessageBox.question(self, "删除技能", f"确定删除技能「{target.name}」？")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        start_worker(
            lambda: self._require_client().delete_skill(target.name),
            lambda _r: (show_toast(self, f"已删除 {target.name}"), self.refresh()),
            lambda detail, _s: show_toast(self, f"删除失败：{detail}", error=True),
        )

    def _require_client(self):
        assert self._session.client is not None, "会话未建立"
        return self._session.client
