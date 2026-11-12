"""平台管理 · 全局技能面板：本地快检 → 服务端预检 → 上传（三级校验）。"""

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

_MAX_ZIP_BYTES = 10 * 1024 * 1024  # 与服务端 skills_manager.MAX_SKILL_ZIP_BYTES 对齐
_ZIP_MAGIC = b"PK\x03\x04"


def _quick_check(path: Path) -> str:
    """浏览器/客户端本地快检：后缀、非空、大小、zip 魔数；通过返回空串。"""

    if path.suffix.lower() != ".zip":
        return "仅接受 .zip 技能包"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"无法读取文件：{exc}"
    if size == 0:
        return "文件内容为空"
    if size > _MAX_ZIP_BYTES:
        return f"压缩包超过 {_MAX_ZIP_BYTES // (1024 * 1024)}MB 上限"
    try:
        with path.open("rb") as fh:
            if fh.read(4) != _ZIP_MAGIC:
                return "文件不是有效的 zip 压缩包（魔数不符）"
    except OSError as exc:
        return f"无法读取文件：{exc}"
    return ""


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


class AdminSkillsPanel(QWidget):
    """全局技能管理（对所有租户生效）；预检不过不上传，冲突可覆盖。"""

    def __init__(self, session: SessionManager, parent: object = None) -> None:
        super().__init__(parent)
        self._session = session
        self._skills: list[SkillInfo] = []

        hint = QLabel(
            "全局技能对所有租户生效；上传前执行 本地快检 → 服务端预检 → 正式安装 三级校验"
        )
        hint.setProperty("hint", True)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["名称", "描述", "文件数", "更新时间", "状态"])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        upload_btn = QPushButton("上传全局技能包")
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
            self._require_client().admin_list_skills,
            self._on_loaded,
            lambda detail, _s: show_toast(self, f"加载全局技能失败：{detail}", error=True),
        )

    def _on_loaded(self, skills: list[SkillInfo]) -> None:
        self._skills = skills
        self._table.setRowCount(len(skills))
        for row, s in enumerate(skills):
            self._table.setItem(row, 0, QTableWidgetItem(s.name))
            self._table.setItem(row, 1, QTableWidgetItem(s.description))
            self._table.setItem(row, 2, QTableWidgetItem(str(s.file_count)))
            self._table.setItem(row, 3, QTableWidgetItem(_fmt_time(s.updated_at)))
            self._table.setItem(row, 4, QTableWidgetItem("有效" if s.valid else f"无效：{s.error}"))

    # ── 上传三级校验 ───────────────────────────────────────────────────

    def _on_pick_upload(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "选择 zip 技能包", "", "Zip 压缩包 (*.zip)")
        if not path_str:
            return
        path = Path(path_str)
        local_error = _quick_check(path)
        if local_error:
            show_toast(self, f"本地快检未通过：{local_error}", error=True)
            return
        start_worker(
            lambda: self._require_client().admin_validate_skill(path),
            lambda result: self._on_validated(path, result),
            lambda detail, _s: show_toast(self, f"预检请求失败：{detail}", error=True),
        )

    def _on_validated(self, path: Path, result: dict) -> None:
        errors = [str(e) for e in result.get("errors", [])]
        warnings = [str(w) for w in result.get("warnings", [])]
        if errors:
            QMessageBox.warning(self, "预检未通过", "\n".join(errors))
            return
        overwrite = False
        if warnings:
            confirm = QMessageBox.question(
                self, "预检警告", "\n".join(warnings) + "\n\n是否继续（覆盖安装）？"
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        start_worker(
            lambda: self._require_client().admin_upload_skill(path, overwrite=overwrite),
            lambda info: (show_toast(self, f"全局技能 {info.name} 已安装"), self.refresh()),
            lambda detail, _s: show_toast(self, f"安装失败：{detail}", error=True),
        )

    # ── 删除 ──────────────────────────────────────────────────────────

    def _on_delete(self) -> None:
        row = self._table.currentRow()
        if not (0 <= row < len(self._skills)):
            show_toast(self, "请先选择技能")
            return
        target = self._skills[row]
        confirm = QMessageBox.question(
            self, "删除全局技能", f"确定删除全局技能「{target.name}」？所有租户立即失效。"
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        start_worker(
            lambda: self._require_client().admin_delete_skill(target.name),
            lambda _r: (show_toast(self, f"已删除 {target.name}"), self.refresh()),
            lambda detail, _s: show_toast(self, f"删除失败：{detail}", error=True),
        )

    def _require_client(self):
        assert self._session.client is not None, "会话未建立"
        return self._session.client
