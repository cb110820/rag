"""平台管理 · 租户面板：租户 CRUD + 配额配置 + 用量统计（platform_admin 专用）。"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.state.session import SessionManager
from desktop.ui.widgets.toast import show_toast
from desktop.workers.base import start_worker

_STATUSES = ["active", "suspended"]
_PLANS = ["free", "pro", "enterprise"]
_DEFAULT_TENANT_ID = "default"


class TenantEditDialog(QDialog):
    """新建/编辑租户表单；payload() 返回服务端请求体。"""

    def __init__(self, tenant: dict[str, Any] | None = None, parent: object = None) -> None:
        super().__init__(parent)
        self._editing = tenant is not None
        self.setWindowTitle("编辑租户" if self._editing else "新建租户")

        self._name = QLineEdit((tenant or {}).get("name", ""))
        self._slug = QLineEdit((tenant or {}).get("slug", ""))
        self._slug.setEnabled(not self._editing)  # slug 创建后不可改
        self._status = QComboBox()
        self._status.addItems(_STATUSES)
        self._status.setCurrentText((tenant or {}).get("status", "active"))
        self._plan = QComboBox()
        self._plan.addItems(_PLANS)
        self._plan.setCurrentText((tenant or {}).get("plan", "free"))
        self._max_users = QSpinBox(minimum=1, maximum=100000)
        self._max_users.setValue(int((tenant or {}).get("max_users", 5)))
        self._max_storage = QSpinBox(minimum=1, maximum=1048576)
        self._max_storage.setValue(int((tenant or {}).get("max_storage_mb", 100)))
        self._max_facts = QSpinBox(minimum=0, maximum=1000000)
        self._max_facts.setValue(int((tenant or {}).get("max_facts", 200)))

        form = QFormLayout()
        form.addRow("名称", self._name)
        form.addRow("slug", self._slug)
        form.addRow("状态", self._status)
        form.addRow("套餐", self._plan)
        form.addRow("最大用户数", self._max_users)
        form.addRow("存储上限 (MB)", self._max_storage)
        form.addRow("事实记忆上限", self._max_facts)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self._error = QLabel("")
        self._error.setStyleSheet("color: #e11d48;")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._error)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if not self._name.text().strip():
            self._error.setText("名称不能为空")
            return
        if not self._editing and not self._slug.text().strip():
            self._error.setText("slug 不能为空")
            return
        self.accept()

    def payload(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self._name.text().strip(),
            "status": self._status.currentText(),
            "plan": self._plan.currentText(),
            "max_users": self._max_users.value(),
            "max_storage_mb": self._max_storage.value(),
            "max_facts": self._max_facts.value(),
        }
        if not self._editing:
            data["slug"] = self._slug.text().strip()
        return data


class TenantsPanel(QWidget):
    def __init__(self, session: SessionManager, parent: object = None) -> None:
        super().__init__(parent)
        self._session = session
        self._tenants: list[dict[str, Any]] = []

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["名称", "slug", "状态", "套餐", "用户上限", "事实上限"]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        new_btn = QPushButton("新建")
        new_btn.clicked.connect(self._on_new)
        edit_btn = QPushButton("编辑")
        edit_btn.clicked.connect(self._on_edit)
        del_btn = QPushButton("删除")
        del_btn.clicked.connect(self._on_delete)
        stats_btn = QPushButton("统计")
        stats_btn.clicked.connect(self._on_stats)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh)
        toolbar = QHBoxLayout()
        for btn in (new_btn, edit_btn, del_btn, stats_btn, refresh_btn):
            toolbar.addWidget(btn)
        toolbar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table, 1)
        layout.addLayout(toolbar)

        self.refresh()

    def refresh(self) -> None:
        start_worker(
            self._require_client().list_tenants,
            self._on_loaded,
            lambda detail, _s: show_toast(self, f"加载租户失败：{detail}", error=True),
        )

    def _on_loaded(self, tenants: list[dict[str, Any]]) -> None:
        self._tenants = tenants
        self._table.setRowCount(len(tenants))
        for row, t in enumerate(tenants):
            self._table.setItem(row, 0, QTableWidgetItem(str(t.get("name", ""))))
            self._table.setItem(row, 1, QTableWidgetItem(str(t.get("slug", ""))))
            self._table.setItem(row, 2, QTableWidgetItem(str(t.get("status", ""))))
            self._table.setItem(row, 3, QTableWidgetItem(str(t.get("plan", ""))))
            self._table.setItem(row, 4, QTableWidgetItem(str(t.get("max_users", ""))))
            self._table.setItem(row, 5, QTableWidgetItem(str(t.get("max_facts", ""))))

    def _selected(self) -> dict[str, Any] | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._tenants):
            return self._tenants[row]
        return None

    def _on_new(self) -> None:
        dialog = TenantEditDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        start_worker(
            lambda: self._require_client().create_tenant(dialog.payload()),
            lambda _t: (show_toast(self, "租户已创建"), self.refresh()),
            lambda detail, _s: show_toast(self, f"创建失败：{detail}", error=True),
        )

    def _on_edit(self) -> None:
        tenant = self._selected()
        if tenant is None:
            show_toast(self, "请先选择租户")
            return
        dialog = TenantEditDialog(tenant, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        tenant_id = str(tenant.get("id", ""))
        start_worker(
            lambda: self._require_client().update_tenant(tenant_id, dialog.payload()),
            lambda _t: (show_toast(self, "已保存"), self.refresh()),
            lambda detail, _s: show_toast(self, f"保存失败：{detail}", error=True),
        )

    def _on_delete(self) -> None:
        tenant = self._selected()
        if tenant is None:
            show_toast(self, "请先选择租户")
            return
        tenant_id = str(tenant.get("id", ""))
        if tenant_id == _DEFAULT_TENANT_ID:
            show_toast(self, "default 租户不可删除", error=True)
            return
        confirm = QMessageBox.warning(
            self, "删除租户",
            f"确定删除租户「{tenant.get('name')}」？其用户/会话/记忆/知识库将全部清除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        start_worker(
            lambda: self._require_client().delete_tenant(tenant_id),
            lambda _r: (show_toast(self, "租户已删除"), self.refresh()),
            lambda detail, _s: show_toast(self, f"删除失败：{detail}", error=True),
        )

    def _on_stats(self) -> None:
        tenant = self._selected()
        if tenant is None:
            show_toast(self, "请先选择租户")
            return
        tenant_id = str(tenant.get("id", ""))

        def _show(stats: dict[str, Any]) -> None:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"租户统计 · {tenant.get('name')}")
            dialog.resize(420, 320)
            view = QPlainTextEdit()
            view.setReadOnly(True)
            lines = [f"{key}: {value}" for key, value in stats.items()]
            view.setPlainText("\n".join(lines))
            layout = QVBoxLayout(dialog)
            layout.addWidget(view)
            dialog.exec()

        start_worker(
            lambda: self._require_client().tenant_stats(tenant_id),
            _show,
            lambda detail, _s: show_toast(self, f"读取统计失败：{detail}", error=True),
        )

    def _require_client(self):
        assert self._session.client is not None, "会话未建立"
        return self._session.client
