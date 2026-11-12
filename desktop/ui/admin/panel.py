"""平台管理页容器：租户管理与全局技能两个 tab。"""

from __future__ import annotations

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from desktop.state.session import SessionManager
from desktop.ui.admin.skills import AdminSkillsPanel
from desktop.ui.admin.tenants import TenantsPanel


class AdminPanel(QWidget):
    def __init__(self, session: SessionManager, parent: object = None) -> None:
        super().__init__(parent)
        tabs = QTabWidget()
        tabs.addTab(TenantsPanel(session), "租户管理")
        tabs.addTab(AdminSkillsPanel(session), "全局技能")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)
