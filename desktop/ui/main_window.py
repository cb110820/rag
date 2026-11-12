"""主窗口：左侧导航 + QStackedWidget 页面栈；按角色显隐管理页。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from desktop import config
from desktop.state.session import SessionManager
from desktop.ui.admin.panel import AdminPanel
from desktop.ui.chat.panel import ChatPanel
from desktop.ui.knowledge.panel import KnowledgePanel
from desktop.ui.memory.panel import MemoryPanel
from desktop.ui.skills.panel import SkillsPanel
from desktop.ui.wiki.panel import WikiPanel
from desktop.workers.base import shutdown_all


class _PlaceholderPage(QWidget):
    """后续阶段替换为正式面板的占位页。"""

    def __init__(self, text: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setProperty("hint", True)
        layout.addWidget(label)


class MainWindow(QMainWindow):
    def __init__(self, session: SessionManager, parent: object = None) -> None:
        super().__init__(parent)
        self._session = session
        # 退出登录标记：app 主循环据此决定"回登录页"还是"退出进程"
        self.logout_requested = False
        self.setWindowTitle("RAG 知识助手 · 桌面端")
        self.resize(1100, 720)

        # ── 导航 ──
        self._nav = QListWidget()
        self._nav.setObjectName("navList")
        self._nav.setFixedWidth(156)
        self._nav.addItem("对话")
        self._nav.addItem("Wiki")
        self._nav.addItem("知识库")
        self._nav.addItem("记忆")
        self._nav.addItem("技能")
        if session.is_platform_admin:
            self._nav.addItem("平台管理")

        # ── 页面栈 ──
        self._pages = QStackedWidget()
        self._pages.addWidget(ChatPanel(session))
        self._pages.addWidget(WikiPanel(session))
        self._pages.addWidget(KnowledgePanel(session))
        self._pages.addWidget(MemoryPanel(session))
        self._pages.addWidget(SkillsPanel(session))
        if session.is_platform_admin:
            self._pages.addWidget(AdminPanel(session))
        self._nav.currentRowChanged.connect(self._pages.setCurrentIndex)
        self._nav.setCurrentRow(0)

        # ── 头部（通栏深藏青）：用户 + 退出 ──
        header_widget = QWidget()
        header_widget.setObjectName("appHeader")
        user_label = QLabel(session.email)
        logout_btn = QPushButton("退出登录")
        logout_btn.clicked.connect(self._on_logout)
        header = QHBoxLayout(header_widget)
        header.setContentsMargins(12, 4, 12, 4)
        header.addStretch(1)
        header.addWidget(user_label)
        header.addWidget(logout_btn)

        # 内容区统一留白
        pages_wrap = QWidget()
        pages_layout = QVBoxLayout(pages_wrap)
        pages_layout.setContentsMargins(12, 10, 12, 12)
        pages_layout.addWidget(self._pages)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._nav)
        body_layout.addWidget(pages_wrap, 1)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(header_widget)
        root.addWidget(body, 1)
        self.setCentralWidget(central)

        config.restore_window_state(self)
        session.start_auto_refresh(self)

    def _on_logout(self) -> None:
        config.save_window_state(self)
        self._session.logout()
        # 标记后关窗：app 主循环检测到该标记会重新弹出登录对话框
        self.logout_requested = True
        self.close()

    def closeEvent(self, event: object) -> None:  # noqa: N802
        config.save_window_state(self)
        # 逐页收尾 + 等待全部 worker 线程结束，避免进程退出时
        # QThread 仍在运行导致的 fail-fast（0xC0000409）崩溃
        for i in range(self._pages.count()):
            page_shutdown = getattr(self._pages.widget(i), "shutdown", None)
            if callable(page_shutdown):
                page_shutdown()
        shutdown_all()
        super().closeEvent(event)
