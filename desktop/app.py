"""QApplication 组装入口：先尝试免登录恢复会话，失败则弹登录框。"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from desktop.config import APP_NAME
from desktop.state.session import SessionManager
from desktop.ui.login_dialog import LoginDialog
from desktop.ui.main_window import MainWindow
from desktop.ui.theme import apply_theme

logger = logging.getLogger(__name__)


def run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # 全局兜底：任何未捕获异常都进日志，避免桌面端静默退出
    sys.excepthook = lambda exc_type, exc, tb: logger.critical(
        "未捕获异常", exc_info=(exc_type, exc, tb)
    )
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    apply_theme(app)

    session = SessionManager()
    # 登录 ↔ 主窗口主循环：关窗退出进程；退出登录则回到登录对话框
    while True:
        if not session.try_restore():
            dialog = LoginDialog(session)
            accepted = dialog.exec() == LoginDialog.DialogCode.Accepted
            dialog.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            if not accepted:
                return 0

        window = MainWindow(session)
        window.show()
        exit_code = app.exec()
        # 有序析构：先销毁窗口再让 QApplication 退出。
        # 否则 run() 返回后 app/window 由 GC 以不确定顺序回收，
        # Qt 在解释器 teardown 阶段可能 fail-fast（0xC0000409）
        logout = window.logout_requested
        window.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        if not logout:
            return exit_code
        # 退出登录：refresh token 已被 logout 清除，新建空会话回登录页
        session = SessionManager()
