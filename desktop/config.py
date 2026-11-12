"""桌面端本地配置：QSettings 持久化 base_url、登录邮箱、窗口几何。"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget

APP_NAME = "RagDesktop"
_ORG = "RagDesktop"
_DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def settings() -> QSettings:
    return QSettings(_ORG, APP_NAME)


def get_base_url() -> str:
    return str(settings().value("base_url", _DEFAULT_BASE_URL))


def set_base_url(url: str) -> None:
    settings().setValue("base_url", url.strip().rstrip("/"))


def get_email() -> str:
    return str(settings().value("email", ""))


def set_email(email: str) -> None:
    settings().setValue("email", email.strip())


def save_window_state(widget: QWidget) -> None:
    s = settings()
    s.setValue("window/geometry", widget.saveGeometry())
    s.setValue("window/state", widget.windowState())


def restore_window_state(widget: QWidget) -> None:
    s = settings()
    geometry = s.value("window/geometry")
    if geometry is not None:
        widget.restoreGeometry(geometry)
