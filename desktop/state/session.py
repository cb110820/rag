"""登录态与 token 生命周期管理。

access_token 仅保存在内存（RagApiClient 内）；
refresh_token 优先存系统 keyring，keyring 不可用时降级 QSettings（Base64 混淆）。
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from PySide6.QtCore import QObject, QTimer

from desktop import config
from desktop.api.client import RagApiClient
from desktop.api.errors import ApiError
from desktop.workers.base import start_worker

logger = logging.getLogger(__name__)

_KEYRING_SERVICE = "RagDesktop"
_KEYRING_KEY = "refresh_token"
_QSETTINGS_KEY = "refresh_token_obf"


def _store_refresh(token: str) -> None:
    try:
        import keyring

        keyring.set_password(_KEYRING_SERVICE, _KEYRING_KEY, token)
        return
    except Exception as exc:
        logger.warning("keyring 不可用，refresh_token 降级存 QSettings: %s", exc)
    config.settings().setValue(
        _QSETTINGS_KEY, base64.b64encode(token.encode()).decode()
    )


def _load_refresh() -> str:
    try:
        import keyring

        token = keyring.get_password(_KEYRING_SERVICE, _KEYRING_KEY)
        if token:
            return str(token)
    except Exception as exc:
        logger.warning("keyring 读取失败，尝试 QSettings 降级: %s", exc)
    obf = str(config.settings().value(_QSETTINGS_KEY, ""))
    if obf:
        try:
            return base64.b64decode(obf.encode()).decode()
        except Exception:
            return ""
    return ""


def _clear_refresh() -> None:
    try:
        import keyring

        keyring.delete_password(_KEYRING_SERVICE, _KEYRING_KEY)
    except Exception:
        pass
    config.settings().remove(_QSETTINGS_KEY)


class SessionManager:
    """当前登录会话：API 客户端 + 用户信息 + 令牌持久化。"""

    def __init__(self) -> None:
        self.client: RagApiClient | None = None
        self.user: dict[str, Any] = {}
        self.expires_in: int = 0

    @property
    def is_logged_in(self) -> bool:
        return self.client is not None and self.client.has_session

    @property
    def role(self) -> str:
        return str(self.user.get("role", ""))

    @property
    def email(self) -> str:
        return str(self.user.get("email", ""))

    @property
    def is_platform_admin(self) -> bool:
        return self.role == "platform_admin"

    def login(self, base_url: str, email: str, password: str) -> None:
        """登录并建立会话；失败抛 ApiError 由 UI 层展示。"""

        client = RagApiClient(base_url)
        result = client.login(email, password)
        self.client = client
        self.user = result.user
        self.expires_in = result.expires_in
        _store_refresh(result.refresh_token)
        config.set_base_url(base_url)
        config.set_email(email)

    def try_restore(self) -> bool:
        """启动时用已存 refresh_token 换新会话（免登录）；任何失败都回退登录页。"""

        refresh = _load_refresh()
        if not refresh:
            return False
        client = RagApiClient(config.get_base_url())
        try:
            result = client.restore_session(refresh)
        except ApiError as exc:
            logger.info("会话恢复失败（回退登录页）: %s", exc)
            _clear_refresh()
            return False
        self.client = client
        self.user = result.user
        _store_refresh(client_refresh(self))  # 轮换后的新 refresh 重新落盘
        return True

    def logout(self) -> None:
        if self.client is not None:
            self.client.logout()
        self.client = None
        self.user = {}
        _clear_refresh()

    def start_auto_refresh(self, parent: QObject) -> None:
        """登录后启动主动刷新定时器：到期前 60s 换新令牌（401 被动刷新兜底）。

        轮换成功的新 refresh_token 会重新落盘；刷新失败仅记日志，
        由后续请求的 401 被动刷新或重新登录兜底。
        """

        if not self.is_logged_in:
            return
        interval_s = max((self.expires_in or 1800) - 60, 300)
        timer = QTimer(parent)
        timer.setInterval(interval_s * 1000)

        def _tick() -> None:
            client = self.client
            if client is None or not client.has_session:
                timer.stop()
                return
            start_worker(client.refresh_now, self._on_refreshed, self._on_refresh_failed)

        timer.timeout.connect(_tick)
        timer.start()
        self._refresh_timer = timer

    def _on_refreshed(self, _result: object) -> None:
        _store_refresh(client_refresh(self))

    def _on_refresh_failed(self, detail: str, _status: int) -> None:
        logger.warning("主动刷新 token 失败: %s", detail)


def client_refresh(session: SessionManager) -> str:
    """取客户端当前持有的 refresh_token（轮换刷新后可能已更新）。"""

    assert session.client is not None
    return session.client._refresh_token  # noqa: SLF001 — 同模块受控访问
