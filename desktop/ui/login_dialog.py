"""登录对话框：品牌标题 + 邮箱/密码 + 可折叠的服务地址（卡片式，样式由全局主题渲染）。"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from desktop import config
from desktop.state.session import SessionManager
from desktop.workers.base import start_worker


class LoginDialog(QDialog):
    """登录成功则 accept()；失败在对话框内联展示错误。"""

    def __init__(self, session: SessionManager, parent: object = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setObjectName("loginDialog")
        self.setWindowTitle("登录 · RAG 知识助手")
        self.setMinimumWidth(400)
        self.setModal(True)

        title = QLabel("RAG 知识助手")
        title.setObjectName("loginTitle")
        subtitle = QLabel("登录以继续使用企业知识库与记忆助手")
        subtitle.setObjectName("loginSubtitle")

        self._email = QLineEdit(config.get_email())
        self._email.setPlaceholderText("邮箱")
        self._password = QLineEdit()
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        self._password.setPlaceholderText("密码")
        self._password.returnPressed.connect(self._on_login)

        self._advanced = QCheckBox("服务地址")
        self._base_url = QLineEdit(config.get_base_url())
        self._base_url.setVisible(False)
        self._advanced.toggled.connect(self._base_url.setVisible)

        self._error = QLabel("")
        self._error.setStyleSheet("color: #e11d48;")
        self._error.setWordWrap(True)
        self._error.setVisible(False)

        self._login_btn = QPushButton("登 录")
        self._login_btn.setProperty("primary", True)
        self._login_btn.setDefault(True)
        self._login_btn.clicked.connect(self._on_login)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)

        form = QFormLayout()
        form.setSpacing(10)
        form.addRow("邮箱", self._email)
        form.addRow("密码", self._password)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(self._login_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(form)
        layout.addWidget(self._advanced)
        layout.addWidget(self._base_url)
        layout.addWidget(self._error)
        layout.addLayout(buttons)

    def _on_login(self) -> None:
        email = self._email.text().strip()
        password = self._password.text()
        base_url = self._base_url.text().strip()
        if not email or not password:
            self._show_error("请输入邮箱和密码")
            return
        if not base_url:
            self._show_error("请输入服务地址")
            return
        self._login_btn.setEnabled(False)
        self._login_btn.setText("登录中…")
        self._error.setVisible(False)

        start_worker(
            lambda: self._session.login(base_url, email, password),
            self._on_login_ok,
            self._on_login_error,
        )

    def _on_login_ok(self, _result: object) -> None:
        self.accept()

    def _on_login_error(self, detail: str, _status: int) -> None:
        self._login_btn.setEnabled(True)
        self._login_btn.setText("登 录")
        self._show_error(detail)

    def _show_error(self, text: str) -> None:
        self._error.setText(text)
        self._error.setVisible(True)
        self.adjustSize()
