"""顶部滑入式通知条（toast）。"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QWidget

_INFO_STYLE = (
    "background: rgba(26, 43, 43, 0.92); color: #fff; padding: 9px 18px;"
    "border-radius: 8px; font-size: 13px;"
)
_ERROR_STYLE = (
    "background: rgba(225, 29, 72, 0.94); color: #fff; padding: 9px 18px;"
    "border-radius: 8px; font-size: 13px;"
)


def show_toast(parent: QWidget, text: str, *, error: bool = False) -> None:
    """在 parent 顶部居中弹一条自动消失的通知。"""

    toast = QLabel(text, parent)
    toast.setStyleSheet(_ERROR_STYLE if error else _INFO_STYLE)
    toast.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    toast.adjustSize()
    x = max((parent.width() - toast.width()) // 2, 0)
    toast.move(x, 12)
    toast.raise_()
    toast.show()
    QTimer.singleShot(3200, toast.deleteLater)
