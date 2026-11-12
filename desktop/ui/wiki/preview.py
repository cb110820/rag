"""Markdown 实时预览：markdown 库渲染 HTML → QTextBrowser（150ms 防抖）。"""

from __future__ import annotations

import markdown
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QTextBrowser

_EXTENSIONS = ["fenced_code", "tables", "nl2br", "sane_lists"]

_CSS = """
body { font-family: "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 14px; }
pre { background: #f5f5f5; padding: 8px; border-radius: 6px; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 10px; color: #666; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ddd; padding: 4px 10px; }
"""


def render_markdown(text: str) -> str:
    body = markdown.markdown(text or "", extensions=_EXTENSIONS)
    return f"<html><head><style>{_CSS}</style></head><body>{body}</body></html>"


class MarkdownPreview(QTextBrowser):
    """跟随编辑器内容防抖刷新的预览组件。"""

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._do_render)
        self._pending = ""

    def schedule_render(self, text: str) -> None:
        self._pending = text
        self._timer.start()

    def _do_render(self) -> None:
        self.setHtml(render_markdown(self._pending))
