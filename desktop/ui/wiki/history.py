"""Wiki 版本历史对话框：版本列表 + 查看正文 + 与当前 diff + 回滚。"""

from __future__ import annotations

import difflib
import html
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from desktop.api.client import RagApiClient
from desktop.api.models import WikiRevision
from desktop.ui.widgets.toast import show_toast
from desktop.workers.base import start_worker


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _diff_html(old: str, new: str, old_name: str, new_name: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=old_name, tofile=new_name, lineterm="",
    )
    parts: list[str] = []
    for line in lines:
        escaped = html.escape(line)
        if line.startswith("+") and not line.startswith("+++"):
            parts.append(f"<div style='background:#e6f4ea;color:#137333'>{escaped}</div>")
        elif line.startswith("-") and not line.startswith("---"):
            parts.append(f"<div style='background:#fdecea;color:#c5221f'>{escaped}</div>")
        elif line.startswith("@@"):
            parts.append(f"<div style='color:#888'>{escaped}</div>")
        else:
            parts.append(f"<div>{escaped}</div>")
    body = "".join(parts) or "<div style='color:#888'>（无差异）</div>"
    return (
        "<html><body style='font-family:Consolas,monospace;font-size:13px;"
        f"white-space:pre'>{body}</body></html>"
    )


class _TextViewDialog(QDialog):
    """只读正文查看弹窗。"""

    def __init__(self, title: str, content: str, parent: object = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 480)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(content)
        layout = QVBoxLayout(self)
        layout.addWidget(view)


class HistoryDialog(QDialog):
    """某页面的版本历史：append-only，回滚即把旧版本正文存为新版本。"""

    def __init__(
        self,
        client: RagApiClient,
        slug: str,
        current_content: str,
        on_reverted,
        parent: object = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._slug = slug
        self._current_content = current_content
        self._on_reverted = on_reverted
        self._revisions: list[WikiRevision] = []
        self.setWindowTitle(f"版本历史 · {slug}")
        self.resize(760, 520)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_selected)

        view_btn = QPushButton("查看正文")
        view_btn.clicked.connect(self._on_view)
        diff_btn = QPushButton("与当前对比")
        diff_btn.clicked.connect(self._on_diff)
        revert_btn = QPushButton("回滚到此版本")
        revert_btn.clicked.connect(self._on_revert)
        buttons = QHBoxLayout()
        buttons.addWidget(view_btn)
        buttons.addWidget(diff_btn)
        buttons.addWidget(revert_btn)
        buttons.addStretch(1)

        hint = QLabel("选择左侧版本后执行操作；回滚不会删除任何历史")
        hint.setProperty("hint", True)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self._list, 1)
        layout.addLayout(buttons)

        start_worker(
            lambda: self._client.wiki_history(self._slug),
            self._on_loaded,
            lambda detail, _s: show_toast(self, f"加载历史失败：{detail}", error=True),
        )

    def _on_loaded(self, revisions: list[WikiRevision]) -> None:
        self._revisions = revisions
        self._list.clear()
        for rev in revisions:
            label = f"{_fmt_time(rev.created_at)}　{rev.summary or '（无摘要）'}"
            QListWidgetItem(label, self._list)

    def _selected(self) -> WikiRevision | None:
        row = self._list.currentRow()
        if 0 <= row < len(self._revisions):
            return self._revisions[row]
        return None

    def _on_selected(self, _row: int) -> None:
        pass

    def _on_view(self) -> None:
        rev = self._selected()
        if rev is None:
            show_toast(self, "请先选择版本")
            return
        start_worker(
            lambda: self._client.get_wiki_revision(self._slug, rev.id),
            lambda detail: _TextViewDialog(
                f"版本 {_fmt_time(rev.created_at)}", detail.content, self
            ).exec(),
            lambda detail, _s: show_toast(self, f"读取失败：{detail}", error=True),
        )

    def _on_diff(self) -> None:
        rev = self._selected()
        if rev is None:
            show_toast(self, "请先选择版本")
            return

        def _show(detail) -> None:
            dialog = QDialog(self)
            dialog.setWindowTitle("差异对比 · 历史版本 → 当前")
            dialog.resize(760, 520)
            browser = QTextBrowser()
            browser.setHtml(
                _diff_html(detail.content, self._current_content,
                           f"历史 {_fmt_time(rev.created_at)}", "当前")
            )
            layout = QVBoxLayout(dialog)
            layout.addWidget(browser)
            dialog.exec()

        start_worker(
            lambda: self._client.get_wiki_revision(self._slug, rev.id),
            _show,
            lambda detail, _s: show_toast(self, f"对比失败：{detail}", error=True),
        )

    def _on_revert(self) -> None:
        rev = self._selected()
        if rev is None:
            show_toast(self, "请先选择版本")
            return
        confirm = QMessageBox.question(
            self, "回滚确认", f"把页面回滚到 {_fmt_time(rev.created_at)} 的版本？"
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        start_worker(
            lambda: self._client.revert_wiki_page(self._slug, rev.id, "回滚到历史版本"),
            lambda _page: (self._on_reverted(), self.accept()),
            lambda detail, _s: show_toast(self, f"回滚失败：{detail}", error=True),
        )
