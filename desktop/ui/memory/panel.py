"""记忆面板：画像（/memory 虚拟文件）与事实记忆（save_memory 条目）双 tab。"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from desktop.api.models import FactItem, FactList, MemoryProfile
from desktop.state.session import SessionManager
from desktop.ui.widgets.toast import show_toast
from desktop.workers.base import start_worker


def _fmt_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


class _ContentDialog(QDialog):
    def __init__(self, title: str, content: str, parent: object = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 420)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(content)
        layout = QVBoxLayout(self)
        layout.addWidget(view)


class MemoryPanel(QWidget):
    def __init__(self, session: SessionManager, parent: object = None) -> None:
        super().__init__(parent)
        self._session = session

        clear_btn = QPushButton("清空全部记忆")
        clear_btn.setProperty("danger", True)
        clear_btn.clicked.connect(self._on_clear_all)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh)
        top = QHBoxLayout()
        top.addWidget(refresh_btn)
        top.addStretch(1)
        top.addWidget(clear_btn)

        self._tabs = QTabWidget()
        self._profile_tab = self._build_profile_tab()
        self._facts_tab = self._build_facts_tab()
        self._tabs.addTab(self._profile_tab, "画像记忆")
        self._tabs.addTab(self._facts_tab, "事实记忆")

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._tabs, 1)

        self.refresh()

    # ── 画像 tab ───────────────────────────────────────────────────────

    def _build_profile_tab(self) -> QWidget:
        tab = QWidget()
        self._profile_quota = QProgressBar()
        self._profile_quota.setFormat("条目 %v/%m · 占比 %p%")
        self._profile_bytes = QLabel("")
        self._profile_table = QTableWidget(0, 3)
        self._profile_table.setHorizontalHeaderLabels(["key", "大小", "更新时间"])
        self._profile_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._profile_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        view_btn = QPushButton("查看内容")
        view_btn.clicked.connect(self._on_view_profile)
        del_btn = QPushButton("删除选中")
        del_btn.clicked.connect(self._on_delete_profile)
        buttons = QHBoxLayout()
        buttons.addWidget(view_btn)
        buttons.addWidget(del_btn)
        buttons.addStretch(1)

        layout = QVBoxLayout(tab)
        quota_row = QHBoxLayout()
        quota_row.addWidget(self._profile_quota, 1)
        quota_row.addWidget(self._profile_bytes)
        layout.addLayout(quota_row)
        layout.addWidget(self._profile_table, 1)
        layout.addLayout(buttons)
        return tab

    def _on_profile_loaded(self, profile: MemoryProfile) -> None:
        self._profile = profile
        self._profile_quota.setMaximum(max(profile.max_items, 1))
        self._profile_quota.setValue(profile.item_count)
        self._profile_bytes.setText(f"总大小 {_fmt_bytes(profile.total_bytes)}")
        self._profile_table.setRowCount(len(profile.items))
        for row, item in enumerate(profile.items):
            self._profile_table.setItem(row, 0, QTableWidgetItem(str(item.get("key", ""))))
            self._profile_table.setItem(
                row, 1, QTableWidgetItem(_fmt_bytes(int(item.get("size_bytes", 0))))
            )
            self._profile_table.setItem(
                row, 2, QTableWidgetItem(str(item.get("updated_at") or "—"))
            )

    def _selected_profile_key(self) -> str | None:
        row = self._profile_table.currentRow()
        items = getattr(self, "_profile", None)
        if items is not None and 0 <= row < len(items.items):
            return str(items.items[row].get("key", ""))
        return None

    def _on_view_profile(self) -> None:
        key = self._selected_profile_key()
        if not key:
            show_toast(self, "请先选择条目")
            return
        start_worker(
            lambda: self._require_client().get_profile_content(key),
            lambda content: _ContentDialog(f"画像记忆 · {key}", content, self).exec(),
            lambda detail, _s: show_toast(self, f"读取失败：{detail}", error=True),
        )

    def _on_delete_profile(self) -> None:
        key = self._selected_profile_key()
        if not key:
            show_toast(self, "请先选择条目")
            return
        confirm = QMessageBox.question(self, "删除画像记忆", f"确定删除 {key}？")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        start_worker(
            lambda: self._require_client().delete_profile_item(key),
            lambda _r: (show_toast(self, "已删除"), self.refresh()),
            lambda detail, _s: show_toast(self, f"删除失败：{detail}", error=True),
        )

    # ── 事实 tab ───────────────────────────────────────────────────────

    def _build_facts_tab(self) -> QWidget:
        tab = QWidget()
        self._facts_quota = QProgressBar()
        self._facts_quota.setFormat("条目 %v/%m · 占比 %p%")
        self._facts_table = QTableWidget(0, 3)
        self._facts_table.setHorizontalHeaderLabels(["内容", "类别", "创建时间"])
        self._facts_table.setColumnWidth(0, 420)
        self._facts_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._facts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        del_btn = QPushButton("删除选中")
        del_btn.clicked.connect(self._on_delete_fact)
        buttons = QHBoxLayout()
        buttons.addWidget(del_btn)
        buttons.addStretch(1)

        layout = QVBoxLayout(tab)
        layout.addWidget(self._facts_quota)
        layout.addWidget(self._facts_table, 1)
        layout.addLayout(buttons)
        return tab

    def _on_facts_loaded(self, facts: FactList) -> None:
        self._facts = facts
        self._facts_quota.setMaximum(max(facts.max_items, 1))
        self._facts_quota.setValue(facts.item_count)
        self._facts_table.setRowCount(len(facts.items))
        for row, fact in enumerate(facts.items):
            self._facts_table.setItem(row, 0, QTableWidgetItem(fact.text))
            self._facts_table.setItem(row, 1, QTableWidgetItem(fact.category))
            self._facts_table.setItem(row, 2, QTableWidgetItem(fact.created_at or "—"))

    def _selected_fact(self) -> FactItem | None:
        row = self._facts_table.currentRow()
        facts = getattr(self, "_facts", None)
        if facts is not None and 0 <= row < len(facts.items):
            return facts.items[row]
        return None

    def _on_delete_fact(self) -> None:
        fact = self._selected_fact()
        if fact is None:
            show_toast(self, "请先选择条目")
            return
        confirm = QMessageBox.question(self, "删除事实记忆", f"确定删除「{fact.text[:40]}」？")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        start_worker(
            lambda: self._require_client().delete_fact(fact.key),
            lambda _r: (show_toast(self, "已删除"), self.refresh()),
            lambda detail, _s: show_toast(self, f"删除失败：{detail}", error=True),
        )

    # ── 清空与刷新 ─────────────────────────────────────────────────────

    def _on_clear_all(self) -> None:
        confirm = QMessageBox.warning(
            self, "清空全部记忆",
            "将删除你的全部画像与事实记忆，且不可恢复。确定继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        start_worker(
            self._require_client().clear_memory,
            lambda result: (
                show_toast(
                    self,
                    f"已清空：画像 {result.get('deleted', 0)} 项，"
                    f"事实 {result.get('facts_deleted', 0)} 条",
                ),
                self.refresh(),
            ),
            lambda detail, _s: show_toast(self, f"清空失败：{detail}", error=True),
        )

    def refresh(self) -> None:
        client = self._require_client()
        start_worker(
            client.list_profile,
            self._on_profile_loaded,
            lambda detail, _s: show_toast(self, f"加载画像失败：{detail}", error=True),
        )
        start_worker(
            client.list_facts,
            self._on_facts_loaded,
            lambda detail, _s: show_toast(self, f"加载事实失败：{detail}", error=True),
        )

    def _require_client(self):
        assert self._session.client is not None, "会话未建立"
        return self._session.client
