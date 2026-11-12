"""HITL 审批卡片（移植 web/app.js 简洁行式卡片逻辑）。

- 头部：⚠ 待审批 N 项 + 批准/拒绝按钮
- 每项：工具名 + 一行摘要（path → fact_id → content 截断 → JSON 截断）
- 「完整参数」可展开查看格式化 JSON
- 决策后整卡坍缩为一行结果文本（show_result）
"""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from desktop.api.models import InterruptPayload
from desktop.ui.theme import restyle

TOOL_LABELS = {
    "forget_memory": "删除长期记忆",
    "write_file": "写入文件（长期记忆）",
    "edit_file": "修改文件（长期记忆）",
}


def summarize_args(args: dict[str, Any]) -> str:
    """从参数里提炼一行摘要：优先路径/fact_id，其次 content 截断，兜底紧凑 JSON。"""

    if not isinstance(args, dict):
        return ""
    path = args.get("path")
    if isinstance(path, str):
        return path
    fact_id = args.get("fact_id")
    if isinstance(fact_id, str):
        return f"fact_id: {fact_id}"
    content = args.get("content")
    if isinstance(content, str):
        return content if len(content) <= 60 else content[:60] + "…"
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= 80 else s[:80] + "…"


class ApprovalCard(QFrame):
    """一张审批卡片对应一批 interrupts；decided 发出 "approve" / "reject"。"""

    decided = Signal(str)

    def __init__(self, interrupts: list[InterruptPayload], parent: object = None) -> None:
        super().__init__(parent)
        self.interrupts = interrupts
        self.setObjectName("approvalCard")

        head = QHBoxLayout()
        title = QLabel(f"⚠ 待审批 <b>{len(interrupts)}</b> 项")
        title.setProperty("cardTitle", True)
        title.setTextFormat(Qt.TextFormat.RichText)
        head.addWidget(title)
        head.addStretch(1)
        self._approve_btn = QPushButton("批准")
        self._reject_btn = QPushButton("拒绝")
        self._approve_btn.setProperty("primary", True)
        self._approve_btn.clicked.connect(lambda: self.decided.emit("approve"))
        self._reject_btn.clicked.connect(lambda: self.decided.emit("reject"))
        head.addWidget(self._reject_btn)
        head.addWidget(self._approve_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.addLayout(head)
        for it in interrupts:
            layout.addWidget(self._build_item(it))

    def _build_item(self, it: InterruptPayload) -> QWidget:
        item = QWidget()
        item_layout = QVBoxLayout(item)
        item_layout.setContentsMargins(0, 2, 0, 2)
        item_layout.setSpacing(2)

        label = TOOL_LABELS.get(it.tool, it.tool)
        summary = summarize_args(it.args)
        summary_html = f"　<span style='color:#8fa3a1'>{summary}</span>" if summary else ""
        line = QLabel(f"<b>{label}</b>{summary_html}")
        line.setTextFormat(Qt.TextFormat.RichText)
        item_layout.addWidget(line)

        toggle = QToolButton()
        toggle.setText("完整参数")
        toggle.setCheckable(True)
        detail = QPlainTextEdit(json.dumps(it.args, ensure_ascii=False, indent=2))
        detail.setReadOnly(True)
        detail.setMaximumHeight(120)
        detail.setVisible(False)
        toggle.toggled.connect(detail.setVisible)
        item_layout.addWidget(toggle)
        item_layout.addWidget(detail)
        return item

    def set_busy(self) -> None:
        """已提交决策，等待 /chat/resume 返回。"""

        self._approve_btn.setEnabled(False)
        self._reject_btn.setEnabled(False)

    def show_result(self, text: str) -> None:
        """决策后整卡坍缩为一行结果（与 Web 端 resolved 行为一致）。"""

        layout = self.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                _delete_layout(child.layout())
        result = QLabel(text)
        layout.addWidget(result)
        # 切换为已决议外观：灰白结果卡
        self.setObjectName("approvalCardResolved")
        restyle(self)


def _delete_layout(layout: object) -> None:
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
            child.widget().deleteLater()
        elif child.layout() is not None:
            _delete_layout(child.layout())
