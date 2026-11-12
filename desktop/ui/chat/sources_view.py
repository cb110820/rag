"""RAG 检索来源 chips（横向滚动条式展示，样式由全局主题渲染）。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QWidget


class SourcesView(QScrollArea):
    """展示一轮对话的检索来源：source（+section）小胶囊，横向可滚。"""

    def __init__(self, sources: list[dict[str, Any]], parent: object = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setFixedHeight(36)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)
        label = QLabel("来源：")
        label.setProperty("muted", True)
        layout.addWidget(label)
        for src in sources:
            name = str(src.get("source", ""))
            section = str(src.get("section", "") or "")
            text = f"{name} · {section}" if section else name
            chip = QLabel(text)
            chip.setProperty("sourceChip", True)
            chip.setToolTip(f"相关度 {src.get('score', 0):.3f}")
            layout.addWidget(chip)
        layout.addStretch(1)
        self.setWidget(container)
