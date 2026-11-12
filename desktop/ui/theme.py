"""桌面端全局主题：工程工具风（参照用户提供的诊断工具截图）。

高饱和蓝导航轨 · 深藏青头部 · 白面板 · 紧凑高密度布局 · 小圆角。
在 app 启动时经 ``apply_theme(app)`` 一次性装配（Fusion 基础风格 + 全局字体 + QSS）。
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

# ── 设计令牌 ─────────────────────────────────────────────────────────
PRIMARY = "#2B7CE6"          # 高饱和工程蓝
PRIMARY_HOVER = "#1E6AD0"
PRIMARY_ACTIVE = "#1A5FC4"
PRIMARY_SOFT = "#E8F1FD"     # 浅蓝选中底
PRIMARY_BORDER = "#B9D4F5"

NAVY = "#123A63"             # 深藏青（头部栏）
NAV_BG = "#1E6FD9"           # 导航轨主蓝
NAV_HOVER = "#3E8BEB"
NAV_SELECTED_BG = "#FFFFFF"

DANGER = "#E11D48"
DANGER_SOFT = "#FDECEF"
WARNING = "#D97706"
WARNING_SOFT = "#FFF7E8"
WARNING_BORDER = "#F0DFAE"
WARNING_TEXT = "#8A5A00"

BG = "#EEF3F8"               # 浅蓝灰底
SURFACE = "#FFFFFF"
SURFACE_2 = "#F6F9FC"
SURFACE_3 = "#EAF1F9"
BORDER = "#D5E0EC"
BORDER_STRONG = "#BECDDD"
TEXT_1 = "#1B2A3A"
TEXT_2 = "#4A5D72"
TEXT_3 = "#8A99AC"

FONT_FAMILY = "Microsoft YaHei"
MONO_FAMILY = "Cascadia Code, Consolas, JetBrains Mono, Microsoft YaHei"

QSS = f"""
/* ══ 基础 ═══════════════════════════════════════════════════════ */
QWidget {{
  background: {BG};
  color: {TEXT_1};
  font-size: 13px;
  selection-background-color: {PRIMARY_BORDER};
  selection-color: {TEXT_1};
}}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{
  background: {NAVY}; color: #fff; border: none;
  padding: 5px 8px; border-radius: 3px;
}}

/* ══ 按钮 ═══════════════════════════════════════════════════════ */
QPushButton {{
  background: {SURFACE};
  border: 1px solid {BORDER_STRONG};
  border-radius: 3px;
  padding: 5px 14px;
  color: {TEXT_1};
}}
QPushButton:hover {{ border-color: {PRIMARY}; color: {PRIMARY}; }}
QPushButton:pressed {{ background: {PRIMARY_SOFT}; }}
QPushButton:disabled {{ color: {TEXT_3}; background: {SURFACE_3}; border-color: {BORDER}; }}
QPushButton[primary="true"] {{
  background: {PRIMARY}; border: 1px solid {PRIMARY}; color: #fff;
}}
QPushButton[primary="true"]:hover {{
  background: {PRIMARY_HOVER}; border-color: {PRIMARY_HOVER}; color: #fff;
}}
QPushButton[primary="true"]:pressed {{ background: {PRIMARY_ACTIVE}; }}
QPushButton[danger="true"] {{
  color: {DANGER}; border-color: #F0B7C3;
}}
QPushButton[danger="true"]:hover {{
  background: {DANGER_SOFT}; border-color: {DANGER}; color: {DANGER};
}}

/* ══ 输入控件 ═══════════════════════════════════════════════════ */
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox {{
  background: {SURFACE};
  border: 1px solid {BORDER_STRONG};
  border-radius: 3px;
  padding: 5px 8px;
  selection-background-color: {PRIMARY_BORDER};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {{
  border: 1px solid {PRIMARY};
}}
QLineEdit:disabled, QPlainTextEdit:disabled {{ background: {SURFACE_3}; color: {TEXT_3}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
  background: {SURFACE}; border: 1px solid {BORDER_STRONG};
  selection-background-color: {PRIMARY_SOFT}; selection-color: {TEXT_1};
  outline: none;
}}

/* ══ 列表 / 表格（高密度数据网格） ═══════════════════════════════ */
QListWidget, QTableWidget, QTreeWidget {{
  background: {SURFACE};
  border: 1px solid {BORDER_STRONG};
  border-radius: 0;
  outline: none;
  gridline-color: {BORDER};
}}
QListWidget::item {{ padding: 5px 8px; border: none; }}
QListWidget::item:selected {{ background: {PRIMARY_SOFT}; color: {PRIMARY_ACTIVE}; }}
QListWidget::item:hover {{ background: {SURFACE_3}; }}
QTableWidget::item {{ padding: 4px 6px; border: none; }}
QTableWidget::item:selected {{ background: {PRIMARY_SOFT}; color: {TEXT_1}; }}
QHeaderView::section {{
  background: {SURFACE_3}; color: {TEXT_2};
  border: none; border-right: 1px solid {BORDER}; border-bottom: 1px solid {BORDER_STRONG};
  padding: 6px 8px; font-weight: 500;
}}

/* ══ 左侧导航轨（高饱和蓝） ══════════════════════════════════════ */
QListWidget#navList {{
  background: {NAV_BG};
  border: none;
  border-radius: 0;
  padding: 8px 0;
  font-size: 14px;
}}
QListWidget#navList::item {{
  padding: 11px 18px;
  margin: 0;
  border-radius: 0;
  color: rgba(255, 255, 255, 0.85);
}}
QListWidget#navList::item:hover {{ background: {NAV_HOVER}; color: #fff; }}
QListWidget#navList::item:selected {{
  background: {NAV_SELECTED_BG}; color: {PRIMARY_ACTIVE}; font-weight: 600;
}}

/* ══ Tab ════════════════════════════════════════════════════════ */
QTabWidget::pane {{
  border: 1px solid {BORDER_STRONG};
  background: {SURFACE}; top: -1px;
}}
QTabBar::tab {{
  background: {SURFACE_3}; color: {TEXT_2};
  padding: 7px 18px; margin-right: 2px;
  border: 1px solid {BORDER}; border-bottom: none;
}}
QTabBar::tab:hover {{ color: {PRIMARY_ACTIVE}; }}
QTabBar::tab:selected {{
  background: {SURFACE}; color: {PRIMARY_ACTIVE}; font-weight: 600;
  border-top: 2px solid {PRIMARY};
}}

/* ══ 进度条 ═════════════════════════════════════════════════════ */
QProgressBar {{
  background: {SURFACE_3}; border: 1px solid {BORDER}; border-radius: 3px;
  height: 10px; text-align: center; color: {TEXT_2}; font-size: 11px;
}}
QProgressBar::chunk {{ background: {PRIMARY}; }}

/* ══ 滚动条 ═════════════════════════════════════════════════════ */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 1px; }}
QScrollBar::handle:vertical {{
  background: #C9D6E4; border-radius: 3px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #A7BCD2; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 1px; }}
QScrollBar::handle:horizontal {{
  background: #C9D6E4; border-radius: 3px; min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: #A7BCD2; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ══ 工具栏 / 分割条 ════════════════════════════════════════════ */
QToolBar {{
  background: {SURFACE}; border: none;
  border-bottom: 1px solid {BORDER}; spacing: 4px; padding: 3px 6px;
}}
QToolButton {{
  background: transparent; border: 1px solid transparent; border-radius: 3px;
  padding: 4px 10px; color: {TEXT_2};
}}
QToolButton:hover {{
  background: {SURFACE_3}; border-color: {BORDER_STRONG}; color: {TEXT_1};
}}
QToolButton:checked {{
  background: {PRIMARY_SOFT}; color: {PRIMARY_ACTIVE}; border-color: {PRIMARY_BORDER};
}}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background: {PRIMARY}; }}

/* ══ 聊天气泡（按角色动态属性，QTextBrowser 承载） ═══════════════ */
QTextBrowser[bubbleRole="user"] {{
  background: #DCEBFD; border: 1px solid {PRIMARY_BORDER};
  border-radius: 4px; padding: 9px 13px; font-size: 14px;
}}
QTextBrowser[bubbleRole="assistant"] {{
  background: {SURFACE}; border: 1px solid {BORDER_STRONG};
  border-radius: 4px; padding: 9px 13px; font-size: 14px;
}}
QTextBrowser[bubbleRole="notice"] {{
  background: {WARNING_SOFT}; color: {WARNING_TEXT};
  border: 1px solid {WARNING_BORDER};
  border-radius: 4px; padding: 8px 13px; font-size: 13px;
}}

/* ══ 审批卡片 ═══════════════════════════════════════════════════ */
QFrame#approvalCard {{
  background: {WARNING_SOFT};
  border: 1px solid {WARNING_BORDER};
  border-left: 3px solid {WARNING};
  border-radius: 4px;
}}
QFrame#approvalCardResolved {{
  background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 4px;
}}
QFrame#approvalCard QLabel {{ background: transparent; border: none; }}
QFrame#approvalCard QToolButton {{ color: {TEXT_3}; font-size: 12px; }}
QFrame#approvalCard QPlainTextEdit {{ background: {SURFACE}; }}
QFrame#approvalCardResolved QLabel {{
  background: transparent; border: none; color: {TEXT_3};
}}
QLabel[cardTitle="true"] {{ color: {WARNING_TEXT}; font-weight: 600; }}

/* ══ 来源 chips ═════════════════════════════════════════════════ */
QLabel[sourceChip="true"] {{
  background: {PRIMARY_SOFT}; color: {PRIMARY_ACTIVE};
  border: 1px solid {PRIMARY_BORDER};
  border-radius: 3px; padding: 3px 10px; font-size: 12px;
}}

/* ══ 辅助文本 ═══════════════════════════════════════════════════ */
QLabel[muted="true"] {{ color: {TEXT_3}; font-size: 12px; background: transparent; }}
QLabel[hint="true"] {{ color: {TEXT_3}; font-size: 12px; background: transparent; }}

/* ══ 头部栏（深藏青） ════════════════════════════════════════════ */
QWidget#appHeader {{
  background: {NAVY}; border-bottom: 1px solid {NAVY};
}}
QWidget#appHeader QLabel {{ background: transparent; color: rgba(255, 255, 255, 0.75); }}
QWidget#appHeader QPushButton {{
  background: transparent; border: 1px solid rgba(255, 255, 255, 0.35);
  color: rgba(255, 255, 255, 0.85); border-radius: 3px;
}}
QWidget#appHeader QPushButton:hover {{
  background: rgba(255, 255, 255, 0.12); color: #fff; border-color: rgba(255, 255, 255, 0.6);
}}

/* ══ 菜单 ═══════════════════════════════════════════════════════ */
QMenu {{
  background: {SURFACE}; border: 1px solid {BORDER_STRONG};
  padding: 3px;
}}
QMenu::item {{ padding: 6px 22px; }}
QMenu::item:selected {{ background: {PRIMARY_SOFT}; color: {PRIMARY_ACTIVE}; }}

/* ══ 消息框 / 输入框按钮 ═════════════════════════════════════════ */
QMessageBox QPushButton, QInputDialog QPushButton {{ min-width: 72px; }}

/* ══ 登录对话框 ═════════════════════════════════════════════════ */
QDialog#loginDialog {{
  background: {SURFACE}; border: 1px solid {BORDER_STRONG}; border-radius: 6px;
}}
QDialog#loginDialog QLabel {{ background: transparent; }}
QLabel#loginTitle {{ color: {NAVY}; font-size: 19px; font-weight: 600; }}
QLabel#loginSubtitle {{ color: {TEXT_3}; font-size: 12px; padding-bottom: 6px; }}
"""


def apply_theme(app: QApplication) -> None:
    """装配全局主题：Fusion 基础风格 + 全局字体 + 全局 QSS。"""

    app.setStyle("Fusion")
    app.setFont(QFont(FONT_FAMILY, 9))
    app.setStyleSheet(QSS)


def restyle(widget) -> None:
    """动态属性变更后刷新样式（unpolish → polish）。"""

    widget.style().unpolish(widget)
    widget.style().polish(widget)
