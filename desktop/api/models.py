"""桌面端 API 数据模型（字段与服务端响应一一对应）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoginResult:
    access_token: str
    refresh_token: str
    expires_in: int
    user: dict[str, Any]


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ThreadInfo:
    id: str
    title: str
    created_at: float
    updated_at: float


@dataclass
class ThreadDetail(ThreadInfo):
    messages: list[Message] = field(default_factory=list)


@dataclass
class InterruptPayload:
    id: str
    tool: str
    args: dict[str, Any]


@dataclass
class ChatResult:
    """对话结果：status=ok 时 answer 有效；pending_approval 时渲染审批卡片。"""

    status: str  # "ok" | "pending_approval"
    answer: str
    interrupts: list[InterruptPayload] = field(default_factory=list)


@dataclass
class SourceItem:
    source: str
    section: str = ""
    score: float = 0.0


@dataclass
class WikiPageSummary:
    id: str
    slug: str
    title: str
    created_at: float
    updated_at: float


@dataclass
class WikiPage(WikiPageSummary):
    content: str = ""
    author_id: str = ""
    summary: str = ""


@dataclass
class WikiRevision:
    id: str
    author_id: str
    summary: str
    created_at: float


@dataclass
class WikiRevisionDetail(WikiRevision):
    content: str = ""


@dataclass
class WikiSearchHit:
    slug: str
    title: str
    updated_at: float
    snippet: str
    title_match: bool


@dataclass
class WikiAttachment:
    id: str
    filename: str
    size_bytes: int
    content_type: str
    created_at: float


@dataclass
class KnowledgeFile:
    source_id: str
    name: str
    size_bytes: int
    modified_at: float


@dataclass
class UploadResult:
    filename: str
    source_id: str
    size_bytes: int
    overwritten: bool
    index_refreshed: bool
    subdir: str = ""
    document_count: int | None = None
    chunk_count: int | None = None


@dataclass
class SkillInfo:
    name: str
    description: str
    path: str
    files: list[str]
    file_count: int
    updated_at: float
    valid: bool
    error: str = ""


@dataclass
class MemoryProfile:
    """画像记忆清单（/memory 下的虚拟文件条目 + 配额用量）。"""

    items: list[dict[str, Any]]
    total_bytes: int
    item_count: int
    max_items: int
    max_file_bytes: int


@dataclass
class FactItem:
    key: str
    text: str
    category: str
    size_bytes: int
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class FactList:
    items: list[FactItem]
    item_count: int
    max_items: int
