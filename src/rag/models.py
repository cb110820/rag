"""RAG 流水线中通用的数据结构定义。

使用 frozen dataclass 保证不可变性，避免检索过程中意外修改状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """从磁盘加载的原始文档。"""

    source_id: str
    text: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """文档分块后的最小检索单元。"""

    chunk_id: str
    source_id: str
    text: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """一次检索返回的单个结果。"""

    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好的字典，用于传给 LLM 工具返回值。

        父子分块：``content`` 优先返回 ``parent_text``（整个章节上下文），
        让 LLM 获得完整语义；同时保留 ``chunk_text`` 字段供调试/溯源。
        """

        md = self.chunk.metadata
        parent_text = md.get("parent_text")
        content = parent_text if parent_text else self.chunk.text
        return {
            "source": self.chunk.source_id,
            "chunk_id": self.chunk.chunk_id,
            "content": content,
            "chunk_text": self.chunk.text,
            "score": round(float(self.score), 6),
            # 来源层级：wiki = Wiki 沉淀的权威知识，raw = 原始文档兜底
            "tier": md.get("tier", "raw"),
            "metadata": md,
        }