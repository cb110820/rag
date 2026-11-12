"""文档分块。

策略：结构感知（先按标题切分章节，再在章节内做定长+重叠切分）。

- 标题识别：行首 `#` / `##` / `###` … 视为章节边界，记录层级路径 section_path；
- 章节内分块：沿用固定长度 + 自然边界对齐，保证块语义完整；
- 无标题文档：整体作为单一章节，行为与旧版一致（向后兼容）。
"""

from __future__ import annotations

import logging
import re

from src.rag.models import Chunk, SourceDocument

logger = logging.getLogger(__name__)

# 匹配 Markdown 标题行：1~6 个 # 后接空格与标题文本
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _split_sections(text: str) -> list[tuple[list[str], int, int]]:
    """按标题把文本切分为章节，返回 [(section_path, start, end), ...]。

    section_path 为从根到当前章节的标题文本列表（不含 # 符号）。
    无标题时返回单个章节，section_path 为空列表。
    """

    lines = text.split("\n")
    sections: list[tuple[list[str], int, int]] = []
    # heading_stack: [(level, title_text)]
    heading_stack: list[tuple[int, str]] = []
    current_start = 0
    current_path: list[str] = []
    char_offset = 0

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            # 先结束当前章节
            sections.append((list(current_path), current_start, char_offset))
            level = len(m.group(1))
            title = m.group(2).strip()
            # 弹栈：移除层级 >= 当前的标题
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_path = [t for _, t in heading_stack]
            current_start = char_offset + len(line) + 1  # +1 for the newline
        char_offset += len(line) + 1  # +1 for the newline

    # 最后一个章节
    sections.append((list(current_path), current_start, len(text)))
    return sections


def _chunk_section(
    text: str,
    source_id: str,
    section_path: list[str],
    start: int,
    end: int,
    doc_metadata: dict,
    chunk_size: int,
    overlap: int,
    base_index: int,
) -> list[Chunk]:
    """对单个章节做定长+自然边界分块（父子分块）。

    - ``chunk.text``（子块）：短小精悍，用于 embedding / BM25 / 重排打分；
    - ``metadata['parent_text']``（父块）：整个章节的完整文本，检索命中后
      返回给 LLM 作为上下文，保证语义完整性。
    """

    section_text = text[start:end]
    cleaned = re.sub(r"[ \t]+", " ", section_text).strip()
    if not cleaned:
        return []
    # 父块文本：章节原文（含标题路径前缀，便于 LLM 理解上下文位置）
    parent_text = cleaned
    if section_path:
        parent_text = f"{' / '.join(section_path)}\n{cleaned}"
    chunks: list[Chunk] = []
    pos = 0
    index = base_index
    while pos < len(cleaned):
        tentative_end = min(pos + chunk_size, len(cleaned))
        chunk_end = tentative_end
        if tentative_end < len(cleaned):
            boundary = max(
                cleaned.rfind(mark, pos, tentative_end)
                for mark in ("\n", "。", "！", "？", ". ")
            )
            if boundary > pos + chunk_size // 2:
                chunk_end = boundary + 1
        body = cleaned[pos:chunk_end].strip()
        if body:
            meta = dict(doc_metadata)
            meta["section_path"] = section_path
            meta["parent_text"] = parent_text
            if section_path:
                meta["section_title"] = section_path[-1]
            chunks.append(
                Chunk(
                    chunk_id=f"{source_id}#{index}",
                    source_id=source_id,
                    text=body,
                    start=start + pos,
                    end=start + chunk_end,
                    metadata=meta,
                )
            )
            index += 1
        if chunk_end >= len(cleaned):
            break
        pos = max(pos + 1, chunk_end - overlap)
    return chunks


def chunk_document(
    document: SourceDocument, chunk_size: int = 400, overlap: int = 80
) -> list[Chunk]:
    """将单个文档切分为多个 Chunk（结构感知 + 父子分块）。

    先按 Markdown 标题切分章节，每节独立做定长+重叠分块；
    每个块携带 section_path（标题层级路径）和 parent_text（整个章节上下文）。
    子块（chunk.text）用于检索打分，父块（parent_text）返回给 LLM。
    """

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    text = document.text
    sections = _split_sections(text)
    chunks: list[Chunk] = []
    index = 0
    for section_path, start, end in sections:
        section_chunks = _chunk_section(
            text,
            document.source_id,
            section_path,
            start,
            end,
            document.metadata,
            chunk_size,
            overlap,
            index,
        )
        chunks.extend(section_chunks)
        index += len(section_chunks)
    logger.debug(
        "文档 %s 分块完成: %d 个章节 → %d 个块（chunk_size=%d, overlap=%d, 原文 %d 字符）",
        document.source_id,
        len(sections),
        len(chunks),
        chunk_size,
        overlap,
        len(text),
    )
    return chunks


def chunk_documents(
    documents: list[SourceDocument], chunk_size: int = 400, overlap: int = 80
) -> list[Chunk]:
    """批量分块：对文档列表中每个文档调用 chunk_document 并展平结果。"""

    return [
        chunk for document in documents for chunk in chunk_document(document, chunk_size, overlap)
    ]
