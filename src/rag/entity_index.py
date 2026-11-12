"""实体索引（仿 WeKnora ENTITY_SEARCH）。

利用 LLM 沉淀生成的 ``entity-*`` Wiki 镜像页（chunk metadata 携带
frontmatter 的 ``kind``/``title``/``aliases``）建立 名称→分块 映射；
查询文本命中实体名或别名时，把对应实体页补进检索候选，参与重排与
Wiki 层级提权（镜像天然 tier="wiki"）。

模块边界红线：本模块 **禁止 import src.curation**。``normalize_entity_name``
与 curation/markdown.py 同名函数逐字一致（测试双 import 断言等价），
修改任一侧必须同步另一侧。
"""

from __future__ import annotations

import logging
import re

from src.rag.models import Chunk

logger = logging.getLogger(__name__)

# 与 src/curation/markdown.py 的 normalize_entity_name 保持逐字一致
_ENTITY_NORM_RE = re.compile(r"[\s\W_]+", re.UNICODE)

# 参与匹配的最短归一化键长（避免单字符中文/数字噪声命中）
_MIN_KEY_LEN = 2


def normalize_entity_name(name: str) -> str:
    """归一化实体名用于"同一实体"匹配：小写、去空白与标点，保留中日韩字符。"""

    return _ENTITY_NORM_RE.sub("", name.lower())


class EntityIndex:
    """名称/别名 → 实体分块 的归一化映射索引。"""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._by_key: dict[str, Chunk] = {}
        self._keys_by_chunk: dict[str, list[str]] = {}
        self._build(chunks)

    @classmethod
    def build(cls, chunks: list[Chunk]) -> EntityIndex:
        return cls(chunks)

    def _build(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            md = chunk.metadata
            if md.get("kind") != "entity":
                continue
            aliases = md.get("aliases") or []
            names = [md.get("title") or "", *aliases]
            keys = [
                normalize_entity_name(n)
                for n in names
                if n and len(normalize_entity_name(n)) >= _MIN_KEY_LEN
            ]
            for key in dict.fromkeys(keys):  # 去重且保持顺序
                # 同名实体后写覆盖：镜像按 slug 唯一，正常不冲突
                self._by_key[key] = chunk
            self._keys_by_chunk[chunk.chunk_id] = list(dict.fromkeys(keys))
        if self._by_key:
            logger.info("实体索引构建完成：%d 个可匹配名称", len(self._by_key))

    def lookup(self, query: str) -> list[Chunk]:
        """查询文本命中实体名/别名时返回对应实体分块（长键优先、去重）。"""

        query_norm = normalize_entity_name(query)
        if not query_norm or not self._by_key:
            return []
        hits: list[Chunk] = []
        seen: set[str] = set()
        # 长键优先：更具体的实体名优先命中
        for key in sorted(self._by_key, key=len, reverse=True):
            if len(key) > len(query_norm):
                continue
            if key in query_norm:
                chunk = self._by_key[key]
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    hits.append(chunk)
        return hits
