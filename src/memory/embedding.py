"""语义记忆的嵌入适配器。

把项目的 :class:`~src.rag.embeddings.EmbeddingProvider`（bge-m3 / hash）
适配为 LangGraph Store 索引要求的嵌入函数签名
``Callable[[Sequence[str]], list[list[float]]]``。

设计要点：

- **懒加载**：Store 在应用启动时就需要 ``index`` 配置（含维度），但 bge-m3
  权重很大，不能因此让每次启动都加载模型。Provider 在首次真正嵌入时才构造，
  维度按配置静态声明（bge-m3=1024，hash=768）；
- **同步函数**：两个 Provider 均为同步 CPU 实现，``AsyncPostgresStore`` /
  ``AsyncSqliteStore`` 内部会把同步嵌入函数放到线程中执行，不阻塞事件循环；
- 维度与模型的对应关系集中在 :class:`Settings.memory_vector_dims`，
  切换嵌入模型会改变维度，需使用全新的 namespace/库（与 RAG 缓存同理）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache

from src.rag.config import get_settings

logger = logging.getLogger("src.memory")


@lru_cache(maxsize=2)
def _get_provider(kind: str):
    """按类型懒加载嵌入 Provider（bge-m3 首次加载会下载/读取权重）。"""

    if kind == "bge-m3":
        from src.rag.embeddings import BGEM3EmbeddingProvider

        settings = get_settings()
        logger.info("语义记忆使用 bge-m3 嵌入: %s", settings.embedding_model)
        return BGEM3EmbeddingProvider(settings.embedding_model)

    from src.rag.embeddings import HashEmbeddingProvider

    logger.info("语义记忆使用 hash 嵌入（离线兜底，召回质量有限）")
    return HashEmbeddingProvider(dimensions=get_settings().memory_vector_dims)


def memory_embed(texts: Sequence[str]) -> list[list[float]]:
    """Store 索引使用的嵌入函数（同步）。"""

    if not texts:
        return []
    settings = get_settings()
    kind = settings.memory_embedding_kind
    started = logging.getLogger("src.memory")
    started.debug("记忆向量嵌入: kind=%s, 条数=%d", kind, len(texts))
    provider = _get_provider(kind)
    vectors = provider.embed(list(texts))
    return vectors.tolist()
