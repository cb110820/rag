"""重排（Reranking）。

在混合召回给出的候选上做更精细的相关性排序。
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from src.rag.models import SearchResult

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """重排器协议。"""

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...


class IdentityReranker:
    """直通重排器：不做任何重排，仅截断到 top_k。"""

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        return results[:top_k]


class BGEReranker:
    """基于 BAAI/bge-reranker-v2-m3 交叉编码器的精细重排。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'rag' extra to use BGE reranking: uv sync --extra rag"
            ) from exc
        logger.info("正在加载 BGE 重排模型 %s（首次运行需下载权重）…", model_name)
        started = time.perf_counter()
        self._model = FlagReranker(model_name, use_fp16=False)
        logger.info("BGE 重排模型加载完成，耗时 %.1fs", time.perf_counter() - started)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """用交叉编码器对 (query, chunk) 对打分，按分数降序截取 top_k。"""

        if not results:
            return []
        started = time.perf_counter()
        scores = self._model.compute_score(
            [[query, item.chunk.text] for item in results], normalize=True
        )
        reranked = [
            SearchResult(item.chunk, float(score))
            for item, score in zip(results, scores, strict=True)
        ]
        final = sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
        logger.debug(
            "BGE 重排 %d → %d 个结果，耗时 %.0fms",
            len(results),
            len(final),
            (time.perf_counter() - started) * 1000,
        )
        return final