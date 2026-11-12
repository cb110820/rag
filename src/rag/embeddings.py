"""文本嵌入（向量化）。

定义 EmbeddingProvider 协议，并提供两种实现：
- BGEM3EmbeddingProvider：基于 BAAI/bge-m3 模型，生产环境使用
- HashEmbeddingProvider：基于哈希的离线兜底实现，供测试和轻量评估使用
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """嵌入器协议，所有实现必须提供 embed 方法。"""

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbeddingProvider:
    """基于哈希的确定性嵌入兜底实现。

    将每个 token 通过 blake2b 哈希映射到向量维度，统计频次后归一化。
    虽然效果远不如真正的模型，但不依赖外部下载，适合测试和 smoke evaluation。
    """

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        started = time.perf_counter()
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[\w\u4e00-\u9fff]", text.lower())
            for token in tokens:
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                matrix[row, int.from_bytes(digest, "little") % self.dimensions] += 1
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        vectors = matrix / np.maximum(norms, 1e-12)
        logger.debug(
            "hash 嵌入 %d 条文本 → 维度 %d，耗时 %.0fms",
            len(texts),
            self.dimensions,
            (time.perf_counter() - started) * 1000,
        )
        return vectors


class BGEM3EmbeddingProvider:
    """基于 FlagEmbedding 的 BGE-M3 嵌入实现。"""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'rag' extra to use BGE-M3: uv sync --extra rag"
            ) from exc
        logger.info("正在加载 BGE-M3 嵌入模型 %s（首次运行需下载权重）…", model_name)
        started = time.perf_counter()
        self._model = BGEM3FlagModel(model_name, use_fp16=False)
        logger.info("BGE-M3 嵌入模型加载完成，耗时 %.1fs", time.perf_counter() - started)

    def embed(self, texts: list[str]) -> np.ndarray:
        """对文本列表做 dense 嵌入并 L2 归一化，返回 (N, dim) 的 float32 矩阵。"""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        started = time.perf_counter()
        result = self._model.encode(texts, return_dense=True, return_sparse=True)
        vectors = np.asarray(result["dense_vecs"], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.maximum(norms, 1e-12)
        logger.debug(
            "BGE-M3 嵌入 %d 条文本 → 维度 %d，耗时 %.0fms",
            len(texts),
            normalized.shape[1],
            (time.perf_counter() - started) * 1000,
        )
        return normalized