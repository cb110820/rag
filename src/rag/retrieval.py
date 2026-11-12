"""混合检索：Dense（向量余弦/FAISS）+ Sparse（BM25）。

两路召回各自排序后用 Reciprocal Rank Fusion（RRF）融合分数，
再取 top_k 作为最终候选。

冲突解决：同一文档族（如 policies_v1 / policies_v2）存在多个版本时，
只保留最新版本的片段；时效性（生效/失效日期）会对分数做加权。
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.embeddings import EmbeddingProvider
from src.rag.models import Chunk, SearchResult

logger = logging.getLogger(__name__)

# 文件名中剥离版本后缀，得到文档族名：policies_v2.md → policies
_FAMILY_RE = re.compile(r"^(.+?)[_-]v?\d+(?:\.\d+)*$", re.IGNORECASE)


def _document_family(source_id: str) -> str:
    """从 source_id 提取文档族名（去掉版本后缀与扩展名）。"""

    stem = Path(source_id).stem
    m = _FAMILY_RE.match(stem)
    return m.group(1) if m else stem


def _parse_version(version: str | None) -> tuple[int, ...]:
    """把版本号字符串解析为整数元组，便于比较；无版本返回空元组。"""

    if not version:
        return ()
    parts = re.findall(r"\d+", version)
    return tuple(int(p) for p in parts) if parts else ()


def _is_effective(effective: str | None, expiry: str | None, today: date) -> str:
    """判断时效性状态：'current' / 'future' / 'expired' / 'unknown'。"""

    today_iso = today.isoformat()
    if effective and expiry:
        if effective <= today_iso <= expiry:
            return "current"
        if today_iso < effective:
            return "future"
        return "expired"
    if effective and today_iso < effective:
        return "future"
    if expiry and today_iso > expiry:
        return "expired"
    return "unknown"


def _recency_factor(status: str) -> float:
    """时效性加权因子：当前有效略加权，过期/未生效降权。"""

    return {"current": 1.10, "unknown": 1.0, "future": 0.85, "expired": 0.60}.get(
        status, 1.0
    )


def _version_key(chunk: Chunk) -> tuple:
    """版本比较键：(version_tuple, effective_date, doc_mtime, tier)，越大越新。

    末位为层级裁决：Wiki 镜像（沉淀层）在版本/日期完全打平时胜过原始文档，
    避免镜像的 doc_mtime（物化时间）反而把镜像整体淘汰。
    """

    md = chunk.metadata
    return (
        _parse_version(md.get("version")),
        md.get("effective_date") or "",
        md.get("doc_mtime") or 0.0,
        1 if md.get("tier") == "wiki" else 0,
    )


def resolve_conflicts(
    results: list[SearchResult], today: date | None = None
) -> list[SearchResult]:
    """解决文档版本冲突：同一文档族只保留最新版本的片段，并做时效性加权。

    - 按文档族分组（policies_v1 / policies_v2 → 同属 policies 族）；
    - 每组内选出版本键最大的版本，丢弃旧版本的所有片段；
    - 对保留片段按生效/失效日期调整分数（当前有效略加权，过期降权）。
    """

    if not results:
        return []
    today = today or date.today()

    # 按文档族分组
    families: dict[str, list[SearchResult]] = defaultdict(list)
    for item in results:
        families[_document_family(item.chunk.source_id)].append(item)

    kept: list[SearchResult] = []
    for family, items in families.items():
        if len(items) == 1:
            kept.append(items[0])
            continue
        # 选最新版本
        best_key = max(_version_key(it.chunk) for it in items)
        latest = [it for it in items if _version_key(it.chunk) == best_key]
        if len(latest) < len(items):
            dropped = len(items) - len(latest)
            logger.debug(
                "冲突过滤：文档族 %s 丢弃 %d 个旧版本片段，保留最新版本 %d 个",
                family, dropped, len(latest),
            )
        kept.extend(latest)

    # 时效性加权
    boosted: list[SearchResult] = []
    for item in kept:
        md = item.chunk.metadata
        status = _is_effective(md.get("effective_date"), md.get("expiry_date"), today)
        factor = _recency_factor(status)
        if factor != 1.0:
            boosted.append(SearchResult(item.chunk, item.score * factor))
        else:
            boosted.append(item)

    return sorted(boosted, key=lambda it: it.score, reverse=True)


def apply_tier_boost(
    results: list[SearchResult], factor: float = 1.15
) -> list[SearchResult]:
    """来源分层提权（知识治理 · RAG 兜底）：Wiki 镜像片段分数乘以 factor。

    Wiki 是人工沉淀的权威知识，原始文档是兜底参考：在相关性接近时让
    Wiki 优先出现在候选前列。仅重排序，不丢片段；调用方需在提权后
    自行截断 top_k。输入需已按分数降序，输出保持降序。
    """

    if factor <= 1.0 or not results:
        return results
    boosted = [
        SearchResult(it.chunk, it.score * factor)
        if it.chunk.metadata.get("tier") == "wiki"
        else it
        for it in results
    ]
    logger.debug(
        "来源分层提权：factor=%.2f，wiki 片段 %d / 共 %d",
        factor,
        sum(1 for it in results if it.chunk.metadata.get("tier") == "wiki"),
        len(results),
    )
    return sorted(boosted, key=lambda it: it.score, reverse=True)


def tokenize(text: str) -> list[str]:
    """中英文分词：英文按单词，中文按单字切分。"""

    latin = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return latin + chinese


class HybridRetriever:
    """混合召回器。

    内部同时维护：
    - BM25Okapi 实例（稀疏/关键词检索，按文档族分片）
    - 所有 chunk 的 dense 向量矩阵 + FAISS 索引（稠密/语义检索）

    向量索引策略：
    - 小规模（< HNSW_THRESHOLD）：IndexFlatIP 暴力精确搜索，零精度损失；
    - 大规模（>= HNSW_THRESHOLD）：IndexHNSWFlat 近似搜索，查询复杂度 O(log N)，
      支撑百万级 chunk 的毫秒级检索。
    """

    # 低于该块数用精确索引，高于则启用 HNSW 近似索引
    HNSW_THRESHOLD = 5000
    # HNSW 图参数：M 越大精度越高但内存越大；efConstruction 影响建索引精度与速度
    HNSW_M = 32
    HNSW_EF_CONSTRUCTION = 200
    # 查询时的搜索深度：越大越精确但越慢
    HNSW_EF_SEARCH = 128

    # BM25 分片大小：每个分片独立构建 BM25Okapi，避免百万级单索引内存爆炸
    BM25_SHARD_SIZE = 10_000

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: EmbeddingProvider,
        corpus_vectors: np.ndarray | None = None,
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder
        # BM25 分片：每片 BM25_SHARD_SIZE 个块，查询时 RRF 合并各分片结果
        self._bm25_shards: list[tuple[BM25Okapi, list[int]]] = []
        if chunks:
            tokenized = [tokenize(chunk.text) for chunk in chunks]
            for start in range(0, len(chunks), self.BM25_SHARD_SIZE):
                end = min(start + self.BM25_SHARD_SIZE, len(chunks))
                shard_tokens = tokenized[start:end]
                shard_indices = list(range(start, end))
                self._bm25_shards.append((BM25Okapi(shard_tokens), shard_indices))
            logger.debug(
                "BM25 稀疏索引构建完成：%d 个块 → %d 个分片（每片≤%d）",
                len(chunks), len(self._bm25_shards), self.BM25_SHARD_SIZE,
            )
        if corpus_vectors is not None and len(chunks):
            # 由上层（增量嵌入缓存）预计算的块向量，避免重建索引时重复嵌入
            self._vectors = np.asarray(corpus_vectors, dtype=np.float32)
        else:
            self._vectors = (
                embedder.embed([chunk.text for chunk in chunks]) if chunks else np.empty((0, 0))
            )
        self._faiss_index = None
        self._use_hnsw = False
        if chunks:
            try:
                import faiss

                dim = self._vectors.shape[1]
                if len(chunks) >= self.HNSW_THRESHOLD:
                    # 大规模：HNSW 近似索引，查询 O(log N)
                    self._faiss_index = faiss.IndexHNSWFlat(
                        dim, self.HNSW_M, faiss.METRIC_INNER_PRODUCT
                    )
                    self._faiss_index.hnsw.efConstruction = self.HNSW_EF_CONSTRUCTION
                    self._faiss_index.hnsw.efSearch = self.HNSW_EF_SEARCH
                    self._use_hnsw = True
                    logger.info(
                        "Dense 索引使用 FAISS HNSW（%d 块，M=%d，efConstruction=%d，efSearch=%d）",
                        len(chunks), self.HNSW_M, self.HNSW_EF_CONSTRUCTION, self.HNSW_EF_SEARCH,
                    )
                else:
                    # 小规模：精确暴力搜索，零精度损失
                    self._faiss_index = faiss.IndexFlatIP(dim)
                    logger.debug("Dense 索引使用 FAISS Flat（%d 块，精确搜索）", len(chunks))
                self._faiss_index.add(self._vectors.astype(np.float32))
            except ImportError:
                logger.debug("未安装 faiss，Dense 检索退化为 numpy 矩阵乘法")

    def _dense_ranking(self, query: str, limit: int) -> list[int]:
        """Dense 语义检索：优先走 FAISS，否则退化为 numpy 矩阵乘法。"""

        if not self.chunks:
            return []
        vector = self.embedder.embed([query]).astype(np.float32)
        if self._faiss_index is not None:
            _, indices = self._faiss_index.search(vector, min(limit, len(self.chunks)))
            return [int(index) for index in indices[0] if index >= 0]
        scores = self._vectors @ vector[0]
        return np.argsort(-scores)[:limit].tolist()

    def _sparse_ranking(self, query: str, limit: int) -> list[int]:
        """BM25 关键词检索：查询全部分片，用 RRF 合并结果（避免跨分片分数不可比）。"""

        if not self._bm25_shards:
            return []
        query_tokens = tokenize(query)
        # 每个分片取 limit 个候选，再用 RRF 融合
        per_shard_k = min(limit, self.BM25_SHARD_SIZE)
        fused: dict[int, float] = defaultdict(float)
        for bm25, shard_indices in self._bm25_shards:
            scores = bm25.get_scores(query_tokens)
            ranked = np.argsort(-scores)[:per_shard_k]
            for rank, local_idx in enumerate(ranked, start=1):
                global_idx = shard_indices[int(local_idx)]
                fused[global_idx] += 1.0 / (60 + rank)
        return sorted(fused, key=fused.get, reverse=True)[:limit]

    def search(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 24,
        *,
        dense_query: str | None = None,
        sparse_query: str | None = None,
    ) -> list[SearchResult]:
        """执行混合检索。

        Args:
            query: 用户查询文本（同时用于 dense 和 sparse，除非被覆盖）。
            top_k: 最终返回的结果条数。
            candidate_k: 单路召回时的候选数（通常 > top_k，多融合后更稳）。
            dense_query: Dense 检索专用查询（HyDE 假想答案），None 时用 query。
            sparse_query: Sparse 检索专用查询（扩展关键词），None 时用 query。

        Returns:
            按融合分数降序排列的 SearchResult 列表。
        """

        if not query.strip():
            raise ValueError("query must not be empty")
        started = time.perf_counter()
        # HyDE：Dense 用假想答案的 embedding，语义空间更接近文档原文
        d_query = dense_query or query
        # Query Expansion：Sparse 用扩展后的关键词，提升 BM25 精确匹配
        s_query = sparse_query or query
        dense_hits = self._dense_ranking(d_query, candidate_k)
        sparse_hits = self._sparse_ranking(s_query, candidate_k)
        logger.debug(
            "召回阶段 query=%r：dense %d 条 / sparse %d 条候选",
            query[:60],
            len(dense_hits),
            len(sparse_hits),
        )
        scores: dict[int, float] = defaultdict(float)
        for ranking in (dense_hits, sparse_hits):
            for rank, index in enumerate(ranking, start=1):
                scores[index] += 1.0 / (60 + rank)
        ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
        logger.debug(
            "RRF 融合完成 → %d 个结果，耗时 %.0fms",
            len(ordered),
            (time.perf_counter() - started) * 1000,
        )
        return [SearchResult(self.chunks[index], scores[index]) for index in ordered]