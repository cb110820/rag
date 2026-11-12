"""RAG 流水线编排。

KnowledgeBase 是 RAG 子系统的统一入口，内部串联：
load_documents → chunk_documents → HybridRetriever（初始化嵌入） → Reranker
对外只需调用 kb.search(query, top_k)。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.rag.chunking import chunk_document
from src.rag.config import get_settings
from src.rag.embedding_cache import CachedCorpusEmbedder
from src.rag.embeddings import BGEM3EmbeddingProvider, EmbeddingProvider, HashEmbeddingProvider
from src.rag.entity_index import EntityIndex
from src.rag.models import SearchResult
from src.rag.parser import load_document, scan_document_stats
from src.rag.reranker import BGEReranker, IdentityReranker, Reranker
from src.rag.retrieval import HybridRetriever, apply_tier_boost, resolve_conflicts

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """知识库检索引擎。

    初始化时会一次性加载文档、分块、构建向量索引和 BM25 索引。
    之后每次 search 只是在内存中做混合召回 + 重排，非常快。
    """

    def __init__(
        self,
        root: Path,
        *,
        embedding: str = "bge-m3",
        reranker: str = "bge-reranker-v2-m3",
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        embedding_cache_path: Path | None = None,
    ) -> None:
        self.root = root
        self.embedder: EmbeddingProvider = (
            BGEM3EmbeddingProvider(embedding_model)
            if embedding == "bge-m3"
            else HashEmbeddingProvider()
        )
        self.reranker: Reranker = (
            BGEReranker(reranker_model) if reranker == "bge-reranker-v2-m3" else IdentityReranker()
        )
        # 缓存命名空间含模型标识：切换嵌入模型/维度后旧向量自动失效
        if embedding == "bge-m3":
            namespace = f"bge-m3:{embedding_model}"
        else:
            namespace = f"hash:{getattr(self.embedder, 'dimensions', 'na')}"
        cache_path = embedding_cache_path or get_settings().embedding_cache_path
        # 缓存器在多次 refresh 之间复用，连接和已积累的缓存都保持有效
        self.cached_embedder = CachedCorpusEmbedder(self.embedder, namespace, cache_path)
        # 增量 refresh 状态：文件清单 + 每个文档的分块结果（避免未变更文档重读重切）
        self._doc_manifest: dict[str, tuple[float, int]] = {}
        self._doc_chunks: dict[str, list] = {}
        self.documents: list = []
        self.chunks: list = []
        self.retriever = None
        self.entity_index = EntityIndex.build([])
        logger.info(
            "初始化知识库：root=%s，嵌入=%s，重排=%s",
            root,
            embedding,
            reranker,
        )
        self.refresh()

    def refresh(self) -> None:
        """增量重建索引：仅重新处理变更文件，未变更文档复用内存中的分块结果。

        - 用 (mtime, size) 判断文件是否变更；
        - 无变更时直接返回（毫秒级）；
        - 有变更时只重读变更文件、重切分块，索引仍需整体重建（纯内存运算）。
        """

        started = time.perf_counter()
        current_stats = scan_document_stats(self.root)
        # 快速路径：清单完全一致，无需任何处理
        if current_stats == self._doc_manifest:
            logger.debug("知识库无变更，跳过 refresh")
            return

        logger.info("开始增量重建知识库索引…")
        try:
            old_manifest = self._doc_manifest
            changed = [sid for sid, st in current_stats.items() if old_manifest.get(sid) != st]
            deleted = [sid for sid in old_manifest if sid not in current_stats]

            # 处理变更/新增文档：加载 + 分块
            for source_id in changed:
                doc = load_document(self.root, source_id)
                if doc is not None:
                    self._doc_chunks[source_id] = chunk_document(doc)
                else:
                    # 内容为空的文件从分块映射中移除
                    self._doc_chunks.pop(source_id, None)

            # 处理删除文档
            for source_id in deleted:
                self._doc_chunks.pop(source_id, None)

            self._doc_manifest = current_stats

            # 展平所有文档的分块，按 source_id 排序保证稳定
            self.chunks = []
            for source_id in sorted(self._doc_chunks):
                self.chunks.extend(self._doc_chunks[source_id])

            if not self.chunks:
                self.retriever = HybridRetriever([], self.embedder)
                self.entity_index = EntityIndex.build([])
                logger.info(
                    "知识库索引重建完成：0 个文档 → 0 个块，耗时 %.2fs",
                    time.perf_counter() - started,
                )
                return

            corpus_vectors = self.cached_embedder.embed_corpus(
                [chunk.text for chunk in self.chunks]
            )
            self.cached_embedder.prune([chunk.text for chunk in self.chunks])
            self.retriever = HybridRetriever(
                self.chunks, self.embedder, corpus_vectors=corpus_vectors
            )
            self.entity_index = EntityIndex.build(self.chunks)
        except Exception:
            logger.exception("知识库索引重建失败，旧索引保留（首次构建则服务不可检索）")
            raise
        logger.info(
            "知识库索引重建完成：%d 个文档 → %d 个块"
            "（变更 %d / 删除 %d，嵌入新计算 %d / 缓存命中 %d），耗时 %.2fs",
            len(self._doc_chunks),
            len(self.chunks),
            len(changed),
            len(deleted),
            self.cached_embedder.last_misses,
            self.cached_embedder.last_hits,
            time.perf_counter() - started,
        )

    def search(
        self,
        query: str,
        top_k: int = 6,
        *,
        dense_query: str | None = None,
        sparse_query: str | None = None,
        enhanced_queries: list[str] | None = None,
        crag_threshold: float = 0.3,
    ) -> list[SearchResult]:
        """混合召回 → 冲突过滤 → 重排 → CRAG 自我纠正 → 返回最终候选。

        召回阶段先取 top_k * 4（至少 20）个候选，冲突过滤只保留同文档族
        的最新版本，再由重排器精排后截到 top_k。

        CRAG（Corrective RAG）：若首轮重排后最高分低于 ``crag_threshold``，
        说明检索结果可能不相关，自动用 ``enhanced_queries`` 中的查询重试，
        取分数最高的一轮结果返回。无 enhanced_queries 时跳过纠正。
        """

        started = time.perf_counter()

        def _retrieve(dq: str | None, sq: str | None) -> list[SearchResult]:
            candidates = self.retriever.search(
                query,
                top_k=max(top_k * 4, 20),
                dense_query=dq,
                sparse_query=sq,
            )
            candidates = self._inject_entities(query, candidates)
            candidates = resolve_conflicts(candidates)
            # 先多留一倍候选再提权截断：让排位靠后的 Wiki 片段有机会
            # 凭层级权威进入最终 top_k（知识治理：Wiki 权威、原始文档兜底）
            reranked = self.reranker.rerank(query, candidates, top_k * 2)
            return apply_tier_boost(reranked, get_settings().wiki_tier_boost)[:top_k]

        results = _retrieve(dense_query, sparse_query)

        # CRAG 自我纠正：最高分过低时用增强查询重试
        if enhanced_queries and results:
            top_score = results[0].score
            if top_score < crag_threshold:
                logger.info(
                    "CRAG 触发：首轮最高分 %.3f < 阈值 %.3f，尝试 %d 个增强查询",
                    top_score, crag_threshold, len(enhanced_queries),
                )
                best_results = results
                best_score = top_score
                for eq in enhanced_queries:
                    alt = _retrieve(eq, eq)
                    if alt and alt[0].score > best_score:
                        best_results = alt
                        best_score = alt[0].score
                if best_results is not results:
                    logger.info(
                        "CRAG 纠正成功：最高分 %.3f → %.3f",
                        top_score, best_score,
                    )
                    results = best_results

        sources = [item.chunk.source_id for item in results]
        logger.info(
            "检索完成 query=%r → %d 个命中，来源=%s，耗时 %.0fms",
            query[:60],
            len(results),
            sources,
            (time.perf_counter() - started) * 1000,
        )
        return results

    def _inject_entities(
        self, query: str, candidates: list[SearchResult]
    ) -> list[SearchResult]:
        """实体检索（ENTITY_SEARCH）：补入查询命中的、未被召回的实体页。

        - 只在 ``entity_search_enabled`` 且索引非空时生效；
        - 已被向量/BM25 召回的实体页（按 source_id）不重复注入；
        - 注入分块赋基分 ``max(候选分) * 0.9`` 进入重排候选池，由重排器
          按真实相关性定序，镜像 tier="wiki" 的提权照常生效。
        """

        settings = get_settings()
        if not settings.entity_search_enabled or self.entity_index is None:
            return candidates
        max_hits = settings.entity_search_max_hits
        if max_hits <= 0 or not candidates:
            return candidates
        seen = {item.chunk.source_id for item in candidates}
        hits = [
            chunk
            for chunk in self.entity_index.lookup(query)
            if chunk.source_id not in seen
        ][:max_hits]
        if not hits:
            return candidates
        base = max(item.score for item in candidates) * 0.9
        logger.info(
            "实体检索注入 %d 个实体页: %s",
            len(hits),
            [h.source_id for h in hits],
        )
        return [*candidates, *(SearchResult(chunk=h, score=base) for h in hits)]