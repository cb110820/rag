"""增量嵌入缓存与知识库增量重建测试。"""

from __future__ import annotations

import numpy as np

from src.rag.embedding_cache import CachedCorpusEmbedder
from src.rag.pipeline import KnowledgeBase


class CountingEmbedder:
    """记录被请求嵌入的文本条数，模拟昂贵的嵌入模型。"""

    dimensions = 4

    def __init__(self) -> None:
        self.embedded = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embedded += len(texts)
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            matrix[row, row % self.dimensions] = float(len(text) % 7) + 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def _make_cache(
    tmp_path, namespace: str = "test:v1"
) -> tuple[CachedCorpusEmbedder, CountingEmbedder]:
    provider = CountingEmbedder()
    cache = CachedCorpusEmbedder(provider, namespace, tmp_path / "embeddings.sqlite3")
    return cache, provider


def test_first_call_computes_all_second_call_all_hits(tmp_path) -> None:
    cache, provider = _make_cache(tmp_path)
    texts = ["第一段文本", "第二段文本", "第三段文本"]

    first = cache.embed_corpus(texts)
    assert provider.embedded == 3
    assert cache.last_hits == 0
    assert cache.last_misses == 3
    assert first.shape == (3, 4)

    second = cache.embed_corpus(texts)
    assert provider.embedded == 3  # 没有新的模型调用
    assert cache.last_hits == 3
    assert cache.last_misses == 0
    np.testing.assert_array_equal(second, first)


def test_partial_change_only_embeds_new_chunks(tmp_path) -> None:
    cache, provider = _make_cache(tmp_path)
    cache.embed_corpus(["甲", "乙"])
    assert provider.embedded == 2

    cache.embed_corpus(["甲", "乙", "丙"])
    assert provider.embedded == 3  # 只嵌入新增的"丙"
    assert cache.last_hits == 2
    assert cache.last_misses == 1


def test_cache_persists_across_instances(tmp_path) -> None:
    db_path = tmp_path / "embeddings.sqlite3"
    provider1 = CountingEmbedder()
    cache1 = CachedCorpusEmbedder(provider1, "test:v1", db_path)
    texts = ["持久化文本一", "持久化文本二"]
    vectors1 = cache1.embed_corpus(texts)
    cache1.close()

    # 模拟服务重启：新实例、新嵌入器，但磁盘缓存还在
    provider2 = CountingEmbedder()
    cache2 = CachedCorpusEmbedder(provider2, "test:v1", db_path)
    vectors2 = cache2.embed_corpus(texts)
    assert provider2.embedded == 0
    np.testing.assert_array_equal(vectors1, vectors2)


def test_namespace_change_forces_recompute(tmp_path) -> None:
    cache, provider = _make_cache(tmp_path, namespace="model-a")
    cache.embed_corpus(["同一段文本"])
    assert provider.embedded == 1

    cache_b, provider_b = _make_cache(tmp_path, namespace="model-b")
    cache_b.embed_corpus(["同一段文本"])
    assert provider_b.embedded == 1  # 换模型命名空间后必须重新嵌入


def test_prune_removes_deleted_chunks(tmp_path) -> None:
    cache, _provider = _make_cache(tmp_path)
    cache.embed_corpus(["保留块", "待删块"])
    assert cache.prune(["保留块", "待删块"]) == 0

    deleted = cache.prune(["保留块"])
    assert deleted == 1
    # 剪枝后"待删块"需要重新嵌入
    texts = ["保留块", "待删块"]
    cache.embed_corpus(texts)
    assert cache.last_misses == 1
    assert cache.last_hits == 1


def test_knowledge_base_incremental_refresh_and_restart(tmp_path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    content_a = "退款政策：30 天内可以申请退款，需要提供订单号。" * 3
    content_b = "架构规范：所有对外服务必须统一经过 API 网关。" * 3
    (knowledge_dir / "a.md").write_text(content_a, encoding="utf-8")
    cache_path = tmp_path / "cache.sqlite3"

    kb = KnowledgeBase(
        knowledge_dir, embedding="hash", reranker="none", embedding_cache_path=cache_path
    )
    initial_chunks = len(kb.chunks)
    assert initial_chunks == 1
    assert kb.cached_embedder.last_misses == 1

    # 无变化重建：跳过嵌入（增量 refresh 直接返回，不调用 embed_corpus）
    misses_before = kb.cached_embedder.last_misses
    kb.refresh()
    assert kb.cached_embedder.last_misses == misses_before

    # 新增文件：只嵌入新文件的块
    (knowledge_dir / "b.md").write_text(content_b, encoding="utf-8")
    kb.refresh()
    b_chunks = sum(1 for chunk in kb.chunks if chunk.source_id == "b.md")
    assert b_chunks == 1
    assert kb.cached_embedder.last_misses == 1
    assert kb.cached_embedder.last_hits == 1

    # 新内容可被检索到
    results = kb.search("对外服务是不是必须走 API 网关", top_k=2)
    assert results
    assert results[0].chunk.source_id == "b.md"

    # 模拟重启：冷启动也不应重新嵌入
    kb2 = KnowledgeBase(
        knowledge_dir, embedding="hash", reranker="none", embedding_cache_path=cache_path
    )
    assert len(kb2.chunks) == 2
    assert kb2.cached_embedder.last_misses == 0

    # 删除文件：块与缓存均被剔除
    (knowledge_dir / "a.md").unlink()
    kb2.refresh()
    assert all(chunk.source_id != "a.md" for chunk in kb2.chunks)
    assert kb2.cached_embedder.prune([chunk.text for chunk in kb2.chunks]) == 0
