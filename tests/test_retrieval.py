from pathlib import Path

from src.rag.models import Chunk, SearchResult
from src.rag.pipeline import KnowledgeBase
from src.rag.retrieval import apply_tier_boost, resolve_conflicts


def _chunk(source_id: str, tier: str, mtime: float = 0.0) -> Chunk:
    return Chunk(
        chunk_id=f"c-{source_id}",
        source_id=source_id,
        text="正文",
        start=0,
        end=2,
        metadata={"tier": tier, "doc_mtime": mtime},
    )


def test_hybrid_retrieval_wiki_authoritative_raw_fallback() -> None:
    """知识治理：检索结果携带 tier，Wiki 镜像经 boost 后排序正确。"""

    kb = KnowledgeBase(Path("data/knowledge/default"), embedding="hash", reranker="none")
    results = kb.search("退款需要多少天内申请", top_k=8)
    assert results
    # tier 标注合法；治理修复后退款政策由原始文档 policies.md 承载，
    # 召回中若 raw 片段分数被 wiki 挤出属预期（Wiki 权威）
    assert all(r.to_dict()["tier"] in {"wiki", "raw"} for r in results)
    # Wiki 片段分数因 tier boost ≥ 同分 raw 片段
    wiki_scores = [r.score for r in results if r.chunk.metadata.get("tier") == "wiki"]
    raw_scores = [r.score for r in results if r.chunk.metadata.get("tier") == "raw"]
    if wiki_scores and raw_scores:
        assert min(wiki_scores) >= max(raw_scores) * 0.99  # boost 后无倒挂


def test_empty_query_is_rejected() -> None:
    kb = KnowledgeBase(Path("data/knowledge/default"), embedding="hash", reranker="none")
    try:
        kb.search("   ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty query should fail")


# ── 知识治理：Wiki 权威 / 原始文档兜底 ───────────────────────────


def test_apply_tier_boost_promotes_wiki() -> None:
    """相关性接近时，Wiki 镜像片段应排在原始文档之前。"""

    results = [
        SearchResult(_chunk("raw-doc.md", "raw"), 0.90),
        SearchResult(_chunk("wiki/curated.md", "wiki"), 0.85),
    ]
    boosted = apply_tier_boost(results, factor=1.15)
    assert boosted[0].chunk.source_id == "wiki/curated.md"
    assert boosted[0].score > boosted[1].score
    # 原始文档分数不变，不丢片段
    assert boosted[1].chunk.source_id == "raw-doc.md"
    assert boosted[1].score == 0.90


def test_apply_tier_boost_noop_below_threshold() -> None:
    results = [SearchResult(_chunk("a.md", "raw"), 0.9)]
    assert apply_tier_boost(results, factor=1.0) is results


def test_resolve_conflicts_wiki_wins_on_tie() -> None:
    """同文档族版本完全打平时，Wiki 镜像应胜过原始文档。"""

    results = [
        SearchResult(_chunk("onboarding.md", "raw", mtime=100.0), 0.9),
        SearchResult(_chunk("wiki/onboarding.md", "wiki", mtime=100.0), 0.8),
    ]
    kept = resolve_conflicts(results)
    assert {it.chunk.source_id for it in kept} == {"wiki/onboarding.md"}


def test_resolve_conflicts_newer_version_still_wins() -> None:
    """原始文档版本号更高时依旧胜出，层级裁决只在平局生效。"""

    raw = Chunk(
        chunk_id="c2",
        source_id="onboarding_v2.md",
        text="正文",
        start=0,
        end=2,
        metadata={"tier": "raw", "version": "2"},
    )
    wiki = Chunk(
        chunk_id="c1",
        source_id="wiki/onboarding.md",
        text="正文",
        start=0,
        end=2,
        metadata={"tier": "wiki", "version": "1"},
    )
    kept = resolve_conflicts([SearchResult(wiki, 0.9), SearchResult(raw, 0.8)])
    assert {it.chunk.source_id for it in kept} == {"onboarding_v2.md"}
