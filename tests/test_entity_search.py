"""实体检索（ENTITY_SEARCH）测试：别名命中、去重、提权、开关、归一化等价。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag.entity_index import EntityIndex, normalize_entity_name
from src.rag.models import Chunk, SearchResult
from src.rag.pipeline import KnowledgeBase
from src.rag.retrieval import apply_tier_boost

pytestmark = pytest.mark.usefixtures("isolated_db")


def _entity_chunk(source_id: str = "wiki/entity-wang-xiao-ming.md",
                  name: str = "王小明", aliases: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"c-{source_id}", source_id=source_id,
        text=f"# {name}\n财务负责人", start=0, end=10,
        metadata={
            "kind": "entity", "title": name, "tier": "wiki",
            "aliases": aliases or ["老张"],
        },
    )


def _kb(index: EntityIndex) -> KnowledgeBase:
    kb = object.__new__(KnowledgeBase)
    kb.entity_index = index
    return kb


def _settings(**overrides):
    base = dict(
        entity_search_enabled=True,
        entity_search_max_hits=2,
        wiki_tier_boost=1.15,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_alias_hit_recalls_entity_page() -> None:
    idx = EntityIndex.build([_entity_chunk()])

    hits = idx.lookup("王小明的审批权限是多少")

    assert len(hits) == 1
    assert hits[0].source_id == "wiki/entity-wang-xiao-ming.md"


def test_alias_name_hit_via_alias() -> None:
    idx = EntityIndex.build([_entity_chunk()])

    assert idx.lookup("老张负责什么")  # 别名命中


def test_no_false_hit_on_unrelated_query() -> None:
    idx = EntityIndex.build([_entity_chunk()])

    assert idx.lookup("报销流程") == []


def test_dedup_when_already_recalled() -> None:
    """已被向量/BM25 召回的实体页（按 source_id）不重复注入。"""

    chunk = _entity_chunk()
    kb = _kb(EntityIndex.build([chunk]))
    existing = SearchResult(chunk=chunk, score=0.8)
    kb.__dict__["entity_index"] = EntityIndex.build([chunk])

    injected = kb._inject_entities("王小明 审批", [existing])

    assert injected == [existing]  # 无新增


def test_injection_base_score_and_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.rag.pipeline.get_settings",
                        lambda: _settings(entity_search_max_hits=2))
    chunk_a = _entity_chunk("wiki/entity-a.md", "王小明")
    chunk_b = _entity_chunk("wiki/entity-b.md", "李四", ["小李"])
    kb = _kb(EntityIndex.build([chunk_a, chunk_b]))
    cand = SearchResult(chunk=Chunk("x", "raw/doc.md", "正文", 0, 5, {}), score=0.6)

    injected = kb._inject_entities("王小明和李四", [cand])

    assert len(injected) == 3
    new_scores = [r.score for r in injected[1:]]
    assert all(abs(s - 0.6 * 0.9) < 1e-9 for s in new_scores)  # 基分 max*0.9


def test_max_hits_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.rag.pipeline.get_settings",
                        lambda: _settings(entity_search_max_hits=1))
    chunk_a = _entity_chunk("wiki/entity-a.md", "王小明")
    chunk_b = _entity_chunk("wiki/entity-b.md", "李四")
    kb = _kb(EntityIndex.build([chunk_a, chunk_b]))
    cand = SearchResult(chunk=Chunk("x", "raw/doc.md", "正文", 0, 5, {}), score=0.6)

    injected = kb._inject_entities("王小明和李四", [cand])

    assert len(injected) == 2  # 最多注入 1 个


def test_disabled_no_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.rag.pipeline.get_settings",
                        lambda: _settings(entity_search_enabled=False))
    chunk = _entity_chunk()
    kb = _kb(EntityIndex.build([chunk]))
    cand = SearchResult(chunk=Chunk("x", "raw/doc.md", "正文", 0, 5, {}), score=0.6)

    assert kb._inject_entities("王小明", [cand]) == [cand]


def test_injected_entity_keeps_wiki_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """注入的实体镜像保持 tier=wiki，提权后分数应高于同分原始片段。"""

    monkeypatch.setattr("src.rag.pipeline.get_settings", lambda: _settings())
    ent = _entity_chunk()
    kb = _kb(EntityIndex.build([ent]))
    cand = SearchResult(chunk=Chunk("x", "raw/doc.md", "正文", 0, 5, {"tier": "raw"}),
                        score=0.6)

    injected = kb._inject_entities("王小明 审批", [cand])
    boosted = apply_tier_boost(injected, 1.15)

    ent_item = next(r for r in boosted if r.chunk.source_id.startswith("wiki/"))
    raw_item = next(r for r in boosted if r.chunk.source_id.startswith("raw/"))
    assert ent_item.chunk.metadata["tier"] == "wiki"
    assert ent_item.score > raw_item.score


def test_normalize_matches_curation() -> None:
    """红线校验：rag 侧归一化与 curation 侧逐字一致。"""

    from src.curation.markdown import normalize_entity_name as curation_norm

    samples = ["王小明", "  AI 助手 v2.0 ", "ＡＢＣ－１２３", "e-mail_USER"]
    for s in samples:
        assert normalize_entity_name(s) == curation_norm(s), s
