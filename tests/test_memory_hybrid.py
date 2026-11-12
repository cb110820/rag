"""记忆混合检索管线测试（dense + BM25 → RRF → recency 精排）。

分两层：
1. 纯函数单测（无 DB、无 LLM）：分词、BM25、加权 RRF、时间衰减、
   管线模式标注与精排序，确定性可复现；
2. 存储集成（isolated_db + hash 嵌入）：端到端 mode 标注、
   kill switch 回退、as-of 走同一管线。
"""

from __future__ import annotations

import datetime as dt
import time

import pytest

from src.memory import retrieval
from src.memory.tools import _recall_memories, _recall_memories_as_of, _save_memory
from src.persistence import database as db

pytestmark = pytest.mark.usefixtures("isolated_db")

TENANT = db.DEFAULT_TENANT_ID
U1 = "hybrid-user-0001"


def _view(key: str, text: str, *, updated_at: str | None = None) -> dict:
    return {
        "key": key,
        "text": text,
        "category": "general",
        "updated_at": updated_at,
        "valid_from": updated_at,
        "created_at": updated_at,
        "status": "active",
    }


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.UTC).isoformat()


# ── 1. 纯函数：分词与 BM25 ─────────────────────────────────────────


def test_tokenize_splits_latin_words_and_cjk_chars() -> None:
    tokens = retrieval.tokenize("Aurora 连接池 timeout 排查")
    assert "aurora" in tokens and "timeout" in tokens
    # 中文按单字切，逐字参与 BM25 匹配
    assert {"连", "接", "池", "排", "查"}.issubset(set(tokens))


def test_bm25_rank_rare_exact_token_wins() -> None:
    candidates = [
        _view("f1", "项目代号 Aurora 的灰度发布步骤"),
        _view("f2", "茶水间咖啡机维修记录与行政通知"),
        _view("f3", "Aurora 数据库连接池耗尽的根因排查"),
    ]
    ranked = retrieval.bm25_rank("Aurora 连接池", candidates, limit=3)
    keys = [key for key, _ in ranked]
    # f3 覆盖全部查询词（Aurora+连接池），BM25 确定地排在仅含 Aurora 的 f1 之前；
    # 零交集事实 f2 不出现
    assert keys == ["f3", "f1"]
    # 平移后分数非负，且首位严格高于末位（保序）
    assert ranked[0][1] > ranked[-1][1] >= 0


def test_bm25_rank_guards_empty_inputs() -> None:
    assert retrieval.bm25_rank("query", [], 5) == []
    assert retrieval.bm25_rank("", [_view("f1", "文本")], 5) == []
    assert retrieval.bm25_rank("query", [_view("f1", "文本")], 0) == []
    # 查询与语料无 token 交集 → 空命中而非负分
    assert retrieval.bm25_rank("zzzz", [_view("f1", " Aurora 记录")], 5) == []


def test_bm25_rank_single_candidate_negative_idf_regression() -> None:
    # 回归：单候选语料下 IDF 必为负，旧实现按 score>0 过滤会误杀全部命中
    ranked = retrieval.bm25_rank(
        "Aurora", [_view("only", "项目代号 Aurora 的发布窗口")], 5
    )
    assert ranked == [("only", 0.0)]  # 平移后非负、保序、不丢命中


# ── 2. 纯函数：加权 RRF 融合 ───────────────────────────────────────


def test_rrf_fuse_rewards_cross_channel_agreement() -> None:
    # b 同时出现在两路，a 只在 dense 首位：跨通道共识的 b 融合分更高
    fused = retrieval.rrf_fuse(
        {"dense": ["a", "b", "c"], "sparse": ["b", "d"]}, rrf_k=60
    )
    assert fused["b"] > fused["a"] > fused["c"]
    # 名次常数生效：rank1 分 = 1/61
    assert fused["a"] == pytest.approx(1 / 61)


def test_rrf_fuse_respects_channel_weights() -> None:
    base = retrieval.rrf_fuse(
        {"dense": ["a", "b"], "sparse": ["b", "a"]}, rrf_k=60
    )
    # 等权 + 对称名次 → a/b 完全打平
    assert base["a"] == pytest.approx(base["b"])

    weighted = retrieval.rrf_fuse(
        {"dense": ["a", "b"], "sparse": ["b", "a"]},
        weights={"dense": 2.0, "sparse": 1.0},
        rrf_k=60,
    )
    # dense 加权后其首位 a 反超
    assert weighted["a"] > weighted["b"]

    # 权重 0 的通道整体跳过
    dense_only = retrieval.rrf_fuse(
        {"dense": ["a", "b"], "sparse": ["b", "a"]},
        weights={"dense": 1.0, "sparse": 0.0},
        rrf_k=60,
    )
    assert set(dense_only) == {"a", "b"}
    assert dense_only["a"] > dense_only["b"]


# ── 3. 纯函数：recency 时间衰减 ────────────────────────────────────


def test_recency_factor_half_life_and_floor() -> None:
    now = 1_000_000.0
    # 无时间戳或关闭衰减 → 中性因子
    assert retrieval.recency_factor(
        None, now=now, half_life_days=30, floor=0.5
    ) == 1.0
    assert retrieval.recency_factor(
        now - 999, now=now, half_life_days=0, floor=0.5
    ) == 1.0
    # 当下 = 1.0；一个半衰期 = 0.5；两个半衰期 = 0.25
    fresh = retrieval.recency_factor(
        now, now=now, half_life_days=30, floor=0.1
    )
    half = retrieval.recency_factor(
        now - 30 * 86400, now=now, half_life_days=30, floor=0.1
    )
    quarter = retrieval.recency_factor(
        now - 60 * 86400, now=now, half_life_days=30, floor=0.1
    )
    assert fresh == pytest.approx(1.0)
    assert half == pytest.approx(0.5)
    assert quarter == pytest.approx(0.25)
    # 衰减下限钳制：0.25 低于 floor=0.5 时取 0.5
    clamped = retrieval.recency_factor(
        now - 60 * 86400, now=now, half_life_days=30, floor=0.5
    )
    assert clamped == pytest.approx(0.5)


# ── 4. 纯函数：完整管线的模式标注与精排序 ──────────────────────────


def test_hybrid_pipeline_mode_labels_and_truncation() -> None:
    now = time.time()
    all_views = [
        _view("f1", "Aurora 灰度发布步骤", updated_at=_iso(now - 10 * 86400)),
        _view("f2", "Aurora 连接池排查", updated_at=_iso(now)),
        _view("f3", "咖啡机维修", updated_at=_iso(now)),
    ]
    # dense 命中 f1/f2；BM25（查询含 Aurora/连接池）命中 f1/f2 → hybrid
    mode, facts = retrieval.hybrid_rank_facts(
        "Aurora 连接池",
        all_views,
        [all_views[0], all_views[1]],
        k=2,
        candidate_k=20,
        rrf_k=60,
        dense_weight=1.0,
        sparse_weight=1.0,
        half_life_days=30,
        recency_floor=0.5,
        now=now,
    )
    assert mode == "hybrid"
    assert len(facts) == 2  # k 截断生效
    assert {f["key"] for f in facts} == {"f1", "f2"}
    # f1/f2 两路名次对称 → RRF 打平，recency 让更新的 f2 居首
    assert facts[0]["key"] == "f2"
    # 最终分已写回视图且为融合精排分（非向量原始分）
    assert all(isinstance(f["score"], float) for f in facts)


def test_hybrid_pipeline_dense_only_and_sparse_only_labels() -> None:
    now = time.time()
    views = [_view("f1", "Aurora 发布", updated_at=_iso(now))]

    # 仅 dense（查询词与语料无字面交集，BM25 零命中）
    mode_sem, facts_sem = retrieval.hybrid_rank_facts(
        "zzzz-no-overlap",
        views,
        views,
        k=5,
        candidate_k=20,
        rrf_k=60,
        dense_weight=1.0,
        sparse_weight=1.0,
        half_life_days=0,
        recency_floor=0.5,
        now=now,
    )
    assert mode_sem == "semantic"
    assert [f["key"] for f in facts_sem] == ["f1"]

    # 仅 sparse（无向量通道）
    mode_kw, facts_kw = retrieval.hybrid_rank_facts(
        "Aurora",
        views,
        [],
        k=5,
        candidate_k=20,
        rrf_k=60,
        dense_weight=1.0,
        sparse_weight=1.0,
        half_life_days=0,
        recency_floor=0.5,
        now=now,
    )
    assert mode_kw == "keyword"
    assert [f["key"] for f in facts_kw] == ["f1"]


# ── 5. 存储集成：端到端 mode 标注与回退 ────────────────────────────


async def test_recall_uses_hybrid_mode_in_hash_embedding() -> None:
    await _save_memory(TENANT, U1, "项目代号 Aurora，每周三同步进度", "project")
    out = await _recall_memories(TENANT, U1, "Aurora 什么时候同步", k=3)
    assert "mode=hybrid" in out
    assert "Aurora" in out


async def test_hybrid_kill_switch_restores_legacy_semantic(isolated_db) -> None:
    isolated_db.memory_hybrid_retrieval_enabled = False
    await _save_memory(TENANT, U1, "项目代号 Aurora，每周三同步进度", "project")
    out = await _recall_memories(TENANT, U1, "Aurora 什么时候同步", k=3)
    assert "mode=semantic" in out
    assert "Aurora" in out


async def test_none_embedding_routes_bm25_keyword_channel(isolated_db) -> None:
    isolated_db.memory_embedding = "none"
    await db.close_db()
    await db.init_db()

    await _save_memory(TENANT, U1, "项目代号 Aurora，每周三同步进度")
    await _save_memory(TENANT, U1, "茶水间咖啡机已经修好")
    out = await _recall_memories(TENANT, U1, "Aurora 项目同步", k=2)
    # 无向量通道：BM25 单路，label 仍为 keyword
    assert "mode=keyword" in out
    assert "Aurora" in out
    assert "咖啡机" not in out


async def test_as_of_recall_uses_same_hybrid_pipeline() -> None:
    text = "团队周报固定在每周五下午三点提交"
    r1 = await _save_memory(TENANT, U1, text, "project")
    old_key = r1.split("memory_id=")[1].split("。")[0]

    time.sleep(1.1)
    await _save_memory(TENANT, U1, text, "project")

    ns = db.user_facts_namespace(TENANT, U1)
    old_value = (await db.get_store().aget(ns, old_key)).value
    mid_ts = (old_value["valid_from"] + old_value["invalid_at"]) / 2
    mid_iso = dt.datetime.fromtimestamp(mid_ts, tz=dt.UTC).isoformat()

    out = await _recall_memories_as_of(TENANT, U1, "周报什么时候", mid_iso, k=5)
    assert "mode=hybrid" in out
    assert old_key in out
