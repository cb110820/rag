"""记忆混合检索管线（multi-channel fusion）。

三阶段结构，与 ``src.rag.retrieval`` 的工程口径保持一致，但针对记忆条目
（单用户 ≤ 配额上限的短文本事实）做轻量化，不引入 FAISS：

1. **多路召回**
   - dense：LangGraph Store 向量索引（bge-m3 / hash），由调用方执行检索后
     把已过噪声门槛、按相似度降序的视图传入；
   - sparse：对候选全集实时构建 ``BM25Okapi`` 做关键词精确匹配
     （单用户事实量级为百条级，逐查询建索引成本可忽略，无需持久化）。
2. **RRF 粗排**：两路分数量纲不可比（余弦相似度 vs BM25 原始分），
   使用加权 Reciprocal Rank Fusion 只依赖名次融合，避免直接加分失真。
3. **recency 精排**：融合分乘以指数时间衰减因子，同等相关下新近确认/
   更新的事实靠前；半衰期与衰减下限可配置，半衰期为 0 时关闭衰减。

模块为纯函数集合：不读全局配置、不访问 Store，便于无 LLM 单测与离线评估。
"""

from __future__ import annotations

import datetime
import re
import time
from collections.abc import Sequence
from typing import Any

from rank_bm25 import BM25Okapi

# 英文/数字按单词切，中文按单字切（与 src.rag.retrieval.tokenize 同算法，
# 内存模块自持一份，避免反向依赖 RAG 重依赖栈）
_LATIN_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """中英文分词：英文按单词、中文按单字，统一小写。"""

    return _LATIN_RE.findall(text.lower()) + _CJK_RE.findall(text)


# ── 第一阶段：BM25 稀疏召回 ─────────────────────────────────────────


def bm25_rank(
    query: str,
    candidates: Sequence[dict[str, Any]],
    limit: int,
    *,
    text_key: str = "text",
    key_key: str = "key",
) -> list[tuple[str, float]]:
    """对候选字典集做 BM25 关键词检索，返回 ``(key, 非负BM25分)`` 降序列表。

    注意 BM25 的 IDF 陷阱：当查询词被全部候选包含（单候选语料必然如此）时，
    ``idf=log((N-n+0.5)/(n+0.5))`` 为负，原始分会小于 0。因此这里：

    1. 先用 token 交集筛出真正有词面命中的候选（零交集不参与排序）；
    2. 在命中子集内按 BM25 原始分排序（相对次序不受负 IDF 影响）；
    3. 分数整体平移减去子集最小分，保证返回值非负且保序
       （RRF 融合只消费名次，episodic 展示再做归一化）。
    """

    if not candidates or limit <= 0:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    query_set = set(query_tokens)
    corpus_tokens = [tokenize(str(item.get(text_key, ""))) for item in candidates]
    # 词面交集为零的候选直接排除，避免负 IDF 把无关文档排进来
    hit_indices = [
        idx for idx, tokens in enumerate(corpus_tokens)
        if query_set & set(tokens)
    ]
    if not hit_indices:
        return []
    bm25 = BM25Okapi(corpus_tokens)
    raw_scores = bm25.get_scores(query_tokens)
    hit_scored = sorted(
        ((float(raw_scores[idx]), idx) for idx in hit_indices),
        reverse=True,
    )[:limit]
    floor_score = hit_scored[-1][0]  # 命中子集最低分 → 平移基准
    return [
        (str(candidates[idx][key_key]), score - floor_score)
        for score, idx in hit_scored
    ]


# ── 第二阶段：加权 RRF 融合 ─────────────────────────────────────────


def rrf_fuse(
    rankings: dict[str, list[str]],
    *,
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
) -> dict[str, float]:
    """多路名次融合：``score = Σ channel_weight / (rrf_k + rank)``。

    Args:
        rankings: 通道名 → 该通道按相关度降序排列的 key 列表。
        weights: 通道权重（缺省全部 1.0）；权重为 0 的通道直接跳过。
        rrf_k: RRF 平滑常数，越大则名次差异对分数影响越平缓。
    """

    fused: dict[str, float] = {}
    for channel, ordered_keys in rankings.items():
        weight = (weights or {}).get(channel, 1.0)
        if weight <= 0:
            continue
        for rank, key in enumerate(ordered_keys, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (rrf_k + rank)
    return fused


# ── 第三阶段：recency 时间衰减精排 ─────────────────────────────────


def _parse_iso(value: Any) -> float | None:
    """ISO 字符串 → Unix 秒；无法解析返回 None。"""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed.timestamp()


def view_recency_ts(view: dict[str, Any]) -> float | None:
    """取事实的"最近确认时刻"：updated_at 优先，回退 valid_from/created_at。"""

    for field in ("updated_at", "valid_from", "created_at"):
        ts = _parse_iso(view.get(field))
        if ts is not None:
            return ts
    return None


def recency_factor(
    ts: float | None,
    *,
    now: float,
    half_life_days: float,
    floor: float,
) -> float:
    """指数时间衰减因子。

    ``factor = max(floor, 0.5 ** (age_days / half_life_days))``：
    一个半衰期时衰减到 0.5，两个半衰期 0.25……低于 floor 后钳住，
    避免老事实被压到近乎不可召回。``half_life_days <= 0`` 或时间戳缺失
    时返回 1.0（关闭衰减，纯相关性排序）。
    """

    if half_life_days <= 0 or ts is None:
        return 1.0
    age_days = max(0.0, (now - ts) / 86400.0)
    return max(floor, 0.5 ** (age_days / half_life_days))


# ── 管线编排 ───────────────────────────────────────────────────────


def hybrid_rank_facts(
    query: str,
    all_views: Sequence[dict[str, Any]],
    dense_ranked: Sequence[dict[str, Any]],
    *,
    k: int,
    candidate_k: int,
    rrf_k: int,
    dense_weight: float,
    sparse_weight: float,
    half_life_days: float,
    recency_floor: float,
    now: float | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """对已过滤的候选事实视图执行"BM25 + 向量 → RRF → recency"完整管线。

    Args:
        query: 检索语句。
        all_views: 候选全集视图（调用方已完成 active/时点有效性过滤），
            sparse 通道的建索引语料；每个视图必须含 ``key`` 与 ``text``。
        dense_ranked: dense 通道命中视图（已过 min_score 噪声门槛、按向量
            相似度降序）；向量索引关闭时传空序列。
        k: 最终返回条数。
        candidate_k: 单路召回候选池上限。

    Returns:
        ``(mode, views)``：mode 为 ``hybrid``（两路均有命中）/
        ``semantic``（仅 dense）/ ``keyword``（仅 sparse）；
        视图按融合精排分降序，最终分写回每个视图的 ``score`` 字段。
    """

    now = now if now is not None else time.time()
    views_by_key = {str(view["key"]): view for view in all_views}

    # 1) 两路召回（dense 由调用方给出；sparse 实时 BM25）
    dense_keys = [str(view["key"]) for view in dense_ranked][:candidate_k]
    sparse_pairs = bm25_rank(query, list(all_views), candidate_k)
    sparse_keys = [key for key, _ in sparse_pairs]

    rankings = {
        "dense": dense_keys,
        "sparse": sparse_keys,
    }
    weights = {"dense": dense_weight, "sparse": sparse_weight}
    fused = rrf_fuse(rankings, weights=weights, rrf_k=rrf_k)
    if not fused:
        # 两路均无命中：模式 label 已无实际意义，沿用 keyword 口径
        return "keyword", []

    # 2) 模式标注：实际产生命中的通道决定 label
    if dense_keys and sparse_keys:
        mode = "hybrid"
    elif dense_keys:
        mode = "semantic"
    else:
        mode = "keyword"

    # 3) recency 精排：RRF 融合分 × 时间衰减因子
    scored: list[tuple[float, str]] = []
    for key, fused_score in fused.items():
        view = views_by_key.get(key)
        if view is None:
            # dense 命中理论上必在全集中；防御性跳过不一致数据
            continue
        factor = recency_factor(
            view_recency_ts(view),
            now=now,
            half_life_days=half_life_days,
            floor=recency_floor,
        )
        scored.append((fused_score * factor, key))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    final_views: list[dict[str, Any]] = []
    for final_score, key in scored[:k]:
        view = dict(views_by_key[key])
        # 覆盖向量原始分：工具输出展示的是最终融合精排分（四位小数）
        view["score"] = round(final_score, 4)
        final_views.append(view)
    return mode, final_views
