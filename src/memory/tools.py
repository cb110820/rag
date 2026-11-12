"""语义事实记忆工具（主代理可调用，跨会话沉淀与召回用户事实）。

与画像文件 ``/memory/AGENTS.md`` 的分工：

- **画像（AGENTS.md）**：少量、稳定、高度浓缩的身份与偏好（文件式编辑）；
- **事实记忆（本模块）**：事件型/结论型的独立条目，结构化保存、向量召回、
  可单独删除、可设置 TTL，避免单文件无限膨胀。

四个工具：

- ``save_memory(text, category?, ttl_days?)``：保存事实；语义去重命中时
  不再原地覆盖，而是建立"取代链"——新事实以 active 新建，旧事实标记为
  superseded 并保留一个审计窗口（默认 14 天 TTL，到期物理清除），
  另有凭据扫描、条目/长度配额；事实携带 valid_from/invalid_at 双时间轴；
- ``recall_memories(query, k?)``：混合检索召回当前有效（active）事实——
  dense 向量与 sparse BM25 两路并行召回，RRF 融合后按 recency 时间衰减精排；
  superseded 旧版本不进召回；可通过 ``memory_hybrid_retrieval_enabled``
  关闭混合管线回退到单通道旧行为；
- ``recall_memories_as_of(query, as_of, k?)``：按 valid_from/invalid_at
  重建某个历史时点有效的事实（point-in-time 查询，旧版本在失效前可命中）；
- ``forget_memory(memory_id)``：按 id 显式遗忘单条事实。

所有 Store 操作带超时（MEMORY_TIMEOUT_SECONDS）与 tenacity 重试，
符合生产红线；每一步均输出 src.memory 审计日志。

工具内的 user_id 从 LangGraph 运行时 context 获取（与 StoreBackend 同一注入
链路），不进 prompt、不暴露给模型。
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
import sqlite3
import time
from functools import partial
from typing import Any

import anyio
from langchain_core.tools import tool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.memory import retrieval
from src.memory.guard import scan_secret
from src.persistence import database as db
from src.rag.config import get_settings

logger = logging.getLogger("src.memory")

_ALLOWED_CATEGORIES = {
    "general",
    "identity",
    "preference",
    "project",
    "decision",
    "feedback",
}
_FACT_KEY_PREFIX = "fact-"

# 事实状态：active=当前有效（召回/配额口径）；superseded=被新事实取代的旧版本
_FACT_STATUS_ACTIVE = "active"
_FACT_STATUS_SUPERSEDED = "superseded"
# 单用户事实扫描上限：取代链会让总条目数大于 active 配额，扫描窗口不能按
# 配额截断，否则 superseded 堆积会导致 active 计数偏低、配额被绕过
_FACT_SCAN_LIMIT = 1000


# ── 运行时身份 ─────────────────────────────────────────────────────


def _current_identity() -> tuple[str, str]:
    """从 LangGraph 运行时 context 取当前 (tenant_id, user_id)。"""

    from langgraph.runtime import get_runtime

    ctx = get_runtime().context
    return ctx.tenant_id, ctx.user_id


# ── 关键词兜底（向量索引关闭时） ───────────────────────────────────

_TOKEN_RE = re.compile(r"[\w一-鿿]")


def _tokenize(text: str) -> set[str]:
    """与 HashEmbeddingProvider 一致的粗粒度分词（含逐字中文）。"""

    return set(_TOKEN_RE.findall(text.lower()))


def _keyword_rank(query: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """无向量索引时的本地打分：查询 token 被事实覆盖的比例（0~1）。"""

    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for fact in facts:
        overlap = len(q_tokens & _tokenize(fact["text"]))
        score = overlap / len(q_tokens)
        if score > 0:
            scored.append((score, fact))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    for score, fact in scored:
        fact["score"] = round(score, 4)
    return [fact for _, fact in scored]


def _jaccard(text_a: str, text_b: str) -> float:
    """两个文本 token 集合的 Jaccard 相似度（0~1）。"""

    a, b = _tokenize(text_a), _tokenize(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── 混合检索候选收集（dense + sparse 两路 → retrieval 管线） ────────


async def _hybrid_collect(
    store: Any,
    ns: tuple[str, ...],
    query: str,
    k: int,
    settings: Any,
    validity_predicate: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """执行两路并行召回并交给 :mod:`src.memory.retrieval` 做 RRF + recency 精排。

    Args:
        validity_predicate: 对原始 value 字典的有效性谓词（当前 active 过滤
            或 as-of 时点过滤），dense 与全量扫描两条通道使用同一谓词，
            保证融合候选的口径一致。
    """

    candidate_k = max(settings.memory_hybrid_candidate_k, k)

    if settings.memory_vector_enabled:
        # 两路 IO 相互独立：并行取数，尾延迟不随通道数线性增长
        dense_items, all_items = await asyncio.gather(
            store.asearch(ns, query=query, limit=candidate_k),
            store.asearch(ns, limit=_FACT_SCAN_LIMIT),
        )
        # dense 通道保留原有的噪声门槛（hash 近似正交 / 低相似度不进融合）
        min_score = settings.memory_recall_min_score
        dense_views: list[dict[str, Any]] = []
        for item in dense_items:
            raw = item.value or {}
            if not validity_predicate(raw):
                continue
            view = db._fact_to_view(item)
            score = view.get("score")
            if score is not None and score < min_score:
                continue
            dense_views.append(view)
    else:
        # 无向量索引：只有 sparse 一路，全量扫描后走 BM25
        all_items = await store.asearch(ns, limit=_FACT_SCAN_LIMIT)
        dense_views = []

    all_views = [
        db._fact_to_view(item)
        for item in all_items
        if validity_predicate(item.value or {})
    ]
    return retrieval.hybrid_rank_facts(
        query,
        all_views,
        dense_views,
        k=k,
        candidate_k=candidate_k,
        rrf_k=settings.memory_rrf_k,
        dense_weight=settings.memory_dense_channel_weight,
        sparse_weight=settings.memory_sparse_channel_weight,
        half_life_days=settings.memory_recency_half_life_days,
        recency_floor=settings.memory_recency_floor,
    )


# ── Store 操作的超时/重试包装 ──────────────────────────────────────


def _retryable_exceptions() -> tuple[type[BaseException], ...]:
    """可重试异常：超时/网络类 + 数据库偶发错误。

    aiosqlite 抛出的 ``sqlite3.OperationalError``（busy_timeout 耗尽后的
    "database is locked"）不是 OSError 子类；psycopg 的连接抖动/死锁同理。
    不纳入重试会让偶发锁错直接变成用户可见的"记忆服务暂时不可用"。
    """

    exc_types: list[type[BaseException]] = [
        OSError, TimeoutError, RuntimeError, sqlite3.Error,
    ]
    try:
        import psycopg

        exc_types.append(psycopg.OperationalError)
    except ImportError:
        pass
    return tuple(exc_types)


def _memory_retry():
    settings = get_settings()
    return retry(
        retry=retry_if_exception_type(_retryable_exceptions()),
        wait=wait_exponential(multiplier=0.3, min=0.3, max=2),
        stop=stop_after_attempt(settings.memory_max_retries),
        reraise=True,
    )


def _effective_expiry_ts(settings: Any, ttl_days: int) -> float | None:
    """按 TTL 绝对语义计算新条目的过期 Unix 秒；永久保留返回 None。

    过期时刻冗余保存在 value 中：langgraph 1.2.x 的 ``Item`` 不暴露
    store 表的 expires_at 列，去重更新时只能靠 value 里的副本折算剩余
    TTL，否则 ``aput`` 不传 ttl 会把过期时间重置为 Store 默认值
    （如 7 天临时事实被刷成永久），违反 TTL 绝对语义。
    """

    if ttl_days and ttl_days > 0:
        days = float(ttl_days)
    elif settings.memory_default_ttl_days > 0:
        days = float(settings.memory_default_ttl_days)
    else:
        return None
    return time.time() + days * 24 * 3600


def _build_fact_value(
    text: str,
    category: str,
    now: float,
    source: str,
    *,
    supersedes: str | None = None,
    expires_ts: float | None = None,
) -> dict[str, Any]:
    """构造新 active 事实的 value（含阶段 1 软状态 schema）。"""

    value: dict[str, Any] = {
        "text": text,
        "category": category,
        "status": _FACT_STATUS_ACTIVE,
        "created_at": now,
        "updated_at": now,
        # 业务时间轴：事实自何时起生效（as-of 查询的左边界）
        "valid_from": now,
        "source": source,
    }
    if supersedes:
        # 前向指针：新事实取代了哪个旧 key，召回时可低成本标注演变轨迹
        value["supersedes"] = supersedes
    if expires_ts is not None:
        value["expires_at"] = expires_ts
    return value


async def _supersede_fact(
    store: Any,
    ns: tuple[str, ...],
    old_item: Any,
    new_key: str,
    new_value: dict[str, Any],
    settings: Any,
    now: float,
) -> None:
    """把去重命中的旧事实标记为 superseded，并写入新 active 事实。

    取代链语义（不覆盖、不丢历史）：

    - 新事实继承旧事实的绝对过期时刻（临时事实被更新后仍是临时的）；
    - 旧事实保留原文，写入 ``superseded_by`` 反向指针，收尾 TTL 取
      "配置审计窗口"与"原 TTL 剩余时长"的较小者，到期由 sweeper 物理清除。
    """

    old_value: dict[str, Any] = dict(old_item.value or {})
    old_expires = old_value.get("expires_at")
    inherited_expires = (
        old_expires
        if isinstance(old_expires, (int, float)) and old_expires > now
        else None
    )

    # 1) 新 active 事实：继承绝对 TTL，避免临时事实被刷成永久
    put_kwargs: dict[str, Any] = {}
    if inherited_expires is not None:
        put_kwargs["ttl"] = (inherited_expires - now) / 60
        new_value["expires_at"] = inherited_expires
    await store.aput(ns, new_key, new_value, index=["text"], **put_kwargs)

    # 2) 旧版本软失效：审计窗口与原 TTL 谁先到听谁的
    closing_seconds = float(settings.memory_superseded_ttl_days) * 24 * 3600
    if inherited_expires is not None:
        closing_seconds = min(closing_seconds, inherited_expires - now)
    closing_expires = now + closing_seconds
    old_value["status"] = _FACT_STATUS_SUPERSEDED
    old_value["superseded_by"] = new_key
    # 业务时间轴右边界：旧事实自此刻起失效（as-of 查询在此时刻之后不再命中它）
    old_value["invalid_at"] = now
    old_value["updated_at"] = now
    old_value["expires_at"] = closing_expires
    await store.aput(
        ns,
        old_item.key,
        old_value,
        index=["text"],
        ttl=closing_seconds / 60,
    )


# ── 核心实现（显式传 user_id，便于单测直接调用） ───────────────────


async def _save_memory(
    tenant_id: str,
    user_id: str,
    text: str,
    category: str = "general",
    ttl_days: int = 0,
    source: str = "manual",
) -> str:
    settings = get_settings()
    text = (text or "").strip()
    category = (category or "general").strip().lower()[:32]
    if category not in _ALLOWED_CATEGORIES:
        category = "general"
    logger.info(
        "save_memory 开始: tenant=%s user_id=%s, category=%s, ttl_days=%d, 长度=%d 字符",
        tenant_id, user_id, category, ttl_days, len(text),
    )

    if not text:
        return "保存失败：记忆内容为空。"
    if len(text) > settings.memory_fact_max_chars:
        logger.warning("save_memory 超长拒绝: %d > %d", len(text), settings.memory_fact_max_chars)
        return (
            f"保存失败：单条记忆最多 {settings.memory_fact_max_chars} 字符，"
            "请精简为一条独立事实后再保存。"
        )
    rule = scan_secret(text)
    if rule is not None:
        logger.warning(
            "save_memory 凭据拦截: tenant=%s user_id=%s 规则=%s",
            tenant_id, user_id, rule,
        )
        return f"保存失败：内容疑似凭据/密钥（{rule}），记忆中禁止保存此类信息。"

    store = db.get_store()
    ns = db.user_facts_namespace(tenant_id, user_id)

    async def _work() -> dict[str, Any]:
        # 租户级配额（tenants.max_facts，平台运营可配）与全局配置取较小者
        max_items = settings.memory_fact_max_items
        tenant = await db.get_tenant(tenant_id)
        if tenant is not None:
            tenant_max = tenant.get("max_facts")
            if isinstance(tenant_max, int) and tenant_max >= 0:
                max_items = min(max_items, tenant_max)

        # 全量条目（含 superseded 审计副本；配额校验 + hash 模式去重共用）。
        # 扫描窗口不能按配额截断，否则旧版本堆积会让 active 计数失真。
        all_items = await store.asearch(ns, limit=_FACT_SCAN_LIMIT)

        def _is_active(item: Any) -> bool:
            # 旧数据无 status 字段，缺省视为 active
            return str((item.value or {}).get("status", _FACT_STATUS_ACTIVE)) == _FACT_STATUS_ACTIVE

        # 1) 去重：仅 active 条目可被取代；superseded 旧版本不再参与判重，
        #    避免对同一历史反复建立取代链。
        #    bge-m3 等真语义嵌入走向量相似度；hash 嵌入余弦区分度差
        #    （共享高频 token 的不同事实也易超阈值，误合并=旧事实被误失效），
        #    退化为 token Jaccard——近逐字重复才判重。
        duplicate_item: Any | None = None
        if settings.memory_embedding_kind == "bge-m3":
            searched = await store.asearch(ns, query=text, limit=5)
            for item in searched:
                if not _is_active(item):
                    continue
                score = getattr(item, "score", None)
                if score is not None and score >= settings.memory_dedup_threshold:
                    duplicate_item = item
                    break
        else:
            best: tuple[float, Any] | None = None
            for item in all_items:
                if not _is_active(item):
                    continue
                old_text = str((item.value or {}).get("text", ""))
                score = _jaccard(text, old_text)
                if best is None or score > best[0]:
                    best = (score, item)
            if best is not None and best[0] >= settings.memory_dedup_threshold:
                duplicate_item = best[1]

        now = time.time()
        if duplicate_item is not None:
            # 取代链不额外消耗 active 配额：新 active 顶替旧 active，
            # 旧版本转为 superseded 审计副本（不计配额）。
            logger.info(
                "save_memory 命中去重: user_id=%s old_key=%s，建立取代链",
                user_id, duplicate_item.key,
            )
            new_key = _FACT_KEY_PREFIX + db.new_id()
            new_value = _build_fact_value(
                text, category, now, source, supersedes=duplicate_item.key
            )
            await _supersede_fact(
                store, ns, duplicate_item, new_key, new_value, settings, now
            )
            return {"action": "superseded", "key": new_key}

        # 2) 配额校验：只数 active，superseded 副本不占用户配额
        active_count = sum(1 for item in all_items if _is_active(item))
        if active_count >= max_items:
            logger.warning(
                "save_memory 达配额上限: user_id=%s active=%d 上限=%d",
                user_id, active_count, max_items,
            )
            return {"action": "quota_exceeded", "key": "", "max_items": max_items}

        # 3) 新建 active 条目（TTL 缺省时走 Store 默认策略；过期时刻冗余进
        #    value，供取代时折算剩余 TTL——Item 不暴露 expires_at 列）
        key = _FACT_KEY_PREFIX + db.new_id()
        value = _build_fact_value(
            text,
            category,
            now,
            source,
            expires_ts=_effective_expiry_ts(settings, ttl_days),
        )
        aput = partial(store.aput, ns, key, value, index=["text"])
        if ttl_days and ttl_days > 0:
            await aput(ttl=float(ttl_days) * 24 * 60)
        else:
            await aput()
        return {"action": "created", "key": key}

    try:
        with anyio.fail_after(settings.memory_timeout_seconds):
            result = await _memory_retry()(_work)()
    except TimeoutError:
        logger.error("save_memory 超时: user_id=%s", user_id)
        return "保存失败：记忆服务超时，请稍后重试。"
    except Exception as exc:
        logger.exception("save_memory 失败: user_id=%s", user_id)
        return f"保存失败：记忆服务暂时不可用（{type(exc).__name__}）。"

    if result["action"] == "quota_exceeded":
        return (
            f"保存失败：事实记忆已达上限 {result['max_items']} 条，"
            "请先用 forget_memory 删除过时条目。"
        )
    logger.info(
        "save_memory 完成: user_id=%s action=%s key=%s",
        user_id, result["action"], result["key"],
    )
    if result["action"] == "superseded":
        return (
            f"已保存记忆（superseded，旧版本已标记失效并保留审计副本），"
            f"memory_id={result['key']}。"
            "后续可通过 recall_memories 召回新版本，或用 forget_memory 删除。"
        )
    return (
        f"已保存记忆（{result['action']}），memory_id={result['key']}。"
        "后续可通过 recall_memories 召回，或用 forget_memory 删除。"
    )


async def _recall_memories(tenant_id: str, user_id: str, query: str, k: int | None = None) -> str:
    settings = get_settings()
    k = k or settings.memory_recall_top_k
    k = min(max(int(k), 1), 20)
    query = (query or "").strip()
    logger.info(
        "recall_memories 开始: tenant=%s user_id=%s, k=%d, query=%r",
        tenant_id, user_id, k, query[:80],
    )
    if not query:
        return "召回失败：查询内容为空。"

    store = db.get_store()
    ns = db.user_facts_namespace(tenant_id, user_id)

    async def _work() -> dict[str, Any]:
        if settings.memory_hybrid_retrieval_enabled:
            # 混合管线：dense 向量 + sparse BM25 并行召回 → RRF 融合 → recency 精排
            def _active_only(raw: dict[str, Any]) -> bool:
                # superseded 旧版本一律不进召回，避免模型引用已失效事实
                return (
                    str(raw.get("status", _FACT_STATUS_ACTIVE))
                    == _FACT_STATUS_ACTIVE
                )

            mode, facts = await _hybrid_collect(
                store, ns, query, k, settings, _active_only
            )
            return {"mode": mode, "facts": facts}

        # ── kill switch 关闭时的旧单通道行为（回退保障） ──
        if settings.memory_vector_enabled:
            # 放大检索池：superseded 旧版本会被过滤，按 k 直取可能导致有效结果不足
            items = await store.asearch(ns, query=query, limit=max(k * 4, 10))
            # 低于阈值的微小相似度（hash 噪声/向量近似正交）不作为命中；
            # superseded 旧版本一律不进召回，避免模型引用已失效事实
            min_score = settings.memory_recall_min_score
            facts = []
            for item in items:
                view = db._fact_to_view(item)
                if view.get("status") != _FACT_STATUS_ACTIVE:
                    continue
                if view.get("score") is not None and view["score"] >= min_score:
                    facts.append(view)
            facts = facts[:k]
            mode = "semantic"
        else:
            items = await store.asearch(ns, limit=_FACT_SCAN_LIMIT)
            active_views = [
                db._fact_to_view(item)
                for item in items
                if str((item.value or {}).get("status", _FACT_STATUS_ACTIVE))
                == _FACT_STATUS_ACTIVE
            ]
            facts = _keyword_rank(query, active_views)[:k]
            mode = "keyword"
        return {"mode": mode, "facts": facts}

    try:
        with anyio.fail_after(settings.memory_timeout_seconds):
            payload = await _memory_retry()(_work)()
    except TimeoutError:
        logger.error("recall_memories 超时: user_id=%s", user_id)
        return "召回失败：记忆服务超时，请稍后重试。"
    except Exception as exc:
        logger.exception("recall_memories 失败: user_id=%s", user_id)
        return f"召回失败：记忆服务暂时不可用（{type(exc).__name__}）。"

    facts = payload["facts"]
    logger.info(
        "recall_memories 完成: user_id=%s mode=%s 命中=%d",
        user_id, payload["mode"], len(facts),
    )
    if not facts:
        return "未检索到相关历史记忆。"
    lines = [f"[记忆检索 mode={payload['mode']}] 共 {len(facts)} 条："]
    for i, fact in enumerate(facts, 1):
        score = fact.get("score")
        score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        # 前向指针提示：该事实是某旧版本的取代者，模型可知存在历史演变
        supersede_text = (
            f", 已取代旧版本 id={fact['supersedes']}" if fact.get("supersedes") else ""
        )
        lines.append(
            f"{i}. {fact['text']} "
            f"[id={fact['key']}, category={fact['category']}{score_text}{supersede_text}]"
        )
    return "\n".join(lines)


# ── as-of 时点召回（双时间轴：valid_from / invalid_at） ──────────────


def _parse_as_of(as_of: str) -> float | None:
    """把 as_of 入参解析为 Unix 秒；无法识别返回 None。

    接受 ``YYYY-MM-DD``（按 UTC 零点）与 ISO 8601 日期时间；
    朴素时间（无时区后缀）一律按 UTC 解释，保证结果与时区配置无关。
    """

    text = (as_of or "").strip()
    if not text:
        return None
    # 日期粒度：补齐 UTC 零点，避免 fromisoformat 把朴素日期当成午夜本地时间
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        text = text + "T00:00:00+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed.timestamp()


def _fact_valid_at(raw: dict[str, Any], ts: float) -> bool:
    """判断事实在指定 Unix 秒时点是否有效（业务时间轴 + TTL 双约束）。"""

    # 旧数据无 valid_from 时回退 created_at；两者都缺则无法判定生效起点
    valid_from = raw.get("valid_from")
    if valid_from is None:
        valid_from = raw.get("created_at")
    if valid_from is None or float(valid_from) > ts:
        return False
    # 已被新事实取代：invalid_at 起（含该时刻）视为失效，与新事实 valid_from 无缝交接
    invalid_at = raw.get("invalid_at")
    if invalid_at is not None and float(invalid_at) <= ts:
        return False
    # TTL 已过期：到期时点之后同样不可作为当时的有效事实
    expires_at = raw.get("expires_at")
    if expires_at is not None and float(expires_at) <= ts:
        return False
    return True


async def _recall_memories_as_of(
    tenant_id: str, user_id: str, query: str, as_of: str, k: int | None = None
) -> str:
    """召回"截至某个历史时点"有效的事实（point-in-time 记忆重建）。

    与 :func:`_recall_memories` 的区别：后者只看当前 active 事实；本函数按
    valid_from/invalid_at/expires_at 过滤出 ``as_of`` 时点有效的版本，
    superseded 旧版本在其失效时点之前仍可被命中。注意：超过审计窗口
    （memory_superseded_ttl_days）被 sweeper 物理删除的旧版本无法重建。
    """

    settings = get_settings()
    k = k or settings.memory_recall_top_k
    k = min(max(int(k), 1), 20)
    query = (query or "").strip()
    if not query:
        return "召回失败：查询内容为空。"
    ts = _parse_as_of(as_of)
    if ts is None:
        return (
            "召回失败：as_of 时间格式无法识别，请使用 YYYY-MM-DD（如 2026-09-01）"
            "或 ISO 8601 日期时间（如 2026-09-01T12:00:00+00:00）。"
        )

    store = db.get_store()
    ns = db.user_facts_namespace(tenant_id, user_id)
    as_of_iso = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC).isoformat()

    async def _work() -> dict[str, Any]:
        if settings.memory_hybrid_retrieval_enabled:
            # 与当前召回同一套混合管线，仅谓词换成 as-of 时点有效性
            mode, facts = await _hybrid_collect(
                store, ns, query, k, settings,
                lambda raw: _fact_valid_at(raw, ts),
            )
            return {"mode": mode, "facts": facts}

        # ── kill switch 关闭时的旧单通道行为（回退保障） ──
        if settings.memory_vector_enabled:
            # 放大检索池后再用时点谓词过滤（superseded 版本也在候选中）
            items = await store.asearch(ns, query=query, limit=max(k * 4, 10))
            min_score = settings.memory_recall_min_score
            facts = []
            for item in items:
                raw = item.value or {}
                if not _fact_valid_at(raw, ts):
                    continue
                view = db._fact_to_view(item)
                score = view.get("score")
                if score is not None and score < min_score:
                    continue
                facts.append(view)
            facts = facts[:k]
            mode = "semantic"
        else:
            items = await store.asearch(ns, limit=_FACT_SCAN_LIMIT)
            valid_views = [
                db._fact_to_view(item)
                for item in items
                if _fact_valid_at(item.value or {}, ts)
            ]
            facts = _keyword_rank(query, valid_views)[:k]
            mode = "keyword"
        return {"mode": mode, "facts": facts}

    try:
        with anyio.fail_after(settings.memory_timeout_seconds):
            payload = await _memory_retry()(_work)()
    except TimeoutError:
        logger.error("recall_memories_as_of 超时: user_id=%s", user_id)
        return "召回失败：记忆服务超时，请稍后重试。"
    except Exception as exc:
        logger.exception("recall_memories_as_of 失败: user_id=%s", user_id)
        return f"召回失败：记忆服务暂时不可用（{type(exc).__name__}）。"

    facts = payload["facts"]
    logger.info(
        "recall_memories_as_of 完成: user_id=%s as_of=%s mode=%s 命中=%d",
        user_id, as_of_iso, payload["mode"], len(facts),
    )
    if not facts:
        return f"截至 {as_of_iso} 未检索到相关历史记忆。"
    lines = [
        f"[记忆检索 mode={payload['mode']} as_of={as_of_iso}] 共 {len(facts)} 条："
    ]
    for i, fact in enumerate(facts, 1):
        score = fact.get("score")
        score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        window = f"生效于 {fact.get('valid_from') or '未知'}"
        if fact.get("invalid_at"):
            window += f"，失效于 {fact['invalid_at']}"
        lines.append(
            f"{i}. {fact['text']} [id={fact['key']}, category={fact['category']}"
            f"{score_text}, {window}]"
        )
    return "\n".join(lines)


async def _forget_memory(tenant_id: str, user_id: str, memory_id: str) -> str:
    memory_id = (memory_id or "").strip()
    logger.info(
        "forget_memory 开始: tenant=%s user_id=%s memory_id=%s",
        tenant_id, user_id, memory_id,
    )
    if not memory_id:
        return "遗忘失败：必须提供 memory_id。"
    # 只接受本系统生成的 fact key，杜绝畸形输入
    if not memory_id.startswith(_FACT_KEY_PREFIX):
        memory_id = _FACT_KEY_PREFIX + memory_id

    settings = get_settings()

    async def _work() -> bool:
        return await db.delete_user_fact(tenant_id, user_id, memory_id)

    try:
        with anyio.fail_after(settings.memory_timeout_seconds):
            deleted = await _memory_retry()(_work)()
    except TimeoutError:
        logger.error("forget_memory 超时: user_id=%s", user_id)
        return "遗忘失败：记忆服务超时，请稍后重试。"
    except Exception as exc:
        logger.exception("forget_memory 失败: user_id=%s", user_id)
        return f"遗忘失败：记忆服务暂时不可用（{type(exc).__name__}）。"

    logger.info("forget_memory 完成: user_id=%s key=%s deleted=%s", user_id, memory_id, deleted)
    if deleted:
        return f"已删除记忆 {memory_id}，后续回答不再使用该信息。"
    return f"未找到记忆 {memory_id}（可能已过期或已删除）。"


# ── LangChain 工具（薄包装：注入运行时身份） ───────────────────────


@tool
async def save_memory(text: str, category: str = "general", ttl_days: int = 0) -> str:
    """将一条跨会话有价值的用户事实保存到长期记忆（语义去重、可召回、可遗忘）。

    适用：用户身份角色、明确的回答/沟通偏好、项目背景、已定结论、对助手的纠错。
    不适用：密码/密钥/令牌等凭据（会被拒绝）、一次性任务细节、与用户无关的临时信息。

    Args:
        text: 单条事实，一句话陈述，不超过 1000 字符（实际上限以系统配置为准）。
        category: 可选分类：general/identity/preference/project/decision/feedback。
        ttl_days: 可选保留天数；0 表示按系统默认策略（默认永久保留）。
            按绝对时间计算（读取不续期），过期后由后台清扫物理删除，
            可能存在一个清扫周期的滞留。
    """

    tenant_id, user_id = _current_identity()
    return await _save_memory(tenant_id, user_id, text, category, ttl_days)


@tool
async def recall_memories(query: str, k: int = 5) -> str:
    """按语义相似度检索当前用户的历史长期记忆，回答个性化问题前应先调用。

    Args:
        query: 检索语句（自然语言问题或关键事实描述）。
        k: 返回条数（1~20，默认 5）。
    """

    tenant_id, user_id = _current_identity()
    return await _recall_memories(tenant_id, user_id, query, k)


@tool
async def recall_memories_as_of(query: str, as_of: str, k: int = 5) -> str:
    """按语义相似度检索"截至某个历史时点"对当前用户有效的长期记忆。

    用于回答"上个月/当时我告诉过你什么"类需要还原历史状态的问题：
    已被取代的旧版本在其失效时点之前仍会被召回，失效时点之后只返回新版本。

    Args:
        query: 检索语句（自然语言问题或关键事实描述）。
        as_of: 目标时点，接受 YYYY-MM-DD（如 2026-09-01，按 UTC 零点）
            或 ISO 8601 日期时间。
        k: 返回条数（1~20，默认 5）。

    注意：超过系统审计窗口被物理清除的旧版本无法重建；
    需要当前有效事实时优先使用 recall_memories。
    """

    tenant_id, user_id = _current_identity()
    return await _recall_memories_as_of(tenant_id, user_id, query, as_of, k)


@tool
async def forget_memory(memory_id: str) -> str:
    """按 memory_id 删除当前用户的一条长期记忆（用户明确要求"忘掉某事"时调用）。

    Args:
        memory_id: save_memory/recall_memories 返回条目中的 id。
    """

    tenant_id, user_id = _current_identity()
    return await _forget_memory(tenant_id, user_id, memory_id)


MEMORY_TOOLS = [save_memory, recall_memories, recall_memories_as_of, forget_memory]
