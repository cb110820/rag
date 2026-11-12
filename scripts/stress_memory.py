"""长期记忆层并发压测脚本（无 LLM，hash 嵌入）。

在临时 SQLite 库上模拟多用户高并发的 save/recall 混合负载，验证：

1. 无写入丢失：每用户成功保存的事实条目数与预期一致；
2. 无跨用户泄漏：每个用户的专属标记只能被本人召回；
3. 无未捕获异常/超时；
4. 输出 save/recall 的 p50/p95/p99 延迟与吞吐。

用法：

    python scripts/stress_memory.py --users 20 --ops-per-user 20 --concurrency 32

退出码：存在错误、丢失或泄漏时为 1，否则 0。
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
import uuid
from pathlib import Path

from src.memory import embedding as memory_embedding
from src.memory import tools as memory_tools
from src.persistence import database as db
from src.rag import config as config_module
from src.rag.config import Settings

# 压测用户统一归属 default 租户（多租户签名：tenant_id + user_id）
_TENANT = db.DEFAULT_TENANT_ID

# 每条形如“前缀 + 4 个互不相交的汉字”，避免 hash 字符集合向量因前缀相同
# 而触发去重（Jaccard≥0.9）把多条压测事实误判为同一条；池容量支持 40 条/人
_UNIQUE_CHUNKS = [
    "春夏秋冬", "东南西北", "金木水火", "风雨雷电", "日月星辰",
    "山川河海", "花鸟鱼虫", "梅兰竹菊", "琴棋书画", "笔墨纸砚",
    "亭台楼阁", "锅碗瓢盆", "油盐酱醋", "酸甜苦辣", "喜怒哀乐",
    "生死祸福", "兴衰荣辱", "进退取舍", "攻守取舍", "成败得失",
    "龙凤麟龟", "麒麟貔貅", "松柏杨柳", "桃李杏梨", "稻麦黍稷",
    "蚕蜂蝶蚁", "燕雀鹰隼", "豺狼虎豹", "鲸鲨鳄豚", "鹿兔狐貂",
    "丝绸棉麻", "砖瓦灰石", "车船舟马", "弓箭刀枪", "钟鼓锣钹",
    "诗词歌赋", "经史子集", "仁义礼智", "温良恭俭", "忠孝廉耻",
]



async def _patch_settings() -> Settings:
    settings = Settings(
        data_dir=Path(tempfile.mkdtemp(prefix="mem-stress-")) / "data",
        jwt_secret_key="stress-secret-0123456789-0123456789-0123456789",
        rag_embedding="hash",
        memory_embedding="hash",
        ttl_sweep_interval_minutes=0,
    )
    for module in (config_module, db, memory_tools, memory_embedding):
        module.get_settings = lambda: settings  # type: ignore[attr-defined]
    await db.init_db()
    return settings


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(samples)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, round(len(ordered) * p) - 1)
        return round(ordered[max(idx, 0)] * 1000, 1)

    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}


async def _user_worker(
    user_id: str, marker: str, ops: int, latencies: dict[str, list[float]]
) -> tuple[int, list[str]]:
    """单用户混合负载：一半 save 一半 recall，返回成功保存数与错误列表。"""

    errors: list[str] = []
    saved = 0
    sem = asyncio.Semaphore(64)

    async def _run_one(i: int) -> None:
        nonlocal saved
        async with sem:
            if i % 2 == 0:
                chunk = _UNIQUE_CHUNKS[(i // 2) % len(_UNIQUE_CHUNKS)]
                text = f"{marker} 专属事实 {chunk} 第{i}条"
                start = time.perf_counter()
                out = await memory_tools._save_memory(_TENANT, user_id, text, "general")
                latencies["save"].append(time.perf_counter() - start)
                if "已保存" in out and "updated" not in out:
                    saved += 1
                elif "已保存" not in out:
                    errors.append(f"save: {out}")
            else:
                start = time.perf_counter()
                out = await memory_tools._recall_memories(
                    _TENANT, user_id, f"{marker} 专属事实", 5
                )
                latencies["recall"].append(time.perf_counter() - start)
                if out.startswith("召回失败"):
                    errors.append(f"recall: {out}")

    await asyncio.gather(*[_run_one(i) for i in range(ops)])
    return saved, errors


async def main_async(users: int, ops: int) -> int:
    await _patch_settings()
    try:
        latencies: dict[str, list[float]] = {"save": [], "recall": []}
        user_specs = [
            (uuid.uuid4().hex, f"标记{idx:03d}-{uuid.uuid4().hex[:8]}")
            for idx in range(users)
        ]

        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            _user_worker(uid, marker, ops, latencies) for uid, marker in user_specs
        ])
        elapsed = time.perf_counter() - t0

        total_saved = sum(r[0] for r in results)
        errors = [e for _, errs in results for e in errs]

        # ── 完整性校验：条目数一致 ──
        missing: list[str] = []
        # worker 中 i%2==0 走 save（i 从 0 开始）
        expected_saves = sum(1 for i in range(ops) if i % 2 == 0)
        for uid, marker in user_specs:
            facts = await db.list_user_facts(_TENANT, uid)
            if len(facts) != expected_saves:
                missing.append(
                    f"{marker}: 期望 {expected_saves} 实际 {len(facts)}"
                )

        # ── 隔离校验：每个用户的标记只在本人结果中出现 ──
        leaks: list[str] = []

        async def _check_leak(idx: int, uid: str, marker: str) -> None:
            out = await memory_tools._recall_memories(
                _TENANT, uid, f"{marker} 专属事实", 10
            )
            # 本人必须能召回自己的标记
            if marker not in out:
                leaks.append(f"{marker}: 本人未召回")
            # 其他人的标记不应出现
            for other_idx, (_, other_marker) in enumerate(user_specs):
                if other_idx != idx and other_marker in out:
                    leaks.append(f"{marker}: 泄漏了 {other_marker}")

        await asyncio.gather(*[
            _check_leak(i, uid, marker)
            for i, (uid, marker) in enumerate(user_specs)
        ])

        total_ops = users * ops
        print("\n========== 记忆层压测结果 ==========")
        print(f"用户数: {users}  每人操作: {ops}  总操作: {total_ops}")
        print(f"总耗时: {elapsed:.2f}s  吞吐: {total_ops / elapsed:.1f} ops/s")
        print(f"save   : n={len(latencies['save'])}  {_percentiles(latencies['save'])} ms")
        print(f"recall : n={len(latencies['recall'])}  {_percentiles(latencies['recall'])} ms")
        print(f"成功新建事实: {total_saved}")
        print(f"运行错误: {len(errors)}  条目丢失: {len(missing)}  隔离泄漏: {len(leaks)}")
        for sample in (errors[:3] + missing[:3] + leaks[:3]):
            print(f"  ! {sample}")
        print("====================================")
        return 0 if not errors and not missing and not leaks else 1
    finally:
        await db.close_db()


def main() -> int:
    parser = argparse.ArgumentParser(description="记忆层并发压测")
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--ops-per-user", type=int, default=20)
    args = parser.parse_args()
    return asyncio.run(main_async(args.users, args.ops_per_user))


if __name__ == "__main__":
    raise SystemExit(main())
