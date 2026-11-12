"""长期记忆层离线评估（无 LLM）。

直连 Store 与记忆工具，验证记忆系统的结构性契约（不依赖模型行为）：

- 跨会话改写召回（语义/关键词两种模式各跑一遍）；
- 多租户 namespace 隔离；
- 显式遗忘；
- 语义去重；
- 凭据护栏；
- TTL 过期清扫；
- 多事实场景的相关性筛选。

用法：

    python -m src.eval.run_memory                # hash 向量模式
    python -m src.eval.run_memory --mode both    # hash + none 关键词兜底

退出码：全部用例通过为 0，否则 1（可接入 CI 质量门禁）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from src.memory import embedding as memory_embedding
from src.memory import tools as memory_tools
from src.persistence import database as db
from src.rag import config as config_module
from src.rag.config import Settings, configure_logging

logger = logging.getLogger("src.memory.eval")

_GOLDEN = Path(__file__).with_name("memory_golden_dataset.json")


def _user_id(name: str) -> str:
    """逻辑用户名 → 确定性 32 位 hex 用户 id。"""

    return uuid.uuid5(uuid.NAMESPACE_DNS, f"memory-eval:{name}").hex


# 评估用虚拟用户统一归属 default 租户，验证 namespace 隔离仍按 tenant_id 维度生效
_TENANT = db.DEFAULT_TENANT_ID


class CaseFailure(Exception):
    """单个评估步骤的断言失败。"""


class MemoryEvalRunner:
    """按 golden dataset 的步骤序列驱动真实记忆工具。"""

    def __init__(self, cases: list[dict[str, Any]]) -> None:
        self.cases = cases
        self.aliases: dict[str, str] = {}

    async def run(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for case in self.cases:
            self.aliases = {}
            try:
                for step in case["steps"]:
                    await self._step(step)
                results.append({"id": case["id"], "passed": True})
                logger.info("[PASS] %s", case["id"])
            except CaseFailure as exc:
                results.append({"id": case["id"], "passed": False, "reason": str(exc)})
                logger.error("[FAIL] %s: %s", case["id"], exc)
        return results

    async def _step(self, step: dict[str, Any]) -> None:
        op = step["op"]
        handler = getattr(self, f"_op_{op}", None)
        if handler is None:
            raise CaseFailure(f"未知步骤类型: {op}")
        await handler(step)

    async def _op_save(self, step: dict[str, Any]) -> None:
        out = await memory_tools._save_memory(
            _TENANT, _user_id(step["user"]),
            step["text"],
            step.get("category", "general"),
            step.get("ttl_days", 0),
        )
        if "expect_reject" in step:
            kind = step["expect_reject"]
            markers = {
                "credential": "凭据",
                "oversize": "最多",
                "quota": "上限",
                "empty": "为空",
            }
            marker = markers.get(kind, kind)
            if marker not in out:
                raise CaseFailure(f"期望被 {kind} 拒绝，实际返回: {out}")
            return
        if "已保存" not in out:
            raise CaseFailure(f"保存失败: {out}")
        if "superseded" in out:
            action = "superseded"
        elif "updated" in out:
            # 兼容历史输出（阶段 1 前的原地更新动作名）
            action = "updated"
        else:
            action = "created"
        if step.get("expect_action") and step["expect_action"] != action:
            raise CaseFailure(
                f"期望动作 {step['expect_action']}，实际 {action}: {out}"
            )
        if "as" in step:
            self.aliases[step["as"]] = out.split("memory_id=")[1].split("。")[0]

    async def _op_recall(self, step: dict[str, Any]) -> None:
        out = await memory_tools._recall_memories(
            _TENANT, _user_id(step["user"]), step["query"], step.get("k", 5)
        )
        hit = "未检索到" not in out
        if step.get("expect_hit", True) != hit:
            raise CaseFailure(
                f"命中状态不符（期望 expect_hit={step.get('expect_hit', True)}）: {out}"
            )
        for expected in _as_list(step.get("expect_contains")):
            if expected not in out:
                raise CaseFailure(f"召回结果缺少 {expected!r}: {out}")
        for forbidden in _as_list(step.get("must_not_contain")):
            if forbidden in out:
                raise CaseFailure(f"召回结果不应包含 {forbidden!r}: {out}")

    async def _op_forget(self, step: dict[str, Any]) -> None:
        key = self.aliases[step["ref"]]
        out = await memory_tools._forget_memory(_TENANT, _user_id(step["user"]), key)
        if "已删除" not in out:
            raise CaseFailure(f"遗忘失败: {out}")

    async def _op_assert_count(self, step: dict[str, Any]) -> None:
        facts = await db.list_user_facts(_TENANT, _user_id(step["user"]))
        if len(facts) != step["count"]:
            raise CaseFailure(f"条目数期望 {step['count']}，实际 {len(facts)}")

    async def _op_assert_active_count(self, step: dict[str, Any]) -> None:
        # active 口径（superseded 审计副本不计）：召回与配额的真实口径
        facts = await db.list_active_user_facts(_TENANT, _user_id(step["user"]))
        if len(facts) != step["count"]:
            raise CaseFailure(f"active 条目数期望 {step['count']}，实际 {len(facts)}")

    async def _op_save_ttl_minutes(self, step: dict[str, Any]) -> None:
        key = "fact-" + db.new_id()
        await db.get_store().aput(
            db.user_facts_namespace(_TENANT, _user_id(step["user"])),
            key,
            {"text": step["text"], "category": step.get("category", "general")},
            index=["text"],
            ttl=float(step["ttl_minutes"]),
        )
        self.aliases[step["as"]] = key

    async def _op_sleep_seconds(self, step: dict[str, Any]) -> None:
        await asyncio.sleep(float(step["seconds"]))

    async def _op_sweep(self, step: dict[str, Any]) -> None:
        deleted = await db.get_store().sweep_ttl()
        logger.info("sweep_ttl 物理删除 %d 条", deleted)

    async def _op_assert_ref_exists(self, step: dict[str, Any]) -> None:
        key = self.aliases[step["ref"]]
        # ref 是 save_ttl_minutes 产生的，需要用户名；从步骤无法获得时，
        # golden 中该步骤必须带 user
        user = _user_id(step["user"]) if step.get("user") else None
        if user is None:
            raise CaseFailure("assert_ref_exists 步骤必须声明 user")
        item = await db.get_user_fact(_TENANT, user, key)
        exists = item is not None
        if exists != bool(step["exists"]):
            raise CaseFailure(f"ref={key} 存在性期望 {step['exists']}，实际 {exists}")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


async def run_suite(cases: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """在一个全新临时库上跑完整用例集。"""

    tmp = Path(tempfile.mkdtemp(prefix=f"mem-eval-{mode}-"))
    settings = Settings(
        data_dir=tmp / "data",
        jwt_secret_key="memory-eval-secret-0123456789-0123456789-0123",
        rag_embedding="hash",
        memory_embedding=mode,
        ttl_sweep_interval_minutes=0,
    )

    # 不依赖 pytest monkeypatch：直接替换模块级 get_settings
    originals = (
        config_module.get_settings,
        db.get_settings,
        memory_tools.get_settings,
        memory_embedding.get_settings,
    )
    for module in (config_module, db, memory_tools, memory_embedding):
        module.get_settings = lambda: settings  # type: ignore[attr-defined]

    try:
        await db.init_db()
        results = await MemoryEvalRunner(cases).run()
    finally:
        await db.close_db()
        config_module.get_settings, db.get_settings = originals[0], originals[1]
        memory_tools.get_settings = originals[2]
        memory_embedding.get_settings = originals[3]

    passed = sum(1 for r in results if r["passed"])
    return {
        "mode": mode,
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 4),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="长期记忆层无 LLM 评估")
    parser.add_argument("--golden", type=Path, default=_GOLDEN)
    parser.add_argument(
        "--mode",
        choices=("hash", "none", "both"),
        default="both",
        help="hash=向量语义召回；none=无索引关键词兜底",
    )
    args = parser.parse_args()
    configure_logging()

    cases = json.loads(args.golden.read_text(encoding="utf-8"))
    modes = ["hash", "none"] if args.mode == "both" else [args.mode]

    reports = [asyncio.run(run_suite(cases, mode)) for mode in modes]
    print(json.dumps(reports, ensure_ascii=False, indent=2))

    all_passed = all(r["failed"] == 0 for r in reports)
    logger.info("记忆评估质量门禁：%s", "通过" if all_passed else "未达标")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
