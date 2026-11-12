"""LangMem 后台反思提取：对话结束后异步从对话轨迹提炼用户事实记忆。

与热路径（模型在对话中主动调 save_memory）的分工：

- 热路径依赖模型自觉，覆盖用户显式要求"记住"的场景；
- 本模块在每轮对话结束后兜底，用 LangMem ``create_memory_manager`` 从对话
  轨迹中自动提炼跨会话有价值的事实（身份/偏好/项目/结论），主代理生成
  答案时无需分心调用记忆工具。

安全红线（与热路径一致）：

- 所有候选事实一律经由 ``tools._save_memory`` 写入——凭据扫描、长度/配额
  校验、语义去重、超时重试全量生效，LangMem 只做提取，不直接接触 Store；
- 提取失败（LLM 欠费/超时/解析异常）只记日志，绝不影响主对话链路；
- 同一用户同一时间只跑一个提取任务，避免高频对话堆积 LLM 调用。

去重策略：不向提取器回传已有记忆（省一次 Store 读与 prompt token），
重复/近重复事实由写入层的语义去重原地合并（更新而非重复新建）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import anyio

from src.memory.tools import _save_memory
from src.rag.config import get_settings

logger = logging.getLogger("src.memory")

# 提取指令：只提炼与用户相关的持久事实；输出独立单句（与 save_memory
# 的单条事实约定一致）；明确禁止凭据与一次性细节，宁可不提取也不错录。
_REFLECTION_INSTRUCTIONS = (
    "你是长期记忆提取器。从对话中只提炼跨会话有价值的用户事实："
    "身份角色、稳定的沟通/回答偏好、项目背景、已确认的决策与结论、"
    "对助手的明确纠错。"
    "每条事实写成一句独立、完整、可脱离上下文理解的中文陈述。"
    "不要记录：密码/密钥/令牌等任何凭据、一次性任务细节、寒暄、"
    "与用户无关的通用知识、未经用户确认的猜测。"
    "没有值得记录的内容时不要产出任何记忆。"
)

# (tenant_id, user_id) → 运行中的提取任务，用于同人并发去重
_running: dict[tuple[str, str], asyncio.Task] = {}

# 消息 type/role → langmem 对话 role
_ROLE_MAP = {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}


def _coerce_messages(messages: list[Any], max_messages: int) -> list[dict[str, str]]:
    """把 LangGraph 消息对象裁剪为 langmem 可接受的 role/content 字典。

    只保留 human/ai 的文本消息：tool/system 消息对事实提炼无价值且徒增
    token；content 为块列表（多模态）时拼接其中的 text 块。
    """

    conversation: list[dict[str, str]] = []
    for msg in messages[-max_messages:]:
        if isinstance(msg, dict):
            role, content = msg.get("role"), msg.get("content")
        else:
            role, content = getattr(msg, "type", None), getattr(msg, "content", None)
        if isinstance(content, list):  # 多模态块列表：只取文本块
            content = "\n".join(
                str(b.get("text", "")) for b in content if isinstance(b, dict)
            )
        mapped = _ROLE_MAP.get(str(role))
        if mapped is None or not isinstance(content, str) or not content.strip():
            continue
        conversation.append({"role": mapped, "content": content})
    return conversation


async def extract_candidate_facts(
    conversation: list[dict[str, str]], model: Any = None
) -> list[str]:
    """调用 LangMem MemoryManager 提取候选事实文本（纯提取，不写库）。

    Args:
        conversation: ``_coerce_messages`` 产出的 user/assistant 对话。
        model: 提取用聊天模型；None 时按配置解析（默认跟随主模型）。
    """

    from langmem import create_memory_manager  # 延迟导入，降低模块耦合

    if model is None:
        from src.agents.graph import resolve_chat_model

        # 工具调用场景关闭流式：ChatTongyi 流式增量 tool_calls 会触发越界 bug
        model = resolve_chat_model(
            get_settings().memory_reflection_model or None, streaming=False
        )
    manager = create_memory_manager(
        model,
        instructions=_REFLECTION_INSTRUCTIONS,
        # 不回传已有记忆（去重在写入层完成），只允许新增
        enable_updates=False,
        enable_deletes=False,
    )
    import time

    started = time.perf_counter()
    try:
        extracted = await manager.ainvoke({"messages": conversation, "max_steps": 1})
    except Exception:
        logger.exception("LangMem manager 调用异常")
        raise
    logger.info(
        "LangMem manager 返回: 耗时 %.2fs, 提取 %d 条",
        time.perf_counter() - started, len(extracted or []),
    )
    facts: list[str] = []
    for item in extracted or []:
        # ExtractedMemory(id, content)；默认 schema Memory 的事实文本在 .content
        text = str(getattr(item.content, "content", "")).strip()
        if text:
            facts.append(text)
    return facts


async def reflect_and_store(
    tenant_id: str, user_id: str, messages: list[Any], model: Any = None
) -> dict[str, int]:
    """后台反思主流程：提取 → 逐条走 _save_memory 护栏写入，返回统计。

    提取阶段失败（LLM 不可用等）被就地吞掉并记日志；写入阶段的单条失败
    不影响其余候选。返回 ``{"extracted", "saved", "skipped", "failed"}``。
    """

    settings = get_settings()
    stats = {"extracted": 0, "saved": 0, "skipped": 0, "failed": 0}
    conversation = _coerce_messages(messages, settings.memory_reflection_max_messages)
    total_chars = sum(len(m["content"]) for m in conversation)
    logger.info(
        "反思提取开始: tenant=%s user_id=%s 消息=%d 条 / %d 字符",
        tenant_id, user_id, len(conversation), total_chars,
    )
    if not conversation or total_chars < settings.memory_reflection_min_chars:
        logger.info("反思提取跳过: 对话过短（%d 字符）", total_chars)
        return stats

    try:
        candidates = await extract_candidate_facts(conversation, model)
    except Exception:
        # LLM 欠费/超时/结构化输出失败：只记日志，主对话已正常结束
        logger.exception("反思提取失败（不影响主对话）: user_id=%s", user_id)
        return stats
    stats["extracted"] = len(candidates)
    logger.info("反思提取候选: user_id=%s %d 条", user_id, len(candidates))

    for text in candidates:
        try:
            # 统一走热路径同一写入实现：凭据扫描/长度/配额/语义去重全量生效；
            # source=reflection 标记来源，便于事后区分人工保存与反思沉淀
            result = await _save_memory(
                tenant_id, user_id, text, source="reflection"
            )
        except Exception:
            stats["failed"] += 1
            logger.exception("反思写入异常: user_id=%s", user_id)
            continue
        if result.startswith("已保存"):
            stats["saved"] += 1
        else:
            # 护栏拒绝（疑似凭据/超长/配额满）
            stats["skipped"] += 1
    logger.info("反思提取完成: user_id=%s 统计=%s", user_id, stats)
    return stats


async def _reflect_guarded(
    key: tuple[str, str], tenant_id: str, user_id: str, messages: list[Any]
) -> None:
    """带总超时与异常围栏的后台任务体；结束时释放并发槽位。"""

    try:
        with anyio.fail_after(get_settings().memory_reflection_timeout_seconds):
            await reflect_and_store(tenant_id, user_id, messages)
    except TimeoutError:
        logger.error(
            "反思提取超时: user_id=%s（>%ss）",
            user_id, get_settings().memory_reflection_timeout_seconds,
        )
    except Exception:
        logger.exception("反思提取异常（不影响主对话）: user_id=%s", user_id)
    finally:
        _running.pop(key, None)


def schedule_reflection(tenant_id: str, user_id: str, messages: list[Any]) -> bool:
    """在每轮对话完成后 fire-and-forget 调度后台提取，返回是否成功调度。

    - 开关关闭 / 该用户已有任务在跑时不调度（返回 False）；
    - 消息列表做浅快照，避免调用方后续修改影响后台任务。
    """

    if not get_settings().memory_reflection_enabled:
        return False
    key = (tenant_id, user_id)
    if key in _running:
        logger.info("反思提取跳过: 该用户已有任务在运行 user_id=%s", user_id)
        return False
    task = asyncio.create_task(
        _reflect_guarded(key, tenant_id, user_id, list(messages))
    )
    _running[key] = task
    logger.info("反思提取已调度: tenant=%s user_id=%s", tenant_id, user_id)
    return True
