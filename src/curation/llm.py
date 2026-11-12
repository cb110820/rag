"""LLM 结构化提取器：文档全文 → 摘要 + 实体 + 矛盾信号。

约束：
- 模型 ``streaming=False``（ChatTongyi 流式 tool_calls 越界 bug）；
- 输出强制 JSON，解析/校验失败统一抛 ``CurationExtractError``，
  调用方据此保证"故障不留半成品"；
- 提取器可注入（测试用 fake，离线可跑）。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.rag.config import get_settings

logger = logging.getLogger(__name__)

# 实体类型白名单（LLM 输出与中文标签都接受）
_TYPE_ALIASES: dict[str, str] = {
    "person": "person", "人物": "person",
    "concept": "concept", "概念": "concept",
    "org": "org", "机构": "org", "organization": "org",
    "tool": "tool", "工具": "tool",
}

# 喂给 LLM 的既有结论长度上限（每个实体），控制 token
_EXISTING_CONCLUSION_LIMIT = 600


class CurationExtractError(RuntimeError):
    """沉淀提取失败：模型异常、超时或输出不可解析。"""


@dataclass
class ExistingEntity:
    """已存在实体页的匹配信息（仅传本次文档可能涉及的实体）。"""

    slug: str
    name: str
    aliases: list[str]
    conclusion: str  # 既有页正文结论（已截断）


@dataclass
class EntityExtraction:
    name: str
    type: str  # person/concept/org/tool
    aliases: list[str] = field(default_factory=list)
    summary: str = ""
    points: list[str] = field(default_factory=list)
    # 与既有页冲突时的新说法；None 表示无冲突
    contradiction: str | None = None


@dataclass
class CurationExtraction:
    summary: str
    key_points: list[str]
    entities: list[EntityExtraction]


class Extractor(Protocol):
    async def __call__(
        self,
        document_text: str,
        source_id: str,
        existing_entities: list[ExistingEntity],
    ) -> CurationExtraction: ...


# ── prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是企业知识库的 Wiki 沉淀引擎。通读文档原文，把知识提炼为结构化 JSON。

铁律：
1. 只能依据原文，严禁编造原文没有的事实、数字或结论；信息不足时留空，不要猜测。
2. 只输出一个 JSON 对象，不要输出 Markdown、解释或代码块围栏之外的文字。
3. 摘要与要点使用与文档相同的语言；简洁、准确、可回溯。
4. 实体类型 type 只能是 person（人物）、concept（概念）、org（机构）、tool（工具）四类之一。
5. 实体数量最多 {max_entities} 个，只抽取文档中真正重要、跨文档可复用的实体。
6. 对于"既有实体结论"：若原文说法与既有结论一致或只是补充，contradiction 设为 null；
   若原文给出了与既有结论直接冲突的说法（如不同数字、相反规定），contradiction 写明原文的新说法，
   并在 aliases 中给出该实体的其他称呼；不得改写或否定既有结论。
7. JSON 结构：
{{
  "summary": "文档整体摘要，100-300 字",
  "key_points": ["关键要点1", "关键要点2"],
  "entities": [
    {{
      "name": "实体规范名",
      "type": "concept",
      "aliases": ["别名1"],
      "summary": "该实体在本文档中的概述",
      "points": ["事实要点1"],
      "contradiction": null
    }}
  ]
}}
没有可抽取实体时 entities 输出空数组。"""


def build_user_prompt(
    document_text: str, source_id: str, existing: list[ExistingEntity]
) -> str:
    parts = [f"文档来源：{source_id}", "", "文档原文：", document_text]
    if existing:
        parts += ["", "既有实体结论（仅判断是否冲突，不得改写）："]
        for ent in existing:
            alias = "、".join(ent.aliases) if ent.aliases else "（无别名）"
            conclusion = ent.conclusion[:_EXISTING_CONCLUSION_LIMIT]
            parts.append(f"- {ent.name}（别名：{alias}）：{conclusion}")
    parts.append("")
    parts.append("请按系统消息要求只输出 JSON。")
    return "\n".join(parts)


# ── JSON 解析与校验 ──────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_extraction(payload: Any, max_entities: int) -> CurationExtraction:
    """把 LLM 返回的 JSON（dict 或字符串）解析为 CurationExtraction。"""

    if isinstance(payload, str):
        text = payload.strip()
        fence = _FENCE_RE.search(text)
        if fence:
            text = fence.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 容错：截取首个 {...} 块
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise CurationExtractError("模型输出中找不到 JSON 对象") from None
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise CurationExtractError(f"模型输出 JSON 无法解析: {exc}") from exc
    else:
        data = payload

    if not isinstance(data, dict):
        raise CurationExtractError("模型输出 JSON 顶层不是对象")

    summary = str(data.get("summary") or "").strip()
    raw_points = data.get("key_points") or []
    if not isinstance(raw_points, list):
        raise CurationExtractError("key_points 必须是数组")
    key_points = [str(p).strip() for p in raw_points if str(p).strip()]

    raw_entities = data.get("entities") or []
    if not isinstance(raw_entities, list):
        raise CurationExtractError("entities 必须是数组")

    entities: list[EntityExtraction] = []
    seen_names: set[str] = set()
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        etype = _TYPE_ALIASES.get(str(item.get("type") or "").strip().lower(), "concept")
        aliases = [
            str(a).strip()
            for a in (item.get("aliases") or [])
            if str(a).strip() and str(a).strip() != name
        ]
        esummary = str(item.get("summary") or "").strip()
        epoints = [
            str(p).strip() for p in (item.get("points") or []) if str(p).strip()
        ] if isinstance(item.get("points") or [], list) else []
        contradiction = item.get("contradiction")
        contradiction = str(contradiction).strip() if contradiction else None
        entities.append(
            EntityExtraction(
                name=name,
                type=etype,
                aliases=aliases,
                summary=esummary,
                points=epoints,
                contradiction=contradiction,
            )
        )

    entities = entities[:max_entities]  # 过滤清洗后再截断

    if not summary and not key_points and not entities:
        raise CurationExtractError("模型输出为空摘要/空要点/空实体")
    return CurationExtraction(summary=summary, key_points=key_points, entities=entities)


# ── 默认提取器（真实模型） ───────────────────────────────────────────


async def extract_curated(
    document_text: str,
    source_id: str,
    existing_entities: list[ExistingEntity],
    *,
    model: Any | None = None,
    max_entities: int | None = None,
) -> CurationExtraction:
    """调用主模型（streaming=False）做结构化提取；任何失败转 CurationExtractError。"""

    settings = get_settings()
    limit = max_entities or settings.wiki_curation_max_entities
    if model is None:
        from src.agents.graph import resolve_chat_model

        model_spec = settings.wiki_curation_model or None
        model = resolve_chat_model(model_spec, streaming=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_entities=limit)},
        {"role": "user", "content": build_user_prompt(document_text, source_id, existing_entities)},
    ]
    try:
        response = await model.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        return parse_extraction(content, limit)
    except CurationExtractError:
        raise
    except Exception as exc:  # noqa: BLE001 - 模型/网络异常统一转型，保证上游可判定
        logger.warning("Wiki 沉淀提取失败 source=%s: %s", source_id, exc)
        raise CurationExtractError(str(exc)) from exc
