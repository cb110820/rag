"""curation.llm 提取器测试：JSON 容错、故障转型、prompt 约束。"""

from __future__ import annotations

import json

import pytest

from src.curation import llm as cl
from src.curation.llm import (
    CurationExtractError,
    CurationExtraction,
    ExistingEntity,
    extract_curated,
    parse_extraction,
)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    def __init__(self, content: str | None = None, exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.calls: list[list[dict]] = []

    async def ainvoke(self, messages: list[dict]) -> _FakeResponse:
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return _FakeResponse(self.content or "")


# ── parse_extraction 容错 ───────────────────────────────────────────


def test_parse_valid_and_clamp() -> None:
    payload = {
        "summary": "摘要",
        "key_points": ["p1", " ", "p2"],
        "entities": [
            {"name": "X", "type": "概念", "aliases": ["x"], "summary": "s", "points": ["a"]},
            {"name": "", "type": "person"},  # 空名丢弃
            {"name": "X", "type": "tool"},  # 重名丢弃
        ]
        + [{"name": f"E{i}", "type": "tool"} for i in range(20)],
    }
    result = parse_extraction(payload, max_entities=3)
    assert result.summary == "摘要"
    assert result.key_points == ["p1", "p2"]
    assert len(result.entities) == 3  # 截断到 max_entities
    assert result.entities[0].name == "X"
    assert result.entities[0].type == "concept"  # 中文类型映射
    assert result.entities[0].aliases == ["x"]


def test_parse_fenced_json_string() -> None:
    text = "```json\n" + json.dumps({"summary": "s", "key_points": [], "entities": []}) + "\n```"
    result = parse_extraction(text, 10)
    assert result.summary == "s"


def test_parse_bad_json_raises() -> None:
    with pytest.raises(CurationExtractError):
        parse_extraction("完全不是 JSON", 10)
    with pytest.raises(CurationExtractError):
        parse_extraction("{}", 10)  # 全空也算失败


@pytest.mark.parametrize(
    "content,exc",
    [
        ("not json at all", None),
        (None, TimeoutError("model timeout")),
        (None, RuntimeError("api key invalid")),
    ],
)
async def test_extract_failures_wrapped(content, exc) -> None:
    model = _FakeModel(content=content, exc=exc)
    with pytest.raises(CurationExtractError):
        await extract_curated("正文", "a.md", [], model=model, max_entities=5)


async def test_extract_success_passes_existing() -> None:
    payload = json.dumps(
        {"summary": "S", "key_points": ["k"],
         "entities": [{"name": "X", "type": "concept", "contradiction": "新说法 30 天"}]}
    )
    model = _FakeModel(content=payload)
    existing = [ExistingEntity(slug="entity-x", name="X", aliases=["x"], conclusion="7 天")]
    result = await extract_curated("原文", "a.md", existing, model=model, max_entities=5)
    assert isinstance(result, CurationExtraction)
    assert result.entities[0].contradiction == "新说法 30 天"
    # 既有结论进入了 user prompt
    user_msg = model.calls[0][1]["content"]
    assert "既有实体结论" in user_msg and "7 天" in user_msg


def test_prompt_contains_hard_constraints() -> None:
    """TR-3.3 证据：prompt 必须覆盖忠实/禁编造/JSON/类型/数量/冲突语义。"""

    system = cl.SYSTEM_PROMPT.format(max_entities=10)
    for phrase in ["严禁编造", "只输出一个 JSON", "person", "concept", "org", "tool",
                   "最多 10 个", "contradiction", "一致或只是补充", "直接冲突"]:
        assert phrase in system
