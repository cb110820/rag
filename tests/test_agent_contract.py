from src.agents.prompts import MAIN_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT, SUMMARY_PROMPT
from src.agents.subagents import build_rag_subagent


def test_subagent_is_retrieval_only() -> None:
    subagent = build_rag_subagent("openai:test-model")
    assert subagent["name"] == "rag_retriever"
    assert len(subagent["tools"]) == 1
    assert "不做最终决策" in subagent["system_prompt"]
    assert "来源" in subagent["system_prompt"]


def test_supervisor_requires_sources_and_uncertainty() -> None:
    assert "来源" in MAIN_SYSTEM_PROMPT
    assert "不确定" in MAIN_SYSTEM_PROMPT
    assert "只做检索" in RAG_SYSTEM_PROMPT


def test_rag_prompt_guards_open_contradiction_markers() -> None:
    """待裁决矛盾区块不得被当作确定结论，必须列双方与来源。"""

    assert "contradiction:open" in RAG_SYSTEM_PROMPT
    assert "矛盾待裁决" in RAG_SYSTEM_PROMPT
    assert "不得" in RAG_SYSTEM_PROMPT and "确定结论" in RAG_SYSTEM_PROMPT
    assert "contradiction:resolved" in RAG_SYSTEM_PROMPT


def test_summary_prompt_keeps_load_bearing_marker_contract() -> None:
    """摘要 prompt 必须保留 <messages> marker 与 {messages} 占位符（LangChain 承重契约）。"""

    # marker 必须单独成行，且 {messages} 占位符原样保留，中间件靠它 format 历史
    assert "\n<messages>\n" in SUMMARY_PROMPT
    assert "{messages}" in SUMMARY_PROMPT
    # 上游四段式结构不得被自定义块破坏
    for section in ("SESSION INTENT", "SUMMARY", "ARTIFACTS", "NEXT STEPS"):
        assert f"## {section}" in SUMMARY_PROMPT
    # 附加块只拼接一次，marker 仍唯一
    assert SUMMARY_PROMPT.count("<messages>") == 1
    assert SUMMARY_PROMPT.count("critical_facts_preservation") == 2


def test_summary_prompt_forces_critical_numbers_preservation() -> None:
    """摘要 prompt 必须强制原样保留关键数字、单位与审批规则语义。"""

    assert "关键数字" in SUMMARY_PROMPT
    assert "原样保留" in SUMMARY_PROMPT
    # 明确点名金额/阈值场景，防止"大于 5000 需审核"被压缩成模糊表述
    assert "5000" in SUMMARY_PROMPT and "人工审核" in SUMMARY_PROMPT
    assert "条件 + 阈值 + 后果" in SUMMARY_PROMPT
    # 要求以清单落地，无数字时必须显式写"无"，禁止编造
    assert "关键数字与规则" in SUMMARY_PROMPT
    assert "不得编造" in SUMMARY_PROMPT
