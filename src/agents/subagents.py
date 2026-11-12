"""子代理构建函数。

目前只有一个 RAG 检索子代理，按需在此继续扩展其他子代理（如代码执行、网页搜索等）。
"""

from __future__ import annotations

from deepagents.middleware.subagents import SubAgent
from langchain_core.language_models import BaseChatModel

from src.agents.prompts import RAG_SYSTEM_PROMPT
from src.rag.tools import retrieve_knowledge


def build_rag_subagent(model: str | BaseChatModel) -> SubAgent:
    """构建 RAG 检索子代理。

    子代理只拥有 retrieve_knowledge 工具，职责边界清晰：只检索、只总结，
    不做最终决策。最终答案由主代理综合后给出。
    """

    return {
        "name": "rag_retriever",
        "description": (
            "检索内部知识库，回答任何与企业内部内容有关的问题，包括产品文档、政策、"
            "技术规范、流程、设备、测试资料；支持完整问句，也支持关键词、短语或"
            "文档/章节标题式的不完整输入（如“厂测 软件开发总体目标”）；"
            "返回带来源的摘要片段。"
        ),
        "system_prompt": RAG_SYSTEM_PROMPT,
        "model": model,
        "tools": [retrieve_knowledge],
    }