"""查询增强（Query Enhancement）。

生产级 RAG 的查询侧增强，包含两种互补策略：

1. **Query Expansion（查询扩展）**：
   用 LLM 把用户口语化查询改写为更适合检索的关键词组合，
   主要喂给 BM25 稀疏检索（关键词匹配对术语敏感）。

2. **HyDE（Hypothetical Document Embeddings）**：
   让 LLM 先生成一段"假想答案"，用它的 embedding 去做 Dense 检索。
   因为答案的语义空间比问题更接近文档原文，召回率显著提升。

设计上不把 LLM 调用耦合进检索核心（pipeline），而是在工具层（tools.py）
生成增强信号后通过参数传入 ``KnowledgeBase.search``，保持检索核心无 LLM 依赖、
可单独测试。LLM 不可用时自动降级为原查询，保证可用性。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.rag.config import get_settings

logger = logging.getLogger(__name__)

# 降级阈值：LLM 调用失败或返回异常时，直接用原查询
_MAX_RETRIES = 2


class QueryEnhancer:
    """查询增强器，封装 LLM 调用做查询扩展与 HyDE 假想答案生成。"""

    def __init__(self, model: Any | None = None) -> None:
        """Args:
        model: LangChain BaseChatModel 实例；None 时尝试从配置解析，
               解析失败则降级为直通（不增强）。
        """

        self._model = model or self._resolve_model()

    @staticmethod
    def _resolve_model() -> Any | None:
        """从项目配置解析 LLM，失败返回 None（降级为直通）。

        一律 ``streaming=False``：ChatTongyi 流式调用存在 tool_calls
        越界问题，且查询增强只需要一次性补全。
        """

        try:
            from src.agents.graph import resolve_chat_model

            spec = get_settings().query_understand_model or None
            model = resolve_chat_model(spec, streaming=False)
            if isinstance(model, str):
                # openai 兼容端点返回的是字符串 spec，需要构造实例
                from langchain_openai import ChatOpenAI

                model_name = model.split(":", 1)[1] if ":" in model else model
                return ChatOpenAI(model=model_name, temperature=0)
            return model
        except Exception as exc:
            logger.warning("查询增强器 LLM 解析失败，降级为直通: %s", exc)
            return None

    # ── 查询理解（QUERY_UNDERSTAND）──────────────────────────────────

    def rewrite(self, query: str) -> str:
        """把口语化/带上下文指代的问题改写为独立、适合检索的查询。

        返回改写后的查询（空或失败返回原查询，绝不阻断检索）。
        """

        if self._model is None:
            return query
        prompt = (
            "你是检索查询改写器。把下面的用户问题改写为一条独立的检索查询："
            "消除指代与口语化表达，保留核心实体与意图，不新增原文没有的术语。"
            "只输出改写后的查询本身，不要解释。\n\n"
            f"问题：{query}\n改写："
        )
        try:
            resp = self._invoke(prompt)
            rewritten = self._clean(resp)
            return rewritten or query
        except Exception as exc:
            logger.warning("查询改写失败，使用原查询: %s", exc)
            return query

    # ── 查询扩展 ──────────────────────────────────────────────────────

    def expand_query(self, query: str) -> str:
        """把查询改写为更适合 BM25 检索的关键词组合。

        返回扩展后的查询字符串（保留原查询 + 补充同义词/术语），
        LLM 不可用时返回原查询。
        """

        if self._model is None:
            return query
        prompt = (
            "你是检索查询优化器。把下面的用户问题改写为适合全文检索的关键词组合，"
            "保留原问题的核心术语，补充 2-4 个同义术语或上位词，用空格分隔。"
            "只输出关键词，不要解释。\n\n"
            f"问题：{query}\n关键词："
        )
        try:
            resp = self._invoke(prompt)
            expanded = self._clean(resp)
            if not expanded:
                return query
            # 保留原查询，确保不丢失原始意图
            logger.info("查询扩展成功，扩展后: %s", f"{query} {expanded}")
            return f"{query} {expanded}"
        except Exception as exc:
            logger.warning("查询扩展失败，使用原查询: %s", exc)
            return query

    # ── HyDE 假想答案 ─────────────────────────────────────────────────

    def generate_hypothetical(self, query: str) -> str:
        """生成一段假想答案文档，用于 Dense 检索（HyDE）。

        LLM 不可用时返回原查询（退化为普通语义检索）。
        """

        if self._model is None:
            return query
        prompt = (
            "请根据下面的问题，写一段假想的答案段落（200-400 字）。"
            "这段文字不是给用户看的，而是用于在知识库中检索相似文档。"
            "使用专业术语，句式接近技术文档风格，不要加标题或列表。\n\n"
            f"问题：{query}\n假想答案："
        )
        try:
            resp = self._invoke(prompt)
            text = self._clean(resp)
            return text or query
        except Exception as exc:
            logger.warning("HyDE 假想答案生成失败，使用原查询: %s", exc)
            return query

    # ── 内部工具 ──────────────────────────────────────────────────────

    def _invoke(self, prompt: str) -> str:
        """调用 LLM，统一处理同步/异步差异。"""

        model = self._model
        # LangChain ChatModel 通常有 invoke；优先用同步 invoke
        if hasattr(model, "invoke"):
            result = model.invoke(prompt)
            return getattr(result, "content", str(result))
        if callable(model):
            result = model(prompt)
            return getattr(result, "content", str(result))
        raise RuntimeError("LLM 实例没有可用的调用方法")

    @staticmethod
    def _clean(text: str) -> str:
        """清洗 LLM 输出：去除多余空白、Markdown 标记。"""

        text = text.strip()
        # 去掉可能的 markdown 代码块包裹
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        # 合并多余空白
        text = re.sub(r"\s+", " ", text).strip()
        return text
