"""RAG（检索增强生成）核心模块。

本包封装文档加载、分块、嵌入、混合检索、重排等完整的检索流水线，
对外只暴露 KnowledgeBase 入口，避免上层代码依赖具体实现细节。
"""

from src.rag.pipeline import KnowledgeBase

__all__ = ["KnowledgeBase"]