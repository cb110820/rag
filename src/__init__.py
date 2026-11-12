"""DeepAgents RAG 应用包入口。

本项目是一个基于 DeepAgents + LangGraph 的企业知识库问答系统，
采用主代理 + RAG 子代理的架构，通过向量检索 + BM25 混合召回和重排
从内部知识库中检索相关片段，交由主代理综合回答。
"""