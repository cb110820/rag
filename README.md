# DeepAgents + RAG

生产边界清晰的 DeepAgents 示例：主代理只规划、委托和综合答案；知识库检索由
`rag_retriever` 子代理独立完成。检索链包含文档解析、分块、BGE-M3 稠密向量、
BM25、RRF 融合和 BGE reranker，并为缺少模型依赖的开发环境提供可测试的轻量回退。

## 快速开始

```powershell
Copy-Item .env.example .env
uv sync --extra dev --extra rag --extra eval
uv run pytest tests/ -v
uv run python -m src.eval.run
uv run langgraph dev
```

将 `.txt`、`.md`、`.html`、`.pdf` 或 `.docx` 放入 `data/knowledge/`。首次使用
BGE 模型会下载权重。只验证项目结构和词法检索时可省略 `rag`、`eval` extras：

```powershell
uv sync --extra dev
uv run pytest tests/ -v
uv run python -m src.eval.run --embedding hash --reranker none
```

FastAPI 可选入口：

```powershell
uv run uvicorn src.api:app --reload
```

请求示例：

```json
{"message":"退款政策是什么？","thread_id":"demo"}
```

## 架构与边界

- `src/rag/`：解析、分块、嵌入、FAISS/NumPy 向量检索、BM25、RRF 和重排。
- `src/rag/tools.py`：唯一暴露给 RAG 子代理的 LangChain 工具，带超时和重试。
- `src/agents/`：`SubAgent` 配置、主代理提示词和 `create_deep_agent()` 装配。
- `src/eval/`：Golden Dataset 和检索门禁；`--with-ragas` 启用 RAGAS 指标。
- 主代理的 `tools=[]`，因此不直接持有检索工具；它使用内置 `task` 委托。
- 默认使用 `StateBackend`，不会在宿主机执行 Shell。生产持久化应注入
  `CompositeBackend`/`StoreBackend` 与 `PostgresSaver`，不要使用 `LocalShellBackend`
  或 `MemorySaver`。
- `MEMORY_FILES` 只在 `create_deep_agent(memory=...)` 传入，不拼进 prompt。

---

## RAG 检索系统

RAG 子系统负责把本地知识库文档转化为可被语义 + 关键词混合召回的片段，供
`rag_retriever` 子代理查询。主代理**不直接持有检索工具**，只通过 `task` 委托子代理。

### 检索流水线

```
知识库文件 → 解析(load_documents) → 分块(chunk_documents)
    → 嵌入(bge-m3/hash) → 混合召回(HybridRetriever) → 重排(Reranker)
```

| 阶段 | 实现 | 说明 |
|------|------|------|
| 文档解析 | `src/rag/parser.py` | 支持 `.txt`、`.md`、`.html`、`.pdf`、`.docx` |
| 文本分块 | `src/rag/chunking.py` | 固定长度 + 自然边界对齐（段落/句号/中文标点），相邻块重叠避免语义断裂 |
| 嵌入生成 | `src/rag/embeddings.py` | `bge-m3`（稠密+稀疏双路径）；离线兜底用 `HashEmbeddingProvider` |
| 混合检索 | `src/rag/retrieval.py` | Dense（FAISS 优先，退化为 NumPy 矩阵乘）+ Sparse（BM25Okapi），RRF 融合分数 |
| 重排序 | `src/rag/reranker.py` | `bge-reranker-v2-m3`；无模型时用 `IdentityReranker` 直通 |
| 嵌入缓存 | `src/rag/embedding_cache.py` | 磁盘缓存，增量重建时只有新增/改动块才调用嵌入模型 |
| 入口编排 | `src/rag/pipeline.py` | `KnowledgeBase` 统一封装，对外只需 `kb.search(query, top_k)` |

### 检索工具

`src/rag/tools.py` 将 `KnowledgeBase` 封装为 LangChain `@tool`：

- `retrieve_knowledge(query, top_k?)`：从运行时 context 取 `tenant_id`，仅检索**该租户**的知识库
- 带 `timeout` 和失败重试，符合生产红线
- 返回结果必须标注来源文件名

### 多租户隔离

- 知识库文件落盘到 `KNOWLEDGE_DIR/<tenant_id>/`
- `KnowledgeBase` 按租户缓存实例，互不干扰
- 上传前校验租户 `max_storage_mb` 配额

---

## 记忆系统

记忆系统分为两层，互补而非替代：

| 类型 | 存储形式 | 用途 | 生命周期 |
|------|----------|------|----------|
| **画像**（profile） | `/memory/AGENTS.md` 文件 | 少量、稳定的身份与偏好（模型通过文件读写维护） | 长期，手动编辑 |
| **事实记忆**（facts） | LangGraph Store 结构化条目 | 事件型/结论型的独立事实，可向量召回、单条删除、设 TTL | 可设 TTL，后台 sweeper 物理清除 |

### 事实记忆三件套

`src/memory/tools.py` 提供三个 LangChain `@tool`，身份一律从
`get_runtime().context` 取 `(tenant_id, user_id)`，不进 prompt、不暴露给模型：

| 工具 | 签名 | 行为 |
|------|------|------|
| `save_memory` | `(text, category?, ttl_days?)` | 保存事实；语义去重（高相似条目原地更新）、凭据扫描、配额校验 |
| `recall_memories` | `(query, k?)` | 按语义相似度召回；低于 `memory_recall_min_score`（默认 0.2）的命中视为噪声丢弃；无向量索引时退化为关键词覆盖度打分 |
| `forget_memory` | `(memory_id)` | 按 id 显式遗忘单条事实 |

支持的 `category`：`general` / `identity` / `preference` / `project` / `decision` / `feedback`。

### 安全护栏（`src/memory/guard.py`）

写入前强制执行，不靠 prompt 约束：

- **凭据扫描** `scan_secret`：识别 PEM 私钥、AWS Key、GitHub/GitLab Token、Google API Key、Slack Token、OpenAI 风格密钥、JWT、明文凭据赋值，命中即拒绝（错误信息不回显原文）
- **配额校验** `ensure_write_allowed`：单条字符上限、单文件字节上限、每用户条目数上限

### 向量嵌入（`src/memory/embedding.py`）

- 复用 RAG 的 `EmbeddingProvider`，适配为 Store 索引要求的同步嵌入函数
- 三态：`bge-m3`（1024 维，生产）/ `hash`（768 维，离线兜底）/ `none`（纯关键词）
- **懒加载**：应用启动时只声明维度，首次真正嵌入时才加载 bge-m3 权重
- 维度集中在 `Settings.memory_vector_dims`，切换模型需换 namespace

### Store namespace 与多租户

- 事实记忆：`("tenant", tenant_id, "user_facts", user_id)`
- 画像记忆：`("tenant", tenant_id, "user_memory", user_id)`
- 跨租户完全隔离，前缀搜索只命中当前租户数据

### 生产级保障

- 所有 Store 操作带 `timeout` + `tenacity` 指数退避重试
- TTL 采用绝对语义（`refresh_on_read=False`），过期由后台 sweeper 物理删除，不会"复活"
- 去重阈值 `memory_dedup_threshold`：向量开启走语义相似度，关闭时退化为 token Jaccard

## 评估门禁

每次修改检索算法后必须执行 `uv run python -m src.eval.run`。默认门禁为：
Recall@K ≥ 0.80、MRR@K ≥ 0.60。Golden Dataset 的 `source_ids` 必须与知识库的
相对文件路径一致。需要基于 LLM 的 RAGAS 评估时执行：

```powershell
uv run python -m src.eval.run --with-ragas
```

该模式需要有效模型凭据；离线门禁不需要 API key。

