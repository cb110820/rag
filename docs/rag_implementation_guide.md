# RAG 实现详解

> 本文档基于 `src/rag/` 全部模块的实际代码整理，不做任何推测。代码引用使用相对路径 + 行号。

## 1. 概览

RAG 代码全部内聚在 `src/rag/`，围绕一个统一入口 `KnowledgeBase`（[pipeline.py L26-L209](src/rag/pipeline.py#L26-L209)）组织，分 **离线索引** 和 **在线检索** 两条主链：

```
【离线索引】磁盘文件 → parser → chunking → embeddings(+cache) → HybridRetriever 建索引
【在线检索】用户问题 → query_enhancer → dense+sparse 双路召回 → RRF 融合
           → 版本冲突过滤 / 时效加权 → reranker 精排 → CRAG 自纠 → 父子块返回
```

### 1.1 模块地图

| 模块 | 职责 |
|---|---|
| `config.py` | `.env` 配置单例（嵌入/重排模式、top_k、超时重试） |
| `models.py` | 三个不可变 dataclass：`SourceDocument` / `Chunk` / `SearchResult` |
| `parser.py` | 多格式解析 + 元数据提取 + 增量文件扫描 |
| `chunking.py` | 结构感知 + 父子分块 |
| `embeddings.py` | 嵌入 Provider：`bge-m3` / `hash` 两态懒加载 |
| `embedding_cache.py` | 块向量 SQLite 增量缓存 |
| `retrieval.py` | BM25 + FAISS 混合召回、RRF、版本冲突、时效加权 |
| `reranker.py` | 交叉编码器重排 / 直通 |
| `query_enhancer.py` | HyDE 假想答案 + 查询扩展 |
| `pipeline.py` | 编排：增量 refresh + search 全流程 + CRAG |
| `tools.py` | 给子代理的 LangChain `@tool`，租户隔离 / 超时 / 重试 / 来源穿透 |

### 1.2 关键设计原则

- **检索核心与 LLM 解耦**：`HybridRetriever` / `KnowledgeBase` 本身不依赖任何大模型，查询增强只是在工具层传入信号，LLM 不可用时自动降级原查询。
- **多租户硬隔离**：知识库按租户目录 `data/knowledge/<tenant>/` 独立加载，检索工具从 `get_runtime().context.tenant_id` 取租户，禁止参数传入。
- **增量优先**：文件 mtime 扫描只重载变更/新增/删除文件；块向量按文本 SHA256 缓存到 SQLite，未改动块零推理。
- **父子分块**：子块（400 字）用于 embedding / BM25 / 重排；整块章节作为 `parent_text` 喂给 LLM，兼顾检索精度与上下文完整。

---

## 2. 离线索引链

### 2.1 解析（parser.py）

支持 `.txt / .md / .html / .htm / .pdf / .docx`（[L27](src/rag/parser.py#L27)），分别用直接读取、BeautifulSoup、pypdf、python-docx 抽纯文本。

```python
# parser.py L91-L107
def extract_text(path: Path) -> str:
    ...
```

解析时正则抽取元数据（[L66-L88](src/rag/parser.py#L66-L88)）：`effective_date / expiry_date / version / report_no / doc_mtime`，供后面版本与时效过滤使用。

`scan_document_stats`（[L110-L125](src/rag/parser.py#L110-L125)）只取每个文件的 `(mtime, size)` **不读内容**，作为增量重建依据。它递归扫描租户目录，所以 Wiki 镜像子目录里的 md 和普通上传文件一视同仁进入同一索引。

### 2.2 结构感知 + 父子分块（chunking.py）

- 先按 Markdown 标题（`#`~`######`）切成章节，维护标题栈生成 `section_path`（如 `["第二章", "2.1 目标"]`）——`_split_sections`（[L23-L55](src/rag/chunking.py#L23-L55)）。
- 章节内再做 **400 字、80 字重叠** 的定长切分，并向换行 / 句号 / 问号等自然边界回退对齐——`_chunk_section`（[L58-L118](src/rag/chunking.py#L58-L118)）。
- 整块章节存入 `metadata["parent_text"]`；子块 `chunk.text` 用于打分。命中后 `SearchResult.to_dict()` 优先返回 `parent_text`（[models.py L50-L51](src/rag/models.py#L50-L51)），同时保留 `chunk_text` 供溯源。

### 2.3 嵌入（两态可切）（embeddings.py）

| Provider | 说明 | 维度 | 依赖 |
|---|---|---|---|
| `HashEmbeddingProvider`（当前运行模式） | 每个中文字 / 英文词 blake2b 哈希，词频 L2 归一化 | 768 | 零依赖、离线 |
| `BGEM3EmbeddingProvider`（生产语义模式） | FlagEmbedding 加载 `BAAI/bge-m3`，取 `dense_vecs` 做 L2 归一化 | 1024 | 需下载约 2.3GB 权重 |

注意当前代码只用了 BGE-M3 的 dense 向量，稀疏检索仍由独立 BM25 承担。

### 2.4 增量向量缓存（embedding_cache.py）

键为 `版本:模型命名空间:块文本SHA256`（[L80-L82](src/rag/embedding_cache.py#L80-L82)），向量持久化到 SQLite `.cache/embeddings.sqlite3`。

- 未改动块直接读盘零推理。
- 新块批量 64 条送模型。
- `prune()`（[L169-L205](src/rag/embedding_cache.py#L169-L205)）清理被删文件的旧向量，分批扫描避免 OOM。
- **查询向量不走缓存**（一次性文本，无复用价值）。

### 2.5 索引构建（retrieval.py HybridRetriever）

内存态双索引：

**稀疏 BM25**（[L174-L185](src/rag/retrieval.py#L174-L185)）
- 每 10,000 块一片（`BM25_SHARD_SIZE`），避免百万级单索引内存爆炸。
- 分词器 `tokenize`（[L133-L138](src/rag/retrieval.py#L133-L138)）：英文按单词 + 中文按单字。
- 查询全部分片，用 RRF 合并（跨分片分数不可比）。

**稠密 FAISS**（[L195-L216](src/rag/retrieval.py#L195-L216)）
- < 5000 块：`IndexFlatIP` 精确暴力搜索，零精度损失。
- ≥ 5000 块：`IndexHNSWFlat`（M=32, efConstruction=200, efSearch=128）近似搜索，查询 O(log N)。
- 未装 faiss 时自动退化为 numpy 矩阵乘法。

### 2.6 增量重建编排（pipeline.refresh，[L74-L143](src/rag/pipeline.py#L74-L143)）

对比文件清单 `(mtime, size)`：完全一致直接返回（毫秒级）；否则只重读/重切**变更和新增**文件、移除删除文件，再整体重建内存索引（嵌入走缓存）。

Wiki 写操作和知识库上传后就是调它（经契约层 `arequest_refresh`）。

---

## 3. 在线检索链

入口是子代理工具 `retrieve_knowledge`（[tools.py L190-L212](src/rag/tools.py#L190-L212)），内部 `_search_with_timeout` 串联以下环节。

### 3.1 租户隔离 + 容错（tools.py）

- **tenant_id 从 `get_runtime().context.tenant_id` 取**（[L211](src/rag/tools.py#L211)），禁止参数传入。
- 每个租户一个懒加载的 `KnowledgeBase` 单例（[L65-L82](src/rag/tools.py#L65-L82)），目录 `data/knowledge/<tenant>/`。
- `anyio.fail_after(30s)` 超时 + tenacity 指数退避最多 3 次重试（[L123](src/rag/tools.py#L123)、[L190-L197](src/rag/tools.py#L190-L197)）。
- 同步检索丢进 anyio 线程池，不阻塞事件循环。
- 命中后经 **contextvar 回调** `source_emitter`（[L38-L47](src/rag/tools.py#L38-L47)）把来源穿出 task 子图边界给 SSE——deepagents 子代理内部的 ToolMessage 不会自动冒泡，这是穿透边界的唯一方式。

### 3.2 查询增强（query_enhancer.py）

| 策略 | 用途 | 喂给 |
|---|---|---|
| **HyDE**（L91-L111） | 让 LLM 先写一段 200-400 字"假想答案" | Dense 检索（答案语义比问题更贴近文档原文） |
| **查询扩展**（L63-L87） | 把口语问题改写成"原词 + 2~4 个同义/上位术语" | BM25 稀疏检索 |

LLM 不可用时自动降级原查询，检索核心本身**不依赖大模型**。

### 3.3 双路召回 + RRF 融合（HybridRetriever.search，[L249-L296](src/rag/retrieval.py#L249-L296)）

- Dense 用 HyDE 文本做 embedding，Sparse 用扩展关键词做 BM25。
- 各取约 24 候选（`candidate_k = max(top_k * 4, 20)`）。
- 用 **Reciprocal Rank Fusion** 融合：

  $$\text{score} = \sum_{\text{两路}} \frac{1}{60 + \text{该路排名}}$$

只认排名、不认原始分，所以向量分和 BM25 分无需对齐量纲。这也是 hash+直通模式下恒看到分数挤在 **0.024~0.033** 的原因：双路第 1 = 1/61 + 1/61 = 0.0328 封顶。

### 3.4 版本冲突过滤 + 时效加权（retrieval.py resolve_conflicts，[L84-L130](src/rag/retrieval.py#L84-L130)）

- 按文件名正则归「文档族」（`policies_v1` / `policies_v2` 同族），同族只留版本键 `(version, effective_date, mtime)` 最大的版本，丢弃旧版片段。
- 时效加权因子：

  | 状态 | 因子 |
  |---|---|
  | `current`（当前有效） | × 1.10 |
  | `unknown`（无日期） | × 1.0 |
  | `future`（未生效） | × 0.85 |
  | `expired`（已过期） | × 0.60 |

### 3.5 精排（reranker.py）

- **交叉编码器**（`bge-reranker-v2-m3`，[L30-L64](src/rag/reranker.py#L30-L64)）：对 `(query, chunk)` 逐对 `compute_score(normalize=True)`，输出有区分度的 **0~1 相关分** 并重排截 top_k。
- **直通**（`none`，[L23-L27](src/rag/reranker.py#L23-L27)）：只截断，保留 RRF 分。**当前运行模式**。

### 3.6 CRAG 自我纠正（pipeline.search，[L145-L209](src/rag/pipeline.py#L145-L209)）

首轮重排后若最高分 `< 0.3`，自动用扩展查询 / 假想答案各重试一轮，取最高分的一轮。

> **注意**：hash+直通模式下 RRF 分天花板 0.033，永远触发不了"高分"，CRAG 会偏向总重试；切到 BGE 重排（0~1 量纲）后阈值才成立。

### 3.7 父子块返回（SearchResult.to_dict，[models.py L42-L59](src/rag/models.py#L42-L59)）

`content` 优先返回整章 `parent_text`，同时保留 `chunk_text` 供溯源；tools 再去重同 source、抽出 section/score 作为前端来源卡片。

---

## 4. 多租户与 Wiki 衔接

- **物理隔离**：每租户一个目录 + 一个 `KnowledgeBase` 实例，检索工具强制带租户上下文。
- **Wiki 不直接依赖 RAG**：Wiki 写业务库后，经契约层（`src/rag/knowledge_contract.py`）把页面镜像成 `<租户>/wiki/<slug>.md` 再通知 `refresh()`；RAG 只把它当普通文件。
- **直接上传**：落 `<租户>/` 其它目录，同样走 `refresh()`。
- **附件旁路**：原样存到 `data_dir/wiki_attachments/`（知识库目录外），**明确不进索引**。
- **单向写入约束**：`resolve_upload_dir` 默认拒绝 `wiki/` 前缀、`assert_not_wiki_source` 拦截删除端点，防止脱离版本管理的孤儿镜像。

---

## 5. 配置项速查表

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `RAG_EMBEDDING` | `bge-m3` | 嵌入模式：`bge-m3` / `hash` / `none` |
| `RAG_RERANKER` | `bge-reranker-v2-m3` | 重排器：`bge-reranker-v2-m3` / `none` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | FlagEmbedding 模型名 / 本地目录 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | FlagReranker 模型名 / 本地目录 |
| `RAG_TOP_K` | `6` | 检索返回条数 |
| `RAG_TIMEOUT_SECONDS` | `30` | 单次检索超时 |
| `RAG_MAX_RETRIES` | `3` | 失败重试次数 |
| `EMBEDDING_CACHE_PATH` | `.cache/embeddings.sqlite3` | 块向量缓存路径 |
| `knowledge_dir` | `data/knowledge` | 知识库根目录（代码侧） |

> **当前运行值**（来自 `.env`）：`RAG_EMBEDDING=hash`，`RAG_RERANKER=none`——覆盖了 config.py 的 bge 默认值，形成零依赖离线模式。

---

## 6. 设计亮点与当前短板

### 亮点

- **检索核心与 LLM 解耦**：可单测、可离线、可降级，不因 LLM 不可用而完全瘫痪。
- **增量优先**：refresh 靠文件 mtime 比较；向量缓存按文本 SHA256；两者结合避免重复推理。
- **父子分块**：短小子块利于打分，整块章节喂给 LLM 利于作答，解决了"块太小丢语义、块太大稀释相关性"的经典矛盾。
- **版本冲突 + 时效加权**：面向真实制度文档（同一政策 v1/v2 并存），自动选最新版并按生效/过期时间加权。
- **FAISS 大小规模自适应**：< 5000 块精确暴力，≥ 5000 块 HNSW 近似。
- **全链路工程化**：租户硬隔离、来源穿透 contextvar、tenacity 重试 + anyio 超时、完整日志体系。

### 短板 / 注意点

| 问题 | 影响 | 可改善方向 |
|---|---|---|
| 当前 hash 模式只有字符匹配 | 语义问句召回率低 | 切 `bge-m3`（需下载权重） |
| CRAG 阈值 0.3 在 hash 模式量纲错配 | 永远触发重试，多做无用功 | 切重排器后自然成立 |
| 进程内内存态索引 | 多副本部署各自刷新 | 引入独立索引服务（Qdrant / pgvector） |
| BM25 中文按单字切分 | 无专业分词，长词命中弱 | 引入 jieba / 清华 THUOCL 分词 |
| BGE-M3 的 sparse 能力未启用 | 稀疏路完全依赖 BM25 | 用 M3 的 lexical 分支替换独立 BM25 |

---

## 7. 常见问题

### Q1：为什么分数都挤在 0.03 左右？

这不是 bug，是 RRF 融合公式的数学上限：`score = Σ 1/(60+rank)`。双路都第 1 名 = `1/61 + 1/61 ≈ 0.0328`，整个取值区间只有 **0.024 ~ 0.033**。这不是"相似度百分比"，只是排名融合的排序分。切到 `bge-reranker-v2-m3` 后分数变成交叉编码器的真实 0~1 相关分，命中通常 0.6~0.99、不相关 < 0.1。

### Q2：如何切到 BGE 语义模式？

改 `.env`：

```
RAG_EMBEDDING=bge-m3
RAG_RERANKER=bge-reranker-v2-m3
```

首次运行 FlagEmbedding 会自动从 HuggingFace 下载约 4.5GB 权重（可用 `HF_ENDPOINT=https://hf-mirror.com` 加速）。切完重启服务即可，无需清缓存（命名空间含模型标识，不会混入旧向量）。

### Q3：RAG 索引什么时候重建？

- 直接上传文件 → `POST /knowledge/upload` → 调 `refresh_knowledge_base()`
- Wiki 建页/更新/导入/回滚 → 契约层通知 `arequest_refresh()`
- 完全相同的文件集 → `refresh()` 直接返回（毫秒级）

### Q4：检索结果里的 `wiki/xxx.md` 和 `docs/xxx.pdf` 有什么区别？

只是来源路径不同。两者在同一个索引、同一个检索里被统一召回，`source_id` 路径作为来源标注返回给主代理。唯一约束是 `wiki/` 子目录只有 Wiki 模块能写，直接上传会被契约层拦截。

### Q5：附件为什么不进 RAG 索引？

附件落在 `data_dir/wiki_attachments/`（知识库目录 `data/knowledge/` 之外），契约层明确不让它进入 `scan_document_stats` 的扫描范围。这是设计选择：附件是原文的原始材料，通常包含扫描件、截图、大表格等不适合直接检索的内容。如果需要附件内容可搜索，应该先把它的核心信息抽成 Wiki 页面或直接上传解析过的版本到知识库。
