# Wiki + RAG 系统详解（初级开发者友好版）

> 本文档假设你了解 Python 基础和 Web 开发概念，但不需要预先掌握 RAG / 向量数据库 / Agent 等技术。
> 每一节都包含：**它解决什么问题 → 关键代码位置 → 初学者要点**。

---

## 一、这个系统是做什么的？

一句话：**用户上传文档或写 Wiki 页面 → 系统把文档变成可搜索的"知识碎片" → 用户用自然语言提问时，系统找到最相关的碎片，让 AI 基于这些碎片回答。**

类比：就像你有一个图书馆，RAG 系统做了三件事：
1. **入库**：把每本书拆成小卡片，在每张卡片背面写上"这张卡片大概在讲什么"（向量）
2. **检索**：你提问时，找出和你问题最相关的卡片
3. **生成**：把卡片内容交给 AI，让它基于卡片回答你（而不是凭空编造）

Wiki 模块则让"写文档"和"被检索"合二为一——你在 Wiki 里写的页面，自动变成知识库的一部分。

---

## 二、整体架构鸟瞰

```
┌─────────────────────────────────────────────────────┐
│                    用户提问                           │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI 接口层 (src/api/)                │
│  /chat  /wiki/*  /knowledge/*  /memory/*              │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│           DeepAgents 主代理 (src/agents/)             │
│   规划 → 委托 RAG 子代理 → 综合答案                    │
└───────────────────────┬─────────────────────────────┘
                        │ task 委托
                        ▼
┌─────────────────────────────────────────────────────┐
│           RAG 检索引擎 (src/rag/)                     │
│  查询增强 → 混合检索 → 冲突过滤 → 重排序 → CRAG        │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌──────────────────┐  ┌──────────────────┐
│ 知识库文件         │  │  Wiki 数据库       │
│ data/knowledge/   │  │ wiki_pages 表     │
│   <tenant>/       │  │ wiki_revisions 表 │
└──────────────────┘  └──────────────────┘
```

**初学者要点**：先记住这个分层。下面每一层都会展开讲。

---

## 三、文档的旅程：从上传到可被检索

这是理解整个系统的关键。跟着一份文档走完整个流程。

### 第 1 步：文档落盘

用户调用 `POST /knowledge/upload` 上传文件，文件被保存到：

```
data/knowledge/<租户ID>/文件名
```

Wiki 页面（`POST /wiki/pages`）也会被写成一个 Markdown 文件：

```
data/knowledge/<租户ID>/wiki/<slug>.md
```

**为什么要落盘成文件？** 因为后续的解析、分块、嵌入都是围绕文件系统做的，统一了"上传文件"和"Wiki 页面"两条入口。

代码位置：[app.py:281](file:///d:/project/Rag/src/api/app.py#L281)（上传落盘）、[wiki.py:54](file:///d:/project/Rag/src/api/wiki.py#L54)（Wiki 镜像落盘）

### 第 2 步：扫描文档元信息

`scan_document_stats` 不读取文件内容，只记录每个文件的"最后修改时间 + 文件大小"：

```python
# parser.py:106
def scan_document_stats(root: Path) -> dict[str, tuple[float, int]]:
    stats = {}
    for path in root.rglob("*"):
        st = path.stat()
        stats[source_id] = (st.st_mtime, st.st_size)
    return stats
```

**为什么这样做？** 这是**增量索引**的基础。下次刷新时，只要比对这个清单，就能知道哪些文件变了、哪些没变，没变的直接跳过，不用重新解析。

### 第 3 步：解析文档

`load_document` 读取文件内容，并提取元数据（版本号、生效日期、失效日期等）：

```python
# parser.py:124
def load_document(root: Path, source_id: str) -> SourceDocument | None:
    text = _read_path(path).strip()
    meta = {"suffix": path.suffix.lower(), "doc_mtime": path.stat().st_mtime}
    meta.update(_extract_document_metadata(text, source_id))
    return SourceDocument(source_id=source_id, text=text, path=path, metadata=meta)
```

**初学者要点**：`SourceDocument` 是一个不可变数据类（dataclass），包含 `source_id`（来源标识）、`text`（全文）、`metadata`（元数据）。

### 第 4 步：分块（Chunking）

一篇文档可能有几千字，但 AI 的上下文窗口有限，而且检索时需要更细粒度的匹配。所以要把文档切成小块（Chunk）。

```python
# chunking.py:121
def chunk_document(document: SourceDocument, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
```

**两个关键设计**：

1. **结构感知分块**：先按 Markdown 标题（`#`、`##`）切成章节，再在章节内按 400 字符切。这样不会把一个标题和它的内容拆到不同块里。

2. **父子分块**：
   - **子块**（`chunk.text`，400 字符）：小而精确，用于检索打分
   - **父块**（`metadata["parent_text"]`，整个章节）：返回给 AI 做上下文，保证语义完整

```
# 一篇文档被分成这样：
文档
├─ 章节A（parent_text = "标题A\n这是A节的全部内容..."）
│   ├─ 子块1（text = "这是A节的前400字..."）
│   └─ 子块2（text = "这是A节的接下来400字..."）
└─ 章节B
    └─ 子块3
```

**为什么要父子分块？** 检索时用小块精确定位"答案在哪"，但给 AI 看时给整章上下文，避免答案被切散导致 AI 理解不全。

### 第 5 步：嵌入（Embedding）

嵌入 = 把文本变成一串数字（向量），让计算机能计算"两段文本有多相似"。

```
"如何重置密码" → [0.12, -0.34, 0.56, ...]  （1024 维向量）
"忘记密码怎么办" → [0.11, -0.33, 0.58, ...]  （和上面很接近）
"今天天气真好" → [0.91, 0.23, -0.45, ...]    （和上面差很远）
```

项目用的是 `bge-m3` 模型（[embeddings.py](file:///d:/project/Rag/src/rag/embeddings.py)），同时支持稠密向量和稀疏向量。

**嵌入缓存**（[embedding_cache.py](file:///d:/project/Rag/src/rag/embedding_cache.py)）：嵌入计算很耗时，所以把每个块的向量存到 SQLite。下次同样的文本直接读缓存，不用再算。缓存 key 是 `(模型名, 块文本的SHA256)`，换了模型或改了文本都会自动失效。

### 第 6 步：构建索引

分好块、算好向量后，构建两个索引：

1. **FAISS 向量索引**（[retrieval.py:196-216](file:///d:/project/Rag/src/rag/retrieval.py#L196-L216)）：存所有块的向量，用于语义搜索
   - 块数 < 5000：用 `IndexFlatIP`（精确搜索，零精度损失）
   - 块数 ≥ 5000：用 `IndexHNSWFlat`（近似搜索，快很多）

2. **BM25 索引**（[retrieval.py:176-181](file:///d:/project/Rag/src/rag/retrieval.py#L176-L181)）：关键词匹配索引，每 10000 个块分一片

---

## 四、检索的旅程：用户提问时发生了什么

```
用户问："怎么重置密码？"
    │
    ▼
1. 查询增强（QueryEnhancer）
    ├─ HyDE：让 AI 先写一段"假想答案"，用它做语义搜索
    └─ 查询扩展：把"怎么重置密码"扩展成"重置密码 忘记密码 修改密码"
    │
    ▼
2. 混合检索（HybridRetriever）
    ├─ Dense（语义）：FAISS 用假想答案的向量找相似块
    ├─ Sparse（关键词）：BM25 用扩展关键词匹配
    └─ RRF 融合：把两路结果合并排序
    │
    ▼
3. 冲突过滤（resolve_conflicts）
    └─ 同一文档有多个版本时，只留最新版；过期文档降权
    │
    ▼
4. 重排序（BGEReranker）
    └─ 用交叉编码器对前 24 个候选精排，取前 6 个
    │
    ▼
5. CRAG 自我纠正
    └─ 如果最高分太低（< 0.3），用其他查询重试
    │
    ▼
6. 返回结果
    └─ content = parent_text（整章上下文），标注来源
```

### 4.1 为什么需要混合检索？

| 检索方式 | 擅长 | 不擅长 |
|---------|------|--------|
| 向量（Dense） | 语义相似，"重置密码"和"忘记密码"能匹配 | 精确术语、代号搜不到 |
| BM25（Sparse） | 精确关键词匹配 | 同义词、改写匹配差 |

**两者结合** = 语义 + 关键词都能覆盖，这就是"混合检索"。

融合用的是 **RRF（Reciprocal Rank Fusion）**：

```
score = 1/(60 + 向量排名) + 1/(60 + BM25排名)
```

排名越靠前，分数越高。不需要把两种分数归一化就能合并。

### 4.2 为什么需要重排序？

向量和 BM25 都是"粗排"——快但不够准。重排序用一个更强的模型（`bge-reranker-v2-m3`）把 `(问题, 候选块)` 放在一起打分，能更精确地判断相关性。

**流程**：粗排取 24 个 → 重排精排 → 取前 6 个。重排慢但只跑 24 个，可接受。

### 4.3 CRAG 自我纠正

如果检索回来的最高分都很低（< 0.3），说明可能没搜到相关内容。这时系统会自动用"查询扩展"和"假想答案"作为新查询重试，取最好的结果。

代码位置：[pipeline.py:179-199](file:///d:/project/Rag/src/rag/pipeline.py#L179-L199)

---

## 五、Wiki 模块：让文档和检索合一

Wiki 模块解决了一个痛点：**以前写文档和建知识库是两套系统，容易不同步。**

### 数据结构

两张表（[database.py](file:///d:/project/Rag/src/persistence/database.py)）：

| 表 | 作用 |
|----|------|
| `wiki_pages` | 存页面元信息（slug、标题、创建者、时间戳） |
| `wiki_revisions` | 存每个版本的完整内容（append-only，只增不改） |

**为什么分两张表？** 因为要保留版本历史。每次编辑不是覆盖，而是往 `wiki_revisions` 插一行新记录。读当前版本就取最新一行。

### Wiki → RAG 的同步

每次写 Wiki 页面，都会做两件事：

```python
# wiki.py:162
_sync_to_knowledge(tenant_id, slug, title, content)  # 写成 .md 文件
_refresh_if_initialized(tenant_id)                     # 触发索引刷新
```

文件路径：`data/knowledge/<租户ID>/wiki/<slug>.md`

**效果**：你在 Wiki 里写的内容，立刻就能被 RAG 检索到。删 Wiki 页面时，对应的 `.md` 文件也会被删，索引自动更新。

---

## 六、多租户隔离

系统支持多个租户（公司/团队），数据严格隔离。每一层都带 `tenant_id`：

| 层 | 隔离方式 |
|----|----------|
| API | JWT 里解析出 `tenant_id` |
| 数据库 | 所有查询带 `WHERE tenant_id = ?` |
| 知识库 | `data/knowledge/<tenant_id>/` 目录分开 |
| 记忆 | Store namespace 以 `("tenant", tenant_id, ...)` 开头 |

**初学者要点**：你不需要在每个地方都手动传租户 ID——用户登录后，`tenant_id` 会被注入到运行时上下文（`get_runtime().context.tenant_id`），各模块从上下文取即可。

---

## 七、关键代码文件速查表

| 文件 | 职责 | 初学者应该先读 |
|------|------|---------------|
| [models.py](file:///d:/project/Rag/src/rag/models.py) | 数据结构定义（Chunk、SearchResult） | ✅ 最先读，理解数据长什么样 |
| [parser.py](file:///d:/project/Rag/src/rag/parser.py) | 文档加载 + 元数据提取 | ✅ |
| [chunking.py](file:///d:/project/Rag/src/rag/chunking.py) | 文档分块（结构感知 + 父子分块） | ✅ |
| [embeddings.py](file:///d:/project/Rag/src/rag/embeddings.py) | 文本转向量 | |
| [retrieval.py](file:///d:/project/Rag/src/rag/retrieval.py) | 混合检索（FAISS + BM25 + RRF） | 核心 |
| [reranker.py](file:///d:/project/Rag/src/rag/reranker.py) | 重排序 | |
| [pipeline.py](file:///d:/project/Rag/src/rag/pipeline.py) | 流水线编排（串联以上所有） | ✅ 串联全局 |
| [query_enhancer.py](file:///d:/project/Rag/src/rag/query_enhancer.py) | 查询增强（HyDE + 扩展） | |
| [tools.py](file:///d:/project/Rag/src/rag/tools.py) | 暴露给 AI 调用的检索工具 | |
| [wiki.py](file:///d:/project/Rag/src/api/wiki.py) | Wiki CRUD API | |
| [graph.py](file:///d:/project/Rag/src/agents/graph.py) | 主代理构建 | |
| [subagents.py](file:///d:/project/Rag/src/agents/subagents.py) | RAG 子代理定义 | |

---

## 八、初学者阅读路线图

按这个顺序读代码，最容易理解：

1. **[models.py](file:///d:/project/Rag/src/rag/models.py)**（5 分钟）：认识 `Chunk`、`SearchResult` 这两个核心数据结构
2. **[parser.py](file:///d:/project/Rag/src/rag/parser.py)**（10 分钟）：看文档怎么被加载
3. **[chunking.py](file:///d:/project/Rag/src/rag/chunking.py)**（15 分钟）：看文档怎么被切块，理解父子分块
4. **[retrieval.py](file:///d:/project/Rag/src/rag/retrieval.py)**（20 分钟）：看混合检索怎么工作，重点看 `search` 方法
5. **[pipeline.py](file:///d:/project/Rag/src/rag/pipeline.py)**（15 分钟）：看 `KnowledgeBase` 怎么把以上串起来
6. **[tools.py](file:///d:/project/Rag/src/rag/tools.py)**（10 分钟）：看 AI 怎么调用检索
7. **[wiki.py](file:///d:/project/Rag/src/api/wiki.py)**（10 分钟）：看 Wiki 怎么同步到知识库
8. **[graph.py](file:///d:/project/Rag/src/agents/graph.py)** + **[subagents.py](file:///d:/project/Rag/src/agents/subagents.py)**（20 分钟）：看 Agent 怎么组织

**建议**：边读边在 `data/knowledge/default/` 下放一个测试文档，然后调用 `POST /chat` 提问，看日志输出，对照代码理解每一步。

---

## 九、常见问题

**Q: 为什么不直接把整个文档喂给 AI？**
A: 一是上下文窗口有限，长文档塞不下；二是 AI 容易"迷路"，给它精确的相关片段比给整篇文章效果好。

**Q: 向量和关键词为什么要都用？**
A: 向量找语义相似的，关键词找精确匹配的。只用一个会漏掉另一种情况。

**Q: Wiki 和上传文件有什么区别？**
A: 没有本质区别——Wiki 页面最终也是写成 `.md` 文件走同一套索引流水线。区别只在 Wiki 有版本历史和在线编辑能力。

**Q: 多租户怎么保证数据不串？**
A: 每个租户的知识库目录、数据库记录、记忆命名空间都用 `tenant_id` 隔开，代码里所有查询都显式带租户条件。
