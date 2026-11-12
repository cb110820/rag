# RAG 系统升级变更摘要

**升级主题**：文档内冲突解决 —— 结构感知分块 + 时效性标记 + 版本过滤

**日期**：2026-10-18

## 一、改动文件清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `src/rag/parser.py` | 修改 | 新增文档元数据提取（生效/失效日期、版本号、报表号、文件 mtime） |
| `src/rag/chunking.py` | 重写 | 由"定长切块"升级为"结构感知分块"（先按标题切章节，再章节内定长） |
| `src/rag/retrieval.py` | 修改 | 新增 `resolve_conflicts`：同文档族版本过滤 + 时效性加权 |
| `src/rag/pipeline.py` | 修改 | `search` 流程插入冲突过滤环节（召回 → 过滤 → 重排） |
| `src/agents/prompts.py` | 修改 | RAG 子代理 prompt 增加冲突处理规则 |

**未改动**：`models.py`（`metadata` 字段为 `dict`，天然支持扩展，无需改结构）、`embeddings.py`、`reranker.py`、`tools.py`、`config.py`。

## 二、核心变更详情

### 1. 文档元数据提取（parser.py）

从正文与文件名自动解析四类元数据，注入 `SourceDocument.metadata`：

| 字段 | 来源 | 示例 |
|------|------|------|
| `doc_mtime` | `path.stat().st_mtime` | `1760000000.0` |
| `effective_date` | 正文 `生效日期/实施日期/发布日期` | `"2025-01-01"` |
| `expiry_date` | 正文 `失效日期/有效期至/废止日期` | `"2024-12-31"` |
| `version` | 正文 `版本/版本号/Version` 或文件名 `xxx_v2.md` | `"2.0"` |
| `report_no` | 正文 `报表号/编号/文号` | `"RPT-2025-001"` |

日期统一归一化为 ISO `YYYY-MM-DD`，支持 `2024年1月1日`、`2024/1/1`、`2024-01-01` 三种格式。

### 2. 结构感知分块（chunking.py）

**旧策略**：全文按 `chunk_size` 定长切，可能截断章节标题与内容的关联。

**新策略**：两阶段分块

1. **章节切分**：扫描 Markdown 标题行（`#`/`##`/`###`），维护标题层级栈，把文档切为多个章节区间
2. **章节内分块**：每个章节独立执行原有的"定长 + 自然边界对齐 + 重叠"逻辑

每个 Chunk 的 `metadata` 新增：

- `section_path: list[str]`：从根到当前章节的标题路径（如 `['员工手册', '报销政策', '审批流程']`）
- `section_title: str`：当前章节标题（取路径末位）

**向后兼容**：无标题的文档整体作为单一章节，分块结果与旧版一致（现有 2 个 chunking 测试仍通过）。

### 3. 冲突过滤 + 时效性加权（retrieval.py `resolve_conflicts`）

在召回与重排之间插入，处理两类冲突：

**版本过滤**：

- 按文档族分组（`policies_v1.md` / `policies_v2.md` → 同属 `policies` 族）
- 每组选出版本键最大的版本，**丢弃旧版本的所有片段**
- 版本比较键：`(version_tuple, effective_date, doc_mtime)`，依次降级

**时效性加权**（对保留片段调整分数）：

| 状态 | 判定条件 | 加权因子 |
|------|----------|----------|
| `current` | `effective_date ≤ today ≤ expiry_date` | ×1.10 |
| `unknown` | 无日期信息 | ×1.00 |
| `future` | `today < effective_date`（未生效） | ×0.85 |
| `expired` | `today > expiry_date`（已过期） | ×0.60 |

### 4. 流水线接入（pipeline.py）

`search()` 流程变更：

```
旧: 混合召回 → 重排 → 返回
新: 混合召回 → resolve_conflicts → 重排 → 返回
```

冲突过滤在重排前执行，减少重排器的无效候选。

### 5. 子代理冲突指令（prompts.py）

`RAG_SYSTEM_PROMPT` 新增三条规则：

- 矛盾答案优先取版本号更高或生效日期更晚的片段
- 无法判断新旧时，列出矛盾点与各自来源，交主代理综合
- 已过期（`expiry_date` 早于今天）的片段不得作为有效依据

## 三、验证结果

| 检查项 | 结果 |
|--------|------|
| ruff | All checks passed |
| 全量测试 | 80 passed（无回归） |
| 记忆评估（hash/none） | 全通过 |
| 端到端：v1(7天) vs v2(30天) | 只返回 v2，v1 被过滤 |
| 结构感知分块 | `section_path` 正确记录三级标题层级 |
| 时效性加权 | 过期内容 0.5→0.3，有效内容 0.5→0.55 |

## 四、知识库文档约定（推荐）

为充分利用冲突解决能力，建议知识库文档遵循以下约定：

```markdown
# 退款政策

生效日期: 2025-01-01
失效日期: 2099-12-31
版本: 2.0
报表号: POL-2025-001

（正文...）
```

- 多版本文件命名：`policies_v1.md`、`policies_v2.md`
- 日期格式：`YYYY-MM-DD` 或 `YYYY年M月D日`
