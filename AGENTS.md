```
# DeepAgents + RAG 项目约定

## 架构蓝图
- 使用 `create_deep_agent()` 创建主代理，不要手写 StateGraph 编排
- RAG 检索封装为一个独立的子代理（subagent），通过内置的 `task` 工具委托
- 子代理声明方式：使用 `SubAgent` 字典规范，提供 name、description、system_prompt、tools
- 主代理保持“规划+协调”角色，不直接持有检索工具

## 模块边界（严禁越界）
- `src/rag/` 只负责：文档解析、分块、嵌入生成、混合检索（向量+BM25）、重排序
- `src/rag/wiki_schema.py` 只负责：Wiki frontmatter Schema 的容错解析、严格校验、序列化与合并（纯函数，不碰文件系统与 DB）
- `src/rag/lint.py` 只负责：知识健康检查（镜像一致性、Schema 违例、时效性、空正文、未裁决矛盾 contradiction_open；纯函数，业务库 slug 由调用方传入）；矛盾 marker 常量在 lint 本地持有，禁止 rag 反向 import curation
- `src/rag/tools.py` 将检索能力封装成符合 LangChain `@tool` 规范的函数；`retrieve_knowledge` 从 `get_runtime().context.tenant_id` 取租户，仅检索该租户知识库
- `src/curation/` 只负责：LLM Wiki 自动沉淀（原始文档 → 结构化提取 → doc/entity 页面 upsert → 矛盾区块 → index/log → 索引刷新）；依赖方向只能 curation→rag，禁止反向
- `src/curation/markdown.py` 只负责：slug/页面模板/矛盾区块/index/log 渲染等纯函数；`src/curation/llm.py` 只负责结构化提取（模型可注入、JSON 容错、`streaming=False`）；`src/curation/service.py` 负责沉淀编排（per-tenant 锁串行：进程内 asyncio.Lock + postgres advisory lock 跨进程、scan_secret、失败不留半成品）；`src/curation/scheduler.py` 负责上传后 fire-and-forget 调度（开关/DB 任务槽位去重/anyio 超时/异常不冒泡；去重持久化在 `wiki_curation_tasks` 表，UNIQUE(tenant_id, source_id) + lease 过期回收崩溃残留，多副本部署下同一文档全局只有一个运行中任务）
- `src/api/curation.py` 只负责手动沉淀端点（`POST /knowledge/curate` 单篇、`POST /knowledge/curate-all` 批量），与自动调度共用同一 service 与 per-tenant 锁
- `src/memory/embedding.py` 只负责：事实记忆的向量嵌入适配（bge-m3 / hash / none 三态懒加载），被 Store 的 index 配置调用
- `src/memory/tools.py` 只负责：语义事实记忆三件套 `save_memory` / `recall_memories` / `forget_memory`（去重、凭据扫描、配额、TTL、超时重试），不涉及检索实现；身份从 `get_runtime().context` 取 `(tenant_id, user_id)`
- `src/memory/guard.py` 只负责：记忆写入前的凭据扫描、配额校验、内容守卫
- `src/memory/reflection.py` 只负责：LangMem 后台反思提取（/chat 完成后异步从对话提炼事实）；LangMem 只做提取不碰 Store，候选一律经 `tools._save_memory` 护栏写入；提取模型必须 `streaming=False`（ChatTongyi 流式 tool_calls 有越界 bug）；失败仅记日志不影响主对话
- `src/agents/` 定义主代理和子代理的配置，不包含检索实现细节；主代理可挂载记忆工具，RAG 子代理仍为只读
- `src/api/` 只负责：FastAPI 应用组装与业务端点（`app.py` 主入口、`auth.py` 认证、`threads.py` 会话、`memory.py` 记忆、`admin.py` 平台管理、`curation.py` 手动沉淀、`wiki.py` Wiki 页面）
- `src/api/admin.py` 只负责平台管理端点（`/api/admin/tenants` CRUD + stats），全部由 `require_platform_admin` 守护
- `src/persistence/database.py` 的租户隔离：`tenants` 表 + `users/threads` 的 `tenant_id` 外键；Store namespace 一律 `("tenant", tenant_id, "user_memory"|"user_facts", user_id)`
- `src/eval/` 包含 RAGAS 检索评估脚本和记忆无 LLM 评估脚本（`run_memory.py` + `memory_golden_dataset.json`），每次改动记忆策略后必须运行

## 知识治理五原则
- **原始文档不可变**：`/knowledge/upload` 拒绝覆盖同名文件（409），修订走版本化文件名或沉淀到 Wiki；删除保留（配额管理需要）且必须留审计日志
- **Wiki 负责沉淀**：Wiki 页面是人工沉淀的权威知识层，镜像物化到 `KNOWLEDGE_DIR/<tenant>/wiki/<slug>.md` 供检索；另有 LLM 自动沉淀层（src/curation）——上传后自动、或经 `/knowledge/curate(-all)` 手动触发，把原始文档沉淀为 `doc-*` 摘要页与 `entity-*` 实体页，并维护 slug 为 `index`/`log` 的两个特殊系统页；自动发布无审核，回滚靠 wiki_revisions 修订历史；冲突时只追加"矛盾待裁决"区块（不覆盖旧结论），交人工裁决
- **RAG 负责兜底**：检索结果携带 `tier`（wiki/raw），Wiki 片段按 `wiki_tier_boost`（默认 1.15）提权；prompt 规定冲突时以 Wiki 为准，原始文档仅作参考；含 `contradiction:open` 标记的片段不得作为确定结论
- **Schema 约束纪律**：Wiki 正文可携带 YAML frontmatter（title/summary/tags/version/effective_date/expiry_date/kind/sources/aliases 白名单，kind ∈ doc/entity/index/log）；缺省合法，存在则必须严格合法（写路径 400）；镜像始终物化为 frontmatter + 正文结构
- **Lint 保证健康**：`GET /knowledge/lint` 扫描孤儿/缺失镜像、frontmatter 违例、过期/未生效、空正文、未裁决矛盾（contradiction_open warning）；镜像与索引均为可重建的派生数据，修复入口是 `rebuild_wiki_mirrors`

## 生产级红线
- 子代理的 system_prompt 必须明确"只做检索和总结，不做最终决策"
- 主代理的 system_prompt 必须包含"检索结果必须标注来源，不确定时明确说不确定"
- 所有工具必须有超时（timeout）和失败重试机制
- 使用 `CompositeBackend` 或 `StateBackend`，禁止直接使用 `LocalShellBackend` 上生产[citation:14]
- 记忆（memory）文件路径通过 `memory` 参数传入，不要硬编码在 prompt 里
- 记忆工具内的 user_id 必须从 `get_runtime().context.user_id` 获取，禁止从参数/prompt 传入
- 记忆写入必须经过 `scan_secret` 凭据扫描，疑似密钥直接拒绝
- 记忆 TTL 采用绝对语义（`refresh_on_read=False`），过期由后台 sweeper 物理删除
- 语义召回低于 `memory_recall_min_score`（默认 0.2）的命中视为噪声丢弃，避免无关条目污染
- 多租户：所有记忆/知识库/线程 CRUD 必须显式带 `tenant_id`；Store namespace 必须以 `("tenant", tenant_id, ...)` 开头，禁止回退到无前缀 namespace
- 多租户：`tenant_id` 与 `user_id` 一律从 `get_runtime().context` 或登录态 `user` 字典取，禁止从工具参数/prompt 传入
- 多租户：知识库文件落盘到 `KNOWLEDGE_DIR/<tenant_id>/`，上传前校验租户 `max_storage_mb` 配额
- 多租户：`default` 租户不可删除；租户 `status=suspended` 时登录返回 403
- 多租户：平台管理端点（`/api/admin/*`）必须由 `require_platform_admin` 守护，普通成员访问返回 403
- LLM 沉淀是上传主链路的旁路：上传端点只做 fire-and-forget 调度，LLM 超时/失败只能记日志 + 写 log 页，绝不阻断上传或返回错误；提取必须在任何页面写入之前完成，失败不留半成品
- 自动沉淀页面作者一律 `author_id="system"`（不可登录的系统保留账号，由 `ensure_system_user()` 幂等播种，仅供 FK 主体）；index/log 系统页不计业务沉淀统计
- 沉淀写入的新页必须过 wiki_schema 严格校验与 `scan_secret`；同租户沉淀必须经 service 的 per-tenant 锁串行（进程内 asyncio.Lock + postgres advisory lock）；多副本下沉淀去重依赖 `wiki_curation_tasks` 槽位表，禁止退化为进程内 dict 去重；开关 `wiki_curation_enabled` 关闭时自动与手动入口都不产生页面

## 命令
- 测试: `pytest tests/ -v`
- 检索评估: `python -m src.eval.run`
- 记忆评估（hash/none 双模式）: `python -m src.eval.run_memory --mode both`
- 记忆层并发压测: `python scripts/stress_memory.py`
- PG 逻辑备份: `python scripts/backup_postgres.py`
- PG PITR 基准备份: `python scripts/backup_postgres.py --mode base`
- PG 恢复指引: `python scripts/restore_postgres.py --pitr-guide`
- 单租户→多租户迁移: `python scripts/migrate_single_to_multi_tenant.py [--dry-run]`
- 本地启动: `langgraph dev`

## 架构描述
主代理配置：让 Codex 用 create_deep_agent() 创建主代理，传入 system_prompt 定义协调逻辑。参考 DeepAgents 的 GTM agent 实践：主代理负责任务分解、委托检索、整合结果，不直接碰原始文档。

RAG 子代理：让 Codex 实现一个声明式子代理（SubAgent 字典），包含：

name: "rag_retriever"

description: "检索内部知识库，回答关于产品文档、政策、技术规范的问题。返回带来源的摘要片段。"

system_prompt: 定义子代理的行为边界——只检索、只总结、必须引用来源、不编造

tools: 传入你封装好的 RAG 检索 tool

关键节点/流程：DeepAgents 的内置工具已经覆盖了规划（write_todos）、文件读写（read_file/write_file）、子代理委托（task）。让 Codex 不要重复实现这些，只需要配置好子代理的注册和主代理的 system_prompt。

## 技术栈和依赖
Python 3.11+
deepagents（核心，包含 create_deep_agent）
langchain（工具定义）
langchain-mcp-adapters（MCP 支持，可选）
FastAPI + uvicorn（API 层）
Qdrant 或 FAISS（向量存储）
bge-m3（嵌入模型，稠密+稀疏双路径）
bge-reranker-v2-m3（重排序）
RAGAS（评估）
LangSmith（可观测性，DeepAgents 原生支持部署）[citation:5]



## 不要做
- 不要在子代理的 tools 里暴露文件写入或 shell 执行能力
- 不要跳过评估直接改检索策略
- 不要用 MemorySaver 上生产（用 PostgresSaver 或 LangSmith 部署）[citation:5]
- 不要把密码、密钥、令牌等凭据写入记忆（`scan_secret` 会拦截，落盘前失败）
- 不要在记忆工具中硬编码 user_id，必须从 runtime context 注入
- 不要关闭 TTL sweeper（`ttl_sweep_interval_minutes > 0`），否则过期事实无法被物理清除
- 不要用 `refresh_on_read=True`（langgraph sqlite 的 GET 路径会"复活"已过期条目）
```

