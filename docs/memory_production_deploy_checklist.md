# 记忆系统生产部署检查清单

> 覆盖 P0 画像记忆治理、P1 Postgres 双后端、P2 语义事实记忆、P3 评估/压测/备份。
> 每一项均对应代码中真实存在的配置项或脚本，部署前逐项勾选。

---

## 一、环境变量（`.env`）

### 必配（生产不可留空）

| 变量 | 说明 | 建议值 | 校验 |
|---|---|---|---|
| `STORAGE_BACKEND` | 存储后端，生产必须 `postgres` | `postgres` | 非 sqlite |
| `POSTGRES_DSN` | psycopg 连接串（非 SQLAlchemy 语法） | `postgresql://user:pwd@host:5432/db?sslmode=require` | `?sslmode=require` |
| `JWT_SECRET_KEY` | JWT 签名密钥，多副本必须显式配置 | `secrets.token_urlsafe(48)` 输出 | ≥32 字节，非默认空 |

### 记忆治理护栏

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_MAX_FILE_BYTES` | `8192` | 单个画像文件最大字节，防画像膨胀 |
| `MEMORY_MAX_ITEMS` | `20` | 每用户画像条目上限 |

### 语义事实记忆

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_EMBEDDING` | 空（跟随 `RAG_EMBEDDING`） | 生产建议显式设 `bge-m3`；无 GPU 可设 `hash`；纯关键词设 `none` |
| `MEMORY_RECALL_TOP_K` | `5` | 每次召回条数 |
| `MEMORY_RECALL_MIN_SCORE` | `0.2` | cosine 相似度下限，低于此值丢弃噪声 |
| `MEMORY_FACT_MAX_CHARS` | `1000` | 单条事实最大字符数 |
| `MEMORY_FACT_MAX_ITEMS` | `200` | 每用户事实条目上限 |
| `MEMORY_DEDUP_THRESHOLD` | `0.9` | 去重相似度阈值，≥ 则更新旧条目 |
| `MEMORY_DEFAULT_TTL_DAYS` | `0` | 默认 TTL 天数，`0` 不过期；建议设 `90` 或 `180` |
| `MEMORY_TIMEOUT_SECONDS` | `10` | Store 操作超时（秒） |
| `MEMORY_MAX_RETRIES` | `2` | 失败重试次数 |
| `TTL_SWEEP_INTERVAL_MINUTES` | `60` | 过期事实物理清扫间隔，`0` 则关闭（不推荐） |

### 长会话压缩

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_SUMMARY_ENABLED` | `true` | 开启自动摘要旧消息 |
| `MEMORY_SUMMARY_TRIGGER_TOKENS` | `6000` | 触发摘要的 token 阈值 |
| `MEMORY_SUMMARY_KEEP_MESSAGES` | `20` | 摘要后保留的最近原文条数 |

### 可观测性

| 变量 | 说明 |
|---|---|
| `LANGSMITH_TRACING` | 生产建议 `true`，追踪规划/检索/记忆调用链 |
| `LANGSMITH_API_KEY` | LangSmith 密钥 |
| `LANGSMITH_PROJECT` | 项目名（如 `rag-deepagents`） |

---

## 二、存储后端：PostgreSQL

### 镜像与扩展

- [ ] 使用 `pgvector/pgvector:pg16` 镜像（裸 `postgres:16` 无法 `CREATE EXTENSION vector`）
- [ ] 确认 `vector` 扩展已创建（`AsyncPostgresStore` 建索引时自动执行 `CREATE EXTENSION IF NOT EXISTS`）
- [ ] 确认 `langgraph` 扩展已创建（checkpoint 表依赖）

### 连接池

| 变量 | 默认 | 说明 |
|---|---|---|
| `PG_POOL_MIN` | `1` | 业务/saver/store 三池最小连接数 |
| `PG_POOL_MAX` | `10` | 每池最大连接数；副本数 × 3 × PG_POOL_MAX ≤ PG `max_connections` |

### TLS

- [ ] `POSTGRES_DSN` 包含 `?sslmode=require`
- [ ] 云数据库（RDS/Cloud SQL）启用 SSL 证书校验

---

## 三、记忆护栏验证（上线前必跑）

```bash
# 1. 单元测试（含记忆治理 20 + 语义记忆 12 用例）
pytest tests/ -v

# 2. 记忆评估（hash + none 双模式，无 LLM，可接 CI）
python -m src.eval.run_memory --mode both

# 3. 并发压测（多用户 save/recall，校验隔离与完整性）
python scripts/stress_memory.py
```

**通过标准**：
- [ ] pytest 全部绿
- [ ] `run_memory` 两种模式各 7/7
- [ ] `stress_memory` 零错误、零跨用户泄漏

---

## 四、备份与 PITR

### 逻辑备份（日常误删恢复）

```bash
python scripts/backup_postgres.py            # pg_dump 到 .dump
python scripts/backup_postgres.py --keep 14  # 保留最近 14 份
```

### PITR 基准备份（时间点恢复）

```bash
python scripts/backup_postgres.py --mode base   # pg_basebackup
```

### 恢复

```bash
python scripts/restore_postgres.py <file.dump> --yes
python scripts/restore_postgres.py --pitr-guide   # 打印 PITR 操作步骤
```

### 上线前确认

- [ ] `docker-compose.yml` 中 `archive_mode=on`、`wal_level=replica`
- [ ] `rag_wal_archive` 卷与 `rag_pgdata` 卷分开保存
- [ ] WAL 归档目录定期同步到对象存储（S3/OSS）
- [ ] 生产环境每日定时逻辑备份 + 每周基准备份
- [ ] 做过一次恢复演练并记录 RTO/RPO

---

## 五、安全红线

- [ ] `JWT_SECRET_KEY` 非空、随机、≥32 字节；多副本共享同一值
- [ ] `scan_secret` 已拦截 8 类凭据（AWS Key、GitHub Token、JWT、私钥、`api_key=` 等）
- [ ] 记忆工具的 `user_id` 从 `get_runtime().context.user_id` 注入，不通过 prompt 传入
- [ ] RAG 子代理**未挂载**记忆工具（只读）
- [ ] `CompositeBackend` 路由：`/skills/` 只读、`/memory/` 经 `GuardedStoreBackend`
- [ ] 不使用 `MemorySaver`，生产使用 `PostgresSaver`
- [ ] 不使用 `LocalShellBackend`，使用 `CompositeBackend` 或 `StateBackend`

---

## 六、TTL 生命周期

- [ ] `TTL_SWEEP_INTERVAL_MINUTES > 0`（关闭 sweeper 会导致过期事实滞留）
- [ ] `refresh_on_read=False`（langgraph sqlite 的 GET 路径会"复活"过期条目，绝对 TTL 语义）
- [ ] `MEMORY_DEFAULT_TTL_DAYS` 设为合理值（如 90/180），非 `0` 永久
- [ ] 启动日志中可见 `启动 TTL 后台清扫器`
- [ ] 关闭时日志中可见 `停止 TTL 后台清扫器`

---

## 七、嵌入模型（`MEMORY_EMBEDDING`）

| 取值 | 依赖 | 适用场景 |
|---|---|---|
| `bge-m3` | 下载 BAAI/bge-m3 权重（首次启动） | 生产推荐，1024 维稠密向量 |
| `hash` | 无外部依赖，768 维字符哈希 | 无 GPU/无外网环境的次优方案 |
| `none` | 不建向量索引 | 纯关键词召回，无需 sqlite-vec/pgvector |

- [ ] 若选 `bge-m3`，确认首次启动可访问模型权重（HuggingFace / ModelScope）
- [ ] 若选 `none`，确认可接受关键词打分的召回质量

---

## 八、上线后巡检（首周每日）

```bash
# 1. 健康检查
curl http://<host>:<port>/health
# 预期 200，且 "database":"ok"

# 2. 记忆 API 冒烟
curl -H "Authorization: Bearer <token>" http://<host>:<port>/memory/facts
# 预期 200，返回空列表或已有事实

# 3. TTL 清扫日志
grep "TTL 后台清扫器" <app.log>

# 4. LangSmith Trace 检查
# 打开 LANGSMITH_PROJECT，确认 save/recall/forget 调用链可观测
```

- [ ] `/health` 返回 200 且数据库探活 `ok`
- [ ] `/memory/facts` 返回正常，无 500
- [ ] 日志中无 `MemoryRejectedError` 误杀、无 `asyncio.TimeoutError`
- [ ] LangSmith 可看到记忆工具调用链
- [ ] PG 连接数稳定，未接近 `max_connections`
- [ ] 备份文件按时生成、归档卷持续增长

---

## 九、回滚预案

- [ ] 保留上一版本镜像 tag
- [ ] PG 逻辑备份可 1 命令恢复：`python scripts/restore_postgres.py <file.dump> --yes`
- [ ] 若记忆层异常，可临时设 `MEMORY_EMBEDDING=none` 降级为关键词召回，不影响其他功能
- [ ] 若 PG 不可用，`STORAGE_BACKEND=sqlite` 仅适合应急单机，恢复后需手动迁回（`scripts/migrate_sqlite_to_postgres.py` 不迁 checkpoint）

---

## 十、命令速查

| 用途 | 命令 |
|---|---|
| 全量测试 | `pytest tests/ -v` |
| 检索评估 | `python -m src.eval.run` |
| 记忆评估 | `python -m src.eval.run_memory --mode both` |
| 记忆压测 | `python scripts/stress_memory.py` |
| PG 逻辑备份 | `python scripts/backup_postgres.py` |
| PG PITR 基准备份 | `python scripts/backup_postgres.py --mode base` |
| PG 恢复 | `python scripts/restore_postgres.py <file.dump> --yes` |
| PITR 指引 | `python scripts/restore_postgres.py --pitr-guide` |
| 本地启动 | `langgraph dev` |
