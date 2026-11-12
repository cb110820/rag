# 多租户隔离设计方案

> 目标：将当前"单租户 + 全用户共享知识库"的架构，升级为"租户级数据隔离"，
> 确保不同租户的知识库、记忆、会话、技能互不可见，同时保留平台管理员的全局管理能力。

---

## 一、现状分析

| 资源 | 当前实现 | 隔离风险 |
|---|---|---|
| 知识库 | 单个 `KNOWLEDGE_DIR`，全局共享 `KnowledgeBase` 单例 | ❌ 所有用户检索到同一份文档 |
| 会话（threads） | `threads.user_id` 归属校验 | ⚠️ 仅用户级，无租户维度 |
| 画像记忆 | Store namespace `("user_memory", user_id)` | ⚠️ 用户级隔离，无租户维度 |
| 事实记忆 | Store namespace `("user_facts", user_id)` | ⚠️ 用户级隔离，无租户维度 |
| 技能 | 单个 `SKILLS_DIR`，全局共享 | ❌ 所有用户共享同一份技能 |
| 认证 | JWT，无租户概念 | ❌ token 不含租户信息 |
| RBAC | 无 | ❌ 无平台管理员/租户管理员角色 |

---

## 二、设计原则

1. **租户为第一隔离维度**：所有用户数据归属到 tenant，知识库/技能/记忆按 tenant 隔离
2. **最小侵入**：复用现有 Store namespace 机制和 user_id 归属校验，不重写持久化层
3. **显式优于隐式**：tenant_id 从 JWT 注入，不依赖请求参数（防篡改）
4. **平台管理员豁免收敛**：仅 `/api/admin/*` 前缀的显式管理端点跳过租户隔离，其余一律带 tenant_id 条件
5. **向后兼容**：现有 sqlite/postgres 数据通过迁移脚本自动归属到 `default` 租户

---

## 三、数据模型

### 3.1 新增 `tenants` 表

```sql
CREATE TABLE tenants (
    id           TEXT PRIMARY KEY,           -- nanoid
    name         TEXT NOT NULL,              -- 租户显示名
    slug         TEXT UNIQUE NOT NULL,       -- URL/路径友好标识
    status       TEXT NOT NULL DEFAULT 'active',  -- active | disabled
    plan         TEXT NOT NULL DEFAULT 'free',    -- free | pro | enterprise
    max_users    INTEGER NOT NULL DEFAULT 5,
    max_storage_mb INTEGER NOT NULL DEFAULT 100,
    max_facts    INTEGER NOT NULL DEFAULT 200,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
```

### 3.2 `users` 表增加字段

```sql
ALTER TABLE users ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id);
ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'member';  -- member | admin | owner
```

- `role=owner`：租户所有者，可管理本租户用户、配额
- `role=admin`：租户管理员，同 owner 但不能删除租户
- `role=member`：普通成员

### 3.3 `threads` 表增加字段

```sql
ALTER TABLE threads ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id);
```

（冗余字段，便于按租户统计与查询，避免每次 join users）

### 3.4 Store namespace 升级

| 记忆类型 | 原 namespace | 新 namespace |
|---|---|---|
| 画像记忆 | `("user_memory", user_id)` | `("tenant", tenant_id, "user_memory", user_id)` |
| 事实记忆 | `("user_facts", user_id)` | `("tenant", tenant_id, "user_facts", user_id)` |

**优势**：`asearch(("tenant", tenant_id))` 可一次性检索某租户全部事实；`purge_tenant` 可前缀删除整个租户数据。

---

## 四、认证与授权

### 4.1 JWT 载荷扩展

```python
payload = {
    "sub": user_id,
    "tenant_id": tenant_id,       # 新增
    "role": role,                 # 新增
    "jti": ..., "type": "access", "iat": ..., "exp": ...
}
```

- `tenant_id` 从 DB 读取（不接受客户端传入），防篡改
- 登录时校验 `tenants.status == 'active'`，禁用租户返回 403

### 4.2 `get_current_user` 扩展

返回的 user dict 包含 `tenant_id` 和 `role`。

### 4.3 新增依赖：`require_tenant_active`

```python
async def require_tenant_active(user: CurrentUser) -> dict:
    tenant = await db.get_tenant(user["tenant_id"])
    if tenant is None or tenant["status"] != "active":
        raise HTTPException(403, "租户已禁用")
    return tenant
```

所有业务端点（`/chat`, `/threads/*`, `/memory/*`, `/knowledge/*`, `/skills/*`）增加此依赖。

### 4.4 平台管理员

- 新增独立的 `platform_admin` 角色（存储在 users 表，`tenant_id` 可为空或指向特殊 `_platform` 租户）
- 仅 `/api/admin/*` 端点允许 platform_admin 访问，可跨租户管理
- **严禁**用"tenant_id 为空"作为全局跳过隔离的开关（避免越权）

---

## 五、知识库隔离

### 5.1 目录结构

```
KNOWLEDGE_DIR/
├── <tenant_id_1>/
│   ├── doc1.pdf
│   └── subdir/doc2.md
├── <tenant_id_2>/
│   └── policy.pdf
└── _shared/              # 可选：跨租户共享文档（需显式授权）
```

### 5.2 `KnowledgeBase` 按租户缓存

`get_knowledge_base()` 改为 `get_knowledge_base(tenant_id)`，`@lru_cache` 改为以 tenant_id 为键：

```python
@lru_cache(maxsize=64)
def get_knowledge_base(tenant_id: str) -> KnowledgeBase:
    tenant_dir = settings.knowledge_dir / tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)
    return KnowledgeBase(tenant_dir, ...)
```

### 5.3 `retrieve_knowledge` 工具接入租户

```python
@tool
async def retrieve_knowledge(query: str, top_k: int = 6) -> str:
    runtime = get_runtime()
    tenant_id = runtime.context.tenant_id
    kb = get_knowledge_base(tenant_id)
    ...
```

- 从 `get_runtime().context.tenant_id` 取租户，不从工具参数取
- 检索范围严格限定在该租户目录

### 5.4 上传/列表接口

- `POST /knowledge/upload`：文件写入 `KNOWLEDGE_DIR/<tenant_id>/`
- `GET /knowledge/files`：只列 `KNOWLEDGE_DIR/<tenant_id>/` 下的文件
- 配额校验：上传前检查 `max_storage_mb`

---

## 六、技能隔离

### 6.1 目录结构

```
SKILLS_DIR/
├── _global/              # 平台预置技能，所有租户可见
├── <tenant_id_1>/
│   └── custom_skill/
└── <tenant_id_2>/
    └── ...
```

### 6.2 技能加载

`list_skills` 合并 `_global/` 和 `<tenant_id>/` 两个目录。
租户上传的技能写入 `<tenant_id>/`，不影响其他租户。

---

## 七、记忆隔离（已有 user 隔离，补充 tenant 维度）

### 7.1 namespace 升级

`database.py` 中的 namespace 构造函数：

```python
def user_memory_namespace(tenant_id, user_id):
    return ("tenant", tenant_id, "user_memory", user_id)

def user_facts_namespace(tenant_id, user_id):
    return ("tenant", tenant_id, "user_facts", user_id)
```

### 7.2 工具层

`save_memory` / `recall_memories` / `forget_memory` 从 runtime context 同时取 `user_id` 和 `tenant_id`。

### 7.3 租户级清除

```python
async def purge_tenant_data(tenant_id):
    # 前缀删除 ("tenant", tenant_id) 下所有 namespace
    # 清除 threads、checkpoints、users、facts
```

---

## 八、运行时上下文

`/chat` 端点注入：

```python
result = await agent.ainvoke(
    {"messages": [...]},
    config={"configurable": {"thread_id": body.thread_id}},
    context={"user_id": user["id"], "tenant_id": user["tenant_id"]},
)
```

`AgentContext` 模型增加 `tenant_id` 字段。

---

## 九、配额管理

每个租户可配置：
- `max_users`：成员数上限
- `max_storage_mb`：知识库存储上限
- `max_facts`：事实记忆条目上限（覆盖默认 `memory_fact_max_items`）

上传/保存记忆前校验配额，超限返回 403。

---

## 十、平台管理 API（`/api/admin/*`）

仅 `platform_admin` 角色可访问：

| 端点 | 作用 |
|---|---|
| `GET /api/admin/tenants` | 列出全部租户 |
| `POST /api/admin/tenants` | 创建租户 |
| `PATCH /api/admin/tenants/{id}` | 修改租户（状态、配额） |
| `DELETE /api/admin/tenants/{id}` | 删除租户（级联清除全部数据） |
| `GET /api/admin/tenants/{id}/stats` | 租户统计（用户数、存储、事实数） |
| `GET /api/admin/tenants/{id}/users` | 租户用户列表 |

---

## 十一、数据迁移

`scripts/migrate_single_to_multi_tenant.py`：

1. 创建 `default` 租户
2. 现有 users 全部归属 `default`，角色设为 `member`（首个用户设为 `owner`）
3. 现有 threads 补充 `tenant_id='default'`
4. Store 中现有记忆 namespace 迁移：
   - `("user_memory", X)` → `("tenant", "default", "user_memory", X)`
   - `("user_facts", X)` → `("tenant", "default", "user_facts", X)`
5. 现有知识库文件移动到 `KNOWLEDGE_DIR/default/`

---

## 十二、实施阶段

| 阶段 | 内容 | 风险 |
|---|---|---|
| **P4-1** | tenants 表 + users/threads 加字段 + JWT 扩展 + 租户状态校验 | 低，纯加字段 |
| **P4-2** | 知识库按租户隔离（目录 + KB 缓存 + 工具 context） | 中，需改 KnowledgeBase 单例 |
| **P4-3** | Store namespace 加 tenant 维度 + 迁移脚本 | 中，需迁移现有数据 |
| **P4-4** | 技能租户隔离 + 配额校验 | 低 |
| **P4-5** | 平台管理 API + 角色权限 | 中，新增角色体系 |
| **P4-6** | 测试 + 评估更新 + AGENTS.md 更新 | 低 |

---

## 十三、验证要点

- [ ] 用户 A（租户 1）上传的文档，用户 B（租户 2）检索不到
- [ ] 用户 A 的画像/事实记忆，用户 B 无法读取
- [ ] 禁用租户后，该租户所有用户登录返回 403
- [ ] 平台管理员可跨租户管理，但普通业务端点仍受租户隔离
- [ ] 配额超限返回 403
- [ ] 数据迁移后原有数据可正常访问
- [ ] 现有 74 个测试 + 记忆评估 14/14 保持全绿
