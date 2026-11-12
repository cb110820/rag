"""持久化基础设施（生产级多用户记忆，支持 SQLite / PostgreSQL 双后端）。

通过 ``STORAGE_BACKEND`` 配置切换：

- ``sqlite``（默认，本地开发/单机）：单库文件 ``.data/app.db``（WAL），
  业务连接、``AsyncSqliteSaver``、``AsyncSqliteStore`` 各自独立连接；
- ``postgres``（多副本生产）：三类访问者各持独立的 psycopg 异步连接池，
  LangGraph 多轮对话用 ``AsyncPostgresSaver``，每用户长期记忆用
  ``AsyncPostgresStore``。业务 DDL 与 SQLite 版同构。

公共数据访问层（users/refresh_tokens/threads/记忆）对上层屏蔽后端差异，
auth/threads/API 模块不感知具体驱动。

所有时间戳统一为 Unix 秒（float），ID 统一为 uuid4 hex。
"""

from __future__ import annotations

import contextlib
import datetime
import inspect
import logging
import os
import secrets
import socket
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import TTLConfig
from langgraph.store.sqlite.aio import AsyncSqliteStore

from src.rag.config import get_settings

logger = logging.getLogger("src.memory")

# 内置默认租户：单租户部署或迁移前的存量数据均归属此租户
DEFAULT_TENANT_ID = "default"
DEFAULT_TENANT_NAME = "默认租户"
DEFAULT_TENANT_SLUG = "default"

# 系统保留账号：LLM 自动沉淀等无人值守写入的作者主体。
# 仅用于满足 wiki_pages.created_by / wiki_revisions.author_id 外键，
# 密码为随机不可用串，禁止登录；全局唯一，可被任意租户页面引用。
SYSTEM_USER_ID = "system"
SYSTEM_USER_EMAIL = "system@local.invalid"

# 用户角色：member（普通成员）/ admin（租户管理员）/ owner（租户所有者）
# / platform_admin（平台管理员，可跨租户管理，仅 /api/admin/* 端点放行）
ROLE_MEMBER = "member"
ROLE_ADMIN = "admin"
ROLE_OWNER = "owner"
ROLE_PLATFORM_ADMIN = "platform_admin"
TENANT_ROLES = (ROLE_MEMBER, ROLE_ADMIN, ROLE_OWNER, ROLE_PLATFORM_ADMIN)

# 本进程唯一工作节点标识（多副本部署下区分持锁者，用于沉淀任务表的
# 崩溃残留归属判断与日志排查）
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


# ── SQLite 兼容修复：Store TTL 过期时间格式 ────────────────────────
# langgraph 1.2.x 的 sqlite Store 用 tz-aware datetime 作为绑定参数写入
# expires_at；Python sqlite3 默认适配器会将其序列化为
# ``2026-10-15 03:31:03.564299+00:00``，而清扫 SQL 用
# ``CURRENT_TIMESTAMP``（``2026-10-15 03:31:03``）做字符串比较——
# 前者永远是"更大"的长字符串，导致 TTL 条目永不过期。
# 这里注册秒级、无时区后缀的适配器，使两边格式可按字典序比较。
# 本项目业务表时间戳统一用 Unix 秒，不依赖 datetime 的默认适配行为。
sqlite3.register_adapter(
    datetime.datetime,
    lambda value: value.astimezone(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S"),
)

# 模块级单例：由 FastAPI lifespan 调 init_db/close_db 管理生命周期
_backend: str = "sqlite"
# SQLite FTS5 是否可用（不可用时 Wiki 搜索自动回退 LIKE）
_wiki_fts_available: bool = False
_biz_conn: aiosqlite.Connection | None = None
_saver: Any = None
_store: Any = None
# SQLite：saver/store 各自独立连接（避免事务交叉）
_saver_conn: aiosqlite.Connection | None = None
_store_conn: aiosqlite.Connection | None = None
# Postgres：业务/saver/store 各自独立连接池
_biz_pool: Any = None
_saver_pool: Any = None
_store_pool: Any = None


def new_id() -> str:
    """生成全局唯一 ID（用户、会话、token jti 统一使用）。"""

    return uuid.uuid4().hex


# ── DDL ────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tenants (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL DEFAULT 'active',
    plan            TEXT NOT NULL DEFAULT 'free',
    max_users       INTEGER NOT NULL DEFAULT 5,
    max_storage_mb  INTEGER NOT NULL DEFAULT 100,
    max_facts       INTEGER NOT NULL DEFAULT 200,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name  TEXT NOT NULL,
    tenant_id     TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id),
    role          TEXT NOT NULL DEFAULT 'member',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    jti        TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    expires_at REAL NOT NULL,
    revoked    INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id);

CREATE TABLE IF NOT EXISTS threads (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id  TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id),
    title      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_threads_user_updated ON threads(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_threads_tenant ON threads(tenant_id);

CREATE TABLE IF NOT EXISTS wiki_pages (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id),
    slug        TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_by  TEXT NOT NULL REFERENCES users(id),
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    UNIQUE(tenant_id, slug)
);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_tenant ON wiki_pages(tenant_id);

CREATE TABLE IF NOT EXISTS wiki_revisions (
    id          TEXT PRIMARY KEY,
    page_id     TEXT NOT NULL REFERENCES wiki_pages(id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id),
    content     TEXT NOT NULL,
    author_id   TEXT NOT NULL REFERENCES users(id),
    summary     TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wiki_revisions_page ON wiki_revisions(page_id, created_at DESC);

CREATE TABLE IF NOT EXISTS wiki_attachments (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id),
    page_id      TEXT NOT NULL REFERENCES wiki_pages(id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    stored_name  TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    content_type TEXT NOT NULL DEFAULT '',
    uploaded_by  TEXT NOT NULL REFERENCES users(id),
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wiki_attachments_page ON wiki_attachments(page_id);
CREATE INDEX IF NOT EXISTS idx_wiki_attachments_tenant ON wiki_attachments(tenant_id);

-- LLM 沉淀任务槽位（多副本部署下跨进程去重）：
-- UNIQUE(tenant_id, source_id) 保证同一文档全局只有一个运行中任务；
-- started_at 过期（lease 超时）视为 worker 崩溃残留，可被重新抢占。
CREATE TABLE IF NOT EXISTS wiki_curation_tasks (
    tenant_id  TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    source_id  TEXT NOT NULL,
    worker_id  TEXT NOT NULL,
    started_at REAL NOT NULL,
    UNIQUE(tenant_id, source_id)
);
"""

# SQLite Wiki 全文检索（FTS5 trigram）。独立于 _SCHEMA_SQL 执行：
# 部分裁剪版 SQLite 未编译 FTS5，此时降级为 LIKE 检索，不阻断启动。
# trigram 对 ≥3 字符的词建索引（中文友好）；更短的词由应用层回退 LIKE。
_WIKI_FTS_SQLITE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
    page_id   UNINDEXED,
    tenant_id UNINDEXED,
    slug      UNINDEXED,
    title,
    content,
    tokenize = 'trigram'
);

-- 新版本写入后：替换该页面的 FTS 行（FTS 中每个页面只保留最新版本）。
-- 注意 update_wiki_page 先 INSERT 版本后 UPDATE 标题，此处取到的可能是
-- 旧标题，由 wiki_fts_page_au 触发器在标题更新后纠正。
CREATE TRIGGER IF NOT EXISTS wiki_fts_revision_ai AFTER INSERT ON wiki_revisions BEGIN
    DELETE FROM wiki_fts WHERE page_id = NEW.page_id;
    INSERT INTO wiki_fts(page_id, tenant_id, slug, title, content)
    SELECT NEW.page_id, NEW.tenant_id, p.slug, p.title, NEW.content
    FROM wiki_pages p WHERE p.id = NEW.page_id;
END;

CREATE TRIGGER IF NOT EXISTS wiki_fts_page_au AFTER UPDATE OF title ON wiki_pages BEGIN
    UPDATE wiki_fts SET title = NEW.title WHERE page_id = NEW.id;
END;

-- 页面删除（级联删除版本）时清理 FTS 行。
CREATE TRIGGER IF NOT EXISTS wiki_fts_page_ad AFTER DELETE ON wiki_pages BEGIN
    DELETE FROM wiki_fts WHERE page_id = OLD.id;
END;
"""

# Postgres 版：REAL→DOUBLE PRECISION；revoked 用 SMALLINT（与 SQLite 0/1
# 读写路径完全一致，避免布尔类型隐式转换差异）；逐条执行。
_SCHEMA_SQL_PG: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS tenants (
        id              TEXT PRIMARY KEY,
        name            TEXT NOT NULL,
        slug            TEXT NOT NULL UNIQUE,
        status          TEXT NOT NULL DEFAULT 'active',
        plan            TEXT NOT NULL DEFAULT 'free',
        max_users       INTEGER NOT NULL DEFAULT 5,
        max_storage_mb  INTEGER NOT NULL DEFAULT 100,
        max_facts       INTEGER NOT NULL DEFAULT 200,
        created_at      DOUBLE PRECISION NOT NULL,
        updated_at      DOUBLE PRECISION NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        id            TEXT PRIMARY KEY,
        email         TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        display_name  TEXT NOT NULL,
        tenant_id     TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id),
        role          TEXT NOT NULL DEFAULT 'member',
        created_at    DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id)",
    """
    CREATE TABLE IF NOT EXISTS refresh_tokens (
        jti        TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_hash TEXT NOT NULL,
        expires_at DOUBLE PRECISION NOT NULL,
        revoked    SMALLINT NOT NULL DEFAULT 0,
        created_at DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id)",
    """
    CREATE TABLE IF NOT EXISTS threads (
        id         TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        tenant_id  TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id),
        title      TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_threads_user_updated ON threads(user_id, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_threads_tenant ON threads(tenant_id)",
    """
    CREATE TABLE IF NOT EXISTS wiki_pages (
        id          TEXT PRIMARY KEY,
        tenant_id   TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id),
        slug        TEXT NOT NULL,
        title       TEXT NOT NULL,
        created_by  TEXT NOT NULL REFERENCES users(id),
        created_at  DOUBLE PRECISION NOT NULL,
        updated_at  DOUBLE PRECISION NOT NULL,
        UNIQUE(tenant_id, slug)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_wiki_pages_tenant ON wiki_pages(tenant_id)",
    """
    CREATE TABLE IF NOT EXISTS wiki_revisions (
        id          TEXT PRIMARY KEY,
        page_id     TEXT NOT NULL REFERENCES wiki_pages(id) ON DELETE CASCADE,
        tenant_id   TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id),
        content     TEXT NOT NULL,
        author_id   TEXT NOT NULL REFERENCES users(id),
        summary     TEXT NOT NULL DEFAULT '',
        created_at  DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_wiki_revisions_page "
    "ON wiki_revisions(page_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS wiki_attachments (
        id           TEXT PRIMARY KEY,
        tenant_id    TEXT NOT NULL DEFAULT 'default' REFERENCES tenants(id),
        page_id      TEXT NOT NULL REFERENCES wiki_pages(id) ON DELETE CASCADE,
        filename     TEXT NOT NULL,
        stored_name  TEXT NOT NULL,
        size_bytes   INTEGER NOT NULL,
        content_type TEXT NOT NULL DEFAULT '',
        uploaded_by  TEXT NOT NULL REFERENCES users(id),
        created_at   DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_wiki_attachments_page ON wiki_attachments(page_id)",
    "CREATE INDEX IF NOT EXISTS idx_wiki_attachments_tenant ON wiki_attachments(tenant_id)",
    """
    CREATE TABLE IF NOT EXISTS wiki_curation_tasks (
        tenant_id  TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
        source_id  TEXT NOT NULL,
        worker_id  TEXT NOT NULL,
        started_at DOUBLE PRECISION NOT NULL,
        UNIQUE(tenant_id, source_id)
    )
    """,
)

# Postgres Wiki 全文检索：pg_trgm 的 GIN 索引可加速 ILIKE '%kw%'（中文友好）。
# 独立于 _SCHEMA_SQL_PG 执行：扩展创建权限不足时降级为顺序扫描，不阻断启动。
_WIKI_TRGM_PG_SQL: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "CREATE INDEX IF NOT EXISTS idx_wiki_pages_title_trgm "
    "ON wiki_pages USING gin (title gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS idx_wiki_revisions_content_trgm "
    "ON wiki_revisions USING gin (content gin_trgm_ops)",
)


async def _configure_aiosqlite(conn: aiosqlite.Connection) -> None:
    """WAL + 外键 + 忙等待，保证多连接并发写入不丢锁错误。"""

    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = aiosqlite.Row


async def _sqlite_column_exists(conn: aiosqlite.Connection, table: str, column: str) -> bool:
    """若表不存在则视为"无需迁移"（建表 SQL 会一次性创建全部列）。"""

    async with conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ) as cur:
        if await cur.fetchone() is None:
            return True
    async with conn.execute(f"PRAGMA table_info({table})") as cur:
        rows = await cur.fetchall()
    return any(row[1] == column for row in rows)


async def _migrate_sqlite_schema(conn: aiosqlite.Connection) -> None:
    """对存量 SQLite 库补充多租户字段（幂等，列已存在则跳过）。"""

    migrations = [
        ("users", "tenant_id", "TEXT NOT NULL DEFAULT 'default'"),
        ("users", "role", "TEXT NOT NULL DEFAULT 'member'"),
        ("threads", "tenant_id", "TEXT NOT NULL DEFAULT 'default'"),
    ]
    for table, column, definition in migrations:
        if not await _sqlite_column_exists(conn, table, column):
            logger.info("存量库迁移: 为 %s 表新增列 %s", table, column)
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    await conn.commit()


async def _migrate_postgres_schema(pool: Any) -> None:
    """对存量 Postgres 库补充多租户字段（幂等，列已存在则跳过）。"""

    migrations = [
        ("users", "tenant_id", "TEXT NOT NULL DEFAULT 'default'"),
        ("users", "role", "TEXT NOT NULL DEFAULT 'member'"),
        ("threads", "tenant_id", "TEXT NOT NULL DEFAULT 'default'"),
    ]
    async with pool.connection() as conn:
        for table, column, definition in migrations:
            cur = await conn.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = %s",
                (table, column),
            )
            if await cur.fetchone() is None:
                logger.info("存量库迁移: 为 %s 表新增列 %s", table, column)
                await conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                )


async def ensure_default_tenant() -> None:
    """确保默认租户存在（首次启动创建）；存量数据均归属此租户。"""

    tenant = await get_tenant(DEFAULT_TENANT_ID)
    if tenant is None:
        now = time.time()
        await _execute(
            "INSERT INTO tenants (id, name, slug, status, plan, max_users, "
            "max_storage_mb, max_facts, created_at, updated_at) "
            "VALUES (?, ?, ?, 'active', 'free', 5, 100, 200, ?, ?)",
            (
                DEFAULT_TENANT_ID,
                DEFAULT_TENANT_NAME,
                DEFAULT_TENANT_SLUG,
                now,
                now,
            ),
        )
        logger.info("已创建默认租户: id=%s", DEFAULT_TENANT_ID)


async def ensure_system_user() -> None:
    """确保系统保留账号存在（LLM 自动沉淀写入的作者主体）。

    幂等：账号已存在则不改动；password_hash 写入随机串（任何登录校验
    都无法通过），不签发 token。
    """

    existing = await _one(
        "SELECT id FROM users WHERE id = ?", (SYSTEM_USER_ID,)
    )
    if existing is not None:
        return
    await _execute(
        "INSERT INTO users (id, email, password_hash, display_name, tenant_id, "
        "role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            SYSTEM_USER_ID,
            SYSTEM_USER_EMAIL,
            "!disabled-" + secrets.token_hex(32),
            "系统（自动沉淀）",
            DEFAULT_TENANT_ID,
            ROLE_MEMBER,
            time.time(),
        ),
    )
    logger.info("已创建系统保留账号: id=%s", SYSTEM_USER_ID)


def _store_index_config(settings: Any) -> dict[str, Any] | None:
    """构造 Store 向量索引配置（仅索引 value 中的 ``text`` 字段）。

    MEMORY_EMBEDDING=none（或 RAG_EMBEDDING=none）时返回 None，
    语义召回退化为应用层关键词打分，Store 不建向量表/扩展。
    """

    if not settings.memory_vector_enabled:
        logger.info(
            "语义记忆向量索引关闭（embedding=%s），召回走关键词兜底",
            settings.memory_embedding_kind,
        )
        return None

    from src.memory.embedding import memory_embed

    return {
        "dims": settings.memory_vector_dims,
        "embed": memory_embed,
        "fields": ["text"],
    }


def _store_ttl_config(settings: Any) -> TTLConfig:
    """Store TTL 策略：读取时刷新/过滤过期项；默认 TTL 0 表示不过期。"""

    default_minutes = (
        settings.memory_default_ttl_days * 24 * 60
        if settings.memory_default_ttl_days > 0
        else None
    )
    # 注意：langgraph 1.2.x sqlite Store 的 GET 路径不做过期过滤，且
    # refresh_on_read=True 会在读取时把已过期条目的 expires_at 整体顺延
    # （"滑动续期复活"）。记忆 TTL 采用绝对语义（自写入起固定保留期），
    # 因此关闭读取续期；过期条目的物理清除依赖周期 sweeper
    # （默认 60 分钟一次，日级 TTL 下最大过期滞留即该间隔）。
    return TTLConfig(
        refresh_on_read=False,
        omit_expired=True,
        default_ttl=default_minutes,
        sweep_interval_minutes=settings.ttl_sweep_interval_minutes or None,
    )


# ── 初始化 / 关闭 ──────────────────────────────────────────────────


async def init_db() -> None:
    """初始化全部持久化连接与表结构（幂等，应用启动时调用一次）。"""

    global _backend

    settings = get_settings()
    backend = settings.storage_backend.strip().lower()
    if backend == "sqlite":
        _backend = "sqlite"
        await _init_sqlite(settings)
    elif backend == "postgres":
        _backend = "postgres"
        await _init_postgres(settings)
    else:
        raise RuntimeError(
            f"不支持的 STORAGE_BACKEND={backend!r}，仅支持 sqlite / postgres"
        )

    await ensure_default_tenant()
    await ensure_system_user()
    deleted = await purge_expired_refresh_tokens()

    # TTL 后台清扫：周期性物理删除过期记忆（读取路径另有惰性过期兜底）
    if settings.ttl_sweep_interval_minutes > 0:
        await _store.start_ttl_sweeper(settings.ttl_sweep_interval_minutes)
        logger.info(
            "记忆 TTL 后台清扫已启动: 间隔=%d 分钟, 默认 TTL=%s",
            settings.ttl_sweep_interval_minutes,
            f"{settings.memory_default_ttl_days} 天"
            if settings.memory_default_ttl_days > 0
            else "不过期",
        )

    logger.info(
        "持久化层已就绪: backend=%s；checkpointer 与长期记忆 Store 已初始化"
        "（向量索引=%s）；清理过期 refresh token %d 条",
        backend,
        "on" if settings.memory_vector_enabled else "off",
        deleted,
    )


async def _init_sqlite_wiki_fts(conn: aiosqlite.Connection) -> None:
    """创建 Wiki FTS5 虚拟表与同步触发器，并在首次启用时全量回填。

    FTS5 不可用（裁剪版 SQLite）时仅记录告警，Wiki 搜索回退 LIKE。
    """

    global _wiki_fts_available
    try:
        await conn.executescript(_WIKI_FTS_SQLITE_SQL)
        _wiki_fts_available = True
    except Exception as exc:  # pragma: no cover - 取决于 SQLite 编译选项
        _wiki_fts_available = False
        logger.warning("SQLite 未启用 FTS5（%s），Wiki 全文搜索回退 LIKE", exc)
        return

    # 幂等回填：FTS 为空（首次启用/旧库升级）时灌入每个页面的最新版本。
    fts_count = await _one_scalar(conn, "SELECT COUNT(*) FROM wiki_fts")
    if fts_count:
        return
    await conn.execute(
        "INSERT INTO wiki_fts(page_id, tenant_id, slug, title, content) "
        "SELECT p.id, p.tenant_id, p.slug, p.title, r.content "
        "FROM wiki_pages p "
        "JOIN wiki_revisions r ON r.page_id = p.id "
        "AND r.created_at = ("
        "  SELECT MAX(created_at) FROM wiki_revisions WHERE page_id = p.id"
        ")"
    )


async def _one_scalar(conn: Any, sql: str, params: Sequence[Any] = ()) -> Any:
    """在初始化阶段（_biz_conn 尚未赋值）对指定连接取单值。

    仅用于 SQLite 初始化路径，SQL 直接使用 ``?`` 占位符。
    """

    async with conn.execute(sql, params) as cur:
        row = await cur.fetchone()
    return row[0] if row is not None else None


async def _init_sqlite(settings: Any) -> None:
    global _biz_conn, _saver, _store, _saver_conn, _store_conn

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db_path = settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 业务连接
    biz_conn = await aiosqlite.connect(str(db_path))
    await _configure_aiosqlite(biz_conn)
    # 先执行列迁移（为旧库补 tenant_id/role 等列），再跑建表/建索引 SQL，
    # 否则 CREATE INDEX ON users(tenant_id) 会因列不存在而失败
    await _migrate_sqlite_schema(biz_conn)
    await biz_conn.executescript(_SCHEMA_SQL)
    await _init_sqlite_wiki_fts(biz_conn)
    await biz_conn.commit()

    # 2) LangGraph checkpoint 连接（多轮对话短期记忆）
    saver_conn = await aiosqlite.connect(str(db_path))
    await _configure_aiosqlite(saver_conn)
    saver = AsyncSqliteSaver(saver_conn)
    await saver.setup()

    # 3) 长期记忆 Store（异步实现；deepagents 的 StoreBackend 走 aget/aput，
    #    同步版 SqliteStore 不支持 async，必须使用 AsyncSqliteStore）。
    #    关键：Store 要求 autocommit 连接（其官方 from_conn_string 工厂即用
    #    isolation_level=None，内部自行 BEGIN/管理事务）；若用 sqlite3 模块
    #    默认的 legacy 隔离模式，迁移 INSERT 会留下未提交事务并长期持有写锁，
    #    导致其他连接写入时报 "database is locked"。
    store_conn = await aiosqlite.connect(str(db_path), isolation_level=None)
    await store_conn.execute("PRAGMA journal_mode=WAL")
    await store_conn.execute("PRAGMA synchronous=NORMAL")
    await store_conn.execute("PRAGMA foreign_keys=ON")
    await store_conn.execute("PRAGMA busy_timeout=5000")
    ttl_config = _store_ttl_config(settings)
    try:
        # 向量索引依赖 sqlite-vec（已随依赖安装）；不可用时降级为无索引 Store，
        # recall_memories 会自动切换到关键词兜底，不影响应用启动。
        store = AsyncSqliteStore(
            store_conn, index=_store_index_config(settings), ttl=ttl_config
        )
        await store.setup()
    except Exception as exc:
        if not settings.memory_vector_enabled:
            raise
        logger.warning(
            "SQLite 向量索引初始化失败（%s），降级为无索引 Store（关键词召回）",
            exc,
        )
        store = AsyncSqliteStore(store_conn, ttl=ttl_config)
        await store.setup()

    _biz_conn = biz_conn
    _saver, _saver_conn = saver, saver_conn
    _store, _store_conn = store, store_conn


async def _init_postgres_wiki_trgm(pool: Any) -> None:
    """启用 pg_trgm 并创建 GIN 三元组索引加速 Wiki ILIKE 检索。

    扩展/索引创建失败（通常是权限不足）时降级为顺序扫描，不阻断启动。
    """

    try:
        async with pool.connection() as conn:
            for statement in _WIKI_TRGM_PG_SQL:
                await conn.execute(statement)
    except Exception as exc:  # pragma: no cover - 取决于 PG 权限/版本
        logger.warning("Postgres pg_trgm 索引不可用（%s），Wiki 搜索走顺序扫描", exc)


async def _init_postgres(settings: Any) -> None:
    """初始化 Postgres 三连接池 + 业务表 + LangGraph saver/store。"""

    global _biz_pool, _saver, _saver_pool, _store, _store_pool

    dsn = settings.postgres_dsn.strip()
    if not dsn:
        raise RuntimeError("STORAGE_BACKEND=postgres 但未配置 POSTGRES_DSN")

    # 延迟导入：未安装 postgres 可选依赖、或纯 SQLite 部署时不影响启动
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.store.postgres.aio import AsyncPostgresStore
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise RuntimeError(
            "STORAGE_BACKEND=postgres 需要安装 postgres 依赖组："
            "uv sync --extra postgres（或 pip install "
            '"langgraph-checkpoint-postgres" "psycopg[binary]"）'
        ) from exc

    # data_dir 仍用于持久化 .jwt_secret（单机文件型密钥，多副本应改用环境变量）
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    common = dict(min_size=settings.pg_pool_min, max_size=settings.pg_pool_max)
    # 业务池：dict_row 使查询行直接表现为 dict（与 sqlite Row 经 _row_to_dict
    # 转换后的形态一致）；saver/store 池由 LangGraph 自行管理行工厂，不覆盖。
    biz_pool = AsyncConnectionPool(
        dsn, open=False, timeout=30, kwargs={"row_factory": dict_row}, **common
    )
    saver_pool = AsyncConnectionPool(dsn, open=False, timeout=30, **common)
    store_pool = AsyncConnectionPool(dsn, open=False, timeout=30, **common)
    for pool in (biz_pool, saver_pool, store_pool):
        await pool.open()

    try:
        # 先迁移（补列），再跑建表/建索引，避免索引引用尚未存在的列
        await _migrate_postgres_schema(biz_pool)
        async with biz_pool.connection() as conn:
            for statement in _SCHEMA_SQL_PG:
                await conn.execute(statement)
        await _init_postgres_wiki_trgm(biz_pool)

        saver = AsyncPostgresSaver(saver_pool)
        await saver.setup()
        ttl_config = _store_ttl_config(settings)
        try:
            # 向量索引依赖 pgvector 扩展（官方镜像内置）；扩展不可用时
            # 降级为无索引 Store（关键词召回），不阻塞启动。
            store = AsyncPostgresStore(
                store_pool, index=_store_index_config(settings), ttl=ttl_config
            )
            await store.setup()
        except Exception as exc:
            if not settings.memory_vector_enabled:
                raise
            logger.warning(
                "Postgres 向量索引初始化失败（%s），降级为无索引 Store（关键词召回）",
                exc,
            )
            store = AsyncPostgresStore(store_pool, ttl=ttl_config)
            await store.setup()
    except Exception:
        # 建表失败必须释放已打开的连接，否则 lifespan 重试会泄漏连接池
        for pool in (biz_pool, saver_pool, store_pool):
            await pool.close()
        raise

    _biz_pool, _saver_pool, _store_pool = biz_pool, saver_pool, store_pool
    _saver, _store = saver, store


async def close_db() -> None:
    """关闭全部连接/连接池（应用停机时调用）。"""

    global _backend, _biz_conn, _saver, _store, _saver_conn, _store_conn
    global _biz_pool, _saver_pool, _store_pool, _wiki_fts_available

    if _store is not None:
        stop = getattr(_store, "stop_ttl_sweeper", None)
        if stop is not None:
            result = stop()
            if inspect.isawaitable(result):
                await result
    for conn in (_biz_conn, _saver_conn, _store_conn):
        if conn is not None:
            await conn.close()
    for pool in (_biz_pool, _saver_pool, _store_pool):
        if pool is not None:
            await pool.close()

    _backend = "sqlite"
    _wiki_fts_available = False
    _biz_conn = _saver_conn = _store_conn = None
    _biz_pool = _saver_pool = _store_pool = None
    _saver = _store = None
    logger.info("持久化层连接已关闭")


def get_saver() -> Any:
    if _saver is None:
        raise RuntimeError("数据库尚未初始化，请先 await init_db()")
    return _saver


def get_store() -> Any:
    if _store is None:
        raise RuntimeError("数据库尚未初始化，请先 await init_db()")
    return _store


# ── 后端无关的数据访问小工具 ───────────────────────────────────────


def _to_pg(sql: str) -> str:
    """aiosqlite 风格的 ? 占位符转换为 psycopg 的 %s。"""

    return sql.replace("?", "%s")


def _conn() -> aiosqlite.Connection:
    if _biz_conn is None:
        raise RuntimeError("数据库尚未初始化，请先 await init_db()")
    return _biz_conn


def _row_to_dict(row: Any) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


async def _execute(sql: str, params: Sequence[Any] = ()) -> None:
    """写操作（INSERT/UPDATE/DELETE），自动提交（SQLite）。"""

    if _backend == "postgres":
        async with _biz_pool.connection() as conn:
            await conn.execute(_to_pg(sql), params)
        return
    await _conn().execute(sql, params)
    await _conn().commit()


async def _execute_rowcount(sql: str, params: Sequence[Any] = ()) -> int:
    """写操作并返回受影响行数。"""

    if _backend == "postgres":
        async with _biz_pool.connection() as conn:
            cur = await conn.execute(_to_pg(sql), params)
            return int(cur.rowcount)
    cur = await _conn().execute(sql, params)
    await _conn().commit()
    return cur.rowcount


async def _one(sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
    if _backend == "postgres":
        async with _biz_pool.connection() as conn:
            cur = await conn.execute(_to_pg(sql), params)
            row = await cur.fetchone()
    else:
        async with _conn().execute(sql, params) as cur:
            row = await cur.fetchone()
    return _row_to_dict(row)


async def _all(sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    if _backend == "postgres":
        async with _biz_pool.connection() as conn:
            cur = await conn.execute(_to_pg(sql), params)
            rows = await cur.fetchall()
    else:
        async with _conn().execute(sql, params) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


def is_unique_violation(exc: BaseException) -> bool:
    """判断异常是否为唯一约束冲突（兼容 sqlite3 与 psycopg 驱动）。"""

    if type(exc).__module__ == "sqlite3":
        return "UNIQUE" in str(exc).upper()
    return getattr(exc, "sqlstate", None) == "23505"


# ── 跨进程协作原语（沉淀调度多副本部署用） ──────────────────────────


@contextlib.asynccontextmanager
async def try_named_lock(name: str) -> AsyncIterator[bool]:
    """跨进程互斥尝试锁，yield 是否获取成功。

    - postgres：会话级 advisory lock（``pg_try_advisory_lock`` + hashtext），
      在独立连接上持锁，任务期间该连接不归还连接池；会话锁不受
      事务回滚影响，显式 unlock 或连接关闭时释放；
    - sqlite：单机部署无跨进程互斥需求，恒 yield True（进程内互斥由
      调用方自己的锁保证）。
    """

    if _backend != "postgres" or _biz_pool is None:
        yield True
        return
    async with _biz_pool.connection() as conn:
        cur = await conn.execute(
            "SELECT pg_try_advisory_lock(hashtext(?))", (name,)
        )
        row = await cur.fetchone()
        acquired = bool(row[0]) if row else False
        try:
            yield acquired
        finally:
            if acquired:
                # 先回滚获取锁语句遗留的隐式事务（session 级锁不受影响），
                # 再显式释放；防止带着 aborted 事务执行 unlock 报错
                with contextlib.suppress(Exception):
                    await conn.rollback()
                with contextlib.suppress(Exception):
                    await conn.execute(
                        "SELECT pg_advisory_unlock(hashtext(?))", (name,)
                    )
                with contextlib.suppress(Exception):
                    await conn.commit()


async def claim_curation_task(
    tenant_id: str, source_id: str, lease_seconds: float
) -> bool:
    """跨进程抢占沉淀任务槽位，返回是否抢占成功。

    - 先清除该文档已过 lease 期限的崩溃残留行，再 INSERT；
    - UNIQUE(tenant_id, source_id) 冲突 → 已有任务在运行（含其他 worker
      未过期的 lease），返回 False；
    - lease_seconds 应显著大于单次沉淀最大耗时（超时 + 兜底宽限），
      避免误回收仍在运行的任务。
    """

    stale_before = time.time() - lease_seconds
    await _execute(
        "DELETE FROM wiki_curation_tasks "
        "WHERE tenant_id = ? AND source_id = ? AND started_at < ?",
        (tenant_id, source_id, stale_before),
    )
    try:
        await _execute(
            "INSERT INTO wiki_curation_tasks "
            "(tenant_id, source_id, worker_id, started_at) VALUES (?, ?, ?, ?)",
            (tenant_id, source_id, WORKER_ID, time.time()),
        )
    except Exception as exc:
        if is_unique_violation(exc):
            return False
        raise
    logger.debug(
        "沉淀任务槽位已抢占: tenant=%s source=%s worker=%s",
        tenant_id, source_id, WORKER_ID,
    )
    return True


async def release_curation_task(tenant_id: str, source_id: str) -> None:
    """释放本 worker 持有的沉淀任务槽位（任务结束时调用，幂等）。"""

    await _execute(
        "DELETE FROM wiki_curation_tasks "
        "WHERE tenant_id = ? AND source_id = ? AND worker_id = ?",
        (tenant_id, source_id, WORKER_ID),
    )


async def health_check() -> dict[str, str]:
    """轻量探活：业务连接 SELECT 1，供 /health 与负载均衡使用。"""

    if _backend == "postgres":
        async with _biz_pool.connection() as conn:
            await conn.execute("SELECT 1")
    else:
        await _conn().execute("SELECT 1")
    return {"status": "ok", "storage_backend": _backend}


# ── tenants（租户） ────────────────────────────────────────────────


async def create_tenant(
    *,
    name: str,
    slug: str,
    status: str = "active",
    plan: str = "free",
    max_users: int = 5,
    max_storage_mb: int = 100,
    max_facts: int = 200,
) -> str:
    """创建租户，返回 tenant_id；slug 重复时抛底层唯一约束异常。"""

    tenant_id = new_id()
    now = time.time()
    await _execute(
        "INSERT INTO tenants (id, name, slug, status, plan, max_users, "
        "max_storage_mb, max_facts, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (tenant_id, name, slug, status, plan, max_users, max_storage_mb, max_facts, now, now),
    )
    return tenant_id


async def get_tenant(tenant_id: str) -> dict[str, Any] | None:
    return await _one(
        "SELECT id, name, slug, status, plan, max_users, max_storage_mb, "
        "max_facts, created_at, updated_at FROM tenants WHERE id = ?",
        (tenant_id,),
    )


async def get_tenant_by_slug(slug: str) -> dict[str, Any] | None:
    return await _one(
        "SELECT id, name, slug, status, plan, max_users, max_storage_mb, "
        "max_facts, created_at, updated_at FROM tenants WHERE slug = ?",
        (slug,),
    )


async def list_tenants() -> list[dict[str, Any]]:
    return await _all(
        "SELECT id, name, slug, status, plan, max_users, max_storage_mb, "
        "max_facts, created_at, updated_at FROM tenants ORDER BY created_at ASC"
    )


async def update_tenant(
    tenant_id: str,
    *,
    name: str | None = None,
    status: str | None = None,
    plan: str | None = None,
    max_users: int | None = None,
    max_storage_mb: int | None = None,
    max_facts: int | None = None,
) -> bool:
    """更新租户字段；返回是否实际更新（不存在为 False）。"""

    fields: list[str] = []
    params: list[Any] = []
    if name is not None:
        fields.append("name = ?")
        params.append(name)
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if plan is not None:
        fields.append("plan = ?")
        params.append(plan)
    if max_users is not None:
        fields.append("max_users = ?")
        params.append(max_users)
    if max_storage_mb is not None:
        fields.append("max_storage_mb = ?")
        params.append(max_storage_mb)
    if max_facts is not None:
        fields.append("max_facts = ?")
        params.append(max_facts)
    if not fields:
        return False
    fields.append("updated_at = ?")
    params.append(time.time())
    params.append(tenant_id)
    rows = await _execute_rowcount(
        f"UPDATE tenants SET {', '.join(fields)} WHERE id = ?", tuple(params)
    )
    return rows > 0


async def delete_tenant(tenant_id: str) -> bool:
    """物理删除租户；其下 users/threads 由外键 ON DELETE 级联清除。
    记忆数据需另行按 namespace 前缀清除（见 purge_tenant_memory）。"""

    rows = await _execute_rowcount("DELETE FROM tenants WHERE id = ?", (tenant_id,))
    return rows > 0


async def get_tenant_stats(tenant_id: str) -> dict[str, Any]:
    """租户统计：用户数、会话数、事实记忆数。"""

    user_row = await _one("SELECT COUNT(*) AS c FROM users WHERE tenant_id = ?", (tenant_id,))
    thread_row = await _one("SELECT COUNT(*) AS c FROM threads WHERE tenant_id = ?", (tenant_id,))
    # 事实记忆在 Store 中，按 namespace 前缀统计
    fact_items = await get_store().asearch(("tenant", tenant_id, "user_facts"), limit=10000)
    return {
        "tenant_id": tenant_id,
        "user_count": int(user_row["c"]) if user_row else 0,
        "thread_count": int(thread_row["c"]) if thread_row else 0,
        "fact_count": len(fact_items),
    }


# ── users ─────────────────────────────────────────────────────────


async def create_user(
    *,
    email: str,
    password_hash: str,
    display_name: str,
    tenant_id: str = DEFAULT_TENANT_ID,
    role: str = ROLE_MEMBER,
) -> str:
    """创建用户，返回 user_id；邮箱重复时抛底层唯一约束异常。"""

    user_id = new_id()
    now = time.time()
    await _execute(
        "INSERT INTO users (id, email, password_hash, display_name, tenant_id, "
        "role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, email, password_hash, display_name, tenant_id, role, now),
    )
    return user_id


async def get_user_by_email(email: str) -> dict[str, Any] | None:
    return await _one(
        "SELECT id, email, password_hash, display_name, tenant_id, role, created_at "
        "FROM users WHERE email = ?",
        (email,),
    )


async def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    return await _one(
        "SELECT id, email, password_hash, display_name, tenant_id, role, created_at "
        "FROM users WHERE id = ?",
        (user_id,),
    )


async def list_users_by_tenant(tenant_id: str) -> list[dict[str, Any]]:
    """列出某租户下全部用户（不含 password_hash）。"""

    return await _all(
        "SELECT id, email, display_name, tenant_id, role, created_at "
        "FROM users WHERE tenant_id = ? ORDER BY created_at ASC",
        (tenant_id,),
    )


async def count_users() -> int:
    row = await _one("SELECT COUNT(*) AS c FROM users")
    return int(row["c"]) if row else 0


async def count_users_by_tenant(tenant_id: str) -> int:
    row = await _one("SELECT COUNT(*) AS c FROM users WHERE tenant_id = ?", (tenant_id,))
    return int(row["c"]) if row else 0


async def delete_user_row(user_id: str) -> None:
    """物理删除用户行；threads/refresh_tokens 由外键 ON DELETE CASCADE 清除。"""

    await _execute("DELETE FROM users WHERE id = ?", (user_id,))


# ── refresh tokens ────────────────────────────────────────────────


async def create_refresh_token(
    *, jti: str, user_id: str, token_hash: str, expires_at: float
) -> None:
    await _execute(
        "INSERT INTO refresh_tokens (jti, user_id, token_hash, expires_at, "
        "revoked, created_at) VALUES (?, ?, ?,  ?, 0, ?)",
        (jti, user_id, token_hash, expires_at, time.time()),
    )


async def get_refresh_token(jti: str) -> dict[str, Any] | None:
    return await _one(
        "SELECT jti, user_id, token_hash, expires_at, revoked, created_at "
        "FROM refresh_tokens WHERE jti = ?",
        (jti,),
    )


async def revoke_refresh_token(jti: str) -> None:
    await _execute(
        "UPDATE refresh_tokens SET revoked = 1 WHERE jti = ?", (jti,)
    )


async def purge_expired_refresh_tokens() -> int:
    """物理删除已过期的 refresh token（含未撤销的），返回删除条数。"""

    return await _execute_rowcount(
        "DELETE FROM refresh_tokens WHERE expires_at < ?", (time.time(),)
    )


# ── threads（会话归属元数据；消息本体在 checkpoint 中） ────────────


async def create_thread(
    *, thread_id: str, user_id: str, tenant_id: str, title: str
) -> dict[str, Any]:
    now = time.time()
    await _execute(
        "INSERT INTO threads (id, user_id, tenant_id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (thread_id, user_id, tenant_id, title, now, now),
    )
    return {
        "id": thread_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
    }


async def get_thread(thread_id: str) -> dict[str, Any] | None:
    return await _one(
        "SELECT id, user_id, tenant_id, title, created_at, updated_at FROM threads WHERE id = ?",
        (thread_id,),
    )


async def list_threads(user_id: str) -> list[dict[str, Any]]:
    return await _all(
        "SELECT id, user_id, tenant_id, title, created_at, updated_at FROM threads "
        "WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    )


async def search_user_threads(
    tenant_id: str,
    user_id: str,
    *,
    limit: int,
    updated_after: float | None = None,
) -> list[dict[str, Any]]:
    """检索某用户在指定租户下的会话元数据（情景记忆候选）。

    多租户红线：tenant_id 显式入参并参与过滤，绝不仅凭 user_id 回退；
    按 updated_at 倒序，可选时间窗（Unix 秒）裁剪久远会话。
    """

    sql = (
        "SELECT id, user_id, tenant_id, title, created_at, updated_at FROM threads "
        "WHERE tenant_id = ? AND user_id = ?"
    )
    params: list[Any] = [tenant_id, user_id]
    if updated_after is not None:
        sql += " AND updated_at >= ?"
        params.append(updated_after)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(int(limit))
    return await _all(sql, tuple(params))


async def rename_thread(thread_id: str, title: str) -> None:
    await _execute(
        "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
        (title, time.time(), thread_id),
    )


async def touch_thread(thread_id: str) -> None:
    """会话有新消息时更新 updated_at（用于历史列表排序）。"""

    await _execute(
        "UPDATE threads SET updated_at = ? WHERE id = ?", (time.time(), thread_id)
    )


async def delete_thread_row(thread_id: str) -> None:
    await _execute("DELETE FROM threads WHERE id = ?", (thread_id,))


async def delete_checkpoint_thread(thread_id: str) -> None:
    """删除 LangGraph checkpoint 中的会话消息与状态。"""

    await get_saver().adelete_thread(thread_id)


async def list_all_thread_ids() -> Sequence[str]:
    """运维用途：列出全部会话 ID。"""

    rows = await _all("SELECT id FROM threads")
    return [r["id"] for r in rows]


# ── 每用户长期记忆（Store；namespace = ("tenant", tenant_id, "user_memory", user_id)） ──

# 记忆枚举上限：每用户实际条目数受 memory_max_items 限制，这里只需大于它
_MEMORY_LIST_LIMIT = 1000


def user_memory_namespace(tenant_id: str, user_id: str) -> tuple[str, ...]:
    """返回某用户的长期记忆 Store namespace（含租户隔离）。"""

    return ("tenant", tenant_id, "user_memory", user_id)


def _iso_time(value: Any) -> str | None:
    """Store 时间戳归一化为 ISO 字符串（PG 返回 datetime，SQLite 返回 str）。"""

    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _epoch_to_iso(value: Any) -> str | None:
    """事实 value 内的 Unix 秒时间戳归一化为 UTC ISO 字符串。"""

    if value is None:
        return None
    try:
        return datetime.datetime.fromtimestamp(float(value), tz=datetime.UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _store_item_to_view(item: Any, *, include_content: bool) -> dict[str, Any]:
    raw = item.value or {}
    content = raw.get("content", "")
    if isinstance(content, list):
        content = "\n".join(str(line) for line in content)
    content = str(content)
    view: dict[str, Any] = {
        "key": item.key,
        "size_bytes": len(content.encode("utf-8")),
        "created_at": _iso_time(getattr(item, "created_at", None)),
        "updated_at": _iso_time(getattr(item, "updated_at", None)),
    }
    if include_content:
        view["content"] = content
    return view


async def list_user_memory(tenant_id: str, user_id: str) -> list[dict[str, Any]]:
    """列出用户全部记忆条目的元信息（不含正文），按 key 排序。"""

    items = await get_store().asearch(
        user_memory_namespace(tenant_id, user_id), limit=_MEMORY_LIST_LIMIT
    )
    views = [_store_item_to_view(item, include_content=False) for item in items]
    return sorted(views, key=lambda v: v["key"])


async def get_user_memory_item(tenant_id: str, user_id: str, key: str) -> dict[str, Any] | None:
    """读取用户单个记忆条目（含正文）；不存在返回 None。"""

    item = await get_store().aget(user_memory_namespace(tenant_id, user_id), key)
    if item is None:
        return None
    return _store_item_to_view(item, include_content=True)


async def delete_user_memory_item(tenant_id: str, user_id: str, key: str) -> bool:
    """删除用户单个记忆条目；返回是否实际删除（不存在为 False）。"""

    ns = user_memory_namespace(tenant_id, user_id)
    if await get_store().aget(ns, key) is None:
        return False
    await get_store().adelete(ns, key)
    return True


async def purge_user_memory(tenant_id: str, user_id: str) -> int:
    """清空用户画像记忆（/memory/ 文件），返回删除条目数。"""

    items = await get_store().asearch(
        user_memory_namespace(tenant_id, user_id), limit=_MEMORY_LIST_LIMIT
    )
    for item in items:
        # prefix 搜索可能命中子 namespace，删除必须使用条目自身的精确 ns
        await get_store().adelete(item.namespace, item.key)
    if items:
        logger.info(
            "已清空用户画像记忆: tenant=%s user=%s, 条目数=%d",
            tenant_id, user_id, len(items),
        )
    return len(items)


# ── 语义事实记忆（namespace = ("user_facts", user_id)，向量索引） ──

# 与画像记忆使用不同顶级 namespace：Store 的 asearch 是前缀匹配，
# 分开顶级 namespace 可避免事实条目污染画像计数与列表。
_FACTS_LIST_LIMIT = 1000


def user_facts_namespace(tenant_id: str, user_id: str) -> tuple[str, ...]:
    """返回某用户语义事实记忆的 Store namespace（含租户隔离）。"""

    return ("tenant", tenant_id, "user_facts", user_id)


def _fact_to_view(item: Any) -> dict[str, Any]:
    raw = item.value or {}
    text = str(raw.get("text", ""))
    # 时序字段（双时间轴）：
    # valid_from=事实业务生效时刻；invalid_at=被取代而失效的时刻（None=仍有效）。
    # 旧数据没有 valid_from 时回退到 created_at，保证历史条目可参与 as-of 查询。
    valid_from = raw.get("valid_from")
    if valid_from is None:
        valid_from = raw.get("created_at")
    # 旧数据没有 status 字段时默认 active：新 schema 对历史条目向后兼容
    return {
        "key": item.key,
        "text": text,
        "category": str(raw.get("category", "general")),
        "size_bytes": len(text.encode("utf-8")),
        "score": getattr(item, "score", None),
        "created_at": _iso_time(getattr(item, "created_at", None)),
        "updated_at": _iso_time(getattr(item, "updated_at", None)),
        "valid_from": _epoch_to_iso(valid_from),
        "invalid_at": _epoch_to_iso(raw.get("invalid_at")),
        # active=当前有效（可召回、计配额）；superseded=被新事实取代的旧版本
        # （不进召回、不计配额，审计窗口内仍可在管理列表查看）
        "status": str(raw.get("status", "active")),
        "superseded_by": raw.get("superseded_by"),
        "supersedes": raw.get("supersedes"),
        "source": str(raw.get("source", "manual")),
    }


async def list_user_facts(tenant_id: str, user_id: str) -> list[dict[str, Any]]:
    """列出用户全部事实记忆（无查询，按更新时间倒序）。

    含 superseded 旧版本，供管理端审计；模型召回与配额口径请用
    :func:`list_active_user_facts`。
    """

    items = await get_store().asearch(
        user_facts_namespace(tenant_id, user_id), limit=_FACTS_LIST_LIMIT
    )
    views = [_fact_to_view(item) for item in items]
    return sorted(views, key=lambda v: v["updated_at"] or "", reverse=True)


async def list_active_user_facts(tenant_id: str, user_id: str) -> list[dict[str, Any]]:
    """列出当前有效（status=active）的事实记忆，旧数据缺省视为 active。"""

    all_views = await list_user_facts(tenant_id, user_id)
    return [v for v in all_views if v.get("status") == "active"]


async def get_user_fact(tenant_id: str, user_id: str, key: str) -> dict[str, Any] | None:
    item = await get_store().aget(user_facts_namespace(tenant_id, user_id), key)
    return _fact_to_view(item) if item is not None else None


async def delete_user_fact(tenant_id: str, user_id: str, key: str) -> bool:
    """删除单条事实记忆；不存在返回 False。"""

    ns = user_facts_namespace(tenant_id, user_id)
    if await get_store().aget(ns, key) is None:
        return False
    await get_store().adelete(ns, key)
    logger.info("已删除事实记忆: tenant=%s user=%s key=%s", tenant_id, user_id, key)
    return True


async def purge_user_facts(tenant_id: str, user_id: str) -> int:
    """清空用户全部事实记忆，返回删除条目数。"""

    items = await get_store().asearch(
        user_facts_namespace(tenant_id, user_id), limit=_FACTS_LIST_LIMIT
    )
    for item in items:
        await get_store().adelete(item.namespace, item.key)
    if items:
        logger.info(
            "已清空用户事实记忆: tenant=%s user=%s, 条目数=%d",
            tenant_id, user_id, len(items),
        )
    return len(items)


async def purge_user_data(tenant_id: str, user_id: str) -> dict[str, int]:
    """账户注销级联：checkpoint 对话 → 事实记忆 → 画像记忆 → 用户业务行。

    顺序保证：先清 LangGraph 侧数据（独立连接/连接池），最后删 users 行，
    threads 与 refresh_tokens 由外键 ON DELETE CASCADE 物理清除。
    任一步失败都不会留下"业务已删但记忆残留"的半成品。
    """

    thread_ids = [t["id"] for t in await list_threads(user_id)]
    for thread_id in thread_ids:
        await delete_checkpoint_thread(thread_id)

    fact_count = await purge_user_facts(tenant_id, user_id)
    memory_count = await purge_user_memory(tenant_id, user_id)
    await delete_user_row(user_id)

    logger.warning(
        "账户数据已级联清除: tenant=%s user=%s, 会话=%d, 画像记忆=%d, 事实记忆=%d",
        tenant_id,
        user_id,
        len(thread_ids),
        memory_count,
        fact_count,
    )
    return {"threads": len(thread_ids), "memories": memory_count, "facts": fact_count}


async def purge_tenant_memory(tenant_id: str) -> int:
    """清空某租户下全部记忆（画像 + 事实），用于租户删除时级联清理。"""

    total = 0
    for ns_prefix in (
        ("tenant", tenant_id, "user_memory"),
        ("tenant", tenant_id, "user_facts"),
    ):
        items = await get_store().asearch(ns_prefix, limit=100000)
        for item in items:
            await get_store().adelete(item.namespace, item.key)
            total += 1
    if total:
        logger.warning("已清空租户全部记忆: tenant=%s, 条目数=%d", tenant_id, total)
    return total


# ── 轻量 Wiki（wiki_pages + wiki_revisions，版本历史） ──────────────


async def create_wiki_page(
    *, tenant_id: str, slug: str, title: str, content: str, author_id: str,
    summary: str = "",
) -> dict[str, Any]:
    """创建 Wiki 页面并写入首个版本，返回页面信息（不含 content）。"""

    now = time.time()
    page_id = new_id()
    revision_id = new_id()
    await _execute(
        "INSERT INTO wiki_pages (id, tenant_id, slug, title, created_by, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (page_id, tenant_id, slug, title, author_id, now, now),
    )
    await _execute(
        "INSERT INTO wiki_revisions "
        "(id, page_id, tenant_id, content, author_id, summary, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (revision_id, page_id, tenant_id, content, author_id, summary, now),
    )
    return {
        "id": page_id,
        "tenant_id": tenant_id,
        "slug": slug,
        "title": title,
        "created_by": author_id,
        "created_at": now,
        "updated_at": now,
    }


async def get_wiki_page(tenant_id: str, slug: str) -> dict[str, Any] | None:
    """读取 Wiki 页面当前版本（含 content）。"""

    page = await _one(
        "SELECT id, tenant_id, slug, title, created_by, created_at, updated_at "
        "FROM wiki_pages WHERE tenant_id = ? AND slug = ?",
        (tenant_id, slug),
    )
    if page is None:
        return None
    revision = await _one(
        "SELECT content, author_id, summary, created_at FROM wiki_revisions "
        "WHERE page_id = ? ORDER BY created_at DESC LIMIT 1",
        (page["id"],),
    )
    if revision is None:
        return None
    return {
        **page,
        "content": revision["content"],
        "author_id": revision["author_id"],
        "summary": revision["summary"],
    }


async def list_wiki_pages(tenant_id: str) -> list[dict[str, Any]]:
    """列出租户全部 Wiki 页面（不含 content，按更新时间倒序）。"""

    return await _all(
        "SELECT id, tenant_id, slug, title, created_by, created_at, updated_at "
        "FROM wiki_pages WHERE tenant_id = ? ORDER BY updated_at DESC",
        (tenant_id,),
    )


async def update_wiki_page(
    *, tenant_id: str, slug: str, title: str, content: str, author_id: str,
    summary: str = "",
) -> dict[str, Any] | None:
    """更新 Wiki 页面：写入新版本 + 更新页面元信息。页面不存在返回 None。"""

    page = await _one(
        "SELECT id FROM wiki_pages WHERE tenant_id = ? AND slug = ?",
        (tenant_id, slug),
    )
    if page is None:
        return None
    now = time.time()
    revision_id = new_id()
    await _execute(
        "INSERT INTO wiki_revisions "
        "(id, page_id, tenant_id, content, author_id, summary, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (revision_id, page["id"], tenant_id, content, author_id, summary, now),
    )
    await _execute(
        "UPDATE wiki_pages SET title = ?, updated_at = ? WHERE id = ?",
        (title, now, page["id"]),
    )
    return {
        "id": page["id"],
        "tenant_id": tenant_id,
        "slug": slug,
        "title": title,
        "updated_at": now,
    }


async def delete_wiki_page(tenant_id: str, slug: str) -> bool:
    """删除 Wiki 页面及其全部版本。返回是否实际删除。"""

    page = await _one(
        "SELECT id FROM wiki_pages WHERE tenant_id = ? AND slug = ?",
        (tenant_id, slug),
    )
    if page is None:
        return False
    await _execute("DELETE FROM wiki_pages WHERE id = ?", (page["id"],))
    return True


async def list_wiki_revisions(tenant_id: str, slug: str) -> list[dict[str, Any]]:
    """列出页面的版本历史（不含 content，按时间倒序）。"""

    page = await _one(
        "SELECT id FROM wiki_pages WHERE tenant_id = ? AND slug = ?",
        (tenant_id, slug),
    )
    if page is None:
        return []
    return await _all(
        "SELECT id, author_id, summary, created_at FROM wiki_revisions "
        "WHERE page_id = ? ORDER BY created_at DESC",
        (page["id"],),
    )


async def get_wiki_revision(
    tenant_id: str, slug: str, revision_id: str
) -> dict[str, Any] | None:
    """读取指定页面某个历史版本的正文（供 diff 对比与回滚）。

    必须同时匹配租户与页面归属，跨租户/跨页面猜测 revision_id 一律返回 None。
    """

    return await _one(
        "SELECT r.id AS id, r.content AS content, r.summary AS summary, "
        "r.author_id AS author_id, r.created_at AS created_at "
        "FROM wiki_revisions r "
        "JOIN wiki_pages p ON p.id = r.page_id "
        "WHERE p.tenant_id = ? AND p.slug = ? AND r.id = ?",
        (tenant_id, slug, revision_id),
    )


async def search_wiki_pages(
    tenant_id: str, keyword: str, limit: int = 20
) -> list[dict[str, Any]]:
    """在当前租户 Wiki 页面的标题与最新版本正文中做关键词检索。

    检索路径按后端自动选择：
    - SQLite + FTS5 可用且关键词 ≥3 字符：走 FTS5 trigram 索引（MATCH 短语），
      异常时回退 LIKE；
    - SQLite 短词（1-2 字符，trigram 无法建索引）或 FTS5 不可用：LIKE；
    - Postgres：ILIKE（若 pg_trgm 扩展可用则由 GIN 索引加速）。

    只匹配每个页面的最新版本，标题命中优先排序。返回 slug/title/updated_at
    及最新正文（供上层生成匹配摘要片段）。
    """

    kw = keyword.strip()
    if not kw:
        return []
    pattern = _like_pattern(kw)

    # SQLite FTS5 trigram：仅对 ≥3 字符的关键词生效
    if _backend == "sqlite" and _wiki_fts_available and len(kw) >= 3:
        match_expr = '"' + kw.replace('"', '""') + '"'
        try:
            return await _all(
                "SELECT p.slug AS slug, p.title AS title, "
                "p.updated_at AS updated_at, f.content AS content "
                "FROM wiki_fts f "
                "JOIN wiki_pages p ON p.id = f.page_id "
                "WHERE f.tenant_id = ? AND wiki_fts MATCH ? "
                "ORDER BY CASE WHEN p.title LIKE ? ESCAPE '\\' THEN 0 ELSE 1 END, "
                "p.updated_at DESC LIMIT ?",
                (tenant_id, match_expr, pattern, limit),
            )
        except Exception as exc:  # pragma: no cover - MATCH 语法兜底
            logger.debug("FTS5 MATCH 失败，回退 LIKE：%s", exc)

    like_op = "ILIKE" if _backend == "postgres" else "LIKE"
    return await _all(
        "SELECT p.slug AS slug, p.title AS title, p.updated_at AS updated_at, "
        "r.content AS content "
        "FROM wiki_pages p "
        "JOIN wiki_revisions r "
        "  ON r.page_id = p.id "
        "  AND r.created_at = ("
        "    SELECT MAX(created_at) FROM wiki_revisions WHERE page_id = p.id"
        "  ) "
        f"WHERE p.tenant_id = ? AND (p.title {like_op} ? ESCAPE '\\' "
        f"OR r.content {like_op} ? ESCAPE '\\') "
        f"ORDER BY CASE WHEN p.title {like_op} ? ESCAPE '\\' THEN 0 ELSE 1 END, "
        "p.updated_at DESC LIMIT ?",
        (tenant_id, pattern, pattern, pattern, limit),
    )


async def add_wiki_attachment(
    *,
    tenant_id: str,
    page_id: str,
    filename: str,
    stored_name: str,
    size_bytes: int,
    content_type: str,
    uploaded_by: str,
) -> dict[str, Any]:
    """新增 Wiki 页面附件记录，返回附件元信息。"""

    now = time.time()
    attachment_id = new_id()
    await _execute(
        "INSERT INTO wiki_attachments "
        "(id, tenant_id, page_id, filename, stored_name, size_bytes, "
        " content_type, uploaded_by, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            attachment_id, tenant_id, page_id, filename, stored_name,
            size_bytes, content_type, uploaded_by, now,
        ),
    )
    return {
        "id": attachment_id,
        "tenant_id": tenant_id,
        "page_id": page_id,
        "filename": filename,
        "stored_name": stored_name,
        "size_bytes": size_bytes,
        "content_type": content_type,
        "uploaded_by": uploaded_by,
        "created_at": now,
    }


async def list_wiki_attachments(tenant_id: str, page_id: str) -> list[dict[str, Any]]:
    """列出某个 Wiki 页面的全部附件（按时间倒序）。"""

    return await _all(
        "SELECT id, tenant_id, page_id, filename, stored_name, size_bytes, "
        "content_type, uploaded_by, created_at FROM wiki_attachments "
        "WHERE tenant_id = ? AND page_id = ? ORDER BY created_at DESC",
        (tenant_id, page_id),
    )


async def get_wiki_attachment(
    tenant_id: str, attachment_id: str
) -> dict[str, Any] | None:
    """按附件 id 读取附件（租户 + id 双重匹配，防跨租户越权）。

    附带 JOIN 出的 slug，供上层定位附件落盘目录。
    """

    return await _one(
        "SELECT a.id AS id, a.tenant_id AS tenant_id, a.page_id AS page_id, "
        "p.slug AS slug, a.filename AS filename, a.stored_name AS stored_name, "
        "a.size_bytes AS size_bytes, a.content_type AS content_type, "
        "a.uploaded_by AS uploaded_by, a.created_at AS created_at "
        "FROM wiki_attachments a "
        "JOIN wiki_pages p ON p.id = a.page_id "
        "WHERE a.tenant_id = ? AND a.id = ?",
        (tenant_id, attachment_id),
    )


async def delete_wiki_attachment(
    tenant_id: str, attachment_id: str
) -> dict[str, Any] | None:
    """删除附件记录，返回被删行（含 stored_name 供删除文件）；不存在返回 None。"""

    row = await get_wiki_attachment(tenant_id, attachment_id)
    if row is None:
        return None
    await _execute("DELETE FROM wiki_attachments WHERE id = ?", (attachment_id,))
    return row


def _like_pattern(keyword: str) -> str:
    """转义 LIKE 通配符（%、_、\\）后包裹成包含匹配模式。"""

    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"
