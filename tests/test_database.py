"""持久化层单元测试：用户、refresh token、会话元数据、Store 隔离。

每个用例都跑在 conftest 提供的临时 SQLite 上（WAL + 外键已开启）。
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from src.persistence import database as db

pytestmark = pytest.mark.usefixtures("isolated_db")


# ── users ─────────────────────────────────────────────────────────


async def test_create_and_get_user() -> None:
    user_id = await db.create_user(
        email="alice@example.com",
        password_hash="scrypt$x$y",
        display_name="Alice",
    )
    assert len(user_id) == 32  # uuid4 hex

    fetched = await db.get_user_by_email("alice@example.com")
    assert fetched is not None
    assert fetched["id"] == user_id
    assert fetched["display_name"] == "Alice"
    assert (await db.get_user_by_id(user_id))["email"] == "alice@example.com"
    # init_db 会幂等播种 system 保留账号（自动沉淀 FK 主体），故默认租户共 2 个用户
    assert await db.count_users() == 2


async def test_duplicate_email_rejected() -> None:
    await db.create_user(
        email="dup@example.com", password_hash="h1", display_name="A"
    )
    with pytest.raises(sqlite3.IntegrityError):
        await db.create_user(
            email="dup@example.com", password_hash="h2", display_name="B"
        )


async def test_get_missing_user_returns_none() -> None:
    assert await db.get_user_by_email("nope@example.com") is None
    assert await db.get_user_by_id(db.new_id()) is None


# ── refresh tokens ────────────────────────────────────────────────


async def test_refresh_token_revoke_and_purge() -> None:
    user_id = await db.create_user(
        email="t@example.com", password_hash="h", display_name="T"
    )
    jti = db.new_id()
    await db.create_refresh_token(
        jti=jti,
        user_id=user_id,
        token_hash="deadbeef",
        expires_at=time.time() + 3600,
    )
    record = await db.get_refresh_token(jti)
    assert record is not None
    assert record["revoked"] == 0

    await db.revoke_refresh_token(jti)
    assert (await db.get_refresh_token(jti))["revoked"] == 1

    # 过期 token 由启动清理任务物理删除
    old_jti = db.new_id()
    await db.create_refresh_token(
        jti=old_jti,
        user_id=user_id,
        token_hash="old",
        expires_at=time.time() - 1,
    )
    assert await db.purge_expired_refresh_tokens() >= 1
    assert await db.get_refresh_token(old_jti) is None


async def test_user_delete_cascades_tokens_and_threads() -> None:
    user_id = await db.create_user(
        email="cas@example.com", password_hash="h", display_name="Cas"
    )
    await db.create_refresh_token(
        jti=db.new_id(),
        user_id=user_id,
        token_hash="h",
        expires_at=time.time() + 100,
    )
    await db.create_thread(
        thread_id="t-1", user_id=user_id,
        tenant_id=db.DEFAULT_TENANT_ID, title="x",
    )

    await db._conn().execute("DELETE FROM users WHERE id = ?", (user_id,))
    await db._conn().commit()

    async with db._conn().execute(
        "SELECT COUNT(*) AS c FROM refresh_tokens WHERE user_id = ?", (user_id,)
    ) as cur:
        assert (await cur.fetchone())["c"] == 0
    assert await db.list_threads(user_id) == []


# ── threads ────────────────────────────────────────────────────────


async def test_thread_crud_and_listing_order() -> None:
    user_id = await db.create_user(
        email="th@example.com", password_hash="h", display_name="Th"
    )
    other_id = await db.create_user(
        email="other@example.com", password_hash="h", display_name="Other"
    )
    await db.create_thread(
        thread_id="a", user_id=user_id,
        tenant_id=db.DEFAULT_TENANT_ID, title="A",
    )
    await asyncio.sleep(0.02)
    await db.create_thread(
        thread_id="b", user_id=user_id,
        tenant_id=db.DEFAULT_TENANT_ID, title="B",
    )

    # b 更新，a 随后被 touch → a 排最前（updated_at DESC）
    await asyncio.sleep(0.02)
    await db.touch_thread("a")

    threads = await db.list_threads(user_id)
    assert [t["id"] for t in threads] == ["a", "b"]
    # 其他用户的会话不可见
    assert await db.list_threads(other_id) == []

    await db.rename_thread("a", "新标题")
    assert (await db.get_thread("a"))["title"] == "新标题"

    await db.delete_thread_row("a")
    assert await db.get_thread("a") is None


# ── saver / store 单例与用户级记忆隔离 ────────────────────────────


async def test_saver_and_store_are_async_and_ready() -> None:
    assert isinstance(db.get_saver(), AsyncSqliteSaver)
    assert isinstance(db.get_store(), AsyncSqliteStore)
    # 删除不存在的会话也不应报错（DELETE 端点依赖此行为幂等）
    await db.delete_checkpoint_thread("no-such-thread")


async def test_long_term_memory_namespace_isolation() -> None:
    """Store 层防御纵深：不同 user_id 的 namespace 互不可见。"""

    alice = await db.create_user(
        email="alice@example.com", password_hash="h", display_name="Alice"
    )
    bob = await db.create_user(
        email="bob@example.com", password_hash="h", display_name="Bob"
    )
    store = db.get_store()
    await store.aput(
        ("user_memory", alice), "/AGENTS.md", {"content": "爱丽丝的秘密"}
    )
    await store.aput(
        ("user_memory", bob), "/AGENTS.md", {"content": "Bob notes"}
    )

    alice_item = await store.aget(("user_memory", alice), "/AGENTS.md")
    bob_item = await store.aget(("user_memory", bob), "/AGENTS.md")
    assert alice_item is not None and alice_item.value["content"] == "爱丽丝的秘密"
    assert bob_item is not None and bob_item.value["content"] == "Bob notes"

    # 跨 namespace 用同名 key 读取 → 查无此条，证明前缀隔离生效
    assert await store.aget(("user_memory", bob), "/AGENTS.md" + "x") is None
    stranger = db.new_id()
    assert await store.aget(("user_memory", stranger), "/AGENTS.md") is None
