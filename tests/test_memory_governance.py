"""长期记忆治理测试。

覆盖三层：
1. 纯策略：凭据扫描、字节/条目数上限；
2. GuardedStoreBackend：写入/编辑拦截、违禁内容不落盘、覆盖不受条目数限制；
3. HTTP 接口：/memory 查看/删除、跨用户隔离、注销账户级联清除。

接口测试与 test_auth 相同：不经过 lifespan，db 由 isolated_db 夹具准备，
使用 httpx AsyncClient + ASGITransport 以复用同一事件循环的连接。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.agents.graph import GuardedStoreBackend
from src.api import auth
from src.api.memory import router as memory_router
from src.memory.guard import MemoryRejectedError, ensure_write_allowed, scan_secret
from src.persistence import database as db

pytestmark = pytest.mark.usefixtures("isolated_db")


# ── 1. 纯策略函数 ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "secret",
    [
        "-----BEGIN RSA PRIVATE KEY-----\nMIIE...",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_0123456789abcdefABCDEF0123456789",
        "xoxb-1234567890-abcdEfghIJKL",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signaturepart",
        'api_key = "sk-abcdef1234567890ABCDEF"',
        "password: hunter2hunter2",
    ],
)
def test_scan_secret_detects_credentials(secret: str) -> None:
    assert scan_secret(secret) is not None


@pytest.mark.parametrize(
    "benign",
    [
        "# 用户记忆\n- 岗位：数据团队负责人\n- 偏好：列表式回答",
        "用户提到旧 token 已经过期，需要重新申请",
        "项目代号 Aurora-7，每周三同步进度",
    ],
)
def test_scan_secret_allows_benign_memory(benign: str) -> None:
    assert scan_secret(benign) is None


def test_size_and_item_limits_raise() -> None:
    with pytest.raises(MemoryRejectedError, match="超过上限"):
        ensure_write_allowed(
            "x" * 257,
            max_file_bytes=256,
            new_file=True,
            item_count=0,
            max_items=10,
        )
    with pytest.raises(MemoryRejectedError, match="条目数"):
        ensure_write_allowed(
            "ok", max_file_bytes=256, new_file=True, item_count=10, max_items=10
        )


# ── 2. GuardedStoreBackend ────────────────────────────────────────


def _guarded_backend(
    user_id: str = "user-guard",
    *,
    tenant_id: str = db.DEFAULT_TENANT_ID,
    max_file_bytes: int = 8192,
    max_items: int = 20,
) -> GuardedStoreBackend:
    return GuardedStoreBackend(
        store=db.get_store(),
        namespace=lambda rt: ("tenant", tenant_id, "user_memory", user_id),
        max_file_bytes=max_file_bytes,
        max_items=max_items,
    )


async def test_guarded_write_persists_benign_memory() -> None:
    backend = _guarded_backend()
    result = await backend.awrite("/AGENTS.md", "# 用户记忆\n- 岗位：产品经理")
    assert result.error is None
    item = await db.get_store().aget(
        ("tenant", db.DEFAULT_TENANT_ID, "user_memory", "user-guard"),
        "/AGENTS.md",
    )
    assert item is not None
    assert "产品经理" in item.value["content"]


async def test_guarded_write_rejects_credentials_without_persisting() -> None:
    backend = _guarded_backend(user_id="user-secret")
    result = await backend.awrite("/AGENTS.md", "aws key is AKIAIOSFODNN7EXAMPLE")
    assert result.error is not None
    assert "凭据" in result.error
    item = await db.get_store().aget(
        ("tenant", db.DEFAULT_TENANT_ID, "user_memory", "user-secret"),
        "/AGENTS.md",
    )
    assert item is None  # 违禁内容绝不全量落盘


async def test_guarded_edit_blocked_and_file_unchanged() -> None:
    backend = _guarded_backend(user_id="user-edit")
    await backend.awrite("/AGENTS.md", "# 用户记忆\n- 偏好：简洁回答")

    result = await backend.aedit(
        "/AGENTS.md",
        "简洁回答",
        "简洁回答\n密钥 AKIAIOSFODNN7EXAMPLE",
    )
    assert result.error is not None
    read = await backend.aread("/AGENTS.md")
    assert read.file_data is not None
    assert "AKIA" not in read.file_data["content"]


async def test_guarded_enforces_item_count_but_allows_overwrite() -> None:
    backend = _guarded_backend(user_id="user-cap", max_items=2)
    assert (await backend.awrite("/a.md", "第一条")).error is None
    assert (await backend.awrite("/b.md", "第二条")).error is None
    third = await backend.awrite("/c.md", "第三条")
    assert third.error is not None and "条目数" in third.error

    # 覆盖已有文件不应受条目数限制
    overwrite = await backend.awrite("/a.md", "第一条（更新）")
    assert overwrite.error is None


async def test_guarded_enforces_file_size() -> None:
    backend = _guarded_backend(user_id="user-size", max_file_bytes=100)
    result = await backend.awrite("/AGENTS.md", "x" * 101)
    assert result.error is not None and "字节" in result.error


# ── 3. HTTP 接口 / 隔离 / 注销级联 ─────────────────────────────────


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(memory_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register_and_token(
    client: AsyncClient, email: str, password: str = "Passw0rd!"
) -> tuple[str, str]:
    resp = await client.post(
        "/auth/register", json={"email": email, "password": password}
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return body["user"]["id"], body["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_memory_api_lifecycle_and_isolation(client: AsyncClient) -> None:
    user_a, token_a = await _register_and_token(client, "alice@example.com")
    _, token_b = await _register_and_token(client, "bob@example.com")

    # 模型通过守卫后端为 A 写入一条记忆
    backend = _guarded_backend(user_a)
    assert (await backend.awrite("/AGENTS.md", "# 用户记忆\n- 岗位：数据团队负责人")).error is None

    # A 能看到记忆与配额
    resp = await client.get("/memory", headers=_auth(token_a))
    assert resp.status_code == 200
    body = resp.json()
    assert body["item_count"] == 1
    assert body["items"][0]["key"] == "/AGENTS.md"
    assert body["total_bytes"] > 0
    assert body["max_items"] == 20

    content = await client.get(
        "/memory/content", params={"key": "/AGENTS.md"}, headers=_auth(token_a)
    )
    assert content.status_code == 200
    assert "数据团队负责人" in content.json()["content"]

    # B 的 namespace 完全隔离，看不到 A 的记忆
    bob_list = await client.get("/memory", headers=_auth(token_b))
    assert bob_list.json()["item_count"] == 0
    bob_content = await client.get(
        "/memory/content", params={"key": "/AGENTS.md"}, headers=_auth(token_b)
    )
    assert bob_content.status_code == 404

    # A 删除单条
    deleted = await client.request(
        "DELETE", "/memory/item", params={"key": "/AGENTS.md"}, headers=_auth(token_a)
    )
    assert deleted.status_code == 204
    again = await client.request(
        "DELETE", "/memory/item", params={"key": "/AGENTS.md"}, headers=_auth(token_a)
    )
    assert again.status_code == 404

    # 畸形 key 拒绝
    bad = await client.request(
        "DELETE", "/memory/item", params={"key": "../etc/passwd"}, headers=_auth(token_a)
    )
    assert bad.status_code == 400

    # 未认证 401
    assert (await client.get("/memory")).status_code == 401


async def test_memory_clear_endpoint(client: AsyncClient) -> None:
    user_id, token = await _register_and_token(client, "carol@example.com")
    backend = _guarded_backend(user_id, max_items=20)
    await backend.awrite("/AGENTS.md", "画像")
    await backend.awrite("/notes.md", "其他结论")

    resp = await client.request("DELETE", "/memory", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    assert await db.list_user_memory(db.DEFAULT_TENANT_ID, user_id) == []


async def test_account_deletion_cascades_all_memory(client: AsyncClient) -> None:
    user_id, token = await _register_and_token(client, "dave@example.com")

    # 会话业务行 + 长期记忆 + refresh token 各留一份数据
    await db.create_thread(
        thread_id=db.new_id(), user_id=user_id,
        tenant_id=db.DEFAULT_TENANT_ID, title="旧会话",
    )
    backend = _guarded_backend(user_id)
    await backend.awrite("/AGENTS.md", "# 用户记忆\n- 岗位：风控")
    await auth._issue_token_pair(user_id, db.DEFAULT_TENANT_ID, db.ROLE_MEMBER)

    result = await db.purge_user_data(db.DEFAULT_TENANT_ID, user_id)
    assert result == {"threads": 1, "memories": 1, "facts": 0}

    # 用户行、会话、记忆、token 全部清除
    assert await db.get_user_by_id(user_id) is None
    assert await db.list_threads(user_id) == []
    assert await db.list_user_memory(db.DEFAULT_TENANT_ID, user_id) == []

    # HTTP 路径：注销后旧 access token 立即失效
    assert (await client.get("/auth/me", headers=_auth(token))).status_code == 401


async def test_delete_account_endpoint(client: AsyncClient) -> None:
    user_id, token = await _register_and_token(client, "erin@example.com")
    backend = _guarded_backend(user_id)
    await backend.awrite("/AGENTS.md", "待注销用户的记忆")

    resp = await client.request("DELETE", "/auth/account", headers=_auth(token))
    assert resp.status_code == 204
    assert await db.get_user_by_id(user_id) is None
    assert await db.list_user_memory(db.DEFAULT_TENANT_ID, user_id) == []
