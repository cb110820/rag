"""多租户隔离测试（P4）。

覆盖：
1. 跨租户记忆隔离：A 租户用户的事实记忆 B 租户用户不可见；
2. 禁用租户登录拒绝（403）；
3. 平台管理 API 鉴权：非 platform_admin 返回 403，platform_admin 可 CRUD 租户；
4. 默认租户不可删除。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api import auth
from src.api.admin import router as admin_router
from src.api.memory import router as memory_router
from src.memory.tools import _save_memory
from src.persistence import database as db

pytestmark = pytest.mark.usefixtures("isolated_db")

TENANT_A = db.DEFAULT_TENANT_ID
TENANT_B = "tenant-b"


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(memory_router)
    app.include_router(admin_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _register(
    client: AsyncClient, email: str, tenant_id: str = TENANT_A
) -> tuple[str, str]:
    # 非 default 租户必须先在 tenants 表中存在（users.tenant_id 外键约束）。
    # create_tenant 会生成随机 id，这里用固定 slug 反查真实 id。
    actual_tenant = tenant_id
    if tenant_id != TENANT_A:
        existing = await db.get_tenant_by_slug(f"slug-{tenant_id}")
        if existing is None:
            await db.create_tenant(name=tenant_id, slug=f"slug-{tenant_id}")
            existing = await db.get_tenant_by_slug(f"slug-{tenant_id}")
        actual_tenant = existing["id"]

    resp = await client.post(
        "/auth/register", json={"email": email, "password": "Passw0rd!"}
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    if tenant_id != TENANT_A:
        await db._conn().execute(
            "UPDATE users SET tenant_id = ? WHERE id = ?",
            (actual_tenant, body["user"]["id"]),
        )
        await db._conn().commit()
        body["user"]["tenant_id"] = actual_tenant
    return body["user"]["id"], body["access_token"]


# ── 1. 跨租户记忆隔离 ──────────────────────────────────────────────


async def test_fact_memory_isolated_across_tenants(client: AsyncClient) -> None:
    """A 租户写入的事实记忆，B 租户用户通过 API 看不到。"""

    uid_a, token_a = await _register(client, "ta@example.com", TENANT_A)
    _, token_b = await _register(client, "tb@example.com", TENANT_B)

    # A 写入一条事实记忆
    await _save_memory(TENANT_A, uid_a, "A 租户的项目代号是 Aurora", "project")

    # A 自己能看到
    resp_a = await client.get("/memory/facts", headers=_auth(token_a))
    assert resp_a.status_code == 200
    assert resp_a.json()["item_count"] == 1

    # B 看不到 A 的事实
    resp_b = await client.get("/memory/facts", headers=_auth(token_b))
    assert resp_b.status_code == 200
    assert resp_b.json()["item_count"] == 0


# ── 2. 禁用租户登录拒绝 ────────────────────────────────────────────


async def test_suspended_tenant_login_forbidden(client: AsyncClient) -> None:
    """租户被设为 suspended 后，其用户登录返回 403。"""

    # 建一个新租户并禁用
    tenant_id = await db.create_tenant(name="Suspended Co", slug="suspended-co", status="suspended")
    # 直接在该租户下创建用户（绕过 register 的 default 租户逻辑）
    await db.create_user(
        email="suspended@example.com",
        password_hash=auth.hash_password("Passw0rd!"),
        display_name="Suspended User",
        tenant_id=tenant_id,
    )

    resp = await client.post(
        "/auth/login",
        json={"email": "suspended@example.com", "password": "Passw0rd!"},
    )
    assert resp.status_code == 403


# ── 3. 平台管理 API 鉴权 ───────────────────────────────────────────


async def _make_platform_admin(email: str = "admin@example.com") -> str:
    """直接在 DB 中创建一个 platform_admin 用户，返回其 access token。"""

    user_id = await db.create_user(
        email=email,
        password_hash=auth.hash_password("Passw0rd!"),
        display_name="Platform Admin",
        tenant_id=db.DEFAULT_TENANT_ID,
        role=db.ROLE_PLATFORM_ADMIN,
    )
    tokens = await auth._issue_token_pair(
        user_id, db.DEFAULT_TENANT_ID, db.ROLE_PLATFORM_ADMIN
    )
    return tokens["access_token"]


async def test_non_admin_cannot_access_admin_api(client: AsyncClient) -> None:
    """普通 member 用户访问 /api/admin/tenants 返回 403。"""

    _, token = await _register(client, "member@example.com")
    resp = await client.get("/api/admin/tenants", headers=_auth(token))
    assert resp.status_code == 403


async def test_admin_can_crud_tenants(client: AsyncClient) -> None:
    """platform_admin 可创建、查询、列表、删除租户（default 除外）。"""

    admin_token = await _make_platform_admin()

    # 列表：至少有 default
    lst = await client.get("/api/admin/tenants", headers=_auth(admin_token))
    assert lst.status_code == 200
    assert any(t["slug"] == "default" for t in lst.json())

    # 创建
    created = await client.post(
        "/api/admin/tenants",
        headers=_auth(admin_token),
        json={"name": "Acme Inc", "slug": "acme-inc"},
    )
    assert created.status_code == 201
    tid = created.json()["id"]
    assert created.json()["max_storage_mb"] == 100  # 默认值

    # 查询
    got = await client.get(f"/api/admin/tenants/{tid}", headers=_auth(admin_token))
    assert got.status_code == 200
    assert got.json()["name"] == "Acme Inc"

    # 更新配额
    patched = await client.patch(
        f"/api/admin/tenants/{tid}",
        headers=_auth(admin_token),
        json={"max_storage_mb": 500, "status": "suspended"},
    )
    assert patched.status_code == 200
    assert patched.json()["max_storage_mb"] == 500
    assert patched.json()["status"] == "suspended"

    # 统计
    stats = await client.get(
        f"/api/admin/tenants/{tid}/stats", headers=_auth(admin_token)
    )
    assert stats.status_code == 200
    assert stats.json()["user_count"] == 0

    # 删除
    deleted = await client.delete(
        f"/api/admin/tenants/{tid}", headers=_auth(admin_token)
    )
    assert deleted.status_code == 204

    # 删除后查不到
    gone = await client.get(f"/api/admin/tenants/{tid}", headers=_auth(admin_token))
    assert gone.status_code == 404


async def test_default_tenant_cannot_be_deleted(client: AsyncClient) -> None:
    """default 租户受保护，平台管理员也不能删除。"""

    admin_token = await _make_platform_admin()
    resp = await client.delete(
        f"/api/admin/tenants/{db.DEFAULT_TENANT_ID}", headers=_auth(admin_token)
    )
    assert resp.status_code == 400


async def test_duplicate_slug_rejected(client: AsyncClient) -> None:
    """创建租户时 slug 重复返回 409。"""

    admin_token = await _make_platform_admin()
    await client.post(
        "/api/admin/tenants",
        headers=_auth(admin_token),
        json={"name": "First", "slug": "dup-slug"},
    )
    dup = await client.post(
        "/api/admin/tenants",
        headers=_auth(admin_token),
        json={"name": "Second", "slug": "dup-slug"},
    )
    assert dup.status_code == 409
