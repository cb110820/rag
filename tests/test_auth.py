"""认证模块单元测试。

两层覆盖：

1. 纯函数：scrypt 哈希、JWT 签发/解码/过期/类型、凭据格式校验；
2. ASGI 接口（不经过 lifespan，db 由 isolated_db 夹具准备）：
   注册/登录/me/refresh 轮换/登出/限流/注册开关。

注意：业务库连接是 aiosqlite，绑定在 pytest-asyncio 的事件循环上，
因此接口测试必须使用 ``httpx.AsyncClient + ASGITransport``
（同一事件循环）；starlette ``TestClient`` 自带独立 portal，
跨循环复用连接会死锁。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api import auth
from src.persistence import database as db
from src.rag.config import Settings

pytestmark = pytest.mark.usefixtures("isolated_db")


# ── scrypt 密码哈希 ───────────────────────────────────────────────


def test_scrypt_hash_roundtrip() -> None:
    stored = auth.hash_password("Passw0rd!")
    assert stored.startswith("scrypt$32768$8$1$")
    assert auth.verify_password("Passw0rd!", stored) is True
    assert auth.verify_password("wrong-password", stored) is False


def test_scrypt_uses_random_salt_and_rejects_tampered_hash() -> None:
    h1 = auth.hash_password("same-pass-123")
    h2 = auth.hash_password("same-pass-123")
    assert h1 != h2  # 随机盐 → 同一密码两次哈希不同

    assert auth.verify_password("x", "not-a-valid-hash") is False
    assert auth.verify_password("x", "bcrypt$32768$8$1$aa$bb") is False
    scheme, n, r, p, salt, digest = h1.split("$")
    tampered = "$".join([scheme, n, r, p, salt, "0" * len(digest)])
    assert auth.verify_password("same-pass-123", tampered) is False


# ── JWT ───────────────────────────────────────────────────────────


async def test_token_encode_and_typed_decode() -> None:
    token, jti = auth._encode_token("user-1", "access", 600)
    payload = auth._decode_token(token, "access")
    assert payload["sub"] == "user-1"
    assert payload["jti"] == jti
    assert payload["type"] == "access"

    # 类型不匹配（拿 access token 当 refresh 用）必须拒绝
    with pytest.raises(auth.HTTPException) as exc:
        auth._decode_token(token, "refresh")
    assert exc.value.status_code == 401


def test_expired_and_forged_token_rejected() -> None:
    secret = auth.get_jwt_secret()
    expired = jwt.encode(
        {"sub": "u", "jti": "j", "type": "access", "iat": 0, "exp": int(time.time()) - 10},
        secret,
        algorithm="HS256",
    )
    with pytest.raises(auth.HTTPException) as exc:
        auth._decode_token(expired, "access")
    assert exc.value.status_code == 401

    forged = jwt.encode(
        {"sub": "u", "type": "access", "exp": int(time.time()) + 100},
        "wrong-secret",
        algorithm="HS256",
    )
    with pytest.raises(auth.HTTPException) as exc:
        auth._decode_token(forged, "access")
    assert exc.value.status_code == 401


def test_validate_credentials() -> None:
    assert auth._validate_credentials("  Alice@Example.COM ", "abcdefgh") == (
        "alice@example.com"
    )
    with pytest.raises(auth.HTTPException) as exc:
        auth._validate_credentials("bad-email", "abcdefgh")
    assert exc.value.status_code == 400
    with pytest.raises(auth.HTTPException) as exc:
        auth._validate_credentials("a@b.com", "short")
    assert exc.value.status_code == 400


# ── ASGI 接口级（httpx 同循环 ASGI 传输） ─────────────────────────


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    app.include_router(auth.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(
    client: AsyncClient, email: str, password: str = "Passw0rd!"
) -> dict:
    resp = await client.post(
        "/auth/register", json={"email": email, "password": password}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def test_register_login_me_flow(client: AsyncClient) -> None:
    data = await _register(client, "alice@example.com")
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 30 * 60
    # 未提供 display_name 时用邮箱前缀
    assert data["user"]["display_name"] == "alice"
    assert data["user"]["email"] == "alice@example.com"

    # /me 需要令牌
    assert (await client.get("/auth/me")).status_code == 401
    me = await client.get(
        "/auth/me", headers={"Authorization": f"Bearer {data['access_token']}"}
    )
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"

    # 伪造令牌
    forged = await client.get(
        "/auth/me", headers={"Authorization": "Bearer not-a-jwt"}
    )
    assert forged.status_code == 401

    # 登录（邮箱大小写不敏感）
    ok = await client.post(
        "/auth/login",
        json={"email": "ALICE@example.com", "password": "Passw0rd!"},
    )
    assert ok.status_code == 200
    bad = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "wrong-pass"},
    )
    assert bad.status_code == 401


async def test_duplicate_register_and_weak_password(client: AsyncClient) -> None:
    await _register(client, "dup@example.com")
    again = await client.post(
        "/auth/register",
        json={"email": "dup@example.com", "password": "Passw0rd!"},
    )
    assert again.status_code == 409

    weak = await client.post(
        "/auth/register", json={"email": "x@y.com", "password": "123"}
    )
    assert weak.status_code == 422


async def test_refresh_rotation_and_replay_guard(client: AsyncClient) -> None:
    tokens = await _register(client, "rot@example.com")

    # 拿 access token 调刷新接口 → 类型错误
    wrong_type = await client.post(
        "/auth/refresh", json={"refresh_token": tokens["access_token"]}
    )
    assert wrong_type.status_code == 401

    rotated = await client.post(
        "/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert rotated.status_code == 200
    new_pair = rotated.json()
    assert new_pair["refresh_token"] != tokens["refresh_token"]

    # 旧 refresh token 重放必须失败（轮换后已撤销）
    replay = await client.post(
        "/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert replay.status_code == 401

    # 新 access 可用
    me = await client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {new_pair['access_token']}"},
    )
    assert me.status_code == 200


async def test_logout_revokes_refresh_token(client: AsyncClient) -> None:
    tokens = await _register(client, "lo@example.com")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    # 未认证不能登出
    no_auth = await client.post(
        "/auth/logout", json={"refresh_token": tokens["refresh_token"]}
    )
    assert no_auth.status_code == 401

    logged_out = await client.post(
        "/auth/logout",
        json={"refresh_token": tokens["refresh_token"]},
        headers=headers,
    )
    assert logged_out.status_code == 204

    # 登出后 refresh 立即失效
    replay = await client.post(
        "/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert replay.status_code == 401

    # DB 中记录确已撤销
    payload = jwt.decode(
        tokens["refresh_token"],
        auth.get_jwt_secret(),
        algorithms=["HS256"],
    )
    record = await db.get_refresh_token(payload["jti"])
    assert record is not None and record["revoked"] == 1


async def test_login_rate_limit_after_8_failures(client: AsyncClient) -> None:
    await _register(client, "limited@example.com")
    for _ in range(auth._LOGIN_MAX_FAILURES):
        resp = await client.post(
            "/auth/login",
            json={"email": "limited@example.com", "password": "bad-pass-1"},
        )
        assert resp.status_code == 401
    # 第 9 次在窗口内直接被限流，即使密码正确也不放行
    blocked = await client.post(
        "/auth/login",
        json={"email": "limited@example.com", "password": "Passw0rd!"},
    )
    assert blocked.status_code == 429


async def test_registration_disabled_returns_403(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed_settings = Settings(
        jwt_secret_key="test-secret-key-please-do-not-use-in-production-0123456789",
        allow_registration=False,
    )
    monkeypatch.setattr(auth, "get_settings", lambda: closed_settings)
    resp = await client.post(
        "/auth/register",
        json={"email": "late@example.com", "password": "Passw0rd!"},
    )
    assert resp.status_code == 403
