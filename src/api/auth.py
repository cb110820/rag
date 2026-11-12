"""账户认证：注册/登录/令牌刷新/登出/当前用户。

生产级要点：

- 密码哈希使用标准库 ``hashlib.scrypt``（OWASP 推荐算法之一），
  每用户独立随机盐，不依赖 bcrypt/passlib 等需要额外原生编译的包；
- 双令牌：access token（默认 30 分钟，无状态 JWT）+ refresh token
  （默认 7 天，SHA-256 落库、可撤销）；刷新时轮换并作废旧 refresh token；
- 登录失败限流：同一邮箱 + 来源 IP 滑动窗口内最多失败 N 次；
- JWT 密钥优先读环境变量 ``JWT_SECRET_KEY``，未配置时生成一次性密钥
  持久化到 .data/.jwt_secret 并在启动日志告警（仅适合单机部署）。
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Annotated, Any

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.persistence import database as db
from src.rag.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# ── 常量 ──────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_PASSWORD_MIN = 8
_PASSWORD_MAX = 128
_ALGORITHM = "HS256"

# scrypt 参数（OWASP 建议档：n=32768, r=8, p=1，工作区约 32MB/次）
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_SALT_BYTES = 16
# OpenSSL scrypt 默认 maxmem=32MiB，与工作区大小卡边会报 memory limit exceeded，
# 显式放宽到 64MiB
_SCRYPT_MAXMEM = 64 * 1024 * 1024

# 登录失败限流：每 (邮箱, IP) 窗口内最多失败次数
_LOGIN_MAX_FAILURES = 8
_LOGIN_WINDOW_SECONDS = 5 * 60

# ── 登录失败限流（单实例内存计数；多副本部署需换 Redis） ──────────
_login_failures: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = Lock()


# ── 密码哈希（scrypt） ────────────────────────────────────────────


def hash_password(password: str) -> str:
    """生成 scrypt 密码哈希字符串：scrypt$n$r$p$salt_hex$hash_hex。"""

    salt = secrets.token_bytes(_SCRYPT_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        maxmem=_SCRYPT_MAXMEM,
        dklen=_SCRYPT_DKLEN,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """常量时间比较校验密码；哈希格式非法时一律返回 False。"""

    try:
        scheme, n_s, r_s, p_s, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n_s),
            r=int(r_s),
            p=int(p_s),
            maxmem=_SCRYPT_MAXMEM,
            dklen=len(hash_hex) // 2,
        )
        return hmac.compare_digest(derived.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


# ── JWT 密钥与签发 ────────────────────────────────────────────────


def get_jwt_secret() -> str:
    """获取 JWT 签名密钥：环境变量优先，否则读取/生成持久化随机密钥。"""

    settings = get_settings()
    if settings.jwt_secret_key.strip():
        return settings.jwt_secret_key.strip()

    path = settings.jwt_secret_file
    if path.exists():
        secret = path.read_text(encoding="utf-8").strip()
        if secret:
            return secret

    secret = secrets.token_urlsafe(48)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secret, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows 下 chmod 语义有限，忽略即可
        pass
    logger.warning(
        "未配置 JWT_SECRET_KEY，已生成随机密钥并持久化到 %s。"
        "多副本/容器化部署时必须在环境变量中显式配置固定密钥，"
        "否则各实例签发的令牌无法互认、重启后旧令牌失效。",
        path,
    )
    return secret


def _now() -> int:
    return int(time.time())


def _encode_token(
    user_id: str,
    token_type: str,
    expires_delta_seconds: int,
    tenant_id: str = "",
    role: str = "member",
) -> tuple[str, str]:
    """签发 JWT，返回 (原始 token, jti)。"""

    jti = db.new_id()
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "jti": jti,
        "type": token_type,
        "iat": _now(),
        "exp": _now() + expires_delta_seconds,
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=_ALGORITHM), jti


async def _issue_token_pair(user_id: str, tenant_id: str, role: str) -> dict[str, Any]:
    """签发 access + refresh 令牌对，refresh 的哈希落库。"""

    settings = get_settings()
    access_token, _ = _encode_token(
        user_id, "access", settings.access_token_ttl_minutes * 60, tenant_id, role
    )
    refresh_token, refresh_jti = _encode_token(
        user_id, "refresh", settings.refresh_token_ttl_days * 24 * 3600, tenant_id, role
    )
    await db.create_refresh_token(
        jti=refresh_jti,
        user_id=user_id,
        token_hash=hashlib.sha256(refresh_token.encode("utf-8")).hexdigest(),
        expires_at=float(_now() + settings.refresh_token_ttl_days * 24 * 3600),
    )
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_ttl_minutes * 60,
    }


def _decode_token(token: str, expected_type: str) -> dict[str, Any]:
    """解码并校验 JWT 类型；任何异常统一转 401。"""

    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="令牌已过期") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="无效令牌") from exc
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=401, detail="令牌类型错误")
    return payload


# ── 请求模型 ───────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=_PASSWORD_MIN, max_length=_PASSWORD_MAX)
    display_name: str = Field(default="", max_length=64)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=_PASSWORD_MAX)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=10, max_length=4096)


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    tenant_id: str
    role: str


def _public_user(user: dict[str, Any]) -> UserResponse:
    return UserResponse(
        id=user["id"],
        email=user["email"],
        display_name=user["display_name"],
        tenant_id=user["tenant_id"],
        role=user["role"],
    )


# ── 当前用户依赖 ──────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)
    ] = None,
) -> dict[str, Any]:
    """从 Authorization: Bearer <access_token> 解析当前登录用户。"""

    if credentials is None:
        raise HTTPException(status_code=401, detail="未提供认证令牌")
    payload = _decode_token(credentials.credentials, "access")
    user = await db.get_user_by_id(payload["sub"])
    if user is None:
        raise HTTPException(status_code=401, detail="账户不存在")
    return user


CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]


async def require_tenant_active(user: CurrentUser) -> dict[str, Any]:
    """校验当前用户所属租户处于 active 状态；禁用租户返回 403。"""

    tenant = await db.get_tenant(user["tenant_id"])
    if tenant is None or tenant["status"] != "active":
        raise HTTPException(status_code=403, detail="租户已被禁用")
    return user


async def require_platform_admin(user: CurrentUser) -> dict[str, Any]:
    """仅平台管理员可访问的端点依赖；非 platform_admin 返回 403。"""

    if user.get("role") != db.ROLE_PLATFORM_ADMIN:
        raise HTTPException(status_code=403, detail="需要平台管理员权限")
    return user


PlatformAdminUser = Annotated[dict[str, Any], Depends(require_platform_admin)]


# ── 登录限流 ──────────────────────────────────────────────────────


def _rate_key(email: str, ip: str) -> str:
    return f"{email.lower()}|{ip}"


def _record_login_failure(key: str) -> None:
    now = time.time()
    with _rate_lock:
        bucket = _login_failures[key]
        bucket.append(now)
        while bucket and now - bucket[0] > _LOGIN_WINDOW_SECONDS:
            bucket.popleft()


def _clear_login_failures(key: str) -> None:
    with _rate_lock:
        _login_failures.pop(key, None)


def _login_blocked(key: str) -> bool:
    now = time.time()
    with _rate_lock:
        bucket = _login_failures[key]
        while bucket and now - bucket[0] > _LOGIN_WINDOW_SECONDS:
            bucket.popleft()
        return len(bucket) >= _LOGIN_MAX_FAILURES


def _client_ip(request: Request) -> str:
    # 反向代理（nginx 等）后的真实 IP；直连时取 peer
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── 路由 ──────────────────────────────────────────────────────────


def _validate_credentials(email: str, password: str) -> str:
    """校验邮箱与密码强度，返回规范化后的邮箱。"""

    email_norm = email.strip().lower()
    if not _EMAIL_RE.match(email_norm):
        raise HTTPException(status_code=400, detail="邮箱格式不正确")
    if len(password) < _PASSWORD_MIN:
        raise HTTPException(
            status_code=400, detail=f"密码至少 {_PASSWORD_MIN} 位"
        )
    return email_norm


@router.post("/register", response_model=dict, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, request: Request) -> dict[str, Any]:
    """自助注册账户并返回令牌对。"""

    settings = get_settings()
    if not settings.allow_registration:
        raise HTTPException(status_code=403, detail="当前系统未开放自助注册")

    email = _validate_credentials(body.email, body.password)
    display_name = body.display_name.strip() or email.split("@")[0]

    try:
        user_id = await db.create_user(
            email=email,
            password_hash=hash_password(body.password),
            display_name=display_name[:64],
            tenant_id=db.DEFAULT_TENANT_ID,
            role=db.ROLE_MEMBER,
        )
    except Exception as exc:
        if db.is_unique_violation(exc):
            # users.email 唯一索引冲突 → 邮箱已注册（兼容 sqlite/postgres 驱动）
            raise HTTPException(status_code=409, detail="该邮箱已注册") from exc
        raise

    logger.info("新用户注册: id=%s email=%s ip=%s", user_id, email, _client_ip(request))
    user = await db.get_user_by_id(user_id)
    assert user is not None
    tokens = await _issue_token_pair(user_id, user["tenant_id"], user["role"])
    tokens["user"] = _public_user(user).model_dump()
    return tokens


@router.post("/login", response_model=dict)
async def login(body: LoginRequest, request: Request) -> dict[str, Any]:
    """邮箱 + 密码登录，成功返回令牌对，失败计入限流。"""

    email = body.email.strip().lower()
    ip = _client_ip(request)
    key = _rate_key(email, ip)
    if _login_blocked(key):
        logger.warning("登录被限流: email=%s ip=%s", email, ip)
        raise HTTPException(
            status_code=429,
            detail=f"连续失败次数过多，请 {_LOGIN_WINDOW_SECONDS // 60} 分钟后再试",
        )

    user = await db.get_user_by_email(email)
    # 无论用户是否存在都执行一次哈希比较，降低用户枚举的时间侧信道
    dummy = "scrypt$32768$8$1$" + "00" * 16 + "$" + "00" * 32
    valid = verify_password(body.password, user["password_hash"]) if user else (
        verify_password(body.password, dummy)
    )
    if not user or not valid:
        _record_login_failure(key)
        logger.warning("登录失败: email=%s ip=%s", email, ip)
        raise HTTPException(status_code=401, detail="邮箱或密码错误")

    _clear_login_failures(key)
    # 租户状态校验：禁用租户的用户无法登录
    tenant = await db.get_tenant(user["tenant_id"])
    if tenant is None or tenant["status"] != "active":
        logger.warning("登录被拒绝：租户已禁用 tenant=%s user=%s", user["tenant_id"], email)
        raise HTTPException(status_code=403, detail="租户已被禁用，请联系管理员")
    tokens = await _issue_token_pair(user["id"], user["tenant_id"], user["role"])
    tokens["user"] = _public_user(user).model_dump()
    logger.info("用户登录成功: id=%s email=%s ip=%s", user["id"], email, ip)
    return tokens


@router.post("/refresh", response_model=dict)
async def refresh_tokens(body: RefreshRequest) -> dict[str, Any]:
    """用 refresh token 换新令牌对（刷新轮换：旧 refresh token 立即作废）。"""

    payload = _decode_token(body.refresh_token, "refresh")
    jti = payload.get("jti", "")
    record = await db.get_refresh_token(jti)
    if record is None or record["revoked"]:
        raise HTTPException(status_code=401, detail="刷新令牌已失效")
    if record["expires_at"] < time.time():
        raise HTTPException(status_code=401, detail="刷新令牌已过期")

    presented_hash = hashlib.sha256(
        body.refresh_token.encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(presented_hash, record["token_hash"]):
        raise HTTPException(status_code=401, detail="刷新令牌不匹配")

    user = await db.get_user_by_id(record["user_id"])
    if user is None:
        raise HTTPException(status_code=401, detail="账户不存在")

    # 轮换：旧 refresh token 撤销，签发全新一对
    await db.revoke_refresh_token(jti)
    tokens = await _issue_token_pair(user["id"], user["tenant_id"], user["role"])
    tokens["user"] = _public_user(user).model_dump()
    return tokens


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    body: RefreshRequest,
    user: CurrentUser,
) -> None:
    """登出：撤销传入的 refresh token（仅能撤销本人的）。"""

    try:
        payload = jwt.decode(
            body.refresh_token, get_jwt_secret(), algorithms=[_ALGORITHM]
        )
    except jwt.InvalidTokenError:
        # 令牌本身非法也视为登出成功（客户端清除本地令牌即可）
        return
    if payload.get("type") == "refresh" and payload.get("sub") == user["id"]:
        await db.revoke_refresh_token(payload.get("jti", ""))
        logger.info("用户登出: id=%s", user["id"])


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser) -> UserResponse:
    """返回当前登录用户信息。"""

    return _public_user(user)


@router.delete("/account", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(user: CurrentUser) -> None:
    """注销当前账户并级联清除全部数据。

    清除范围：全部会话 checkpoint 消息、每用户长期记忆 namespace、
    refresh token（外键级联）、用户业务行。操作不可逆。
    """

    result = await db.purge_user_data(user["tenant_id"], user["id"])
    logger.warning(
        "账户已注销: id=%s email=%s, 清除会话=%d 条, 清除长期记忆=%d 条",
        user["id"],
        user["email"],
        result["threads"],
        result["memories"],
    )
