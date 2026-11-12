"""平台管理 API（仅 platform_admin 角色可访问）。

提供租户的全生命周期管理：创建、查询、更新、删除与用量统计。
删除租户会级联清除其下用户/会话（外键）并物理清除该租户全部记忆
（namespace 前缀 ``("tenant", <id>, ...)``）。
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.auth import PlatformAdminUser
from src.persistence import database as db
from src.rag.config import get_settings
from src.skills import manager as skills_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
_ALLOWED_STATUS = {"active", "suspended", "deleted"}
_ALLOWED_PLANS = {"free", "pro", "enterprise"}


class TenantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(min_length=2, max_length=64)
    status: str = "active"
    plan: str = "free"
    max_users: int = Field(default=5, ge=1, le=100000)
    max_storage_mb: int = Field(default=100, ge=1, le=1048576)
    max_facts: int = Field(default=200, ge=0, le=1000000)


class TenantUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    status: str | None = None
    plan: str | None = None
    max_users: int | None = Field(default=None, ge=1, le=100000)
    max_storage_mb: int | None = Field(default=None, ge=1, le=1048576)
    max_facts: int | None = Field(default=None, ge=0, le=1000000)


def _validate_slug(slug: str) -> str:
    if not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=400,
            detail="slug 仅允许小写字母/数字/连字符，2~64 字符且以字母数字开头",
        )
    return slug


def _validate_status(status: str | None) -> None:
    if status is not None and status not in _ALLOWED_STATUS:
        raise HTTPException(
            status_code=400,
            detail=f"status 必须为 {sorted(_ALLOWED_STATUS)} 之一",
        )


def _validate_plan(plan: str | None) -> None:
    if plan is not None and plan not in _ALLOWED_PLANS:
        raise HTTPException(
            status_code=400,
            detail=f"plan 必须为 {sorted(_ALLOWED_PLANS)} 之一",
        )


@router.get("/tenants", response_model=list[dict[str, Any]])
async def admin_list_tenants(
    user: PlatformAdminUser,
) -> list[dict[str, Any]]:
    """列出全部租户（平台管理员专用）。"""

    logger.info("平台管理员列出全部租户: admin=%s", user["email"])
    return await db.list_tenants()


@router.post("/tenants", response_model=dict[str, Any], status_code=201)
async def admin_create_tenant(
    body: TenantCreate,
    user: PlatformAdminUser,
) -> dict[str, Any]:
    """创建新租户。"""

    _validate_slug(body.slug)
    _validate_status(body.status)
    _validate_plan(body.plan)
    if await db.get_tenant_by_slug(body.slug) is not None:
        raise HTTPException(status_code=409, detail=f"slug '{body.slug}' 已被占用")
    tenant_id = await db.create_tenant(
        name=body.name,
        slug=body.slug,
        status=body.status,
        plan=body.plan,
        max_users=body.max_users,
        max_storage_mb=body.max_storage_mb,
        max_facts=body.max_facts,
    )
    logger.info(
        "平台管理员创建租户: admin=%s, tenant=%s, slug=%s",
        user["email"], tenant_id, body.slug,
    )
    tenant = await db.get_tenant(tenant_id)
    assert tenant is not None
    return tenant


@router.get("/tenants/{tenant_id}", response_model=dict[str, Any])
async def admin_get_tenant(
    tenant_id: str,
    user: PlatformAdminUser,
) -> dict[str, Any]:
    """查询单个租户详情。"""

    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="租户不存在")
    return tenant


@router.patch("/tenants/{tenant_id}", response_model=dict[str, Any])
async def admin_update_tenant(
    tenant_id: str,
    body: TenantUpdate,
    user: PlatformAdminUser,
) -> dict[str, Any]:
    """更新租户字段（仅传入的字段会被修改）。"""

    if await db.get_tenant(tenant_id) is None:
        raise HTTPException(status_code=404, detail="租户不存在")
    _validate_status(body.status)
    _validate_plan(body.plan)
    updated = await db.update_tenant(
        tenant_id,
        name=body.name,
        status=body.status,
        plan=body.plan,
        max_users=body.max_users,
        max_storage_mb=body.max_storage_mb,
        max_facts=body.max_facts,
    )
    if not updated:
        raise HTTPException(status_code=400, detail="未提供任何可更新字段")
    logger.info("平台管理员更新租户: admin=%s, tenant=%s", user["email"], tenant_id)
    tenant = await db.get_tenant(tenant_id)
    assert tenant is not None
    return tenant


@router.delete("/tenants/{tenant_id}", status_code=204)
async def admin_delete_tenant(
    tenant_id: str,
    user: PlatformAdminUser,
) -> None:
    """删除租户：级联清除用户/会话并物理清除该租户全部记忆。

    default 租户禁止删除。
    """

    if tenant_id == db.DEFAULT_TENANT_ID:
        raise HTTPException(status_code=400, detail="default 租户不可删除")
    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="租户不存在")
    # 先清除该租户在 Store 中的全部记忆（namespace 前缀删除）
    facts_purged = await db.purge_tenant_memory(tenant_id)
    deleted = await db.delete_tenant(tenant_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="删除租户失败")
    logger.info(
        "平台管理员删除租户: admin=%s, tenant=%s, 清除记忆条目=%d",
        user["email"], tenant_id, facts_purged,
    )


@router.get("/tenants/{tenant_id}/stats", response_model=dict[str, Any])
async def admin_tenant_stats(
    tenant_id: str,
    user: PlatformAdminUser,
) -> dict[str, Any]:
    """查询租户用量统计（用户数、会话数、事实记忆数）。"""

    if await db.get_tenant(tenant_id) is None:
        raise HTTPException(status_code=404, detail="租户不存在")
    return await db.get_tenant_stats(tenant_id)


# ── 全局技能管理（platform_admin 专用） ─────────────────────────────
#
# 全局技能安装到 skills_dir 根目录，对所有租户可见；代理每轮对话都会
# 重扫 /skills/ 源（/chat 与 /chat/stream 注入 skills_metadata=None），
# 因此上传/删除后无需重启即可在下一条消息生效。


class SkillAdminResponse(BaseModel):
    name: str
    description: str
    path: str
    files: list[str]
    updated_at: float
    valid: bool
    error: str = ""


class SkillValidateResponse(BaseModel):
    """上传前预检结果：校验不通过也返回 200，由 errors 承载原因。"""

    valid: bool
    name: str = ""
    description: str = ""
    files: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []


def _skill_payload(info: skills_manager.SkillInfo) -> dict[str, Any]:
    return {
        "name": info.name,
        "description": info.description,
        "path": info.path,
        "files": info.files,
        "updated_at": info.updated_at,
        "valid": info.valid,
        "error": info.error,
    }


def _global_skills_root():
    return get_settings().skills_dir.resolve()


@router.get("/skills", response_model=list[SkillAdminResponse])
async def admin_list_skills(user: PlatformAdminUser) -> list[dict[str, Any]]:
    """列出全部全局共享技能（平台管理员专用）。"""

    infos = await anyio.to_thread.run_sync(
        skills_manager.list_skills, _global_skills_root()
    )
    return [_skill_payload(info) for info in infos]


@router.post("/skills/validate", response_model=SkillValidateResponse)
async def admin_validate_skill(
    user: PlatformAdminUser,
    file: Annotated[UploadFile, File()],
) -> dict[str, Any]:
    """上传前预检 zip 技能包：执行与安装完全一致的全部校验，但不写入磁盘。

    校验未通过同样返回 200，由 errors 字段承载原因，便于前端统一展示；
    同名技能已存在属于可恢复场景，放入 warnings 提示勾选覆盖上传。
    """

    raw_name = file.filename or ""
    logger.info("平台管理员预检技能包: admin=%s, 文件名=%r", user["email"], raw_name)
    if not raw_name.lower().endswith(".zip"):
        return {"valid": False, "errors": ["仅接受 .zip 技能包"]}

    content = await file.read()
    try:
        inspection = skills_manager.inspect_skill_zip(content)
    except skills_manager.SkillInstallError as exc:
        logger.info("技能包预检未通过: admin=%s, %s", user["email"], exc)
        return {"valid": False, "errors": [str(exc)]}

    warnings: list[str] = []
    if (_global_skills_root() / inspection.name).is_dir():
        warnings.append(
            f"技能 '{inspection.name}' 已存在，上传时需勾选「覆盖同名技能」"
        )
    logger.info(
        "技能包预检通过: admin=%s, 技能=%s, %d 个文件",
        user["email"], inspection.name, len(inspection.files),
    )
    return {
        "valid": True,
        "name": inspection.name,
        "description": inspection.description,
        "files": inspection.files,
        "warnings": warnings,
    }


@router.post("/skills/upload", response_model=SkillAdminResponse, status_code=201)
async def admin_upload_skill(
    user: PlatformAdminUser,
    file: Annotated[UploadFile, File()],
    overwrite: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    """上传 zip 技能包并安装到全局技能目录，对所有租户生效。

    zip 布局要求与 /skills/upload 一致：根目录 SKILL.md 或唯一顶层目录
    ``<name>/SKILL.md``；安装经暂存目录原子替换，失败不留半成品。
    技能在安装后的下一条对话消息自动加载到模型，无需重启服务。
    """

    raw_name = file.filename or ""
    logger.info(
        "平台管理员上传全局技能: admin=%s, 文件名=%r, overwrite=%s",
        user["email"], raw_name, overwrite,
    )
    if not raw_name.lower().endswith(".zip"):
        raise HTTPException(status_code=415, detail="仅接受 .zip 技能包")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(content) > skills_manager.MAX_SKILL_ZIP_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"压缩包超过 "
                f"{skills_manager.MAX_SKILL_ZIP_BYTES // (1024 * 1024)}MB 上限"
            ),
        )

    root = _global_skills_root()
    root.mkdir(parents=True, exist_ok=True)
    try:
        info = await anyio.to_thread.run_sync(
            lambda: skills_manager.install_skill_from_zip(root, content, overwrite=overwrite)
        )
    except skills_manager.SkillConflictError as exc:
        logger.warning("全局技能冲突: admin=%s, %s", user["email"], exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except skills_manager.SkillInstallError as exc:
        logger.warning("全局技能安装被拒绝: admin=%s, %s", user["email"], exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "全局技能已就绪: admin=%s, 技能=%s, %d 个文件（下一轮对话自动加载到模型）",
        user["email"], info.name, len(info.files),
    )
    return _skill_payload(info)


@router.delete("/skills/{name}", status_code=204)
async def admin_delete_skill(name: str, user: PlatformAdminUser) -> None:
    """按名称删除全局共享技能。"""

    logger.info("平台管理员删除全局技能: admin=%s, 技能=%s", user["email"], name)
    try:
        await anyio.to_thread.run_sync(
            skills_manager.delete_skill, _global_skills_root(), name
        )
    except skills_manager.SkillInstallError as exc:
        # “不存在”归为 404，其余非法输入归为 400
        status = 404 if "不存在" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
