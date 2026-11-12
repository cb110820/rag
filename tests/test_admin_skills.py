"""平台管理员全局技能管理 API 测试。

覆盖：
1. 鉴权：非 platform_admin 访问 /api/admin/skills* 一律 403；
2. 上传：合法 zip 安装到全局技能根目录，重复上传 409、overwrite 覆盖成功；
3. 列表与删除：安装后可见、删除后消失、重复删除 404；
4. 非法输入：非 zip 后缀 415、损坏 zip 400。
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api import admin as admin_module
from src.api import auth
from src.api.admin import router as admin_router
from src.persistence import database as db
from src.rag.config import Settings

pytestmark = pytest.mark.usefixtures("isolated_db")


@pytest.fixture
def skills_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Settings
) -> Path:
    """把 admin 模块的 skills_dir 指向临时目录，避免污染项目 skills/。"""

    root = tmp_path / "skills"
    isolated_db.skills_dir = root
    monkeypatch.setattr(admin_module, "get_settings", lambda: isolated_db)
    return root


@pytest.fixture
async def client(skills_root: Path) -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(admin_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _member_token(client: AsyncClient, email: str = "member@example.com") -> str:
    resp = await client.post(
        "/auth/register", json={"email": email, "password": "Passw0rd!"}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["access_token"]


async def _admin_token(email: str = "admin@example.com") -> str:
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


def _skill_zip(name: str = "pdf-tools", description: str = "PDF 处理技能") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            f"{name}/SKILL.md",
            f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\n步骤说明。\n",
        )
        zf.writestr(f"{name}/helper.py", "print('hi')\n")
    return buf.getvalue()


def _upload_payload(zip_bytes: bytes, overwrite: bool = False) -> dict:
    return {
        "files": {"file": ("skill.zip", zip_bytes, "application/zip")},
        "data": {"overwrite": "true" if overwrite else "false"},
    }


# ── 1. 鉴权 ────────────────────────────────────────────────────────


async def test_non_admin_forbidden(client: AsyncClient) -> None:
    """普通 member 访问全局技能管理端点一律 403。"""

    token = await _member_token(client)
    assert (await client.get("/api/admin/skills", headers=_auth(token))).status_code == 403
    assert (
        await client.post(
            "/api/admin/skills/upload",
            headers=_auth(token),
            **_upload_payload(_skill_zip()),
        )
    ).status_code == 403
    assert (
        await client.delete("/api/admin/skills/pdf-tools", headers=_auth(token))
    ).status_code == 403


# ── 2. 上传 / 列表 / 删除全流程 ─────────────────────────────────────


async def test_admin_upload_list_delete_skill(
    client: AsyncClient, skills_root: Path
) -> None:
    token = await _admin_token()

    # 上传安装到全局根目录
    resp = await client.post(
        "/api/admin/skills/upload",
        headers=_auth(token),
        **_upload_payload(_skill_zip()),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "pdf-tools"
    assert body["valid"] is True
    assert (skills_root / "pdf-tools" / "SKILL.md").is_file()

    # 列表可见
    listed = await client.get("/api/admin/skills", headers=_auth(token))
    assert listed.status_code == 200
    names = [s["name"] for s in listed.json()]
    assert names == ["pdf-tools"]

    # 重复上传未覆盖 → 409
    conflict = await client.post(
        "/api/admin/skills/upload",
        headers=_auth(token),
        **_upload_payload(_skill_zip()),
    )
    assert conflict.status_code == 409

    # overwrite=true → 覆盖成功
    overwritten = await client.post(
        "/api/admin/skills/upload",
        headers=_auth(token),
        **_upload_payload(_skill_zip(description="新版描述"), overwrite=True),
    )
    assert overwritten.status_code == 201
    assert overwritten.json()["description"] == "新版描述"

    # 删除 → 204，列表清空，文件消失
    deleted = await client.delete("/api/admin/skills/pdf-tools", headers=_auth(token))
    assert deleted.status_code == 204
    assert not (skills_root / "pdf-tools").exists()
    listed_after = await client.get("/api/admin/skills", headers=_auth(token))
    assert listed_after.json() == []

    # 重复删除 → 404
    missing = await client.delete("/api/admin/skills/pdf-tools", headers=_auth(token))
    assert missing.status_code == 404


# ── 3. 上传前预检端点 ───────────────────────────────────────────────


async def test_validate_endpoint_reports_metadata(client: AsyncClient) -> None:
    """合法技能包：预检返回 valid=True 与解析出的元数据，且不写入磁盘。"""

    token = await _admin_token()
    resp = await client.post(
        "/api/admin/skills/validate",
        headers=_auth(token),
        files={"file": ("skill.zip", _skill_zip(), "application/zip")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["name"] == "pdf-tools"
    assert body["description"] == "PDF 处理技能"
    assert set(body["files"]) == {"SKILL.md", "helper.py"}
    assert body["warnings"] == []


async def test_validate_endpoint_does_not_install(
    client: AsyncClient, skills_root: Path
) -> None:
    """预检不得在技能根目录留下任何文件。"""

    token = await _admin_token()
    await client.post(
        "/api/admin/skills/validate",
        headers=_auth(token),
        files={"file": ("skill.zip", _skill_zip(), "application/zip")},
    )
    assert not (skills_root / "pdf-tools").exists()


async def test_validate_endpoint_returns_errors_for_bad_zip(client: AsyncClient) -> None:
    """损坏 zip：仍返回 200，valid=False 且 errors 承载原因。"""

    token = await _admin_token()
    resp = await client.post(
        "/api/admin/skills/validate",
        headers=_auth(token),
        files={"file": ("skill.zip", b"definitely not a zip", "application/zip")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert body["errors"] and "zip" in body["errors"][0]


async def test_validate_endpoint_rejects_non_zip_suffix(client: AsyncClient) -> None:
    token = await _admin_token()
    resp = await client.post(
        "/api/admin/skills/validate",
        headers=_auth(token),
        files={"file": ("skill.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert ".zip" in body["errors"][0]


async def test_validate_endpoint_warns_on_conflict(client: AsyncClient) -> None:
    """同名技能已安装时：预检仍通过，但 warnings 提示需覆盖上传。"""

    token = await _admin_token()
    # 先正式安装一个技能
    installed = await client.post(
        "/api/admin/skills/upload",
        headers=_auth(token),
        **_upload_payload(_skill_zip()),
    )
    assert installed.status_code == 201

    resp = await client.post(
        "/api/admin/skills/validate",
        headers=_auth(token),
        files={"file": ("skill.zip", _skill_zip(), "application/zip")},
    )
    body = resp.json()
    assert body["valid"] is True
    assert any("pdf-tools" in w and "覆盖" in w for w in body["warnings"])


async def test_validate_endpoint_forbidden_for_member(client: AsyncClient) -> None:
    token = await _member_token(client)
    resp = await client.post(
        "/api/admin/skills/validate",
        headers=_auth(token),
        files={"file": ("skill.zip", _skill_zip(), "application/zip")},
    )
    assert resp.status_code == 403


# ── 4. 非法输入 ────────────────────────────────────────────────────


async def test_upload_rejects_non_zip_suffix(client: AsyncClient) -> None:
    token = await _admin_token()
    resp = await client.post(
        "/api/admin/skills/upload",
        headers=_auth(token),
        files={"file": ("skill.txt", b"not a zip", "text/plain")},
        data={"overwrite": "false"},
    )
    assert resp.status_code == 415


async def test_upload_rejects_corrupt_zip(client: AsyncClient) -> None:
    token = await _admin_token()
    resp = await client.post(
        "/api/admin/skills/upload",
        headers=_auth(token),
        files={"file": ("skill.zip", b"definitely not a zip", "application/zip")},
        data={"overwrite": "false"},
    )
    assert resp.status_code == 400
