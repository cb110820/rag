"""知识治理五原则 API 级测试。

覆盖：
1. 原始文档不可变：/knowledge/upload 同名文件返回 409；
2. Schema 约束纪律：Wiki 写路径 frontmatter 校验（400），合法页面镜像含 frontmatter；
3. Lint 保证健康：GET /knowledge/lint 报告缺失/孤儿镜像。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from src.api.app import upload_knowledge_file
from src.persistence import database as db
from src.rag import knowledge_contract as kc

pytestmark = pytest.mark.usefixtures("isolated_db")

TENANT = db.DEFAULT_TENANT_ID


class _FakeUploadFile:
    """最小 UploadFile 替身：upload 端点只读 filename 与 read()。"""

    def __init__(self, filename: str, data: bytes) -> None:
        self.filename = filename
        self._data = data
        self.content_type = "text/markdown"

    async def read(self) -> bytes:
        return self._data


@pytest.fixture
async def knowledge_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """把契约层的知识库/数据根目录重定向到临时目录。"""

    fake = SimpleNamespace(
        knowledge_dir=tmp_path / "knowledge",
        data_dir=tmp_path / "data",
    )
    monkeypatch.setattr(kc, "get_settings", lambda: fake)
    return fake.knowledge_dir


def _user(email: str = "u@example.com") -> dict:
    return {"id": "user-1", "email": email, "tenant_id": TENANT}


# ── 1. 原始文档不可变 ─────────────────────────────────────────────


async def test_upload_rejects_overwrite(knowledge_root: Path) -> None:
    first_content = "# 报告\n\n正文".encode()
    first = await upload_knowledge_file(
        _user(), _FakeUploadFile("report.md", first_content)
    )
    assert first.index_refreshed is False  # KB 未初始化，跳过主动刷新
    assert (knowledge_root / TENANT / "report.md").exists()

    with pytest.raises(HTTPException) as exc:
        await upload_knowledge_file(
            _user(), _FakeUploadFile("report.md", "# 修订版".encode())
        )
    assert exc.value.status_code == 409
    assert "不可覆盖" in exc.value.detail
    # 原文件内容未被篡改
    assert (knowledge_root / TENANT / "report.md").read_bytes() == first_content


async def test_upload_versioned_filename_allowed(knowledge_root: Path) -> None:
    await upload_knowledge_file(_user(), _FakeUploadFile("report.md", b"v1"))
    second = await upload_knowledge_file(
        _user(), _FakeUploadFile("report_v2.md", b"v2")
    )
    assert second.source_id == "report_v2.md"
    assert {p.name for p in (knowledge_root / TENANT).iterdir()} == {
        "report.md",
        "report_v2.md",
    }


# ── 2+3. Wiki Schema 纪律与 Lint（走完整 app 路由） ────────────────


@pytest.fixture
async def client(
    knowledge_root: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:
    # wiki 端点的镜像物化经 anyio 线程池执行，仍读取同一份 fake settings
    from src.api.app import app as fastapi_app

    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _token(client: AsyncClient, email: str) -> str:
    resp = await client.post(
        "/auth/register", json={"email": email, "password": "Passw0rd!"}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_wiki_frontmatter_discipline(
    client: AsyncClient, knowledge_root: Path
) -> None:
    token = await _token(client, "wiki-a@example.com")
    headers = _auth(token)

    # 未知字段 → 400
    bad = await client.post(
        "/wiki/pages",
        headers=headers,
        json={"slug": "bad", "title": "坏页面", "content": "---\nowner: 张三\n---\n\n正文"},
    )
    assert bad.status_code == 400
    assert "owner" in bad.json()["detail"]

    # YAML 损坏 → 400
    broken = await client.post(
        "/wiki/pages",
        headers=headers,
        json={"slug": "broken", "title": "坏YAML", "content": "---\n: :\n---\n\n正文"},
    )
    assert broken.status_code == 400

    # 纯正文（无 frontmatter）合法 → 201，镜像自动补全 frontmatter
    ok = await client.post(
        "/wiki/pages",
        headers=headers,
        json={"slug": "clean", "title": "干净页面", "content": "# 干净页面\n\n正文"},
    )
    assert ok.status_code == 201, ok.text
    mirror = knowledge_root / TENANT / "wiki" / "clean.md"
    assert mirror.exists()
    text = mirror.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "title: 干净页面" in text
    assert "正文" in text

    # 合法 frontmatter 页面 → 201，tags 保留进镜像
    ok2 = await client.post(
        "/wiki/pages",
        headers=headers,
        json={
            "slug": "tagged",
            "title": "带标签",
            "content": "---\ntags:\n  - 部署\nversion: v2\n---\n\n正文",
        },
    )
    assert ok2.status_code == 201, ok2.text
    tagged = (knowledge_root / TENANT / "wiki" / "tagged.md").read_text(
        encoding="utf-8"
    )
    assert "tags:" in tagged
    assert "部署" in tagged

    # lint：两个健康镜像，无 error
    lint = await client.get("/knowledge/lint", headers=headers)
    assert lint.status_code == 200
    report = lint.json()
    assert report["error_count"] == 0
    assert report["checked_files"] == 2


async def test_wiki_lint_reports_mirror_drift(
    client: AsyncClient, knowledge_root: Path
) -> None:
    token = await _token(client, "wiki-b@example.com")
    headers = _auth(token)
    created = await client.post(
        "/wiki/pages",
        headers=headers,
        json={"slug": "drift", "title": "漂移页面", "content": "正文"},
    )
    assert created.status_code == 201

    # 镜像被外力删除 → mirror_missing error
    (knowledge_root / TENANT / "wiki" / "drift.md").unlink()
    report = (await client.get("/knowledge/lint", headers=headers)).json()
    assert report["error_count"] == 1
    assert report["issues"][0]["code"] == "mirror_missing"

    # 写入孤儿镜像 → mirror_orphan error
    orphan = knowledge_root / TENANT / "wiki" / "orphan.md"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("---\ntitle: 孤儿\n---\n\n正文", encoding="utf-8")
    report = (await client.get("/knowledge/lint", headers=headers)).json()
    codes = {issue["code"] for issue in report["issues"]}
    assert codes == {"mirror_missing", "mirror_orphan"}
    assert report["error_count"] == 2


async def test_lint_requires_auth(client: AsyncClient) -> None:
    resp = await client.get("/knowledge/lint")
    assert resp.status_code in {401, 403}
