"""curation.scheduler 测试：上传调度、开关、并发去重、超时围栏、
跨进程槽位去重（DB 持久化 + lease 崩溃恢复）。"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from src.api.app import upload_knowledge_file
from src.curation import scheduler as sch
from src.curation import service as svc
from src.curation.llm import CurationExtractError, CurationExtraction
from src.persistence import database as db
from src.rag import knowledge_contract as kc

pytestmark = pytest.mark.usefixtures("isolated_db")

TENANT = db.DEFAULT_TENANT_ID


class _FakeUpload:
    def __init__(self, filename: str, data: bytes) -> None:
        self.filename = filename
        self._data = data
        self.content_type = "text/markdown"

    async def read(self) -> bytes:
        return self._data


@pytest_asyncio.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    settings = SimpleNamespace(
        knowledge_dir=tmp_path / "knowledge",
        data_dir=tmp_path / "data",
        wiki_curation_enabled=True,
        wiki_curation_timeout_seconds=10.0,
        wiki_curation_max_chars=60000,
    )
    monkeypatch.setattr(kc, "get_settings", lambda: settings)
    monkeypatch.setattr(sch, "get_settings", lambda: settings)
    monkeypatch.setattr(svc, "get_settings", lambda: settings)
    kc.tenant_root(TENANT).mkdir(parents=True, exist_ok=True)
    return settings


async def _drain(timeout: float = 10.0) -> None:
    """等待后台沉淀任务全部结束。

    任务体含 to_thread / aiosqlite 线程往返，``sleep(0)`` 空转无法给线程池
    留出墙钟时间；直接 gather 任务对象本身，超时兜底。
    任务内异常已全部消化，``gather`` 不会再抛。
    """

    tasks = list(sch._running.values())
    if tasks:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
        )
    assert not sch._running


@pytest_asyncio.fixture(autouse=True)
async def _no_leaked_background_tasks(isolated_db) -> None:
    """用例结束（且在 isolated_db 关库之前）取消并回收残留后台任务。"""

    yield
    tasks = list(sch._running.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    sch._running.clear()


# ── TR-5.1 上传后自动调度，失败不阻断 ───────────────────────────────


async def test_upload_schedules_curation(env: SimpleNamespace,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    extraction = CurationExtraction(summary="S", key_points=["k"], entities=[])

    async def fake_extract(text, source_id, existing):
        return extraction

    monkeypatch.setattr(svc, "extract_curated", fake_extract)

    user = {"id": "user-1", "email": "u@example.com", "tenant_id": TENANT}
    resp = await upload_knowledge_file(
        user, _FakeUpload("report.md", "# 报告\n正文".encode())
    )
    assert resp.source_id == "report.md"  # 上传照常 200
    assert sch.is_curation_running(TENANT, "report.md")
    await _drain()

    slugs = {p["slug"] for p in await db.list_wiki_pages(TENANT)}
    assert "doc-report" in slugs  # 后台沉淀完成
    assert not sch._running  # 槽位释放


async def test_upload_unaffected_when_extraction_fails(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_extract(text, source_id, existing):
        raise CurationExtractError("LLM 不可用")

    monkeypatch.setattr(svc, "extract_curated", fake_extract)
    user = {"id": "user-1", "email": "u@example.com", "tenant_id": TENANT}
    resp = await upload_knowledge_file(
        user, _FakeUpload("report.md", b"# x")
    )
    assert resp.source_id == "report.md"  # 无异常冒泡
    await _drain()

    slugs = {p["slug"] for p in await db.list_wiki_pages(TENANT)}
    assert "doc-report" not in slugs
    log_page = await db.get_wiki_page(TENANT, "log")
    assert log_page is not None and "沉淀失败" in log_page["content"]


# ── TR-5.2 开关关闭不调度 ────────────────────────────────────────────


async def test_disabled_no_schedule(env: SimpleNamespace) -> None:
    env.wiki_curation_enabled = False
    assert await sch.schedule_curation(TENANT, "x.md") is False
    assert not sch._running


# ── TR-5.2 并发去重：同文档第二次调度被拒，提取器只跑一次 ──────────


async def test_dedup_concurrent_same_source(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_extract(text, source_id, existing):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return CurationExtraction(summary="S", key_points=[], entities=[])

    monkeypatch.setattr(svc, "extract_curated", fake_extract)
    (kc.tenant_root(TENANT) / "a.md").write_text("正文", encoding="utf-8")

    assert await sch.schedule_curation(TENANT, "a.md") is True
    await asyncio.wait_for(entered.wait(), timeout=5)
    # 任务在跑时再次调度 → 拒绝
    assert await sch.schedule_curation(TENANT, "a.md") is False
    release.set()
    await _drain()
    assert calls == 1
    assert not sch._running


# ── TR-5.3 超时围栏释放槽位、不留半成品 ─────────────────────────────


async def test_timeout_releases_slot(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.wiki_curation_timeout_seconds = 0.05

    async def slow_extract(text, source_id, existing):
        await asyncio.sleep(5)
        return CurationExtraction(summary="S", key_points=[], entities=[])

    monkeypatch.setattr(svc, "extract_curated", slow_extract)
    (kc.tenant_root(TENANT) / "slow.md").write_text("正文", encoding="utf-8")

    await sch.schedule_curation(TENANT, "slow.md")
    # 等待超时围栏触发
    for _ in range(300):
        await asyncio.sleep(0.01)
        if not sch._running:
            break
    assert not sch.is_curation_running(TENANT, "slow.md")
    slugs = {p["slug"] for p in await db.list_wiki_pages(TENANT)}
    assert not any(s.startswith(("doc-", "entity-")) for s in slugs)


# ── 多副本改造：DB 持久化任务槽位（跨进程去重 + lease 崩溃恢复） ────


async def test_claim_slot_dedup_and_release() -> None:
    """同一 (tenant, source) 槽位只能被抢占一次，释放后可再次抢占。"""

    assert await db.claim_curation_task(TENANT, "a.md", lease_seconds=600) is True
    # 其他 worker（同进程模拟）再来抢占 → 唯一约束拒绝
    assert await db.claim_curation_task(TENANT, "a.md", lease_seconds=600) is False
    await db.release_curation_task(TENANT, "a.md")
    assert await db.claim_curation_task(TENANT, "a.md", lease_seconds=600) is True
    await db.release_curation_task(TENANT, "a.md")


async def test_claim_slot_purges_stale_lease() -> None:
    """崩溃 worker 的残留行超过 lease 期限后，可被新任务重新抢占。"""

    assert await db.claim_curation_task(TENANT, "stale.md", lease_seconds=600) is True
    # 把 started_at 拨回 lease 期限之前（模拟 worker 崩溃后时间流逝）
    await db._execute(
        "UPDATE wiki_curation_tasks SET started_at = ? "
        "WHERE tenant_id = ? AND source_id = ?",
        (time.time() - 10_000, TENANT, "stale.md"),
    )
    assert await db.claim_curation_task(TENANT, "stale.md", lease_seconds=600) is True
    await db.release_curation_task(TENANT, "stale.md")


async def test_schedule_skips_when_other_worker_holds_slot(
    env: SimpleNamespace,
) -> None:
    """槽位被其他 worker 持有（DB 行存在）时调度被拒，不创建任务。"""

    (kc.tenant_root(TENANT) / "held.md").write_text("正文", encoding="utf-8")
    # 模拟另一进程已抢占（lease 未过期）
    assert await db.claim_curation_task(TENANT, "held.md", lease_seconds=600) is True
    assert await sch.schedule_curation(TENANT, "held.md") is False
    assert not sch._running
    await db.release_curation_task(TENANT, "held.md")


async def test_schedule_releases_db_slot_on_completion(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """沉淀任务结束后释放 DB 槽位，后续调度可再次成功。"""

    async def fake_extract(text, source_id, existing):
        return CurationExtraction(summary="S", key_points=[], entities=[])

    monkeypatch.setattr(svc, "extract_curated", fake_extract)
    (kc.tenant_root(TENANT) / "done.md").write_text("正文", encoding="utf-8")

    assert await sch.schedule_curation(TENANT, "done.md") is True
    await _drain()
    # 槽位已释放：重新调度应成功
    assert await sch.schedule_curation(TENANT, "done.md") is True
    await _drain()
    row = await db._one(
        "SELECT COUNT(*) AS c FROM wiki_curation_tasks WHERE source_id = ?",
        ("done.md",),
    )
    assert row is not None and row["c"] == 0


async def test_service_busy_when_cross_process_lock_held(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """advisory 锁未获取时 curate_document 抛 CurationBusyError，不留页面。"""

    @contextlib.asynccontextmanager
    async def locked(name: str):
        yield False

    monkeypatch.setattr(db, "try_named_lock", locked)
    (kc.tenant_root(TENANT) / "busy.md").write_text("正文", encoding="utf-8")

    with pytest.raises(svc.CurationBusyError):
        await svc.curate_document(TENANT, "busy.md")
    slugs = {p["slug"] for p in await db.list_wiki_pages(TENANT)}
    assert not any(s.startswith(("doc-", "entity-")) for s in slugs)
