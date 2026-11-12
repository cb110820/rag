"""非流式 /chat 引用溯源（citations）测试：去重合并、pending 无 sources、SSE 不变。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.api.auth import get_current_user
from src.persistence import database as db
from src.rag.tools import _source_emitters

pytestmark = pytest.mark.usefixtures("isolated_db")

TENANT = db.DEFAULT_TENANT_ID
_user_state: dict[str, str] = {}


def _current_user() -> dict:
    return {"id": _user_state["id"], "email": "u@example.com", "tenant_id": TENANT}


@pytest_asyncio.fixture(autouse=True)
async def _seed_user(isolated_db) -> None:
    """threads.user_id 有外键约束，先播种测试用户并记录真实 id。"""

    _user_state["id"] = await db.create_user(
        email="u@example.com", password_hash="h", display_name="测试用户"
    )


@pytest_asyncio.fixture
async def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    app.dependency_overrides[get_current_user] = _current_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


class _FakeAgent:
    """模拟 DeepAgents 代理：执行期间经 source_emitter 通道发射检索来源。"""

    def __init__(self, sources: list[list[dict]] | None = None) -> None:
        self.sources = sources or []
        self.invocations = 0

    async def ainvoke(self, _input: dict, config=None, context=None) -> dict[str, Any]:
        self.invocations += 1
        for batch in self.sources:
            for callback in list(_source_emitters.get() or []):
                callback(batch)
        return {"messages": [SimpleNamespace(content="答案正文")]}


async def test_chat_returns_deduped_sources(client) -> None:
    """多轮检索来源按 source 去重、后发覆盖先发。"""

    thread_id = (
        await db.create_thread(
            thread_id="thr-t", tenant_id=TENANT, user_id=_user_state["id"], title="t"
        )
    )["id"]
    app.state.agent = _FakeAgent(sources=[
        [{"source": "a.md", "section": "旧段落", "score": 0.5, "tier": "raw"}],
        [
            {"source": "a.md", "section": "新段落", "score": 0.9, "tier": "wiki"},
            {"source": "wiki/entity-x.md", "section": "", "score": 0.8, "tier": "wiki"},
        ],
    ])

    resp = await client.post("/chat", json={"message": "问个问题", "thread_id": thread_id})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["answer"] == "答案正文"
    by_source = {s["source"]: s for s in body["sources"]}
    assert set(by_source) == {"a.md", "wiki/entity-x.md"}
    assert by_source["a.md"]["section"] == "新段落"  # 后发覆盖先发
    assert by_source["a.md"]["tier"] == "wiki"


async def test_chat_pending_approval_no_sources(client) -> None:
    """HITL 挂起：不返回 sources。"""

    from types import SimpleNamespace as NS

    thread_id = (
        await db.create_thread(
            thread_id="thr-t2", tenant_id=TENANT, user_id=_user_state["id"], title="t2"
        )
    )["id"]
    agent = _FakeAgent(sources=[[{"source": "a.md", "score": 0.5, "tier": "raw"}]])

    async def pending_ainvoke(_input, config=None, context=None):
        agent.invocations += 1
        return {"messages": [NS(content="", tool_calls=[])],
                "__interrupt__": [{"id": "i1", "tool": "shell", "args": {}}]}

    app.state.agent = SimpleNamespace(ainvoke=pending_ainvoke)

    resp = await client.post("/chat", json={"message": "审批一下", "thread_id": thread_id})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "pending_approval"
    assert body["sources"] == []


async def test_chat_without_sources_is_empty_list(client) -> None:
    """代理未做检索（无回调发射）：sources 为空列表，不报错。"""

    thread_id = (
        await db.create_thread(
            thread_id="thr-t3", tenant_id=TENANT, user_id=_user_state["id"], title="t3"
        )
    )["id"]
    app.state.agent = _FakeAgent()

    resp = await client.post("/chat", json={"message": "你好", "thread_id": thread_id})

    assert resp.status_code == 200, resp.text
    assert resp.json()["sources"] == []
