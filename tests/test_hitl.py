"""HITL 审批检查点 API 测试（/chat 挂起 + /chat/resume 恢复）。

用 FakeAgent 模拟 LangGraph 中断语义，覆盖：
1. 正常对话：status=ok 且带 answer；
2. 挂起：status=pending_approval，interrupts 载荷含工具名与参数；
3. 恢复：approve 后返回最终答案；重复恢复 409；
4. 越权：非会话属主调 /chat/resume 返回 404；
5. 拒绝：reject 决策透传给代理并返回其反馈。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.types import Command

from src.api import auth
from src.api.app import ChatResponse
from src.api.app import chat as chat_handler
from src.api.app import chat_resume as resume_handler
from src.persistence import database as db

pytestmark = pytest.mark.usefixtures("isolated_db")


class _FakeInterrupt:
    """模拟 langgraph Interrupt 对象（HITL 中间件载荷格式）。"""

    def __init__(self, interrupt_id: str, tool: str, args: dict) -> None:
        self.id = interrupt_id
        self.value = {"action_request": {"action": tool, "args": args}}


class FakeAgent:
    """可中断的伪代理：首轮可配置挂起，resume 按决策给出反馈。"""

    def __init__(self) -> None:
        self.interrupt_next = False
        self.pending = False
        self.last_decisions: list[dict] | None = None

    async def ainvoke(self, payload, config=None, context=None):
        if isinstance(payload, Command):
            if not self.pending:
                raise RuntimeError("GraphInterrupt: no pending interrupt")
            self.pending = False
            self.last_decisions = payload.resume["decisions"]
            dtype = self.last_decisions[0]["type"]
            if dtype == "approve":
                text = "已按批准执行该操作"
            else:
                text = "好的，已取消该操作"
            return {"messages": [SimpleNamespace(content=text)]}
        if self.interrupt_next:
            self.interrupt_next = False
            self.pending = True
            return {
                "__interrupt__": [
                    _FakeInterrupt("itr-1", "forget_memory", {"fact_id": "f-123"})
                ],
                "messages": [],
            }
        return {"messages": [SimpleNamespace(content="普通回答")]}


@pytest.fixture
def fake_agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture
async def client(fake_agent: FakeAgent) -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    app.include_router(auth.router)
    app.add_api_route("/chat", chat_handler, methods=["POST"], response_model=ChatResponse)
    app.add_api_route(
        "/chat/resume", resume_handler, methods=["POST"], response_model=ChatResponse
    )
    app.state.agent = fake_agent
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _user_with_thread(email: str = "hitl@example.com") -> tuple[str, str]:
    user_id = await db.create_user(
        email=email,
        password_hash=auth.hash_password("Passw0rd!"),
        display_name="HITL User",
    )
    thread = await db.create_thread(
        thread_id=db.new_id(), user_id=user_id, tenant_id=db.DEFAULT_TENANT_ID,
        title="新对话",
    )
    tokens = await auth._issue_token_pair(user_id, db.DEFAULT_TENANT_ID, "member")
    return tokens["access_token"], thread["id"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_chat_normal_ok(client: AsyncClient) -> None:
    token, tid = await _user_with_thread()
    resp = await client.post(
        "/chat", json={"message": "你好", "thread_id": tid}, headers=_auth(token)
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["answer"] == "普通回答"
    assert data["interrupts"] == []


async def test_chat_pending_approval_payload(client: AsyncClient, fake_agent: FakeAgent) -> None:
    token, tid = await _user_with_thread()
    fake_agent.interrupt_next = True
    resp = await client.post(
        "/chat", json={"message": "忘掉我的偏好", "thread_id": tid}, headers=_auth(token)
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "pending_approval"
    assert data["answer"] == ""
    assert len(data["interrupts"]) == 1
    it = data["interrupts"][0]
    assert it["id"] == "itr-1"
    assert it["tool"] == "forget_memory"
    assert it["args"] == {"fact_id": "f-123"}


async def test_resume_approve_then_duplicate_409(
    client: AsyncClient, fake_agent: FakeAgent
) -> None:
    token, tid = await _user_with_thread()
    fake_agent.interrupt_next = True
    await client.post(
        "/chat", json={"message": "忘掉", "thread_id": tid}, headers=_auth(token)
    )

    resp = await client.post(
        "/chat/resume",
        json={"thread_id": tid, "decisions": [{"type": "approve"}]},
        headers=_auth(token),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["answer"] == "已按批准执行该操作"
    assert fake_agent.last_decisions == [{"type": "approve"}]

    # 重复审批：interrupt 已被消费 → 409
    resp = await client.post(
        "/chat/resume",
        json={"thread_id": tid, "decisions": [{"type": "approve"}]},
        headers=_auth(token),
    )
    assert resp.status_code == 409, resp.text
    assert "已处理" in resp.json()["detail"]


async def test_resume_reject(client: AsyncClient, fake_agent: FakeAgent) -> None:
    token, tid = await _user_with_thread()
    fake_agent.interrupt_next = True
    await client.post(
        "/chat", json={"message": "忘掉", "thread_id": tid}, headers=_auth(token)
    )
    resp = await client.post(
        "/chat/resume",
        json={"thread_id": tid, "decisions": [{"type": "reject", "message": "不要删"}]},
        headers=_auth(token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["answer"] == "好的，已取消该操作"
    assert fake_agent.last_decisions == [{"type": "reject", "message": "不要删"}]


async def test_resume_non_owner_404(client: AsyncClient, fake_agent: FakeAgent) -> None:
    token_a, tid_a = await _user_with_thread("owner@example.com")
    token_b, _ = await _user_with_thread("intruder@example.com")
    fake_agent.interrupt_next = True
    await client.post(
        "/chat", json={"message": "忘掉", "thread_id": tid_a}, headers=_auth(token_a)
    )
    # B 用户尝试恢复 A 的会话
    resp = await client.post(
        "/chat/resume",
        json={"thread_id": tid_a, "decisions": [{"type": "approve"}]},
        headers=_auth(token_b),
    )
    assert resp.status_code == 404, resp.text


async def test_resume_unauthenticated(client: AsyncClient) -> None:
    resp = await client.post(
        "/chat/resume", json={"thread_id": "t", "decisions": [{"type": "approve"}]}
    )
    assert resp.status_code in (401, 403)
