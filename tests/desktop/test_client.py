"""RagApiClient 端点契约与 401 无感刷新重试测试（respx mock，不依赖真实服务）。"""

import httpx
import pytest
import respx

from desktop.api.client import RagApiClient
from desktop.api.errors import ApiAuthExpiredError, ApiError

BASE = "http://testserver"


def _client() -> RagApiClient:
    return RagApiClient(BASE)


@respx.mock
def test_login_success_sets_tokens() -> None:
    respx.post(f"{BASE}/auth/login").mock(
        return_value=httpx.Response(
            200,
            json={
                "access_token": "at-1",
                "refresh_token": "rt-1",
                "token_type": "bearer",
                "expires_in": 1800,
                "user": {"id": "u1", "email": "a@b.c", "role": "member"},
            },
        )
    )
    client = _client()
    result = client.login("a@b.c", "pw")
    assert result.access_token == "at-1"
    assert result.user["role"] == "member"
    assert client.has_session


@respx.mock
def test_login_403_tenant_suspended() -> None:
    respx.post(f"{BASE}/auth/login").mock(
        return_value=httpx.Response(403, json={"detail": "租户已被禁用，请联系管理员"})
    )
    with pytest.raises(ApiError) as exc_info:
        _client().login("a@b.c", "pw")
    assert exc_info.value.status_code == 403
    assert "租户已被禁用" in exc_info.value.detail


@respx.mock
def test_401_triggers_refresh_and_retries_once() -> None:
    respx.get(f"{BASE}/threads").mock(
        side_effect=[
            httpx.Response(401, json={"detail": "token 过期"}),
            httpx.Response(
                200, json=[{"id": "t1", "title": "会话", "created_at": 1.0, "updated_at": 2.0}]
            ),
        ]
    )
    refresh_route = respx.post(f"{BASE}/auth/refresh").mock(
        return_value=httpx.Response(
            200, json={"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 1800}
        )
    )
    client = _client()
    client.set_tokens("at-1", "rt-1")

    threads = client.list_threads()
    assert [t.id for t in threads] == ["t1"]
    assert refresh_route.called
    # 轮换式刷新后本地令牌对已原子替换，重试请求携带新 access token
    retry_request = respx.get(f"{BASE}/threads").calls[-1].request
    assert retry_request.headers["Authorization"] == "Bearer at-2"


@respx.mock
def test_refresh_failure_clears_session() -> None:
    respx.get(f"{BASE}/threads").mock(
        return_value=httpx.Response(401, json={"detail": "token 过期"})
    )
    respx.post(f"{BASE}/auth/refresh").mock(
        return_value=httpx.Response(401, json={"detail": "刷新令牌已失效"})
    )
    client = _client()
    client.set_tokens("at-1", "rt-1")

    with pytest.raises(ApiAuthExpiredError):
        client.list_threads()
    assert not client.has_session


@respx.mock
def test_error_detail_parsed_from_response() -> None:
    respx.get(f"{BASE}/threads/xxx").mock(
        return_value=httpx.Response(404, json={"detail": "会话不存在"})
    )
    client = _client()
    client.set_tokens("at", "rt")
    with pytest.raises(ApiError) as exc_info:
        client.get_thread("xxx")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "会话不存在"


@respx.mock
def test_delete_knowledge_file_quotes_subdir_path() -> None:
    route = respx.delete(url__regex=r".*/knowledge/files/.*").mock(
        return_value=httpx.Response(204)
    )
    client = _client()
    client.set_tokens("at", "rt")
    client.delete_knowledge_file("subdir/a b.txt")
    raw_path = route.calls[-1].request.url.raw_path
    assert raw_path == b"/knowledge/files/subdir/a%20b.txt"


@respx.mock
def test_chat_stream_dispatches_events() -> None:
    body = (
        'event: token\ndata: {"text": "你"}\n\n'
        'event: token\ndata: {"text": "好"}\n\n'
        'event: sources\ndata: {"sources": [{"source": "s1", "section": "", "score": 0.8}]}\n\n'
        "event: done\ndata: {}\n\n"
    )
    respx.post(f"{BASE}/chat/stream").mock(
        return_value=httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )
    )
    client = _client()
    client.set_tokens("at", "rt")

    import threading

    events: list[tuple[str, dict]] = []
    client.chat_stream("hi", "t1", lambda e, d: events.append((e, d)), threading.Event())
    assert [e for e, _ in events] == ["token", "token", "sources", "done"]
    assert events[0][1] == {"text": "你"}


@respx.mock
def test_resume_chat_decisions_payload() -> None:
    route = respx.post(f"{BASE}/chat/resume").mock(
        return_value=httpx.Response(
            200, json={"status": "ok", "answer": "已删除", "interrupts": []}
        )
    )
    client = _client()
    client.set_tokens("at", "rt")
    result = client.resume_chat("t1", [{"type": "approve"}])
    assert result.status == "ok"
    import json

    sent = json.loads(route.calls[-1].request.read())
    assert sent == {"thread_id": "t1", "decisions": [{"type": "approve"}]}
