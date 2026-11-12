"""桌面端 API 客户端：封装 Rag 后端全部端点。

线程模型：UI 主线程与 QThread worker 共用本客户端；``httpx.Client``
按线程惰性创建（thread-local），避免跨线程共享连接。
认证：access_token 仅内存持有；401 时用 refresh_token 无感刷新并重试一次
（refresh 轮换式：成功后必须原子替换本地令牌对，旧 refresh 立即作废）。
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import httpx

from desktop.api.errors import ApiAuthExpiredError, ApiError, ApiNetworkError
from desktop.api.models import (
    ChatResult,
    FactItem,
    FactList,
    InterruptPayload,
    KnowledgeFile,
    LoginResult,
    MemoryProfile,
    Message,
    SkillInfo,
    ThreadDetail,
    ThreadInfo,
    UploadResult,
    WikiAttachment,
    WikiPage,
    WikiPageSummary,
    WikiRevision,
    WikiRevisionDetail,
    WikiSearchHit,
)
from desktop.api.sse import parse_sse_lines

# 流式读取不设读超时（长生成场景），仅保留连接超时
_STREAM_TIMEOUT = httpx.Timeout(30.0, read=None)


def _raise_for_status(resp: httpx.Response) -> None:
    """把非 2xx 响应转成 ApiError，detail 优先取服务端 JSON 的 detail 字段。"""

    if resp.is_success:
        return
    detail = resp.text[:200]
    try:
        raw = resp.json().get("detail")
        if isinstance(raw, str):
            detail = raw
        elif raw:
            detail = json.dumps(raw, ensure_ascii=False)[:200]
    except ValueError:
        pass
    raise ApiError(resp.status_code, detail)


def _quote_path_param(value: str) -> str:
    """路径参数逐段编码（支持 source_id 这类含子目录的多段路径）。"""

    return "/".join(quote(seg, safe="") for seg in value.split("/"))


class RagApiClient:
    """Rag 后端端点方法族；构造时传入服务 base_url。"""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._access_token = ""
        self._refresh_token = ""
        self._refresh_lock = threading.Lock()
        # httpx.Client 不可跨线程共享：每个使用线程各持一个实例
        self._local = threading.local()

    # ── 认证态 ────────────────────────────────────────────────────────

    def set_tokens(self, access_token: str, refresh_token: str) -> None:
        self._access_token = access_token
        self._refresh_token = refresh_token

    def clear_tokens(self) -> None:
        self._access_token = ""
        self._refresh_token = ""

    @property
    def has_session(self) -> bool:
        return bool(self._access_token)

    def _auth_headers(self) -> dict[str, str]:
        if not self._access_token:
            return {}
        return {"Authorization": f"Bearer {self._access_token}"}

    def _thread_client(self) -> httpx.Client:
        client = getattr(self._local, "client", None)
        if client is None:
            client = httpx.Client(base_url=self.base_url, timeout=30.0)
            self._local.client = client
        return client

    # ── 通用请求（401 → 无感刷新 → 重试一次）─────────────────────────

    def _request(
        self, method: str, path: str, *, _retried: bool = False, **kwargs: Any
    ) -> httpx.Response:
        headers = {**self._auth_headers(), **kwargs.pop("headers", {})}
        try:
            resp = self._thread_client().request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiNetworkError(f"无法连接服务器：{exc}") from exc
        if resp.status_code == 401 and not _retried and self._refresh_token:
            self._refresh()
            return self._request(method, path, _retried=True, **kwargs)
        _raise_for_status(resp)
        return resp

    def _refresh(self) -> None:
        """轮换式刷新：成功则原子替换令牌对；失败清空会话并抛 AuthExpired。"""

        with self._refresh_lock:
            try:
                resp = self._thread_client().post(
                    "/auth/refresh", json={"refresh_token": self._refresh_token}
                )
            except httpx.HTTPError as exc:
                raise ApiNetworkError(f"无法连接服务器：{exc}") from exc
            if resp.status_code != 200:
                self.clear_tokens()
                raise ApiAuthExpiredError()
            data = resp.json()
            self.set_tokens(data["access_token"], data["refresh_token"])

    @staticmethod
    def _file_tuple(path: Path) -> tuple[str, bytes, str]:
        mime, _ = mimetypes.guess_type(str(path))
        return (path.name, path.read_bytes(), mime or "application/octet-stream")

    # ── 认证 ──────────────────────────────────────────────────────────

    def login(self, email: str, password: str) -> LoginResult:
        try:
            resp = self._thread_client().post(
                "/auth/login", json={"email": email, "password": password}
            )
        except httpx.HTTPError as exc:
            raise ApiNetworkError(f"无法连接服务器：{exc}") from exc
        _raise_for_status(resp)
        data = resp.json()
        result = LoginResult(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_in=int(data.get("expires_in", 0)),
            user=data.get("user") or {},
        )
        self.set_tokens(result.access_token, result.refresh_token)
        return result

    def logout(self) -> None:
        if self._access_token:
            try:
                self._request("POST", "/auth/logout")
            except ApiError:
                pass  # 登出失败不阻塞本地清理
        self.clear_tokens()

    def me(self) -> dict[str, Any]:
        resp = self._request("GET", "/auth/me")
        return dict(resp.json())

    def restore_session(self, refresh_token: str) -> LoginResult:
        """用已存的 refresh_token 换新会话（启动免登录）。

        失败抛 ApiAuthExpiredError / ApiNetworkError，调用方据此回退到登录页。
        """

        self.set_tokens("", refresh_token)
        self._refresh()
        user = self.me()
        return LoginResult(self._access_token, self._refresh_token, 0, user)

    def refresh_now(self) -> None:
        """主动刷新令牌（定时器用），与 401 被动刷新共用轮换逻辑。"""

        self._refresh()

    # ── 会话 ──────────────────────────────────────────────────────────

    def create_thread(self, title: str = "新对话") -> ThreadInfo:
        resp = self._request("POST", "/threads", json={"title": title})
        return ThreadInfo(**resp.json())

    def list_threads(self) -> list[ThreadInfo]:
        resp = self._request("GET", "/threads")
        return [ThreadInfo(**row) for row in resp.json()]

    def get_thread(self, thread_id: str) -> ThreadDetail:
        resp = self._request("GET", f"/threads/{quote(thread_id, safe='')}")
        data = resp.json()
        messages = [Message(**m) for m in data.pop("messages", [])]
        return ThreadDetail(**data, messages=messages)

    def rename_thread(self, thread_id: str, title: str) -> ThreadInfo:
        resp = self._request(
            "PATCH", f"/threads/{quote(thread_id, safe='')}", json={"title": title}
        )
        return ThreadInfo(**resp.json())

    def delete_thread(self, thread_id: str) -> None:
        self._request("DELETE", f"/threads/{quote(thread_id, safe='')}")

    # ── 对话 ──────────────────────────────────────────────────────────

    @staticmethod
    def _chat_result(data: dict[str, Any]) -> ChatResult:
        return ChatResult(
            status=data.get("status", "ok"),
            answer=data.get("answer", ""),
            interrupts=[InterruptPayload(**it) for it in data.get("interrupts", [])],
        )

    def chat(self, message: str, thread_id: str) -> ChatResult:
        resp = self._request(
            "POST", "/chat", json={"message": message, "thread_id": thread_id}
        )
        return self._chat_result(resp.json())

    def resume_chat(self, thread_id: str, decisions: list[dict[str, Any]]) -> ChatResult:
        resp = self._request(
            "POST", "/chat/resume", json={"thread_id": thread_id, "decisions": decisions}
        )
        return self._chat_result(resp.json())

    def chat_stream(
        self,
        message: str,
        thread_id: str,
        on_event: Callable[[str, dict[str, Any]], None],
        cancel_event: threading.Event,
    ) -> None:
        """POST /chat/stream：逐 SSE 事件回调 on_event(event, data)。

        cancel_event 置位后尽快返回；服务端流结束（done/error）后自然返回。
        401 时无感刷新并重连一次。
        """

        payload = {"message": message, "thread_id": thread_id}
        for attempt in range(2):
            try:
                with self._thread_client().stream(
                    "POST",
                    "/chat/stream",
                    json=payload,
                    headers=self._auth_headers(),
                    timeout=_STREAM_TIMEOUT,
                ) as resp:
                    if resp.status_code == 401 and attempt == 0 and self._refresh_token:
                        self._refresh()
                        continue
                    _raise_for_status(resp)
                    for event, data in parse_sse_lines(resp.iter_lines()):
                        if cancel_event.is_set():
                            return
                        on_event(event, data)
                    return
            except httpx.HTTPError as exc:
                raise ApiNetworkError(f"连接中断：{exc}") from exc

    # ── 知识库 ────────────────────────────────────────────────────────

    def upload_knowledge(self, path: Path, subdir: str = "") -> UploadResult:
        resp = self._request(
            "POST",
            "/knowledge/upload",
            files={"file": self._file_tuple(path)},
            data={"subdir": subdir},
        )
        return UploadResult(**resp.json())

    def list_knowledge_files(self) -> list[KnowledgeFile]:
        resp = self._request("GET", "/knowledge/files")
        return [KnowledgeFile(**row) for row in resp.json()]

    def delete_knowledge_file(self, source_id: str) -> None:
        self._request("DELETE", f"/knowledge/files/{_quote_path_param(source_id)}")

    # ── 技能（租户）──────────────────────────────────────────────────

    def list_skills(self) -> list[SkillInfo]:
        resp = self._request("GET", "/skills")
        return [SkillInfo(**row) for row in resp.json()]

    def upload_skill(self, path: Path, overwrite: bool = False) -> SkillInfo:
        resp = self._request(
            "POST",
            "/skills/upload",
            files={"file": self._file_tuple(path)},
            data={"overwrite": str(overwrite).lower()},
        )
        return SkillInfo(**resp.json())

    def delete_skill(self, name: str) -> None:
        self._request("DELETE", f"/skills/{quote(name, safe='')}")

    # ── 记忆 ──────────────────────────────────────────────────────────

    def list_profile(self) -> MemoryProfile:
        resp = self._request("GET", "/memory")
        return MemoryProfile(**resp.json())

    def get_profile_content(self, key: str) -> str:
        resp = self._request("GET", "/memory/content", params={"key": key})
        return str(resp.json().get("content", ""))

    def delete_profile_item(self, key: str) -> None:
        self._request("DELETE", "/memory/item", params={"key": key})

    def list_facts(self) -> FactList:
        resp = self._request("GET", "/memory/facts")
        data = resp.json()
        return FactList(
            items=[FactItem(**row) for row in data.get("items", [])],
            item_count=data.get("item_count", 0),
            max_items=data.get("max_items", 0),
        )

    def delete_fact(self, key: str) -> None:
        self._request("DELETE", "/memory/fact", params={"key": key})

    def clear_memory(self) -> dict[str, int]:
        resp = self._request("DELETE", "/memory")
        return dict(resp.json())

    # ── Wiki ──────────────────────────────────────────────────────────

    def search_wiki(self, query: str) -> list[WikiSearchHit]:
        resp = self._request("GET", "/wiki/search", params={"q": query})
        return [WikiSearchHit(**row) for row in resp.json()]

    def list_wiki_pages(self) -> list[WikiPageSummary]:
        resp = self._request("GET", "/wiki/pages")
        return [WikiPageSummary(**row) for row in resp.json()]

    def get_wiki_page(self, slug: str) -> WikiPage:
        resp = self._request("GET", f"/wiki/pages/{quote(slug, safe='')}")
        return WikiPage(**resp.json())

    def create_wiki_page(
        self, slug: str, title: str, content: str, summary: str = ""
    ) -> WikiPage:
        resp = self._request(
            "POST",
            "/wiki/pages",
            json={"slug": slug, "title": title, "content": content, "summary": summary},
        )
        return WikiPage(**resp.json())

    def update_wiki_page(
        self, slug: str, title: str, content: str, summary: str = ""
    ) -> WikiPage:
        resp = self._request(
            "PUT",
            f"/wiki/pages/{quote(slug, safe='')}",
            json={"title": title, "content": content, "summary": summary},
        )
        return WikiPage(**resp.json())

    def delete_wiki_page(self, slug: str) -> None:
        self._request("DELETE", f"/wiki/pages/{quote(slug, safe='')}")

    def wiki_history(self, slug: str) -> list[WikiRevision]:
        resp = self._request("GET", f"/wiki/pages/{quote(slug, safe='')}/history")
        return [WikiRevision(**row) for row in resp.json()]

    def get_wiki_revision(self, slug: str, revision_id: str) -> WikiRevisionDetail:
        resp = self._request(
            "GET",
            f"/wiki/pages/{quote(slug, safe='')}/revisions/{quote(revision_id, safe='')}",
        )
        return WikiRevisionDetail(**resp.json())

    def revert_wiki_page(self, slug: str, revision_id: str, summary: str = "") -> WikiPage:
        resp = self._request(
            "POST",
            f"/wiki/pages/{quote(slug, safe='')}/revert",
            json={"revision_id": revision_id, "summary": summary},
        )
        return WikiPage(**resp.json())

    def import_wiki_document(
        self, path: Path, slug: str = "", title: str = "", summary: str = "导入文档"
    ) -> WikiPage:
        resp = self._request(
            "POST",
            "/wiki/import",
            files={"file": self._file_tuple(path)},
            data={"slug": slug, "title": title, "summary": summary},
        )
        return WikiPage(**resp.json())

    # ── Wiki 附件 ─────────────────────────────────────────────────────

    def list_wiki_attachments(self, slug: str) -> list[WikiAttachment]:
        resp = self._request("GET", f"/wiki/pages/{quote(slug, safe='')}/attachments")
        return [WikiAttachment(**row) for row in resp.json()]

    def upload_wiki_attachment(self, slug: str, path: Path) -> WikiAttachment:
        resp = self._request(
            "POST",
            f"/wiki/pages/{quote(slug, safe='')}/attachments",
            files={"file": self._file_tuple(path)},
        )
        return WikiAttachment(**resp.json())

    def download_wiki_attachment(self, attachment_id: str) -> tuple[str, bytes]:
        """下载附件，返回 (文件名, 内容)；文件名从 Content-Disposition 解析。"""

        resp = self._request(
            "GET", f"/wiki/attachments/{quote(attachment_id, safe='')}"
        )
        filename = "attachment"
        disposition = resp.headers.get("content-disposition", "")
        match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", disposition)
        if match:
            filename = unquote(match.group(1))
        return filename, resp.content

    def delete_wiki_attachment(self, attachment_id: str) -> None:
        self._request("DELETE", f"/wiki/attachments/{quote(attachment_id, safe='')}")

    # ── 平台管理（platform_admin）─────────────────────────────────────

    def list_tenants(self) -> list[dict[str, Any]]:
        resp = self._request("GET", "/api/admin/tenants")
        return list(resp.json())

    def create_tenant(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._request("POST", "/api/admin/tenants", json=payload)
        return dict(resp.json())

    def get_tenant(self, tenant_id: str) -> dict[str, Any]:
        resp = self._request("GET", f"/api/admin/tenants/{quote(tenant_id, safe='')}")
        return dict(resp.json())

    def update_tenant(self, tenant_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._request(
            "PATCH", f"/api/admin/tenants/{quote(tenant_id, safe='')}", json=payload
        )
        return dict(resp.json())

    def delete_tenant(self, tenant_id: str) -> None:
        self._request("DELETE", f"/api/admin/tenants/{quote(tenant_id, safe='')}")

    def tenant_stats(self, tenant_id: str) -> dict[str, Any]:
        resp = self._request(
            "GET", f"/api/admin/tenants/{quote(tenant_id, safe='')}/stats"
        )
        return dict(resp.json())

    def admin_list_skills(self) -> list[SkillInfo]:
        resp = self._request("GET", "/api/admin/skills")
        return [SkillInfo(**row) for row in resp.json()]

    def admin_validate_skill(self, path: Path) -> dict[str, Any]:
        """预检 zip 技能包（不落盘）；errors/warnings 在响应体内承载。"""

        resp = self._request(
            "POST", "/api/admin/skills/validate", files={"file": self._file_tuple(path)}
        )
        return dict(resp.json())

    def admin_upload_skill(self, path: Path, overwrite: bool = False) -> SkillInfo:
        resp = self._request(
            "POST",
            "/api/admin/skills/upload",
            files={"file": self._file_tuple(path)},
            data={"overwrite": str(overwrite).lower()},
        )
        return SkillInfo(**resp.json())

    def admin_delete_skill(self, name: str) -> None:
        self._request("DELETE", f"/api/admin/skills/{quote(name, safe='')}")
