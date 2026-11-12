"""会话（对话线程）管理 API。

会话元数据（标题、归属、时间）存业务表 threads；
消息本体由 LangGraph checkpointer 按 thread_id 持久化，
本模块通过 ``agent.aget_state`` 重建历史消息。
所有接口强制会话归属校验：用户只能访问自己的会话，
访问他人会话一律返回 404（不泄露资源是否存在）。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.auth import CurrentUser
from src.persistence import database as db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/threads", tags=["threads"])

_DEFAULT_TITLE = "新对话"
_TITLE_MAX = 100


class CreateThreadRequest(BaseModel):
    title: str = Field(default=_DEFAULT_TITLE, max_length=_TITLE_MAX)


class RenameThreadRequest(BaseModel):
    title: str = Field(min_length=1, max_length=_TITLE_MAX)


class MessageResponse(BaseModel):
    role: str
    content: str


class ThreadResponse(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float


class ThreadDetailResponse(ThreadResponse):
    messages: list[MessageResponse]


async def _owned_thread(thread_id: str, user: dict[str, Any]) -> dict[str, Any]:
    """取出会话并校验归属；不存在或不属于当前用户均返回 404。"""

    thread = await db.get_thread(thread_id)
    if thread is None or thread["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="会话不存在")
    return thread


def _message_content(content: Any) -> str:
    """LangChain 消息 content 可能是字符串或块列表，统一抽取纯文本。"""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _extract_messages(state_values: dict[str, Any]) -> list[MessageResponse]:
    """从 checkpoint 状态中重建聊天消息（过滤工具调用等内部消息）。"""

    raw_messages = state_values.get("messages") or []
    messages: list[MessageResponse] = []
    for m in raw_messages:
        mtype = getattr(m, "type", "")
        # 仅保留人类消息与有正文的 AI 消息；工具消息、纯工具调用的中间轮次不展示
        if mtype == "human":
            messages.append(MessageResponse(role="user", content=_message_content(m.content)))
        elif mtype == "ai":
            text = _message_content(m.content).strip()
            if text:
                messages.append(MessageResponse(role="assistant", content=text))
    return messages


@router.post("", response_model=ThreadResponse, status_code=201)
async def create_thread(
    body: CreateThreadRequest, user: CurrentUser
) -> ThreadResponse:
    """创建一条空会话，返回服务端分配的 thread_id。"""

    title = body.title.strip() or _DEFAULT_TITLE
    thread = await db.create_thread(
        thread_id=db.new_id(),
        user_id=user["id"],
        tenant_id=user["tenant_id"],
        title=title[:_TITLE_MAX],
    )
    logger.info("新建会话: thread_id=%s user=%s", thread["id"], user["email"])
    return ThreadResponse(**thread)


@router.get("", response_model=list[ThreadResponse])
async def list_threads(user: CurrentUser) -> list[ThreadResponse]:
    """列出当前用户的全部会话（按最近活跃倒序）。"""

    rows = await db.list_threads(user["id"])
    return [ThreadResponse(**t) for t in rows]


@router.get("/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread(
    thread_id: str, user: CurrentUser, request: Request
) -> ThreadDetailResponse:
    """获取会话详情与完整消息历史（从 checkpoint 重建）。"""

    thread = await _owned_thread(thread_id, user)
    agent = request.app.state.agent
    snapshot = await agent.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )
    messages = _extract_messages(snapshot.values or {})
    return ThreadDetailResponse(
        id=thread["id"],
        title=thread["title"],
        created_at=thread["created_at"],
        updated_at=thread["updated_at"],
        messages=messages,
    )


@router.patch("/{thread_id}", response_model=ThreadResponse)
async def rename_thread(
    thread_id: str, body: RenameThreadRequest, user: CurrentUser
) -> ThreadResponse:
    """重命名会话。"""

    thread = await _owned_thread(thread_id, user)
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="标题不能为空")
    await db.rename_thread(thread_id, title[:_TITLE_MAX])
    thread["title"] = title[:_TITLE_MAX]
    return ThreadResponse(**thread)


@router.delete("/{thread_id}", status_code=204)
async def delete_thread(thread_id: str, user: CurrentUser) -> None:
    """删除会话：同时清除业务元数据与 checkpoint 中的全部消息。"""

    await _owned_thread(thread_id, user)
    # 先清 checkpoint（失败则元数据保留，不会出现幽灵会话）
    await db.delete_checkpoint_thread(thread_id)
    await db.delete_thread_row(thread_id)
    logger.info("删除会话: thread_id=%s user=%s", thread_id, user["email"])
