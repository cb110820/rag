"""长期记忆管理 API（账户自助查看与删除）。

所有接口强制按登录态用户隔离，记忆分两类载体：

- 画像记忆（``/memory/`` 虚拟文件，namespace=("user_memory", uid)）；
- 语义事实记忆（save_memory 写入的独立条目，namespace=("user_facts", uid)）。

端点：

- GET    /memory          列出画像记忆条目（元信息 + 用量配额）
- GET    /memory/content  读取单条画像记忆正文（?key=/AGENTS.md）
- DELETE /memory/item     删除单条画像记忆（?key=/AGENTS.md）
- GET    /memory/facts    列出全部语义事实记忆
- DELETE /memory/fact     删除单条语义事实（?key=fact-xxxx）
- DELETE /memory          清空本人全部记忆（画像 + 事实）
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.api.auth import CurrentUser
from src.persistence import database as db
from src.rag.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryItemMeta(BaseModel):
    key: str
    size_bytes: int
    created_at: str | None = None
    updated_at: str | None = None


class MemoryListResponse(BaseModel):
    items: list[MemoryItemMeta]
    total_bytes: int
    item_count: int
    max_items: int
    max_file_bytes: int


class MemoryContentResponse(MemoryItemMeta):
    content: str


class FactItem(BaseModel):
    key: str
    text: str
    category: str
    size_bytes: int
    created_at: str | None = None
    updated_at: str | None = None
    # 业务时间轴：事实生效时刻 / 被取代而失效的时刻（None=仍有效）
    valid_from: str | None = None
    invalid_at: str | None = None
    # 软状态字段：active/superseded 与取代链指针；旧数据缺省 active
    status: str = "active"
    superseded_by: str | None = None
    supersedes: str | None = None
    source: str = "manual"


class FactListResponse(BaseModel):
    items: list[FactItem]
    item_count: int
    max_items: int


class MemoryClearedResponse(BaseModel):
    deleted: int
    facts_deleted: int


def _validate_key(key: str) -> str:
    """校验记忆 key：非空、单段虚拟文件路径，杜绝畸形输入。"""

    key = (key or "").strip()
    if not key or not key.startswith("/") or ".." in key or "\x00" in key:
        raise HTTPException(status_code=400, detail="非法的记忆 key")
    return key


@router.get("", response_model=MemoryListResponse)
async def list_memory(user: CurrentUser) -> MemoryListResponse:
    """列出当前用户的全部长期记忆条目与配额用量。"""

    items = await db.list_user_memory(user["tenant_id"], user["id"])
    settings = get_settings()
    logger.info("查看记忆列表: user=%s, 条目数=%d", user["email"], len(items))
    return MemoryListResponse(
        items=[MemoryItemMeta(**item) for item in items],
        total_bytes=sum(item["size_bytes"] for item in items),
        item_count=len(items),
        max_items=settings.memory_max_items,
        max_file_bytes=settings.memory_max_file_bytes,
    )


@router.get("/content", response_model=MemoryContentResponse)
async def get_memory_content(
    user: CurrentUser, key: str = Query(min_length=1, max_length=512)
) -> MemoryContentResponse:
    """读取单条记忆的正文内容。"""

    key = _validate_key(key)
    item = await db.get_user_memory_item(user["tenant_id"], user["id"], key)
    if item is None:
        raise HTTPException(status_code=404, detail="记忆条目不存在")
    return MemoryContentResponse(**item)


@router.delete("/item", status_code=204)
async def delete_memory_item(
    user: CurrentUser, key: str = Query(min_length=1, max_length=512)
) -> None:
    """删除单条记忆（模型不可见该删除动作的历史，立即生效）。"""

    key = _validate_key(key)
    deleted = await db.delete_user_memory_item(user["tenant_id"], user["id"], key)
    if not deleted:
        raise HTTPException(status_code=404, detail="记忆条目不存在")
    logger.info("用户删除单条记忆: user=%s key=%s", user["email"], key)


@router.get("/facts", response_model=FactListResponse)
async def list_facts(user: CurrentUser) -> FactListResponse:
    """列出当前用户的全部语义事实记忆（含正文，条目均为短文本）。"""

    items = await db.list_user_facts(user["tenant_id"], user["id"])
    settings = get_settings()
    logger.info("查看事实记忆列表: user=%s, 条目数=%d", user["email"], len(items))
    return FactListResponse(
        items=[FactItem(**item) for item in items],
        item_count=len(items),
        max_items=settings.memory_fact_max_items,
    )


def _validate_fact_key(key: str) -> str:
    """校验事实记忆 key：必须形如 fact-<32位 hex>。"""

    key = (key or "").strip()
    if not re.fullmatch(r"fact-[0-9a-f]{32}", key):
        raise HTTPException(status_code=400, detail="非法的事实记忆 id")
    return key


@router.delete("/fact", status_code=204)
async def delete_fact(
    user: CurrentUser, key: str = Query(min_length=1, max_length=128)
) -> None:
    """删除单条语义事实记忆。"""

    key = _validate_fact_key(key)
    if not await db.delete_user_fact(user["tenant_id"], user["id"], key):
        raise HTTPException(status_code=404, detail="事实记忆不存在")
    logger.info("用户删除事实记忆: user=%s key=%s", user["email"], key)


@router.delete("", response_model=MemoryClearedResponse)
async def clear_memory(user: CurrentUser) -> MemoryClearedResponse:
    """清空当前用户的全部长期记忆（画像文件 + 语义事实）。"""

    facts_deleted = await db.purge_user_facts(user["tenant_id"], user["id"])
    deleted = await db.purge_user_memory(user["tenant_id"], user["id"])
    logger.info(
        "用户清空全部长期记忆: user=%s, 画像=%d, 事实=%d",
        user["email"], deleted, facts_deleted,
    )
    return MemoryClearedResponse(deleted=deleted, facts_deleted=facts_deleted)
