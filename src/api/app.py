"""FastAPI 对外服务入口。

端点：
- /health 健康检查（免鉴权）；
- /auth/* 注册、登录、令牌刷新、登出、当前用户；
- /threads/* 会话管理（登录可见，仅能访问本人会话）；
- /chat 对话接口（登录 + 会话归属校验，多轮记忆与长期记忆按账户隔离）；
- /knowledge/* 知识库上传/列表、/skills/* 技能管理（登录可用）。
应用通过 lifespan 初始化持久化层（SQLite + checkpointer + Store），
并在此刻构建一次全局代理图单例。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Annotated

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.api import auth
from src.persistence import database as db
from src.rag.config import configure_logging

# 必须在导入会触发模型/代理初始化的模块之前完成日志配置
configure_logging()

from src.agents.graph import build_agent  # noqa: E402
from src.api.admin import router as admin_router  # noqa: E402
from src.api.curation import router as curation_router  # noqa: E402
from src.api.memory import router as memory_router  # noqa: E402
from src.api.threads import router as threads_router  # noqa: E402
from src.api.wiki import router as wiki_router  # noqa: E402
from src.memory import reflection  # noqa: E402
from src.rag import knowledge_contract as kc  # noqa: E402
from src.rag.config import APP_DIR, get_settings  # noqa: E402
from src.rag.lint import lint_tenant_knowledge  # noqa: E402
from src.rag.parser import SUPPORTED_SUFFIXES  # noqa: E402
from src.rag.tools import source_emitter  # noqa: E402
from src.skills import manager as skills_manager  # noqa: E402
from src.skills.manager import MAX_SKILL_ZIP_BYTES, SkillInfo  # noqa: E402

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：初始化持久化层与代理图，停机时关闭连接。"""

    await db.init_db()
    # 首次启动若无任何账户，给出明显提示，方便部署后创建管理员/首个账户
    user_count = await db.count_users()
    if user_count == 0:
        settings = get_settings()
        reg_hint = "开放" if settings.allow_registration else "未开放"
        logger.warning("当前系统中没有任何账户（自助注册：%s），请先注册首个账户", reg_hint)

    app.state.agent = build_agent(
        checkpointer=db.get_saver(),
        store=db.get_store(),
    )
    logger.info("应用启动完成，代理图已就绪")
    try:
        yield
    finally:
        await db.close_db()


app = FastAPI(title="DeepAgents RAG API", version="0.2.0", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(threads_router)
app.include_router(memory_router)
app.include_router(admin_router)
app.include_router(wiki_router)
app.include_router(curation_router)
# 前端静态页面目录：开发态为项目根/web，打包态为 exe 同级/web（首启时
# 由 launcher 从 bundle 种子目录拷贝，便于直接替换/定制 index.html）
_WEB_DIR = APP_DIR / "web"

# 单次上传大小上限，防止大文件占满内存
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# 首轮对话自动取标题的长度
_AUTO_TITLE_LEN = 18
_DEFAULT_THREAD_TITLE = "新对话"


class ChatRequest(BaseModel):
    """聊天请求体。thread_id 必须是当前用户已创建的会话。"""

    message: str = Field(min_length=1, max_length=20_000)
    thread_id: str = Field(min_length=1, max_length=128)


class InterruptPayload(BaseModel):
    """一个待审批的工具调用（HITL 检查点挂起时返回给前端）。"""

    id: str
    tool: str
    args: dict


class ChatResponse(BaseModel):
    """聊天响应体。status=pending_approval 时 answer 为空，前端渲染审批卡片。

    ``sources``：本轮回答引用的知识库片段（来源/层级/相关度），由检索
    工具经 source_emitter 收集；pending_approval 时不收集。默认空列表，
    旧客户端零感知。
    """

    status: str = "ok"  # ok | pending_approval
    answer: str = ""
    interrupts: list[InterruptPayload] = []
    sources: list[dict] = []


class ResumeRequest(BaseModel):
    """HITL 审批恢复请求：decisions 与挂起时返回的 interrupts 按下标一一对应。"""

    thread_id: str = Field(min_length=1, max_length=128)
    # 每项：{"type": "approve"} / {"type": "reject", "message": "..."} / {"type": "edit", ...}
    decisions: list[dict] = Field(min_length=1)


def _interrupt_to_payload(it: object) -> InterruptPayload:
    """把 langgraph Interrupt 对象转成前端可渲染的审批载荷（防御式解析）。"""

    value = getattr(it, "value", None)
    tool, args = "unknown", {}
    if isinstance(value, dict):
        single = value.get("action_request") or {}
        if single:
            tool = single.get("action") or single.get("name") or tool
            args = single.get("args") or {}
        else:
            multiple = value.get("action_requests") or []
            if multiple and isinstance(multiple[0], dict):
                tool = multiple[0].get("name") or multiple[0].get("action") or tool
                args = multiple[0].get("args") or {}
    return InterruptPayload(id=str(getattr(it, "id", "")), tool=tool, args=args)


def _pending_interrupts_from_result(result: dict) -> list[InterruptPayload]:
    return [_interrupt_to_payload(it) for it in result.get("__interrupt__") or []]


async def _pending_interrupts_from_state(
    agent: object, thread_id: str
) -> tuple[list[InterruptPayload], list]:
    """流式结束后从检查点状态读取挂起的审批与消息快照（一次 aget_state）。"""

    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    payloads: list[InterruptPayload] = []
    for task in getattr(state, "tasks", []) or []:
        for it in getattr(task, "interrupts", []) or []:
            payloads.append(_interrupt_to_payload(it))
    messages = (getattr(state, "values", None) or {}).get("messages", [])
    return payloads, messages


def _result_answer(result: dict) -> str:
    """从代理结果提取最终文本答案（兼容 content 为块列表的情况）。"""

    content = result["messages"][-1].content
    if isinstance(content, str):
        return content
    return "\n".join(str(block.get("text", block)) for block in content)


def _sse(event: str, data: str) -> str:
    """格式化一条 SSE 事件（data 为已序列化的 JSON 字符串）。"""

    return f"event: {event}\ndata: {data}\n\n"


class UploadResponse(BaseModel):
    """知识库上传响应体。"""

    filename: str
    source_id: str
    subdir: str = ""
    size_bytes: int
    # 知识库单例尚未在本进程初始化时不主动刷新（避免上传时下载嵌入模型权重），
    # 文件已落盘，首次检索构建索引时自然会纳入新文件
    index_refreshed: bool
    document_count: int | None = None
    chunk_count: int | None = None


class KnowledgeFile(BaseModel):
    """知识库中的单个文件信息。"""

    source_id: str
    name: str
    size_bytes: int
    modified_at: float


class SkillResponse(BaseModel):
    """技能元数据，对应 skills/<name>/ 目录。"""

    name: str
    description: str
    path: str
    files: list[str]
    file_count: int
    updated_at: float
    valid: bool
    error: str = ""

    @classmethod
    def from_info(cls, info: SkillInfo) -> SkillResponse:
        return cls(
            name=info.name,
            description=info.description,
            path=info.path,
            files=info.files,
            file_count=len(info.files),
            updated_at=info.updated_at,
            valid=info.valid,
            error=info.error,
        )


@app.get("/health")
async def health() -> dict[str, str]:
    """健康检查端点（免鉴权）：含持久化层探活，DB 不可用时返回 503。"""

    try:
        return await db.health_check()
    except Exception as exc:
        logger.warning("健康检查失败，持久化层不可用: %s", exc)
        raise HTTPException(status_code=503, detail="storage unavailable") from exc


@app.post("/knowledge/upload", response_model=UploadResponse)
async def upload_knowledge_file(
    user: auth.CurrentUser,
    file: Annotated[UploadFile, File()],
    subdir: Annotated[str, Form()] = "",
) -> UploadResponse:
    """上传文档到知识库目录并触发索引刷新。

    - 仅接受 parser 支持的格式（txt/md/html/pdf/docx）
    - 文件落盘到 KNOWLEDGE_DIR（可通过 subdir 表单字段指定安全子目录）
    - 原始文档不可变：同名文件已存在时返回 409，不做覆盖
    - 知识库已在运行时初始化才同步刷新索引，否则等待下次检索自然重建
    """

    raw_name = file.filename or ""
    logger.info(
        "收到上传请求: 用户=%s, 原始文件名=%r, subdir=%r",
        user["email"], raw_name, subdir,
    )
    filename = kc.sanitize_filename(raw_name)
    if not filename:
        logger.warning("上传拒绝：文件名无效（%r）", raw_name)
        raise HTTPException(status_code=400, detail="文件名无效")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        logger.warning("上传拒绝：不支持的文件类型 %s（%s）", suffix, filename)
        raise HTTPException(
            status_code=415,
            detail=(
                f"不支持的文件类型 '{suffix}'，"
                f"支持的格式: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            ),
        )

    content = await file.read()
    if not content:
        logger.warning("上传拒绝：文件内容为空（%s）", filename)
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > MAX_UPLOAD_BYTES:
        logger.warning("上传拒绝：文件 %s 大小 %d bytes 超过上限", filename, len(content))
        raise HTTPException(status_code=413, detail="文件超过 50MB 上限")

    root = kc.tenant_root(user["tenant_id"])
    try:
        target_dir = kc.resolve_upload_dir(user["tenant_id"], subdir)
    except kc.KnowledgePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename

    # 知识治理原则 1——原始文档不可变：同名文件一律拒绝覆盖。
    # 修订请以新版本文件名上传（如 report_v2.md），或沉淀到 Wiki 页面
    if target.exists():
        logger.warning(
            "上传拒绝：原始文档不可覆盖 tenant=%s, filename=%s, 操作人=%s",
            user["tenant_id"], filename, user["email"],
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"原始文档不可覆盖：'{filename}' 已存在。"
                "如需更新，请以带版本号的新文件名上传（如 report_v2.md），"
                "或将修订沉淀到 Wiki 页面"
            ),
        )

    # 配额校验：租户知识库总存储不得超过 max_storage_mb
    tenant = await db.get_tenant(user["tenant_id"])
    max_bytes = (tenant or {}).get("max_storage_mb")
    if max_bytes:
        existing_bytes = sum(
            p.stat().st_size for p in root.rglob("*") if p.is_file()
        ) if root.exists() else 0
        projected = existing_bytes + len(content)
        limit = int(max_bytes) * 1024 * 1024
        if projected > limit:
            logger.warning(
                "上传拒绝：租户存储超额 tenant=%s, projected=%d, limit=%d",
                user["tenant_id"], projected, limit,
            )
            raise HTTPException(
                status_code=413,
                detail=f"租户知识库存储已达上限 {max_bytes}MB，请先删除旧文档",
            )

    await anyio.to_thread.run_sync(target.write_bytes, content)
    logger.info(
        "文件已落盘: %s（%d bytes，上传者=%s，tenant=%s）",
        target,
        len(content),
        user["email"],
        user["tenant_id"],
    )

    saved_path = target.resolve()
    source_id = saved_path.relative_to(root).as_posix()
    rel_parent = saved_path.relative_to(root).parent.as_posix()
    saved_subdir = "" if rel_parent == "." else rel_parent

    # 只在该租户知识库已初始化（说明模型权重已就绪）时刷新，避免上传请求
    # 承担 BGE 模型首次加载/下载的高昂成本
    index_refreshed = await anyio.to_thread.run_sync(
        kc.is_initialized, user["tenant_id"]
    )
    document_count = None
    chunk_count = None
    if index_refreshed:
        kb = await anyio.to_thread.run_sync(
            kc.refresh_knowledge_base, user["tenant_id"]
        )
        document_count = len(kb.documents)
        chunk_count = len(kb.chunks)
        logger.info(
            "索引已同步刷新: tenant=%s, %d 个文档 / %d 个块",
            user["tenant_id"], document_count, chunk_count,
        )
    else:
        logger.info("知识库尚未初始化，跳过主动刷新（将在首次检索时纳入新文件）")

    # LLM Wiki 自动沉淀：文件落盘后 fire-and-forget 调度，不阻塞响应，
    # 沉淀失败只记日志，不改变上传结果（知识治理：Wiki 负责沉淀）
    try:
        from src.curation.scheduler import schedule_curation

        await schedule_curation(user["tenant_id"], source_id)
    except Exception:  # noqa: BLE001 - 调度本身绝不能影响上传
        logger.exception("调度 Wiki 沉淀失败（不影响上传）: source=%s", source_id)

    return UploadResponse(
        filename=filename,
        source_id=source_id,
        subdir=saved_subdir,
        size_bytes=len(content),
        index_refreshed=index_refreshed,
        document_count=document_count,
        chunk_count=chunk_count,
    )


@app.get("/knowledge/files", response_model=list[KnowledgeFile])
async def list_knowledge_files(user: auth.CurrentUser) -> list[KnowledgeFile]:
    """列出当前租户知识库目录中的所有受支持文档，供前端知识库面板展示。"""

    root = kc.tenant_root(user["tenant_id"])
    logger.debug("查询知识库文件列表: tenant=%s, root=%s", user["tenant_id"], root)
    if not root.exists():
        return []
    files: list[KnowledgeFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stat = path.stat()
        files.append(
            KnowledgeFile(
                source_id=path.relative_to(root).as_posix(),
                name=path.name,
                size_bytes=stat.st_size,
                modified_at=stat.st_mtime,
            )
        )
    return files


@app.delete("/knowledge/files/{source_id:path}", status_code=204)
async def delete_knowledge_file(
    source_id: str, user: auth.CurrentUser
) -> None:
    """删除租户知识库中的文件并增量刷新索引。

    - source_id 为相对租户根目录的 POSIX 相对路径（支持子目录）；
    - 路径越界与 wiki/ 保留区校验统一由 Wiki↔RAG 契约层负责，杜绝路径穿越
      与"镜像被删但 DB 版本还在"的不一致。
    """

    normalized = source_id.replace("\\", "/").lstrip("/")
    if not normalized or normalized in {".", ".."}:
        raise HTTPException(status_code=400, detail="非法的文件标识")
    try:
        kc.assert_not_wiki_source(normalized)
        target = kc.resolve_tenant_file(user["tenant_id"], normalized)
    except kc.KnowledgePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    if target.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="不支持的文件类型")

    target.unlink()
    logger.info(
        "知识库文件已删除: tenant=%s, source_id=%s, 操作人=%s",
        user["tenant_id"], normalized, user["email"],
    )
    await kc.arequest_refresh(user["tenant_id"])


# ── 知识健康检查（知识治理 · Lint 保证健康） ────────────────────────


class KnowledgeLintIssue(BaseModel):
    """单条知识健康问题。"""

    severity: str
    code: str
    source_id: str
    message: str


class KnowledgeLintReport(BaseModel):
    """知识健康检查报告：问题列表 + 各级别计数 + 检查文件数。"""

    issues: list[KnowledgeLintIssue]
    error_count: int
    warning_count: int
    info_count: int
    checked_files: int


@app.get("/knowledge/lint", response_model=KnowledgeLintReport)
async def lint_knowledge(user: auth.CurrentUser) -> KnowledgeLintReport:
    """知识健康检查：扫描当前租户 Wiki 镜像区的一致性与 Schema 纪律。

    检查项：孤儿/缺失镜像（与业务库比对）、frontmatter 缺失或违例、
    过期/未生效文档、空正文。原始文档不可变且不受 Schema 约束，不在
    深度检查范围内。
    """

    root = kc.tenant_root(user["tenant_id"])
    pages = await db.list_wiki_pages(user["tenant_id"])
    wiki_slugs = {row["slug"] for row in pages}
    result = await anyio.to_thread.run_sync(lint_tenant_knowledge, root, wiki_slugs)
    counts = {"error": 0, "warning": 0, "info": 0}
    for issue in result.issues:
        counts[issue["severity"]] = counts.get(issue["severity"], 0) + 1
    logger.info(
        "知识健康检查完成: tenant=%s, checked=%d, error=%d, warning=%d, info=%d",
        user["tenant_id"], result.checked_files,
        counts["error"], counts["warning"], counts["info"],
    )
    return KnowledgeLintReport(
        issues=[KnowledgeLintIssue(**issue) for issue in result.issues],
        error_count=counts["error"],
        warning_count=counts["warning"],
        info_count=counts["info"],
        checked_files=result.checked_files,
    )


def settings_skills_root() -> Path:
    """返回已解析的技能根目录。"""

    return get_settings().skills_dir.resolve()


@app.get("/skills", response_model=list[SkillResponse])
async def list_skills_endpoint(user: auth.CurrentUser) -> list[SkillResponse]:
    """列出当前租户可见的全部技能（全局共享 + 租户私有）。"""

    infos = await anyio.to_thread.run_sync(
        skills_manager.list_skills_for_tenant,
        settings_skills_root(),
        user["tenant_id"],
    )
    logger.debug("查询技能列表: tenant=%s, %d 个", user["tenant_id"], len(infos))
    return [SkillResponse.from_info(info) for info in infos]


@app.post("/skills/upload", response_model=SkillResponse, status_code=201)
async def upload_skill(
    user: auth.CurrentUser,
    file: Annotated[UploadFile, File()],
    overwrite: Annotated[bool, Form()] = False,
) -> SkillResponse:
    """上传 zip 技能包并安装到 skills 目录。

    zip 布局要求：根目录为 ``SKILL.md``，或唯一顶层目录 ``<name>/SKILL.md``；
    frontmatter 必须包含与目录名一致的 name 和非空 description。
    安装经过暂存目录原子替换，失败不会留下半成品。技能在下一次对话时自动生效。
    """

    raw_name = file.filename or ""
    logger.info(
        "收到技能上传: 用户=%s, 文件名=%r, overwrite=%s",
        user["email"], raw_name, overwrite,
    )
    if not raw_name.lower().endswith(".zip"):
        raise HTTPException(status_code=415, detail="仅接受 .zip 技能包")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(content) > MAX_SKILL_ZIP_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"压缩包超过 {MAX_SKILL_ZIP_BYTES // (1024 * 1024)}MB 上限",
        )

    root = skills_manager.tenant_skills_root(settings_skills_root(), user["tenant_id"])
    root.mkdir(parents=True, exist_ok=True)
    try:
        info = await anyio.to_thread.run_sync(
            lambda: skills_manager.install_skill_from_zip(root, content, overwrite=overwrite)
        )
    except skills_manager.SkillConflictError as exc:
        logger.warning("技能冲突: tenant=%s, %s", user["tenant_id"], exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except skills_manager.SkillInstallError as exc:
        logger.warning("技能安装被拒绝: tenant=%s, %s", user["tenant_id"], exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "租户私有技能已就绪: tenant=%s, 技能=%s, %d 个文件, 上传者=%s",
        user["tenant_id"], info.name, len(info.files), user["email"],
    )
    return SkillResponse.from_info(info)


@app.delete("/skills/{name}", status_code=204)
async def remove_skill(name: str, user: auth.CurrentUser) -> None:
    """按名称删除租户私有技能（全局共享技能不可通过此接口删除）。"""

    logger.info(
        "收到技能删除请求: tenant=%s, 技能=%s, 操作人=%s",
        user["tenant_id"], name, user["email"],
    )
    root = skills_manager.tenant_skills_root(settings_skills_root(), user["tenant_id"])
    try:
        await anyio.to_thread.run_sync(skills_manager.delete_skill, root, name)
    except skills_manager.SkillInstallError as exc:
        # “不存在”归为 404，其余非法输入归为 400
        status = 404 if "不存在" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, http_request: Request, user: auth.CurrentUser) -> ChatResponse:
    """将用户消息交给 DeepAgents 代理图处理并返回答案。

    - 会话必须存在且属于当前用户（否则 404，杜绝越权访问他人对话）；
    - thread_id 作为 LangGraph checkpoint 键，自动获得多轮上下文；
    - user_id 通过运行时 context 注入，长期记忆按账户隔离。
    """

    started = time.perf_counter()
    thread = await db.get_thread(body.thread_id)
    if thread is None or thread["user_id"] != user["id"]:
        logger.warning(
            "拒绝访问会话: thread_id=%s, 请求用户=%s", body.thread_id, user["email"]
        )
        raise HTTPException(status_code=404, detail="会话不存在")

    logger.info(
        "对话请求: thread_id=%s, user=%s, 消息长度=%d 字符",
        body.thread_id,
        user["email"],
        len(body.message),
    )
    try:
        # 收集检索引用（仿 WeKnora citation assembly）：检索工具在 anyio
        # 工作线程内回调，用锁保证并发安全；按 source 去重、后发覆盖先发，
        # 与前端流式 sourceMap 语义一致
        collected: dict[str, dict] = {}
        collected_lock = threading.Lock()

        def _collect_sources(items: list[dict]) -> None:
            with collected_lock:
                for item in items:
                    collected[item.get("source", "")] = item

        with source_emitter(_collect_sources):
            result = await http_request.app.state.agent.ainvoke(
                {
                    "messages": [{"role": "user", "content": body.message}],
                    # 强制 SkillsMiddleware 每轮重扫 /skills/ 源：
                    # 管理员上传/删除技能后无需重启，下一条消息即加载到模型
                    "skills_metadata": None,
                },
                config={"configurable": {"thread_id": body.thread_id}},
                # 关键：运行时上下文，驱动 StoreBackend 的每用户记忆 namespace 与租户隔离
                context={"user_id": user["id"], "tenant_id": user["tenant_id"]},
            )
    except Exception as exc:
        logger.exception("代理执行失败: thread_id=%s, user=%s", body.thread_id, user["email"])
        raise HTTPException(status_code=502, detail="Agent execution failed") from exc

    # HITL：命中审批工具，代理已在检查点挂起 → 返回待审批载荷
    pending = _pending_interrupts_from_result(result)
    if pending:
        logger.info(
            "HITL 挂起待审批: thread_id=%s, user=%s, 工具=%s",
            body.thread_id,
            user["email"],
            [p.tool for p in pending],
        )
        return ChatResponse(status="pending_approval", interrupts=pending)

    answer = _result_answer(result)

    # 首轮消息自动生成会话标题；之后每轮刷新 updated_at（列表排序用）
    if thread["title"] == _DEFAULT_THREAD_TITLE:
        auto_title = re.sub(r"\s+", " ", body.message).strip()[:_AUTO_TITLE_LEN]
        if auto_title:
            await db.rename_thread(body.thread_id, auto_title)
    else:
        await db.touch_thread(body.thread_id)

    logger.info(
        "对话完成: thread_id=%s, user=%s，答案 %d 字符，耗时 %.2fs",
        body.thread_id,
        user["email"],
        len(answer),
        time.perf_counter() - started,
    )
    # LangMem 后台反思：本轮完成后异步提炼事实记忆（失败仅记日志，不影响响应）
    reflection.schedule_reflection(
        user["tenant_id"], user["id"], result.get("messages", [])
    )
    return ChatResponse(answer=answer, sources=list(collected.values()))


@app.post("/chat/resume", response_model=ChatResponse)
async def chat_resume(
    body: ResumeRequest, http_request: Request, user: auth.CurrentUser
) -> ChatResponse:
    """HITL 审批恢复：携带 decisions 恢复挂起的代理执行。

    - 会话归属校验与 /chat 一致（非属主 404）；
    - 恢复后可能再次挂起（同条消息触发多个敏感操作）→ 再次返回 pending；
    - 重复审批（interrupt 已被消费）→ 409。
    """

    thread = await db.get_thread(body.thread_id)
    if thread is None or thread["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="会话不存在")

    logger.info(
        "HITL 审批恢复: thread_id=%s, user=%s, decisions=%s",
        body.thread_id,
        user["email"],
        [d.get("type") for d in body.decisions],
    )
    try:
        result = await http_request.app.state.agent.ainvoke(
            Command(resume={"decisions": body.decisions}),
            config={"configurable": {"thread_id": body.thread_id}},
            context={"user_id": user["id"], "tenant_id": user["tenant_id"]},
        )
    except Exception as exc:
        # interrupt 已被消费（重复点击）或无挂起状态时 resume 会抛错
        logger.warning(
            "HITL 恢复失败: thread_id=%s, user=%s, %s",
            body.thread_id, user["email"], exc,
        )
        raise HTTPException(status_code=409, detail="该审批已处理或已过期") from exc

    pending = _pending_interrupts_from_result(result)
    if pending:
        logger.info(
            "HITL 再次挂起: thread_id=%s, 工具=%s",
            body.thread_id, [p.tool for p in pending],
        )
        return ChatResponse(status="pending_approval", interrupts=pending)

    await db.touch_thread(body.thread_id)
    # 审批通过后本轮对话真正结束，同样触发后台反思提取
    reflection.schedule_reflection(
        user["tenant_id"], user["id"], result.get("messages", [])
    )
    return ChatResponse(answer=_result_answer(result))


@app.post("/chat/stream")
async def chat_stream(
    body: ChatRequest, http_request: Request, user: auth.CurrentUser
) -> StreamingResponse:
    """SSE 流式对话：逐 token 推送答案，并推送 RAG 检索来源。

    事件类型：
    - ``token``     {"text": "..."}            主代理答案增量；
    - ``sources``   [{"source","section","score"}]  检索到的知识库来源（去重全量）；
    - ``interrupt`` {"interrupts": [...]}       HITL 挂起待审批（随后结束，恢复走 /chat/resume）；
    - ``done``      {}                          本轮结束；
    - ``error``     {"detail": "..."}           执行失败。
    """

    thread = await db.get_thread(body.thread_id)
    if thread is None or thread["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="会话不存在")

    async def event_generator():
        # agent.astream 在独立 pump 任务中运行；检索来源由工具在子代理
        # 内部（可能处于 anyio 工作线程）经 source_emitter 回调线程安全地
        # 投递给本队列，从而绕开 task 子图不冒泡内部事件的限制。
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _emit_sources(sources: list[dict]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ("sources", sources))

        async def _pump() -> None:
            try:
                with source_emitter(_emit_sources):
                    async for message, metadata in http_request.app.state.agent.astream(
                        {
                            "messages": [{"role": "user", "content": body.message}],
                            # 与 /chat 一致：每轮重扫技能目录，新上传技能即时生效
                            "skills_metadata": None,
                        },
                        config={"configurable": {"thread_id": body.thread_id}},
                        context={"user_id": user["id"], "tenant_id": user["tenant_id"]},
                        stream_mode="messages",
                    ):
                        # 仅转发主图层级的 AI 文本 token：
                        # 主图节点 namespace 形如 "model:<run-id>"（不含 "|"），
                        # 子代理（task 子图）内的节点 namespace 形如 "task:<id>|model:<id>"
                        namespace = (metadata or {}).get("langgraph_checkpoint_ns", "")
                        is_ai = type(message).__name__ in {"AIMessage", "AIMessageChunk"}
                        text = getattr(message, "content", "")
                        if is_ai and "|" not in namespace and isinstance(text, str) and text:
                            queue.put_nowait(("token", text))
            except Exception:
                logger.exception(
                    "流式代理执行失败: thread_id=%s, user=%s",
                    body.thread_id, user["email"],
                )
                queue.put_nowait(("fatal", None))
                return
            queue.put_nowait(("end", None))

        pump_task = asyncio.create_task(_pump())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "token":
                    yield _sse(
                        "token",
                        json.dumps({"text": payload}, ensure_ascii=False),
                    )
                elif kind == "sources":
                    yield _sse(
                        "sources",
                        json.dumps({"sources": payload}, ensure_ascii=False),
                    )
                elif kind == "fatal":
                    yield _sse("error", json.dumps({"detail": "Agent execution failed"}))
                    return
                elif kind == "end":
                    break
        finally:
            # 客户端断开等情况下取消后台 pump，避免 agent 继续空跑
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump_task

        # HITL：流结束后从检查点状态检测挂起的审批，通知前端渲染审批卡片
        pending, messages = await _pending_interrupts_from_state(
            http_request.app.state.agent, body.thread_id
        )
        if pending:
            logger.info(
                "HITL 流式挂起待审批: thread_id=%s, 工具=%s",
                body.thread_id, [p.tool for p in pending],
            )
            yield _sse(
                "interrupt",
                json.dumps(
                    {"interrupts": [p.model_dump() for p in pending]},
                    ensure_ascii=False,
                ),
            )
        else:
            # 正常结束（非挂起）才触发后台反思提取
            reflection.schedule_reflection(user["tenant_id"], user["id"], messages)

        if thread["title"] == _DEFAULT_THREAD_TITLE:
            auto_title = re.sub(r"\s+", " ", body.message).strip()[:_AUTO_TITLE_LEN]
            if auto_title:
                await db.rename_thread(body.thread_id, auto_title)
        else:
            await db.touch_thread(body.thread_id)
        yield _sse("done", "{}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 静态前端必须最后挂载：mount 到 "/" 会吞掉所有未被前面路由匹配的路径，
# html=True 使访问根路径时返回 index.html
if _WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
    logger.info("前端静态页面已挂载: %s → /", _WEB_DIR)
else:
    logger.warning("前端目录不存在，跳过页面挂载: %s", _WEB_DIR)
