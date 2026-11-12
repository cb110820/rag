"""DeepAgents 代理图构建。

主代理（rag_supervisor）负责规划和综合，在遇到需要知识库检索的问题时，
会通过 task 委托 RAG 子代理（rag_retriever）去调用 retrieve_knowledge 工具。

多用户记忆隔离（生产级）：

- **短期记忆（多轮对话）**：全局共享一个持久化 ``AsyncSqliteSaver``，
  LangGraph 按 ``thread_id`` 天然隔离；会话归属由业务层 threads 表鉴权；
- **长期记忆（跨会话事实）**：``/memory/AGENTS.md`` 路由到 ``StoreBackend``，
  namespace 在运行时由 ``Runtime.context.user_id`` 决定，
  形如 ``("user_memory", "<user_id>")``。代理图是全局单例，
  但每次 ainvoke 传入不同的 ``context={"user_id": ...}``，
  模型读写的记忆文件天然按账户隔离；
- **技能目录只读**：``/skills/`` 包装为只读后端，模型只能查看员工技能，
  write/edit/delete 一律拒绝；只有 ``/memory/`` 路径允许模型沉淀记忆。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from deepagents.backends.protocol import DeleteResult, EditResult, WriteResult
from deepagents.backends.utils import file_data_to_string, perform_string_replacement
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from src.agents.prompts import MAIN_SYSTEM_PROMPT, SUMMARY_PROMPT
from src.agents.subagents import build_rag_subagent
from src.memory import episodic
from src.memory.guard import MemoryRejectedError, ensure_write_allowed
from src.memory.tools import MEMORY_TOOLS
from src.rag.config import get_settings
from src.skills import list_skills

logger = logging.getLogger(__name__)


class AgentContext(BaseModel):
    """每次代理执行的运行时上下文（由 API 层从登录态注入）。"""

    user_id: str
    tenant_id: str


class ReadOnlyFilesystemBackend(FilesystemBackend):
    """只读文件后端：用于挂载员工技能目录。

    读方法（ls/read/glob/grep）全部继承磁盘实现；
    任何写、改、删操作直接抛 PermissionError，防止模型篡改技能包。
    """

    def write(self, file_path: str, content: str):  # type: ignore[override]
        raise PermissionError(f"技能目录为只读，禁止写入: {file_path}")

    def edit(  # type: ignore[override]
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        raise PermissionError(f"技能目录为只读，禁止修改: {file_path}")

    def delete(self, file_path: str):  # type: ignore[override]
        raise PermissionError(f"技能目录为只读，禁止删除: {file_path}")


class GuardedStoreBackend(StoreBackend):
    """带安全/容量护栏的长期记忆后端。

    在 ``StoreBackend`` 的虚拟文件能力上增加强制策略（不依赖 prompt 自觉）：

    - 写入/编辑内容经过凭据扫描，命中密钥特征直接拒绝（工具错误返回给模型，
      错误信息不含命中原文）；
    - 单文件字节上限、每用户记忆条目数上限；
    - 所有写入/编辑/删除操作输出审计日志（user_id、路径、字节数）。

    编辑路径会先在本地计算替换后的完整内容再校验，保证被拦内容绝不全量
    落盘；sync/async 两套方法均覆盖（图内 ainvoke 走 async 版本）。
    """

    def __init__(
        self,
        *,
        max_file_bytes: int,
        max_items: int,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._max_file_bytes = max_file_bytes
        self._max_items = max_items

    # ── 内部辅助 ────────────────────────────────────────────────────

    def _audit(
        self,
        action: str,
        namespace: tuple[str, ...],
        file_path: str,
        *,
        size_bytes: int | None = None,
    ) -> None:
        user_id = namespace[-1] if namespace else "?"
        logger.info(
            "长期记忆%s: user_id=%s path=%s size_bytes=%s",
            action,
            user_id,
            file_path,
            size_bytes if size_bytes is not None else "-",
        )

    def _validate(
        self,
        namespace: tuple[str, ...],
        file_path: str,
        content: str,
        *,
        new_file: bool,
        item_count: int,
    ) -> str | None:
        """策略校验；通过返回 None，违规返回可直接展示给模型的错误文案。"""

        try:
            ensure_write_allowed(
                content,
                max_file_bytes=self._max_file_bytes,
                new_file=new_file,
                item_count=item_count,
                max_items=self._max_items,
            )
        except MemoryRejectedError as exc:
            logger.warning(
                "长期记忆写入被拒: user_id=%s path=%s",
                namespace[-1] if namespace else "?",
                file_path,
            )
            return str(exc)
        return None

    def _item_count(self, store: BaseStore, namespace: tuple[str, ...]) -> int:
        return len(store.search(namespace, limit=self._max_items + 1))

    async def _aitem_count(
        self, store: BaseStore, namespace: tuple[str, ...]
    ) -> int:
        items = await store.asearch(namespace, limit=self._max_items + 1)
        return len(items)

    # ── write ───────────────────────────────────────────────────────

    def write(self, file_path: str, content: str) -> WriteResult:  # type: ignore[override]
        store = self._get_store()
        namespace = self._get_namespace()
        new_file = store.get(namespace, file_path) is None
        error = self._validate(
            namespace,
            file_path,
            content,
            new_file=new_file,
            item_count=self._item_count(store, namespace),
        )
        if error is not None:
            return WriteResult(error=error)
        result = super().write(file_path, content)
        if result.error is None:
            self._audit("写入", namespace, file_path, size_bytes=len(content.encode()))
        return result

    async def awrite(  # type: ignore[override]
        self, file_path: str, content: str
    ) -> WriteResult:
        store = self._get_store()
        namespace = self._get_namespace()
        new_file = await store.aget(namespace, file_path) is None
        error = self._validate(
            namespace,
            file_path,
            content,
            new_file=new_file,
            item_count=await self._aitem_count(store, namespace),
        )
        if error is not None:
            return WriteResult(error=error)
        result = await super().awrite(file_path, content)
        if result.error is None:
            self._audit("写入", namespace, file_path, size_bytes=len(content.encode()))
        return result

    # ── edit：先本地推演替换结果再校验，违禁内容绝不全量落盘 ──────────

    def _preview_edit(
        self,
        old_content: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
    ) -> tuple[str | None, str | None]:
        """返回 (new_content, error)：与父类相同的替换语义。"""

        replacement = perform_string_replacement(
            old_content, old_string, new_string, replace_all
        )
        if isinstance(replacement, str):
            return None, replacement
        new_content, _occurrences = replacement
        return new_content, None

    def edit(  # type: ignore[override]
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        store = self._get_store()
        namespace = self._get_namespace()
        item = store.get(namespace, file_path)
        if item is None:
            return EditResult(error=f"Error: File '{file_path}' not found")
        old_content = file_data_to_string(self._convert_store_item_to_file_data(item))
        new_content, preview_error = self._preview_edit(
            old_content, old_string, new_string, replace_all
        )
        if preview_error is not None:
            return EditResult(error=preview_error)
        assert new_content is not None
        error = self._validate(
            namespace, file_path, new_content, new_file=False, item_count=0
        )
        if error is not None:
            return EditResult(error=error)
        result = super().edit(file_path, old_string, new_string, replace_all)
        if result.error is None:
            self._audit(
                "修改", namespace, file_path, size_bytes=len(new_content.encode())
            )
        return result

    async def aedit(  # type: ignore[override]
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        store = self._get_store()
        namespace = self._get_namespace()
        item = await store.aget(namespace, file_path)
        if item is None:
            return EditResult(error=f"Error: File '{file_path}' not found")
        old_content = file_data_to_string(self._convert_store_item_to_file_data(item))
        new_content, preview_error = self._preview_edit(
            old_content, old_string, new_string, replace_all
        )
        if preview_error is not None:
            return EditResult(error=preview_error)
        assert new_content is not None
        error = self._validate(
            namespace, file_path, new_content, new_file=False, item_count=0
        )
        if error is not None:
            return EditResult(error=error)
        result = await super().aedit(file_path, old_string, new_string, replace_all)
        if result.error is None:
            self._audit(
                "修改", namespace, file_path, size_bytes=len(new_content.encode())
            )
        return result

    # ── delete：审计 ────────────────────────────────────────────────

    def delete(self, file_path: str) -> DeleteResult:  # type: ignore[override]
        namespace = self._get_namespace()
        result = super().delete(file_path)
        if result.error is None:
            self._audit("删除", namespace, file_path)
        return result

    async def adelete(self, file_path: str) -> DeleteResult:  # type: ignore[override]
        namespace = self._get_namespace()
        result = await super().adelete(file_path)
        if result.error is None:
            self._audit("删除", namespace, file_path)
        return result


def resolve_chat_model(
    model_spec: str | None = None, *, streaming: bool = True
) -> BaseChatModel | str:
    """根据 MODEL_PROVIDER 解析模型：字符串（openai 系）或已构造的模型实例。

    阿里云部分密钥仅对 DashScope 原生端点授权、OpenAI 兼容端点会返回
    InvalidApiKey，因此 dashscope 提供方走 ChatTongyi 原生 SDK 协议。

    Args:
        model_spec: 模型标识，None 时用配置的主模型。
        streaming: token 级流式开关。主代理置 True 配合 /chat/stream；
            后台反思等工具调用场景必须置 False——ChatTongyi 流式增量
            tool_calls 存在 subtract_client_response 越界 bug。
    """

    settings = get_settings()
    spec = model_spec or settings.model
    provider = settings.model_provider.strip().lower()
    if provider == "dashscope":
        from langchain_community.chat_models.tongyi import ChatTongyi

        model_name = spec.split(":", 1)[1] if ":" in spec else spec
        logger.info("使用 DashScope 原生模型: %s（streaming=%s）", model_name, streaming)
        return ChatTongyi(
            model=model_name,
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
            # 开启 token 级流式，配合 /chat/stream 的 stream_mode="messages"
            streaming=streaming,
        )
    logger.info("使用 OpenAI 兼容模型: %s", spec)
    return spec


def build_agent(
    *,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    model: str | None = None,
    memory_files: Sequence[str] | None = None,
):
    """构建完整的 DeepAgents 主代理（应用启动时调用一次，全局单例复用）。

    Args:
        checkpointer: 持久化检查点存（AsyncSqliteSaver），承载多轮对话。
        store: 持久化键值存储（SqliteStore），承载每用户长期记忆。
        model: LLM 模型标识符，None 时按 MODEL_PROVIDER/MODEL 配置解析。
        memory_files: 额外的全局记忆文件（可选，配置项 MEMORY_FILES）。
    """

    settings = get_settings()
    selected_model = resolve_chat_model(model)
    global_memory_files = list(
        memory_files if memory_files is not None else settings.memory_file_paths
    )

    # 复合后端：
    # - /skills/ → 磁盘技能目录（只读包装）
    # - /memory/ → 长期记忆 Store，namespace 按运行时 user_id 动态隔离
    # - 默认    → StateBackend（会话内临时文件不落盘）
    skills_root = settings.skills_dir.resolve()
    skills_root.mkdir(parents=True, exist_ok=True)
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/": ReadOnlyFilesystemBackend(
                root_dir=skills_root, virtual_mode=True
            ),
            "/memory/": GuardedStoreBackend(
                store=store,
                namespace=lambda rt: (
                    "tenant", rt.context.tenant_id, "user_memory", rt.context.user_id
                ),
                max_file_bytes=settings.memory_max_file_bytes,
                max_items=settings.memory_max_items,
            ),
        },
    )

    middleware: list = [
        # 工具层开放写能力（模型需要 edit_file 沉淀长期记忆），
        # 路径级权限由后端保证：/skills/ 只读、/memory/ 可写
        FilesystemMiddleware(
            backend=backend,
            tools=["ls", "read_file", "glob", "grep", "write_file", "edit_file"],
        ),
        TodoListMiddleware(),
    ]
    # 长会话防膨胀（deepagents 非破坏式压缩）：超过 token 阈值时，模型请求只携带
    # 摘要 + 最近 N 条原文；state["messages"] 原始消息不删除，被压缩原文同时
    # offload 到 backend 的 /conversation_history/<session>.md（随 checkpoint 持久化），
    # 摘要消息带 read_file 回溯指针，模型可按需读回完整历史。
    # 自定义摘要 prompt 强制原样保留金额/阈值等关键数字与审批规则，
    # 防止滚动摘要把"大于 5000 需人工审核"这类约束压缩成模糊表述。
    if settings.memory_summary_enabled:
        middleware.append(
            SummarizationMiddleware(
                model=selected_model,
                backend=backend,
                trigger=[("tokens", settings.memory_summary_trigger_tokens)],
                keep=("messages", settings.memory_summary_keep_messages),
                summary_prompt=SUMMARY_PROMPT,
            )
        )

    available_skills = [s for s in list_skills(skills_root) if s.valid]
    logger.info(
        "构建 DeepAgents 代理图：子代理=[rag_retriever]，"
        "长期记忆=每用户独立 AGENTS.md（Store 隔离）+ 语义事实工具=%d 个，"
        "上下文压缩=%s（%d tokens 触发/保留 %d 条），全局记忆文件=%d 个，"
        "可用技能=%d 个 %s",
        len(MEMORY_TOOLS),
        "on" if settings.memory_summary_enabled else "off",
        settings.memory_summary_trigger_tokens,
        settings.memory_summary_keep_messages,
        len(global_memory_files),
        len(available_skills),
        [s.name for s in available_skills],
    )
    agent = create_deep_agent(
        model=selected_model,
        # 主代理仅持有个人记忆工具；RAG 检索仍只在 rag_retriever 子代理手中。
        # 语义事实三件套 + as-of 时点召回 + 跨会话情景记忆检索
        tools=[*MEMORY_TOOLS, *episodic.EPISODIC_TOOLS],
        system_prompt=MAIN_SYSTEM_PROMPT,
        subagents=[build_rag_subagent(selected_model)],
        middleware=middleware,
        backend=backend,
        skills=["/skills/"],
        # 每用户私有长期记忆（Store 动态 namespace）+ 可选的全局记忆文件
        memory=["/memory/AGENTS.md", *global_memory_files],
        checkpointer=checkpointer,
        store=store,
        context_schema=AgentContext,
        # HITL 审批检查点：命中配置工具的调用前挂起，等 /chat/resume 恢复；
        # 中断状态随检查点落盘，服务重启后审批仍可恢复
        interrupt_on={name: True for name in settings.hitl_interrupt_tools},
        name="rag_supervisor",
    )
    # 情景记忆工具通过该图读取 checkpoint 历史（依赖倒置，避免 memory→agents 反向依赖）
    episodic.register_agent_graph(agent)
    logger.info("代理图构建完成：rag_supervisor")
    return agent
