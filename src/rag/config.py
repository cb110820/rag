"""全局配置管理。

通过 pydantic-settings 从 .env 文件和环境变量加载配置，
提供 get_settings() 单例供全项目复用。
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from uvicorn.logging import AccessFormatter

# 统一日志格式：时间 | 级别 | 模块名 | 消息
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# uvicorn 访问日志格式（参数名固定：client_addr / request_line / status_code）
_ACCESS_LOG_FORMAT = (
    '%(asctime)s | %(levelname)-7s | uvicorn.access               '
    '| %(client_addr)s - "%(request_line)s" %(status_code)s'
)
# 项目内所有业务 logger 都在 src 命名空间下
_PROJECT_LOGGER_NAME = "src"
# 需要统一接管的 uvicorn logger（uvicorn.error 默认冒泡到 uvicorn，无需单独挂 handler）
_UVICORN_LOGGER_NAME = "uvicorn"
_UVICORN_ACCESS_LOGGER_NAME = "uvicorn.access"
# 应用根目录与只读资源目录：
# - 开发态：项目根目录（src/rag/config.py 向上两级），两者相同；
# - PyInstaller 打包态：APP_DIR 为 exe 所在目录（可写，存放 .env/.data/logs/
#   .cache/skills/data 等运行期数据），RESOURCE_DIR 为 _internal  bundle
#   目录（只读，存放随包发布的种子资源），由 launcher 在首启时拷贝到 APP_DIR。
FROZEN: bool = bool(getattr(sys, "frozen", False))
APP_DIR: Path = (
    Path(sys.executable).resolve().parent
    if FROZEN
    else Path(__file__).resolve().parents[2]
)
RESOURCE_DIR: Path = (
    Path(str(getattr(sys, "_MEIPASS", APP_DIR))) if FROZEN else APP_DIR
)
# 持久化日志文件：<APP_DIR>/logs/app.log，按天切割，保留 14 天
LOG_FILE_PATH = APP_DIR / "logs" / "app.log"
_LOG_BACKUP_DAYS = 14


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Windows 友好的按天切割 handler。

    同机多进程（多 worker 或历史遗留进程）仍持有 app.log 句柄时，
    ``os.rename`` 会抛 PermissionError（WinError 32）。此时放弃本次切割、
    重开当前文件继续写，保证日志系统绝不因切割失败而中断，下个自然日
    会再次尝试切割。
    """

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except PermissionError:
            logging.getLogger(__name__).warning(
                "日志切割被跳过：%s 正被其他进程占用，继续写入当前文件",
                self.baseFilename,
            )
            if self.stream:
                self.stream.close()
            self.stream = self._open()


class _DualFormatter(logging.Formatter):
    """同一文件 handler 复用时的双格式分发器。

    业务/uvicorn 主日志走标准格式，uvicorn.access 记录走 AccessFormatter
    （解析 5 元组参数注入 client_addr/request_line/status_code）。
    """

    def __init__(
        self,
        business_fmt: str,
        access_fmt: str,
        datefmt: str,
        access_logger_name: str,
    ) -> None:
        super().__init__(business_fmt, datefmt)
        self._access_formatter = AccessFormatter(access_fmt, datefmt)
        self._access_logger_name = access_logger_name

    def format(self, record: logging.LogRecord) -> str:
        if record.name == self._access_logger_name:
            return self._access_formatter.format(record)
        return super().format(record)


def _load_env_into_environ() -> None:
    """把项目 .env 的键值注入 os.environ，供 openai SDK 等第三方库读取。

    pydantic-settings 只会把 .env 映射到 Settings 字段（extra 键还会被忽略），
    不会写入进程环境；而 ChatOpenAI 初始化时直接从 os.environ 读取
    OPENAI_API_KEY / OPENAI_BASE_URL，因此需要在最早的配置模块中显式引导。
    已存在的真实环境变量优先，不被 .env 覆盖。
    """

    # 优先当前工作目录（开发态习惯），打包态回退到 exe 所在目录
    env_path = Path(".env")
    if not env_path.exists() and FROZEN:
        env_path = APP_DIR / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_into_environ()


def configure_logging(level: str | None = None) -> logging.Logger:
    """幂等配置项目日志，返回 src 命名空间的根 logger。

    - 只给 ``src`` logger 绑定 handler 并设置 propagate=False，
      避免 FastAPI/uvicorn 自身 handler 叠加导致同一条日志打印两次；
    - 同时接管 uvicorn / uvicorn.access 两个 logger，使 HTTP 访问日志、
      启动/错误日志也带上精确到秒的完整日期（默认 uvicorn 输出无时间），
      并与业务日志统一写入终端和 logs/app.log；
    - 两个输出渠道：stderr（终端实时查看）+ logs/app.log（持久化，
      每天零点切割，归档为 app.log.YYYY-MM-DD，保留 14 天）；
    - 日志文件显式使用 UTF-8，保证 Windows 下中文不乱码；
    - 重复调用安全（已有 handler 时仅调整级别），适配 uvicorn reload、
      测试多入口等场景。

    Args:
        level: 显式日志级别（如 ``"DEBUG"``），None 时读 Settings.log_level。
    """

    project_logger = logging.getLogger(_PROJECT_LOGGER_NAME)
    # 必须在首次发出 INFO 前设置级别，否则未初始化的 logger 继承根 logger
    # 默认的 WARNING，启动横幅会被静默丢弃
    project_logger.setLevel((level or get_settings().log_level).upper())
    if not project_logger.handlers:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
        # 必须使用 uvicorn 自带的 AccessFormatter：访问记录以 5 个位置参数
        # 传入 logger.info，由它解析并注入 client_addr / request_line /
        # status_code 字段，标准 Formatter 无法识别这些字段会抛异常
        access_formatter = AccessFormatter(
            _ACCESS_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT
        )

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)

        # 全进程只保留一个文件 handler：Windows 下两个 handler 同时打开
        # app.log 会导致切割 rename 时互相占用（WinError 32）。双格式由
        # _DualFormatter 按 logger 名分发；SafeTimedRotatingFileHandler
        # 再兜底跨进程占用，切割失败也不中断日志。
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_handler = SafeTimedRotatingFileHandler(
            LOG_FILE_PATH,
            when="midnight",
            backupCount=_LOG_BACKUP_DAYS,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            _DualFormatter(
                _LOG_FORMAT,
                _ACCESS_LOG_FORMAT,
                _LOG_DATE_FORMAT,
                _UVICORN_ACCESS_LOGGER_NAME,
            )
        )

        project_logger.addHandler(stream_handler)
        project_logger.addHandler(file_handler)
        project_logger.propagate = False

        # 接管 uvicorn 主 logger（uvicorn.error 无独立 handler，
        # 默认冒泡到 uvicorn，会自动复用这里的格式）
        uvicorn_logger = logging.getLogger(_UVICORN_LOGGER_NAME)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(stream_handler)
        uvicorn_logger.addHandler(file_handler)
        uvicorn_logger.setLevel(logging.INFO)
        uvicorn_logger.propagate = False

        # 访问日志 stderr 通道需要独立的 AccessFormatter；文件通道复用
        # 上面的共享 file_handler（双格式分发），避免二次打开同一文件
        access_stream_handler = logging.StreamHandler(sys.stderr)
        access_stream_handler.setFormatter(access_formatter)
        access_logger = logging.getLogger(_UVICORN_ACCESS_LOGGER_NAME)
        access_logger.handlers.clear()
        access_logger.addHandler(access_stream_handler)
        access_logger.addHandler(file_handler)
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False

        project_logger.info(
            "日志文件已启用: %s（按天切割，保留 %d 天；业务日志与 uvicorn "
            "访问日志均已带精确到秒的完整日期）",
            LOG_FILE_PATH,
            _LOG_BACKUP_DAYS,
        )
    return project_logger


class Settings(BaseSettings):
    """应用级配置项。

    字段名会自动映射到同名环境变量（大小写不敏感），也可在 .env 文件中设置。
    """

    # 打包态 .env 固定在 exe 同级目录；开发态沿用当前工作目录的 .env
    model_config = SettingsConfigDict(
        env_file=str(APP_DIR / ".env") if FROZEN else ".env",
        extra="ignore",
    )

    # 模型提供方：openai（含兼容端点，值用 openai:model 形式）或 dashscope（阿里云原生）
    model_provider: str = "openai"
    model: str = "openai:gpt-5.5"
    knowledge_dir: Path = Path("data/knowledge")
    # 员工技能（Agent Skills）根目录，每个技能一个含 SKILL.md 的子目录
    skills_dir: Path = Path("skills")
    # 嵌入/重排提供方：生产用 bge-m3；本地无模型权重时可设 hash / none 零依赖运行
    rag_embedding: str = "bge-m3"
    rag_reranker: str = "bge-reranker-v2-m3"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # 语料块嵌入向量的 SQLite 缓存路径，重启后无需重新嵌入
    embedding_cache_path: Path = Path(".cache/embeddings.sqlite3")
    # 日志级别：DEBUG / INFO / WARNING / ERROR
    log_level: str = "INFO"
    rag_top_k: int = Field(default=6, ge=1, le=50)
    # Wiki 镜像（沉淀层）相对原始文档（兜底层）的分数提权系数，
    # 用于"Wiki 权威、原始文档兜底"的来源分层
    wiki_tier_boost: float = Field(default=1.15, gt=1.0, le=2.0)
    rag_timeout_seconds: float = Field(default=30.0, gt=0)
    rag_max_retries: int = Field(default=3, ge=1, le=10)
    memory_files: str = ""

    # ── LLM Wiki 自动沉淀（src/curation/） ──────────────────────────
    # 上传后自动沉淀总开关；失败只记日志，不阻断上传
    wiki_curation_enabled: bool = True
    # 沉淀专用模型规格，空串跟随主模型；后台一律 streaming=False
    wiki_curation_model: str = ""
    # 单文档沉淀总超时（含 LLM 调用）
    wiki_curation_timeout_seconds: float = Field(default=180.0, gt=0)
    # 单文档最多抽取实体数（LLM Wiki 实践建议 5-15）
    wiki_curation_max_entities: int = Field(default=10, ge=5, le=15)
    # 喂给 LLM 的文档全文字符上限（超长截断）
    wiki_curation_max_chars: int = Field(default=60000, ge=1000)

    # ── 查询理解（仿 WeKnora QUERY_UNDERSTAND，检索前 LLM 改写） ─────
    # 开关：默认关闭（每次检索 +1 次 LLM 调用；关闭时零开销，eval 基线不受影响）
    query_understand_enabled: bool = False
    # 查询理解独立超时（秒）；超时/失败降级为原始查询，绝不阻断检索
    query_understand_timeout_seconds: float = Field(default=3.5, gt=0)
    # 查询理解模型规格，空串跟随主模型；一律 streaming=False
    query_understand_model: str = ""

    # ── 实体检索（仿 WeKnora ENTITY_SEARCH，利用 entity-* Wiki 页召回） ──
    # 纯离线（无 LLM）：基于实体名/别名归一化匹配
    entity_search_enabled: bool = True
    # 单次检索最多注入的未召回实体页数量（防止挤占 top_k）
    entity_search_max_hits: int = Field(default=1, ge=0, le=5)

    # ── 持久化后端（生产部署） ─────────────────────────────────────
    # sqlite：单机默认（本地开发/小规模部署）；postgres：多副本生产部署
    storage_backend: str = "sqlite"
    # psycopg 连接串，STORAGE_BACKEND=postgres 时必填，
    # 形如 postgresql://user:password@host:5432/dbname?sslmode=require
    postgres_dsn: str = ""
    # 业务/checkpointer/store 三类访问者各自独立的连接池尺寸
    pg_pool_min: int = Field(default=1, ge=0)
    pg_pool_max: int = Field(default=10, ge=1)

    # ── 长期记忆治理（模型写 /memory/ 时强制生效，prompt 约束之外的硬护栏） ──
    # 单个记忆文件（如 /memory/AGENTS.md）的最大字节数
    memory_max_file_bytes: int = Field(default=8192, ge=256)
    # 每个用户 namespace 下允许的记忆文件/条目数上限
    memory_max_items: int = Field(default=20, ge=1)

    # ── 语义事实记忆（P2：save_memory/recall_memories/forget_memory） ──
    # 向量嵌入来源：留空时跟随 RAG_EMBEDDING；可显式设 bge-m3 / hash / none。
    # none 时关闭向量索引，recall_memories 退化为本地关键词打分。
    memory_embedding: str = ""
    # 语义记忆召回条数；低于该相似度的命中视为噪声丢弃
    memory_recall_top_k: int = Field(default=5, ge=1, le=20)
    memory_recall_min_score: float = Field(default=0.2, ge=0.0, le=1.0)
    # 单条事实最大字符数 / 每用户事实条目上限 / 写入去重的相似度阈值
    memory_fact_max_chars: int = Field(default=1000, ge=10)
    memory_fact_max_items: int = Field(default=200, ge=1)
    memory_dedup_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    # 事实默认 TTL（天）：0 表示不过期；工具调用可显式传 ttl_days
    memory_default_ttl_days: int = Field(default=0, ge=0)
    # 被新事实取代（superseded）的旧版本保留天数：审计窗口内可回溯，
    # 到期由现有 TTL sweeper 物理清除；若原条目 TTL 更早则以更早者为准
    memory_superseded_ttl_days: int = Field(default=14, ge=1)
    # 记忆工具超时与重试（生产红线：所有工具必须有超时与失败重试）
    memory_timeout_seconds: float = Field(default=10.0, gt=0)
    memory_max_retries: int = Field(default=2, ge=1, le=5)

    # ── 记忆混合检索（dense 向量 + sparse BM25 → RRF 粗排 → recency 精排） ──
    # 总开关：关闭后回退到单通道旧行为（向量模式仅 semantic、none 模式仅 coverage 打分）
    memory_hybrid_retrieval_enabled: bool = True
    # 单路召回候选池上限（dense 与 sparse 各自的取数深度，融合后再截 top_k）
    memory_hybrid_candidate_k: int = Field(default=20, ge=5, le=100)
    # RRF 平滑常数：越大则名次间差距越平缓（与 RAG HybridRetriever 同口径 60）
    memory_rrf_k: int = Field(default=60, ge=1)
    # 两路通道权重（加权 RRF）：调大某路等于在融合时给该路更多话语权
    memory_dense_channel_weight: float = Field(default=1.0, gt=0)
    memory_sparse_channel_weight: float = Field(default=1.0, gt=0)
    # recency 精排半衰期（天）：同等相关下每过一个半衰期，老事实权重减半；
    # 0 = 关闭时间衰减，纯按 RRF 相关性排序
    memory_recency_half_life_days: float = Field(default=30.0, ge=0)
    # 衰减下限：老事实最低保留的权重比例，防止被压到不可召回
    memory_recency_floor: float = Field(default=0.5, ge=0.1, le=1.0)

    # ── 长会话上下文压缩（SummarizationMiddleware） ──
    memory_summary_enabled: bool = True
    # 消息历史超过该 token 数触发摘要；触发后保留最近 N 条原文
    memory_summary_trigger_tokens: int = Field(default=6000, ge=500)
    memory_summary_keep_messages: int = Field(default=20, ge=2)

    # ── LangMem 后台反思提取（对话结束后异步提炼事实记忆） ──
    # 总开关：关闭后 /chat 系列端点不再调度后台提取任务
    memory_reflection_enabled: bool = True
    # 提取模型：留空跟随主模型（MODEL_PROVIDER/MODEL）
    memory_reflection_model: str = ""
    # 每次提取取对话末尾 N 条消息（控制 LLM token 成本）
    memory_reflection_max_messages: int = Field(default=12, ge=2, le=100)
    # 整个后台提取流程的超时（秒）；超时仅记日志，不影响主对话
    memory_reflection_timeout_seconds: float = Field(default=90.0, gt=0)
    # 对话文本总长度低于该值不触发提取（寒暄类对话无提炼价值）
    memory_reflection_min_chars: int = Field(default=8, ge=0)

    # ── 跨会话情景记忆（episodic：检索历史会话，而非事实条目） ──
    # 总开关：关闭后 search_past_conversations 直接返回未启用提示
    memory_episodic_enabled: bool = True
    # 每次检索最多扫描的候选会话数（按 updated_at 倒序、时间窗内取）
    memory_episodic_max_threads: int = Field(default=10, ge=1, le=50)
    # 候选会话的时间窗（天）：updated_at 早于该窗口的会话不参与检索
    memory_episodic_since_days: int = Field(default=90, ge=1)
    # 最终返回的会话条数 / 每个会话摘录的最大字符数
    memory_episodic_top_k: int = Field(default=3, ge=1, le=10)
    memory_episodic_max_chars: int = Field(default=600, ge=100)
    # 跨会话检索需要逐个读取 checkpoint，超时单独放宽
    memory_episodic_timeout_seconds: float = Field(default=15.0, gt=0)

    # ── Human-in-the-loop 审批检查点 ──
    # 命中列表中的工具调用前挂起，等待前端 /chat/resume 审批后恢复执行；
    # 空列表 = 关闭审批（全部直通）
    hitl_interrupt_tools: list[str] = Field(
        default_factory=lambda: ["forget_memory", "write_file", "edit_file"]
    )

    # ── 记忆 TTL 后台清扫 ──
    # 后台清扫间隔（分钟）；0 表示不启动清扫协程（仍在读取时惰性过期）
    ttl_sweep_interval_minutes: int = Field(default=60, ge=0)

    # ── 账户与多用户记忆（生产级） ─────────────────────────────────
    # 业务数据根目录：账号、会话元数据、LangGraph checkpoint、长期记忆均在其中
    data_dir: Path = Path(".data")
    # JWT 签名密钥；生产环境务必通过环境变量 JWT_SECRET_KEY 注入。
    # 留空时首次启动自动生成并持久化到 .data/.jwt_secret（仅适合单机/开发）
    jwt_secret_key: str = ""
    # access token 有效期（分钟）与 refresh token 有效期（天）
    access_token_ttl_minutes: int = Field(default=30, ge=1, le=1440)
    refresh_token_ttl_days: int = Field(default=7, ge=1, le=365)
    # 是否开放自助注册（False 时仅管理员脚本建号）
    allow_registration: bool = True

    @property
    def db_path(self) -> Path:
        """账号 / 会话 / checkpoint / 长期记忆共用的 SQLite 文件。"""

        return self.data_dir / "app.db"

    @property
    def jwt_secret_file(self) -> Path:
        return self.data_dir / ".jwt_secret"

    @property
    def memory_file_paths(self) -> tuple[str, ...]:
        """将逗号分隔的 memory_files 字符串解析为文件路径元组。"""

        return tuple(item.strip() for item in self.memory_files.split(",") if item.strip())

    @property
    def memory_embedding_kind(self) -> str:
        """语义记忆嵌入类型：显式 MEMORY_EMBEDDING 优先，否则跟随 RAG_EMBEDDING。"""

        kind = (self.memory_embedding or self.rag_embedding).strip().lower()
        return kind if kind in {"bge-m3", "hash", "none"} else "hash"

    @property
    def memory_vector_enabled(self) -> bool:
        """是否启用 Store 向量索引（none 时关闭，召回走关键词兜底）。"""

        return self.memory_embedding_kind != "none"

    @property
    def memory_vector_dims(self) -> int:
        """向量维度：必须与实际嵌入模型一致（建索引时固定，不可中途变更）。"""

        return 1024 if self.memory_embedding_kind == "bge-m3" else 768


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局 Settings 单例，首次调用时读取 .env 并缓存。"""

    return Settings()