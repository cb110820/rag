"""文档加载与解析。

递归扫描知识目录，支持多种格式（txt/md/html/pdf/docx），
统一输出纯文本的 SourceDocument 列表。

解析时自动提取文档元数据，用于后续版本过滤与时效性判断：
- doc_mtime：文件修改时间戳
- effective_date / expiry_date：生效/失效日期（ISO 格式；frontmatter
  优先，正文正则兜底）
- version：版本号（frontmatter → 正文 → 文件名，前者优先）
- report_no：报表号/文档编号
- tier：来源层级（"wiki" = Wiki 沉淀镜像，"raw" = 原始上传文档），
  供检索层做权威提权

Markdown 文本若携带合法 YAML frontmatter，会在正文剥离后仅作为
metadata 生效，避免 YAML 噪声污染分块与向量。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

# HTML解析库，用于抽取HTML/HTM文件的纯文本内容
from bs4 import BeautifulSoup

# 文档解析库，用于抽取DOCX文件的纯文本内容
from docx import Document

# PDF解析库，用于抽取PDF文件的纯文本内容
from pypdf import PdfReader

# ── 项目内部导入 ────────────────────────────────────────────────
# 契约层仅依赖 config，无循环导入；用于按 source_id 判定来源层级
from src.rag.knowledge_contract import is_wiki_source

# 文档模型，用于存储解析后的文档内容
from src.rag.models import SourceDocument
from src.rag.wiki_schema import parse_frontmatter

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf", ".docx"}

# ── 元数据提取正则 ────────────────────────────────────────────────
# 生效日期：生效日期 / 生效时间 / 有效起始 / 实施日期
_EFFECTIVE_RE = re.compile(
    r"(?:生效日期|生效时间|有效起始|实施日期|发布日期)[:：\s]*"
    r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)",
    re.IGNORECASE,
)
# 失效日期：失效日期 / 失效时间 / 有效期至 / 废止日期
_EXPIRY_RE = re.compile(
    r"(?:失效日期|失效时间|有效期至|废止日期|过期日期)[:：\s]*"
    r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)",
    re.IGNORECASE,
)
# 版本号：版本 / 版本号 / Version 后跟数字
_VERSION_RE = re.compile(
    r"(?:版本(?:号)?|Version|Rev)[:：\s]*[Vv]?(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
# 文件名中的版本号：policies_v2.md / report_20240101.md
_FILENAME_VERSION_RE = re.compile(r"[_-]v?(\d+(?:\.\d+)*)$", re.IGNORECASE)
# 报表号/编号：报表号 / 编号 / 文号
_REPORT_NO_RE = re.compile(
    r"(?:报表号|编号|文号|文档编号)[:：\s]*([A-Za-z0-9][A-Za-z0-9_\-./]*)",
    re.IGNORECASE,
)


def _normalize_date(raw: str) -> str | None:
    """把 '2024年1月1日' / '2024-01-01' / '2024/1/1' 统一为 ISO 'YYYY-MM-DD'。"""

    m = re.match(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", raw)
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def _extract_document_metadata(text: str, source_id: str) -> dict:
    """从正文与文件名提取时效性/版本元数据。"""

    meta: dict = {}
    m = _EFFECTIVE_RE.search(text)
    if m:
        meta["effective_date"] = _normalize_date(m.group(1))
    m = _EXPIRY_RE.search(text)
    if m:
        meta["expiry_date"] = _normalize_date(m.group(1))
    m = _VERSION_RE.search(text)
    if m:
        meta["version"] = m.group(1)
    m = _REPORT_NO_RE.search(text)
    if m:
        meta["report_no"] = m.group(1).strip()
    # 文件名兜底版本号：xxx_v2.md / xxx_20240101.md
    if "version" not in meta:
        stem = Path(source_id).stem
        fm = _FILENAME_VERSION_RE.search(stem)
        if fm:
            meta["version"] = fm.group(1)
    return meta


def extract_text(path: Path) -> str:
    """根据文件后缀选择对应的解析器，返回纯文本内容。

    公开给 Wiki 文档导入等场景复用：给定任意受支持后缀的文件路径，
    统一抽取纯文本，无需先将文件放入知识库目录。
    """

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix in {".html", ".htm"}:
        return BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser").get_text("\n")
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    raise ValueError(f"Unsupported document type: {suffix}")


def scan_document_stats(root: Path) -> dict[str, tuple[float, int]]:
    """轻量扫描知识库目录，返回 {source_id: (mtime, size)}，不读取文件内容。

    用于增量 refresh 时快速判断哪些文件发生了变更，避免百万级文档全量重读。
    """

    if not root.exists():
        return {}
    stats: dict[str, tuple[float, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        source_id = path.relative_to(root).as_posix()
        st = path.stat()
        stats[source_id] = (st.st_mtime, st.st_size)
    return stats


def _finalize_document(path: Path, source_id: str, text: str) -> SourceDocument | None:
    """统一装配 SourceDocument：剥离 frontmatter、合并元数据、标注层级。

    - frontmatter 值权威，覆盖正文正则提取的同名字段；
    - frontmatter 块从正文剥离后再提取元数据/分块，避免 YAML 噪声；
    - 剥离后正文为空的文档视为无效，返回 None。
    """

    fm, body = parse_frontmatter(text)
    body = body.strip()
    if not body:
        return None
    meta: dict = {"suffix": path.suffix.lower(), "doc_mtime": path.stat().st_mtime}
    meta.update(_extract_document_metadata(body, source_id))
    if fm:
        meta.update({k: v for k, v in fm.items() if isinstance(v, (str, list))})
    meta["tier"] = "wiki" if is_wiki_source(source_id) else "raw"
    return SourceDocument(source_id=source_id, text=body, path=path, metadata=meta)


def load_document(root: Path, source_id: str) -> SourceDocument | None:
    """加载单个文档（按 source_id 相对路径），内容为空时返回 None。"""

    path = root / source_id
    if not path.exists():
        return None
    text = extract_text(path).strip()
    if not text:
        return None
    return _finalize_document(path, source_id, text)


def load_documents(root: Path) -> list[SourceDocument]:
    """递归扫描 root 目录，加载所有支持格式的文档。

    source_id 使用相对于 root 的 POSIX 路径，方便在检索结果中标注来源。
    """

    if not root.exists():
        logger.warning("知识库目录不存在，按空知识库处理: %s", root)
        return []
    logger.info("开始扫描知识库目录: %s", root)
    documents: list[SourceDocument] = []
    skipped = 0
    for path in sorted(
        item for item in root.rglob("*") if item.suffix.lower() in SUPPORTED_SUFFIXES
    ):
        source_id = path.relative_to(root).as_posix()
        logger.debug(
            "解析文档 %s（%s, %d bytes）",
            source_id,
            path.suffix.lower(),
            path.stat().st_size,
        )
        text = extract_text(path).strip()
        doc = _finalize_document(path, source_id, text) if text else None
        if doc is not None:
            documents.append(doc)
            logger.debug("文档 %s 解析完成，提取 %d 字符", source_id, len(doc.text))
        else:
            skipped += 1
            logger.warning("文档 %s 解析后内容为空，已跳过", source_id)
    logger.info(
        "知识库扫描完成: 加载 %d 个文档（跳过 %d 个空文档），共 %d 字符",
        len(documents),
        skipped,
        sum(len(doc.text) for doc in documents),
    )
    return documents