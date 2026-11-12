"""Wiki 页面 frontmatter Schema（知识治理 · Schema 约束纪律）。

Wiki 页面正文允许携带 YAML frontmatter（``---`` 包裹的键值块），承载
结构化元数据：title / summary / tags / version / effective_date /
expiry_date，以及 LLM 自动沉淀层使用的 kind / sources / aliases。

纪律规则：
- frontmatter **缺省合法**：纯正文页面视为未声明元数据（迁移容错）；
- frontmatter **存在则必须严格合法**：白名单之外的键、类型错误、超长值
  一律视为违例，由调用方（Wiki 写路径）拒绝；
- 解析必须容错：正文以水平分隔线 ``---`` 开头是合法 Markdown，不能误判
  为 frontmatter。

本模块只做解析/校验/序列化，不碰文件系统与数据库。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import yaml

# frontmatter 白名单键（Schema 纪律：未知键一律违例）
ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "title",
        "summary",
        "tags",
        "version",
        "effective_date",
        "expiry_date",
        # LLM 自动沉淀层元数据：
        # kind=页面类型（doc 文档摘要 / entity 实体 / index 总目录 / log 日志）
        # sources=来源原始文档 source_id 列表；aliases=实体别名
        "kind",
        "sources",
        "aliases",
    }
)

# 自动沉淀页 kind 枚举
ALLOWED_KINDS: frozenset[str] = frozenset({"doc", "entity", "index", "log"})

# 各字段长度纪律
_TITLE_MAX = 200
_SUMMARY_MAX = 200
_VERSION_MAX = 20
_TAG_MAX = 30
_TAGS_MAX = 10
# sources/aliases 列表纪律（LLM 自动沉淀页）
_SOURCE_MAX = 300
_SOURCES_MAX = 30
_ALIAS_MAX = 100
_ALIASES_MAX = 20

# merge_frontmatter 时做并集去重的列表字段
_UNION_KEYS: frozenset[str] = frozenset({"sources", "aliases"})

# 开头扫描 frontmatter 闭合线的行数上限（防止超长伪 frontmatter）
_MAX_PROBE_LINES = 50

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def has_frontmatter_block(text: str) -> bool:
    """文本是否以合法包裹的 frontmatter 结构开头（不校验 YAML 内容）。

    即：首个非空行为 ``---``，且在开头 :data:`_MAX_PROBE_LINES` 行内
    存在闭合行 ``---``。用于写路径区分「无 frontmatter」与「结构存在
    但 YAML 无法解析」两种情况。
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines = normalized.split("\n")
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != "---":
        return False
    for probe in range(idx + 1, min(idx + 1 + _MAX_PROBE_LINES, len(lines))):
        if lines[probe].strip() == "---":
            return True
    return False


def parse_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """从容错解析 Markdown 文本，返回 ``(frontmatter | None, body)``。

    视为 frontmatter 的全部条件（任一不满足则整体视为正文）：
    - 文本首个非空行为 ``---``（允许 BOM/空白）；
    - 在开头 :data:`_MAX_PROBE_LINES` 行内存在闭合行 ``---``；
    - 两线之间的内容经 ``yaml.safe_load`` 解析为非空 dict。

    返回的 body 为剥离 frontmatter 块后的正文（保留其后所有内容）。
    """

    # 统一换行并剥离 BOM，避免 Windows 写入的 \r\n 干扰行匹配
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines = normalized.split("\n")
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != "---":
        return None, text

    open_line = idx
    close_line = None
    for probe in range(open_line + 1, min(open_line + 1 + _MAX_PROBE_LINES, len(lines))):
        if lines[probe].strip() == "---":
            close_line = probe
            break
    if close_line is None:
        return None, text

    block = "\n".join(lines[open_line + 1 : close_line])
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError:
        # YAML 语法错误：返回 None 让调用方区分处理（写路径可结合
        # has_frontmatter_block 判定报 400，读路径视为纯正文）
        return None, text

    if not isinstance(data, dict):
        return None, text

    body = "\n".join(lines[close_line + 1 :]).lstrip("\n")
    return data, body


def validate_frontmatter(meta: dict[str, Any]) -> list[str]:
    """校验 frontmatter 是否符合 Schema，返回违例描述列表（空列表 = 合法）。"""

    errors: list[str] = []
    unknown = sorted(set(meta) - ALLOWED_KEYS)
    if unknown:
        allowed = ", ".join(sorted(ALLOWED_KEYS))
        errors.append(f"未知字段: {', '.join(unknown)}（允许的字段: {allowed}）")

    title = meta.get("title")
    if title is not None:
        if not isinstance(title, str) or not title.strip():
            errors.append("title 必须是非空字符串")
        elif len(title.strip()) > _TITLE_MAX:
            errors.append(f"title 超过 {_TITLE_MAX} 字符上限")

    summary = meta.get("summary")
    if summary is not None:
        if not isinstance(summary, str):
            errors.append("summary 必须是字符串")
        elif len(summary.strip()) > _SUMMARY_MAX:
            errors.append(f"summary 超过 {_SUMMARY_MAX} 字符上限")

    tags = meta.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            errors.append("tags 必须是字符串列表")
        else:
            if len(tags) > _TAGS_MAX:
                errors.append(f"tags 超过 {_TAGS_MAX} 项上限")
            if any(not t.strip() for t in tags):
                errors.append("tags 中存在空白项")
            if any(len(t.strip()) > _TAG_MAX for t in tags):
                errors.append(f"tags 单项超过 {_TAG_MAX} 字符上限")

    version = meta.get("version")
    if version is not None:
        if not isinstance(version, str) or not version.strip():
            errors.append("version 必须是非空字符串")
        elif len(version.strip()) > _VERSION_MAX:
            errors.append(f"version 超过 {_VERSION_MAX} 字符上限")

    for key in ("effective_date", "expiry_date"):
        value = meta.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not _ISO_DATE_RE.match(value.strip()):
            errors.append(f"{key} 必须是 ISO 日期（YYYY-MM-DD）")
        else:
            try:
                date.fromisoformat(value.strip())
            except ValueError:
                errors.append(f"{key} 不是有效日期: {value!r}")

    kind = meta.get("kind")
    if kind is not None:
        if not isinstance(kind, str) or kind.strip() not in ALLOWED_KINDS:
            allowed = "/".join(sorted(ALLOWED_KINDS))
            errors.append(f"kind 必须是 {allowed} 之一")

    errors.extend(_validate_string_list(meta, "sources", _SOURCE_MAX, _SOURCES_MAX))
    errors.extend(_validate_string_list(meta, "aliases", _ALIAS_MAX, _ALIASES_MAX))

    return errors


def _validate_string_list(
    meta: dict[str, Any], key: str, item_max: int, list_max: int
) -> list[str]:
    """校验字符串列表字段：类型、非空项、单项长度、列表项数。"""

    value = meta.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
        return [f"{key} 必须是字符串列表"]
    errors: list[str] = []
    if len(value) > list_max:
        errors.append(f"{key} 超过 {list_max} 项上限")
    if any(not t.strip() for t in value):
        errors.append(f"{key} 中存在空白项")
    if any(len(t.strip()) > item_max for t in value):
        errors.append(f"{key} 单项超过 {item_max} 字符上限")
    return errors


def normalize_frontmatter(meta: dict[str, Any]) -> dict[str, Any]:
    """规整 frontmatter：字符串字段去首尾空白，日期规整为 ISO 日期。

    调用方必须先通过 :func:`validate_frontmatter` 校验；本函数不校验，
    只做写入前的清洁（保证镜像中的 frontmatter 无多余空白）。
    """

    clean: dict[str, Any] = {}
    for key in sorted(ALLOWED_KEYS & set(meta)):
        value = meta[key]
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        if isinstance(value, list):
            value = [t.strip() for t in value if isinstance(t, str) and t.strip()]
            if not value:
                continue
        clean[key] = value
    return clean


def dump_frontmatter(meta: dict[str, Any]) -> str:
    """把元数据序列化为 ``---\\n...\\n---\\n`` 块（含结尾空行）。

    ``allow_unicode`` 保证中文原样输出；``sort_keys`` 输出键序稳定，
    保证相同元数据物化结果幂等。
    """

    body = yaml.safe_dump(
        meta, allow_unicode=True, sort_keys=True, default_flow_style=False
    ).rstrip("\n")
    return f"---\n{body}\n---\n"


def merge_frontmatter(
    base: dict[str, Any] | None, override: dict[str, Any]
) -> dict[str, Any]:
    """合并元数据：override 中的非空值覆盖 base 同名字段。

    用于镜像物化时以页面 title 为准（override）覆盖正文自带 title（base）。
    sources/aliases 为列表并集语义（跨文档沉淀时保留全部来源/别名），
    按首次出现顺序去重；其余字段 override 非空即覆盖。
    """

    merged: dict[str, Any] = dict(base or {})
    for key, value in override.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if key in _UNION_KEYS and isinstance(merged.get(key), list) and isinstance(value, list):
            merged[key] = _union_strings(merged[key], value)
        else:
            merged[key] = value
    return {k: v for k, v in merged.items() if k in ALLOWED_KEYS}


def _union_strings(*lists: list[Any]) -> list[str]:
    """多个字符串列表按首次出现顺序去重并集（去首尾空白、丢弃空项）。"""

    seen: set[str] = set()
    union: list[str] = []
    for lst in lists:
        for item in lst:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                union.append(cleaned)
    return union
