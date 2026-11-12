"""自动沉淀层页面工程（纯函数）。

集中管理：
- 系统身份与保留 slug（index/log）、页面 slug 前缀（doc-/entity-）；
- 文档摘要页 / 实体页正文与 frontmatter 模板（来源强制标注）；
- "矛盾待裁决"区块的机器可识别 marker（lint 复用）；
- index 总目录渲染、log 追加格式。

本模块不碰文件系统与数据库，所有时间/ID 由参数注入，保证可测与幂等。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

from src.rag import knowledge_contract as kc
from src.rag.wiki_schema import validate_frontmatter

# ── 常量 ────────────────────────────────────────────────────────────

SYSTEM_AUTHOR = "system"
INDEX_SLUG = "index"
LOG_SLUG = "log"
DOC_PREFIX = "doc-"
ENTITY_PREFIX = "entity-"

# 实体类型：内部标识 → 中文展示
ENTITY_TYPES: dict[str, str] = {
    "person": "人物",
    "concept": "概念",
    "org": "机构",
    "tool": "工具",
}

# slug 中可读部分预算：80(总长) - 前缀 - "-" - 8(短哈希)
_SLUG_BUDGET = 80 - 1 - 8

# 矛盾区块 marker（lint 据 open/resolved 计数判断未裁决项）
CONTRADICTION_OPEN = "<!-- contradiction:open -->"
CONTRADICTION_CLOSE = "<!-- contradiction:close -->"
CONTRADICTION_RESOLVED = "<!-- contradiction:resolved -->"


# ── slug 与名称 ─────────────────────────────────────────────────────


def _ascii_slug(text: str) -> str:
    """拉丁字母/数字转小写连字符 slug；纯中文等非 ASCII 文本返回空串。"""

    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _prefixed_slug(prefix: str, readable_source: str, stem: str) -> str:
    """生成前缀 slug：可读部分优先，非 ASCII/超长时追加短哈希保证唯一稳定。"""

    readable = _ascii_slug(stem)
    budget = _SLUG_BUDGET - len(prefix)
    if not readable:
        slug = f"{prefix}{_short_hash(readable_source)}"
    elif len(readable) > budget:
        slug = f"{prefix}{readable[:budget].rstrip('-')}-{_short_hash(readable_source)}"
    else:
        slug = f"{prefix}{readable}"
    assert kc.is_valid_wiki_slug(slug), f"生成了非法 slug: {slug!r}"
    return slug


def doc_slug(source_id: str) -> str:
    """原始文档 source_id → 文档摘要页 slug（确定性，同文档重复沉淀稳定）。"""

    stem = PurePosixPath(source_id.replace("\\", "/")).stem
    return _prefixed_slug(DOC_PREFIX, source_id, stem)


def doc_slug_disambiguated(source_id: str) -> str:
    """跨目录同名文档碰撞时的消歧 slug：doc-<stem>-<hash8(完整 source_id)>。"""

    normalized = source_id.replace("\\", "/")
    stem = PurePosixPath(normalized).stem
    digest = _short_hash(normalized)
    readable = _ascii_slug(stem)
    # 前缀 + "-" + 8 位哈希的总预算受 slug 80 长度约束
    budget = _SLUG_BUDGET - len(DOC_PREFIX) - len(digest) - 1
    base = readable[:budget].rstrip("-") if readable else digest
    slug = f"{DOC_PREFIX}{base}-{digest}"
    assert kc.is_valid_wiki_slug(slug), f"生成了非法 slug: {slug!r}"
    return slug


def entity_slug(name: str) -> str:
    """实体名 → 实体页 slug（确定性；中文名走短哈希）。"""

    return _prefixed_slug(ENTITY_PREFIX, normalize_entity_name(name), name)


def normalize_entity_name(name: str) -> str:
    """归一化实体名用于"同一实体"匹配：小写、去空白与标点，保留中日韩字符。"""

    return re.sub(r"[\s\W_]+", "", name.lower(), flags=re.UNICODE)


def entity_aliases(name: str, aliases: Iterable[str]) -> list[str]:
    """去重清洗别名（含规范名），剔除空白与超长项由 Schema 校验兜底。"""

    seen: set[str] = set()
    result: list[str] = []
    for raw in [name, *aliases]:
        cleaned = (raw or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


# ── 页面模板 ─────────────────────────────────────────────────────────


def _source_lines(sources: Iterable[str]) -> str:
    return "\n".join(f"- `{s}`" for s in dict.fromkeys(sources) if s)


def build_doc_page(
    source_id: str,
    summary: str,
    key_points: Iterable[str],
    entities: Iterable[tuple[str, str]] = (),
) -> tuple[dict[str, Any], str]:
    """构造文档摘要页 (frontmatter, body)。

    :param entities: ``(实体名, slug)`` 可迭代表，渲染为相对链接。
    """

    summary = (summary or "").strip()
    points = [p.strip() for p in key_points if p and p.strip()]
    fm: dict[str, Any] = {"kind": "doc", "sources": [source_id]}
    if summary:
        fm["summary"] = summary[:200]

    sections = ["## 文档摘要", "", summary or "（LLM 未生成摘要）", ""]
    if points:
        sections += ["## 关键要点", ""]
        sections += [f"- {p}" for p in points]
        sections.append("")
    linked = [(n, s) for n, s in entities if n and s]
    if linked:
        sections += ["## 关联实体", ""]
        sections += [f"- [{n}]({s}.md)" for n, s in linked]
        sections.append("")
    sections += ["## 来源", "", _source_lines([source_id]), ""]
    return fm, "\n".join(sections)


def build_entity_page(
    name: str,
    entity_type: str,
    aliases: Iterable[str],
    summary: str,
    points: Iterable[str],
    sources: Iterable[str],
) -> tuple[dict[str, Any], str]:
    """构造实体页 (frontmatter, body)。类型标识写正文，不新增 schema 键。"""

    type_label = ENTITY_TYPES.get(entity_type, "概念")
    alias_list = [a for a in aliases if a and a.strip() and a.strip() != name]
    source_list = list(dict.fromkeys(s for s in sources if s))
    summary = (summary or "").strip()
    points = [p.strip() for p in points if p and p.strip()]

    fm: dict[str, Any] = {"kind": "entity", "sources": source_list[:30]}
    if alias_list:
        fm["aliases"] = alias_list[:20]

    head = f"> 类型：{type_label}"
    if alias_list:
        head += "　|　别名：" + "、".join(alias_list)
    sections = [head, ""]
    if summary:
        sections += ["## 概述", "", summary, ""]
    if points:
        sections += ["## 要点", ""]
        sections += [f"- {p}" for p in points]
        sections.append("")
    sections += ["## 来源", "", _source_lines(source_list), ""]
    return fm, "\n".join(sections)


def entity_type_from_body(body: str) -> str | None:
    """从实体页正文首行 ``> 类型：X`` 提取中文类型（index 分组用）。"""

    m = re.search(r"^>\s*类型：([^\s|｜]+)", body.strip(), flags=re.MULTILINE)
    return m.group(1).strip() if m else None


# ── 矛盾区块 ─────────────────────────────────────────────────────────


def wrap_contradiction(
    today_iso: str, new_claim: str, new_source: str, old_claim: str, old_source: str
) -> str:
    """生成标准"矛盾待裁决"区块（含成对 marker 与双方说法/来源）。"""

    return (
        f"{CONTRADICTION_OPEN}\n"
        f"## ⚠️ 矛盾待裁决（{today_iso}）\n\n"
        f"- 新说法（来源：`{new_source}`）：{new_claim.strip()}\n"
        f"- 既有说法（来源：`{old_source}`）：{old_claim.strip()}\n"
        f"{CONTRADICTION_CLOSE}\n"
    )


def append_contradiction(body: str, block: str) -> str:
    """在页面正文末尾追加矛盾区块；完全相同的区块不重复追加（幂等）。"""

    base = body.rstrip()
    if block.strip() in base:
        return body
    return f"{base}\n\n{block.rstrip()}\n"


_OPEN_BLOCK_RE = re.compile(
    re.escape(CONTRADICTION_OPEN) + r".*?" + re.escape(CONTRADICTION_CLOSE),
    re.DOTALL,
)


def open_contradiction_blocks(body: str) -> list[str]:
    """提取正文中全部"未裁决"矛盾区块（open…close 且段内无 resolved 标记）。"""

    return [
        m.group(0).strip()
        for m in _OPEN_BLOCK_RE.finditer(body)
        if CONTRADICTION_RESOLVED not in m.group(0)
    ]


def carry_open_contradictions(new_body: str, old_body: str) -> str:
    """再沉淀整体重建正文时，把旧页未裁决区块迁移到新正文。

    安全不变量：未经人工裁决（出现 resolved 标记）的矛盾区块不得因
    再次沉淀而消失；已存在的相同区块不重复追加。
    """

    body = new_body.rstrip()
    for block in open_contradiction_blocks(old_body):
        if block not in body:
            body = f"{body}\n\n{block}"
    return body + "\n"


def has_open_contradiction(text: str) -> bool:
    """是否存在未裁决矛盾：open 标记数多于 resolved 标记数。"""

    return text.count(CONTRADICTION_OPEN) > text.count(CONTRADICTION_RESOLVED)


# ── index / log ──────────────────────────────────────────────────────


def render_index(entries: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """渲染总目录页。entries 项含 slug/title/kind，实体页可带 entity_type。

    分区：文档摘要页 → 实体页（按类型分组）→ 其他页面；index/log 排除。
    """

    docs: list[dict[str, Any]] = []
    entities: dict[str, list[dict[str, Any]]] = {label: [] for label in ENTITY_TYPES.values()}
    others: list[dict[str, Any]] = []
    for item in entries:
        slug = item["slug"]
        if slug in {INDEX_SLUG, LOG_SLUG}:
            continue
        if item.get("kind") == "doc":
            docs.append(item)
        elif item.get("kind") == "entity":
            label = item.get("entity_type") or "概念"
            entities.setdefault(label, []).append(item)
        else:
            others.append(item)

    def _line(item: dict[str, Any]) -> str:
        return f"- [{item['title']}]({item['slug']}.md)"

    sections = ["## 文档摘要", ""]
    sections += [_line(i) for i in sorted(docs, key=lambda x: x["title"])] or ["_（暂无）_"]
    sections.append("")
    sections += ["## 实体页", ""]
    for label in ENTITY_TYPES.values():
        items = sorted(entities.get(label, []), key=lambda x: x["title"])
        if not items:
            continue
        sections += [f"### {label}", ""]
        sections += [_line(i) for i in items]
        sections.append("")
    sections += ["## 其他", ""]
    sections += [_line(i) for i in sorted(others, key=lambda x: x["title"])] or ["_（暂无）_"]
    sections.append("")
    return {"kind": "index"}, "\n".join(sections)


def log_page_initial() -> tuple[dict[str, Any], str]:
    """log 页初始正文。"""

    return {"kind": "log"}, "## 沉淀记录\n"


def log_entry(
    timestamp: str, source_id: str, ok: bool, stats: dict[str, Any] | None = None,
    reason: str = "",
) -> str:
    """生成一条 append-only 日志行。"""

    stats = stats or {}
    if ok:
        detail = (
            f"新建 {stats.get('created', 0)} 页 / 更新 {stats.get('updated', 0)} 页 / "
            f"实体 {stats.get('entities', 0)} / 矛盾 {stats.get('contradictions', 0)}"
        )
        return f"- {timestamp} `{source_id}` 沉淀成功：{detail}"
    return f"- {timestamp} `{source_id}` 沉淀失败：{reason or '未知原因'}"


def append_log(body: str, entry: str) -> str:
    """append-only 追加日志行，保留历史正文。"""

    base = body.rstrip()
    if not base:
        base = "## 沉淀记录"
    return f"{base}\n{entry}\n"


# ── 校验兜底 ─────────────────────────────────────────────────────────


def assert_valid_frontmatter(fm: dict[str, Any]) -> None:
    """系统生成的 frontmatter 必须合法；违例即编程错误，直接抛出。"""

    errors = validate_frontmatter(fm)
    if errors:
        raise ValueError("自动沉淀页 frontmatter 不合法: " + "；".join(errors))
