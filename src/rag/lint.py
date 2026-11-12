"""知识健康检查（知识治理 · Lint 保证健康）。

对租户知识库的 Wiki 镜像区做一致性/纪律扫描，产出结构化 issue 列表：

- ``mirror_orphan``：镜像存在但业务库无对应页面（error，可经重建镜像清理）；
- ``mirror_missing``：业务库页面存在但镜像缺失（error，可经重建镜像修复）；
- ``frontmatter_missing``：镜像缺少 frontmatter（warning，Schema 纪律，
  可经重建镜像自动补全）；
- ``frontmatter_invalid``：frontmatter 违反 Schema 约束（error）；
- ``content_expired`` / ``content_future``：文档已过期 / 尚未生效；
- ``content_empty``：镜像正文为空（warning）；
- ``contradiction_open``：存在未裁决的"矛盾待裁决"区块（warning，
  由 LLM 自动沉淀层写入，人工裁决后关闭标记）。

纯函数模块：只读文件系统与入参，不碰数据库；业务库 slug 集合由调用方
（API 端点）传入并负责鉴权。深度检查仅覆盖 ``wiki/`` 镜像区——原始文档
不可变且不受 Schema 纪律约束，无需逐次扫描。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TypedDict

from src.rag import knowledge_contract as kc
from src.rag.wiki_schema import parse_frontmatter, validate_frontmatter

logger = logging.getLogger(__name__)

# 矛盾区块 marker（与 src.curation.markdown 的常量保持字节一致）
_CONTRADICTION_OPEN = "<!-- contradiction:open -->"
_CONTRADICTION_RESOLVED = "<!-- contradiction:resolved -->"


class LintIssue(TypedDict):
    """单条健康问题：severity / code / source_id / message。"""

    severity: str  # "error" | "warning" | "info"
    code: str
    source_id: str
    message: str


@dataclass
class LintResult:
    """一次 lint 的产出：问题列表 + 实际检查的镜像文件数。"""

    issues: list[LintIssue] = field(default_factory=list)
    checked_files: int = 0


def lint_tenant_knowledge(
    tenant_root: Path, wiki_slugs: set[str], today: date | None = None
) -> LintResult:
    """扫描租户 ``wiki/`` 镜像区，返回健康问题列表。

    :param tenant_root: 租户知识库根目录 ``KNOWLEDGE_DIR/<tenant_id>/``；
    :param wiki_slugs: 业务库当前存在的 Wiki 页面 slug 集合；
    :param today: 时效性判断基准日，缺省取当天。
    """

    today = today or date.today()
    result = LintResult()
    mirror_dir = tenant_root / kc.WIKI_DIR_NAME
    found_slugs: set[str] = set()

    if mirror_dir.is_dir():
        for path in sorted(mirror_dir.glob("*.md")):
            source_id = f"{kc.WIKI_DIR_NAME}/{path.name}"
            slug = kc.slug_from_wiki_source(source_id)
            if slug is None:
                result.issues.append(
                    LintIssue(
                        severity="warning",
                        code="mirror_naming",
                        source_id=source_id,
                        message="文件名不符合 Wiki slug 命名契约，不受镜像托管",
                    )
                )
                continue
            found_slugs.add(slug)
            if slug not in wiki_slugs:
                result.issues.append(
                    LintIssue(
                        severity="error",
                        code="mirror_orphan",
                        source_id=source_id,
                        message=f"镜像存在但业务库无对应页面（slug={slug}），可经重建镜像清理",
                    )
                )
                continue
            _lint_markdown_file(path, source_id, result, today)

    for slug in sorted(wiki_slugs - found_slugs):
        result.issues.append(
            LintIssue(
                severity="error",
                code="mirror_missing",
                source_id=kc.wiki_source_id(slug),
                message=f"业务库页面存在但镜像缺失（slug={slug}），可经重建镜像修复",
            )
        )

    result.checked_files = len(found_slugs)
    return result


def _lint_markdown_file(
    path: Path, source_id: str, result: LintResult, today: date
) -> None:
    """对单个镜像文件做 Schema/时效/正文检查。"""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        result.issues.append(
            LintIssue(
                severity="error",
                code="read_failed",
                source_id=source_id,
                message=f"镜像文件无法读取: {exc}",
            )
        )
        return

    # 矛盾标记检测（marker 契约由 src.curation.markdown 定义；此处本地比对，
    # 避免 rag 反向依赖 curation 包）
    if text.count(_CONTRADICTION_OPEN) > text.count(_CONTRADICTION_RESOLVED):
        result.issues.append(
            LintIssue(
                severity="warning",
                code="contradiction_open",
                source_id=source_id,
                message="存在未裁决的知识矛盾（contradiction:open），请人工核对后关闭标记",
            )
        )

    fm, body = parse_frontmatter(text)
    if fm is None:
        result.issues.append(
            LintIssue(
                severity="warning",
                code="frontmatter_missing",
                source_id=source_id,
                message="镜像缺少 frontmatter（Schema 纪律），可经重建镜像自动补全",
            )
        )
        return

    errors = validate_frontmatter(fm)
    if errors:
        result.issues.append(
            LintIssue(
                severity="error",
                code="frontmatter_invalid",
                source_id=source_id,
                message="frontmatter 违反 Schema 约束: " + "；".join(errors),
            )
        )

    today_iso = today.isoformat()
    expiry = fm.get("expiry_date")
    if isinstance(expiry, str) and _is_iso_date(expiry) and expiry < today_iso:
        result.issues.append(
            LintIssue(
                severity="warning",
                code="content_expired",
                source_id=source_id,
                message=f"文档已过期（expiry_date={expiry}），检索时会被降权",
            )
        )
    effective = fm.get("effective_date")
    if isinstance(effective, str) and _is_iso_date(effective) and effective > today_iso:
        result.issues.append(
            LintIssue(
                severity="info",
                code="content_future",
                source_id=source_id,
                message=f"文档尚未生效（effective_date={effective}）",
            )
        )

    if not body.strip():
        result.issues.append(
            LintIssue(
                severity="warning",
                code="content_empty",
                source_id=source_id,
                message="镜像正文为空（仅剩 frontmatter）",
            )
        )


def _is_iso_date(value: str) -> bool:
    """字符串是否为可比较的 ISO 日期（YYYY-MM-DD）。"""

    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True
