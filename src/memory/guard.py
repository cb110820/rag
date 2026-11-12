"""长期记忆写入护栏（prompt 约束之外的强制安全层）。

模型通过 write_file/edit_file 写 ``/memory/`` 时，仅靠 system prompt 禁止
保存凭据并不可靠；本模块在后端写入路径上做强制校验：

- ``scan_secret``：识别常见凭据形态（私钥、云厂商密钥、JWT、明文凭据赋值），
  命中即拒绝，且错误信息不回显原文；
- ``ensure_write_allowed``：统一校验凭据、单文件字节上限、每用户条目数上限。

策略违规抛 :class:`MemoryRejectedError`，由后端包装层转成工具错误返回给模型，
不打断整轮对话。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("src.memory")


class MemoryRejectedError(ValueError):
    """记忆内容触发安全/容量策略，拒绝写入。"""


# (规则名, 编译正则)。规则宁可保守——误拦时模型可改写措辞后重试，
# 但漏掉凭据会直接进入跨会话持久化存储。
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PEM 私钥", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("AWS Access Key ID", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub/GitLab Token", re.compile(r"\b(?:gh[pousr]_|glpat-)[A-Za-z0-9_-]{20,}\b")),
    ("Google API Key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack Token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("OpenAI 风格密钥", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("JWT 令牌", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    # 明文凭据赋值：password/secret/token/api_key = "value"（值至少 8 位）
    (
        "明文凭据赋值",
        re.compile(
            r"(?i)(?:password|passwd|secret|token|api[_-]?key)\s*[:=]\s*"
            r"[\"']?[^\s\"']{8,}"
        ),
    ),
)


def scan_secret(content: str) -> str | None:
    """扫描内容中的凭据特征。

    Returns:
        命中的规则名称；未命中返回 None。
    """

    for rule_name, pattern in _SECRET_PATTERNS:
        if pattern.search(content):
            return rule_name
    return None


def ensure_write_allowed(
    content: str,
    *,
    max_file_bytes: int,
    new_file: bool,
    item_count: int,
    max_items: int,
) -> None:
    """写入前统一策略校验，违规则抛 :class:`MemoryRejectedError`。

    Args:
        content: 即将落盘的完整文件内容。
        max_file_bytes: 单文件字节上限。
        new_file: 是否为新建文件（编辑已有文件不做条目数校验）。
        item_count: 该用户 namespace 下当前条目数。
        max_items: 每用户条目数上限。
    """

    rule = scan_secret(content)
    if rule is not None:
        logger.warning(
            "记忆写入被安全策略拦截: 规则=%s, 内容长度=%d 字符", rule, len(content)
        )
        raise MemoryRejectedError(
            f"记忆内容疑似包含凭据或密钥（命中规则：{rule}），已拒绝写入。"
            "记忆中禁止保存任何密码、密钥、令牌，请删除该内容后重试。"
        )

    size = len(content.encode("utf-8"))
    if size > max_file_bytes:
        logger.warning("记忆写入超过单文件上限: %d > %d bytes", size, max_file_bytes)
        raise MemoryRejectedError(
            f"记忆文件大小为 {size} 字节，超过上限 {max_file_bytes} 字节，"
            "请先精简或删除过时内容后再写入。"
        )

    if new_file and item_count >= max_items:
        logger.warning(
            "记忆写入超过条目数上限: 当前=%d, 上限=%d", item_count, max_items
        )
        raise MemoryRejectedError(
            f"记忆条目数已达上限 {max_items}，请先合并或删除旧记忆后再新建。"
        )
