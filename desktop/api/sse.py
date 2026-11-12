"""SSE（Server-Sent Events）行解析：纯函数，便于单测。

服务端格式（src/api/app.py 的 ``_sse``）：

    event: <name>\\ndata: <json>\\n\\n

兼容多行 data、CRLF 与流末尾无空行收尾的情况。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any


def parse_sse_lines(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """把 SSE 行流解析成 ``(事件名, data JSON)`` 序列。"""

    event = "message"
    data_parts: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r")
        if not line:  # 空行：分发已累积的事件
            if data_parts:
                yield event, json.loads("\n".join(data_parts))
            event, data_parts = "message", []
            continue
        if line.startswith(":"):  # 注释/心跳行
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
    # 流末尾没有空行收尾时，分发残留事件
    if data_parts:
        yield event, json.loads("\n".join(data_parts))
