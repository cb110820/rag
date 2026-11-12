"""SSE 行解析纯函数测试。"""

from desktop.api.sse import parse_sse_lines


def test_parse_single_event() -> None:
    lines = ["event: token", 'data: {"text": "你好"}', ""]
    assert list(parse_sse_lines(lines)) == [("token", {"text": "你好"})]


def test_parse_multiple_events_in_order() -> None:
    lines = [
        "event: token",
        'data: {"text": "a"}',
        "",
        "event: sources",
        'data: {"sources": [{"source": "s1", "section": "", "score": 0.9}]}',
        "",
        "event: done",
        "data: {}",
        "",
    ]
    events = list(parse_sse_lines(lines))
    assert [e for e, _ in events] == ["token", "sources", "done"]
    assert events[1][1]["sources"][0]["source"] == "s1"


def test_parse_interrupt_event() -> None:
    payload = (
        '{"interrupts": [{"id": "i1", "tool": "forget_memory",'
        ' "args": {"memory_id": "fact-x"}}]}'
    )
    lines = ["event: interrupt", f"data: {payload}", ""]
    events = list(parse_sse_lines(lines))
    assert events == [
        ("interrupt", {"interrupts": [{"id": "i1", "tool": "forget_memory",
                                       "args": {"memory_id": "fact-x"}}]})
    ]


def test_multiline_data_and_comment_and_crlf() -> None:
    lines = [": heartbeat", "event: token", 'data: {"text":', 'data:  "拼接"}', "", ""]
    events = list(parse_sse_lines(line + "\r" for line in lines))
    assert events == [("token", {"text": "拼接"})]


def test_trailing_event_without_blank_line() -> None:
    lines = ["event: done", "data: {}"]
    assert list(parse_sse_lines(lines)) == [("done", {})]
