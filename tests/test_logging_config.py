"""日志容错测试：Windows 文件占用切割不中断 + 双格式分发。"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from src.rag import config


def test_safe_handler_swallows_permission_error_on_rollover(
    tmp_path: Path, monkeypatch
) -> None:
    """app.log 被其他进程占用导致 rename 失败时，跳过切割且日志仍可写。"""

    log_file = tmp_path / "app.log"
    handler = config.SafeTimedRotatingFileHandler(
        log_file, when="midnight", backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    def _locked_rollover(self) -> None:  # 模拟 Windows WinError 32
        raise PermissionError(
            "[WinError 32] 另一个程序正在使用此文件，进程无法访问"
        )

    monkeypatch.setattr(TimedRotatingFileHandler, "doRollover", _locked_rollover)

    handler.doRollover()  # 不应抛异常

    # 切割失败后 handler 必须仍能正常写日志（流已重开）
    record = logging.LogRecord(
        "src", logging.INFO, __file__, 1, "切割后的日志", None, None
    )
    handler.emit(record)
    handler.close()
    assert "切割后的日志" in log_file.read_text(encoding="utf-8")


def test_dual_formatter_dispatches_by_logger_name() -> None:
    """业务记录走标准格式，uvicorn.access 记录走访问格式。"""

    formatter = config._DualFormatter(
        config._LOG_FORMAT,
        config._ACCESS_LOG_FORMAT,
        config._LOG_DATE_FORMAT,
        config._UVICORN_ACCESS_LOGGER_NAME,
    )

    business = logging.LogRecord(
        "src.api.app", logging.INFO, __file__, 1, "业务消息", None, None
    )
    out = formatter.format(business)
    assert "src.api.app" in out and "业务消息" in out

    # uvicorn 访问记录：args 为固定 5 元组
    access = logging.LogRecord(
        config._UVICORN_ACCESS_LOGGER_NAME,
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1234", "GET", "/health", "1.1", 200),
        None,
    )
    out_access = formatter.format(access)
    assert "uvicorn.access" in out_access
    assert '127.0.0.1:1234 - "GET /health HTTP/1.1" 200' in out_access
