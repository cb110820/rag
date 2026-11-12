"""通用后台调用 worker：把阻塞 API 调用丢到 QThread，结果经 Signal 回主线程。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal

from desktop.api.errors import ApiError


class CallWorker(QThread):
    """执行 fn() 并发出 result(Any) / error(detail, status_code)。"""

    result = Signal(object)
    error = Signal(str, int)  # (可读描述, HTTP 状态码；网络错误为 0，未知异常为 -1)

    def __init__(self, fn: Callable[[], Any], parent: object = None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.result.emit(self._fn())
        except ApiError as exc:
            self.error.emit(exc.detail, exc.status_code)
        except Exception as exc:  # 兜底：任何未预期异常都不让线程静默死亡
            self.error.emit(f"操作失败：{exc}", -1)


# 活跃 worker 强引用注册表：QThread 包装对象若被 Python GC 提前回收
# （C++ 线程仍在运行），Qt 直接 abort（0xC0000409 fail-fast）。
# 这里统一管理生命周期，调用方无需自行持有引用。
_ACTIVE: set[CallWorker] = set()
_ACTIVE_LOCK = threading.Lock()


def start_worker(
    fn: Callable[[], Any],
    on_result: Callable[[Any], None],
    on_error: Callable[[str, int], None] | None = None,
) -> CallWorker:
    """便捷入口：创建、接线、启动并托管 worker 生命周期（finished 后自动释放）。"""

    worker = CallWorker(fn)
    worker.result.connect(on_result)
    if on_error is not None:
        worker.error.connect(on_error)
    with _ACTIVE_LOCK:
        _ACTIVE.add(worker)

    def _cleanup() -> None:
        with _ACTIVE_LOCK:
            _ACTIVE.discard(worker)
        worker.deleteLater()

    worker.finished.connect(_cleanup)
    worker.start()
    return worker


def shutdown_all(timeout_ms: int = 3000) -> None:
    """窗口关闭前调用：等待全部活跃 worker 结束；超时仍在运行的强制终止。

    terminate 是最后手段（线程可能在 I/O 中硬停），但优于进程退出时
    QThread 存活导致的 fail-fast（0xC0000409）崩溃。
    """

    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            running = [w for w in _ACTIVE if w.isRunning()]
        if not running:
            break
        running[0].wait(100)
    with _ACTIVE_LOCK:
        stragglers = [w for w in _ACTIVE if w.isRunning()]
    for worker in stragglers:
        worker.terminate()
        worker.wait(1000)
