"""start_worker 生命周期回归：worker 未被 Python GC 提前回收（0xC0000409 根因修复）。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import time

import pytest
from PySide6.QtCore import QElapsedTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from desktop.workers.base import _ACTIVE, start_worker


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _wait_for(cond, timeout_ms: int = 5000) -> bool:
    """轮询等待条件成立（跨线程 queued 信号投递有延迟），返回是否等到。"""

    timer = QElapsedTimer()
    timer.start()
    while not cond() and timer.elapsed() < timeout_ms:
        QTest.qWait(50)
    return cond()


def test_start_worker_holds_reference_until_finished(qapp) -> None:
    done: list[int] = []

    def slow_fn() -> int:
        time.sleep(0.3)
        return 42

    # 返回值刻意丢弃（调用方不持有引用），随后强制 GC：
    # 修复前 QThread 包装被回收 → Qt abort；修复后注册表持有强引用
    start_worker(slow_fn, done.append)
    gc.collect()
    assert len(_ACTIVE) >= 1

    assert _wait_for(lambda: done == [42])
    # finished 后注册表自动释放，不泄漏
    assert _wait_for(lambda: len(_ACTIVE) == 0)


def test_start_worker_error_path_also_released(qapp) -> None:
    errors: list[tuple[str, int]] = []

    def failing() -> None:
        raise RuntimeError("boom")

    start_worker(failing, lambda _r: None, lambda d, s: errors.append((d, s)))
    gc.collect()
    assert _wait_for(lambda: errors == [("操作失败：boom", -1)])
    assert _wait_for(lambda: len(_ACTIVE) == 0)
