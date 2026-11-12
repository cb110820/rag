"""pytest 公共夹具：每个用例使用独立临时数据库与固定 JWT 密钥。

- 不触碰开发库 ``.data/app.db``；
- 测试库嵌入固定为 hash（零下载、确定性），语义记忆索引走 sqlite-vec；
- ``get_settings()`` 在所有通过 ``from ... import get_settings`` 持有的
  模块中均被替换；
- 每个用例后清空登录限流计数，避免用例间相互污染。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from src.api import auth
from src.memory import embedding as memory_embedding
from src.memory import episodic as memory_episodic
from src.memory import reflection as memory_reflection
from src.memory import tools as memory_tools
from src.persistence import database as db
from src.rag import config as config_module
from src.rag.config import Settings

_TEST_JWT_SECRET = "test-secret-key-please-do-not-use-in-production-0123456789"


def make_test_settings(tmp_path: Path) -> Settings:
    """构造测试用 Settings：hash 嵌入 + 临时数据目录。"""

    return Settings(
        data_dir=tmp_path / "data",
        jwt_secret_key=_TEST_JWT_SECRET,
        allow_registration=True,
        rag_embedding="hash",
        # 测试中默认不启后台清扫协程，需要 TTL 清扫的用例手动调 sweep_ttl
        ttl_sweep_interval_minutes=0,
        # 测试中默认关闭 LangMem 后台反思，避免误触真实 LLM；反思用例显式打开
        memory_reflection_enabled=False,
        # 测试中默认关闭上传后自动沉淀，避免误触真实 LLM；
        # 沉淀调度/服务用例自行打开并注入 fake 提取器
        wiki_curation_enabled=False,
    )


@pytest_asyncio.fixture
async def isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Settings]:
    """初始化临时 SQLite（含 saver/store），用例结束后关闭，返回测试 Settings。"""

    test_settings = make_test_settings(tmp_path)
    # 各模块通过 ``from ... import get_settings`` 拿到的是模块级名字，
    # 必须逐一替换
    monkeypatch.setattr(config_module, "get_settings", lambda: test_settings)
    monkeypatch.setattr(db, "get_settings", lambda: test_settings)
    monkeypatch.setattr(auth, "get_settings", lambda: test_settings)
    monkeypatch.setattr(memory_tools, "get_settings", lambda: test_settings)
    monkeypatch.setattr(memory_embedding, "get_settings", lambda: test_settings)
    monkeypatch.setattr(memory_reflection, "get_settings", lambda: test_settings)
    monkeypatch.setattr(memory_episodic, "get_settings", lambda: test_settings)
    # curation 子系统在模块内持有 get_settings 名字，逐一替换
    from src.curation import llm as curation_llm
    from src.curation import scheduler as curation_scheduler
    from src.curation import service as curation_service

    monkeypatch.setattr(curation_llm, "get_settings", lambda: test_settings)
    monkeypatch.setattr(curation_scheduler, "get_settings", lambda: test_settings)
    monkeypatch.setattr(curation_service, "get_settings", lambda: test_settings)

    auth._login_failures.clear()

    await db.init_db()
    try:
        yield test_settings
    finally:
        await db.close_db()
        auth._login_failures.clear()
