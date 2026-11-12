"""查询理解（QUERY_UNDERSTAND）测试：门控、改写、超时/失败降级、streaming=False。"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from src.rag import tools as rag_tools

pytestmark = pytest.mark.usefixtures("isolated_db")

QUERY = "退款流程是什么"


def _settings(**overrides):
    base = dict(
        rag_timeout_seconds=10.0,
        rag_max_retries=2,
        query_understand_enabled=False,
        query_understand_timeout_seconds=5.0,
        query_understand_model="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeEnhancer:
    def __init__(self, *, rewritten="改写后的检索查询", fail=False, slow=0.0) -> None:
        self.rewritten = rewritten
        self.fail = fail
        self.slow = slow
        self.calls: list[str] = []

    def rewrite(self, query: str) -> str:
        self.calls.append("rewrite")
        if self.fail:
            raise RuntimeError("LLM 挂了")
        if self.slow:
            time.sleep(self.slow)
        return self.rewritten

    def generate_hypothetical(self, query: str) -> str:
        self.calls.append("hyde")
        return "假想答案段落"

    def expand_query(self, query: str) -> str:
        self.calls.append("expand")
        return "退款 流程 政策"


@pytest.fixture
def search_stub(monkeypatch: pytest.MonkeyPatch):
    """替换租户 KB 为记录型 stub，捕获 kb.search 入参。"""

    captured: dict = {}

    class _Stub:
        def search(self, query, top_k, *, dense_query=None, sparse_query=None,
                   enhanced_queries=None):
            captured.update(
                query=query, top_k=top_k, dense=dense_query,
                sparse=sparse_query, enhanced=list(enhanced_queries or []),
            )
            return []

    monkeypatch.setattr(rag_tools, "get_knowledge_base", lambda tenant_id: _Stub())
    return captured


@pytest.fixture
def enhancer_spy(monkeypatch: pytest.MonkeyPatch):
    holder: dict = {}

    def install(enhancer) -> _FakeEnhancer:
        holder["obj"] = enhancer
        monkeypatch.setattr(rag_tools, "_enhancer", enhancer)
        monkeypatch.setattr(rag_tools, "_get_enhancer", lambda: enhancer)
        return enhancer

    holder["install"] = install
    return holder


async def test_enhance_disabled_skips_llm(
    monkeypatch: pytest.MonkeyPatch, search_stub, enhancer_spy
) -> None:
    """开关关闭（默认）时不做任何 LLM 调用，检索用原始查询。"""

    monkeypatch.setattr(rag_tools, "get_settings",
                        lambda: _settings(query_understand_enabled=False))
    enh = enhancer_spy["install"](_FakeEnhancer())

    await rag_tools._search_with_timeout("t1", QUERY, 6)

    assert enh.calls == []  # 零 LLM 调用
    assert search_stub["query"] == QUERY
    assert search_stub["dense"] is None and search_stub["sparse"] is None
    assert search_stub["enhanced"] == []


async def test_rewrite_used_as_main_query(
    monkeypatch: pytest.MonkeyPatch, search_stub, enhancer_spy
) -> None:
    """开关开启：改写查询作为主查询，hyde/扩展/原查询进 CRAG 备选。"""

    monkeypatch.setattr(rag_tools, "get_settings",
                        lambda: _settings(query_understand_enabled=True))
    enhancer_spy["install"](_FakeEnhancer())

    await rag_tools._search_with_timeout("t1", QUERY, 6)

    assert search_stub["query"] == "改写后的检索查询"
    assert search_stub["dense"] == "假想答案段落"
    assert search_stub["sparse"] == "退款 流程 政策"
    # 备选不含主查询本身；原查询应作为备选保底
    assert "改写后的检索查询" not in search_stub["enhanced"]
    assert QUERY in search_stub["enhanced"]
    assert len(search_stub["enhanced"]) <= 2


async def test_rewrite_timeout_degrades_to_raw_query(
    monkeypatch: pytest.MonkeyPatch, search_stub, enhancer_spy
) -> None:
    """改写超时：整体降级为原始查询，检索照常执行。"""

    monkeypatch.setattr(rag_tools, "get_settings", lambda: _settings(
        query_understand_enabled=True, query_understand_timeout_seconds=0.05,
    ))
    enhancer_spy["install"](_FakeEnhancer(slow=0.4))

    await rag_tools._search_with_timeout("t1", QUERY, 6)

    assert search_stub["query"] == QUERY
    assert search_stub["dense"] is None and search_stub["sparse"] is None


async def test_rewrite_failure_returns_original(
    monkeypatch: pytest.MonkeyPatch, search_stub, enhancer_spy
) -> None:
    """LLM 异常：降级为原始查询，不向调用方抛错。"""

    monkeypatch.setattr(rag_tools, "get_settings",
                        lambda: _settings(query_understand_enabled=True))
    enhancer_spy["install"](_FakeEnhancer(fail=True))

    await rag_tools._search_with_timeout("t1", QUERY, 6)

    assert search_stub["query"] == QUERY


async def test_model_resolved_streaming_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """查询理解模型必须以 streaming=False 解析（ChatTongyi 红线）。"""

    captured: dict = {}
    fake_model = object()

    def fake_resolve(spec=None, streaming=True):
        captured.update(spec=spec, streaming=streaming)
        return fake_model

    import src.agents.graph as graph

    monkeypatch.setattr(graph, "resolve_chat_model", fake_resolve)
    monkeypatch.setattr(rag_tools, "get_settings",
                        lambda: _settings(query_understand_model=""))

    model = rag_tools.QueryEnhancer._resolve_model()

    assert model is fake_model
    assert captured["streaming"] is False
