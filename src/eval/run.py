"""离线评估脚本。

用 golden dataset（标注好的问题 + 期望 source_ids）评估检索质量，
输出 Recall@K、MRR@K 等指标；可选集成 Ragas 做更丰富的 LLM-as-Judge 评估。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from pathlib import Path

from src.rag.config import configure_logging, get_settings
from src.rag.pipeline import KnowledgeBase

logger = logging.getLogger(__name__)


def load_golden(path: Path) -> list[dict[str, object]]:
    """加载黄金评估集（JSON 格式）。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_ids(source_id: str) -> set[str]:
    """来源 id 的可比形式集合。

    多租户根目录下 source_id 形如 ``<tenant>/wiki/x.md``，而 golden
    dataset 用相对 id（``policies.md`` / ``wiki/refund-policy.md``）。
    比对时同时给出剥离租户前缀的形式，避免多租户布局导致的假阴性。
    """

    if "/" in source_id:
        return {source_id, source_id.split("/", 1)[1]}
    return {source_id}


def retrieval_metrics(
    kb: KnowledgeBase, samples: list[dict[str, object]], top_k: int
) -> dict[str, float]:
    """计算检索指标：Recall@K 和 MRR@K。

    Recall@K = 期望来源在检索结果中的占比
    MRR@K = 第一个相关结果排名的倒数的平均值
    来源 id 比对经 :func:`_candidate_ids` 归一化（剥离租户前缀）。
    """

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for sample in samples:
        expected = set(sample["source_ids"])
        results = kb.search(str(sample["question"]), top_k=top_k)
        sources = [result.chunk.source_id for result in results]
        matched = {sid for source in sources for sid in _candidate_ids(source)}
        recalls.append(len(expected.intersection(matched)) / len(expected))
        reciprocal_ranks.append(
            next(
                (
                    1 / rank
                    for rank, source in enumerate(sources, 1)
                    if _candidate_ids(source) & expected
                ),
                0,
            )
        )
    return {
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr_at_k": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def _install_ragas_langchain_compat() -> None:
    """为 Ragas 0.3.x 打补丁：补一个缺失的 langchain_community.vertexai 模块。

    Ragas 在 isinstance 检查时引用了 ChatVertexAI，但项目并没有配置 VertexAI。
    注入一个空类即可让 Ragas 正常运行，不必额外安装 VertexAI 相关包。
    """

    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        module = types.ModuleType(module_name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = module


def run_ragas(samples: list[dict[str, object]], kb: KnowledgeBase, top_k: int) -> None:
    """用 Ragas 做 LLM-as-Judge 评估（可选）。"""

    _install_ragas_langchain_compat()
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import ContextRelevance
    except ImportError as exc:
        raise RuntimeError("Install the eval extra: uv sync --extra eval") from exc
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=str(item["question"]),
                retrieved_contexts=[
                    result.chunk.text for result in kb.search(str(item["question"]), top_k)
                ],
                response=str(item["reference"]),
                reference=str(item["reference"]),
            )
            for item in samples
        ]
    )
    print(evaluate(dataset=dataset, metrics=[ContextRelevance()]))


def main() -> int:
    """命令行入口：加载评估集 → 构建 KB → 计算指标 → 可选跑 Ragas → 返回退出码。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden", type=Path, default=Path(__file__).with_name("golden_dataset.json")
    )
    parser.add_argument("--embedding", choices=("bge-m3", "hash"), default="hash")
    parser.add_argument("--reranker", choices=("bge-reranker-v2-m3", "none"), default="none")
    parser.add_argument("--with-ragas", action="store_true")
    args = parser.parse_args()
    configure_logging()
    settings = get_settings()
    samples = load_golden(args.golden)
    logger.info(
        "开始离线评估：%d 个样本，embedding=%s，reranker=%s",
        len(samples),
        args.embedding,
        args.reranker,
    )
    kb = KnowledgeBase(settings.knowledge_dir, embedding=args.embedding, reranker=args.reranker)
    metrics = retrieval_metrics(kb, samples, settings.rag_top_k)
    logger.info("评估完成：Recall@K=%.3f, MRR@K=%.3f", metrics["recall_at_k"], metrics["mrr_at_k"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.with_ragas:
        run_ragas(samples, kb, settings.rag_top_k)
    passed = metrics["recall_at_k"] >= 0.8 and metrics["mrr_at_k"] >= 0.6
    logger.info("质量门禁：%s", "通过" if passed else "未达标")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())