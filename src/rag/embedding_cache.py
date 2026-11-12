"""语料块嵌入向量的磁盘缓存（增量嵌入）。

知识库每次重建索引都会重新解析、分块，但嵌入是其中唯一昂贵的步骤
（BGE-M3 模型推理）。本模块以 ``(模型命名空间, 块文本 SHA256)`` 为键，
把向量持久化到 SQLite：

- 未改动的文件 → 块文本一致 → 直接命中缓存，零模型调用；
- 改动的文件 → 只有内容发生变化的块需要重新嵌入，其余块仍命中；
- 新增文件 / 删除文件 → 只计算新块，删除块通过 :meth:`prune` 清理；
- 服务重启后缓存仍在磁盘上，冷启动也不必重新嵌入整个知识库。

键中包含模型命名空间（如 ``bge-m3:BAAI/bge-m3``），切换嵌入模型不会
读到错误向量。SQLite 不可用时自动降级为直连嵌入器，不影响主流程。
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.rag.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

# 键版本前缀：嵌入逻辑或存储格式发生不兼容变化时递增，使旧缓存整体失效
_KEY_VERSION = "v1"


class CachedCorpusEmbedder:
    """带磁盘缓存的语料嵌入器，仅供知识库块向量使用。

    查询（query）向量不经过此缓存——查询文本是一次性的，缓存没有复用价值，
    反而会让数据库无限膨胀；查询仍直接调用底层嵌入器。
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        namespace: str,
        db_path: Path,
        batch_size: int = 64,
    ) -> None:
        self.provider = provider
        self._namespace = namespace
        self._db_path = Path(db_path)
        self._batch_size = batch_size
        self._lock = threading.Lock()
        # 最近一次 embed_corpus 的统计，供日志与测试断言
        self.last_hits = 0
        self.last_misses = 0
        self._conn: sqlite3.Connection | None = None
        self._enabled = False
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    embedding_key TEXT PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.commit()
            self._enabled = True
            logger.info("嵌入缓存已启用: %s（命名空间 %s）", self._db_path, namespace)
        except sqlite3.Error as exc:
            self._enabled = False
            logger.warning("嵌入缓存初始化失败，降级为不缓存: %s", exc)

    def _make_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{_KEY_VERSION}:{self._namespace}:{digest}"

    def embed_corpus(self, texts: list[str]) -> np.ndarray:
        """嵌入一批语料块，命中缓存的直接读盘，未命中的批量调用模型。

        返回顺序与 ``texts`` 严格一致；空输入返回空矩阵。
        """

        if not texts:
            self.last_hits = 0
            self.last_misses = 0
            return np.empty((0, 0), dtype=np.float32)

        started = time.perf_counter()
        if not self._enabled or self._conn is None:
            return self.provider.embed(texts)

        keys = [self._make_key(text) for text in texts]
        with self._lock:
            return self._embed_locked(texts, keys, started)

    def _embed_locked(
        self, texts: list[str], keys: list[str], started: float
    ) -> np.ndarray:
        assert self._conn is not None
        vectors: dict[str, np.ndarray] = {}
        miss_indices: list[int] = []
        try:
            rows = self._conn.execute(
                f"SELECT embedding_key, dim, vector FROM embeddings WHERE embedding_key IN "
                f"({','.join('?' for _ in keys)})",
                keys,
            ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("嵌入缓存读取失败，本次全部重新嵌入: %s", exc)
            return self.provider.embed(texts)

        for key, raw in ((row[0], row[2]) for row in rows):
            vectors[key] = np.frombuffer(raw, dtype=np.float32).copy()

        miss_indices = [index for index, key in enumerate(keys) if key not in vectors]
        # 同 key 的旧维度记录在插入时由 INSERT OR REPLACE 覆盖；
        # 命名空间已含模型/维度标识，正常情况下不会命中到异维向量

        new_vectors: dict[str, np.ndarray] = {}
        for begin in range(0, len(miss_indices), self._batch_size):
            batch_indices = miss_indices[begin : begin + self._batch_size]
            batch_texts = [texts[index] for index in batch_indices]
            batch_vectors = self.provider.embed(batch_texts)
            for index, vector in zip(batch_indices, batch_vectors, strict=True):
                new_vectors[keys[index]] = np.asarray(vector, dtype=np.float32)

        if new_vectors:
            now = time.time()
            payload = [
                (key, vector.shape[0], vector.tobytes(), now)
                for key, vector in new_vectors.items()
            ]
            try:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO embeddings "
                    "(embedding_key, dim, vector, created_at) VALUES (?, ?, ?, ?)",
                    payload,
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.warning("嵌入缓存写入失败（不影响本次检索）: %s", exc)

        vectors.update(new_vectors)
        # 防御：若向量维度不一致（异常缓存/模型配置错误），放弃缓存直连模型
        dims = {vector.shape[0] for vector in vectors.values()}
        if len(dims) != 1:
            logger.warning("缓存向量维度不一致 %s，本次全部重新嵌入", dims)
            return self.provider.embed(texts)

        result = np.stack([vectors[key] for key in keys]).astype(np.float32)
        self.last_hits = len(texts) - len(miss_indices)
        self.last_misses = len(miss_indices)
        logger.info(
            "语料嵌入：共 %d 块，缓存命中 %d，新计算 %d，耗时 %.0fms",
            len(texts),
            self.last_hits,
            self.last_misses,
            (time.perf_counter() - started) * 1000,
        )
        return result

    def prune(self, live_texts: list[str], batch_size: int = 5000) -> int:
        """删除不属于当前任何语料块的缓存行（含被删文件、旧模型命名空间）。

        分批扫描 + 批量删除，避免百万级缓存行时一次性加载全部 key 导致 OOM。
        返回删除的行数。
        """

        if not self._enabled or self._conn is None:
            return 0
        live_keys = {self._make_key(text) for text in live_texts}
        deleted = 0
        with self._lock:
            try:
                assert self._conn is not None
                # 分批扫描全表，每批取 batch_size 个 key 与 live_keys 比对
                offset = 0
                while True:
                    rows = self._conn.execute(
                        "SELECT embedding_key FROM embeddings LIMIT ? OFFSET ?",
                        (batch_size, offset),
                    ).fetchall()
                    if not rows:
                        break
                    stale = [row[0] for row in rows if row[0] not in live_keys]
                    if stale:
                        self._conn.executemany(
                            "DELETE FROM embeddings WHERE embedding_key = ?",
                            [(key,) for key in stale],
                        )
                        self._conn.commit()
                        deleted += len(stale)
                    offset += len(rows)
                if deleted:
                    logger.info("嵌入缓存剪枝：移除 %d 个过期块向量", deleted)
            except sqlite3.Error as exc:
                logger.warning("嵌入缓存剪枝失败（不影响检索）: %s", exc)
        return deleted

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._enabled = False

    def __enter__(self) -> CachedCorpusEmbedder:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
