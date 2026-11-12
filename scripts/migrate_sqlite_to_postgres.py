"""SQLite → PostgreSQL 冷切换迁移脚本（生产部署一次性使用）。

迁移范围：
- ✅ users / refresh_tokens（普通标量列，直接行拷贝，ON CONFLICT 跳过）
- ✅ store（每用户长期记忆）：value 为 JSON 文本 → PG jsonb，
  时间列 'YYYY-MM-DD HH:MM:SS'（UTC）→ timestamptz
- ⛔ threads / checkpoints / writes：SQLite checkpoint 为 msgpack BLOB 且
  无 blobs 表，PG 3.x 为 JSONB + checkpoint_blobs 分离结构，序列化格式不兼容，
  冷切换不迁移对话历史；旧 .data/app.db 保留为只读归档，PG 上线后会话从空开始。

安全约束：
- 目标库 users 表必须为空（防止污染已有生产库），可用 --force 覆盖该检查；
- 迁移期间应用必须停机，避免源库继续写入；
- 幂等：全部 INSERT 使用 ON CONFLICT DO NOTHING，可安全重跑。

用法（PowerShell）：
    $env:STORAGE_BACKED="postgres"  # 不需要；脚本直接用 --dsn / POSTGRES_DSN
    python -m scripts.migrate_sqlite_to_postgres `
        --sqlite .data/app.db `
        --dsn "postgresql://rag:rag@localhost:5432/rag" --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

# 允许从项目根直接 python scripts/migrate_...py 运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psycopg.types.json import Json  # noqa: E402
from psycopg_pool import AsyncConnectionPool  # noqa: E402

from src.persistence.database import _SCHEMA_SQL_PG  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | migrate-pg | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("migrate-pg")

_BUSINESS_TABLES = ("users", "refresh_tokens")


def _parse_utc(value: object) -> datetime | None:
    """SQLite store 时间列（UTC 文本）→ 带时区 datetime。"""

    if value is None or value == "":
        return None
    text = str(value)
    for parser in (
        lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S"),
        lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%z"),
        lambda s: datetime.fromisoformat(s),
    ):
        try:
            dt = parser(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    logger.warning("无法解析时间值，按 NULL 处理: %r", value)
    return None


def _read_sqlite(path: Path) -> dict[str, list[tuple]]:
    if not path.exists():
        raise FileNotFoundError(f"源 SQLite 文件不存在: {path}")
    src = sqlite3.connect(str(path))
    src.row_factory = sqlite3.Row
    try:
        data: dict[str, list[tuple]] = {}
        existing = {
            r[0]
            for r in src.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for table in (*_BUSINESS_TABLES, "threads", "store"):
            if table in existing:
                rows = src.execute(f"SELECT * FROM {table}").fetchall()
                data[table] = [tuple(row) for row in rows]
                cols = [d[1] for d in src.execute(f"PRAGMA table_info({table})")]
                logger.info("源表 %s: %d 行, 列=%s", table, len(rows), cols)
            else:
                data[table] = []
                logger.info("源表 %s 不存在，按 0 行处理", table)
        return data
    finally:
        src.close()


async def migrate(sqlite_path: Path, dsn: str, *, force: bool) -> None:
    data = _read_sqlite(sqlite_path)

    # 延迟导入：未装 postgres 依赖时给出明确提示
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore

    pool = AsyncConnectionPool(dsn, open=False, min_size=1, max_size=5, timeout=30)
    await pool.open()
    try:
        # 1) 建表（业务 DDL + LangGraph saver/store 官方 setup）
        async with pool.connection() as conn:
            for statement in _SCHEMA_SQL_PG:
                await conn.execute(statement)
        await AsyncPostgresSaver(pool).setup()
        await AsyncPostgresStore(pool).setup()
        logger.info("目标库 schema 已就绪（业务表 + checkpoints + store）")

        # 2) 安全闸：目标 users 非空则拒绝（除非 --force）
        async with pool.connection() as conn:
            target_users = (
                await (await conn.execute("SELECT COUNT(*) FROM users")).fetchone()
            )[0]
        if target_users and not force:
            raise RuntimeError(
                f"目标库已有 {target_users} 个用户，拒绝迁移以防污染生产库；"
                "确认要合并重跑请加 --force"
            )

        # 3) 业务表行拷贝
        async with pool.connection() as conn:
            for table in _BUSINESS_TABLES:
                rows = data[table]
                if not rows:
                    continue
                cols = [
                    r[1]
                    for r in await (
                        await conn.execute(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name=%s ORDER BY ordinal_position",
                            (table,),
                        )
                    ).fetchall()
                ]
                placeholders = ",".join(["%s"] * len(cols))
                collist = ",".join(cols)
                sql = (
                    f"INSERT INTO {table} ({collist}) VALUES ({placeholders}) "
                    "ON CONFLICT DO NOTHING"
                )
                await conn.executemany(sql, rows)
                logger.info("业务表 %s: 迁入 %d 行", table, len(rows))

            # 4) 长期记忆 store：JSON 文本 → jsonb，文本时间 → timestamptz
            store_rows = data["store"]
            inserted = 0
            for prefix, key, value, created_at, updated_at, expires_at, ttl in store_rows:
                try:
                    payload = json.loads(bytes(value))
                except (json.JSONDecodeError, TypeError):
                    logger.warning("store 行 value 非合法 JSON，跳过: %s %s", prefix, key)
                    continue
                await conn.execute(
                    "INSERT INTO store (prefix, key, value, created_at, updated_at, "
                    "expires_at, ttl_minutes) VALUES (%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT DO NOTHING",
                    (
                        prefix,
                        key,
                        Json(payload),
                        _parse_utc(created_at),
                        _parse_utc(updated_at),
                        _parse_utc(expires_at),
                        ttl,
                    ),
                )
                inserted += 1
            logger.info("长期记忆 store: 迁入 %d/%d 行", inserted, len(store_rows))

        # 5) 校验
        async with pool.connection() as conn:
            for table in (*_BUSINESS_TABLES, "store"):
                count = (
                    await (await conn.execute(f"SELECT COUNT(*) FROM {table}"))
                ).fetchone()[0]
                logger.info("校验 目标表 %s: %d 行", table, count)

        skipped_threads = len(data["threads"])
        logger.info("=" * 60)
        logger.info("迁移完成：业务表 %d 用户 / store %d 条记忆", len(data["users"]), inserted)
        logger.info(
            "已跳过 %d 个会话的对话历史（checkpoint 格式不兼容冷切换），"
            "旧库 %s 已保留为只读归档",
            skipped_threads,
            sqlite_path,
        )
        logger.info("下一步：将 .env 切到 STORAGE_BACKEND=postgres 后启动应用")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SQLite → PostgreSQL 冷切换迁移")
    parser.add_argument("--sqlite", default=".data/app.db", help="源 SQLite 路径")
    parser.add_argument(
        "--dsn",
        default=None,
        help="目标 Postgres DSN（缺省读 POSTGRES_DSN 环境变量）",
    )
    parser.add_argument("--yes", action="store_true", help="确认执行（免交互）")
    parser.add_argument(
        "--force",
        action="store_true",
        help="目标库非空时仍执行（依赖 ON CONFLICT 跳过冲突行）",
    )
    args = parser.parse_args()

    import os

    dsn = args.dsn or os.environ.get("POSTGRES_DSN", "").strip()
    if not dsn:
        parser.error("必须通过 --dsn 或 POSTGRES_DSN 环境变量提供目标库连接串")
    if not args.yes:
        parser.error("冷切换迁移需显式加 --yes 确认（请确保应用已停机）")

    logger.info("开始冷切换迁移: %s -> %s", args.sqlite, dsn.split("@")[-1])
    asyncio.run(migrate(Path(args.sqlite), dsn, force=args.force))


if __name__ == "__main__":
    main()
