"""PostgreSQL 备份脚本：逻辑备份（日常）+ 物理基准备份（PITR）。

两种模式：

- ``--mode logical``（默认）：``pg_dump -Fc`` 生成压缩的自定义格式转储，
  带时间戳、自动轮转保留最近 N 份；用于误删数据/表级恢复，
  恢复入口：``scripts/restore_postgres.py``。
- ``--mode base``：``pg_basebackup`` 拉取 PGDATA 物理基础备份（tar.gz），
  配合 docker-compose 已开启的 WAL 归档，构成 PITR（时间点恢复）能力。

连接串默认取环境变量 ``POSTGRES_DSN``（psycopg 语法），也可用 ``--dsn`` 覆盖。
``pg_dump`` / ``pg_basebackup`` 必须在 PATH 中（本机安装 PostgreSQL 客户端，
或在 postgres 容器内挂载本目录执行）。

用法：

    python scripts/backup_postgres.py                      # 逻辑备份
    python scripts/backup_postgres.py --mode base          # PITR 基准备份
    python scripts/backup_postgres.py --keep 30 --dir backups/pg
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _resolve_dsn(cli_dsn: str | None) -> str:
    dsn = cli_dsn or os.environ.get("POSTGRES_DSN", "")
    if not dsn:
        raise SystemExit("未提供连接串：请设置 POSTGRES_DSN 或传 --dsn")
    return dsn


def _require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise SystemExit(
            f"未找到 {name}：请安装 PostgreSQL 客户端工具并加入 PATH，"
            "或在 postgres 容器内执行本脚本。"
        )
    return binary


def _rotate(directory: Path, pattern: str, keep: int) -> list[Path]:
    """按 mtime 倒序保留最近 keep 份，删除更旧的备份。"""

    backups = sorted(
        directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    removed = []
    for old in backups[keep:]:
        old.unlink()
        removed.append(old)
    return removed


def backup_logical(dsn: str, backup_dir: Path, keep: int) -> Path:
    pg_dump = _require_binary("pg_dump")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = backup_dir / f"rag-{stamp}.dump"

    _log(f"开始逻辑备份: {target.name}")
    cmd = [pg_dump, dsn, "-Fc", "-f", str(target), "-v"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        target.unlink(missing_ok=True)
        _log(result.stderr[-2000:])
        raise SystemExit(f"pg_dump 失败，退出码 {result.returncode}")

    removed = _rotate(backup_dir, "rag-*.dump", keep)
    size_mb = target.stat().st_size / 1024 / 1024
    _log(f"逻辑备份完成: {target}（{size_mb:.2f} MB），轮转删除 {len(removed)} 份")
    for old in removed:
        _log(f"  已轮转: {old.name}")
    return target


def backup_base(dsn: str, backup_dir: Path, keep: int) -> Path:
    """PITR 基础备份：pg_basebackup -Ft -X fetch，再打包为 tar.gz。"""

    pg_basebackup = _require_binary("pg_basebackup")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    staging = backup_dir / f"base-{stamp}"
    staging.mkdir()
    target = backup_dir / f"rag-base-{stamp}.tar.gz"

    _log(f"开始物理基准备份: {target.name}（PITR）")
    cmd = [
        pg_basebackup,
        f"--dbname={dsn}",
        "-D", str(staging),
        "-Ft",            # tar 格式：base.tar.gz + pg_wal.tar.gz
        "-X", "fetch",     # 备份结束时回收所需 WAL
        "-z",              # gzip 压缩
        "-P",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        _log(result.stderr[-2000:])
        raise SystemExit(f"pg_basebackup 失败，退出码 {result.returncode}")

    shutil.make_archive(str(target.with_suffix("").with_suffix("")), "gztar", staging)
    shutil.rmtree(staging, ignore_errors=True)
    removed = _rotate(backup_dir, "rag-base-*.tar.gz", keep)
    size_mb = target.stat().st_size / 1024 / 1024
    _log(
        f"基准备份完成: {target}（{size_mb:.2f} MB）。"
        "PITR 还依赖 docker-compose 的 WAL 归档卷 rag_wal_archive；"
        f"轮转删除 {len(removed)} 份"
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="PostgreSQL 备份（logical/base PITR）")
    parser.add_argument("--dsn", default=None, help="psycopg 连接串，默认取 POSTGRES_DSN")
    parser.add_argument("--dir", default="backups/postgres", help="备份输出目录")
    parser.add_argument("--keep", type=int, default=14, help="保留份数（默认 14）")
    parser.add_argument(
        "--mode", choices=("logical", "base"), default="logical",
        help="logical=pg_dump 日常备份；base=pg_basebackup PITR 基准备份",
    )
    args = parser.parse_args()

    dsn = _resolve_dsn(args.dsn)
    backup_dir = Path(args.dir)
    if args.mode == "logical":
        backup_logical(dsn, backup_dir, args.keep)
    else:
        backup_base(dsn, backup_dir, args.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
