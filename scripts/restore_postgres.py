"""PostgreSQL 恢复脚本。

支持两种恢复：

1. **逻辑转储恢复**（默认）：用 ``pg_restore --clean --if-exists`` 把
   ``scripts/backup_postgres.py`` 产出的 ``.dump`` 恢复到目标库
   （会先 DROP 再重建同名对象，属于破坏性操作，必须显式 ``--yes``）。

       python scripts/restore_postgres.py backups/postgres/rag-xxxx.dump --yes

2. **PITR（时间点恢复）**：本脚本只打印标准操作指引，具体步骤需在
   数据库服务器/容器上手工执行（涉及停库、替换 PGDATA、写 recovery 配置，
   不宜由应用侧脚本自动完成）：

       python scripts/restore_postgres.py --pitr-guide

PITR 前置条件：
- 已通过 ``backup_postgres.py --mode base`` 取得物理基础备份；
- docker-compose 已开启 ``archive_mode=on`` 且 WAL 归档卷
  ``rag_wal_archive`` 中有连续归档（基础备份之后的 WAL 不得有缺口）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PITR_GUIDE = """
======================== PostgreSQL PITR 操作指引 ========================
0. 选定恢复目标时间（UTC），例如 2026-10-15 04:00:00+00。

1. 停止应用与数据库容器：
     docker compose stop app
     docker compose stop postgres

2. 备份当前数据卷（防止恢复操作本身造成二次损坏）：
     docker run --rm -v rag_rag_pgdata:/data -v ${PWD}/backups:/backup alpine \\
       tar czf /backup/pgdata-before-restore.tar.gz -C /data .

3. 清空数据卷并展开物理基础备份（base.tar.gz / pg_wal.tar.gz 来自
   backups/postgres/rag-base-*.tar.gz）：
     docker run --rm -v rag_rag_pgdata:/data -v ${PWD}/backups:/backup alpine sh -c \\
       "rm -rf /data/* && tar xzf /backup/rag-base-<时间戳>.tar.gz -C /data
        && tar xzf /data/pg_wal.tar.gz -C /data/pg_wal"

4. 配置恢复目标与归档来源（在 PGDATA 根目录）：
     touch /var/lib/postgresql/data/recovery.signal
     并在 postgresql.auto.conf 追加：
       restore_command = 'cp /var/lib/postgresql/wal_archive/%f %p'
       recovery_target_time = '2026-10-15 04:00:00+00'
       recovery_target_action = 'promote'
   （容器内可通过临时 alpine 容器挂载 rag_rag_pgdata 写入）

5. 启动数据库，PG 自动进入恢复模式，回放 WAL 到目标时间后提升为主库：
     docker compose start postgres
     docker logs -f rag-postgres     # 看到 "recovery stopping before ..." 与
                                     # "database system is ready to accept
                                     # connections" 即恢复完成

6. 恢复完成后立即做一次全新基准备份，并重新拉起应用：
     python scripts/backup_postgres.py --mode base
     docker compose start app

注意：恢复目标时间之后提交的事务会丢失；WAL 归档有缺口则无法越过缺口。
=========================================================================
"""


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def restore_logical(dump: Path, dsn: str) -> None:
    pg_restore = shutil.which("pg_restore")
    if not pg_restore:
        raise SystemExit("未找到 pg_restore：请安装 PostgreSQL 客户端工具。")
    if not dump.is_file():
        raise SystemExit(f"转储文件不存在: {dump}")

    _log(f"开始恢复 {dump} → {dsn.split('@')[-1]}")
    cmd = [
        pg_restore,
        dsn,
        "--clean",
        "--if-exists",
        "--no-owner",
        "--single-transaction",
        "-v",
        str(dump),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    # pg_restore -v 的进度输出走 stderr，不能仅凭 stderr 有内容判失败
    if result.returncode != 0:
        _log(result.stderr[-3000:])
        raise SystemExit(f"pg_restore 失败，退出码 {result.returncode}")
    _log("恢复完成。")


def main() -> int:
    parser = argparse.ArgumentParser(description="PostgreSQL 恢复（逻辑/PITR 指引）")
    parser.add_argument("dump", nargs="?", type=Path, help="逻辑转储 .dump 文件路径")
    parser.add_argument("--dsn", default=None, help="目标库连接串，默认取 POSTGRES_DSN")
    parser.add_argument("--pitr-guide", action="store_true", help="打印 PITR 操作指引")
    parser.add_argument(
        "--yes", action="store_true",
        help="确认执行破坏性恢复（DROP + 重建同名对象）",
    )
    args = parser.parse_args()

    if args.pitr_guide:
        print(PITR_GUIDE)
        return 0

    if args.dump is None:
        parser.error("必须提供 .dump 文件路径，或使用 --pitr-guide")
    if not args.yes:
        raise SystemExit("逻辑恢复是破坏性操作，请确认后追加 --yes 执行。")

    dsn = args.dsn or os.environ.get("POSTGRES_DSN", "")
    if not dsn:
        raise SystemExit("未提供连接串：请设置 POSTGRES_DSN 或传 --dsn")
    restore_logical(args.dump, dsn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
