"""单租户 → 多租户数据迁移脚本。

将 P4 之前的单租户数据迁移到多租户命名空间：

1. **Store 记忆 namespace 升级**：
   - ``("user_memory", <uid>)`` → ``("tenant", "default", "user_memory", <uid>)``
   - ``("user_facts", <uid>)`` → ``("tenant", "default", "user_facts", <uid>)``

2. **知识库文件**：``knowledge_dir/*`` → ``knowledge_dir/default/*``
   （已在租户子目录下的文件跳过，避免重复迁移）。

3. **技能**：已有技能保持全局共享（``skills_root/<name>/``），无需迁移。

用法::

    python scripts/migrate_single_to_multi_tenant.py [--dry-run]

``--dry-run`` 只打印将要执行的操作，不实际修改数据。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil

from src.persistence import database as db
from src.rag.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("migrate")


async def migrate_store_namespaces(dry_run: bool) -> tuple[int, int]:
    """迁移 Store 中的画像记忆与事实记忆 namespace 到多租户结构。"""

    store = db.get_store()
    migrated_memory = 0
    migrated_facts = 0

    # 画像记忆：prefix = ("user_memory",)
    memory_items = await store.asearch(("user_memory",), limit=100000)
    for item in memory_items:
        old_ns = item.namespace
        if len(old_ns) < 2 or old_ns[0] != "user_memory":
            continue
        user_id = old_ns[1]
        new_ns = ("tenant", "default", "user_memory", user_id)
        logger.info(
            "[Store] user_memory: %s → %s (key=%s)",
            old_ns, new_ns, item.key,
        )
        if not dry_run:
            await store.aput(new_ns, item.key, item.value, index=["text"])
            await store.adelete(old_ns, item.key)
        migrated_memory += 1

    # 事实记忆：prefix = ("user_facts",)
    fact_items = await store.asearch(("user_facts",), limit=100000)
    for item in fact_items:
        old_ns = item.namespace
        if len(old_ns) < 2 or old_ns[0] != "user_facts":
            continue
        user_id = old_ns[1]
        new_ns = ("tenant", "default", "user_facts", user_id)
        logger.info(
            "[Store] user_facts: %s → %s (key=%s)",
            old_ns, new_ns, item.key,
        )
        if not dry_run:
            # 事实记忆保留 text 索引；TTL 不迁移（过期条目本应被清扫）
            await store.aput(new_ns, item.key, item.value, index=["text"])
            await store.adelete(old_ns, item.key)
        migrated_facts += 1

    return migrated_memory, migrated_facts


def migrate_knowledge_files(dry_run: bool) -> int:
    """将知识库根目录下的文档迁入 default 租户子目录。"""

    settings = get_settings()
    root = settings.knowledge_dir.resolve()
    if not root.is_dir():
        logger.info("[KB] 知识库目录不存在，跳过: %s", root)
        return 0

    default_dir = root / "default"
    moved = 0
    for entry in sorted(root.iterdir()):
        # 跳过已有的租户子目录与隐藏目录
        if entry.is_dir() and entry.name in {"default"} or entry.name.startswith("."):
            continue
        target = default_dir / entry.name
        logger.info("[KB] %s → %s", entry, target)
        if not dry_run:
            default_dir.mkdir(parents=True, exist_ok=True)
            if target.exists():
                logger.warning("[KB] 目标已存在，跳过: %s", target)
                continue
            shutil.move(str(entry), str(target))
        moved += 1
    return moved


async def main() -> None:
    parser = argparse.ArgumentParser(description="单租户 → 多租户数据迁移")
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印操作，不实际修改"
    )
    args = parser.parse_args()

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    logger.info("===== 开始迁移（%s） =====", mode)

    await db.init_db()
    try:
        mem, facts = await migrate_store_namespaces(args.dry_run)
        kb = migrate_knowledge_files(args.dry_run)
        logger.info(
            "===== 迁移完成（%s）: Store 画像=%d, 事实=%d, KB 文件=%d =====",
            mode, mem, facts, kb,
        )
    finally:
        await db.close_db()


if __name__ == "__main__":
    asyncio.run(main())
