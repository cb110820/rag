"""PyInstaller 打包入口（开发态也可用：python launcher.py）。

打包后的运行布局（onedir）：

    RagAssistant/
    ├── RagAssistant.exe
    ├── .env                # 外置可编辑（模型密钥/JWT 密钥），构建时拷贝
    ├── web/                # 前端（首启从 _internal 种子拷贝，之后可自行替换）
    ├── skills/             # 员工技能（可写，上传的技能落这里）
    ├── data/knowledge/     # 知识库种子文档（可写）
    ├── .cache/             # 嵌入缓存（运行期生成）
    ├── .data/              # 账号/会话/checkpoint/长期记忆（运行期生成）
    ├── logs/               # 日志（运行期生成）
    └── _internal/          # PyInstaller 依赖与只读种子资源

启动行为：
1. 打包态切换工作目录到 exe 所在目录，保证所有相对路径（.env/.data/skills
   等）落在可写的应用目录；
2. 首启时把 bundle 内的只读种子资源（web/skills/data）拷贝到应用目录，
   已存在则不覆盖用户数据；
3. 启动 uvicorn 并（可选）自动打开浏览器。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import traceback
from pathlib import Path

# 打包态：exe 所在目录；开发态：项目根（launcher.py 所在目录）
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = Path(str(getattr(sys, "_MEIPASS", APP_DIR)))

# 首启时从 bundle 拷贝到应用目录的种子资源（相对路径，仅当目标不存在时拷贝）
_SEED_RESOURCES = ("web", "skills", "data")


def _seed_runtime_resources() -> None:
    """把随包只读种子资源首次释放到 exe 同级可写目录。

    - 目录存在即视为已初始化，绝不覆盖用户后续上传的技能/知识库/自定义前端；
    - 逐文件跳过已存在项，因此发布新版本只补新文件，旧文件保持用户版本。
    """

    for rel in _SEED_RESOURCES:
        src = RESOURCE_DIR / rel
        if not src.exists():
            continue
        dst = APP_DIR / rel
        for item in src.rglob("*"):
            rel_path = item.relative_to(src)
            target = dst / rel_path
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepAgents RAG 助手")
    parser.add_argument("--host", default=os.environ.get("RAG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RAG_PORT", "8000")))
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    return parser.parse_args()


def main() -> int:
    if getattr(sys, "frozen", False):
        os.chdir(APP_DIR)
    _seed_runtime_resources()

    args = _parse_args()

    # 延迟导入：chdir 与种子释放必须发生在配置模块读取路径之前
    import uvicorn

    print("=" * 64)
    print("  DeepAgents RAG 助手")
    print(f"  应用目录 : {APP_DIR}")
    print(f"  访问地址 : http://{args.host}:{args.port}/")
    print("  关闭服务 : 直接关闭本窗口，或按 Ctrl+C")
    print("=" * 64)

    if not args.no_browser:
        # 给 uvicorn 1.5 秒启动时间后再拉起默认浏览器
        threading.Timer(
            1.5,
            lambda: __import__("webbrowser").open(f"http://{args.host}:{args.port}/"),
        ).start()

    # 直接传 app 对象而非 "src.api:app" 字符串，避免打包后字符串导入失败
    from src.api.app import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001 —— 顶层兜底，避免双击启动时窗口一闪而过
        traceback.print_exc()
        if getattr(sys, "frozen", False):
            print("\n启动失败，按回车键关闭窗口……")
            try:
                input()
            except EOFError:
                pass
        raise
