"""PBR 贴图生成器 · 重写版 —— 程序入口。

在 PyCharm 里直接右键运行本文件即可。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    try:
        from pbr_app.ui import run
    except ImportError as e:
        print("缺少界面依赖：", e)
        print("请执行：  pip install -r requirements.txt")
        return 1
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
