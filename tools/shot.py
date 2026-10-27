"""界面截图工具（开发用）。

用法：
    python tools/shot.py [源图路径]
不传源图时用程序化生成的一块「苔藓石头」样图，避免依赖外部素材。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pbr_app.ui import MainWindow  # noqa: E402
from pbr_core.imgio import save_image  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "output")


def fbm(size: int = 512, octaves: int = 5, seed: int = 7, base: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), dtype=np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        g = rng.random((n, n)).astype(np.float32)
        # 双线性放大
        idx = (np.arange(size) + 0.5) / size * n - 0.5
        i0 = np.floor(idx).astype(np.int32)
        t = (idx - i0).astype(np.float32)
        t = t * t * (3 - 2 * t)
        i0c = np.clip(i0, 0, n - 1)
        i1c = np.clip(i0 + 1, 0, n - 1)
        a = g[np.ix_(i0c, i0c)]
        b = g[np.ix_(i0c, i1c)]
        c = g[np.ix_(i1c, i0c)]
        d = g[np.ix_(i1c, i1c)]
        tx, ty = t[None, :], t[:, None]
        acc += amp * ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty)
        total += amp
        amp *= 0.52
    acc /= total
    return (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)


def moss_stone(size: int = 512, seed: int = 7) -> np.ndarray:
    h = fbm(size, 5, seed, base=2)
    fine = fbm(size, 4, seed + 11, base=4)
    h = np.clip(h * 0.86 + fine * 0.14, 0, 1)

    # 灰石 -> 苔藓绿
    stone = np.stack([0.34 + h * 0.34, 0.33 + h * 0.33, 0.30 + h * 0.30], axis=2)
    moss = np.stack([0.16 + h * 0.30, 0.30 + h * 0.42, 0.10 + h * 0.20], axis=2)
    mask = np.clip((h - 0.42) * 3.2, 0, 1)[..., None]
    rgb = stone * (1 - mask) + moss * mask
    rgb *= (0.86 + 0.28 * fbm(size, 4, seed + 23))[..., None]
    return np.clip(rgb, 0, 1)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    src = sys.argv[1] if len(sys.argv) > 1 else ""
    if not src or not os.path.exists(src):
        src = os.path.join(OUT, "_sample_moss.png")
        if not os.path.exists(src):
            save_image(moss_stone(512, 3), src)

    app = QApplication(sys.argv[:1])
    w = MainWindow()
    w.resize(1500, 930)
    w.show()
    app.processEvents()
    w.load(src)
    for _ in range(30):
        app.processEvents()

    w.grab().save(os.path.join(OUT, "ui_dash.png"))
    w.switch_tab(1)
    for _ in range(30):
        app.processEvents()
    w.grab().save(os.path.join(OUT, "ui_pack.png"))
    w.switch_tab(0)

    print("ok", w.size())
    QTimer.singleShot(0, app.quit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
