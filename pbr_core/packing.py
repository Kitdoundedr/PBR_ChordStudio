"""通道打包：把多张灰度图合并成一张 RGBA，供引擎省采样。

注意：打包图里的通道含义由使用者约定，本模块只保证
    通道顺序 = R, G, B, A （绝不交换）
并且 alpha 默认写满 255 —— 之前那种导出后“半透明”的结果就是因为 alpha 没填。
"""

from __future__ import annotations

import numpy as np

# 常见打包约定
PRESETS: dict[str, tuple[str, str, str, str]] = {
    "ORM  (UE/Unity 通用)": ("OCC", "RGH", "MET", "A1"),
    "RMA  (Rough/Metal/AO)": ("RGH", "MET", "OCC", "A1"),
    "MRAO (Metal/Rough/AO)": ("MET", "RGH", "OCC", "A1"),
    "RHAM (本工具旧版)": ("RGH", "HGT", "OCC", "MET"),
}


def pack(r, g, b, a=None, default_alpha: int = 255) -> np.ndarray:
    """四张灰度图 -> uint8 RGBA。缺省通道填 0，alpha 缺省填满。

    尺寸取第一张非空的通道，所以只往 G/B/A 里塞东西也能打包。
    """
    ref = next((x for x in (r, g, b, a) if x is not None), None)
    if ref is None:
        raise ValueError("没有任何通道可打包")
    h, w = np.asarray(ref).shape[:2]

    def _c(x):
        if x is None:
            return np.zeros((h, w), dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[..., 0]
        return np.clip(x, 0.0, 1.0)

    arrs = [_c(r), _c(g), _c(b), _c(a)]
    out = np.stack(arrs, axis=2)
    if a is None:
        out[..., 3] = default_alpha / 255.0
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def unpack(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] != 4:
        raise ValueError("unpack 需要 (H,W,4) 的图")
    f = a.astype(np.float32) / 255.0
    return f[..., 0], f[..., 1], f[..., 2], f[..., 3]
