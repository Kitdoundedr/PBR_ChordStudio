"""由基础颜色 / 高度派生其余 PBR 贴图。

所有函数统一输入 (H,W,3) 或 (H,W) 的 float32 [0,1]，输出 (H,W) float32 [0,1]。
这些都是“可调的经验算法”，不是物理测量，参数都开放给 UI 调整。
"""

from __future__ import annotations

import numpy as np

from .imgio import box_blur, gaussian_blur, to_luma


def height_from_color(rgb: np.ndarray, contrast: float = 1.0, invert: bool = False) -> np.ndarray:
    """亮度当高度：亮=凸，暗=凹。contrast 拉伸对比。"""
    h = to_luma(np.asarray(rgb, dtype=np.float32))
    h = np.clip((h - 0.5) * float(contrast) + 0.5, 0.0, 1.0)
    return 1.0 - h if invert else h


def ambient_occlusion(height: np.ndarray, radius: float = 12.0, intensity: float = 1.0) -> np.ndarray:
    """由高度近似 AO：比周围低的区域（缝隙）变暗。

    返回 1=开阔，0=完全遮蔽。
    """
    h = np.asarray(height, dtype=np.float32)
    if h.ndim == 3:
        h = to_luma(h)
    blurred = gaussian_blur(h, max(1.0, radius))
    occ = np.clip((blurred - h) * 6.0 * float(intensity), 0.0, 1.0)
    return np.clip(1.0 - occ, 0.0, 1.0)


def roughness_from_color(
    rgb: np.ndarray,
    base: float = 0.5,
    detail: float = 0.35,
    radius: float = 8.0,
) -> np.ndarray:
    """亮度越低越粗糙（脏/暗处更糙），再叠加一层局部起伏。"""
    lum = to_luma(np.asarray(rgb, dtype=np.float32))
    local = lum - box_blur(lum, max(1, int(radius)))
    r = float(base) + (0.5 - lum) * float(detail) + local * 0.5
    return np.clip(r, 0.02, 1.0)


def metallic_from_color(rgb: np.ndarray, value: float = 0.0, edge_boost: float = 0.0) -> np.ndarray:
    """金属度默认给常量（木头、石头、瓦片这类基本为 0）。"""
    h, w = np.asarray(rgb).shape[:2]
    m = np.full((h, w), float(value), dtype=np.float32)
    if edge_boost > 0:
        m = np.clip(m + curvature(rgb) * max(0.0, edge_boost), 0.0, 1.0)
    return m


def curvature(rgb: np.ndarray, radius: float = 3.0, bias: float = 0.5, strength: float = 1.0) -> np.ndarray:
    """曲率：凸出部分亮、凹陷部分暗，常用来做边缘磨损遮罩。"""
    h = to_luma(np.asarray(rgb, dtype=np.float32))
    if radius > 1.0:
        h = gaussian_blur(h, radius)
    lap = (
        np.roll(h, 1, 0) + np.roll(h, -1, 0) + np.roll(h, 1, 1) + np.roll(h, -1, 1) - 4.0 * h
    )
    return np.clip(bias - lap * 8.0 * float(strength), 0.0, 1.0)


def glossiness(roughness: np.ndarray) -> np.ndarray:
    """光泽 = 1 - 粗糙。"""
    return np.clip(1.0 - np.asarray(roughness, dtype=np.float32), 0.0, 1.0)


def shininess(roughness: np.ndarray, lo: float = 2.0, hi: float = 128.0) -> np.ndarray:
    """把粗糙度映射到传统高光指数，供老引擎使用。"""
    g = glossiness(roughness)
    return np.clip(lo + (hi - lo) * g, 0.0, 255.0) / 255.0


def displacement(height: np.ndarray, scale: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """位移/高度输出，留出 scale / bias 便于对齐模型表面。"""
    h = np.asarray(height, dtype=np.float32)
    return np.clip(h * float(scale) + float(bias), 0.0, 1.0)
