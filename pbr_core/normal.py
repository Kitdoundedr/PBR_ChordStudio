"""法线贴图：生成、编码、修复、校验。

================= 关键约定（粉色的根源就在这里）=================
切线空间法线贴图 = 单位向量 (nx, ny, nz) 映射到 0..255：
    R = nx * 0.5 + 0.5      -> 一般围绕 128
    G = ny * 0.5 + 0.5      -> 一般围绕 128
    B = nz * 0.5 + 0.5      -> 永远 >= 128，平坦处接近 255
所以一张正常的法线贴图整体偏蓝紫；如果蓝色通道被写到了红色通道，
就会得到 R≈255 / G≈128 / B≈128 —— 也就是“整张粉掉”。
本模块的 encode_normal() 强制 通道顺序 = (X, Y, Z) -> (R, G, B)，
并且在保存前提供 assert_channel_order() 自检。
=================================================================
"""

from __future__ import annotations

import numpy as np

from .imgio import gaussian_blur, to_luma

EPS = 1e-8


# ------------------------------------------------------------------ 编码 / 解码


def encode_normal(n: np.ndarray) -> np.ndarray:
    """把 float 法线 (H,W,3) 从 [-1,1] 编码成 uint8 RGB (H,W,3)。

    这里显式地按 (X, Y, Z) 顺序堆叠，绝不依赖任何库的默认通道顺序。
    """
    n = np.asarray(n, dtype=np.float32)
    if n.ndim != 3 or n.shape[2] != 3:
        raise ValueError("encode_normal 需要 (H,W,3) 的法线数组")
    # 归一化，避免强度过大导致向量长度偏离 1
    length = np.sqrt((n * n).sum(axis=2, keepdims=True))
    n = n / np.maximum(length, EPS)
    out = n * 0.5 + 0.5
    out[..., 2] = np.clip(out[..., 2], 0.5, 1.0)  # nz 恒为正，B 恒 >= 128
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def decode_normal(rgb) -> np.ndarray:
    """把 uint8 / float RGB 解回 float 法线 (H,W,3)，范围 [-1,1]。"""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("decode_normal 需要至少 3 通道的图")
    a = a[..., :3].astype(np.float32)
    if a.dtype != np.float32 or a.max() > 1.5:
        a = a / 255.0
    return a * 2.0 - 1.0


# ------------------------------------------------------------------ 由高度生成


def height_to_normal(
    height: np.ndarray,
    strength: float = 1.0,
    flip_y: bool = False,
    invert: bool = False,
    blur: float = 0.0,
    clamp_z: float = 1.0,
) -> np.ndarray:
    """高度图 -> 法线贴图（uint8 RGB）。

    strength : 强度倍数，1.0 = 按像素梯度原始大小
    flip_y   : True 得到 OpenGL(+Y) 约定，False 为 DirectX(-Y)
               UE / Unity 用 DirectX，Godot / Blender 部分管线用 OpenGL
    invert   : 高度反相（凹 <-> 凸）
    blur     : 生成前对高度做高斯平滑，去掉噪点引起的高频毛刺
    """
    h = np.asarray(height, dtype=np.float32)
    if h.ndim == 3:
        h = to_luma(h)
    h = np.clip(h, 0.0, 1.0)

    if invert:
        h = 1.0 - h
    if blur > 0.5:
        h = gaussian_blur(h, blur)

    # 梯度：gx 沿列(x 向右)，gy 沿行(行号向下)
    gy, gx = np.gradient(h.astype(np.float32))

    nx = -gx * strength
    # 行号向下，而 +V 向上，所以这里符号取正
    ny = gy * strength
    if flip_y:
        ny = -ny

    nz = np.full_like(nx, float(clamp_z))

    normal = np.stack([nx, ny, nz], axis=2)
    return encode_normal(normal)


def luma_to_normal(
    rgb: np.ndarray,
    strength: float = 2.0,
    detail_radius: float = 2.0,
    flip_y: bool = False,
    invert: bool = False,
) -> np.ndarray:
    """从颜色贴图的亮度细节推法线（没有高度图时的快速方案）。

    先做高通（原图 - 模糊图）去掉大块明暗，避免整体起伏把法线压垮。

    尺度用 |detail| 的 95 分位，而不是标准差：标准差会被少量极值带偏，
    曾经导致任何照片都被放大成逐像素雪花噪点。用分位数 + 硬截断后，
    贴图的结构边缘保留、传感器噪点不会被当成细节。
    """
    lum = to_luma(np.asarray(rgb, dtype=np.float32))
    detail = lum - gaussian_blur(lum, max(1.5, float(detail_radius)))

    scale = float(np.percentile(np.abs(detail), 95)) if detail.size else 0.0
    if scale < 1e-4:
        scale = float(detail.std())
    if scale > 1e-4:
        detail = np.clip(detail / scale, -1.0, 1.0) * 0.18

    h = 0.5 + detail
    return height_to_normal(h, strength=strength, flip_y=flip_y, invert=invert)


# ------------------------------------------------------------------ 诊断


def channel_stats(rgb: np.ndarray) -> dict:
    """统计三通道，并判断 Z 轴是否在正确通道（B）。"""
    a = np.asarray(rgb)[..., :3].astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    means = [float(a[..., i].mean()) for i in range(3)]
    stds = [float(a[..., i].std()) for i in range(3)]
    # Z 轴特征：均值最高、方差最小
    z_channel = int(np.argmax([means[i] - stds[i] for i in range(3)]))
    return {
        "mean": means,
        "std": stds,
        "z_channel": "RGB"[z_channel],
        "is_correct_order": z_channel == 2,
        "is_flat": stds[0] < 0.01 and stds[1] < 0.01,
    }


def is_pink(rgb: np.ndarray) -> bool:
    """判断是否呈“粉色法线”特征：R 高、G/B 居中。"""
    st = channel_stats(rgb)
    m = st["mean"]
    return m[0] > 0.75 and 0.35 < m[1] < 0.65 and 0.35 < m[2] < 0.65


# ------------------------------------------------------------------ 修复


def repair_normal(
    rgb: np.ndarray,
    swap_rb: bool | None = None,
    flip_g: bool = False,
    rebuild_z: bool = False,
) -> np.ndarray:
    """修复已有的错误法线贴图，返回 uint8 RGB。

    swap_rb   : None = 自动判断（R 通道像 Z 就换），True/False = 强制
    flip_g    : 翻转绿通道（DirectX <-> OpenGL）
    rebuild_z : 丢弃原 B，由 XY 重新计算 Z = sqrt(1 - x^2 - y^2)
                （用于 B 通道被压坏、或原图存了 DXT5nm 之类只有两通道的情况）
    """
    a = np.asarray(rgb)[..., :3]
    auto = is_pink(a) or not channel_stats(a)["is_correct_order"]
    do_swap = auto if swap_rb is None else bool(swap_rb)

    if do_swap:
        a = a[..., ::-1]  # R <-> B

    n = decode_normal(a)
    if flip_g:
        n[..., 1] = -n[..., 1]

    if rebuild_z:
        xy2 = n[..., 0] ** 2 + n[..., 1] ** 2
        n[..., 2] = np.sqrt(np.clip(1.0 - xy2, 1e-6, 1.0))

    return encode_normal(n)


# ------------------------------------------------------------------ 自检


def assert_channel_order(rgb: np.ndarray, strict: bool = False) -> None:
    """保存前的守门员：确认 B 通道是 Z 轴。

    strict=True 时直接抛异常，防止错误数据落盘。
    """
    st = channel_stats(rgb)
    if st["is_flat"]:
        return
    if not st["is_correct_order"]:
        msg = (
            "法线贴图通道顺序异常：Z 轴出现在 {} 通道（应为 B）。"
            " 请检查是否混用了 BGR 数据。".format(st["z_channel"])
        )
        if strict:
            raise ValueError(msg)
        print("[警告] " + msg)
