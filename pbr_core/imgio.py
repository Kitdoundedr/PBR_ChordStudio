"""图像读写与基础滤波。

设计原则：
1. 全程使用 float32 RGB 数组，形状 (H, W, 3)，取值范围 [0, 1]。
2. 不引入 OpenCV —— OpenCV 默认 BGR 顺序，和 PIL / 大多数 DCC 的 RGB 顺序混用
   是法线贴图“导出变粉”的头号来源。本工程只用 PIL + numpy，通道顺序固定为 RGB。
3. 保存法线这类数据贴图时不写 ICC / sRGB 信息，不做任何色彩空间转换。
"""

from __future__ import annotations

import os
import struct
from typing import Optional

import numpy as np
from PIL import Image

# 允许读取超大贴图（8K / 16K）
Image.MAX_IMAGE_PIXELS = None

# 支持的扩展名
READ_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".webp")
WRITE_EXTS = (".png", ".tga", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg",
              ".webp", ".ico", ".gif", ".dds")

# 这些格式存不了 alpha，导出 RGBA 时要降级
NO_ALPHA_EXTS = (".jpg", ".jpeg", ".bmp", ".gif", ".ico")

# 有损格式 —— 法线 / 粗糙度这类数据贴图用它会被压出块状噪声
LOSSY_EXTS = (".jpg", ".jpeg", ".webp", ".gif")


def load_rgb(path: str) -> np.ndarray:
    """读取任意贴图为 float32 RGB [0,1]。

    16 位图按 65535 归一化，8 位图按 255 归一化。
    """
    im = Image.open(path)
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(im).astype(np.float32)
        peak = float(arr.max()) if arr.size else 0.0
        scale = 65535.0 if peak > 255.0 else 255.0
        arr = np.clip(arr / scale, 0.0, 1.0)
        arr = np.repeat(arr[..., None], 3, axis=2)
    else:
        arr = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
    return np.ascontiguousarray(arr)


def load_gray(path: str) -> np.ndarray:
    """读取为单通道 float32 [0,1]（用亮度加权，而不是简单三通道平均）。"""
    return to_luma(load_rgb(path))


def to_luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 亮度。"""
    return (
        rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    ).astype(np.float32)


DDS_MAGIC = 0x20534444  # "DDS "


def save_dds(arr: np.ndarray, path: str) -> str:
    """写未压缩的 32 位 DDS（BGRA）。

    PIL 能读 DDS 但不会写，而各家 DCC / 引擎都认这个头。这里手写 128 字节
    DDS_HEADER —— 不压缩是为了不碰 BC1/BC5 的块压缩：法线贴图一旦被 DXT 压过，
    边缘会出现明显的色块，在引擎里就是一片马赛克。
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=2)
    if a.dtype != np.uint8:
        a = np.clip(np.asarray(a, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    has_alpha = a.shape[2] == 4
    if has_alpha:
        bgra = np.dstack([a[..., 2], a[..., 1], a[..., 0], a[..., 3]])
    else:
        bgra = np.dstack([a[..., 2], a[..., 1], a[..., 0],
                          np.full((h, w), 255, dtype=np.uint8)])

    # DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT
    flags = 0x1 | 0x2 | 0x4 | 0x8 | 0x1000
    pflags = 0x40 | (0x1 if has_alpha else 0)  # DDPF_RGB (+ DDPF_ALPHAPIXELS)

    header = struct.pack("<I", DDS_MAGIC)
    header += struct.pack("<7I", 124, flags, h, w, w * 4, 0, 0)   # size/flags/h/w/pitch/depth/mips
    header += b"\x00" * 44                                        # reserved1[11]
    header += struct.pack("<4I", 32, pflags, 0, 32)               # pf size/flags/fourCC/bitcount
    header += struct.pack("<4I", 0x00FF0000, 0x0000FF00,
                          0x000000FF, 0xFF000000 if has_alpha else 0)
    header += struct.pack("<4I", 0x1000, 0, 0, 0)                 # caps / caps2-4
    header += b"\x00" * 4                                          # reserved2

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(bgra.tobytes())
    return path


def _to_pil(a: np.ndarray) -> Image.Image:
    if a.ndim == 2:
        return Image.fromarray(a, mode="L")
    if a.shape[2] == 3:
        return Image.fromarray(a, mode="RGB")
    if a.shape[2] == 4:
        return Image.fromarray(a, mode="RGBA")
    raise ValueError(f"通道数不支持: {a.shape}")


def save_image(arr: np.ndarray, path: str, bit_depth: int = 8,
               quality: int = 95) -> str:
    """保存数组（按扩展名自动选编码方式）。

    - arr 为 (H,W) 单通道或 (H,W,3) 三通道，(H,W,4) 四通道
    - 值域 [0,1] 的 float，或 0..255 的 uint8
    - 支持 png / tga / tif / bmp / jpg / webp / ico / gif / dds
    单通道 16 位用 PNG 的 I;16 保存；多通道 16 位降级为 8 位并给出提示。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in WRITE_EXTS:
        raise ValueError(f"不支持的输出格式: {ext}")

    a = np.asarray(arr)
    if a.dtype != np.uint8 and a.dtype != np.uint16:
        a = np.clip(a.astype(np.float32), 0.0, 1.0)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if bit_depth == 16 and a.ndim == 2 and ext == ".png":
        u16 = np.clip(a * 65535.0 + 0.5, 0, 65535).astype(np.uint16) if a.dtype != np.uint16 else a
        Image.fromarray(u16).save(path)  # PIL 会写成 16 位灰度 PNG
        return path

    if a.dtype == np.uint16:
        a = (a / 65535.0 * 255.0 + 0.5).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if ext == ".dds":
        return save_dds(a, path)

    im = _to_pil(a)

    # 数据贴图不写任何色彩配置，避免被读取端当成 sRGB 做 gamma 变换
    if ext in (".jpg", ".jpeg"):
        im.convert("RGB").save(path, quality=quality, subsampling=0)
    elif ext == ".webp":
        im.save(path, quality=quality, method=6)
    elif ext == ".gif":
        im.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256).save(path)
    elif ext == ".ico":
        # ICO 单边最大 256，超了 Pillow 会直接报错，先按比例缩
        w, h = im.size
        if max(w, h) > 256:
            k = 256.0 / max(w, h)
            im = im.resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS)
        im.save(path, sizes=[im.size])
    elif ext == ".bmp":
        im.convert("RGB").save(path)
    elif ext in (".tif", ".tiff"):
        im.save(path, compression="tiff_lzw")
    else:
        im.save(path)
    return path


# ---------------------------------------------------------------- 滤波工具


def _box_blur_1d(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """沿指定轴做一次盒式模糊，用积分图实现，O(N)。"""
    if radius < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius + 1, radius)
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * radius + 1)).astype(np.float32)


def box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """二维盒式模糊（可处理单通道或多通道）。"""
    if radius < 1:
        return a.astype(np.float32)
    out = _box_blur_1d(a, radius, 0)
    out = _box_blur_1d(out, radius, 1)
    return out


def gaussian_blur(a: np.ndarray, radius: float) -> np.ndarray:
    """三次盒式模糊近似高斯，速度远快于真高斯，贴图场景足够。"""
    if radius <= 0.5:
        return a.astype(np.float32)
    r = max(1, int(round(radius / 1.7)))  # 三次盒式等效半径
    out = a.astype(np.float32)
    for _ in range(3):
        out = box_blur(out, r)
    return out


def resize(arr: np.ndarray, size: tuple[int, int], resample: Optional[int] = None) -> np.ndarray:
    """缩放数组到 (W, H)。"""
    if size is None:
        return arr
    a = np.asarray(arr)
    if a.ndim == 2:
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), mode="L")
        im = im.resize(size, Image.LANCZOS if resample is None else resample)
        return np.asarray(im).astype(np.float32) / 255.0
    im = Image.fromarray((np.clip(a[..., :3], 0, 1) * 255).astype(np.uint8), mode="RGB")
    im = im.resize(size, Image.LANCZOS if resample is None else resample)
    return np.asarray(im).astype(np.float32) / 255.0
