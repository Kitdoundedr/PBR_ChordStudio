"""导出：命名规则 + 落盘 + 落盘后自检。

命名沿用 PBR 行业约定的后缀（和 Substance / DCC 导出预设一致），
导出前可以自由勾选需要哪些图。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .imgio import save_image
from .normal import assert_channel_order, channel_stats

# 后缀 -> 中文名
SUFFIXES: dict[str, str] = {
    "COL": "基础色 BaseColor",
    "NRM": "法线 Normal",
    "RGH": "粗糙度 Roughness",
    "MET": "金属度 Metallic",
    "OCC": "环境光遮蔽 AO",
    "DIS": "置换/高度 Height",
    "CUR": "曲率 Curvature",
    "GLS": "光泽度 Glossiness",
    "SHN": "高光指数 Shininess",
    "EMI": "自发光 Emissive",
    "TRS": "透射 Transmission",
    "COA": "清漆 Coat",
}

# 这些是线性数据贴图，导入引擎时必须关掉 sRGB
LINEAR_MAPS = {"NRM", "RGH", "MET", "OCC", "DIS", "CUR", "GLS", "SHN"}


@dataclass
class ExportResult:
    path: str
    suffix: str
    size: tuple[int, int]
    ok: bool = True
    note: str = ""


def build_name(stem: str, suffix: str, pattern: str = "{stem}_{suffix}", ext: str = ".png") -> str:
    return pattern.format(stem=stem, suffix=suffix) + ext


def export_maps(
    maps: dict[str, np.ndarray],
    out_dir: str,
    stem: str,
    suffixes: list[str] | None = None,
    pattern: str = "{stem}_{suffix}",
    ext: str = ".png",
    bit_depth: int = 8,
) -> list[ExportResult]:
    """把生成好的贴图写到磁盘，并逐张回读校验。"""
    os.makedirs(out_dir, exist_ok=True)
    results: list[ExportResult] = []
    pick = suffixes or list(maps.keys())

    for suf in pick:
        if suf not in maps:
            continue
        arr = maps[suf]
        if arr is None:
            continue

        # 法线：落盘前守门，通道顺序不对直接拒写
        if suf == "NRM":
            assert_channel_order(arr, strict=True)
            rgb = arr
        else:
            rgb = arr

        path = os.path.join(out_dir, build_name(stem, suf, pattern, ext))
        save_image(rgb, path, bit_depth=bit_depth)
        res = _verify_saved(path, suf, rgb.shape[:2])
        results.append(res)
    return results


def _verify_saved(path: str, suffix: str, expect_size: tuple[int, int]) -> ExportResult:
    """回读刚写出的文件，独立校验一遍（不复用内存里的数组）。"""
    try:
        im = Image.open(path)
        im.load()
        size = im.size
        note = ""
        ok = True

        if suffix == "NRM":
            arr = np.asarray(im.convert("RGB"))
            st = channel_stats(arr)
            if not st["is_correct_order"] and not st["is_flat"]:
                ok = False
                note = f"通道顺序异常，Z 在 {st['z_channel']} 通道"
            else:
                note = "B={:.0f} (Z轴正确)".format(st["mean"][2] * 255)

        if im.mode == "RGBA":
            alpha = np.asarray(im)[..., 3]
            if alpha.max() < 255:
                ok = False
                note = (note + "；" if note else "") + f"alpha 未填满 (max={alpha.max()})"

        if size != expect_size[::-1] and size != expect_size:
            ok = False
            note = (note + "；" if note else "") + f"尺寸异常 {size}"

        return ExportResult(path=path, suffix=suffix, size=size, ok=ok, note=note)
    except Exception as e:  # noqa: BLE001
        return ExportResult(path=path, suffix=suffix, size=(0, 0), ok=False, note=f"回读失败: {e}")
