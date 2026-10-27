"""生成流水线：一份参数 -> 一整套贴图。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import maps as M
from .imgio import gaussian_blur, load_rgb, to_luma
from .normal import height_to_normal, luma_to_normal
from .packing import PRESETS, pack


@dataclass
class PBRParams:
    """所有可调参数。GUI 直接绑定到这个对象上。"""

    # 法线
    normal_strength: float = 3.0
    normal_blur: float = 1.0
    normal_flip_y: bool = False        # False = DirectX(UE/Unity)
    normal_invert: bool = False
    normal_from: str = "color"         # "color" | "height"
    normal_detail_radius: float = 2.0

    # 高度 / 置换
    height_contrast: float = 1.0
    height_invert: bool = False

    # AO
    ao_radius: float = 12.0
    ao_intensity: float = 1.0

    # 粗糙度
    rough_base: float = 0.55
    rough_detail: float = 0.35

    # 金属度
    metal_value: float = 0.0
    metal_edge_boost: float = 0.0

    # 曲率
    curv_radius: float = 3.0
    curv_strength: float = 1.0


@dataclass
class PBRSet:
    """生成结果。maps 里是可直接保存的数组（法线为 uint8 RGB）。"""

    maps: dict[str, np.ndarray] = field(default_factory=dict)
    packed: dict[str, np.ndarray] = field(default_factory=dict)
    base_color: np.ndarray | None = None

    def preview(self, key: str) -> np.ndarray:
        if key in self.maps:
            return self.maps[key]
        return self.packed[key]


def generate(
    base_color: np.ndarray,
    params: PBRParams,
    height: np.ndarray | None = None,
    masks: dict[str, np.ndarray] | None = None,
) -> PBRSet:
    """从基础色（可选外部高度图）生成整套 PBR 贴图。

    masks 可以覆盖任意一张图的默认算法，例如 {"MET": 手绘金属遮罩}。
    """
    rgb = np.asarray(base_color, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    rgb = rgb[..., :3]

    # --- 高度
    if height is not None:
        hgt = np.asarray(height, dtype=np.float32)
        if hgt.ndim == 3:
            hgt = to_luma(hgt)
        if params.height_invert:
            hgt = 1.0 - hgt
    else:
        hgt = M.height_from_color(rgb, params.height_contrast, params.height_invert)

    # --- 法线（核心）
    if params.normal_from == "height" or height is not None:
        nrm = height_to_normal(
            hgt,
            strength=params.normal_strength,
            flip_y=params.normal_flip_y,
            invert=False,
            blur=params.normal_blur,
        )
    else:
        nrm = luma_to_normal(
            rgb,
            strength=params.normal_strength,
            detail_radius=params.normal_detail_radius,
            flip_y=params.normal_flip_y,
            invert=params.normal_invert,
        )

    # --- 其余通道
    out: dict[str, np.ndarray] = {
        "COL": rgb,
        "NRM": nrm,
        "DIS": M.displacement(hgt),
        "OCC": M.ambient_occlusion(hgt, params.ao_radius, params.ao_intensity),
        "RGH": M.roughness_from_color(rgb, params.rough_base, params.rough_detail),
        "MET": M.metallic_from_color(rgb, params.metal_value, params.metal_edge_boost),
        "CUR": M.curvature(rgb, params.curv_radius, 0.5, params.curv_strength),
    }
    out["GLS"] = M.glossiness(out["RGH"])
    out["SHN"] = M.shininess(out["RGH"])
    out["EMI"] = np.zeros_like(hgt)
    out["TRS"] = np.zeros_like(hgt)
    out["COA"] = np.zeros_like(hgt)

    if masks:
        for k, v in masks.items():
            if v is not None:
                out[k] = np.asarray(v, dtype=np.float32)

    # --- 打包图
    packed: dict[str, np.ndarray] = {}
    for preset_name, (c0, c1, c2, c3) in PRESETS.items():
        key = preset_name.split()[0]
        vals = []
        for c in (c0, c1, c2):
            if c == "A1":
                vals.append(np.ones_like(hgt))
            else:
                vals.append(out.get(c))
        a = None if c3 != "MET" else out.get("MET")
        packed[key] = pack(vals[0], vals[1], vals[2], a)

    return PBRSet(maps=out, packed=packed, base_color=rgb)


def load_source(path: str) -> np.ndarray:
    """读取源贴图，顺便做超大图保护。"""
    rgb = load_rgb(path)
    # 超过 4096 的图，法线梯度容易全是噪点，交给调用方决定是否缩放
    return rgb
