"""通道顺序回归测试：确保法线永远是 B=Z，打包图 alpha 永远是 255。

直接运行：  python tests/test_channels.py
或者 PyCharm 里对文件名右键 Run。
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbr_core.exporter import export_maps  # noqa: E402
from pbr_core.normal import (  # noqa: E402
    channel_stats,
    encode_normal,
    height_to_normal,
    repair_normal,
)
from pbr_core.packing import pack  # noqa: E402
from pbr_core.pipeline import PBRParams, generate  # noqa: E402


def _dummy_color(size=128) -> np.ndarray:
    """造一张有明暗起伏的假颜色图。"""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 0.5 + 0.35 * np.sin(x / 7.0) * np.cos(y / 9.0)
    return np.stack([base, base * 0.9, base * 0.8], axis=2).clip(0, 1)


def test_encode_channel_order():
    """Z 必须落在 B 通道。"""
    n = np.zeros((8, 8, 3), dtype=np.float32)
    n[..., 2] = 1.0  # 平坦朝上的法线
    enc = encode_normal(n)
    assert enc[..., 2].min() == 255, "Z 没写进 B 通道"
    assert abs(int(enc[..., 0].mean()) - 128) <= 1
    assert abs(int(enc[..., 1].mean()) - 128) <= 1


def test_height_normal_not_pink():
    h = _dummy_color()[..., 0]
    nrm = height_to_normal(h, strength=4.0)
    st = channel_stats(nrm)
    assert st["is_correct_order"], f"Z 跑到了 {st['z_channel']} 通道"


def test_pipeline_all_maps():
    res = generate(_dummy_color(), PBRParams(normal_strength=4.0))
    st = channel_stats(res.maps["NRM"])
    assert st["is_correct_order"]
    for suf in ("RGH", "MET", "OCC", "GLS", "CUR"):
        a = res.maps[suf]
        assert a.min() >= 0.0 and a.max() <= 1.0, f"{suf} 超出 0..1"


def test_packed_alpha_full():
    h = _dummy_color()[..., 0]
    rgba = pack(h, h, h)
    assert rgba[..., 3].min() == 255, "打包图 alpha 没填满"


def test_repair_pink():
    """把正确法线故意反转通道，再修回来。"""
    h = _dummy_color()[..., 0]
    good = height_to_normal(h, strength=4.0)
    pink = good[..., ::-1].copy()
    assert not channel_stats(pink)["is_correct_order"]
    fixed = repair_normal(pink)
    assert channel_stats(fixed)["is_correct_order"]


def test_export_roundtrip():
    res = generate(_dummy_color(), PBRParams(normal_strength=4.0))
    with tempfile.TemporaryDirectory() as d:
        results = export_maps(res.maps, d, "T_Test", ["NRM", "OCC"])
        assert all(r.ok for r in results), [r.note for r in results]
        assert os.path.exists(os.path.join(d, "T_Test_NRM.png"))


def test_export_rejects_broken_normal():
    """通道反转的法线必须被拦下来，不允许落盘。"""
    h = _dummy_color()[..., 0]
    pink = height_to_normal(h, strength=4.0)[..., ::-1].copy()
    with tempfile.TemporaryDirectory() as d:
        try:
            export_maps({"NRM": pink}, d, "bad")
        except ValueError:
            return
    raise AssertionError("错误的法线居然被写盘了")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    raise SystemExit(1 if failed else 0)
