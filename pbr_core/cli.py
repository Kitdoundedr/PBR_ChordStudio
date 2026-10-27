"""命令行入口：不打开 GUI 也能生成 / 校验 / 修复。

用法：
    python -m pbr_core.cli gen   -i color.png -o out --name T_Rock_A01
    python -m pbr_core.cli check -i normal.png
    python -m pbr_core.cli fix   -i pink_normal.png -o fixed/
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

from .exporter import SUFFIXES, export_maps
from .imgio import load_rgb, save_image
from .normal import channel_stats, is_pink, repair_normal
from .pipeline import PBRParams, generate


def _print_stats(name: str, arr: np.ndarray) -> None:
    st = channel_stats(arr)
    print(f"  {name:<6} R={st['mean'][0] * 255:5.1f} G={st['mean'][1] * 255:5.1f} "
          f"B={st['mean'][2] * 255:5.1f}   Z轴在 {st['z_channel']} "
          f"{'OK' if st['is_correct_order'] else '<-- 异常'}")


def cmd_gen(args) -> int:
    rgb = load_rgb(args.input)
    params = PBRParams(
        normal_strength=args.strength,
        normal_blur=args.blur,
        normal_flip_y=args.flip_y,
        normal_from=args.normal_from,
        ao_radius=args.ao_radius,
        rough_base=args.rough,
    )
    height = load_rgb(args.height) if args.height else None
    if height is not None and height.ndim == 3:
        height = height.mean(axis=2)

    res = generate(rgb, params, height=height)

    stem = args.name or os.path.splitext(os.path.basename(args.input))[0]
    suffixes = SUFFIXES.keys() if args.set == "all" else args.set.split(",")
    results = export_maps(res.maps, args.out, stem, list(suffixes))

    print(f"输出目录: {os.path.abspath(args.out)}")
    bad = 0
    for r in results:
        flag = "OK " if r.ok else "BAD"
        print(f"  [{flag}] {os.path.basename(r.path):<44} {r.note}")
        bad += 0 if r.ok else 1
    print(f"共 {len(results)} 张，异常 {bad} 张")
    return 0 if bad == 0 else 1


def cmd_check(args) -> int:
    for p in args.input:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        st = channel_stats(rgb)
        print(f"\n{p}")
        _print_stats("数值", rgb)
        if st["is_flat"]:
            print("  判定: 平坦法线（没有细节）")
        elif st["is_correct_order"]:
            print("  判定: 正常，Z 轴在 B 通道")
        elif is_pink(rgb):
            print("  判定: 粉色法线 —— Z 轴跑到了 R 通道（通道顺序被反转）")
            print("  修复: python -m pbr_core.cli fix -i <文件> -o <目录>")
        else:
            print("  判定: 通道顺序异常")
    return 0


def cmd_fix(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    for p in args.input:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        fixed = repair_normal(rgb, swap_rb=None if not args.swap else True,
                              flip_g=args.flip_g, rebuild_z=args.rebuild_z)
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(args.out, stem + args.suffix + ".png")
        save_image(fixed, out)
        st = channel_stats(fixed)
        print(f"{os.path.basename(p)} -> {out}")
        _print_stats("修复后", fixed)
        if not st["is_correct_order"]:
            print("  [警告] 修复后仍异常，请加 --rebuild-z 重算 B 通道")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pbr_core", description="PBR 贴图生成 / 校验 / 修复")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="由基础色生成整套贴图")
    g.add_argument("-i", "--input", required=True)
    g.add_argument("-o", "--out", default="out")
    g.add_argument("--name")
    g.add_argument("--set", default="NRM,RGH,MET,OCC,DIS,CUR")
    g.add_argument("--height", default=None, help="改用外部高度图")
    g.add_argument("--normal-from", dest="normal_from", default="color", choices=["color", "height"])
    g.add_argument("--strength", type=float, default=3.0)
    g.add_argument("--blur", type=float, default=1.0)
    g.add_argument("--flip-y", dest="flip_y", action="store_true", help="输出 OpenGL 约定(翻转绿通道)")
    g.add_argument("--ao-radius", dest="ao_radius", type=float, default=12.0)
    g.add_argument("--rough", type=float, default=0.55)
    g.set_defaults(func=cmd_gen)

    c = sub.add_parser("check", help="检查法线贴图通道顺序")
    c.add_argument("-i", "--input", nargs="+", required=True)
    c.set_defaults(func=cmd_check)

    f = sub.add_parser("fix", help="修复粉色 / 通道反转的法线")
    f.add_argument("-i", "--input", nargs="+", required=True)
    f.add_argument("-o", "--out", default="fixed")
    f.add_argument("--swap", action="store_true", help="强制交换 R/B")
    f.add_argument("--flip-g", dest="flip_g", action="store_true")
    f.add_argument("--rebuild-z", dest="rebuild_z", action="store_true")
    f.add_argument("--suffix", default="_fixed")
    f.set_defaults(func=cmd_fix)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
