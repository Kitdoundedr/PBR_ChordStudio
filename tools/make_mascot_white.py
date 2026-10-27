"""纯白背景视频 -> 透明看板娘动画（WebP）。

跟 make_mascot.py 的区别，都是为了这种「棚拍白底」素材准备的：

1. **方形补边推理**：竖屏 9:16 直接压成 1024×1024 会让模型看到变形的人体，
   这里先用白底把画面补成正方形再送进 ISNet-anime，跑完再把 alpha 裁回来。
2. **白底去污染**：白背景的抗锯齿边缘像素其实是 ``C = a·F + (1-a)·255``，
   直接拿来当半透明色贴到深色面板上就是一圈白毛边。反解出真实前景色
   ``F = (C - (1-a)) / a``，边缘就干净了。
3. **统一裁切**：14 秒里人物会挥手、位置有微动。所有帧共用同一个包围盒
   （取并集 + 留白），动画才不会左右抖动；最后按比例塞进目标框，人物不变形。

用法：
    python tools/make_mascot_white.py \
        --video "C:/.../竖屏14秒挥手加站姿-纯白背景.mp4" \
        --out assets/mascot_white.webp --fps 12 --height 720 --box 144x256
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_mascot import DEFAULT_MODEL, ffmpeg_exe, load_model, video_info  # noqa: E402

MODEL_INPUT = 1024


# ------------------------------------------------------------------ 抽帧


def extract(video: str, workdir: str, fps: int, height: int) -> list[str]:
    w, h = video_info(video)["size"]
    if h <= 0:
        h = 1920
    k = height / float(h)
    ow, oh = max(2, int(round(w * k)) // 2 * 2), max(2, int(round(h * k)) // 2 * 2)
    cmd = [ffmpeg_exe(), "-hide_banner", "-v", "error", "-y", "-i", video,
           "-vf", f"fps={fps},scale={ow}:{oh}:flags=lanczos",
           os.path.join(workdir, "f_%04d.png")]
    subprocess.run(cmd, check=True)
    return sorted(glob.glob(os.path.join(workdir, "f_*.png")))


# ------------------------------------------------------------------ 抠图


def alpha_square(sess, rgb: Image.Image, threshold: float) -> np.ndarray:
    """白底补正方形 -> ISNet 推理 -> 裁回原尺寸的 alpha。"""
    w, h = rgb.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(rgb, ((s - w) // 2, (s - h) // 2))

    big = canvas.resize((MODEL_INPUT, MODEL_INPUT), Image.BICUBIC)
    x = np.asarray(big, dtype=np.float32)[:, :, ::-1] / 127.5 - 1.0   # BGR, [-1,1]
    x = np.transpose(x, (2, 0, 1))[None, ...]

    m = np.asarray(sess.run(None, {"img": x})[0], dtype=np.float32)[0, 0]
    lo, hi = float(m.min()), float(m.max())
    if hi - lo > 1e-6:
        m = (m - lo) / (hi - lo)

    a = Image.fromarray((m * 255).astype(np.uint8)).resize((s, s), Image.BILINEAR)
    a = np.asarray(a, dtype=np.float32) / 255.0
    ox, oy = (s - w) // 2, (s - h) // 2
    a = a[oy:oy + h, ox:ox + w]
    if threshold > 0:
        a[a < threshold] = 0.0
    return a


def decontaminate(rgb: Image.Image, alpha: np.ndarray) -> Image.Image:
    """去掉白背景混进边缘的颜色（反解 straight alpha 前景色）。"""
    c = np.asarray(rgb, dtype=np.float32) / 255.0
    a = alpha[..., None]
    fg = np.where(a > 0.004, (c - (1.0 - a)) / np.maximum(a, 0.004), 0.0)
    out = np.concatenate([np.clip(fg, 0.0, 1.0) * 255.0, a * 255.0], axis=2)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGBA")


def union_bbox(frames: list[Image.Image], cutoff: int = 24) -> tuple[int, int, int, int]:
    """所有帧 alpha 的并集包围盒，避免动画期间人物被裁掉。"""
    acc = None
    for im in frames:
        m = np.asarray(im.split()[3]) > cutoff
        acc = m if acc is None else (acc | m)
    ys, xs = np.where(acc)
    if len(xs) == 0:
        w, h = frames[0].size
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ------------------------------------------------------------------ 输出


def fit(frames: list[Image.Image], box: tuple[int, int], pad: float) -> list[Image.Image]:
    """统一裁切后按比例塞进 box，人物不变形，四周透明留白。"""
    bw, bh = box
    x0, y0, x1, y1 = union_bbox(frames)
    # 留一点呼吸空间，边缘不贴死
    mw = int((x1 - x0) * pad)
    mh = int((y1 - y0) * pad)
    x0, y0 = max(0, x0 - mw), max(0, y0 - mh)
    x1 = min(frames[0].width, x1 + mw)
    y1 = min(frames[0].height, y1 + mh)

    cw, ch = x1 - x0, y1 - y0
    k = min(bw / cw, bh / ch)
    tw, th = max(1, int(cw * k)), max(1, int(ch * k))
    ox, oy = (bw - tw) // 2, (bh - th) // 2

    out: list[Image.Image] = []
    for im in frames:
        crop = im.crop((x0, y0, x1, y1)).resize((tw, th), Image.LANCZOS)
        canvas = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        canvas.paste(crop, (ox, oy))
        out.append(canvas)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="assets/mascot_white.webp")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--height", type=int, default=720, help="抠图时的工作高度")
    ap.add_argument("--box", default="144x256", help="最终动画尺寸，如 144x256")
    ap.add_argument("--pad", type=float, default=0.02, help="包围盒外扩比例")
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--quality", type=int, default=86)
    ap.add_argument("--dump-frames", default="", help="可选：把成品逐帧存成 PNG 目录")
    args = ap.parse_args()

    bw, bh = (int(v) for v in args.box.lower().split("x"))
    info = video_info(args.video)
    print(f"视频 {info['size'][0]}x{info['size'][1]}  {info['duration']:.1f}s -> {bw}x{bh} @ {args.fps}fps")

    workdir = tempfile.mkdtemp(prefix="mascot_white_")
    try:
        files = extract(args.video, workdir, args.fps, args.height)
        if not files:
            print("没抽到帧")
            return 1
        print(f"[1/3] 抽到 {len(files)} 帧")

        sess = load_model(args.model)
        frames: list[Image.Image] = []
        for i, p in enumerate(files, 1):
            rgb = Image.open(p).convert("RGB")
            a = alpha_square(sess, rgb, args.threshold)
            frames.append(decontaminate(rgb, a))
            if i % 20 == 0 or i == len(files):
                print(f"[2/3] 抠图 {i}/{len(files)}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    frames = fit(frames, (bw, bh), args.pad)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    frames[0].save(args.out, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / args.fps)), loop=0,
                   quality=args.quality, method=4, allow_mixed=False)
    print(f"[3/3] 输出 {bw}x{bh}  {len(frames)} 帧  "
          f"{os.path.getsize(args.out) / 1024 / 1024:.2f} MB -> {args.out}")

    if args.dump_frames:
        os.makedirs(args.dump_frames, exist_ok=True)
        for i, im in enumerate(frames):
            im.save(os.path.join(args.dump_frames, f"n_{i:04d}.png"))
        print("逐帧已存 ->", args.dump_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
