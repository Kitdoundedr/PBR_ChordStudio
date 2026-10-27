"""把动画视频抠成透明看板娘，输出 UI 用的循环动画。

用 ISNet-anime (E:\\AiMesh\\isnet-anime.onnx) 逐帧抠图，输出透明背景的
GIF / APNG，塞进左下角卡片循环播放。

用法：
    python tools/make_mascot.py --video animation.mp4 --out assets/mascot.gif
    python tools/make_mascot.py --video animation.mp4 --out assets/mascot.png --format apng

流程：ffmpeg 抽帧到临时目录 -> 逐帧推理 alpha -> 缩放合成 -> 存动画
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_MODEL = r"E:\AiMesh\isnet-anime.onnx"
MODEL_INPUT = 1024


# ------------------------------------------------------------------ ffmpeg


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        raise SystemExit("需要 ffmpeg：pip install imageio-ffmpeg  (" + str(e) + ")")


def video_info(video: str) -> dict:
    """从 ffmpeg 输出里抓时长和分辨率。"""
    exe = ffmpeg_exe()
    err = subprocess.run([exe, "-hide_banner", "-i", video],
                         capture_output=True).stderr.decode("utf-8", "ignore")
    info = {"duration": 0.0, "size": (0, 0)}

    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", err)
    if m:
        hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info["duration"] = hh * 3600 + mm * 60 + ss

    m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", err)
    if m:
        info["size"] = (int(m.group(1)), int(m.group(2)))
    return info


def _even(n: float) -> int:
    return max(2, int(round(n / 2)) * 2)


def extract_frames(video: str, workdir: str, fps: int, max_frames: int, side: int) -> list[str]:
    """抽帧成 PNG，返回按顺序排列的路径列表。"""
    exe = ffmpeg_exe()
    info = video_info(video)
    w, h = info["size"] if info["size"][0] else (512, 512)
    k = min(1.0, side / max(w, h))
    ow, oh = _even(w * k), _even(h * k)

    cmd = [exe, "-hide_banner", "-v", "error", "-y", "-i", video,
           "-vf", f"fps={fps},scale={ow}:{oh}:flags=lanczos",
           "-frames:v", str(max_frames),
           os.path.join(workdir, "f_%04d.png")]
    subprocess.run(cmd, check=True)
    return sorted(glob.glob(os.path.join(workdir, "f_*.png")))


# ------------------------------------------------------------------ 抠图


def load_model(path: str):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    avail = ort.get_available_providers()
    prefer = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider") if p in avail]
    return ort.InferenceSession(path, so, providers=prefer + ["CPUExecutionProvider"])


def predict_alpha(sess, rgb_frame: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """单帧推理，返回 HxW float alpha [0,1]。

    ISNet-anime：输入 (1,3,1024,1024) float32、BGR、归一化到 [-1,1]，
    输出单通道 mask。alpha 小于 threshold 的直接压成 0，去掉边缘薄雾。
    """
    h, w = rgb_frame.shape[:2]
    im = Image.fromarray(rgb_frame).convert("RGB").resize(
        (MODEL_INPUT, MODEL_INPUT), Image.BICUBIC)
    x = np.asarray(im, dtype=np.float32)[:, :, ::-1]          # RGB -> BGR
    x = x / 127.5 - 1.0
    x = np.transpose(x, (2, 0, 1))[None, ...]

    mask = sess.run(None, {"img": x})[0]
    a = np.asarray(mask, dtype=np.float32)[0, 0]
    lo, hi = float(a.min()), float(a.max())
    if hi - lo > 1e-6:
        a = (a - lo) / (hi - lo)

    a8 = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    alpha = np.asarray(a8, dtype=np.float32) / 255.0
    if threshold > 0:
        alpha[alpha < threshold] = 0.0
    return alpha


# ------------------------------------------------------------------ 输出


def write_animation(frames: list[Image.Image], out: str, fmt: str, fps: int) -> None:
    dur = int(round(1000 / fps))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    if fmt == "apng":
        frames[0].save(out, format="PNG", save_all=True, append_images=frames[1:],
                       duration=dur, loop=0)
        return

    # GIF 只支持 1bit 透明：把 alpha 轻微羽化，边缘不刺眼
    soft = []
    for im in frames:
        im = im.convert("RGBA")
        a = im.split()[3].filter(ImageFilter.GaussianBlur(0.7))
        im.putalpha(a)
        soft.append(im)
    soft[0].save(out, format="GIF", save_all=True, append_images=soft[1:],
                 duration=dur, loop=0, disposal=2, optimize=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="assets/mascot.gif")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-frames", dest="max_frames", type=int, default=240)
    ap.add_argument("--proc-side", dest="proc_side", type=int, default=640,
                    help="抠图时使用的分辨率（越大越准越慢）")
    ap.add_argument("--max-side", dest="max_side", type=int, default=320,
                    help="最终动画的最大边长")
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--format", choices=["gif", "apng"], default=None)
    ap.add_argument("--no-matting", dest="no_matting", action="store_true")
    args = ap.parse_args()

    fmt = args.format or ("apng" if args.out.lower().endswith((".png", ".apng")) else "gif")
    info = video_info(args.video)
    print(f"视频 {info['size'][0]}x{info['size'][1]}  时长 {info['duration']:.1f}s  -> {fmt}")

    workdir = tempfile.mkdtemp(prefix="mascot_")
    try:
        files = extract_frames(args.video, workdir, args.fps, args.max_frames, args.proc_side)
        print(f"[1/3] 抽到 {len(files)} 帧（{args.proc_side}px）")
        if not files:
            print("没抽到帧，检查视频路径")
            return 1

        sess = None if args.no_matting else load_model(args.model)
        if sess is not None:
            print(f"[2/3] 开始抠图（{len(files)} 帧，CPU 推理）")

        frames: list[Image.Image] = []
        for p in files:
            im = Image.open(p).convert("RGB")
            if sess is not None:
                a = predict_alpha(sess, np.asarray(im), args.threshold)
                rgba = im.convert("RGBA")
                rgba.putalpha(Image.fromarray((a * 255 + 0.5).astype(np.uint8)))
                im = rgba
            frames.append(im)

        w, h = frames[0].size
        k = min(1.0, args.max_side / max(w, h))
        tw, th = _even(w * k), _even(h * k)
        if (tw, th) != (w, h):
            frames = [f.resize((tw, th), Image.LANCZOS) for f in frames]
        print(f"[3/3] 输出 {tw}x{th}  {len(frames)} 帧 -> {args.out}")

        write_animation(frames, args.out, fmt, args.fps)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"完成：{args.out}  {mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
