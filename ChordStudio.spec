# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（onedir，免控制台窗口）。

    .venv\\Scripts\\pyinstaller ChordStudio.spec --noconfirm

为什么用 onedir 而不是 onefile：
  onefile 每次启动都要把整个 Qt 解压到 %TEMP%，冷启动 3~5 秒，还会在 C 盘写临时目录；
  onedir 直接跑现成的 dll，启动 1 秒出头，绿色免安装（整个文件夹拷走即可）。
"""

import os

ROOT = SPECPATH

# 只带真正用到的素材。assets 里的 mascot.png(13M 废弃) / mascot.mp4 / mascot.gif 不进包
DATAS = [
    (os.path.join(ROOT, "assets", "bg.jpg"), "assets"),
    (os.path.join(ROOT, "assets", "mascot_white.webp"), "assets"),
    (os.path.join(ROOT, "assets", "app.ico"), "assets"),
]

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=[],
    datas=DATAS,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # onnxruntime / tkinter / 测试框架只有 tools/ 和 tests/ 用，界面本身不需要
    excludes=["onnxruntime", "tkinter", "unittest", "pytest", "pydoc_data"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChordStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=os.path.join(ROOT, "assets", "app.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ChordStudio",
)
