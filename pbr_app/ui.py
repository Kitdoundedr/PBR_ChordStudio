"""PBR 材质分解工作台 —— 布局严格对齐参考图（ChordStudio/界面预览.png）。

    ┌ 顶栏  ● ChordStudio · CHORD PBR 材质分解                      [拖放图片到软件中] ┐
    │      [贴图预览] [通道打包]           CHORD · Ublsoft   本地离线                  │
    ├ 左栏 152 ─────────┬ 右区（跟随 tab 切换）────────────────────────────────────────┤
    │ ChordStudio       │ ① 贴图预览：左「小图网格 + 参数 + 导出」 | 右「大图预览」      │
    │  CHORD PBR 材质分解 │  ┌ 通道 ────────┐  ┌ 大图 ──────────────────────┐        │
    │  [选择图片]        │  │ ▫ ▫ ▫ ▫      │  │  COL 基础色      1024×1024 │        │
    │  [生成贴图]        │  │ ▫ ▫ ▫ ▫      │  │                            │        │
    │  [导出全部通道]     │  │ ▫ ▫ ▫ ▫      │  │      选中的通道铺满        │        │
    │  ── 推理参数 ──    │  ├ 调整 ────────┤  │                            │        │
    │  分辨率 − 1024 +   │  │ 强度 ──●──   │  └────────────────────────────┘        │
    │  粗糙度步长 − 5 +  │  │ 对比度 ─●──  │                                        │
    │  看板娘（占满剩余）  │  ├ 导出 ────────┤   ② 通道打包：混合选项 + 3D | 合成图   │
    └────────────────────┴──└──────────────┘──────────────────────────────────────────┘

两条导出路径（对应顶部两个 tab）：
    · 贴图预览 —— 单图导出。左栏下部 [导出当前通道] / [导出勾选的 N 张]
    · 通道打包 —— 合成导出。按 R/G/B/A 打包成一张 RGBA（_MASK）
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QGuiApplication,
    QIcon,
    QImage,
    QImageReader,
    QLinearGradient,
    QMovie,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pbr_core.exporter import export_maps
from pbr_core.imgio import gaussian_blur, load_rgb, resize, save_image
from pbr_core.normal import channel_stats, repair_normal
from pbr_core.packing import pack
from pbr_core.pipeline import PBRParams, generate

# ------------------------------------------------------------------ 资源与主题

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS = os.path.join(_ROOT, "assets")
_BG_PATH = next(
    (p for p in (os.path.join(_ASSETS, "bg.jpg"), os.path.join(_ASSETS, "bg_blur.jpg"))
     if os.path.exists(p)), os.path.join(_ASSETS, "bg.jpg"))
_MASCOT_PATH = next(
    (p for p in (os.path.join(_ASSETS, "mascot_white.webp"),
                 os.path.join(_ASSETS, "mascot.webp"),
                 os.path.join(_ASSETS, "mascot.gif"))
     if os.path.exists(p)), os.path.join(_ASSETS, "mascot.gif"))
_ICON_PATH = os.path.join(_ASSETS, "app.ico")

ACCENT = "#8b6cf0"
ACCENT2 = "#6a4fd8"
GOOD = "#54d1a0"
BAD = "#ff6b81"
DIM = "#b8b0d2"

CARD = ("qlineargradient(x1:0,y1:0,x2:0,y2:1, "
        "stop:0 rgba(60,54,80,0.82), stop:1 rgba(34,29,47,0.76))")
CARD_LINE = "rgba(255,255,255,0.24)"

STYLE = """
QWidget { color:#ece8f6; font-family:'Microsoft YaHei UI','Segoe UI'; font-size:12px; background:transparent; }
QFrame#card { background:%(card)s; border:1px solid %(line)s; border-radius:12px; }
QFrame#header { background:%(card)s; border:1px solid %(line)s; border-radius:12px; }
QFrame#chanCard { background:rgba(255,255,255,0.07); border:1px solid %(line)s; border-radius:10px; }
QFrame#chanCard[picked="true"] { border:1px solid rgba(255,255,255,0.60); }
QFrame#thumb { background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.20);
               border-radius:8px; }
QFrame#thumb:hover { background:rgba(255,255,255,0.16); border-color:rgba(255,255,255,0.48); }
QFrame#thumb[picked="true"] { border-color:rgba(255,255,255,0.36); }
QFrame#thumb[sel="true"] { background:rgba(255,255,255,0.26); border:1px solid rgba(255,255,255,0.95); }
QLabel#cardTitle { color:#e6e0f5; font-weight:600; font-size:12px; }
QLabel#hint { color:%(dim)s; font-size:11px; }
QPushButton { background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.24);
              border-radius:7px; padding:6px 10px; color:#eee9f7; }
QPushButton:hover { background:rgba(255,255,255,0.22); border-color:rgba(255,255,255,0.48); }
QPushButton:pressed { background:rgba(255,255,255,0.35); }
QPushButton#accent { background:rgba(255,255,255,0.90); border:1px solid rgba(255,255,255,0.96);
                     color:#211a2f; font-weight:600; padding:7px 12px; }
QPushButton#accent:hover { background:#ffffff; }
QPushButton#accent:pressed { background:rgba(255,255,255,0.76); }
QPushButton#step { padding:2px 6px; border-radius:5px; font-size:13px; }
QPushButton#tiny { padding:3px 8px; font-size:11px; border-radius:6px; }
QPushButton#tab { border:1px solid transparent; border-radius:6px; padding:5px 14px;
                  background:transparent; color:#b5accd; }
QPushButton#tab:hover { background:rgba(255,255,255,0.14); }
QPushButton#tab:checked { background:rgba(255,255,255,0.93); border-color:#ffffff;
                          color:#1d1729; font-weight:600; }
QComboBox { background:rgba(255,255,255,0.09); border:1px solid rgba(255,255,255,0.24);
            border-radius:6px; padding:3px 8px; color:#ece8f6; }
QComboBox:hover { border-color:rgba(255,255,255,0.46); }
QComboBox::drop-down { border:0; width:16px; }
QComboBox QAbstractItemView { background:#2a2438; color:#ece8f6; border:1px solid %(line)s;
                             selection-background-color:%(a2)s; }
QSlider::groove:horizontal { height:4px; background:rgba(255,255,255,0.22); border-radius:2px; }
QSlider::sub-page:horizontal { background:rgba(255,255,255,0.80); border-radius:2px; }
QSlider::handle:horizontal { width:11px; margin:-4px 0; border-radius:5px; background:#fff; }
QPlainTextEdit { background:rgba(14,11,21,0.70); border:1px solid rgba(255,255,255,0.16);
                 border-radius:8px; color:#c2bbd6; font-family:Consolas,monospace; font-size:11px; }
QScrollArea { border:0; background:transparent; }
QScrollArea > QWidget > QWidget { background:transparent; }
QScrollBar:vertical { background:transparent; width:8px; }
QScrollBar::handle:vertical { background:rgba(255,255,255,0.30); border-radius:4px; min-height:26px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:transparent; }
QScrollBar:horizontal { height:0; }
QMenuBar { background:transparent; color:#d3cbeb; }
QMenuBar::item:selected { background:rgba(255,255,255,0.18); border-radius:4px; }
QMenu { background:#2a2438; color:#ece8f6; border:1px solid %(line)s; }
QMenu::item:selected { background:%(a2)s; }
""" % {"card": CARD, "line": CARD_LINE, "a1": ACCENT, "a2": ACCENT2, "dim": DIM}

DEFAULT_MIX = {"R": "RGH", "G": "DIS", "B": "OCC", "A": "MET"}

PRESET_MIX = {
    "RHAM": {"R": "RGH", "G": "DIS", "B": "OCC", "A": "MET"},
    "ORM": {"R": "OCC", "G": "RGH", "B": "MET", "A": "—"},
    "RMA": {"R": "RGH", "G": "MET", "B": "OCC", "A": "—"},
    "MRA": {"R": "MET", "G": "RGH", "B": "OCC", "A": "—"},
    "HDRP Mask Map": {"R": "MET", "G": "OCC", "B": "DIS", "A": "—"},
}

MIX_ITEMS = [
    ("—", "空"), ("COL", "基础色"), ("RGH", "粗糙度"), ("MET", "金属度"),
    ("OCC", "环境吸收"), ("DIS", "高度"), ("CUR", "曲率"), ("GLS", "光泽度"),
    ("SHN", "高光"), ("EMI", "自发光"),
]

# 通道卡片，顺序对齐参考图（COL / NRM 优先）
CHANNELS: list[tuple[str, str]] = [
    ("COL", "基础色"), ("NRM", "法线"), ("RGH", "粗糙度"), ("MET", "金属度"),
    ("OCC", "环境吸收"), ("DIS", "高度"), ("CUR", "曲率"), ("GLS", "光泽度"),
    ("SHN", "高光"), ("EMI", "自发光"), ("TRS", "透射"), ("MASK", "合成"),
]

# 默认参与「导出全部通道」的卡片
CHECKED_BY_DEFAULT = {"COL", "NRM", "RGH", "MET", "OCC", "DIS", "CUR", "GLS"}

# 顶部格式条：顺序即按钮顺序
FORMATS = ["PNG", "TGA", "WebP", "JPG", "ICO", "BMP", "GIF", "TIF", "DDS"]
FMT_EXT = {
    "PNG": ".png", "TGA": ".tga", "WebP": ".webp", "JPG": ".jpg", "ICO": ".ico",
    "BMP": ".bmp", "GIF": ".gif", "TIF": ".tif", "DDS": ".dds",
}
# 有损格式：数据贴图用它导出会被压坏，导出时要在日志里提醒
FMT_LOSSY = {"JPG", "WebP", "GIF"}
# 存不了 alpha 的格式
FMT_NO_ALPHA = {"JPG", "BMP", "GIF", "ICO"}

# 大图下方的说明，让人一眼知道自己看的是什么数据（中文, English）
CHANNEL_HINT: dict[str, tuple[str, str]] = {
    "COL": ("基础色 BaseColor · 源图颜色本身，不带光照",
            "BaseColor · albedo from the source image, no lighting baked in"),
    "NRM": ("法线 Normal · R=nx、G=ny、B=nz，平坦处约 128 / 128 / 255",
            "Normal · R=nx, G=ny, B=nz; flat areas read about 128 / 128 / 255"),
    "RGH": ("粗糙度 Roughness · 越亮越粗糙，白=全漫反射",
            "Roughness · brighter means rougher; white is fully diffuse"),
    "MET": ("金属度 Metallic · 只有接近 0 或 1 的强对比",
            "Metallic · strong contrast, values should sit near 0 or 1"),
    "OCC": ("环境吸收 AO · 凹陷与夹缝处变暗",
            "Ambient Occlusion · cavities and creases get darker"),
    "DIS": ("高度 Height · 灰度，可直接接置换",
            "Height · grayscale displacement data"),
    "CUR": ("曲率 Curvature · 边缘高亮，用来做边缘磨损",
            "Curvature · edges brighten, handy for edge wear"),
    "GLS": ("光泽度 Glossiness · 粗糙度反相，给部分老管线用",
            "Glossiness · inverse of roughness for legacy pipelines"),
    "SHN": ("高光 Specular · 从亮部提的高光遮罩",
            "Specular · highlight mask pulled from bright areas"),
    "EMI": ("自发光 Emissive · 从最亮区域提取",
            "Emissive · extracted from the brightest regions"),
    "TRS": ("透射 Transmission · 越薄越通透",
            "Transmission · thinner areas transmit more"),
    "MASK": ("合成 Mask · 彩色是正常的：R / G / B 各装一张灰度图；A 里是数据不是透明度，"
             "这里按不透明显示，要改回打包页换分配",
             "Packed Mask · the colors are expected: R/G/B each hold a grayscale map. "
             "A carries data, not opacity, so it is shown opaque here — "
             "change the assignment on the Packing tab"),
}

# 界面语言切换用：中文原文 -> English。查不到的（格式名、通道代号）原样保留。
_EN: dict[str, str] = {
    # 顶栏 / 左栏
    "· CHORD PBR 材质分解": "· PBR Material Decomposer",
    "拖放图片到软件中": "Drop image here",
    "也可以点这里选择图片": "Click to pick an image",
    "贴图预览": "Maps",
    "通道打包": "Packing",
    "本地离线": "Offline",
    "CHORD PBR 材质分解": "PBR Material Decomposer",
    "选择图片": "Load Image",
    "生成贴图": "Generate",
    "导出全部通道": "Export All",
    "导出带圆点勾选的所有通道（单图）": "Export every channel marked with a dot",
    "推理参数": "Inference",
    "分辨率": "Size",
    "粗糙度步长": "Rough",
    "越大越快，精度越低": "Bigger = faster, coarser",
    "法线 Y": "Nrm Y",
    "Dx = UE / Unity · Gl = OpenGL": "Dx = UE / Unity · Gl = OpenGL",
    "法线修复": "Fix Normal",
    "修掉导出的粉色法线（通道反转）": "Repair a pink normal map (swapped channels)",
    "导出格式": "Format",
    # 预览页
    "通道": "Channels",
    "源图": "Source",
    "点小图看大图 · 点右上角圆点决定是否参与「导出全部通道」":
        "Click a thumb to preview · the dot toggles batch export",
    "调整": "Adjust",
    "重置": "Reset",
    "把这一个通道的强度 / 对比度退回 1.00": "Reset gain / contrast of this channel to 1.00",
    "强度": "Gain",
    "对比度": "Contrast",
    "强度 = 细节量；对比度 = 明暗反差。法线会自动保值，不会变粉。":
        "Gain = amount of detail, contrast = tonal spread. Normals stay valid.",
    "导出": "Export",
    "格式": "Format",
    "导出当前通道": "Export This Map",
    "把右边大图这张按当前格式写盘": "Write the map shown on the right using the current format",
    "导出勾选的 8 张": "Export 8 checked",
    "导出勾选的通道": "Export checked",
    "点击选择导出目录": "Click to change the output folder",
    "点左侧小图查看大图": "Click a thumbnail to preview",
    "先选择一张贴图，这里会显示该通道的结果": "Load a texture to see this channel",
    "原始输入贴图，所有通道都由它推出来": "The source image every channel is derived from",
    "源图 · 不可调整": "Source · read only",
    # 打包页
    "混合选项": "Channel Packing",
    "按 R/G/B/A 把多张通道打包成一张 RGBA": "Pack channels into a single RGBA image",
    "3D 球照这套分配拆回来预览：换预设球没变 = 通道没丢；"
    "球变了 = 有通道没打进去。":
        "The 3D ball unpacks the result with the same rules: "
        "unchanged after a preset switch = nothing lost; a visible change = a channel is missing.",
    "空 —": "Empty —",
    "预设": "Presets",
    "圆点是整张通道打包的标准做法：ORM 给 UE，"
    "Mask Map 给 HDRP。": "Packed masks are the standard: ORM for UE, Mask Map for HDRP.",
    "3D 材质预览": "3D Preview",
    "3D 打包预览": "3D Pack Preview",
    "左键拖拽转光源 · 滚轮缩放": "Drag to move the light · wheel to zoom",
    "基础色 · 法线 · 粗糙度 · 金属度 · AO": "BaseColor · Normal · Rough · Metal · AO",
    "合成后的材质图": "Packed Result",
    "合成结果会显示在这里": "The packed result shows here",
    "导出 RGBA 贴图": "Export Packed RGBA",
    "单图 + 合成 一起导出": "Export Both",
    "A 写入": "A write",
    "数据": "Data",
    "不透明": "Opaque",
    # 通道名
    "基础色": "BaseColor",
    "法线": "Normal",
    "粗糙度": "Roughness",
    "金属度": "Metallic",
    "环境吸收": "AO",
    "高度": "Height",
    "曲率": "Curvature",
    "光泽度": "Glossiness",
    "高光": "Specular",
    "自发光": "Emissive",
    "透射": "Transmission",
    "合成": "Mask",
    "空": "Empty",
    # 菜单 / 对话框
    "工具": "Tools",
    "帮助": "Help",
    "关于法线通道顺序": "About normal channel order",
    "导出目录…": "Output folder…",
    "法线修复（粉色 / 通道反转）…": "Repair normal (pink / swapped)…",
    "还没生成": "Nothing generated",
    "请先选择一张贴图。": "Load a texture first.",
    "通道为空": "Channel is empty",
    "没有勾选": "Nothing checked",
    "在通道卡片右上角点圆点选中要导出的通道。": "Click the dot on a thumbnail to include it.",
    "单图导出失败": "Export failed",
    "导出失败": "Export failed",
    "合成导出失败": "Packing export failed",
    "有问题": "Problems",
    "完成": "Done",
    "选择基础色贴图": "Choose a base color texture",
    "选择导出目录": "Choose output folder",
    "选择要修复的法线贴图": "Choose a normal map to repair",
}

RESOLUTIONS = [1024, 2048, 4096, 8192]
ROUGH_STEPS = [1, 2, 3, 4, 5, 6, 7, 8]

# 分辨率显示成 1024×1024（2 的幂方图），不是 1024 这种裸数字
RES_LABELS = ["%d×%d" % (v, v) for v in RESOLUTIONS]


# ------------------------------------------------------------------ 图像工具


def _downsample(a: np.ndarray, max_side: int) -> np.ndarray:
    """整数倍面积平均降采样。

    QImage.scaled(SmoothTransformation) 实际是双线性，缩到 1/7 时高频细节会
    直接走样成一片雪花 —— 法线贴图缩略图最容易踩这个坑。这里先按整数倍
    做块平均（等价 mipmap），再交给 Qt 做最后一次小比例缩放。
    """
    h, w = a.shape[:2]
    f = int(np.ceil(max(h, w) / float(max_side)))
    if f <= 1:
        return a
    hh, ww = (h // f) * f, (w // f) * f
    if hh < f or ww < f:
        return a
    src = a[:hh, :ww].astype(np.float32)
    if src.ndim == 2:
        return src.reshape(hh // f, f, ww // f, f).mean(axis=(1, 3))
    c = src.shape[2]
    return src.reshape(hh // f, f, ww // f, f, c).mean(axis=(1, 3))


def arr_to_qimage(arr: np.ndarray, max_side: int = 768) -> QImage:
    """numpy -> QImage。float 按 [0,1] 处理，uint8 直接用。通道固定 RGB / RGBA。"""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=2)
    # 先统一到 0..255 的浮点，再做面积平均，最后落成 uint8
    if a.dtype != np.uint8:
        a = np.clip(np.asarray(a, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    if max(a.shape[:2]) > max_side:
        a = _downsample(a, max_side)
    a = np.clip(np.asarray(a, dtype=np.float32) + 0.5, 0, 255).astype(np.uint8)
    a = np.ascontiguousarray(a)
    if a.shape[2] == 4:
        img = QImage(a.data, a.shape[1], a.shape[0], a.shape[1] * 4, QImage.Format_RGBA8888)
    else:
        img = QImage(a.data, a.shape[1], a.shape[0], a.shape[1] * 3, QImage.Format_RGB888)
    return img.copy()


def np_to_pixmap(arr: np.ndarray, max_side: int = 150, radius: int = 8) -> QPixmap:
    """numpy -> 圆角 QPixmap（通道缩略图 / 材质球用）。"""
    img = arr_to_qimage(arr)
    if max(img.width(), img.height()) > max_side:
        img = img.scaled(max_side, max_side, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    pm = QPixmap(img.size())
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    path = QPainterPath()
    path.addRoundedRect(0, 0, img.width(), img.height(), radius, radius)
    p.setClipPath(path)
    p.drawImage(0, 0, img)
    p.end()
    return pm


class ImagePanel(QWidget):
    """等比自适应显示贴图。

    比 QLabel 强的地方：面板拉多大，图就铺多大（QLabel 的 pixmap 尺寸是死的，
    面板一大就只剩一圈黑边）。checker=True 时先铺棋盘格，能看清 alpha。
    """

    def __init__(self, radius: int = 10, checker: bool = False,
                 placeholder: str = "载入贴图后显示",
                 min_size: tuple[int, int] = (160, 90),
                 max_side: int = 640, cover: bool = False):
        super().__init__()
        self.setMinimumSize(QSize(*min_size))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img: QImage | None = None
        self._radius = radius
        self._checker = checker
        self._placeholder = placeholder
        self._max_side = max_side
        self._cover = cover

    def set_array(self, arr):
        self._img = arr_to_qimage(arr, self._max_side) if arr is not None else None
        self.update()

    def set_placeholder(self, text: str):
        self._placeholder = text
        self.update()

    def clear(self):
        self._img = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        w, h = self.width(), self.height()

        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, self._radius, self._radius)
        p.fillPath(path, QColor(0, 0, 0, 60))
        p.setClipPath(path)

        if self._img is None:
            p.setPen(QColor(DIM))
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            p.end()
            return

        iw, ih = self._img.width(), self._img.height()
        s = (max if self._cover else min)(w / max(1, iw), h / max(1, ih))
        dw, dh = max(1, int(iw * s)), max(1, int(ih * s))
        x, y = (w - dw) // 2, (h - dh) // 2

        if self._checker:
            p.fillRect(max(0, x), max(0, y), min(dw, w), min(dh, h), QColor(28, 24, 38))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(43, 38, 56))
            step = 10
            for gy in range(0, dh, step):
                for gx in range(0, dw, step):
                    if ((gx // step) + (gy // step)) % 2 == 0:
                        p.drawRect(x + gx, y + gy, min(step, dw - gx), min(step, dh - gy))

        p.drawImage(QRectF(x, y, dw, dh), self._img)
        p.end()


# ------------------------------------------------------------------ 基础控件


def make_card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(11, 9, 11, 11)
    lay.setSpacing(7)
    if title:
        t = QLabel(title)
        t.setObjectName("cardTitle")
        lay.addWidget(t)
    return card, lay


def hint(text: str, wrap: bool = True) -> QLabel:
    lb = QLabel(text)
    lb.setObjectName("hint")
    lb.setWordWrap(wrap)
    return lb


def divider() -> QFrame:
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet("background:rgba(255,255,255,0.10);")
    return line


class Slider(QWidget):
    """标签 + 滑块 + 数值。"""

    def __init__(self, text: str, lo: float, hi: float, value: float,
                 decimals: int = 2, label_width: int = 40, height: int = 18):
        super().__init__()
        self.decimals = decimals
        self.scale = 10 ** decimals
        self.setFixedHeight(height)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.name = QLabel(text)
        self.name.setObjectName("hint")
        self.name.setFixedWidth(label_width)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(lo * self.scale), int(hi * self.scale))
        self.slider.setValue(int(round(value * self.scale)))

        self.value_label = QLabel(f"{value:.{decimals}f}")
        self.value_label.setProperty("nol10n", True)
        self.value_label.setFixedWidth(30)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setStyleSheet("color:%s;font-size:11px;" % ACCENT)

        lay.addWidget(self.name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.value_label)

        self._on_change = None
        self.slider.valueChanged.connect(self._changed)

    def _changed(self, v):
        self.value_label.setText(f"{v / self.scale:.{self.decimals}f}")
        if self._on_change:
            self._on_change()

    def on_change(self, fn):
        self._on_change = fn

    def value(self) -> float:
        return self.slider.value() / self.scale

    def reset(self, value: float = 1.0):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(value * self.scale)))
        self.slider.blockSignals(False)
        self.value_label.setText(f"{value:.{self.decimals}f}")


class Stepper(QWidget):
    """数值步进器：− 1024 +"""

    def __init__(self, values: list[int], index: int = 1, on_change=None,
                 labels: list[str] | None = None):
        super().__init__()
        self.values = values
        self.labels = labels or [str(v) for v in values]
        self.idx = index
        self._on_change = on_change
        self.setFixedHeight(26)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        minus = QPushButton("−")
        minus.setObjectName("step")
        minus.setFixedWidth(24)
        minus.setCursor(Qt.PointingHandCursor)
        minus.clicked.connect(lambda: self._step(-1))

        self.value_label = QLabel(self.labels[index])
        self.value_label.setProperty("nol10n", True)
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("color:#e8e4f2;font-weight:600;font-size:12px;")

        plus = QPushButton("+")
        plus.setObjectName("step")
        plus.setFixedWidth(24)
        plus.setCursor(Qt.PointingHandCursor)
        plus.clicked.connect(lambda: self._step(1))

        lay.addWidget(minus)
        lay.addWidget(self.value_label, 1)
        lay.addWidget(plus)

    def _step(self, d: int):
        self.idx = max(0, min(len(self.values) - 1, self.idx + d))
        self.value_label.setText(self.labels[self.idx])
        if self._on_change:
            self._on_change(self.value())

    def value(self) -> int:
        return self.values[self.idx]


class Segmented(QWidget):
    """分段按钮组。选中态是浅白色块，和整体玻璃风格一致。"""

    def __init__(self, items: list[str], active: int = 0, on_change=None,
                 compact: bool = False):
        super().__init__()
        self._on_change = on_change
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        pad = "2px 6px" if compact else "3px 10px"
        fs = "10px" if compact else "11px"
        wrap = QFrame()
        wrap.setStyleSheet(
            "QFrame{background:rgba(255,255,255,0.09);border:1px solid rgba(255,255,255,0.24);"
            "border-radius:7px;}"
            "QPushButton{background:transparent;border:0;border-radius:5px;"
            "padding:%s;color:#cec6e2;font-size:%s;}"
            "QPushButton:hover{background:rgba(255,255,255,0.14);}"
            "QPushButton:checked{background:rgba(255,255,255,0.94);color:#1d1729;font-weight:600;}"
            % (pad, fs))
        wl = QHBoxLayout(wrap)
        wl.setContentsMargins(2, 2, 2, 2)
        wl.setSpacing(0)

        self.buttons: list[QPushButton] = []
        for i, text in enumerate(items):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setChecked(i == active)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, idx=i: self.set_active(idx))
            self.buttons.append(b)
            wl.addWidget(b)
        lay.addWidget(wrap)

    def set_active(self, idx: int):
        for i, b in enumerate(self.buttons):
            b.setChecked(i == idx)
        if self._on_change:
            self._on_change(self.active_text())

    def active_text(self) -> str:
        for b in self.buttons:
            if b.isChecked():
                return b.text()
        return self.buttons[0].text()


class DotBadge(QWidget):
    """右上角圆形勾选徽标。点它 = 切换是否参与批量导出。"""

    def __init__(self, checked: bool, on_toggle):
        super().__init__()
        self.setFixedSize(14, 14)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("参与「导出全部通道」")
        self._checked = checked
        self._on_toggle = on_toggle

    def is_checked(self) -> bool:
        return self._checked

    def set_checked(self, on: bool):
        self._checked = bool(on)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self._checked:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 238))
            p.drawEllipse(0, 0, 13, 13)
            p.setPen(QPen(QColor("#1d1729"), 1.6))
            p.drawLine(3, 7, 6, 10)
            p.drawLine(6, 10, 10, 4)
        else:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 96), 1))
            p.drawEllipse(0, 0, 13, 13)
        p.end()

    def mousePressEvent(self, e):
        self._on_toggle()


# ------------------------------------------------------------------ 看板娘


def _anim_ratio(path: str, fallback: float = 0.5625) -> float:
    """读动画第一帧的尺寸拿宽高比 —— QMovie 本身不暴露这个信息。"""
    try:
        size = QImageReader(path).size()
        if size.isValid() and size.height() > 0:
            return size.width() / size.height()
    except Exception:  # noqa: BLE001
        pass
    return fallback


class MascotCard(QFrame):
    """左栏底部的看板娘。动画尺寸已烘进资源，运行时不缩放（QMovie 缩放会卡）。

    宽高比从资源里读，换一段视频只需要重新生成 webp，不用改代码。
    """

    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(2)

        self.view = QLabel("")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.view.setStyleSheet("background:transparent;")
        lay.addWidget(self.view, 1)

        # 脚下留一条槽，用来放语言切换之类的控件
        self.footer = QHBoxLayout()
        self.footer.setContentsMargins(0, 0, 0, 0)
        self.footer.addStretch(1)
        lay.addLayout(self.footer)

        self.movie: QMovie | None = None
        self._prop = _anim_ratio(_MASCOT_PATH)
        if os.path.exists(_MASCOT_PATH):
            self.movie = QMovie(_MASCOT_PATH)
            self.movie.setCacheMode(QMovie.CacheAll)
            self.view.setMovie(self.movie)
            self.movie.start()
        else:
            self.view.setText("动画未找到")
            self.view.setStyleSheet("color:#8b83a6;")

    def add_footer(self, w: QWidget):
        """把控件放到看板娘脚下（水平居中），动画会自动让出这块高度。"""
        self.footer.addWidget(w)
        self.footer.addStretch(1)   # 前后各一条弹簧 -> 居中对齐看板娘中线

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # QLabel 的 movie 本身是 150x197，这里只调整 label 的显示区域，
        # 交给 QLabel 等比居中，不对 QMovie 做 setScaledSize。
        extra = self.footer.sizeHint().height() + 6 if self.footer.count() > 1 else 0
        h = max(80, self.height() - 12 - extra)
        w = int(min(self.width() - 12, h * self._prop))
        self.view.setFixedSize(w, int(w / self._prop))


# ------------------------------------------------------------------ 3D 材质球


MIX_CH_INDEX = {"R": 0, "G": 1, "B": 2, "A": 3}
BASE_KEYS = ("COL", "NRM", "RGH", "MET", "OCC")


def _shrink_rgba(a: np.ndarray, target: int = 512) -> np.ndarray:
    """打包图整体降采样一次（PIL BOX），比逐通道 LANCZOS 快一个量级。

    球只有 300px 上下，4096 的图贴上去全是沙粒；先缩到 512 再说。
    """
    h, w = a.shape[:2]
    s = target / float(max(h, w))
    if s >= 1.0:
        return a
    arr = a if a.dtype == np.uint8 else np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if arr.ndim != 3:
        return arr
    if arr.shape[2] == 3:
        arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = Image.fromarray(arr[:, :, :4], "RGBA").resize((nw, nh), Image.BOX)
    return np.asarray(im)


def unpack_mix(rgba, mix: dict) -> dict:
    """把打包好的 RGBA 按 R/G/B/A 的分配拆回语义通道。

    球预览的就是这张打包图：R 槽里装了什么，球上对应的那项就用什么。
    同一语义通道被塞进两个槽位时取平均，空分配（—）跳过。
    """
    if rgba is None or not mix:
        return {}
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] < 3:
        return {}
    if a.shape[2] == 3:
        a = np.dstack([a, np.full(a.shape[:2], 255, a.dtype)])

    slots: dict[str, list] = {}
    for ch, key in mix.items():
        if key in ("—", None) or ch not in MIX_CH_INDEX:
            continue
        slots.setdefault(key, []).append(a[..., MIX_CH_INDEX[ch]])

    out: dict[str, np.ndarray] = {}
    for key, layers in slots.items():
        out[key] = layers[0] if len(layers) == 1 else np.mean(layers, axis=0)
    return out


def render_sphere(maps, size: int = 300, azimuth: float = 0.9,
                  elevation: float = 0.85) -> np.ndarray:
    """把整套材质贴到球上做光照预览（纯 numpy，不依赖 OpenGL）。

    maps 传 dict：COL 基础色 / NRM 法线 / RGH 粗糙度 / MET 金属度 / OCC 环境吸收，
    缺哪个就用该通道的中性值（灰 0.62、平法线、粗糙 0.5、非金属、AO 全亮）。
    DIS 高度会被当成一层很轻的凹凸叠上去（打包图里的高度通道也能看出来）。
    也可以只传一张法线数组，等价于 {"NRM": arr}。

    返回 RGBA（球外透明），直接压到透视网格背景上。

    坐标约定：屏幕空间 X 向右、Y 向下、Z 朝向镜头。所以法线图的
    G=+1（"向上"）对应球面切向的 −Y，B = T × N 正好落在画面上方。
    """
    blank = np.zeros((size, size, 4), dtype=np.uint8)
    if maps is None:
        return blank
    m = maps if isinstance(maps, dict) else {"NRM": maps}
    if not m or all(v is None for v in m.values()):
        return blank

    grid_y, grid_x = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    rad = size / 2.0 - 2.0
    sx = (grid_x - c) / rad
    sy = (grid_y - c) / rad
    d2 = sx * sx + sy * sy
    nz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))

    u = np.clip(sx * 0.5 + 0.5, 0.0, 1.0)
    v = np.clip(sy * 0.5 + 0.5, 0.0, 1.0)

    def _f(a):
        a = np.asarray(a)
        if a.dtype == np.uint8:
            return a.astype(np.float32) / 255.0
        return np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)

    def _uv(a):
        h, w = a.shape[:2]
        ix = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
        iy = np.clip((v * (h - 1)).astype(np.int32), 0, h - 1)
        return a[iy, ix]

    def sample1(key, default: float) -> np.ndarray:
        a = m.get(key)
        if a is None:
            return np.full((size, size), default, dtype=np.float32)
        s = _uv(_f(a))
        return s[..., 0] if s.ndim == 3 else s

    def sample3(key, default: float) -> np.ndarray:
        a = m.get(key)
        if a is None:
            return np.full((size, size, 3), default, dtype=np.float32)
        a = _f(a)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=2)
        return _uv(a)[..., :3]

    tn = sample3("NRM", 0.5) * 2.0 - 1.0          # 切空间法线
    if m.get("DIS") is not None:
        # 高度图顺手当凹凸：先糊掉沙粒，梯度只给很轻的 XY 偏转，别盖过法线
        d = gaussian_blur(sample1("DIS", 0.5), 2.0)
        gy, gx = np.gradient(d)
        amp = 1.4 * size / 256.0
        tn = np.stack([tn[..., 0] - gx * amp,
                       tn[..., 1] + gy * amp,
                       tn[..., 2]], axis=-1)
    alb = np.clip(sample3("COL", 0.62), 0.0, 1.0) ** 2.2  # sRGB -> 线性
    rough = np.clip(sample1("RGH", 0.55), 0.04, 1.0)
    metal = np.clip(sample1("MET", 0.0), 0.0, 1.0)
    ao = np.clip(sample1("OCC", 1.0), 0.0, 1.0)

    # 球面切空间：T 是 (1,0,0) 在切平面上的投影，B = T × N
    tx = 1.0 - sx * sx
    ty = -sx * sy
    tz = -sx * nz
    tl = np.sqrt(tx * tx + ty * ty + tz * tz) + 1e-6
    tx, ty, tz = tx / tl, ty / tl, tz / tl

    bx = ty * nz - tz * sy
    by = tz * sx - tx * nz
    bz = tx * sy - ty * sx

    nx = tx * tn[..., 0] + bx * tn[..., 1] + sx * tn[..., 2]
    ny = ty * tn[..., 0] + by * tn[..., 1] + sy * tn[..., 2]
    nzz = tz * tn[..., 0] + bz * tn[..., 1] + nz * tn[..., 2]
    nl = np.sqrt(nx * nx + ny * ny + nzz * nzz) + 1e-6
    nx, ny, nzz = nx / nl, ny / nl, nzz / nl

    # 主光在右上前方（el 越高越顶光，az 越大越偏右）
    lx = np.cos(azimuth) * np.cos(elevation)
    ly = -np.sin(azimuth) * np.cos(elevation)
    lz = np.sin(elevation)
    ll = np.sqrt(lx * lx + ly * ly + lz * lz) + 1e-6
    lx, ly, lz = lx / ll, ly / ll, lz / ll

    ndl = np.clip(nx * lx + ny * ly + nzz * lz, 0.0, 1.0)
    diff = np.clip((ndl + 0.16) / 1.16, 0.0, 1.0) ** 1.2    # 一点 wrap，暗面不糊死

    hx, hy, hz = lx, ly, lz + 1.0                            # 半角向量（视线 = +Z）
    hl = np.sqrt(hx * hx + hy * hy + hz * hz) + 1e-6
    ndh = np.clip((nx * hx + ny * hy + nzz * hz) / hl, 0.0, 1.0)
    ndv = np.clip(nzz, 0.0, 1.0)

    # 粗糙度 -> Blinn 指数：越粗糙高光越宽越弱，越光滑越锐
    spec_pow = np.clip(2.0 / np.maximum(rough, 0.05) ** 4 - 2.0, 8.0, 1024.0)
    spec = ndh ** spec_pow

    metal3 = metal[..., None]
    alb3 = alb
    diff_col = alb3 * (1.0 - metal3) * diff[..., None] * (0.40 + 0.60 * ao[..., None])

    # 粗糙度越高高光越弱：不然整颗球会被一层宽高光糊成灰塑料
    spec_gain = (0.55 * (1.0 - metal) + 1.30 * metal) * (1.0 - rough) ** 1.2
    spec_tint = (1.0 - metal3) + alb3 * metal3
    spec_col = spec[..., None] * spec_gain[..., None] * spec_tint

    # 假环境（上亮下暗 + 光源方向的反射斑）：金属没有漫反射，全靠它撑出体积感
    rx = 2.0 * ndv * nx
    ry = 2.0 * ndv * ny
    rz = 2.0 * ndv * nzz - 1.0
    rl = np.clip(rx * lx + ry * ly + rz * lz, 0.0, 1.0)
    # 假环境：一条"地平线"以上的亮天空 + 光源方向的一块反射斑。
    # 屏幕 Y 向下，反射 y 为正 = 朝地面，所以天空取 0.5 - ry*2.5 做一条利落的交界
    hz = np.clip(0.5 - ry * 2.5, 0.0, 1.0)
    hz = hz * hz * (3.0 - 2.0 * hz)
    env = 0.10 + 0.78 * hz + 0.90 * rl ** 10
    env_col = env[..., None] * (alb3 * metal3 + (1.0 - metal3) * 0.05) * ao[..., None]

    amb = alb3 * (0.055 * ao * (1.0 - metal))[..., None]
    rim = (np.clip(1.0 - ndv, 0.0, 1.0) ** 3 * 0.06)[..., None]
    rim_col = rim * np.array([0.55, 0.68, 1.0], dtype=np.float32)

    col = diff_col + amb + spec_col + env_col + rim_col
    col = np.clip(col, 0.0, 1.0) ** (1.0 / 2.2)

    alpha = np.clip((1.0 - np.sqrt(d2)) * rad * 2.0, 0.0, 1.0)
    out = np.zeros((size, size, 4), dtype=np.float32)
    out[..., :3] = col
    out[..., 3] = alpha
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


class SphereView(QWidget):
    """带透视网格地面的材质球：左键拖拽转光源，滚轮缩放。"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(QSize(200, 170))
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("球上用的是「通道打包」当前 R/G/B/A 分配的打包结果\n"
                        "法线与基础色取自原始通道 · 左键拖拽转光源 · 滚轮缩放")
        self._maps: dict = {}
        self._base: dict = {}
        self._base_cache: dict = {}
        self._mix_used: set[str] = set()
        self._az = 0.9
        self._el = 0.8
        self._zoom = 1.0
        self._full = 200
        self._pm: QPixmap | None = None
        self._drag = None

    def set_material(self, maps):
        """喂整套材质（不经过打包图）。maps 是通道名 -> 数组的 dict。"""
        self._base_cache = {}
        self.set_base(maps)
        self._mix_used = set()
        self._maps = dict(self._base)
        self.refresh()

    def set_base(self, maps):
        """只更新原始通道（法线 / 基础色从这来），留着给 set_packed 用。

        按通道对象逐个比对缓存：4K 图缩放一次就够，不然每帧重算会卡死。
        """
        src = maps if isinstance(maps, dict) else {"NRM": maps}
        cache = self._base_cache
        out = {}
        for k in BASE_KEYS:
            a = src.get(k)
            hit = cache.get(k)
            if a is None:
                out[k] = None
            elif hit is not None and hit[0] is a:
                out[k] = hit[1]
            else:
                out[k] = self._smooth(a)
                cache[k] = (a, out[k])
        self._base = out

    def set_packed(self, rgba, mix: dict, base=None):
        """球预览 = 当前打包图。按 R/G/B/A 的分配把打包图解回语义通道再渲染。

        法线（和没被打包的基础色）从原始通道补 —— 它们本来就不进打包图。
        """
        if base is not None:
            self.set_base(base)
        b = self._base

        maps = {}
        if rgba is not None:
            a = np.asarray(rgba)
            if a.ndim == 3 and a.shape[2] >= 3:
                # 先整体缩到 512，再拆通道；4K 逐通道缩放会拖慢拖动 gamma 的响应
                a = _shrink_rgba(a)
                if a.dtype == np.uint8:
                    a = a.astype(np.float32) / 255.0
                maps = unpack_mix(a, mix)
        if not maps:                       # 四个槽位全空：退回原始通道
            self._mix_used = set()
            self._maps = dict(b)
            self.refresh()
            return

        if "COL" not in maps and b.get("COL") is not None:
            maps["COL"] = b["COL"]
        nrm = b.get("NRM")
        dis = maps.pop("DIS", None)
        if nrm is not None:
            # 高度图当场并进法线，渲染时就不用每帧再算梯度（拖光源才不卡）
            maps["NRM"] = self._with_bump(nrm, dis) if dis is not None else nrm
        self._mix_used = {v for v in mix.values() if v not in ("—", None)}
        self._maps = maps
        self.refresh()

    def set_normal(self, arr):
        """兼容老调用：只给法线。"""
        self.set_material({"NRM": arr})

    @staticmethod
    def _with_bump(nrm: np.ndarray, dis: np.ndarray, strength: float = 1.6):
        """把高度图的梯度叠到法线 xy 上（DIS 打进打包图时球上能看出起伏）。"""
        if nrm is None or dis is None:
            return nrm
        d = np.asarray(dis, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        nr = np.clip(np.asarray(nrm, dtype=np.float32), 0.0, 1.0)
        if nr.ndim == 2:
            nr = np.stack([nr, nr, nr], axis=2)
        if d.shape[:2] != nr.shape[:2]:
            d = resize(d, (nr.shape[1], nr.shape[0]))
        gy, gx = np.gradient(gaussian_blur(d, 2.0))
        amp = strength * nr.shape[1] / 512.0
        out = nr.copy()
        out[..., 0] = np.clip(out[..., 0] - gx * amp, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + gy * amp, 0.0, 1.0)
        return out

    @staticmethod
    def _smooth(arr):
        """球面只是预览：4K 贴图直接采样就是一片雪花噪点。

        先降采样到 ~512px、再非常轻地糊一下（0.65），结构留住、沙粒去掉。
        统一转成 float32 [0,1]，渲染时就不用反复判断位深。
        """
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        else:
            a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
        h, w = a.shape[:2]
        scale = 512.0 / max(h, w)
        if scale < 1.0:
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            a = resize(a, (nw, nh))
        return np.clip(gaussian_blur(a, 0.65), 0.0, 1.0).astype(np.float32)

    def refresh(self, draft: bool = False):
        """重画球。draft=True 用半分辨率（拖光源时保帧率，松手再补高清）。"""
        pad = 6
        avail_w = max(80, self.width() - pad * 2)
        avail_h = max(80, self.height() - pad * 2)
        size = max(80, min(avail_w, avail_h))
        self._full = size
        render_size = max(96, size // 2) if draft else size
        img = render_sphere(self._maps, render_size, self._az, self._el)
        self._pm = np_to_pixmap(img, render_size, radius=0)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(24, 20, 34))
        grad.setColorAt(1.0, QColor(14, 11, 20))
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, 9, 9)
        p.fillPath(path, grad)
        p.setClipPath(path)

        horizon = h * 0.74
        p.setPen(QPen(QColor(139, 108, 240, 52), 1))
        for i in range(-6, 7):
            p.drawLine(QPointF(w / 2 + i * w * 0.06, horizon),
                       QPointF(w / 2 + i * w * 0.34, h))
        for i in range(1, 7):
            y = horizon + (h - horizon) * (i / 6.0) ** 1.8
            p.drawLine(QPointF(0, y), QPointF(w, y))

        if self._pm is not None:
            pm = self._pm
            # draft 渲染的是半分辨率，绘制时拉回正常大小（缩略期间略糊，松手就清）
            target = max(8, int(self._full * self._zoom))
            if pm.width() != target:
                pm = pm.scaled(target, target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            px = (w - pm.width()) // 2
            py = (h - pm.height()) // 2 - 4

            # 接地阴影：没有它球像是浮在网格上
            sh_w, sh_h = pm.width() * 0.62, pm.height() * 0.13
            sh = QRectF(w / 2.0 - sh_w / 2, py + pm.height() - sh_h * 0.62,
                        sh_w, sh_h)
            sg = QRadialGradient(sh.center(), sh_w / 2.0)
            sg.setColorAt(0.0, QColor(0, 0, 0, 96))
            sg.setColorAt(0.5, QColor(0, 0, 0, 48))
            sg.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(sg)
            p.drawEllipse(sh)

            p.drawPixmap(px, py, pm)
        p.end()

    def mousePressEvent(self, e):
        self._drag = e.position()
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        dp = e.position() - self._drag
        self._drag = e.position()
        self._az += dp.x() * 0.012
        self._el = float(np.clip(self._el - dp.y() * 0.010, -1.2, 1.2))
        self.refresh(draft=True)

    def mouseReleaseEvent(self, e):
        self._drag = None
        self.setCursor(Qt.OpenHandCursor)
        self.refresh()

    def wheelEvent(self, e):
        self._zoom = float(np.clip(self._zoom * (1.0015 ** e.angleDelta().y()), 0.6, 1.8))
        self.refresh()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refresh()


# ------------------------------------------------------------------ 通道后处理


def _to01(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    return np.asarray(a, dtype=np.float32)


def apply_channel(key: str, arr, gain: float = 1.0, contrast: float = 1.0):
    """按卡片的「强度 / 对比度」处理单个通道。

    - 灰度通道：先做围绕 0.5 的对比度 S 曲线，再按强度缩放振幅
    - 法线通道：强度缩放切线空间 xy（保持 z 正、重新归一化），
                对比度对 xy 幅值做 tanh 饱和（压平弱细节、放大强细节）
    两者都是 1.0 时原样返回，保证默认不改变数据。
    """
    if arr is None:
        return None
    if abs(gain - 1.0) < 1e-4 and abs(contrast - 1.0) < 1e-4:
        return arr

    if key == "NRM":
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            a = np.clip(np.asarray(a, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
        n = a[..., :3].astype(np.float32) / 255.0 * 2.0 - 1.0
        xy = n[..., :2] * float(gain)
        m = np.linalg.norm(xy, axis=2, keepdims=True)
        xy = xy * (np.tanh(m * float(contrast)) / np.maximum(m, 1e-6))
        nz = np.sqrt(np.clip(1.0 - np.sum(xy * xy, axis=2), 0.0, 1.0))
        out = np.stack([xy[..., 0], xy[..., 1], nz], axis=2) * 0.5 + 0.5
        out = np.clip(out, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2], 0.5, 1.0)   # z 恒为正，防止出现粉色
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)

    x = _to01(arr)
    y = np.clip((x - 0.5) * float(contrast) + 0.5, 0.0, 1.0)
    y = np.clip((y - 0.5) * float(gain) + 0.5, 0.0, 1.0)
    if x.ndim == 2:
        return y
    return y[..., :3]


# ------------------------------------------------------------------ 通道小图


class ThumbTile(QFrame):
    """左栏的小图格子：缩略图 + 通道代号。

    两层状态彼此独立：
      · 选中（sel）—— 点格子把右侧大图切到这张，同一时刻只有一个
      · 勾选（picked）—— 点右上角圆点，决定它是否参与「导出全部通道」
    `checked=None` 表示这是源图格子，没有勾选概念。
    """

    W, H = 70, 92

    def __init__(self, key: str, cn: str, checked: bool | None,
                 on_pick, on_toggle=None):
        super().__init__()
        self.setObjectName("thumb")
        self.key = key
        self.cn = cn
        self.setFixedSize(self.W, self.H)
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("sel", "false")
        self.setProperty("picked", "true" if checked else "false")
        self.setToolTip(f"{key} · {cn}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 3)
        lay.setSpacing(2)

        self.panel = ImagePanel(radius=5, checker=(key == "MASK"),
                                placeholder=cn[:2],
                                min_size=(58, 58), max_side=160, cover=True)
        self.panel.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        lay.addWidget(self.panel, 1)

        row = QHBoxLayout()
        row.setContentsMargins(1, 0, 0, 0)
        row.setSpacing(2)
        code = QLabel(key)
        code.setStyleSheet("color:#c6bee2;font-size:9px;font-weight:700;letter-spacing:0.4px;")
        row.addWidget(code)
        row.addStretch(1)
        self.badge = None
        if checked is not None:
            self.badge = DotBadge(checked, lambda: on_toggle(self.key))
            row.addWidget(self.badge)
        lay.addLayout(row)

        self._on_pick = on_pick

    # -------------------------------------------------- 状态

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._on_pick(self.key)

    def set_array(self, arr):
        self.panel.set_array(arr)

    def apply_lang(self, en: bool):
        name = _EN.get(self.cn, self.cn) if en else self.cn
        self.panel.set_placeholder(name[:2])
        self.setToolTip(f"{self.key} · {name}")

    def set_selected(self, on: bool):
        self.setProperty("sel", "true" if on else "false")
        self._repaint()

    def is_checked(self) -> bool:
        return bool(self.badge and self.badge.is_checked())

    def set_checked(self, on: bool):
        if self.badge is not None:
            self.badge.set_checked(on)
        self.setProperty("picked", "true" if self.is_checked() else "false")
        self._repaint()

    def _repaint(self):
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


# ------------------------------------------------------------------ 主窗口


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChordStudio · PBR 材质分解")
        self.resize(1500, 930)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(STYLE)
        self.setAcceptDrops(True)
        if os.path.exists(_ICON_PATH):
            self.setWindowIcon(QIcon(_ICON_PATH))

        self.src_rgb: np.ndarray | None = None
        self.src_path: str = ""
        self.result = None
        self.current_key = "COL"
        self.cards: dict[str, ThumbTile] = {}
        self.display: dict[str, np.ndarray] = {}
        self._composite_arr = None
        self.out_size = 1024
        self.rough_step = 5
        self.flip_y = False
        self.ext_mode = "png"
        self.lang = "cn"
        self.alpha_mode = "data"     # data = 原样写 A；opaque = 写满 255

        self._chan_params: dict[str, list[float]] = {k: [1.0, 1.0] for k, _ in CHANNELS}
        self._outdir_labels: list[QLabel] = []
        self._fmt_labels: list[QLabel] = []
        self.out_dir = os.path.abspath("output")
        self.ed_name = "texture"

        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(200)
        self._regen_timer.timeout.connect(self.regenerate)

        self._bg = QPixmap(_BG_PATH) if os.path.exists(_BG_PATH) else QPixmap()
        self._bg_scaled: QPixmap | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 4, 10, 8)
        root.setSpacing(6)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(8)
        body.addWidget(self._build_left(), 0)
        body.addWidget(self._build_right(), 1)
        root.addLayout(body, 1)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(50)
        root.addWidget(self.log_box)

        self._build_menu()
        self.menuBar().hide()
        self._update_multi_btn()
        self._set_fmt("PNG")
        self._set_placeholders()
        self.select_channel("COL")
        self.log(self._t("就绪 · 选择或拖入一张贴图 · 左栏小图切换 · 「通道打包」按 R/G/B/A 合成",
                         "Ready · load or drop a texture · click a thumb to preview · Packing tab builds the RGBA"))

    # ---------------------------------------------------------- 顶栏

    def _build_header(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("header")
        wrap.setFixedHeight(64)
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(12, 5, 12, 5)
        lay.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(14, 14)
        dot.setStyleSheet(
            "background:qradialgradient(cx:0.38,cy:0.32,radius:0.95,stop:0 #c3b2ff,stop:1 #4b2fb0);"
            "border-radius:7px;")
        row1.addWidget(dot)

        logo = QLabel("ChordStudio")
        logo.setStyleSheet("font-size:13px;font-weight:700;color:#fff;")
        row1.addWidget(logo)
        row1.addWidget(hint("· CHORD PBR 材质分解", False))
        row1.addStretch(1)

        # 可导出的格式全部收到顶栏，两页共用同一个状态
        self._fmt_title = hint("导出格式", False)
        row1.addWidget(self._fmt_title)
        self.seg_fmt = Segmented(FORMATS, 0, self._set_fmt, compact=True)
        self.seg_fmt.setToolTip("PNG / TGA / WebP / JPG / ICO / BMP / GIF / TIF / DDS")
        row1.addWidget(self.seg_fmt)
        row1.addSpacing(12)

        self.drop_hint = QLabel("拖放图片到软件中")
        self.drop_hint.setAlignment(Qt.AlignCenter)
        self.drop_hint.setFixedSize(126, 26)
        self.drop_hint.setCursor(Qt.PointingHandCursor)
        self.drop_hint.setToolTip("也可以点这里选择图片")
        self.drop_hint.setStyleSheet(
            "border:1px dashed rgba(255,255,255,0.28);border-radius:8px;"
            "color:#c3bad9;font-size:11px;background:rgba(18,14,26,0.72);")
        self.drop_hint.mousePressEvent = lambda e: self.pick_source()  # type: ignore[method-assign]
        row1.addWidget(self.drop_hint)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(4)
        self.tab_prev = QPushButton("贴图预览")
        self.tab_pack = QPushButton("通道打包")
        for i, b in enumerate((self.tab_prev, self.tab_pack)):
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setChecked(i == 0)
            b.clicked.connect(lambda _=False, n=i: self.switch_tab(n))
            row2.addWidget(b)
        row2.addStretch(1)
        row2.addWidget(hint("CHORD · Ublsoft", False))
        row2.addSpacing(8)
        row2.addWidget(hint("本地离线", False))
        lay.addLayout(row2)
        return wrap

    def switch_tab(self, n: int):
        self.tab_prev.setChecked(n == 0)
        self.tab_pack.setChecked(n == 1)
        self.stack.setCurrentIndex(n)

    # ---------------------------------------------------------- 中 / 英

    def _t(self, cn: str, en: str | None = None) -> str:
        """取当前语言的文案。传了 en 就用它，否则查 _EN 表。"""
        if self.lang != "en":
            return cn
        return en if en is not None else _EN.get(cn, cn)

    def _set_lang(self, text: str):
        want = "en" if text.strip().upper() == "EN" else "cn"
        for b in self.seg_lang.buttons:
            b.setChecked(b.text().strip().upper() == want.upper())
        if want == self.lang:
            return
        self.lang = want
        self._apply_lang()
        self.log("UI language -> English" if want == "en" else "界面语言 -> 中文")

    def _apply_lang(self):
        """把所有 QLabel / QPushButton 的文案换一遍。

        第一次进来时把控件当前的中文存进 `cnText` 属性，之后来回切换都读它 ——
        这样不用为每个控件手写一份中英对照，新增控件也自动生效（只要译文进了 _EN）。
        """
        en = self.lang == "en"
        for w in self.findChildren(QLabel) + self.findChildren(QPushButton):
            if w.property("nol10n"):
                continue
            cn = w.property("cnText")
            if cn is None:
                cn = w.text()
                if not cn.strip():
                    continue
                w.setProperty("cnText", cn)
            w.setText(_EN.get(cn, cn) if en else cn)

        self._retranslate_combos()
        self._set_fmt(self.seg_fmt.active_text(), quiet=True)
        self._update_multi_btn()
        self._refresh_outdir()
        self._set_placeholders()
        self._refresh_sphere_label()
        self._refresh_comp_warn()
        self.select_channel(self.current_key)

    def _retranslate_combos(self):
        for cb in self.mix_combos.values():
            cur = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            for k, cn in MIX_ITEMS:
                cb.addItem(self._t("空 —") if k == "—" else f"{self._t(cn)} {k}", k)
            cb.setCurrentIndex(max(0, cb.findData(cur)))
            cb.blockSignals(False)

    def _set_placeholders(self):
        self.big.set_placeholder(self._t("点左侧小图查看大图"))
        self.composite.set_placeholder(self._t("合成结果会显示在这里"))
        self.src_tile.apply_lang(self.lang == "en")
        for tile in self.cards.values():
            tile.apply_lang(self.lang == "en")

    # ---------------------------------------------------------- 左栏

    def _build_left(self) -> QWidget:
        col = QWidget()
        col.setFixedWidth(156)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(7)

        card, lc = make_card()
        name = QLabel("ChordStudio")
        name.setStyleSheet("font-size:16px;font-weight:700;color:#fff;")
        lc.addWidget(name)
        lc.addWidget(hint("CHORD PBR 材质分解"))

        self.btn_load = QPushButton("选择图片")
        self.btn_load.setObjectName("accent")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.clicked.connect(self.pick_source)
        lc.addWidget(self.btn_load)

        self.btn_gen = QPushButton("生成贴图")
        self.btn_gen.setCursor(Qt.PointingHandCursor)
        self.btn_gen.clicked.connect(self.regenerate)
        lc.addWidget(self.btn_gen)

        self.btn_all = QPushButton("导出全部通道")
        self.btn_all.setCursor(Qt.PointingHandCursor)
        self.btn_all.setToolTip("导出带圆点勾选的所有通道（单图）")
        self.btn_all.clicked.connect(self.export_checked)
        lc.addWidget(self.btn_all)

        lc.addSpacing(2)
        lc.addWidget(divider())
        lc.addWidget(hint("推理参数"))

        # 分辨率要显示成 1024×1024，和标签挤一行放不下，标签单独占一行
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(hint("分辨率"))
        self.step_res = Stepper(RESOLUTIONS, 0, self._on_res, labels=RES_LABELS)
        box.addWidget(self.step_res)
        lc.addLayout(box)

        lc.addSpacing(1)

        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(hint("粗糙度步长"))
        self.step_rough = Stepper(ROUGH_STEPS, 4, self._on_rough)
        box.addWidget(self.step_rough)
        lc.addLayout(box)

        lc.addWidget(hint("越大越快，精度越低"))

        row = QHBoxLayout()
        row.addWidget(hint("法线 Y", False))
        self.seg_flip = Segmented(["Dx", "Gl"], 0, self._on_flip)
        row.addWidget(self.seg_flip)
        lc.addLayout(row)
        lc.addWidget(hint("Dx = UE / Unity · Gl = OpenGL"))

        row = QHBoxLayout()
        row.setSpacing(6)
        btn_fix = QPushButton("法线修复")
        btn_fix.setObjectName("tiny")
        btn_fix.setCursor(Qt.PointingHandCursor)
        btn_fix.setToolTip("修掉导出的粉色法线（通道反转）")
        btn_fix.clicked.connect(self.open_repair)
        row.addWidget(btn_fix, 1)
        lc.addLayout(row)
        lay.addWidget(card)

        mascot = MascotCard()
        self.mascot = mascot
        lay.addWidget(mascot, 1)

        # 看板娘脚下：界面语言
        self.seg_lang = Segmented(["CN", "EN"], 0, self._set_lang, compact=True)
        self.seg_lang.setFixedSize(112, 25)
        self.seg_lang.setToolTip("界面语言 / UI language")
        mascot.add_footer(self.seg_lang)
        return col

    def _on_res(self, value: int):
        self.out_size = int(value)
        self.log(self._t(f"分辨率 -> {value}×{value}", f"Size -> {value}×{value}"))
        if value >= 8192:
            self.log(self._t("8192 至少要 4 GB 空闲内存，会慢一会儿",
                             "8192 needs about 4 GB free RAM and takes a while"))
        self.schedule_regen()

    def _on_rough(self, value: int):
        self.rough_step = int(value)
        self.schedule_regen()

    def _on_flip(self, text: str):
        self.flip_y = text == "Gl"
        self.log("法线 Y 方向 -> " + ("OpenGL (+Y)" if self.flip_y else "DirectX (−Y)"))
        self.schedule_regen()

    # ---------------------------------------------------------- 右区（两个 tab）

    def _build_right(self) -> QWidget:
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_preview_page())
        self.stack.addWidget(self._build_pack_page())
        return self.stack

    def _build_preview_page(self) -> QWidget:
        """左：小图选择 + 参数 + 导出；右：选中通道的大图。"""
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # ---------------------------------------------- 左栏：小图 / 参数 / 导出
        # 窗口压到最小高度时这一列会超出可视区，套一层滚动，别让卡片互相压
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        # ① 小图网格
        c1, l1 = make_card("通道")
        grid = QGridLayout()
        grid.setSpacing(5)
        self.src_tile = ThumbTile("SRC", "源图", None, self.select_channel)
        grid.addWidget(self.src_tile, 0, 0)
        for i, (key, cn) in enumerate(CHANNELS):
            tile = ThumbTile(key, cn, key in CHECKED_BY_DEFAULT,
                             on_pick=self.select_channel,
                             on_toggle=self.toggle_checked)
            self.cards[key] = tile
            slot = i + 1
            grid.addWidget(tile, slot // 4, slot % 4)
        # 第 5 列吃掉落差，免得 4 列的格子被拉伸 / 挤出卡片
        grid.setColumnStretch(4, 1)
        l1.addLayout(grid)
        l1.addWidget(hint("点小图看大图 · 点右上角圆点决定是否参与「导出全部通道」"))
        ll.addWidget(c1)

        # ② 强弱 / 对比（只作用于当前选中的通道）
        c2, l2 = make_card("调整")
        top = QHBoxLayout()
        top.setSpacing(6)
        self.lbl_sel = QLabel("COL · 基础色")
        self.lbl_sel.setStyleSheet("color:#e2dcf5;font-size:11px;font-weight:600;")
        top.addWidget(self.lbl_sel)
        top.addStretch(1)
        self.btn_reset = QPushButton("重置")
        self.btn_reset.setObjectName("tiny")
        self.btn_reset.setFixedWidth(46)
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setToolTip("把这一个通道的强度 / 对比度退回 1.00")
        self.btn_reset.clicked.connect(self.reset_current)
        top.addWidget(self.btn_reset)
        l2.addLayout(top)

        self.s_gain = Slider("强度", 0.0, 2.0, 1.0, 2, label_width=42)
        self.s_contrast = Slider("对比度", 0.2, 3.0, 1.0, 2, label_width=42)
        self.s_gain.on_change(self._on_adjust)
        self.s_contrast.on_change(self._on_adjust)
        l2.addWidget(self.s_gain)
        l2.addWidget(self.s_contrast)
        l2.addWidget(hint("强度 = 细节量；对比度 = 明暗反差。法线会自动保值，不会变粉。"))
        ll.addWidget(c2)

        # ③ 导出选项
        c3, l3 = make_card("导出")
        self.lbl_fmt = hint("", False)
        self.lbl_fmt.setStyleSheet("color:#d5cdec;font-size:10px;")
        l3.addWidget(self.lbl_fmt)
        self._fmt_labels.append(self.lbl_fmt)

        self.btn_one = QPushButton("导出当前通道")
        self.btn_one.setObjectName("accent")
        self.btn_one.setCursor(Qt.PointingHandCursor)
        self.btn_one.setToolTip("把右边大图这张按当前格式写盘")
        self.btn_one.clicked.connect(lambda: self.export_one(self.current_key))
        l3.addWidget(self.btn_one)

        self.btn_multi = QPushButton("导出勾选的 8 张")
        self.btn_multi.setCursor(Qt.PointingHandCursor)
        self.btn_multi.clicked.connect(self.export_checked)
        l3.addWidget(self.btn_multi)

        lb_out = hint("", False)
        lb_out.setStyleSheet("color:#8f87ad;font-size:10px;")
        lb_out.setCursor(Qt.PointingHandCursor)
        lb_out.setToolTip("点击选择导出目录")
        lb_out.mousePressEvent = lambda e: self.pick_outdir()  # type: ignore[method-assign]
        self._outdir_labels.append(lb_out)
        l3.addWidget(lb_out)
        ll.addWidget(c3)

        ll.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(332)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(left)
        lay.addWidget(scroll, 0)

        # ---------------------------------------------- 右栏：大图
        c4, l4 = make_card()
        head = QHBoxLayout()
        head.setSpacing(7)
        self.big_code = QLabel("COL")
        self.big_code.setStyleSheet(
            "color:#cfc4ff;background:rgba(139,108,240,0.24);border-radius:4px;"
            "padding:1px 6px;font-size:10px;font-weight:700;letter-spacing:0.5px;")
        self.big_name = QLabel("基础色")
        self.big_name.setStyleSheet("color:#f2eefb;font-size:13px;font-weight:600;")
        head.addWidget(self.big_code)
        head.addWidget(self.big_name)
        head.addStretch(1)
        self.big_info = hint("—", False)
        self.big_info.setStyleSheet("color:#9d95ba;font-size:10px;")
        head.addWidget(self.big_info)
        l4.addLayout(head)

        self.big = ImagePanel(radius=10, checker=False,
                              placeholder="点左侧小图查看大图",
                              min_size=(360, 340), max_side=1400)
        l4.addWidget(self.big, 1)

        self.big_note = hint("", False)
        self.big_note.setStyleSheet("color:#8f87ad;font-size:10px;")
        l4.addWidget(self.big_note)

        lay.addWidget(c4, 1)
        return page

    def _build_pack_page(self) -> QWidget:
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # 左：混合选项 + 3D 材质预览
        left = QWidget()
        left.setFixedWidth(330)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        c1, l1 = make_card("混合选项")
        l1.addWidget(hint("按 R/G/B/A 把多张通道打包成一张 RGBA"))
        l1.addWidget(hint("3D 球照这套分配拆回来预览：换预设球没变 = 通道没丢；"
                          "球变了 = 有通道没打进去。"))

        self.mix_combos: dict[str, QComboBox] = {}
        colors = {"R": "#ff5b6e", "G": "#4bd07a", "B": "#5b8bff", "A": "#d9d4e6"}
        for ch in ("R", "G", "B", "A"):
            row = QHBoxLayout()
            row.setSpacing(7)
            chip = QLabel(ch)
            chip.setFixedSize(19, 19)
            chip.setAlignment(Qt.AlignCenter)
            chip.setStyleSheet("background:%s;color:#17131f;border-radius:4px;"
                               "font-weight:700;font-size:11px;" % colors[ch])
            combo = QComboBox()
            combo.setFixedHeight(26)
            for k, cn in MIX_ITEMS:
                combo.addItem(f"{cn} {k}" if k != "—" else "空 —", k)
            combo.setCurrentIndex(max(0, combo.findData(DEFAULT_MIX[ch])))
            combo.currentIndexChanged.connect(self.update_composite)
            self.mix_combos[ch] = combo
            row.addWidget(chip)
            row.addWidget(combo, 1)
            l1.addLayout(row)

        l1.addWidget(divider())
        l1.addWidget(hint("预设"))
        grid = QGridLayout()
        grid.setSpacing(5)
        for i, name in enumerate(PRESET_MIX):
            b = QPushButton(name)
            b.setFixedHeight(26)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
            grid.addWidget(b, i // 2, i % 2)
        l1.addLayout(grid)
        l1.addWidget(hint("圆点是整张通道打包的标准做法：ORM 给 UE，"
                          "Mask Map 给 HDRP。"))
        ll.addWidget(c1)

        c2, l2 = make_card("3D 打包预览")
        self.sphere = SphereView()
        l2.addWidget(self.sphere, 1)
        self.lbl_sphere_src = hint("球上取自打包图：R=…", False)
        self.lbl_sphere_src.setProperty("nol10n", True)   # 动态文案，自己按语言重写
        self.lbl_sphere_src.setAlignment(Qt.AlignCenter)
        self.lbl_sphere_src.setStyleSheet("color:#8f87ad;font-size:10px;")
        l2.addWidget(self.lbl_sphere_src)
        self.lbl_sphere_warn = hint("", False)
        self.lbl_sphere_warn.setProperty("nol10n", True)   # 动态文案，自己按语言重写
        self.lbl_sphere_warn.setAlignment(Qt.AlignCenter)
        self.lbl_sphere_warn.setStyleSheet("color:#e2a856;font-size:10px;")
        self.lbl_sphere_warn.hide()
        l2.addWidget(self.lbl_sphere_warn)
        cap = hint("左键拖拽转光源 · 滚轮缩放", False)
        cap.setAlignment(Qt.AlignCenter)
        l2.addWidget(cap)
        ll.addWidget(c2, 1)
        lay.addWidget(left, 0)

        # 右：合成结果
        c3, l3 = make_card("合成后的材质图")
        self.composite = ImagePanel(radius=10, checker=True,
                                    placeholder="合成结果会显示在这里",
                                    min_size=(240, 200))
        l3.addWidget(self.composite, 1)

        self.lbl_comp = hint("R:RGH · G:DIS · B:OCC · A:MET", False)
        self.lbl_comp.setAlignment(Qt.AlignCenter)
        l3.addWidget(self.lbl_comp)

        # alpha 承载的是数据（比如 RHAM 的金属度），看图器会当成透明度合成到背景上，
        # A 全 0 时整张图看着就是空的。这行提示负责解释，下面那个开关负责解决。
        self.lbl_comp_warn = hint("", False)
        self.lbl_comp_warn.setProperty("nol10n", True)
        self.lbl_comp_warn.setWordWrap(True)
        self.lbl_comp_warn.setStyleSheet("color:#e2a856;font-size:10px;")
        self.lbl_comp_warn.hide()
        l3.addWidget(self.lbl_comp_warn)

        l3.addWidget(divider())
        self.s_gamma = Slider("Gamma", 0.4, 2.5, 1.0, 2, label_width=48)
        self.s_gamma.on_change(self.update_composite)
        l3.addWidget(self.s_gamma)

        row_a = QHBoxLayout()
        row_a.setSpacing(6)
        row_a.addWidget(hint("A 写入", False))
        self.seg_alpha = Segmented(["数据", "不透明"], 0, self._on_alpha_mode, compact=True)
        row_a.addWidget(self.seg_alpha)
        row_a.addStretch(1)
        l3.addLayout(row_a)

        self.lbl_fmt2 = hint("", False)
        self.lbl_fmt2.setStyleSheet("color:#d5cdec;font-size:10px;")
        l3.addWidget(self.lbl_fmt2)
        self._fmt_labels.append(self.lbl_fmt2)

        self.btn_export_comp = QPushButton("导出 RGBA 贴图")
        self.btn_export_comp.setObjectName("accent")
        self.btn_export_comp.setCursor(Qt.PointingHandCursor)
        self.btn_export_comp.clicked.connect(self.export_composite)
        l3.addWidget(self.btn_export_comp)

        self.btn_export_both = QPushButton("单图 + 合成 一起导出")
        self.btn_export_both.setCursor(Qt.PointingHandCursor)
        self.btn_export_both.clicked.connect(self.export_both)
        l3.addWidget(self.btn_export_both)

        lb_out = hint("", False)
        lb_out.setStyleSheet("color:#8f87ad;font-size:10px;")
        lb_out.setCursor(Qt.PointingHandCursor)
        lb_out.setToolTip("点击选择导出目录")
        lb_out.mousePressEvent = lambda e: self.pick_outdir()  # type: ignore[method-assign]
        self._outdir_labels.append(lb_out)
        l3.addWidget(lb_out)
        self._refresh_outdir()

        lay.addWidget(c3, 1)
        return page

    # ---------------------------------------------------------- 背景

    def _rescale_bg(self):
        if self._bg.isNull():
            self._bg_scaled = None
            return
        self._bg_scaled = self._bg.scaled(max(1, self.width()), max(1, self.height()),
                                          Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._bg_scaled is None or self._bg_scaled.isNull():
            return
        p = QPainter(self)
        p.drawPixmap((self.width() - self._bg_scaled.width()) // 2,
                     (self.height() - self._bg_scaled.height()) // 2,
                     self._bg_scaled)
        p.fillRect(self.rect(), QColor(11, 8, 17, 92))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale_bg()

    def _ensure_on_screen(self):
        """把窗口拉回可见区域。

        双屏 + 负坐标（左侧竖屏）时 Windows 会按主屏信息去算位置，算出来
        整块落在屏幕外，用户只能看到一个任务栏图标。这里统一处理：
          · 已经完整落在某块屏幕可用区内 -> 不动
          · 否则挪到窗口中心所在的那块屏，居中，尺寸按可用区夹一次
        """
        screens = QGuiApplication.screens()
        if not screens:
            return
        g = self.frameGeometry()
        if any(s.availableGeometry().contains(g) for s in screens):
            return

        scr = QGuiApplication.screenAt(g.center()) or QGuiApplication.primaryScreen()
        if scr is None:
            return
        av = scr.availableGeometry()
        # 屏幕比最小尺寸还小时不再压窗口，靠左上角放，至少标题栏和左边缘够得着
        w = max(self.minimumWidth(), min(self.width(), av.width() - 40))
        h = max(self.minimumHeight(), min(self.height(), av.height() - 40))
        if (w, h) != (self.width(), self.height()):
            self.resize(w, h)
        x = av.left() + max(0, (av.width() - w) // 2)
        y = av.top() + max(0, (av.height() - h) // 2)
        self.move(x, y)

    def showEvent(self, e):
        super().showEvent(e)
        # 只在第一次显示时摆一次，之后窗口位置归用户
        if not getattr(self, "_placed_once", False):
            self._placed_once = True
            self._ensure_on_screen()

    # ---------------------------------------------------------- 菜单

    def _build_menu(self):
        m = self.menuBar().addMenu("工具")
        a = QAction("法线修复（粉色 / 通道反转）…", self)
        a.triggered.connect(self.open_repair)
        m.addAction(a)
        b = QAction("导出目录…", self)
        b.triggered.connect(self.pick_outdir)
        m.addAction(b)

        h = self.menuBar().addMenu("帮助")
        c = QAction("关于法线通道顺序", self)
        c.triggered.connect(lambda: QMessageBox.information(
            self, "通道顺序",
            "切线空间法线必须让 Z 轴落在蓝(B)通道：\n\n"
            "  R = nx ≈ 128\n  G = ny ≈ 128\n  B = nz ≈ 255\n\n"
            "R≈255、G/B≈128 就是通道反转，画面整片发粉。\n"
            "UE / Unity 用 DirectX(−Y)；Blender 部分管线用 OpenGL(+Y)。"))
        h.addAction(c)

    # ---------------------------------------------------------- 载入

    def log(self, text: str):
        self.log_box.appendPlainText(text)

    def pick_source(self):
        p, _ = QFileDialog.getOpenFileName(
            self, self._t("选择基础色贴图"), "",
            "图片 (*.png *.jpg *.jpeg *.tga *.tif *.tiff *.bmp *.webp)")
        if p:
            self.load(p)

    def pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, self._t("选择导出目录"), self.out_dir)
        if d:
            self.out_dir = d
            self._refresh_outdir()
            self.log(self._t(f"导出目录 -> {d}", f"Output folder -> {d}"))

    def _refresh_outdir(self):
        full = os.path.abspath(self.out_dir)
        home = os.path.expanduser("~")
        shown = "~" + full[len(home):] if full.startswith(home) else full
        for lb in self._outdir_labels:
            lb.setText("→ " + shown)
            lb.setToolTip(full)

    def load(self, path: str):
        try:
            rgb = load_rgb(path)
        except Exception as e:  # noqa: BLE001
            self.log(f"[失败] 读取 {path}: {e}")
            return
        self.src_rgb = rgb
        self.src_path = path
        h, w = rgb.shape[:2]
        self.ed_name = os.path.splitext(os.path.basename(path))[0]
        self.src_tile.set_array(rgb)
        self.log(self._t(f"载入 {os.path.basename(path)}  ({w}×{h})",
                         f"Loaded {os.path.basename(path)}  ({w}×{h})"))
        self.regenerate()

    # ---------------------------------------------------------- 生成

    def _params(self) -> PBRParams:
        step = self.rough_step
        # 高通半径跟着分辨率走：1024 用 4，4096 用 16，否则大图上只剩逐像素噪点
        detail_radius = max(2.0, self.out_size / 256.0)
        return PBRParams(
            normal_strength=3.0,
            normal_blur=max(0.5, step * 0.25),
            normal_flip_y=self.flip_y,
            normal_detail_radius=detail_radius,
            height_contrast=1.0,
            ao_radius=max(6.0, self.out_size / 85.0),
            rough_base=0.55,
            rough_detail=0.25 + step * 0.03,
        )

    def schedule_regen(self):
        if self.src_rgb is not None:
            self._regen_timer.start()

    def regenerate(self):
        if self.src_rgb is None:
            self.log("先选择一张图片")
            return
        src = self.src_rgb
        if self.out_size and max(src.shape[:2]) != self.out_size:
            src = resize(src, (self.out_size, self.out_size))

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.result = generate(src, self._params())
        finally:
            QApplication.restoreOverrideCursor()

        # 每个通道套用它自己记住的强度 / 对比度
        # MASK 不是独立通道，是其它十一张拼出来的结果，交给 update_composite 填；
        # 这里要是跟着循环，会先把上一张图的合成图塞回卡片上。
        self.display = {}
        self.cards["MASK"].set_array(None)
        for key, tile in self.cards.items():
            if key == "MASK":
                continue
            raw = self._raw_map(key)
            g, c = self._chan_params.get(key, (1.0, 1.0))
            arr = apply_channel(key, raw, g, c)
            self.display[key] = arr
            tile.set_array(arr)

        self.sphere.set_base(self.display)
        self.update_composite()
        self._update_status()
        self.select_channel(self.current_key)
        self.log(self._t(f"生成完成 · {self.out_size}px · 法线 Y={'+' if self.flip_y else '−'}",
                         f"Generated · {self.out_size}px · normal Y={'+' if self.flip_y else '−'}"))

    def _raw_map(self, key: str):
        if key == "MASK":
            return self._composite_arr
        if self.result is None:
            return None
        return self.result.maps.get(key)

    def _update_status(self):
        if not self.result:
            return
        nrm = self.display.get("NRM")
        if nrm is None:
            return
        arr = np.asarray(nrm)
        if arr.dtype != np.uint8:
            arr = np.clip(np.asarray(arr, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
        st = channel_stats(arr)
        ok = st["is_correct_order"]
        self.log(self._t("法线通道 " + ("正常 (Z→B)" if ok else "异常，请检查"),
                         "Normal channels " + ("OK (Z→B)" if ok else "broken, please check")))

    # ---------------------------------------------------------- 单选 / 参数

    @staticmethod
    def _size_text(arr) -> str:
        if arr is None:
            return "—"
        a = np.asarray(arr)
        if a.ndim < 2:
            return "—"
        return f"{a.shape[1]}×{a.shape[0]}"

    def _set_adjust_enabled(self, on: bool):
        for w in (self.s_gain, self.s_contrast, self.btn_reset):
            w.setEnabled(on)

    def select_channel(self, key: str):
        """把右侧大图切到 key；源图没有强度/对比度，用灰掉的控件表达。"""
        if key in self.cards:
            self.current_key = key
        self.src_tile.set_selected(key == "SRC")
        for k, tile in self.cards.items():
            tile.set_selected(k == key)

        if key == "SRC":
            w, h = (self.src_rgb.shape[1], self.src_rgb.shape[0]) if self.src_rgb is not None else (0, 0)
            self.big_code.setText("SRC")
            self.big_name.setText(self._t("源图"))
            self.big_info.setText(f"{w}×{h}" if w else "—")
            self.big_note.setText(self._t("原始输入贴图，所有通道都由它推出来"))
            self.big.set_array(self.src_rgb)
            self.lbl_sel.setText(self._t("源图 · 不可调整"))
            self._set_adjust_enabled(False)
            return

        tile = self.cards[key]
        self.big_code.setText(key)
        self.big_name.setText(self._t(tile.cn))
        arr = self.display.get(key)
        self.big_info.setText(self._size_text(arr))
        self.big.set_array(arr)
        hint_pair = CHANNEL_HINT.get(key, ("", ""))
        self.big_note.setText(
            (hint_pair[1] if self.lang == "en" else hint_pair[0]) if arr is not None
            else self._t("先选择一张贴图，这里会显示该通道的结果"))
        self.lbl_sel.setText(f"{key} · {self._t(tile.cn)}")
        # MASK 是合成结果，不是可调的源通道：给它打开滑杆会把 RGBA 拆成三通道
        self._set_adjust_enabled(arr is not None and key != "MASK")
        g, c = self._chan_params.get(key, (1.0, 1.0))
        self.s_gain.reset(g)
        self.s_contrast.reset(c)

    def _on_adjust(self):
        key = self.current_key
        if key not in self._chan_params or key == "MASK" or self.src_rgb is None:
            return
        self._chan_params[key] = [self.s_gain.value(), self.s_contrast.value()]
        self._reprocess(key)

    def reset_current(self):
        key = self.current_key
        if key not in self._chan_params or key == "MASK":
            return
        self._chan_params[key] = [1.0, 1.0]
        self.s_gain.reset(1.0)
        self.s_contrast.reset(1.0)
        self._reprocess(key)
        self.log(self._t(f"{key} 强度/对比度已重置", f"{key} gain/contrast reset"))

    def _reprocess(self, key: str):
        if key == "MASK":
            return
        g, c = self._chan_params.get(key, (1.0, 1.0))
        raw = self._raw_map(key)
        arr = apply_channel(key, raw, g, c)
        self.display[key] = arr
        self.cards[key].set_array(arr)
        if key == self.current_key:
            self.big.set_array(arr)
            self.big_info.setText(self._size_text(arr))
        if key in ("NRM", "COL", "RGH", "MET", "OCC"):
            self.sphere.set_base(self.display)
            self.update_composite()
        if key == "NRM":
            self._update_status()

    def toggle_checked(self, key: str):
        tile = self.cards[key]
        tile.set_checked(not tile.is_checked())
        n = sum(1 for t in self.cards.values() if t.is_checked())
        self._update_multi_btn()
        state = "参与" if tile.is_checked() else "退出"
        self.log(f"{key} {state}「导出全部通道」· 当前 {n} 个")

    def _update_multi_btn(self):
        n = sum(1 for t in self.cards.values() if t.is_checked() and t.key != "MASK")
        self.btn_multi.setText(
            self._t(f"导出勾选的 {n} 张", f"Export {n} checked") if n
            else self._t("导出勾选的通道", "Export checked"))

    def _set_fmt(self, text: str, quiet: bool = False):
        self.ext_mode = FMT_EXT.get(text, ".png").lstrip(".")
        for b in self.seg_fmt.buttons:
            b.setChecked(b.text() == text)
        for lb in self._fmt_labels:
            lb.setText(self._t(f"当前格式 {text} · 在顶部切换",
                               f"Format {text} · switchable at the top"))
        note = ""
        if text in FMT_NO_ALPHA:
            note = self._t(" · 无 alpha 通道", " · no alpha channel")
        if text in FMT_LOSSY:
            note += self._t(" · 有损，数据贴图勿用", " · lossy, avoid for data maps")
        if not quiet:
            self.log(self._t(f"导出格式 -> {text}", f"Format -> {text}")
                     + f" (.{self.ext_mode}){note}")

    # ---------------------------------------------------------- 合成

    def current_mix(self) -> dict[str, str]:
        return {ch: c.currentData() for ch, c in self.mix_combos.items()}

    def _refresh_sphere_label(self):
        """球下面那行小字：当前哪些通道是从打包图里读的，缺了哪几项。"""
        if not (hasattr(self, "lbl_sphere_src") and hasattr(self, "lbl_sphere_warn")):
            return
        mix = self.current_mix()
        used = {v for v in mix.values() if v not in ("—", None)}
        txt = " · ".join(f"{ch}={mix[ch]}" for ch in ("R", "G", "B", "A")
                         if mix[ch] not in ("—", None))
        self.lbl_sphere_src.setText(self._t("球上取自打包图：", "Sphere reads: ")
                                    + (txt or self._t("没有分配通道", "nothing assigned")))

        miss = [k for k in ("RGH", "MET", "OCC") if k not in used]
        if miss:
            self.lbl_sphere_warn.setText(self._t(
                f"打包图里没有 {'/'.join(miss)}，球上按默认值算",
                f"No {'/'.join(miss)} in the packed map — using defaults"))
            self.lbl_sphere_warn.show()
        else:
            self.lbl_sphere_warn.hide()

    def apply_preset(self, name: str):
        mix = PRESET_MIX[name]
        for ch, c in self.mix_combos.items():
            c.blockSignals(True)
            c.setCurrentIndex(max(0, c.findData(mix[ch])))
            c.blockSignals(False)
        self.update_composite()
        self.log(f"预设 {name}: " + " / ".join(f"{k}={v}" for k, v in mix.items()))

    def _on_alpha_mode(self, _text: str = ""):
        """分段按钮的文案会随语言变，所以用按钮索引判断当前档位，不看文字。"""
        self.alpha_mode = "opaque" if self.seg_alpha.buttons[1].isChecked() else "data"
        self._refresh_comp_warn()

    def _refresh_comp_warn(self):
        """打包图的 alpha 不是用来做透明的，是第四条数据。

        A 里装的是金属度这类数据时，看图器会拿它当透明度：金属度全 0 的图整张透明，
        看着像"什么都没导出"。这行把原因说清楚，并指路到上面的「不透明」开关。
        """
        if not hasattr(self, "lbl_comp_warn"):
            return
        if self._composite_arr is None:
            self.lbl_comp_warn.hide()
            return

        if self.alpha_mode == "opaque":
            self.lbl_comp_warn.setText(self._t(
                "A 写满 255：看图器能正常显示，A 的数据不写进文件",
                "Alpha forced to 255: viewable everywhere, the A channel is not written"))
            self.lbl_comp_warn.show()
            return

        mix = self.current_mix()
        ch = mix.get("A")
        if ch in ("—", None):
            self.lbl_comp_warn.hide()
            return
        amax = int(self._composite_arr[..., 3].max()) if self._composite_arr.shape[2] >= 4 else 255
        if amax >= 255:
            self.lbl_comp_warn.hide()
            return

        if amax == 0:
            msg = self._t(f"A:{ch} 全是 0，导出的图在普通看图器里是全透明（数据本身没错）",
                          f"A:{ch} is all 0, so viewers show the file as fully transparent "
                          f"(the data itself is fine)")
        else:
            msg = self._t(f"A:{ch} 没到 255，导出的图会偏透明（数据本身没错）",
                          f"A:{ch} is below 255, so viewers show the file faded "
                          f"(the data itself is fine)")
        self.lbl_comp_warn.setText(msg + self._t("　切上面「不透明」即可直接看",
                                                "  Switch to Opaque above to view it"))
        self.lbl_comp_warn.show()

    def update_composite(self):
        if self.result is None:
            return
        mix = self.current_mix()

        def get(ch):
            key = mix[ch]
            if key in ("—", None):
                return None
            return self.display.get(key)

        arrs = {ch: get(ch) for ch in ("R", "G", "B", "A")}
        if all(v is None for v in arrs.values()):
            # 四个槽位都空：没有东西可打包，合成图和球都退回原始通道
            self._composite_arr = None
            self.composite.set_array(None)
            self.lbl_comp.setText("")
            self.display.pop("MASK", None)
            mask_tile = self.cards.get("MASK")
            if mask_tile is not None:
                mask_tile.set_array(None)
                if self.current_key == "MASK":
                    self.select_channel("MASK")
            self.sphere.set_packed(None, mix, self.display)
            self._refresh_sphere_label()
            self._refresh_comp_warn()
            return

        rgba = pack(arrs["R"], arrs["G"], arrs["B"], arrs["A"], default_alpha=255)
        g = self.s_gamma.value() if hasattr(self, "s_gamma") else 1.0
        if abs(g - 1.0) > 1e-3:
            f = rgba[..., :3].astype(np.float32) / 255.0
            rgba[..., :3] = np.clip((f ** (1.0 / g)) * 255.0 + 0.5, 0, 255).astype(np.uint8)

        self._composite_arr = rgba
        preview = rgba.copy()
        preview[..., 3] = 255
        self.composite.set_array(preview)

        mask_tile = self.cards.get("MASK")
        if mask_tile is not None:
            self.display["MASK"] = preview
            mask_tile.set_array(preview)

        self.lbl_comp.setText(" · ".join(
            f"{ch}:{mix[ch]}" for ch in ("R", "G", "B", "A") if mix[ch] not in ("—", None)))

        # 3D 球跟着打包图走：把 RGBA 按当前分配解回语义通道再渲染
        self.sphere.set_packed(rgba, mix, self.display)
        self._refresh_sphere_label()
        self._refresh_comp_warn()

        if self.current_key == "MASK":
            self.select_channel("MASK")

    # ---------------------------------------------------------- 导出

    def _ctx(self):
        return self.ed_name or "texture", os.path.abspath(self.out_dir), "." + self.ext_mode

    def _fmt_warn(self, data_map: bool = False) -> str:
        """格式本身的风险提示：丢 alpha / 有损压坏数据贴图。"""
        fmt = self.seg_fmt.active_text()
        if fmt in FMT_NO_ALPHA:
            return self._t(f"注意：{fmt} 存不了 alpha，A 通道会被丢掉",
                           f"Note: {fmt} has no alpha channel, A will be lost")
        if data_map and fmt in FMT_LOSSY:
            return self._t(f"注意：{fmt} 是有损压缩，法线 / 粗糙度这类数据贴图会被压坏",
                           f"Note: {fmt} is lossy, data maps like normals will degrade")
        return ""

    def export_one(self, key: str):
        if self.result is None:
            QMessageBox.warning(self, self._t("还没生成"), self._t("请先选择一张贴图。"))
            return
        # MASK 卡片上显示的是 alpha 被拉满的预览图，直接写盘会把 A 里的数据丢掉，
        # 所以统一走合成导出那条路（它写的是 _composite_arr 本体 + 「A 写入」设置）
        if key == "MASK":
            self.export_composite()
            return
        arr = self.display.get(key)
        if arr is None:
            QMessageBox.warning(self, self._t("通道为空"),
                                self._t(f"{key} 还没有内容。", f"{key} is empty."))
            return
        stem, out_dir, ext = self._ctx()
        try:
            res = export_maps({key: arr}, out_dir, stem, [key], ext=ext)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, self._t("单图导出失败"), str(e))
            self.log(f"[失败] {e}")
            return
        warn = self._fmt_warn(key not in ("COL", "MASK"))
        for r in res:
            self.log(f"  {'OK ' if r.ok else 'BAD'} {os.path.basename(r.path)} {r.note}")
        if warn:
            self.log(warn)

    def export_checked(self):
        if self.result is None:
            QMessageBox.warning(self, self._t("还没生成"), self._t("请先选择一张贴图。"))
            return
        keys = [k for k, c in self.cards.items() if c.is_checked() and k != "MASK"]
        if not keys:
            QMessageBox.warning(self, self._t("没有勾选"),
                                self._t("在通道卡片右上角点圆点选中要导出的通道。"))
            return
        stem, out_dir, ext = self._ctx()
        maps = {k: self.display[k] for k in keys if self.display.get(k) is not None}
        try:
            res = export_maps(maps, out_dir, stem, list(maps), ext=ext)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, self._t("导出失败"), str(e))
            self.log(f"[失败] {e}")
            return
        bad = [r for r in res if not r.ok]
        warn = self._fmt_warn(data_map=True)
        self.log(self._t(f"单图导出 {len(res)} 张 -> {out_dir}", f"Exported {len(res)} maps -> {out_dir}"))
        if warn:
            self.log(warn)
        for r in res:
            self.log(f"  {'OK ' if r.ok else 'BAD'} {os.path.basename(r.path)} {r.note}")
        if bad:
            QMessageBox.warning(self, self._t("有问题"),
                                "\n".join(f"{os.path.basename(r.path)}: {r.note}" for r in bad))

    def export_composite(self):
        if self._composite_arr is None:
            QMessageBox.warning(self, self._t("还没生成"), self._t("请先选择一张贴图。"))
            return
        stem, out_dir, ext = self._ctx()
        path = os.path.join(out_dir, f"{stem}_MASK{ext}")
        arr = self._composite_arr
        if self.alpha_mode == "opaque" and arr.ndim == 3 and arr.shape[2] >= 4:
            arr = arr.copy()
            arr[..., 3] = 255
        try:
            save_image(arr, path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, self._t("合成导出失败"), str(e))
            self.log(f"[失败] {e}")
            return
        self.log(self._t(f"合成导出 -> {path}", f"Packed export -> {path}"))
        if self._fmt_warn():
            self.log(self._fmt_warn())
        if self.lbl_comp_warn.isVisible() and self.alpha_mode == "data":
            self.log(self._t("  A 里装的是数据不是透明度：这张图在常规看图器里会偏透明，"
                             "想直接看就把「A 写入」切成「不透明」再导一次。",
                             "  A carries data, not opacity: this file looks faded in ordinary "
                             "viewers. Switch A write to Opaque and export again to view it."))

    def export_both(self):
        if self.result is None:
            QMessageBox.warning(self, self._t("还没生成"), self._t("请先选择一张贴图。"))
            return
        self.export_checked()
        self.export_composite()

    # ---------------------------------------------------------- 法线修复

    def open_repair(self):
        p, _ = QFileDialog.getOpenFileName(self, self._t("选择要修复的法线贴图"), "",
                                           "图片 (*.png *.jpg *.tga *.tif *.tiff *.bmp)")
        if not p:
            return
        arr = np.asarray(load_rgb(p) * 255 + 0.5, dtype=np.uint8)
        st = channel_stats(arr)
        msg = (f"Z 轴当前在 {st['z_channel']} 通道\n"
               f"R{st['mean'][0]*255:.0f} / G{st['mean'][1]*255:.0f} / B{st['mean'][2]*255:.0f}\n\n"
               + ("检测到粉色法线（通道反转），是否交换 R/B 修复？"
                  if not st["is_correct_order"] else "通道顺序正常，仍要重新编码一遍吗？"))
        if QMessageBox.question(self, "法线修复", msg) != QMessageBox.Yes:
            return
        fixed = repair_normal(arr)
        out = os.path.splitext(p)[0] + "_fixed.png"
        save_image(fixed, out)
        st2 = channel_stats(fixed)
        self.log(f"修复 -> {out}  B={st2['mean'][2]*255:.0f}")
        QMessageBox.information(self, "完成",
                                f"已保存：\n{out}\n\nZ 轴现在在 {st2['z_channel']} 通道。")

    # ---------------------------------------------------------- 拖放

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isfile(p):
                self.load(p)
                break


def run():
    import sys

    # 不设 AppUserModelID 的话，任务栏会归到 python 进程用默认图标
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ChordStudio.PBR")
        except Exception:  # noqa: BLE001
            pass

    app = QApplication.instance() or QApplication(sys.argv)
    if os.path.exists(_ICON_PATH):
        app.setWindowIcon(QIcon(_ICON_PATH))
    w = MainWindow()
    w.show()
    return app.exec()
