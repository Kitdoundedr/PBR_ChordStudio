"""界面层冒烟测试：小图选择 + 强度/对比度 + 两条导出路径。

不弹窗（QT_QPA_PLATFORM=offscreen），直接构造 MainWindow 跑一遍：
  1. 载入样图 -> 生成 -> 12 张通道小图都有像素
  2. 选中小图 -> 右侧大图跟着切换
  3. 拖强度/对比度 -> 只作用于当前通道，且法线仍是 B=Z
  4. 「贴图预览」单图导出 -> 文件落盘
  5. 「通道打包」合成导出 -> _MASK 落盘且 alpha 不是全 0
  6. 顶栏九种格式都能落盘，后缀正确
  7. 中英切换：文案变了，但步进器 / 滑块的数值不被改写
  8. 3D 球按当前 R/G/B/A 分配解包：改分配球就变，分配全空退回原始通道
  9. A 通道写法：数据原样写 / 写满 255，两种都能落盘且只动 alpha
 10. MASK 只读：滑杆灰掉、调参不改合成图、单图导出写的是打包数据而不是预览 alpha
 11. 窗口被摆到屏幕外时自动拉回可用区（双屏负坐标场景）

直接运行：  python tests/test_ui_export.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from pbr_app.ui import CHANNELS, FORMATS, FMT_EXT, MainWindow, apply_channel  # noqa: E402
from pbr_core.imgio import save_image  # noqa: E402
from pbr_core.normal import channel_stats  # noqa: E402


def _sample(path: str) -> str:
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:256, 0:256]
    base = 0.5 + 0.25 * np.sin(x / 9.0) * np.cos(y / 7.0)
    rgb = np.stack([base * 0.8, base, base * 0.7], axis=2)
    rgb = np.clip(rgb + rng.normal(0, 0.02, rgb.shape), 0, 1)
    save_image(rgb.astype(np.float32), path)
    return path


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    tmp = tempfile.mkdtemp(prefix="pbr_ui_")

    src = _sample(os.path.join(tmp, "src.png"))
    w = MainWindow()
    w.out_dir = tmp
    w.out_size = 256
    w.load(src)
    app.processEvents()

    fails: list[str] = []

    # 1. 所有小图都有像素
    if set(w.cards) != {k for k, _ in CHANNELS}:
        fails.append("小图数量不对")
    for key, card in w.cards.items():
        if key == "MASK":
            continue
        if w.display.get(key) is None:
            fails.append(f"{key} 没有生成")

    # 2. 法线仍是 B=Z
    nrm = np.asarray(w.display["NRM"])
    if nrm.dtype != np.uint8:
        nrm = np.clip(nrm * 255, 0, 255).astype(np.uint8)
    st = channel_stats(nrm)
    if not st["is_correct_order"]:
        fails.append(f"法线通道异常 z={st['z_channel']}")

    # 3. 选中 + 滑块只作用于当前通道，且法线不会被拉成粉色
    w.select_channel("NRM")
    app.processEvents()
    if w.current_key != "NRM":
        fails.append("选中小图没有切换 current_key")
    shown = w.big._img
    want = np.asarray(w.display["NRM"]).shape
    if shown is None or (shown.width(), shown.height()) != (want[1], want[0]):
        fails.append("右侧大图没有跟着切到 NRM")
    w.s_gain.slider.setValue(180)                   # 强度 1.80
    app.processEvents()
    if list(w._chan_params["RGH"]) != [1.0, 1.0]:
        fails.append("调 NRM 的强度串到了 RGH")
    w.select_channel("RGH")
    w.s_contrast.slider.setValue(260)               # 对比度 2.60
    app.processEvents()
    nrm2 = np.asarray(w.display["NRM"])
    if nrm2.dtype != np.uint8:
        nrm2 = np.clip(nrm2 * 255, 0, 255).astype(np.uint8)
    if not channel_stats(nrm2)["is_correct_order"]:
        fails.append("拉强度后法线通道翻转了")
    if np.allclose(np.asarray(w.display["NRM"], dtype=np.float32),
                   nrm.astype(np.float32)):
        fails.append("NRM 强度滑块没有生效")

    # 单通道后处理：z 恒为正
    probe = apply_channel("NRM", nrm, 2.0, 3.0)
    if float(probe[..., 2].min()) < 127:
        fails.append("apply_channel 把法线 z 压成负数了（会变粉）")

    # 4. 单图导出
    w.export_one("NRM")
    w.export_checked()
    singles = [f for f in os.listdir(tmp) if f.endswith("_NRM.png") or f.endswith("_RGH.png")]
    if len(singles) < 2:
        fails.append(f"单图导出文件缺失：{os.listdir(tmp)}")

    # 5. 合成导出
    w.export_composite()
    mask = os.path.join(tmp, "src_MASK.png")
    if not os.path.exists(mask):
        fails.append("合成图没有导出")
    else:
        from pbr_core.imgio import load_rgb
        m = load_rgb(mask)
        if float(m.mean()) <= 0.0:
            fails.append("合成图全黑")

    # 6. 顶栏九种格式
    fmt_dir = tempfile.mkdtemp(prefix="pbr_fmt_")
    w.out_dir = fmt_dir
    if [b.text() for b in w.seg_fmt.buttons] != FORMATS:
        fails.append("顶栏格式按钮不齐")
    for i, fmt in enumerate(FORMATS):
        w.seg_fmt.buttons[i].click()
        app.processEvents()
        if w.ext_mode != FMT_EXT[fmt].lstrip("."):
            fails.append(f"{fmt} 没有写进 ext_mode")
        w.export_one("COL")
    got = {os.path.splitext(f)[1].lower() for f in os.listdir(fmt_dir)}
    want_ext = {e.lower() for e in FMT_EXT.values()}
    if got != want_ext:
        fails.append(f"格式落盘不全：{sorted(got)}")

    # 7. 中英切换
    res_before = w.step_res.value_label.text()
    gain_before = w.s_gain.value_label.text()
    w.seg_lang.buttons[1].click()
    app.processEvents()
    if w.lang != "en" or w.btn_load.text() != "Load Image":
        fails.append("切到 English 没生效")
    if w.step_res.value_label.text() != res_before:
        fails.append("语言切换改坏了分辨率数值")
    if w.s_gain.value_label.text() != gain_before:
        fails.append("语言切换改坏了滑块数值")
    w.seg_lang.buttons[0].click()
    app.processEvents()
    if w.lang != "cn" or w.btn_load.text() != "选择图片":
        fails.append("切回中文没生效")

    # 8. 3D 球跟着打包分配走
    def set_mix(m: dict):
        for ch, c in w.mix_combos.items():
            c.blockSignals(True)
            c.setCurrentIndex(max(0, c.findData(m[ch])))
            c.blockSignals(False)
        w.update_composite()
        app.processEvents()

    def sphere_bytes():
        img = w.sphere._pm.toImage()
        return np.frombuffer(img.constBits(), np.uint8).copy()

    set_mix({"R": "OCC", "G": "RGH", "B": "MET", "A": "—"})
    keys = set(w.sphere._maps)
    if not {"OCC", "RGH", "MET"} <= keys:
        fails.append(f"球没有按 ORM 分配取通道：{sorted(keys)}")
    if "DIS" in keys:
        fails.append("没分配 DIS 时球还在读高度")
    orm_px = sphere_bytes()

    set_mix({"R": "RGH", "G": "MET", "B": "OCC", "A": "DIS"})
    if "DIS" in w.sphere._maps:
        fails.append("DIS 没有并进法线（球不该再单独渲染它）")
    if np.array_equal(orm_px, sphere_bytes()):
        fails.append("改了打包分配球没有任何变化")

    set_mix({"R": "—", "G": "—", "B": "—", "A": "—"})
    if w.sphere._maps.get("COL") is None:
        fails.append("分配全空时球没有退回原始通道")

    # 9. A 通道两种写法：数据原样 / 写满 255（RHAM 的 A=金属度，全 0 时会导出全透明图）
    from PIL import Image

    set_mix({"R": "RGH", "G": "DIS", "B": "OCC", "A": "MET"})
    w.seg_fmt.buttons[0].click()                    # 回到 PNG（第 6 步最后停在 DDS）
    app.processEvents()
    alpha_dir = tempfile.mkdtemp(prefix="pbr_alpha_")
    w.out_dir = alpha_dir
    mask = os.path.join(alpha_dir, "src_MASK.png")

    w.seg_alpha.buttons[0].click()                  # 数据
    app.processEvents()
    w.export_composite()
    a = None
    if not os.path.exists(mask):
        fails.append("A 数据模式没有导出")
    else:
        a = np.asarray(Image.open(mask))
        if a.ndim != 3 or a.shape[2] != 4:
            fails.append("合成图丢了 alpha 通道")
        elif not np.array_equal(a[..., 3], w._composite_arr[..., 3]):
            fails.append("数据模式没有原样写 A")
        elif int(a[..., 3].max()) < 255 and w.lbl_comp_warn.isHidden():
            fails.append("A 不满 255 却没给透明提示")

    w.seg_alpha.buttons[1].click()                  # 不透明
    app.processEvents()
    w.export_composite()
    b = np.asarray(Image.open(mask))
    if int(b[..., 3].min()) != 255:
        fails.append("不透明模式 alpha 没写满 255")
    if a is not None and a.ndim == 3 and a.shape[2] == 4 and \
            not np.array_equal(a[..., :3], b[..., :3]):
        fails.append("不透明模式改动了 RGB 数据")

    # 10. MASK 是合成结果，不能当普通通道调；单图导出也必须走合成数据（不然 A 会被拉满）
    set_mix({"R": "RGH", "G": "DIS", "B": "OCC", "A": "MET"})
    w.seg_alpha.buttons[0].click()                  # 回「数据」
    app.processEvents()
    w.select_channel("MASK")
    app.processEvents()
    if w.s_gain.isEnabled() or w.s_contrast.isEnabled():
        fails.append("MASK 的强度/对比度没灰掉（拖了会把 RGBA 拆成三通道）")
    mask_before = np.asarray(w.display["MASK"]).copy()
    w.s_gain.slider.setValue(170)
    w._on_adjust()
    app.processEvents()
    m_after = np.asarray(w.display["MASK"])
    if m_after.ndim != 3 or m_after.shape[2] != 4:
        fails.append(f"MASK 被调参后丢了 alpha：shape={m_after.shape}")
    if m_after.dtype != np.uint8:
        fails.append(f"MASK 被调参后 dtype 变成 {m_after.dtype}")
    if not np.array_equal(mask_before, m_after):
        fails.append("MASK 的滑杆还能改到合成图")

    one_dir = tempfile.mkdtemp(prefix="pbr_mask1_")
    w.out_dir = one_dir
    w.export_one("MASK")
    one_path = os.path.join(one_dir, "src_MASK.png")
    if not os.path.exists(one_path):
        fails.append("MASK 单图导出没落盘")
    else:
        m = np.asarray(Image.open(one_path))
        if m.ndim != 3 or m.shape[2] != 4:
            fails.append("MASK 单图导出丢了 alpha")
        elif not np.array_equal(m[..., 3], w._composite_arr[..., 3]):
            fails.append("MASK 单图导出写的是预览用的 alpha，不是打包数据")

    # 分配全空：MASK 卡片不能留着上一张的合成图
    set_mix({"R": "—", "G": "—", "B": "—", "A": "—"})
    if "MASK" in w.display or w.cards["MASK"].panel._img is not None:
        fails.append("分配全空后 MASK 卡片仍残留旧合成图")

    # 11. 窗口被摆到屏幕外时要能自己拉回来（双屏 / 负坐标竖屏踩过）
    from PySide6.QtGui import QGuiApplication

    w.move(-9000, -9000)
    w._ensure_on_screen()
    app.processEvents()
    av = QGuiApplication.primaryScreen().availableGeometry()
    if not av.contains(w.frameGeometry().topLeft()):
        fails.append(f"越界窗口没拉回：pos={w.pos()} 可用区={av}")
    if (w.width(), w.height()) != (max(w.minimumWidth(), min(1500, av.width() - 40)),
                                   max(w.minimumHeight(), min(930, av.height() - 40))):
        fails.append(f"拉回后尺寸不对：{w.size()}")

    w.close()

    if fails:
        for f in fails:
            print("  FAIL ", f)
        print(f"\n{len(fails)} 项失败")
        return 1
    print("  PASS  test_cards_and_single_export")
    print("  PASS  test_composite_export")
    print("  PASS  test_channel_slider_keeps_normal_blue")
    print("  PASS  test_top_format_bar")
    print("  PASS  test_language_switch_keeps_values")
    print("  PASS  test_sphere_follows_mix")
    print("  PASS  test_alpha_mode_export")
    print("  PASS  test_mask_is_readonly_preview")
    print("  PASS  test_window_kept_on_screen")
    print("\n10/10 通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
