# PBR 贴图生成器 · 重写版

一个不依赖 OpenCV、通道顺序绝对不会搞反的 PBR 贴图工作室。
PyCharm 直接打开本目录即可运行。

---

## 一、粉色法线的根因（针对旧版工具）

对 `T_Ornament_木雕_A01_Bc_NRM.png`（2048×2048, RGB）逐通道统计：

| 通道 | 均值 | 标准差 | 说明 |
|------|------|--------|------|
| R | **252.7** | 5.3 | 几乎条条拉满 —— 这是 Z 轴 |
| G | 127.6 | 15.9 | 切线 Y |
| B | **127.6** | 17.4 | 切线 X |

正常法线应该是 `B≈255`、`R/G≈128`（整体偏蓝紫）。
这张图把 **Z 轴写进了红通道**，于是屏幕上是 `#FF8080` 一片粉。

也就是说：**RGB 三个通道被反转成了 (Z, Y, X)**，等价于 BGR/RGB 顺序搞混。
数学上向量长度仍然是 1，所以任何“看起来像不像法线”的直觉判断都发现不了，
只有对比“哪个通道均值最高 / 方差最小”才能定位。

同一张贴图另有一份 `Downloads/..._NRM.png` 是 `B=249.2 / R=G=127.3`，完全正常，
可以确认问题出在**导出那一步**，不是算法本身。

### 旧版工具最可能的三种写法（对应改法）

```python
# ✗ 错误 1：用 OpenCV 算完直接交给 PIL 保存
arr = cv2.Sobel(...)              # OpenCV 是 BGR 习惯
Image.fromarray(arr).save(...)    # 当成 RGB 存 -> R/B 互换

# ✗ 错误 2：手工合并通道时顺序写反
Image.merge("RGB", (z, y, x))     # 应该是 (x, y, z)

# ✗ 错误 3：本来就已是 RGB，却又做了一次转换
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 反转两次 = 还是错的
Image.fromarray(rgb).save("_NRM.png")
```

本工程的做法：`pbr_core/normal.py::encode_normal()` 里显式
`np.stack([nx, ny, nz], axis=2)`，写死 `(X, Y, Z) -> (R, G, B)`；
导出前 `assert_channel_order(strict=True)` 再拦一道；
导出后 `_verify_saved()` 把文件重新读回来再验一遍。三道关卡，粉色不可能再出现。

---

## 二、目录结构

```
PBRTexStudio/
├── main.py                  程序入口（PyCharm 右键运行 / 打包 EXE 的入口）
├── ChordStudio.spec         PyInstaller 打包配置（onedir）
├── requirements.txt
├── assets/                  bg.jpg 背景 / mascot_white.webp 看板娘 / app.ico 应用图标
├── dist/ChordStudio/        打包产物：文件夹版，双击 ChordStudio.exe
├── dist_onefile/            打包产物：单文件版 ChordStudio_single.exe
├── tools/
│   ├── make_mascot.py       视频 -> ISNet-anime 抠图 -> 透明动画（任意背景）
│   ├── make_mascot_white.py 纯白背景专用：方形补边推理 + 白底去污染 + 统一裁切
│   └── shot.py              界面截图（开发用，默认用程序化样图）
├── tests/
│   ├── test_channels.py     7 条通道顺序回归测试
│   └── test_ui_export.py    界面冒烟：卡片滑块 + 两条导出路径
├── pbr_core/                纯算法层，不依赖 GUI，可单独 import
│   ├── imgio.py             读写 + 盒式/高斯模糊（只用 PIL+numpy）
│   ├── normal.py            法线生成 / 编码 / 诊断 / 修复 / 自检
│   ├── maps.py              AO、粗糙、金属、曲率、光泽…
│   ├── packing.py           RGBA 通道打包（ORM/RMA/RHAM）
│   ├── pipeline.py          PBRParams + 一整套生成
│   ├── exporter.py          命名规则 + 落盘 + 回读校验
│   └── cli.py               命令行：gen / check / fix
└── pbr_app/
    └── ui.py                仪表盘界面（玻璃面板 + 3D 材质球 + 看板娘）
```

## 二·五、界面说明

顶栏一条玻璃条：左边两个 tab（两条导出路径），右边是**九种可导出格式**。
「贴图预览」页是**左选右看**的结构：

```
┌ 顶栏  ● ChordStudio · CHORD PBR 材质分解   [格式 PNG TGA WebP JPG ICO BMP GIF TIF DDS] [拖放图片]┐
│      [贴图预览] [通道打包]                    CHORD · Ublsoft   本地离线                        │
├ 左栏 156 ────────┬ 贴图预览页 ─────────────────────────────────────────────────┤
│ ChordStudio      │ ┌ 通道 ───────────┐  ┌ 大图 ──────────────────────────┐   │
│  选择图片         │ │ ▫ ▫ ▫ ▫  13 格   │  │ COL 基础色         1024×1024  │   │
│  生成贴图         │ │ ▫ ▫ ▫ ▫  源图+12 │  │                              │   │
│  导出全部通道      │ │ ▫ ▫ ▫ ▫  通道    │  │     选中的那张铺满显示        │   │
│  ── 推理参数 ──   │ ├ 调整 ───────────┤  │                              │   │
│  分辨率 1024×1024 │ │ 强度  ───●────  │  └──────────────────────────────┘   │
│  粗糙度步长 − 5 + │ │ 对比度 ──●─────  │  底下说明这通道是什么数据            │
│  法线Y Dx/Gl      │ ├ 导出 ───────────┤                                      │
│  看板娘（上移）    │ │ 当前格式 PNG    │   ← 通道打包页：混合选项 + 3D | 合成图│
│  ── CN | EN ──    │ │ 导出当前通道    │──────────────────────────────────────┘
└───────────────────┴─┴────────────────┘
```

| 区域 | 内容 |
|------|------|
| 顶栏 | 圆点 logo、`贴图预览 / 通道打包` 切换、**九格格式条**、`CHORD · Ublsoft`、右上角虚线拖放框（可点击） |
| 左栏（全局） | 品牌卡 + **选择图片 / 生成贴图 / 导出全部通道** + 推理参数（分辨率四档 `1024×1024 / 2048×2048 / 4096×4096 / 8192×8192`、粗糙度步长、法线 Y）+ 法线修复 + 看板娘 + **看板娘脚下居中的 CN / EN** |
| 贴图预览·左上 | 13 个小图（源图 + 12 通道），**点一下右边就切到那张**；右上角圆点决定是否参与批量导出 |
| 贴图预览·左下 | 强度 / 对比度横条（只作用于当前选中的那张）+ 重置 |
| 贴图预览·左下 | 导出选项：当前格式提示、**导出当前通道**、**导出勾选的 N 张**、导出目录（点击可改） |
| 贴图预览·右侧 | 选中通道的大图，标题栏显示代号、名称、像素尺寸，底部一行说明这通道是什么 |
| 通道打包页 | 混合选项（R/G/B/A + 预设）与 3D 材质预览在左，合成结果大图与 Gamma / **A 写入** / 导出在右 |

### 导出格式（顶栏九格）

| 格式 | 备注 |
|------|------|
| PNG | 默认；唯一支持 16 位单通道的格式 |
| TGA | 32 位带 alpha，DCC 最通用 |
| WebP | 有损（quality 95），体积小 |
| JPG | 有损，**无 alpha** |
| ICO | 最大 256×256，超出会自动等比缩 |
| BMP | 无 alpha |
| GIF | 256 色，无 alpha |
| TIF | LZW 压缩，保留 alpha |
| DDS | **手写 128 字节 DDS_HEADER**，未压缩 32 位 BGRA。<br>PIL 只能读不能写；不压 BC1/BC5 是故意的，法线被 DXT 压过会出色块 |

选中丢 alpha 或有损的格式时，导出日志会直接提示风险（`注意：JPG 存不了 alpha…`、
`注意：JPG 是有损压缩，法线 / 粗糙度这类数据贴图会被压坏`）。

### 中 / 英切换

看板娘脚下的 `CN | EN`。实现上没有为每个控件手写对照表：第一次切换时把控件当前的
中文存进动态属性 `cnText`，之后来回切都读它（`_apply_lang()`），只要译文进了 `_EN`
字典就自动生效。动态文本（勾选数量、通道名、格式提示、导出目录）走 `_t()` 单独处理；
`Stepper` / `Slider` 的数值标签标了 `nol10n`，不会被语言切换改写。

### 两条导出，互相独立

| 按钮 | 位置 | 产物 |
|------|------|------|
| 导出当前通道（单张） | 贴图预览页左下「导出」 | `前缀_COL/NRM/RGH/…`，右侧大图是哪张就出哪张 |
| 导出勾选的 N 张 | 贴图预览页左下「导出」 | 小图右上角圆点勾选的通道一起出，全选等于一整套 |
| 导出全部通道 | 全局左栏 | 同上，入口在左栏（勾选状态共用） |
| 导出 RGBA 贴图 | 通道打包页 | 按 R/G/B/A 规则打包成一张 `前缀_MASK.<当前格式>` |
| 单图 + 合成 一起导出 | 通道打包页 | 上面两件事一起做 |

法线在写盘前会过 `assert_channel_order(strict=True)`，写完立刻回读校验，
日志里会打印 `法线通道 正常 (Z→B)`。

### A 通道是数据，不是透明度

打包图的 alpha 装的是**第四条数据**（RHAM 里就是金属度），看图器却会把它当透明度
合成到背景上：金属度全 0 的图 → alpha 全 0 → 打开看着**整张全透明，像没导出东西**。
数据其实没错，R/G/B 里该有的都在。

打包页右下的 **`A 写入`：`数据` / `不透明`** 就是给这件事准备的：

- `数据`（默认）：原样写 A，交给引擎读 `.a`。用普通看图器看着透明属正常现象
- `不透明`：A 写满 255，任何看图器都能正常显示；A 的数据不会写进文件

选 `数据` 且 A 不满 255 时，合成图下面会有一行橙色提示说明原因；
导出日志里也会重复一遍。想要一张「能直接看」的打包图，除了切 `不透明`，
也可以把 A 槽位改成 `空` —— 那样 alpha 自动补 255。

### 当前通道的「强度 / 对比度」

不是摆设，直接作用于该通道的数据，预览与导出完全一致；每个通道各记一套值，
切回去还在（`_chan_params`）：

- 灰度通道：先做围绕 0.5 的对比度曲线，再按强度缩放振幅
- 法线通道：强度缩放切线空间 xy（重新归一化，z 恒为正），
  对比度对 xy 幅值做 `tanh` 饱和（压平弱细节、放大强细节）
- 两个值都是 1.00 时原样返回，默认不改变数据；`重置` 一键回到 1.00

几个实现细节：

- **3D 材质球**：`render_sphere()` 把**整套材质**贴到球面（不是只贴法线）：
  基础色当 albedo、法线做切空间变换、粗糙度决定高光宽窄、金属度切漫反射并换成
  带色的环境反射、AO 压暗凹陷。纯 numpy 算，不依赖 OpenGL；`SphereView` 负责
  拖拽改光源方向、滚轮缩放、球底接地阴影。
  贴上去之前先降采样到 512px（PIL `BOX` 一次缩整张 RGBA），否则 4K 法线在球面上就是一片雪花

  **球预览的是「打包图」本身**，不是原始通道：`unpack_mix()` 按当前
  `R/G/B/A` 的分配把合成图拆回语义通道（`R=OCC` 就把 R 通道当 AO 用），
  再交给 `render_sphere()`。所以

  - 换预设（RHAM ⇄ ORM ⇄ RMA）球通常**不变** —— 打包只是搬家，语义没变，
    这恰好说明「这张图没丢通道」
  - 漏掉某个通道（比如 A 留空、MET 没打进去）球就**明显变了**，
    球下小字会标出「打包图里没有 MET，球上按默认值算」
  - 法线和基础色本来就不进打包图，从原始通道补；`DIS` 打进去时它的梯度会
    并进法线当凹凸（`_with_bump()`），所以球上能看出起伏

  三个容易写错的地方（旧版就是这么写成"蓝塑料球"的）：

  1. **底必须是基础色**，不能硬编码成某个颜色，否则换什么素材都是同一个球
  2. **法线要走切空间变换** `n = T·nx + B·ny + N·nz`，
     直接把 `nx` 加到屏幕坐标的 `x` 上只是把细节当成微小抖动，方向还是错的
  3. **屏幕 Y 向下**，所以光源的 y 取 `-sin(az)`、反射的"天空"取 `0.5 - ry·2.5`，
     写反了整颗球就变成底光，暗部全在上半部分
- **合成图预览**：A 通道常常是金属度（默认为 0），预览时补满 alpha，
  导出文件仍保留真实 A 数据
- **MASK 卡片是只读的**：它显示的就是打包结果本身，所以图画出来是**彩色的**
  （R / G / B 各装一张灰度图，A 里是数据不是透明度）。三件事跟着定了下来：
  - 强度 / 对比度对它**灰掉**。`apply_channel()` 走的是 `y[..., :3]` 的灰度分支，
    真让它跑一遍会把 RGBA 切成三通道 float，alpha 直接丢
  - 分配全空时卡片要清空，否则会留着上一张图的合成结果
  - 「导出当前通道」落在 MASK 上时改走 `export_composite()`：卡片上那份预览
    alpha 已被拉满，直接写盘等于把 A 里的数据抹了
- **小图填满格子**：`ImagePanel(cover=True)` 按 `max()` 比例缩放后裁切，缩略图不留白，
  大图仍是 `contain`（贴图要能看全）
- **小窗口**：预览页左列套了 `QScrollArea`。窗口压到最小高度（1180×760）时
  卡片会超出可视区，不加滚动就会互相压住
- **缩略图**：`_downsample()` 先做整数倍**面积平均**再交给 Qt。
  直接 `QImage.scaled(SmoothTransformation)` 缩到 1/7 会把法线的高频细节
  走样成雪花噪点，这是最容易踩的坑
- **分辨率步进器**：`Stepper` 支持传一份 `labels`，值还是 `int`（给管线用），
  显示的是 `1024×1024` 这种完整尺寸。156px 的左栏里「标签 + 步长条」放一行会
  把数字压掉，所以「分辨率 / 粗糙度步长」两个标签都改成**独立一行**、步长条独占整行
- **主题**：卡片是 `qlineargradient` 的玻璃底（上亮下暗，`rgba(60,54,80,0.82)` →
  `rgba(34,29,47,0.76)`），边框统一 `rgba(255,255,255,0.24)` 的浅透明色；`选中态`
  （tab / 分段按钮 / 小图边框 / 勾选圆点）一律是浅白色块，和底色拉开层次。
  背景用未模糊的 `bg.jpg` 压一层 92 alpha 暗罩，比早先的模糊版清晰，但卡片内文字仍够看
- **看板娘脚下的语言按钮**：`MascotCard.footer` 前后各一条 `addStretch(1)`，
  按钮落在卡片水平中线上；`resizeEvent` 会按 footer 高度给动画让位
- **窗口位置**：`showEvent` 第一次显示时调 `_ensure_on_screen()`。双屏（尤其左侧
  负坐标的竖屏）下 Windows 会按主屏信息算位置，把窗口整块摆到屏幕外、只剩任务栏图标。
  规则：已经完整落在某块屏的可用区内就**不动**（用户拖过的位置要保住）；否则按
  窗口中心所在屏（找不到就用主屏）居中，尺寸按可用区夹一次但不低于 `minimumSize`，
  屏比最小尺寸还小时贴左上角放，至少标题栏够得着
- **看板娘**：`assets/mascot_white.webp`（169 帧 144×300、8 位 alpha）。
  由 `tools/make_mascot_white.py` 从白底视频生成：**白底去污染**是关键，
  白色抗锯齿边缘其实是 `C = a·F + (1-a)·255`，不反解掉就会在深色面板上留一圈白毛边。
  动画尺寸在生成阶段就烘成显示尺寸，运行时**不能**对 QMovie 调 `setScaledSize`，
  否则每帧重新解码会把界面卡死

重新生成看板娘（需要 `pip install onnxruntime imageio-ffmpeg`）：

```bash
# 纯白背景的素材用这个（方形补边推理 + 去白边 + 统一包围盒裁切）
python tools/make_mascot_white.py --video "竖屏14秒挥手加站姿-纯白背景.mp4" \
       --out assets/mascot_white.webp --fps 12 --height 720 --box 144x300

# 背景不是纯白的素材用旧的通用脚本（输出 apng / gif）
python tools/make_mascot.py --video assets/mascot.mp4 \
       --out assets/mascot.png --format apng --fps 12 --proc-side 640 --max-side 320
```

`--box` 就是成品尺寸，直接用卡片能给的大小（左栏 156px 宽，扣掉内边距是 144）。
宽高比由资源决定、`MascotCard` 运行时从文件里读，换一段视频不用改代码。

## 三、运行

项目里已带好一个可用的虚拟环境 `.venv`（Python 3.13 + numpy / pillow / PySide6）：

- PyCharm 打开本目录后，解释器选 `PBRTexStudio/.venv/Scripts/python.exe` 即可直接运行 `main.py`
- 想自己重建环境：

```bash
pip install -r requirements.txt
python main.py
```

命令行（不用开界面也能干活）：

```bash
# 由基础色生成整套贴图
python -m pbr_core.cli gen -i color.png -o out --name T_Rock_A01 \
       --set NRM,DIS,OCC,RGH,MET --strength 3

# 检查任意法线的通道顺序
python -m pbr_core.cli check -i path/to/x_NRM.png

# 把粉色法线修回来
python -m pbr_core.cli fix -i pink_NRM.png -o fixed
```

## 三·五、打包 EXE

已经打好两份，都在 F 盘工程目录里，绿色免安装、双击即用：

| 产物 | 路径 | 体积 | 说明 |
|------|------|------|------|
| 文件夹版（推荐） | `dist/ChordStudio/ChordStudio.exe` | 139 MB | 启动 ~1.5s，整个 `ChordStudio` 文件夹拷走就能跑 |
| 单文件版 | `dist_onefile/ChordStudio_single.exe` | 59 MB | 就一个 exe，首次启动要解压到 `%TEMP%`，约 5~8s |

两份都带同一个 `assets/app.ico`（16/24/32/48/64/128/256 七种尺寸），
EXE 文件图标、标题栏、任务栏都是它。`run()` 里设了
`SetCurrentProcessExplicitAppUserModelID("ChordStudio.PBR")`，
不然打包后的窗口在任务栏上会被并到 python 进程、显示通用图标。

重新打包（`assets/app.ico` 换了图标要重跑一次，图标是编译进 EXE 资源的，不是读文件）：

```bash
# 文件夹版
.venv\Scripts\pyinstaller ChordStudio.spec --noconfirm

# 单文件版
.venv\Scripts\pyinstaller main.py --onefile --windowed --noconfirm \
    --name ChordStudio_single --icon assets/app.ico \
    --add-data "assets/bg.jpg;assets" \
    --add-data "assets/mascot_white.webp;assets" \
    --add-data "assets/app.ico;assets" \
    --exclude-module onnxruntime --exclude-module tkinter
```

打包要点：

- `_ASSETS` 是按 `__file__` 往上找一层的，PyInstaller 下正好落在 `_MEIPASS`
  （`_internal`），所以 `--add-data` 一律目标写 `assets` 就行，代码不用判断是否 frozen
- **只打三份素材**：`bg.jpg`、`mascot_white.webp`、`app.ico`。`assets/` 里的
  `mascot.png`(13M 废弃 APNG)、`mascot.mp4`、`mascot.gif` 不进包，否则白胖 20M
- 排除 `onnxruntime` / `tkinter`：只有 `tools/` 下的抠图脚本用得到
- 打包出来的 `_internal/PySide6/plugins/imageformats/` 必须有 `qwebp.dll`，
  少了看板娘的 webp 动画就加载不出来（"动画未找到"）
- 别用 `--onefile` 当默认：每次启动都把 139M 解压到 C 盘 `%TEMP%`，冷启动 5 秒以上

## 四、通道规范

| 后缀 | 含义 | 位深 | sRGB |
|------|------|------|------|
| COL | 基础色 | 8 | **开** |
| NRM | 法线（Z 在 **B**） | 8 | 关 |
| RGH | 粗糙度 | 8 | 关 |
| MET | 金属度 | 8 | 关 |
| OCC | 环境光遮蔽 | 8 | 关 |
| DIS | 位移 / 高度 | 8 或 16 | 关 |
| CUR | 曲率 | 8 | 关 |
| GLS / SHN | 光泽度 / 高光指数 | 8 | 关 |

导入 UE / Unity 时，除了 COL 之外**全部要关掉 sRGB**，
否则粗糙度会变成一条 gamma 曲线，法线也会偏色。

DirectX（UE、Unity）与 OpenGL（Blender 部分管线）只差**绿通道是否翻转**，
界面里的 `DirectX / OpenGL` 下拉框就是干这个的，不会动到 R/B。
