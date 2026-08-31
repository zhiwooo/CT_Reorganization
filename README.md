# Tiny CT Workstation

一个轻量级工业CT重建原型系统，集成了完整的CT扫描数据处理和三维重建功能。基于 `Python + PySide6` 构建现代化桌面应用界面，采用 `ASTRA Toolbox` 执行高效的FDK锥束CT重建算法。

## 核心功能

- ✅ **投影数据管理**：支持TIFF/PNG/JPEG多种格式，自动识别和加载投影图像序列
- ✅ **图像校正**：暗场/亮场(Dark/Flat)校正，消除探测器暗电流和光强不均
- ✅ **几何参数配置**：灵活设置源物距(SOD)、源探距(SDD)、投影数量、探测器像素尺寸等
- ✅ **参数校正**：支持旋转中心偏移校正、探测器面内偏转校正、探测器偏移校正等
- ✅ **高效重建**：采用ASTRA FDK_CUDA算法实现GPU加速的三维重建
- ✅ **实时预览**：投影图像和重建切片的实时显示和交互
- ✅ **多格式导出**：
  - 原始体数据（`.npy`格式，便于后处理）
  - PNG切片序列（用于医学影像显示）
  - 配置参数（`config.json`，便于参数管理和再现）

## 使用 uv

如果本机尚未安装 `uv`，先安装：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

创建环境并安装依赖：

```powershell
uv sync
```

如果机器上有 Anaconda，`uv` 可能会优先选择 Anaconda 解释器。当前项目已验证可使用 uv 管理的 CPython 3.12.14：

```powershell
$env:Path = "C:\Users\zhiwoo\.local\bin;$env:Path"
$env:UV_PROJECT_ENVIRONMENT = "D:\Office\工业CT重建\.venv_managed"
uv run python --version
```

启动 GUI：

```powershell
uv run tiny-ct
```

或：

```powershell
uv run python -m tiny_ct_app.main
```

命令行直接重建：

```powershell
uv run tiny-ct-cli --projection-dir ./proj --output-dir ./recon_result --projection-count 360 --sod 200 --sdd 800 --voxel-size 0.2
```

如果需要强制启动 GUI：

```powershell
uv run python -m tiny_ct_app.main --gui
```

## 当前测试数据

工作区已有 `proj` 文件夹，里面包含 360 张投影图，以及 `dark_0000.tif`、`flat_0000.tif`。软件默认参数按照视频中展示的仿真参数设置：

```text
源物距：200 mm
源探距：800 mm
投影数量：360 张
探测器面内偏转：2 deg
探测器横向像素间隔：0.2 mm
探测器纵向像素间隔：0.2 mm
```

## 注意

ASTRA 的 3D FDK 通常需要 CUDA GPU。若环境中没有可用 CUDA 或 ASTRA 未正确安装，界面仍可打开并预览投影，但重建会在日志区提示错误。

## 项目结构

```
src/tiny_ct_app/
├── __init__.py           # 包初始化模块
├── config.py             # 配置数据类定义
├── io_utils.py           # 输入输出工具函数（图像加载、保存等）
├── reconstruction.py     # CT重建核心算法（预处理、FDK重建）
└── main.py              # PySide6 GUI主程序
```

### 模块说明

| 模块 | 功能 | 主要函数/类 |
|------|------|-----------|
| `config.py` | 重建参数配置 | `ReconstructionConfig` 数据类 |
| `io_utils.py` | 文件输入输出 | `load_projection_stack()`, `save_png_stack()`, `normalize_for_display()` |
| `reconstruction.py` | 重建算法实现 | `preprocess_projections()`, `run_fdk_reconstruction()` |
| `main.py` | GUI应用主程序 | `MainWindow`, `ReconstructionWorker`, `ImageView` |

## 使用指南

### 基本工作流

1. **启动应用**
   ```powershell
   uv run tiny-ct
   ```

2. **选择投影目录**
   - 点击"选择目录"按钮选择包含投影图像的文件夹
   - 或直接在文本框输入路径
   - 点击"加载预览"查看投影图像

3. **配置重建参数**
   - **重建参数标签**：配置源物距、源探距、投影数量、探测器像素尺寸等基本几何参数
   - **体数据设置标签**：配置输出体数据的大小和分辨率
   - **参数校正标签**：配置探测器面内偏转、旋转中心偏移等校正参数

4. **执行重建**
   - 点击"开始重建"按钮
   - 监控日志区域查看处理进度
   - 重建完成后自动显示重建结果的中间切片

5. **查看和导出结果**
   - 使用滑块浏览不同的投影或切片
   - 点击"打开保存目录"查看导出的文件：
     - `volume.npy` - 原始体数据（NumPy数组）
     - `config.json` - 本次重建的配置参数
     - `slices/` - 按切片序号保存的PNG图像

### 关键参数说明

#### 几何参数

- **源物距 (SOD - Source Object Distance)**：X射线源到旋转中心的距离，单位mm
- **源探距 (SDD - Source Detector Distance)**：X射线源到探测器的距离，单位mm
  - 自动计算的原点到探测器距离(ODD) = SDD - SOD
- **投影数量**：扫描获取的投影图像总数（典型值：360）
- **探测器像素尺寸**：X和Y方向的物理像素间距，单位mm

#### 校正参数

- **探测器面内偏转**：探测器绕光轴旋转的角度，单位度。用于校正探测器安装误差
- **旋转中心偏移**：旋转中心在投影图像中的像素位移。用于补偿机械对齐误差
- **暗场/亮场校正**：启用时自动加载并应用dark和flat校正图像

#### 体数据参数

- **体尺寸 (X/Y/Z)**：重建体数据的分辨率（像素数）
- **体素大小**：单个体素的物理尺寸，单位mm

### 投影图像格式要求

项目支持以下投影图像命名规则：
- `proj_*.tif` / `proj_*.tiff`（TIFF格式）
- `proj_*.png` / `proj_*.jpg` / `proj_*.jpeg`（其他格式）

若不按规范命名，系统会回退到加载所有图像文件。

建议的目录结构：
```
proj/
├── proj_0000.tif
├── proj_0001.tif
├── ...
├── proj_0359.tif
├── dark_0000.tif          # 可选：暗场图像
└── flat_0000.tif          # 可选：亮场图像
```

## 技术细节

### 投影预处理流程

1. **暗场/亮场校正**（可选）
   - 公式：`corrected = (raw - dark) / (flat - dark)`
   - 用于消除探测器固定噪声和光强不均

2. **强度到衰减系数转换**
   - 归一化到[0, 1]范围
   - 取负对数：`μ = -ln(intensity)`
   - 处理异常值（NaN、Inf）

3. **旋转中心偏移校正**
   - 通过列向平移投影来补偿旋转轴偏移
   - 采用线性插值保持图像质量

4. **探测器偏转校正**
   - 使用scipy.ndimage的旋转函数
   - 可选项，若scipy不可用则跳过

### FDK重建算法

- **算法**：锥束FDK（Feldkamp Davis Kiraly）
- **实现**：ASTRA Toolbox的FDK_CUDA
- **加速**：NVIDIA CUDA并行计算
- **输出**：浮点型体数据

### 显示和导出

- **实时预览**：使用百分位数拉伸(1%-99%)增强对比度
- **PNG导出**：8位灰度，自动对比度调整
- **NumPy导出**：保留原始浮点数据用于后处理

## 故障排查

### 问题：无法导入ASTRA Toolbox

**症状**：出现 "未能导入 ASTRA Toolbox" 错误

**解决方案**：
```powershell
# 检查环境
uv run python -c "import astra; print(astra.__version__)"

# 重新安装依赖
uv sync --refresh
```

### 问题：重建速度很慢或失败

**症状**：重建进度停滞或出现CUDA相关错误

**原因和解决**：
- ❌ 没有NVIDIA显卡或CUDA环境不正确
  - 安装CUDA工具包（与GPU驱动兼容）
  - 重新编译ASTRA Toolbox for CUDA
- ❌ GPU内存不足
  - 减小体数据大小
  - 减少投影数量

### 问题：投影无法加载

**症状**：加载预览时显示文件未找到错误

**原因和解决**：
- ❌ 投影文件名不符合规范
  - 确保文件名为 `proj_XXXX.tif` 格式
  - 或检查是否在指定目录中
- ❌ 文件路径包含特殊字符
  - 使用ASCII字符的目录路径

### 问题：重建结果质量差

**症状**：重建的体数据中出现伪影或模糊

**调整建议**：
- 检查源物距和源探距参数是否正确
- 调整旋转中心偏移参数
- 确保已启用暗场/亮场校正（如果数据存在）
- 尝试调整探测器偏转参数

## 依赖项

| 包名 | 版本 | 用途 |
|-----|------|------|
| `astra-toolbox` | ≥2.2 | CT重建算法库 |
| `pyside6` | ≥6.7 | GUI框架 |
| `numpy` | ≥1.26 | 数值计算 |
| `pillow` | ≥10.0 | 图像处理 |
| `scipy` | ≥1.11 | 科学计算（探测器偏转校正） |
| `tifffile` | ≥2024.2.12 | TIFF文件读写 |

## 开发和扩展

### 添加自定义重建算法

编辑 `reconstruction.py` 的 `run_fdk_reconstruction()` 函数，添加其他ASTRA支持的算法（如SIRT、CGLS等）。

### 自定义参数校正

在 `reconstruction.py` 的 `preprocess_projections()` 函数中添加新的预处理步骤。

### 扩展GUI功能

编辑 `main.py` 的 `MainWindow` 类，添加新的参数控件或功能标签。

## 许可证

本项目遵循开源许可证。详见项目根目录的LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request！在贡献代码前，请：
1. 创建新的特性分支
2. 编写清晰的提交信息
3. 更新相关文档
4. 确保代码符合项目风格

## 参考资源

- [ASTRA Toolbox 文档](https://www.astra-toolbox.com/)
- [PySide6 官方文档](https://doc.qt.io/qtforpython-6/)
- [FDK算法论文](https://doi.org/10.1109/TNS.1984.4868845) - Feldkamp, Davis, Kiraly (1984)
- CT图像重建基础 - 理论和应用
