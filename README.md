# Tiny CT Workstation

一个轻量级工业 CT 重建原型，使用 `Python + PySide6` 构建桌面界面，使用 `ASTRA Toolbox` 执行 FDK 锥束 CT 重建。

当前版本目标是复刻视频里的 Tiny CT 工具核心流程：

- 选择投影图像目录
- 读取 `proj_*.tif/.png/.jpg` 图像序列
- 自动读取可选的 `dark_0000.tif`、`flat_0000.tif`
- 配置源物距、源探距、投影数量、探测器像素尺寸等几何参数
- 支持旋转中心偏移校正
- 支持探测器面内偏转参数
- 使用 ASTRA FDK 重建
- 查看投影图和重建切片
- 导出切片 PNG、体数据 `.npy` 和参数 `config.json`

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

启动软件：

```powershell
uv run tiny-ct
```

或：

```powershell
uv run python -m tiny_ct_app.main
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
