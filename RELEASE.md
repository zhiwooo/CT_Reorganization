# v0.1.0 Release 准备说明

本文档用于整理 Tiny CT Workstation `v0.1.0` 的发布流程。

## 发布目标

`v0.1.0` 是第一个可演示版本，目标是提供一个接近 Tiny CT 视频效果的 Windows 桌面应用：

- PySide6 图形界面
- 投影序列导入和预览
- background / dark / flat 校正
- 旋转中心偏移估计与校正
- 探测器面内偏转和 X/Y 偏移校正
- ASTRA `FDK_CUDA` 三维重建
- `volume.npy`、`config.json`、PNG 切片导出

## 发布前检查

在项目根目录执行：

```powershell
$env:Path = "C:\Users\zhiwoo\.local\bin;$env:Path"
uv sync
uv run pytest -q
uv run python -c "import astra, PySide6; print('ok')"
```

有显示环境时启动 GUI：

```powershell
uv run ct
```

至少确认：

- 软件窗口可以打开
- 可以选择 `proj` 数据目录
- 可以导入并预览投影图
- 可以计算推荐体素尺寸
- 可以自动估计旋转中心
- 小尺寸重建可以成功
- 输出目录包含 `volume.npy`、`config.json` 和可选 `slices/`

## 一键打包

```powershell
powershell -ExecutionPolicy ByPass -File scripts\release.ps1
```

默认发布产物：

```text
release/
└── TinyCT-v0.1.0-windows-x64.zip
```

发布目录结构：

```text
release/TinyCT-v0.1.0-windows-x64/
├── TinyCT/
│   ├── TinyCT.exe
│   └── ...
├── README.md
├── RELEASE.md
└── VERSION.txt
```

如需随包附带示例数据：

```powershell
powershell -ExecutionPolicy ByPass -File scripts\release.ps1 -IncludeSampleData
```

示例数据较大，只建议内部演示包使用。

## 运行环境说明

ASTRA `FDK_CUDA` 依赖 NVIDIA GPU 和 CUDA 运行环境。目标电脑如果没有可用 CUDA，GUI 可以启动，投影预览可以使用，但重建阶段可能失败。

## 版本标记

发布包确认可用后，可以创建 git tag：

```powershell
git tag v0.1.0
git push origin v0.1.0
```
