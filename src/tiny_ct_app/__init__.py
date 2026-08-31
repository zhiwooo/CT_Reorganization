"""
Tiny CT Workstation - 轻量级工业CT重建工作站。

一个集投影数据处理、参数校正和三维重建于一体的CT图像重建系统。
提供PySide6图形界面和完整的重建工作流。

主要模块：
- config: 重建参数配置
- io_utils: 图像文件的输入输出
- reconstruction: CT重建算法核心实现
- main: 用户界面和应用入口

使用示例：
    uv run ct              # 启动GUI应用
    uv run python -m tiny_ct_app.main

典型工作流：
    1. 选择投影图像目录
    2. 加载投影预览
    3. 配置几何参数和校正参数
    4. 执行FDK重建
    5. 查看切片或导出结果

系统要求：
    - Python 3.10+
    - NVIDIA CUDA支持（用于FDK_CUDA加速）
    - 4GB+ GPU显存（推荐）
"""

__version__ = "0.1.0"
__author__ = "CT Reconstruction Team"
__description__ = "Lightweight Industrial CT Reconstruction Workstation"
