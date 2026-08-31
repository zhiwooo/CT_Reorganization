"""
输入输出工具模块。

提供投影图像加载、校正数据读取、图像显示和结果导出等功能。
支持多种格式：TIFF、PNG、JPEG等。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


# 支持的投影图像文件名模式
IMAGE_PATTERNS = ("proj_*.tif", "proj_*.tiff", "proj_*.png", "proj_*.jpg", "proj_*.jpeg")


def find_projection_files(folder: str | Path) -> list[Path]:
    """
    查找投影图像文件。

    首先按照标准命名规则（proj_*.tif/png/jpg等）查找，
    如果没有找到则查找所有图像文件。

    Args:
        folder: 包含投影图像的文件夹路径。

    Returns:
        list[Path]: 找到的投影文件路径列表（按名称排序）。
    """
    base = Path(folder)
    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(sorted(base.glob(pattern)))
    if not files:
        for pattern in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            files.extend(sorted(base.glob(pattern)))
    return files


def read_image(path: str | Path) -> np.ndarray:
    """
    读取单个投影图像。

    支持TIFF和PNG/JPEG格式。如果是彩色图像，仅使用第一个通道。
    返回结果为float32类型的灰度图像。

    Args:
        path: 图像文件路径。

    Returns:
        np.ndarray: float32灰度图像数组，形状为(height, width)。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        IOError: 图像读取失败时抛出。
    """
    source = Path(path)
    if source.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(source)
    else:
        image = np.asarray(Image.open(source))
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.float32)


def load_projection_stack(folder: str | Path, limit: int | None = None) -> tuple[np.ndarray, list[Path]]:
    """
    加载投影图像栈。

    从指定目录加载所有投影图像，返回叠加后的三维数组。

    Args:
        folder: 包含投影图像的文件夹路径。
        limit: 最多加载的图像数量。为None时加载所有图像。

    Returns:
        tuple[np.ndarray, list[Path]]: 
            - np.ndarray: 投影图像栈，形状为(num_projections, height, width)
            - list[Path]: 加载的文件路径列表

    Raises:
        FileNotFoundError: 指定目录中没有找到投影图像时抛出。
    """
    files = find_projection_files(folder)
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No projection images found in {folder}")
    stack = np.stack([read_image(path) for path in files], axis=0)
    return stack, files


def load_optional_calibration(folder: str | Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    加载可选的暗场(dark)和亮场(flat)校正图像。

    查找文件夹中的dark_*.tif和flat_*.tif文件。
    如果存在，返回第一个找到的文件；否则返回None。

    Args:
        folder: 文件夹路径。

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: 
            (dark_image, flat_image) - 如果文件不存在则对应位置为None。
    """
    base = Path(folder)
    dark_files = sorted(base.glob("dark_*.tif")) + sorted(base.glob("dark_*.tiff"))
    flat_files = sorted(base.glob("flat_*.tif")) + sorted(base.glob("flat_*.tiff"))
    dark = read_image(dark_files[0]) if dark_files else None
    flat = read_image(flat_files[0]) if flat_files else None
    return dark, flat


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    将图像标准化为uint8格式用于显示。

    使用1%和99%百分位数进行拉伸，提高对比度。

    Args:
        image: 输入图像数组（任意数值类型）。

    Returns:
        np.ndarray: uint8类型的显示图像，值范围[0, 255]。
    """
    array = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(array, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(np.min(array)), float(np.max(array))
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def save_png_stack(volume: np.ndarray, output_dir: str | Path) -> None:
    """
    将三维体数据导出为PNG切片序列。

    按照 slice_0000.png, slice_0001.png, ... 的格式保存。

    Args:
        volume: 三维体数据数组，形状为(num_slices, height, width)。
        output_dir: 输出目录。如果不存在则自动创建。

    Raises:
        IOError: 文件写入失败时抛出。
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for index, slice_image in enumerate(volume):
        display = normalize_for_display(slice_image)
        Image.fromarray(display).save(target / f"slice_{index:04d}.png")
