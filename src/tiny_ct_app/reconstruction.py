"""
CT重建核心算法模块。

实现投影预处理和FDK锥束CT重建算法。支持暗场/亮场校正、
旋转中心偏移校正、探测器偏转校正等图像预处理功能。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .config import ReconstructionConfig
from .io_utils import load_optional_background, load_optional_calibration, load_projection_stack, save_png_stack


# 进度回调函数类型定义
ProgressCallback = Callable[[str], None]


def _log(callback: ProgressCallback | None, message: str) -> None:
    """
    输出日志信息。

    Args:
        callback: 进度回调函数。为None时不输出。
        message: 日志信息。
    """
    if callback:
        callback(message)


def _shift_columns(image: np.ndarray, offset_px: float) -> np.ndarray:
    """
    通过插值平移图像的列（横向平移）。

    用于旋转中心偏移校正。

    Args:
        image: 输入图像，形状为(height, width)。
        offset_px: 平移量（像素）。正值向右平移，负值向左平移。

    Returns:
        np.ndarray: 平移后的图像，同样形状为(height, width)。
    """
    if abs(offset_px) < 1e-6:
        return image
    x = np.arange(image.shape[1], dtype=np.float32)
    shifted_x = x - float(offset_px)
    return np.vstack([np.interp(shifted_x, x, row, left=row[0], right=row[-1]) for row in image])


def _shift_axis_with_edge_fill(image: np.ndarray, shift: int, axis: int) -> np.ndarray:
    """Shift an image along one axis and fill exposed pixels with edge values."""
    if shift == 0:
        return image

    length = image.shape[axis]
    if length == 0:
        return image

    magnitude = abs(shift)
    if magnitude >= length:
        edge_index = 0 if shift > 0 else length - 1
        edge = np.take(image, edge_index, axis=axis)
        return np.repeat(np.expand_dims(edge, axis=axis), length, axis=axis)

    pad_width = [(0, 0)] * image.ndim
    if shift > 0:
        pad_width[axis] = (magnitude, 0)
        padded = np.pad(image, pad_width, mode="edge")
        slices = [slice(None)] * image.ndim
        slices[axis] = slice(0, length)
    else:
        pad_width[axis] = (0, magnitude)
        padded = np.pad(image, pad_width, mode="edge")
        slices = [slice(None)] * image.ndim
        slices[axis] = slice(magnitude, magnitude + length)

    return padded[tuple(slices)]


def _apply_detector_offsets(
    projections: np.ndarray,
    offset_x_mm: float,
    offset_y_mm: float,
    px_size_x_mm: float,
    px_size_y_mm: float,
    callback: ProgressCallback | None,
) -> np.ndarray:
    """Apply detector translation offsets using a simple integer-pixel shift."""
    if abs(offset_x_mm) < 1e-6 and abs(offset_y_mm) < 1e-6:
        return projections

    x_shift = int(round(offset_x_mm / max(px_size_x_mm, 1e-6)))
    y_shift = int(round(offset_y_mm / max(px_size_y_mm, 1e-6)))
    if x_shift == 0 and y_shift == 0:
        return projections

    _log(callback, f"应用探测器偏移校正：dx={offset_x_mm:.4g} mm, dy={offset_y_mm:.4g} mm")
    corrected = []
    for proj in projections:
        shifted = _shift_axis_with_edge_fill(proj, x_shift, axis=1)
        shifted = _shift_axis_with_edge_fill(shifted, y_shift, axis=0)
        corrected.append(shifted)
    return np.stack(corrected, axis=0).astype(np.float32)


def _apply_detector_roll(projections: np.ndarray, roll_deg: float, callback: ProgressCallback | None) -> np.ndarray:
    """
    应用探测器面内偏转校正。

    用于校正探测器绕光轴旋转的偏转。

    Args:
        projections: 投影图像栈，形状为(num_projections, height, width)。
        roll_deg: 偏转角度（度）。
        callback: 进度回调函数。

    Returns:
        np.ndarray: 校正后的投影栈，float32类型。

    Notes:
        若scipy不可用，跳过校正并发出警告。
    """
    if abs(roll_deg) < 1e-6:
        return projections
    try:
        from scipy import ndimage
    except Exception:
        _log(callback, "scipy 不可用，已跳过探测器面内偏转校正。")
        return projections

    _log(callback, f"应用探测器面内偏转校正：{roll_deg:.4g} deg")
    corrected = [
        ndimage.rotate(proj, roll_deg, reshape=False, order=1, mode="nearest")
        for proj in projections
    ]
    return np.stack(corrected, axis=0).astype(np.float32)


def _apply_background_correction(
    projections: np.ndarray,
    background: np.ndarray,
    callback: ProgressCallback | None,
) -> np.ndarray:
    """Subtract a static detector/background image from every projection."""
    if background.shape != projections.shape[1:]:
        _log(callback, f"背景图尺寸 {background.shape} 与投影尺寸 {projections.shape[1:]} 不一致，已跳过背景矫正。")
        return projections

    _log(callback, "应用背景矫正。")
    corrected = projections - background.astype(np.float32, copy=False)
    return np.maximum(corrected, 0.0).astype(np.float32, copy=False)


def _to_attenuation_for_center_estimation(
    projections: np.ndarray,
    folder: str | Path | None,
    use_background_correction: bool,
    use_dark_flat: bool,
) -> np.ndarray:
    """Build attenuation projections for robust center estimation."""
    data = projections.astype(np.float32, copy=True)

    if folder is not None and use_background_correction:
        background = load_optional_background(folder)
        if background is not None and background.shape == data.shape[1:]:
            data = np.maximum(data - background.astype(np.float32, copy=False), 0.0)

    if folder is not None and use_dark_flat:
        dark, flat = load_optional_calibration(folder)
        if dark is not None and flat is not None and dark.shape == data.shape[1:] and flat.shape == data.shape[1:]:
            data = (data - dark) / np.maximum(flat - dark, 1.0)

    data = np.maximum(data, 1e-6)
    if np.nanmax(data) > 2.0:
        data = data / np.nanmax(data)
    data = -np.log(np.clip(data, 1e-6, None))
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _largest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    """Return inclusive start/end indexes for the largest true run in a 1D mask."""
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        return None

    breaks = np.flatnonzero(np.diff(indexes) > 1)
    starts = np.r_[indexes[0], indexes[breaks + 1]]
    ends = np.r_[indexes[breaks], indexes[-1]]
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def _estimate_center_from_silhouette(projections: np.ndarray) -> float:
    """Estimate correction offset from the projection silhouette center."""
    width = projections.shape[2]
    geometric_center = (width - 1) / 2.0
    sample_count = min(24, projections.shape[0])
    sample_indexes = np.linspace(0, projections.shape[0] - 1, sample_count, dtype=int)
    offsets: list[float] = []

    for index in sample_indexes:
        profile = np.mean(projections[index], axis=0)
        lo, hi = np.percentile(profile, [5.0, 95.0])
        if hi <= lo:
            continue
        mask = profile > (lo + (hi - lo) * 0.25)
        run = _largest_true_run(mask)
        if run is None:
            continue
        start, end = run
        if end - start < max(4, width // 32):
            continue
        object_center = (start + end) / 2.0
        offsets.append(geometric_center - object_center)

    if not offsets:
        raise ValueError("投影主体轮廓不清晰，无法估算旋转中心。")
    return float(np.median(offsets))


def estimate_rotation_center_offset(
    projections: np.ndarray,
    folder: str | Path | None = None,
    use_background_correction: bool = True,
    use_dark_flat: bool = True,
) -> float:
    """
    估算旋转中心偏移。

    先转换为衰减图，再使用多张投影的主体外轮廓中心估算相对
    图像几何中心的校正量。正值表示投影主体偏左，需要向右校正。
    """
    if projections.ndim != 3 or projections.shape[0] < 2:
        raise ValueError("至少需要两张投影才能估算旋转中心。")

    attenuation = _to_attenuation_for_center_estimation(
        projections,
        folder,
        use_background_correction=use_background_correction,
        use_dark_flat=use_dark_flat,
    )
    return _estimate_center_from_silhouette(attenuation)


def preprocess_projections(
    projections: np.ndarray,
    folder: str | Path,
    config: ReconstructionConfig,
    callback: ProgressCallback | None = None,
) -> np.ndarray:
    """
    投影图像预处理。

    包括暗场/亮场校正、对数变换、旋转中心偏移校正、
    探测器面内偏转校正等步骤。

    Args:
        projections: 原始投影图像栈，形状为(num_projections, height, width)。
        folder: 投影图像所在文件夹，用于加载校正图像。
        config: 重建配置对象。
        callback: 进度回调函数。

    Returns:
        np.ndarray: 预处理后的投影栈，float32类型，已准备好进行重建。

    Notes:
        - 对数变换：被用来将强度转换为衰减系数
        - 异常值（NaN、Inf）会被替换为0
    """
    data = projections.astype(np.float32, copy=True)

    if config.use_background_correction:
        background = load_optional_background(folder)
        if background is not None:
            data = _apply_background_correction(data, background, callback)
        else:
            _log(callback, "未找到 background/bg/bkg 图像，跳过背景矫正。")

    if config.use_dark_flat:
        dark, flat = load_optional_calibration(folder)
        if dark is not None and flat is not None:
            _log(callback, "应用暗场/亮场校正。")
            denom = np.maximum(flat - dark, 1.0)
            data = (data - dark) / denom
        else:
            _log(callback, "未找到完整 dark/flat 图像，跳过平场校正。")

    data = np.maximum(data, 1e-6)
    if np.nanmax(data) > 2.0:
        data = data / np.nanmax(data)
    data = -np.log(np.clip(data, 1e-6, None))
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if abs(config.rotation_center_offset_px) > 1e-6:
        _log(callback, f"应用旋转中心偏移：{config.rotation_center_offset_px:.4g} px")
        data = np.stack([_shift_columns(proj, config.rotation_center_offset_px) for proj in data], axis=0)

    data = _apply_detector_offsets(
        data,
        config.detector_offset_x_mm,
        config.detector_offset_y_mm,
        config.detector_pixel_size_x_mm,
        config.detector_pixel_size_y_mm,
        callback,
    )
    data = _apply_detector_roll(data, config.detector_roll_deg, callback)
    return data.astype(np.float32, copy=False)


def run_fdk_reconstruction(
    config: ReconstructionConfig,
    callback: ProgressCallback | None = None,
) -> np.ndarray:
    """
    执行FDK锥束CT重建。

    这是主重建函数，完成以下步骤：
    1. 加载投影图像
    2. 预处理投影（校正+对数变换）
    3. 设置几何参数（投影几何和体数据几何）
    4. 使用ASTRA Toolbox的FDK_CUDA算法进行重建
    5. 导出结果（体数据.npy、配置.json、切片PNG）

    Args:
        config: 重建配置对象，包含所有必要参数。
        callback: 进度回调函数，用于输出日志信息。

    Returns:
        np.ndarray: 重建结果体数据，形状为(volume_size_z, volume_size_y, volume_size_x)。

    Raises:
        RuntimeError: 未安装ASTRA Toolbox或其他重建错误时抛出。
        FileNotFoundError: 无法找到投影图像时抛出。

    Notes:
        - 需要NVIDIA CUDA支持（FDK_CUDA算法）
        - 结果自动保存到 config.output_dir 目录
        - 包括原始体数据(volume.npy)、配置(config.json)、切片(slices/)
    """
    config.validate()

    projection_dir = Path(config.projection_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(callback, f"读取投影目录：{projection_dir}")
    raw, files = load_projection_stack(projection_dir, limit=config.projection_count)
    if len(files) < config.projection_count:
        _log(callback, f"投影数量少于设置值：读取 {len(files)} 张。")
    else:
        _log(callback, f"读取投影：{len(files)} 张，尺寸 {raw.shape[2]} x {raw.shape[1]}")

    projections = preprocess_projections(raw, projection_dir, config, callback)
    detector_rows, detector_cols = projections.shape[1], projections.shape[2]
    angles = np.linspace(0.0, 2.0 * np.pi, projections.shape[0], endpoint=False).astype(np.float32)

    try:
        import astra
    except Exception as exc:
        raise RuntimeError("未能导入 ASTRA Toolbox，请先通过 uv 安装 astra-toolbox。") from exc

    _log(callback, "创建 ASTRA 锥束几何。")
    proj_geom = astra.create_proj_geom(
        "cone",
        config.detector_pixel_size_x_mm,
        config.detector_pixel_size_y_mm,
        detector_rows,
        detector_cols,
        angles,
        config.source_object_distance_mm,
        config.origin_detector_distance_mm,
    )
    vol_geom = astra.create_vol_geom(
        config.volume_size_y,
        config.volume_size_x,
        config.volume_size_z,
        -config.volume_size_x * config.voxel_size_mm / 2.0,
        config.volume_size_x * config.voxel_size_mm / 2.0,
        -config.volume_size_y * config.voxel_size_mm / 2.0,
        config.volume_size_y * config.voxel_size_mm / 2.0,
        -config.volume_size_z * config.voxel_size_mm / 2.0,
        config.volume_size_z * config.voxel_size_mm / 2.0,
    )

    sino = np.transpose(projections, (1, 0, 2))
    sino_id = astra.data3d.create("-sino", proj_geom, sino)
    rec_id = astra.data3d.create("-vol", vol_geom)
    alg_id = None

    try:
        cfg = astra.astra_dict("FDK_CUDA")
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = rec_id
        alg_id = astra.algorithm.create(cfg)
        _log(callback, "开始 FDK_CUDA 重建。")
        astra.algorithm.run(alg_id)
        volume = astra.data3d.get(rec_id).astype(np.float32)
    finally:
        if alg_id is not None:
            astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(rec_id)

    np.save(output_dir / "volume.npy", volume)
    config.save(output_dir / "config.json")
    if config.save_png_slices:
        _log(callback, "导出 PNG 切片。")
        save_png_stack(volume, output_dir / "slices")
    _log(callback, f"重建完成：{output_dir}")
    return volume
