from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .config import ReconstructionConfig
from .io_utils import load_optional_calibration, load_projection_stack, save_png_stack


ProgressCallback = Callable[[str], None]


def _log(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _shift_columns(image: np.ndarray, offset_px: float) -> np.ndarray:
    if abs(offset_px) < 1e-6:
        return image
    x = np.arange(image.shape[1], dtype=np.float32)
    shifted_x = x - float(offset_px)
    return np.vstack([np.interp(shifted_x, x, row, left=row[0], right=row[-1]) for row in image])


def _apply_detector_roll(projections: np.ndarray, roll_deg: float, callback: ProgressCallback | None) -> np.ndarray:
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


def preprocess_projections(
    projections: np.ndarray,
    folder: str | Path,
    config: ReconstructionConfig,
    callback: ProgressCallback | None = None,
) -> np.ndarray:
    data = projections.astype(np.float32, copy=True)

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

    data = _apply_detector_roll(data, config.detector_roll_deg, callback)
    return data.astype(np.float32, copy=False)


def run_fdk_reconstruction(
    config: ReconstructionConfig,
    callback: ProgressCallback | None = None,
) -> np.ndarray:
    try:
        import astra
    except Exception as exc:
        raise RuntimeError("未能导入 ASTRA Toolbox，请先通过 uv 安装 astra-toolbox。") from exc

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
