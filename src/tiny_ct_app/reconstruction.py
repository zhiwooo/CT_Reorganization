"""
CT重建核心算法模块。

实现投影预处理和FDK锥束CT重建算法。支持暗场/亮场校正、
旋转中心偏移校正、探测器偏转校正等图像预处理功能。

主要算法流程：
1. 投影预处理：
   - 背景/暗电流校正：去除检测器的系统噪声
   - 暗场/亮场校正：补偿光源不均匀性（平场校正）
   - 对数变换：根据Beer-Lambert定律转换为衰减系数
   - 旋转中心校正：补偿几何误差（拍摄时旋转中心偏移）
   - 探测器偏移校正：补偿探测器面内/外平移
   - 探测器滚转校正：补偿探测器旋转偏转

2. 几何参数设置：
   - 投影几何：定义X光源、探测器、投影中心距等
   - 体数据几何：定义重建体积的大小和位置

3. FDK重建：
   - 使用ASTRA Toolbox的CUDA加速FDK_CUDA算法
   - 输出3D灰度体数据

4. 结果导出：
   - 保存原始体数据(.npy)
   - 保存配置参数(.json)
   - 导出切片序列(.png)
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

    用于旋转中心偏移校正。使用线性插值保证平滑性，边界外的像素
    使用边界值填充以保持图像尺寸不变。

    Args:
        image: 输入图像，形状为(height, width)。
        offset_px: 平移量（像素）。正值向右平移，负值向左平移。

    Returns:
        np.ndarray: 平移后的图像，同样形状为(height, width)。

    Notes:
        - 使用np.interp进行线性插值，具有反走样效果
        - 边界像素使用row[0]和row[-1]填充超出范围的值
        - 对每一行独立进行平移，适合处理投影图像

    Algorithm:
        对于每一行，将原始x坐标映射为x - offset_px，
        然后使用线性插值计算新位置的像素值。
    """
    if abs(offset_px) < 1e-6:
        return image
    x = np.arange(image.shape[1], dtype=np.float32)
    shifted_x = x - float(offset_px)
    return np.vstack([np.interp(shifted_x, x, row, left=row[0], right=row[-1]) for row in image])


def _shift_axis_with_edge_fill(image: np.ndarray, shift: int, axis: int) -> np.ndarray:
    """
    沿指定轴平移图像，用边界像素填充暴露的像素。

    用于探测器面内/外平移校正。平移后的空白区域使用图像边界值填充。

    Args:
        image: 输入图像，可以是2D或多维数组。
        shift: 平移量（像素）。正值沿轴正方向平移，负值沿轴负方向平移。
        axis: 平移轴编号（0表示行/高度，1表示列/宽度）。

    Returns:
        np.ndarray: 平移后的图像，形状与输入相同。

    Notes:
        - 使用 np.pad 的 'edge' 模式自动用边界值补齐
        - 如果平移量过大（≥轴长），返回全边界值填充的数组
    """
    if shift == 0:
        return image

    length = image.shape[axis]
    if length == 0:
        return image

    magnitude = abs(shift)
    # 极限情况：平移量大于等于轴长，返回全边界值
    if magnitude >= length:
        # 选择移出的边界（正向移动取前边界，负向移动取后边界）
        edge_index = 0 if shift > 0 else length - 1
        edge = np.take(image, edge_index, axis=axis)
        # 用该边界值填满整个轴
        return np.repeat(np.expand_dims(edge, axis=axis), length, axis=axis)

    # 一般情况：使用np.pad进行边界填充平移
    pad_width = [(0, 0)] * image.ndim
    if shift > 0:
        # 正向平移：在前面填充，取后面的数据
        pad_width[axis] = (magnitude, 0)  # (前填充量, 后填充量)
        padded = np.pad(image, pad_width, mode="edge")  # 用边界值填充
        slices = [slice(None)] * image.ndim
        slices[axis] = slice(0, length)  # 取前length个元素
    else:
        # 负向平移：在后面填充，取前面的数据
        pad_width[axis] = (0, magnitude)
        padded = np.pad(image, pad_width, mode="edge")
        slices = [slice(None)] * image.ndim
        slices[axis] = slice(magnitude, magnitude + length)  # 跳过前magnitude个元素

    return padded[tuple(slices)]


def _apply_detector_offsets(
    projections: np.ndarray,
    offset_x_mm: float,
    offset_y_mm: float,
    px_size_x_mm: float,
    px_size_y_mm: float,
    callback: ProgressCallback | None,
) -> np.ndarray:
    """
    应用探测器平移偏移校正。

    将毫米级的探测器偏移转换为像素单位，对每个投影分别应用
    水平和竖直方向的平移，使用边界值填充平移后的空白区域。

    Args:
        projections: 投影图像栈，形状为(num_projections, height, width)。
        offset_x_mm: 探测器水平（列）偏移，单位毫米。
        offset_y_mm: 探测器竖直（行）偏移，单位毫米。
        px_size_x_mm: 探测器水平像素大小，单位毫米。
        px_size_y_mm: 探测器竖直像素大小，单位毫米。
        callback: 进度回调函数。

    Returns:
        np.ndarray: 校正后的投影栈，float32类型。

    Notes:
        - 使用四舍五入转换为整像素数
        - 若两个偏移都为0，直接返回原投影
    """
    if abs(offset_x_mm) < 1e-6 and abs(offset_y_mm) < 1e-6:
        return projections

    # 将毫米级偏移转换为整像素单位
    x_shift = int(round(offset_x_mm / max(px_size_x_mm, 1e-6)))
    y_shift = int(round(offset_y_mm / max(px_size_y_mm, 1e-6)))
    if x_shift == 0 and y_shift == 0:
        return projections

    _log(callback, f"应用探测器偏移校正：dx={offset_x_mm:.4g} mm, dy={offset_y_mm:.4g} mm")
    corrected = []
    for proj in projections:
        # 先平移列（x方向），axis=1表示宽度方向
        shifted = _shift_axis_with_edge_fill(proj, x_shift, axis=1)
        # 再平移行（y方向），axis=0表示高度方向
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
    """
    应用背景/暗电流校正。

    从每个投影中减去静态背景图像（如探测器的暗电流或环境背景）。
    若校正后数值为负，直接置为0（物理上投影强度不能为负）。

    Args:
        projections: 投影图像栈，形状为(num_projections, height, width)。
        background: 背景图像，形状必须为(height, width)，与投影的单张尺寸一致。
        callback: 进度回调函数。

    Returns:
        np.ndarray: 背景校正后的投影栈，float32类型，所有值≥0。

    Notes:
        - 若背景尺寸不匹配，将跳过校正并返回原投影
    """
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
    """
    为旋转中心估计构建衰减投影。

    将原始投影转换为衰减系数投影，过程包括：
    1. 可选的背景和平场校正
    2. 强度归一化
    3. 对数变换（Beer-Lambert定律）
    4. NaN/Inf异常值清理

    衰减投影用于鲁棒地检测物体轮廓，不受强度变化影响。

    Args:
        projections: 原始投影栈，形状为(num_projections, height, width)。
        folder: 包含校正图像的文件夹，若为None则跳过文件系统操作。
        use_background_correction: 是否应用背景校正。
        use_dark_flat: 是否应用暗场/亮场校正。

    Returns:
        np.ndarray: 衰减投影栈，float32类型，无异常值。

    Notes:
        - 对数变换假设投影值在[0, 1]范围内
        - 异常值（NaN/Inf）会被替换为0
    """
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
    """
    找到1D掩码中最长的连续True值段。

    用于检测投影中物体的连续轮廓区域。

    Args:
        mask: 一维布尔数组掩码。

    Returns:
        tuple[int, int] | None: (start_index, end_index) 的包含边界索引对。
            若mask中没有True值，返回None。

    Notes:
        - 返回的索引是包含的（inclusive），即[start, end]都包括在内
    """
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        return None

    breaks = np.flatnonzero(np.diff(indexes) > 1)
    starts = np.r_[indexes[0], indexes[breaks + 1]]
    ends = np.r_[indexes[breaks], indexes[-1]]
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def _estimate_center_from_silhouette(projections: np.ndarray) -> float:
    """
    从投影的物体轮廓中心估计旋转中心偏移。

    在多张投影中分别检测物体的轮廓区域，计算轮廓中心相对于
    几何中心的偏移，取中位数作为最终估计值。

    Args:
        projections: 衰减投影栈，形状为(num_projections, height, width)。

    Returns:
        float: 旋转中心偏移量（像素）。正值表示物体偏左，需向右校正。

    Raises:
        ValueError: 无法从任何投影中清晰检测到物体轮廓。

    Notes:
        - 采样24张或更少投影来加速计算
        - 使用百分位数(5%-95%)检测轮廓，具有鲁棒性
        - 忽略过小的轮廓区域（可能是噪声）
    """
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

    预处理步骤顺序（重要）：
    1. 背景校正（可选）- 减去探测器暗电流
    2. 暗场/亮场校正（可选）- 补偿光源不均匀性（平场校正）
    3. 强度归一化 - 确保值在[0, 1]范围
    4. 对数变换 - 转换为衰减系数 μ = -ln(I/I0)
    5. 旋转中心偏移校正 - 通过列平移补偿几何误差
    6. 探测器平移校正 - 补偿XY方向平移
    7. 探测器滚转校正 - 补偿探测器旋转

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
        - 每个预处理步骤都是可选的，取决于config设置
    """
    # 1. 复制投影为float32格式，准备进行后续运算
    data = projections.astype(np.float32, copy=True)

    # 2. 背景校正（可选）- 去除探测器暗电流噪声
    if config.use_background_correction:
        background = load_optional_background(folder)
        if background is not None:
            data = _apply_background_correction(data, background, callback)
        else:
            _log(callback, "未找到 background/bg/bkg 图像，跳过背景矫正。")

    # 3. 暗场/亮场校正（可选）- 补偿光源不均匀性
    if config.use_dark_flat:
        dark, flat = load_optional_calibration(folder)
        if dark is not None and flat is not None:
            _log(callback, "应用暗场/亮场校正。")
            # 平场校正公式：I_corrected = (I - I_dark) / (I_flat - I_dark)
            denom = np.maximum(flat - dark, 1.0)
            data = (data - dark) / denom
        else:
            _log(callback, "未找到完整 dark/flat 图像，跳过平场校正。")

    # 4. 强度归一化
    data = np.maximum(data, 1e-6)  # 避免对数计算中的零值
    if np.nanmax(data) > 2.0:
        # 如果强度范围超过[0,2]，进行归一化到[0,1]
        data = data / np.nanmax(data)
    
    # 5. 对数变换 - 根据Beer-Lambert定律：I = I0 * exp(-μx)
    # 所以 μ = -ln(I/I0)，当I0=1时，μ = -ln(I)
    data = -np.log(np.clip(data, 1e-6, None))
    # 清理可能产生的异常值
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # 6. 旋转中心偏移校正 - 通过列平移补偿中心偏移
    if abs(config.rotation_center_offset_px) > 1e-6:
        _log(callback, f"应用旋转中心偏移：{config.rotation_center_offset_px:.4g} px")
        data = np.stack([_shift_columns(proj, config.rotation_center_offset_px) for proj in data], axis=0)

    # 7. 探测器平移校正 - 补偿XY方向平移
    data = _apply_detector_offsets(
        data,
        config.detector_offset_x_mm,
        config.detector_offset_y_mm,
        config.detector_pixel_size_x_mm,
        config.detector_pixel_size_y_mm,
        callback,
    )
    
    # 8. 探测器滚转校正 - 补偿探测器旋转（绕光轴旋转）
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

    # 步骤1：加载投影图像
    _log(callback, f"读取投影目录：{projection_dir}")
    raw, files = load_projection_stack(projection_dir, limit=config.projection_count)
    if len(files) < config.projection_count:
        _log(callback, f"投影数量少于设置值：读取 {len(files)} 张。")
    else:
        _log(callback, f"读取投影：{len(files)} 张，尺寸 {raw.shape[2]} x {raw.shape[1]}")

    # 步骤2：预处理投影（校正+对数变换）
    projections = preprocess_projections(raw, projection_dir, config, callback)
    detector_rows, detector_cols = projections.shape[1], projections.shape[2]
    # 生成均匀分布的投影角度（0到2π）
    angles = np.linspace(0.0, 2.0 * np.pi, projections.shape[0], endpoint=False).astype(np.float32)

    # 导入ASTRA Toolbox（用于FDK重建）
    try:
        import astra
    except Exception as exc:
        raise RuntimeError("未能导入 ASTRA Toolbox，请先通过 uv 安装 astra-toolbox。") from exc

    # 步骤3：设置几何参数
    _log(callback, "创建 ASTRA 锥束几何。")
    # 投影几何：定义X光源、投影角度、探测器等参数
    proj_geom = astra.create_proj_geom(
        "cone",
        config.detector_pixel_size_x_mm,
        config.detector_pixel_size_y_mm,
        detector_rows,
        detector_cols,
        angles,
        config.source_object_distance_mm,  # 源到旋转中心的距离
        config.origin_detector_distance_mm,  # 旋转中心到探测器的距离
    )
    # 体数据几何：定义重建体积的大小、位置和体素大小
    vol_geom = astra.create_vol_geom(
        config.volume_size_y,
        config.volume_size_x,
        config.volume_size_z,
        # XYZ坐标范围（单位：毫米）
        -config.volume_size_x * config.voxel_size_mm / 2.0,
        config.volume_size_x * config.voxel_size_mm / 2.0,
        -config.volume_size_y * config.voxel_size_mm / 2.0,
        config.volume_size_y * config.voxel_size_mm / 2.0,
        -config.volume_size_z * config.voxel_size_mm / 2.0,
        config.volume_size_z * config.voxel_size_mm / 2.0,
    )

    # 步骤4：准备正弦图数据并执行FDK重建
    # 转置投影：(num_proj, height, width) -> (height, num_proj, width)
    # ASTRA要求的格式是(detector_rows, num_angles, detector_cols)
    sino = np.transpose(projections, (1, 0, 2))
    sino_id = astra.data3d.create("-sino", proj_geom, sino)
    rec_id = astra.data3d.create("-vol", vol_geom)
    alg_id = None

    try:
        # 配置FDK_CUDA算法（需要NVIDIA GPU）
        cfg = astra.astra_dict("FDK_CUDA")
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = rec_id
        alg_id = astra.algorithm.create(cfg)
        _log(callback, "开始 FDK_CUDA 重建。")
        # 执行重建算法
        astra.algorithm.run(alg_id)
        # 获取重建结果
        volume = astra.data3d.get(rec_id).astype(np.float32)
    finally:
        # 清理ASTRA资源
        if alg_id is not None:
            astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(rec_id)

    # 步骤5：导出结果
    np.save(output_dir / "volume.npy", volume)  # 保存原始体数据
    config.save(output_dir / "config.json")  # 保存配置参数
    if config.save_png_slices:
        _log(callback, "导出 PNG 切片。")
        save_png_stack(volume, output_dir / "slices")  # 导出PNG切片
    _log(callback, f"重建完成：{output_dir}")
    return volume
