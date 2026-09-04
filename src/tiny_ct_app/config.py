"""
CT重建配置管理模块。

本模块定义了CT重建所需的所有参数，包括几何参数、体数据参数等。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ReconstructionConfig:
    """
    CT重建配置数据类。

    包含CT重建所需的所有参数，如源物距(SOD)、源探距(SDD)、
    探测器参数、体数据大小等。所有距离单位为毫米，角度单位为度。
    """
    # 文件和输出路径
    projection_dir: str = ""  # 投影图像目录路径，导入前为空
    output_dir: str = "recon_result"  # 重建结果输出目录

    # 重建算法
    algorithm: str = "FDK"  # 重建算法（目前仅支持FDK）

    # 几何参数（单位：毫米）
    source_object_distance_mm: float = 0.0  # 源物距(SOD): X射线源到旋转中心的距离
    source_detector_distance_mm: float = 0.0  # 源探距(SDD): X射线源到探测器的距离
    projection_count: int = 0  # 投影总数量，由导入图像数量决定

    # 探测器参数
    detector_pixel_size_x_mm: float = 0.0  # 探测器横向像素间隔（mm）
    detector_pixel_size_y_mm: float = 0.0  # 探测器纵向像素间隔（mm）
    detector_roll_deg: float = 0.0  # 探测器面内偏转角度（度）
    detector_offset_x_mm: float = 0.2  # 探测器横向偏移（mm）
    detector_offset_y_mm: float = 0.2  # 探测器纵向偏移（mm）

    # 旋转中心校正
    rotation_center_offset_px: float = 0.0  # 旋转中心偏移（像素）

    # 体数据参数
    volume_size_x: int = 0  # 体数据X方向大小（像素）
    volume_size_y: int = 0  # 体数据Y方向大小（像素）
    volume_size_z: int = 0  # 体数据Z方向大小（像素）
    voxel_size_mm: float = 0.0  # 体素大小（mm）

    # 处理选项
    use_background_correction: bool = True  # 是否使用背景图像校正
    use_dark_flat: bool = True  # 是否使用暗场/亮场校正
    save_png_slices: bool = True  # 是否导出PNG切片

    @property
    def origin_detector_distance_mm(self) -> float:
        """
        计算原点到探测器的距离。

        Returns:
            float: ODD = SDD - SOD（源探距减去源物距）
        """
        return self.source_detector_distance_mm - self.source_object_distance_mm

    def validate(self) -> None:
        """
        校验重建参数，尽早给出清晰错误信息。

        Raises:
            ValueError: 参数不满足锥束CT重建的基本几何或尺寸约束。
        """
        if self.source_object_distance_mm <= 0:
            raise ValueError("源物距 SOD 必须大于 0。")
        if self.source_detector_distance_mm <= self.source_object_distance_mm:
            raise ValueError("源探距 SDD 必须大于源物距 SOD。")
        if self.projection_count <= 0:
            raise ValueError("投影数量必须大于 0。")
        if self.detector_pixel_size_x_mm <= 0 or self.detector_pixel_size_y_mm <= 0:
            raise ValueError("探测器像素间隔必须大于 0。")
        if self.volume_size_x <= 0 or self.volume_size_y <= 0 or self.volume_size_z <= 0:
            raise ValueError("体数据尺寸必须全部大于 0。")
        if self.voxel_size_mm <= 0:
            raise ValueError("体素尺寸必须大于 0。")
        if not self.projection_dir.strip():
            raise ValueError("请先导入投影图像目录。")

    def save(self, path: str | Path) -> None:
        """
        将配置保存为JSON文件。

        Args:
            path: 保存路径。如果父目录不存在则自动创建。

        Raises:
            IOError: 文件写入失败时抛出。
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
