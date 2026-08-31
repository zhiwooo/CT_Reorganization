from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ReconstructionConfig:
    projection_dir: str = "proj"
    output_dir: str = "recon_result"
    algorithm: str = "FDK"
    source_object_distance_mm: float = 200.0
    source_detector_distance_mm: float = 800.0
    projection_count: int = 360
    detector_pixel_size_x_mm: float = 0.2
    detector_pixel_size_y_mm: float = 0.2
    detector_roll_deg: float = 2.0
    detector_offset_x_mm: float = 0.0
    detector_offset_y_mm: float = 0.0
    rotation_center_offset_px: float = 0.0
    volume_size_x: int = 256
    volume_size_y: int = 256
    volume_size_z: int = 256
    voxel_size_mm: float = 0.2
    use_dark_flat: bool = True
    save_png_slices: bool = True

    @property
    def origin_detector_distance_mm(self) -> float:
        return self.source_detector_distance_mm - self.source_object_distance_mm

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
