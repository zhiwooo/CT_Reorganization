from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


IMAGE_PATTERNS = ("proj_*.tif", "proj_*.tiff", "proj_*.png", "proj_*.jpg", "proj_*.jpeg")


def find_projection_files(folder: str | Path) -> list[Path]:
    base = Path(folder)
    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(sorted(base.glob(pattern)))
    if not files:
        for pattern in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            files.extend(sorted(base.glob(pattern)))
    return files


def read_image(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(source)
    else:
        image = np.asarray(Image.open(source))
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.float32)


def load_projection_stack(folder: str | Path, limit: int | None = None) -> tuple[np.ndarray, list[Path]]:
    files = find_projection_files(folder)
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No projection images found in {folder}")
    stack = np.stack([read_image(path) for path in files], axis=0)
    return stack, files


def load_optional_calibration(folder: str | Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    base = Path(folder)
    dark_files = sorted(base.glob("dark_*.tif")) + sorted(base.glob("dark_*.tiff"))
    flat_files = sorted(base.glob("flat_*.tif")) + sorted(base.glob("flat_*.tiff"))
    dark = read_image(dark_files[0]) if dark_files else None
    flat = read_image(flat_files[0]) if flat_files else None
    return dark, flat


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(array, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(np.min(array)), float(np.max(array))
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def save_png_stack(volume: np.ndarray, output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for index, slice_image in enumerate(volume):
        display = normalize_for_display(slice_image)
        Image.fromarray(display).save(target / f"slice_{index:04d}.png")
