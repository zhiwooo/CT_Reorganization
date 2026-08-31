from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tiny_ct_app.config import ReconstructionConfig
from tiny_ct_app.io_utils import (
    find_projection_files,
    load_optional_calibration,
    load_projection_stack,
    save_png_stack,
)
from tiny_ct_app.reconstruction import _apply_detector_offsets


def _write_image(path: Path, value: int) -> None:
    Image.fromarray(np.full((4, 5), value, dtype=np.uint8)).save(path)


def test_calibration_files_are_loaded_without_unpack_error(tmp_path: Path) -> None:
    _write_image(tmp_path / "dark_0000.tif", 2)
    _write_image(tmp_path / "flat_0000.tif", 10)

    dark, flat = load_optional_calibration(tmp_path)

    assert dark is not None
    assert flat is not None
    assert dark.shape == (4, 5)
    assert flat.shape == (4, 5)
    assert np.all(dark == 2)
    assert np.all(flat == 10)


def test_fallback_projection_search_excludes_dark_and_flat(tmp_path: Path) -> None:
    _write_image(tmp_path / "scan_a.tif", 10)
    _write_image(tmp_path / "scan_b.tif", 11)
    _write_image(tmp_path / "dark_0000.tif", 0)
    _write_image(tmp_path / "flat_0000.tif", 255)

    files = find_projection_files(tmp_path)
    stack, loaded_files = load_projection_stack(tmp_path)

    assert [path.name for path in files] == ["scan_a.tif", "scan_b.tif"]
    assert [path.name for path in loaded_files] == ["scan_a.tif", "scan_b.tif"]
    assert stack.shape == (2, 4, 5)


def test_save_png_stack_removes_stale_slices(tmp_path: Path) -> None:
    save_png_stack(np.zeros((3, 4, 5), dtype=np.float32), tmp_path)
    save_png_stack(np.zeros((1, 4, 5), dtype=np.float32), tmp_path)

    assert [path.name for path in sorted(tmp_path.glob("slice_*.png"))] == ["slice_0000.png"]


def test_detector_offsets_handle_shifts_larger_than_image() -> None:
    projections = np.arange(6, dtype=np.float32).reshape(1, 2, 3)

    shifted = _apply_detector_offsets(
        projections,
        offset_x_mm=100.0,
        offset_y_mm=-100.0,
        px_size_x_mm=1.0,
        px_size_y_mm=1.0,
        callback=None,
    )

    assert shifted.shape == projections.shape
    assert np.all(shifted == shifted[0, 0, 0])


def test_config_validation_rejects_invalid_geometry() -> None:
    config = ReconstructionConfig(source_object_distance_mm=200.0, source_detector_distance_mm=100.0)

    try:
        config.validate()
    except ValueError as exc:
        assert "SDD" in str(exc)
    else:
        raise AssertionError("Expected invalid geometry to raise ValueError")
