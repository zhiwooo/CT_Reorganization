from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from tiny_ct_app.config import ReconstructionConfig
from tiny_ct_app.io_utils import (
    find_projection_files,
    load_optional_background,
    load_optional_calibration,
    load_projection_stack,
    save_png_stack,
)
from tiny_ct_app.reconstruction import (
    _apply_background_correction,
    _apply_detector_offsets,
    estimate_rotation_center_offset,
)


def _write_image(path: Path, value: int) -> None:
    Image.fromarray(np.full((4, 5), value, dtype=np.uint8)).save(path)


def test_empty_projection_path_finds_no_files() -> None:
    assert find_projection_files("") == []

    try:
        load_projection_stack("")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected empty projection path to raise FileNotFoundError")


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


def test_background_file_is_loaded(tmp_path: Path) -> None:
    _write_image(tmp_path / "background_0000.tif", 3)

    background = load_optional_background(tmp_path)

    assert background is not None
    assert background.shape == (4, 5)
    assert np.all(background == 3)


def test_fallback_projection_search_excludes_dark_and_flat(tmp_path: Path) -> None:
    _write_image(tmp_path / "scan_a.tif", 10)
    _write_image(tmp_path / "scan_b.tif", 11)
    _write_image(tmp_path / "background_0000.tif", 1)
    _write_image(tmp_path / "bg_0000.tif", 1)
    _write_image(tmp_path / "bkg_0000.tif", 1)
    _write_image(tmp_path / "dark_0000.tif", 0)
    _write_image(tmp_path / "flat_0000.tif", 255)

    files = find_projection_files(tmp_path)
    stack, loaded_files = load_projection_stack(tmp_path)

    assert [path.name for path in files] == ["scan_a.tif", "scan_b.tif"]
    assert [path.name for path in loaded_files] == ["scan_a.tif", "scan_b.tif"]
    assert stack.shape == (2, 4, 5)


def test_background_correction_subtracts_and_clips() -> None:
    projections = np.array([[[5, 10], [2, 1]]], dtype=np.float32)
    background = np.array([[3, 4], [5, 1]], dtype=np.float32)

    corrected = _apply_background_correction(projections, background, callback=None)

    np.testing.assert_array_equal(corrected, np.array([[[2, 6], [0, 0]]], dtype=np.float32))


def test_estimate_rotation_center_offset_returns_zero_for_centered_silhouette() -> None:
    projections = np.ones((4, 6, 9), dtype=np.float32)
    projections[:, :, 2:7] = 0.25

    assert estimate_rotation_center_offset(projections) == 0.0


def test_estimate_rotation_center_offset_returns_correction_for_shifted_silhouette() -> None:
    projections = np.ones((4, 6, 9), dtype=np.float32)
    projections[:, :, 1:6] = 0.25

    assert estimate_rotation_center_offset(projections) == 1.0


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


def test_main_window_starts_with_empty_projection_path() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from tiny_ct_app.main import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    assert app is not None
    assert window.projection_dir.text() == ""
    assert window.center_offset.value() == 0.0
    assert window.voxel.value() == 0.05
    assert window.vol_x.value() == 512
    assert window.vol_y.value() == 512
    assert window.vol_z.value() == 512
    assert window.correction_formula.currentText() == "μ = -ln((max(I - B, 0) - D) / (F - D))"
    assert window.projections is None
    assert window.volume is None
    assert not window.slice_slider.isEnabled()
    assert not window.reconstruct_button.isEnabled()
    assert not window.estimate_center_button.isEnabled()
    assert not window.calc_voxel_button.isEnabled()


def test_main_window_calculates_voxel_size_from_magnification() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from tiny_ct_app.main import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.det_px_x.setValue(0.2)
    window.sod.setValue(200.0)
    window.sdd.setValue(800.0)
    window.projections = np.zeros((2, 512, 512), dtype=np.float32)
    window.update_data_dependent_controls()

    window.calculate_voxel_size()

    assert app is not None
    assert window.calc_voxel_button.isEnabled()
    assert window.voxel.value() == 0.05
    assert window.vol_x.value() == 512
    assert window.vol_y.value() == 512
    assert window.vol_z.value() == 512
