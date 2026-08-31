from __future__ import annotations

import os
from pathlib import Path
import sys
import traceback

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import ReconstructionConfig
from .io_utils import load_projection_stack, normalize_for_display
from .reconstruction import run_fdk_reconstruction


class ImageView(QLabel):
    def __init__(self) -> None:
        super().__init__("请选择投影目录或开始重建")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 520)
        self.setStyleSheet("background: #111; color: #ddd; border: 1px solid #bbb;")
        self._image: np.ndarray | None = None

    def set_array(self, image: np.ndarray) -> None:
        self._image = normalize_for_display(image)
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._image is None:
            return
        h, w = self._image.shape
        qimage = QImage(self._image.data, w, h, w, QImage.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ReconstructionWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, config: ReconstructionConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            volume = run_fdk_reconstruction(self.config, self.log.emit)
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(volume)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tiny CT - CT重建算法验证工具")
        self.resize(1180, 760)
        self.projections: np.ndarray | None = None
        self.volume: np.ndarray | None = None
        self.worker_thread: QThread | None = None
        self.worker: ReconstructionWorker | None = None

        self._build_ui()
        self._load_default_folder()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_reconstruction_tab(), "CT重建")
        tabs.addTab(self._build_calibration_tab(), "参数校正")
        self.setCentralWidget(tabs)

        open_output = QAction("打开保存目录", self)
        open_output.triggered.connect(self.open_output_dir)
        self.menuBar().addAction(open_output)

    def _build_reconstruction_tab(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)

        left = QVBoxLayout()
        left.addWidget(self._build_path_group())
        left.addWidget(self._build_geometry_group())
        left.addWidget(self._build_volume_group())
        left.addWidget(self._build_action_group())
        left.addStretch(1)

        right = QVBoxLayout()
        self.image_view = ImageView()
        right.addWidget(self.image_view, 1)

        slider_row = QHBoxLayout()
        self.slice_label = QLabel("切片/投影：0")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.update_display_index)
        slider_row.addWidget(self.slice_label)
        slider_row.addWidget(self.slice_slider, 1)
        right.addLayout(slider_row)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(145)
        right.addWidget(self.log_box)

        layout.addLayout(left, 0)
        layout.addLayout(right, 1)
        return root

    def _build_calibration_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel("第一版支持通过参数进行几何校正；后续可扩展标定球自动校正和自动旋转中心估计。")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(self._build_correction_group())
        layout.addStretch(1)
        return page

    def _build_path_group(self) -> QGroupBox:
        group = QGroupBox("投影数据")
        layout = QGridLayout(group)
        self.projection_dir = QLineEdit(str(Path.cwd() / "proj"))
        browse = QPushButton("选择目录")
        browse.clicked.connect(self.choose_projection_dir)
        load = QPushButton("加载预览")
        load.clicked.connect(self.load_projection_preview)
        layout.addWidget(QLabel("投影目录"), 0, 0)
        layout.addWidget(self.projection_dir, 0, 1)
        layout.addWidget(browse, 0, 2)
        layout.addWidget(load, 1, 1, 1, 2)
        return group

    def _build_geometry_group(self) -> QGroupBox:
        group = QGroupBox("重建参数")
        form = QFormLayout(group)
        self.algorithm = QComboBox()
        self.algorithm.addItems(["FDK"])
        self.sod = self._double_spin(1.0, 100000.0, 200.0, " mm")
        self.sdd = self._double_spin(1.0, 100000.0, 800.0, " mm")
        self.proj_count = self._spin(1, 20000, 360)
        self.det_px_x = self._double_spin(0.0001, 100.0, 0.2, " mm")
        self.det_px_y = self._double_spin(0.0001, 100.0, 0.2, " mm")
        self.use_dark_flat = QCheckBox("使用 dark/flat")
        self.use_dark_flat.setChecked(True)
        form.addRow("算法", self.algorithm)
        form.addRow("源物距 SOD", self.sod)
        form.addRow("源探距 SDD", self.sdd)
        form.addRow("投影数量", self.proj_count)
        form.addRow("探测器横向像素间隔", self.det_px_x)
        form.addRow("探测器纵向像素间隔", self.det_px_y)
        form.addRow("", self.use_dark_flat)
        return group

    def _build_correction_group(self) -> QGroupBox:
        group = QGroupBox("几何校正")
        form = QFormLayout(group)
        self.det_roll = self._double_spin(-45.0, 45.0, 2.0, " deg")
        self.det_offset_x = self._double_spin(-10000.0, 10000.0, 0.0, " mm")
        self.det_offset_y = self._double_spin(-10000.0, 10000.0, 0.0, " mm")
        self.center_offset = self._double_spin(-10000.0, 10000.0, 0.0, " px")
        form.addRow("探测器面内偏转", self.det_roll)
        form.addRow("探测器横向偏移", self.det_offset_x)
        form.addRow("探测器纵向偏移", self.det_offset_y)
        form.addRow("旋转中心偏移", self.center_offset)
        return group

    def _build_volume_group(self) -> QGroupBox:
        group = QGroupBox("体数据设置")
        form = QFormLayout(group)
        self.vol_x = self._spin(16, 2048, 256)
        self.vol_y = self._spin(16, 2048, 256)
        self.vol_z = self._spin(16, 2048, 256)
        self.voxel = self._double_spin(0.0001, 100.0, 0.2, " mm")
        self.output_dir = QLineEdit(str(Path.cwd() / "recon_result"))
        form.addRow("X 尺寸", self.vol_x)
        form.addRow("Y 尺寸", self.vol_y)
        form.addRow("Z 切片数", self.vol_z)
        form.addRow("体素尺寸", self.voxel)
        form.addRow("保存目录", self.output_dir)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("重建操作")
        layout = QHBoxLayout(group)
        self.reconstruct_button = QPushButton("开始重建")
        self.reconstruct_button.clicked.connect(self.start_reconstruction)
        open_button = QPushButton("打开保存目录")
        open_button.clicked.connect(self.open_output_dir)
        layout.addWidget(self.reconstruct_button)
        layout.addWidget(open_button)
        return group

    def _double_spin(self, low: float, high: float, value: float, suffix: str = "") -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(4)
        box.setValue(value)
        box.setSuffix(suffix)
        return box

    def _spin(self, low: int, high: int, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(low, high)
        box.setValue(value)
        return box

    def _load_default_folder(self) -> None:
        if Path(self.projection_dir.text()).exists():
            self.load_projection_preview()

    def choose_projection_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择投影图像目录", self.projection_dir.text())
        if folder:
            self.projection_dir.setText(folder)
            self.load_projection_preview()

    def load_projection_preview(self) -> None:
        try:
            self.projections, files = load_projection_stack(self.projection_dir.text(), limit=self.proj_count.value())
        except Exception as exc:
            QMessageBox.warning(self, "加载失败", str(exc))
            return
        self.log(f"已加载投影预览：{len(files)} 张，尺寸 {self.projections.shape[2]} x {self.projections.shape[1]}")
        self.volume = None
        self.slice_slider.setRange(0, max(0, len(files) - 1))
        self.slice_slider.setValue(0)
        self.update_display_index(0)

    def update_display_index(self, index: int) -> None:
        if self.volume is not None:
            self.slice_label.setText(f"切片：{index + 1}/{self.volume.shape[0]}")
            self.image_view.set_array(self.volume[index])
        elif self.projections is not None:
            self.slice_label.setText(f"投影：{index + 1}/{self.projections.shape[0]}")
            self.image_view.set_array(self.projections[index])

    def make_config(self) -> ReconstructionConfig:
        return ReconstructionConfig(
            projection_dir=self.projection_dir.text(),
            output_dir=self.output_dir.text(),
            algorithm=self.algorithm.currentText(),
            source_object_distance_mm=self.sod.value(),
            source_detector_distance_mm=self.sdd.value(),
            projection_count=self.proj_count.value(),
            detector_pixel_size_x_mm=self.det_px_x.value(),
            detector_pixel_size_y_mm=self.det_px_y.value(),
            detector_roll_deg=self.det_roll.value(),
            detector_offset_x_mm=self.det_offset_x.value(),
            detector_offset_y_mm=self.det_offset_y.value(),
            rotation_center_offset_px=self.center_offset.value(),
            volume_size_x=self.vol_x.value(),
            volume_size_y=self.vol_y.value(),
            volume_size_z=self.vol_z.value(),
            voxel_size_mm=self.voxel.value(),
            use_dark_flat=self.use_dark_flat.isChecked(),
        )

    def start_reconstruction(self) -> None:
        self.reconstruct_button.setEnabled(False)
        config = self.make_config()
        self.log("提交重建任务。")
        self.worker_thread = QThread()
        self.worker = ReconstructionWorker(config)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_reconstruction_finished)
        self.worker.failed.connect(self.on_reconstruction_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def on_reconstruction_finished(self, volume: object) -> None:
        self.volume = volume
        if isinstance(self.volume, np.ndarray):
            self.slice_slider.setRange(0, max(0, self.volume.shape[0] - 1))
            self.slice_slider.setValue(self.volume.shape[0] // 2)
            self.update_display_index(self.slice_slider.value())
        self.reconstruct_button.setEnabled(True)

    def on_reconstruction_failed(self, message: str) -> None:
        self.log(message)
        QMessageBox.critical(self, "重建失败", message)
        self.reconstruct_button.setEnabled(True)

    def open_output_dir(self) -> None:
        path = Path(self.output_dir.text())
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)

    def log(self, message: str) -> None:
        self.log_box.append(message)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
