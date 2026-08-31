"""
Tiny CT 工作站主程序模块。

提供PySide6 GUI界面，用于加载投影图像、配置几何参数、
执行CT重建以及查看结果。支持实时预览投影和重建切片。
"""

from __future__ import annotations

import argparse
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
    """用于显示投影图像或重建切片的图像查看器。"""

    def __init__(self) -> None:
        """初始化图像查看器。"""
        super().__init__("请选择投影目录或开始重建")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 520)
        self.setStyleSheet("background: #111; color: #ddd; border: 1px solid #bbb;")
        self._image: np.ndarray | None = None

    def set_array(self, image: np.ndarray) -> None:
        """
        设置并显示图像数组。

        Args:
            image: 输入图像数组。
        """
        self._image = normalize_for_display(image)
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802
        """处理窗口大小改变事件。"""
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        """刷新显示的图像（调整大小到当前窗口）。"""
        if self._image is None:
            return
        display = np.ascontiguousarray(self._image)
        h, w = display.shape
        qimage = QImage(display.data, w, h, w, QImage.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ReconstructionWorker(QObject):
    """
    在独立线程中执行CT重建的工作线程。

    信号：
        finished(object): 重建成功完成，参数为重建的体数据
        failed(str): 重建失败，参数为错误信息
        log(str): 进度日志消息
    """

    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, config: ReconstructionConfig) -> None:
        """
        初始化重建工作线程。

        Args:
            config: 重建配置对象。
        """
        super().__init__()
        self.config = config

    def run(self) -> None:
        """执行重建任务。信号会自动发出进度信息。"""
        try:
            volume = run_fdk_reconstruction(self.config, self.log.emit)
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(volume)


class MainWindow(QMainWindow):
    """
    Tiny CT工作站主窗口。

    提供用户界面用于：
    - 加载投影图像
    - 配置重建参数（几何参数、体数据参数、校正参数）
    - 执行CT重建
    - 实时预览投影和重建结果
    """

    def __init__(self) -> None:
        """初始化主窗口。"""
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
        """构建用户界面。"""
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
        """
        从UI控件读取参数并创建重建配置对象。

        Returns:
            ReconstructionConfig: 包含用户输入的所有参数的配置对象。
        """
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
        """
        启动重建任务。

        在独立线程中执行重建以避免阻塞UI。
        重建状态通过信号进行通信。
        """
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
        """
        处理重建完成事件。

        更新显示的体数据并刷新切片视图。

        Args:
            volume: 重建完成的体数据。
        """
        self.volume = volume
        if isinstance(self.volume, np.ndarray):
            self.slice_slider.setRange(0, max(0, self.volume.shape[0] - 1))
            self.slice_slider.setValue(self.volume.shape[0] // 2)
            self.update_display_index(self.slice_slider.value())
        self.reconstruct_button.setEnabled(True)

    def on_reconstruction_failed(self, message: str) -> None:
        """
        处理重建失败事件。

        显示错误对话框并记录错误信息。

        Args:
            message: 错误信息。
        """
        self.log(message)
        QMessageBox.critical(self, "重建失败", message)
        self.reconstruct_button.setEnabled(True)

    def open_output_dir(self) -> None:
        """在文件管理器中打开输出目录。"""
        path = Path(self.output_dir.text())
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)

    def log(self, message: str) -> None:
        """
        在日志框中添加消息。

        Args:
            message: 要添加的消息。
        """
        self.log_box.append(message)


def _build_cli_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Tiny CT workstation")
    parser.add_argument("--projection-dir", default=str(Path.cwd() / "proj"), help="投影图像目录")
    parser.add_argument("--output-dir", default=str(Path.cwd() / "recon_result"), help="重建输出目录")
    parser.add_argument("--projection-count", type=int, default=360, help="读取投影的数量上限")
    parser.add_argument("--sod", type=float, default=200.0, help="源物距（mm）")
    parser.add_argument("--sdd", type=float, default=800.0, help="源探距（mm）")
    parser.add_argument("--detector-pixel-size-x", type=float, default=0.2, help="探测器横向像素间隔（mm）")
    parser.add_argument("--detector-pixel-size-y", type=float, default=0.2, help="探测器纵向像素间隔（mm）")
    parser.add_argument("--detector-roll", type=float, default=2.0, help="探测器面内偏转（deg）")
    parser.add_argument("--detector-offset-x", type=float, default=0.0, help="探测器横向偏移（mm）")
    parser.add_argument("--detector-offset-y", type=float, default=0.0, help="探测器纵向偏移（mm）")
    parser.add_argument("--rotation-center-offset", type=float, default=0.0, help="旋转中心偏移（px）")
    parser.add_argument("--vol-size-x", type=int, default=256, help="重建体数据 X 尺寸")
    parser.add_argument("--vol-size-y", type=int, default=256, help="重建体数据 Y 尺寸")
    parser.add_argument("--vol-size-z", type=int, default=256, help="重建体数据 Z 尺寸")
    parser.add_argument("--voxel-size", type=float, default=0.2, help="体素尺寸（mm）")
    parser.add_argument("--no-dark-flat", action="store_true", help="关闭 dark/flat 校正")
    parser.add_argument("--no-png", action="store_true", help="不导出 PNG 切片")
    parser.add_argument("--gui", action="store_true", help="强制启动 GUI 模式")
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """运行命令行重建模式。"""
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    config = ReconstructionConfig(
        projection_dir=args.projection_dir,
        output_dir=args.output_dir,
        projection_count=args.projection_count,
        source_object_distance_mm=args.sod,
        source_detector_distance_mm=args.sdd,
        detector_pixel_size_x_mm=args.detector_pixel_size_x,
        detector_pixel_size_y_mm=args.detector_pixel_size_y,
        detector_roll_deg=args.detector_roll,
        detector_offset_x_mm=args.detector_offset_x,
        detector_offset_y_mm=args.detector_offset_y,
        rotation_center_offset_px=args.rotation_center_offset,
        volume_size_x=args.vol_size_x,
        volume_size_y=args.vol_size_y,
        volume_size_z=args.vol_size_z,
        voxel_size_mm=args.voxel_size,
        use_dark_flat=not args.no_dark_flat,
        save_png_slices=not args.no_png,
    )

    try:
        volume = run_fdk_reconstruction(config, lambda message: print(f"[tiny-ct] {message}"))
    except Exception as exc:
        print(f"重建失败：{exc}", file=sys.stderr)
        return 1

    print(f"重建完成: shape={volume.shape}, output_dir={config.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    应用程序入口点。

    若命令行带参数，优先执行 CLI 重建；否则启动 GUI。

    Returns:
        int: 应用程序退出代码。
    """
    if argv is None:
        argv = sys.argv[1:]

    if argv and not ("--gui" in argv or "gui" in argv):
        return cli_main(argv)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
