"""Runtime DLL search path setup for the PyInstaller build."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _add_dll_dir(path: Path) -> None:
    if not path.exists():
        return
    path_text = str(path)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(path_text)
    os.environ["PATH"] = path_text + os.pathsep + os.environ.get("PATH", "")


base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
for relative in (
    "PySide6",
    "shiboken6",
    "numpy.libs",
    "scipy.libs",
    "nvidia/cuda_runtime/bin",
    "nvidia/cufft/bin",
):
    _add_dll_dir(base / relative)
