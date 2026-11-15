# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir specification for the standalone Linux ``neurowhisper`` AppDir bundle.

IMPORTANT: Uses ``COLLECT`` (onedir) rather than ``--onefile`` so SquashFS AppImage FUSE-mounts
in <50 ms without unpacking ~220 MB into ``/tmp/_MEIxxxxxx`` on every launch.
"""

from __future__ import annotations

import ctypes.util
import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

SPEC_DIR = Path(SPECPATH).resolve()
LINUX_ROOT = SPEC_DIR.parent
SRC_DIR = LINUX_ROOT / "src"
RESOURCES_DIR = LINUX_ROOT / "resources"
ENTRYPOINT = SRC_DIR / "neurowhisper_linux" / "main.py"

datas = [
    (str(RESOURCES_DIR / "neurowhisper.png"), "resources"),
    (str(RESOURCES_DIR / "neurowhisper.desktop"), "resources"),
]
binaries = []
hiddenimports = [
    *collect_submodules("neurowhisper_linux"),
    "PIL._tkinter_finder",
    "jeepney",
    "jeepney.io",
    "jeepney.io.blocking",
    "pynput.keyboard._xorg",
    "pynput.mouse._xorg",
    "pystray._appindicator",
    "pystray._gtk",
    "pystray._xorg",
    "sounddevice",
]


def _safe_collect_all(pkg_name: str) -> None:
    try:
        pkg_datas, pkg_bins, pkg_hidden = collect_all(pkg_name)
        datas.extend(pkg_datas)
        binaries.extend(pkg_bins)
        hiddenimports.extend(pkg_hidden)
    except Exception:
        pass


def _safe_collect_data(pkg_name: str) -> None:
    try:
        datas.extend(collect_data_files(pkg_name))
    except Exception:
        pass


def _safe_collect_libs(pkg_name: str) -> None:
    try:
        binaries.extend(collect_dynamic_libs(pkg_name))
    except Exception:
        pass


def _safe_copy_metadata(dist_name: str) -> None:
    try:
        datas.extend(copy_metadata(dist_name))
    except Exception:
        pass


# Bundle CustomTkinter themes/fonts, faster-whisper Silero VAD ONNX asset, and ONNX ASR configs
_safe_collect_all("customtkinter")
_safe_collect_data("faster_whisper")
_safe_collect_all("onnx_asr")

# Bundle native C++/Rust shared libraries for offline inference & audio decoding
for _native_pkg in ("ctranslate2", "onnxruntime", "av", "tokenizers", "_sounddevice_data"):
    _safe_collect_libs(_native_pkg)

# Preserve package metadata inspected at runtime by onnx-asr / huggingface_hub / tqdm
for _dist in ("onnx-asr", "onnxruntime", "faster-whisper", "ctranslate2", "huggingface_hub", "tokenizers", "tqdm"):
    _safe_copy_metadata(_dist)

# Bundle system libportaudio.so.2 if present on build machine so fresh Linux desktops work out-of-the-box
_portaudio_candidates = [
    "/usr/lib/x86_64-linux-gnu/libportaudio.so.2",
    "/usr/lib/aarch64-linux-gnu/libportaudio.so.2",
    "/usr/lib64/libportaudio.so.2",
    "/usr/lib/libportaudio.so.2",
]
_found_pa = ctypes.util.find_library("portaudio")
if _found_pa and os.path.isabs(_found_pa):
    _portaudio_candidates.insert(0, _found_pa)
for _pa_path in _portaudio_candidates:
    if os.path.isfile(_pa_path):
        binaries.append((os.path.realpath(_pa_path), "."))
        break

# Deduplicate hiddenimports while preserving order
hiddenimports = list(dict.fromkeys(hiddenimports))

# Exclude Windows-only / heavy unused toolkits to keep AppImage ~220 MB
excludes = [
    "keyboard",
    "torch",
    "torchaudio",
    "torchvision",
    "nemo",
    "matplotlib",
    "scipy",
    "pandas",
    "pytest",
    "IPython",
    "notebook",
]

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(SRC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="neurowhisper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="neurowhisper",
)
