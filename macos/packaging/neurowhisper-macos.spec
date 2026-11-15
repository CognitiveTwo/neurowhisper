# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir .app bundle specification for neurowhisper on macOS.

IMPORTANT: Uses PyInstaller `onedir` (EXE + COLLECT + BUNDLE) — NEVER `--onefile` —
so neurowhisper.app launches instantaneously from `/Applications/neurowhisper.app`
without unpacking 250+ MB into `/tmp/_MEIxxxxxx` or invalidating TCC bundle identity.
"""

from pathlib import Path
import plistlib
import platform
import sys

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)

SPEC_DIR = Path(SPECPATH).resolve()
MACOS_DIR = SPEC_DIR.parent
SRC_DIR = MACOS_DIR / "src"
RES_DIR = MACOS_DIR / "resources"

datas = [
    (str(RES_DIR / "icon.png"), "resources"),
    (str(RES_DIR / "icon.icns"), "resources"),
]
binaries = []
hiddenimports = [
    "neurowhisper_mac",
    "neurowhisper_mac.paths",
    "neurowhisper_mac.config",
    "neurowhisper_mac.ipc",
    "neurowhisper_mac.platform_mac",
    "neurowhisper_mac.menubar",
    "neurowhisper_mac.insights",
    "neurowhisper_mac.gui",
    "neurowhisper_mac.backends",
    "neurowhisper_mac.backends.base",
    "neurowhisper_mac.backends.mlx_whisper_backend",
    "neurowhisper_mac.backends.onnx_asr_backend",
    "neurowhisper_mac.backends.faster_whisper_backend",
    "neurowhisper_mac.backends.openai_backend",
    "PIL._tkinter_finder",
    "pystray._darwin",
    "sounddevice",
    "_sounddevice_data",
    "objc",
    "Foundation",
    "AppKit",
    "Quartz",
    "ApplicationServices",
]


def _safe_collect_all(pkg_name: str) -> None:
    try:
        d, b, h = collect_all(pkg_name)
        datas.extend(d)
        binaries.extend(b)
        hiddenimports.extend(h)
    except Exception:
        pass


def _safe_collect_data(pkg_name: str) -> None:
    try:
        datas.extend(collect_data_files(pkg_name))
    except Exception:
        pass


def _safe_collect_dylibs(pkg_name: str) -> None:
    try:
        binaries.extend(collect_dynamic_libs(pkg_name))
    except Exception:
        pass


def _safe_copy_metadata(dist_name: str) -> None:
    try:
        datas.extend(copy_metadata(dist_name))
    except Exception:
        pass


# UI theme assets, VAD ONNX graphs, ONNX ASR preprocessors, and native dylibs
_safe_collect_all("customtkinter")
_safe_collect_data("faster_whisper")
_safe_collect_all("onnx_asr")
_safe_collect_data("_sounddevice_data")

for _native_pkg in ("onnxruntime", "ctranslate2", "av", "tokenizers", "_sounddevice_data"):
    _safe_collect_dylibs(_native_pkg)

# Fallback: bundle PortAudio dylib from Homebrew prefix if wheel omits _sounddevice_data
if not any("portaudio" in Path(src).name.lower() for src, _ in binaries):
    for _pa_candidate in (
        Path("/opt/homebrew/lib/libportaudio.2.dylib"),
        Path("/usr/local/lib/libportaudio.2.dylib"),
    ):
        if _pa_candidate.exists():
            binaries.append((str(_pa_candidate.resolve()), "."))
            break

# Bundle Apple Silicon MLX + Metal shader library (mlx.metallib) on arm64
if sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64"):
    _safe_collect_all("mlx")
    _safe_collect_all("mlx_whisper")

for _meta_pkg in (
    "onnx-asr",
    "onnxruntime",
    "faster-whisper",
    "huggingface_hub",
    "tokenizers",
    "tqdm",
    "openai",
    "mlx-whisper",
):
    _safe_copy_metadata(_meta_pkg)

with open(RES_DIR / "Info.plist", "rb") as _plist_fh:
    info_plist_dict = plistlib.load(_plist_fh)

a = Analysis(
    [str(SRC_DIR / "neurowhisper_mac" / "main.py")],
    pathex=[str(SRC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pynput", "torch", "torchaudio", "torchvision", "matplotlib", "scipy"],
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
    target_arch=None,
    codesign_identity=None,
    entitlements_file=str(RES_DIR / "entitlements.plist"),
    icon=str(RES_DIR / "icon.icns"),
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

app = BUNDLE(
    coll,
    name="neurowhisper.app",
    icon=str(RES_DIR / "icon.icns"),
    bundle_identifier="com.neuroflash.neurowhisper",
    info_plist=info_plist_dict,
)
