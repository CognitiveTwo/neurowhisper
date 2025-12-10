# -*- mode: python ; coding: utf-8 -*-
"""
WhisperTyper PyInstaller Spec File
Builds a lightweight distributable without CUDA DLLs and models.
Users download CUDA DLLs separately if they have an NVIDIA GPU.
"""

import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

block_cipher = None

# Get the directory containing this spec file
spec_dir = os.path.dirname(os.path.abspath(SPEC))

# Collect ctranslate2 native binaries
ctranslate2_binaries = collect_dynamic_libs('ctranslate2')
ctranslate2_datas = collect_data_files('ctranslate2')

a = Analysis(
    ['whisper_gui.pyw'],
    pathex=[spec_dir],
    binaries=ctranslate2_binaries,
    datas=ctranslate2_datas,
    hiddenimports=[
        'faster_whisper',
        'ctranslate2',
        'huggingface_hub',
        'sounddevice',
        'scipy.io.wavfile',
        'numpy',
        'keyboard',
        'tkinter',
        'tkinter.ttk',
        'tkinter.scrolledtext',
        'tkinter.messagebox',
        # Required for pkg_resources
        'jaraco',
        'jaraco.text',
        'jaraco.functools',
        'jaraco.context',
        'pkg_resources',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude CUDA libraries - users download separately
        'torch',
        'torchvision', 
        'torchaudio',
        # Exclude development tools
        'pip',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out CUDA DLLs from binaries (they might be auto-detected)
cuda_dll_patterns = ['cublas', 'cudnn', 'cudart', 'nvrtc', 'cufft', 'curand', 'cusolver', 'cusparse']
a.binaries = [b for b in a.binaries if not any(pattern in b[0].lower() for pattern in cuda_dll_patterns)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WhisperTyper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one: icon='icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WhisperTyper',
)
