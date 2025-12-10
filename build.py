#!/usr/bin/env python3
"""
WhisperTyper Build Script
Creates a lightweight distributable package.

Usage:
    python build.py

Output:
    dist/WhisperTyper/ - Folder with WhisperTyper.exe and dependencies
    dist/WhisperTyper.zip - Zipped package for distribution
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Configuration
SPEC_FILE = "build.spec"
DIST_NAME = "WhisperTyper"
OUTPUT_DIR = "dist"


def clean_previous_builds():
    """Remove previous build artifacts"""
    print("🧹 Cleaning previous builds...")
    
    dirs_to_clean = ["build", "dist", "__pycache__"]
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"   Removed {d}/")


def run_pyinstaller():
    """Run PyInstaller with the spec file"""
    print("\n📦 Running PyInstaller...")
    
    cmd = [sys.executable, "-m", "PyInstaller", SPEC_FILE, "--noconfirm"]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("❌ PyInstaller failed!")
        sys.exit(1)
    
    print("✅ PyInstaller completed successfully")


def create_readme():
    """Create a README for the distribution"""
    readme_path = os.path.join(OUTPUT_DIR, DIST_NAME, "README.txt")
    
    readme_content = """WhisperTyper - Voice-to-Text Dictation
======================================

QUICK START:
1. Run WhisperTyper.exe
2. Select a model from the dropdown (will download on first use)
3. Press F9 for live dictation, F8 for batch recording

GPU ACCELERATION (Optional):
If you have an NVIDIA GPU and want faster transcription:
1. Download CUDA DLLs from:
   https://github.com/Purfview/whisper-standalone-win/releases/tag/libs
2. Extract the DLL files to this folder (next to WhisperTyper.exe)
3. Restart WhisperTyper

Without CUDA DLLs, WhisperTyper will use CPU mode (slower but works).

HOTKEYS:
- F9: Toggle Live Mode (types as you speak)
- F8: Toggle Batch Mode (records, then transcribes)

MODELS:
- Small: Fast, good for CPU
- Medium: Balanced speed/accuracy
- Large v3: Best quality (needs GPU)

Created by DR.M @ neuroflash.com
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 Created {readme_path}")


def create_zip():
    """Create a ZIP archive of the distribution"""
    print("\n📦 Creating ZIP archive...")
    
    dist_folder = os.path.join(OUTPUT_DIR, DIST_NAME)
    zip_path = os.path.join(OUTPUT_DIR, f"{DIST_NAME}.zip")
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    shutil.make_archive(
        os.path.join(OUTPUT_DIR, DIST_NAME),
        'zip',
        OUTPUT_DIR,
        DIST_NAME
    )
    
    # Get file size
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"✅ Created {zip_path} ({size_mb:.1f} MB)")


def print_summary():
    """Print build summary"""
    dist_folder = os.path.join(OUTPUT_DIR, DIST_NAME)
    
    print("\n" + "=" * 50)
    print("BUILD COMPLETE!")
    print("=" * 50)
    print(f"\n📁 Distribution folder: {os.path.abspath(dist_folder)}")
    print(f"📦 ZIP archive: {os.path.abspath(os.path.join(OUTPUT_DIR, DIST_NAME + '.zip'))}")
    print("\nTo test the build:")
    print(f"   1. Navigate to: {dist_folder}")
    print(f"   2. Run: WhisperTyper.exe")
    print("\nNote: CUDA DLLs are NOT included. Users can:")
    print("   - Use CPU mode (works out of the box)")
    print("   - Download CUDA DLLs separately for GPU acceleration")


def main():
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 50)
    print("WhisperTyper Build Script")
    print("=" * 50)
    
    # Check for PyInstaller
    try:
        import PyInstaller
        print(f"✅ PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller not found. Install with: pip install pyinstaller")
        sys.exit(1)
    
    # Check for spec file
    if not os.path.exists(SPEC_FILE):
        print(f"❌ Spec file not found: {SPEC_FILE}")
        sys.exit(1)
    
    clean_previous_builds()
    run_pyinstaller()
    create_readme()
    create_zip()
    print_summary()


if __name__ == "__main__":
    main()
