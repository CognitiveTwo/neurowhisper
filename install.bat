@echo off
echo ============================================
echo    WhisperTyper - Installation Script
echo ============================================
echo.

:: Check if Python is installed
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
if exist venv (
    echo      Virtual environment already exists, skipping...
) else (
    py -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo ============================================
echo    Installation Complete!
echo ============================================
echo.
echo To run WhisperTyper, double-click: run.bat
echo.
echo NOTE: For GPU acceleration, you'll need CUDA DLLs.
echo       The app will fall back to CPU mode if unavailable.
echo.
pause
