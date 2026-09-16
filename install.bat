@echo off
:: neurowhisper installer (Windows)
:: Creates/updates a private virtual environment. Run this once, then use run.bat.
:: The venv lives in %LOCALAPPDATA% so it is never synced or shared between machines.
setlocal EnableDelayedExpansion
pushd "%~dp0"

echo ============================================
echo    neurowhisper - installer
echo ============================================
echo.

set "VENV_DIR=%LOCALAPPDATA%\neurowhisper\venv"
set "REQ=%~dp0requirements.txt"
set "STAMP=%VENV_DIR%\requirements.installed"

:: ---------------------------------------------------------------- interpreter
set "PYTHON_CMD="
py -3 --version >nul 2>&1 && set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
    python --version >nul 2>&1 && set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    python3 --version >nul 2>&1 && set "PYTHON_CMD=python3"
)
if not defined PYTHON_CMD (
    echo [neurowhisper] No Python interpreter found on this machine.
    echo                Install Python 3.11 or newer from https://www.python.org/downloads/
    echo                and tick "Add python.exe to PATH" during setup.
    popd
    pause
    exit /b 1
)

:: ---------------------------------------------------------------- version gate
if not exist "%VENV_DIR%\Scripts\python.exe" (
    %PYTHON_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    if errorlevel 1 (
        set "PYVER=unknown"
        for /f "tokens=*" %%v in ('%PYTHON_CMD% -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "PYVER=%%v"
        echo [neurowhisper] neurowhisper needs Python 3.11 or newer ^(found !PYVER!^)
        echo                Download a newer Python from https://www.python.org/downloads/
        popd
        pause
        exit /b 1
    )
    echo [neurowhisper] First run on this machine - creating virtual environment...
    echo [neurowhisper] Venv location: %VENV_DIR%
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [neurowhisper] ERROR: Failed to create the virtual environment.
        popd
        pause
        exit /b 1
    )
)

:: ---------------------------------------------------------------- dependencies
set "NEED_INSTALL=0"
if not exist "%STAMP%" set "NEED_INSTALL=1"
if "%NEED_INSTALL%"=="0" (
    fc /b "%STAMP%" "%REQ%" >nul 2>&1
    if errorlevel 1 set "NEED_INSTALL=1"
)

if "%NEED_INSTALL%"=="1" (
    echo [neurowhisper] Installing/updating dependencies ^(this can take a few minutes^)...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REQ%"
    if errorlevel 1 (
        echo [neurowhisper] ERROR: Failed to install dependencies.
        popd
        pause
        exit /b 1
    )
    copy /y "%REQ%" "%STAMP%" >nul
    echo [neurowhisper] Dependencies are up to date.
) else (
    echo [neurowhisper] Dependencies already up to date.
)

echo.
echo ============================================
echo    neurowhisper - installation complete
echo ============================================
echo.
echo Start the app by double-clicking: run.bat
echo.
echo The first launch downloads the Small speech model (~460 MB).
echo That needs an internet connection once and takes a few minutes.
echo.
echo Optional: for NVIDIA GPU acceleration, drop the CUDA/cuDNN DLLs into
echo this folder. Without them the app runs on the CPU.
echo.
popd
pause
endlocal
exit /b 0
