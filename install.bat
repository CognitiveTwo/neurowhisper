@echo off
:: neurowhisper installer (Windows)
:: Creates/updates a private virtual environment. Run this once, then use run.bat.
:: The venv lives in %LOCALAPPDATA% so it is never synced or shared between machines.
setlocal DisableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%LOCALAPPDATA%\neurowhisper\venv"
setlocal EnableDelayedExpansion
pushd "!SCRIPT_DIR!"
set "PYTHONUTF8=1"

echo ============================================
echo    neurowhisper - installer
echo ============================================
echo.

set "REQ=!SCRIPT_DIR!requirements.txt"
set "STAMP=!VENV_DIR!\requirements.installed"

:: Check if an existing venv is healthy (survives base Python upgrades)
set "VENV_OK=0"
if exist "!VENV_DIR!\Scripts\python.exe" (
    "!VENV_DIR!\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "VENV_OK=1"
)

if "%VENV_OK%"=="0" (
    :: ---------------------------------------------------------------- interpreter
    set "PYTHON_CMD="
    set "FOUND_ANY_PY="
    for %%C in ("py -3.13" "py -3.12" "py -3.11" "py -3" "python" "python3") do (
        if not defined PYTHON_CMD (
            %%~C --version >nul 2>&1
            if !errorlevel! equ 0 (
                if not defined FOUND_ANY_PY set "FOUND_ANY_PY=%%~C"
                %%~C -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
                if !errorlevel! equ 0 set "PYTHON_CMD=%%~C"
            )
        )
    )
    if not defined PYTHON_CMD (
        if defined FOUND_ANY_PY (
            set "PYVER=unknown"
            for /f "tokens=*" %%v in ('!FOUND_ANY_PY! -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "PYVER=%%v"
            echo [neurowhisper] neurowhisper needs Python 3.11 or newer ^(found !PYVER!^).
            echo                Download a newer Python from https://www.python.org/downloads/
        ) else (
            echo [neurowhisper] No Python interpreter found on this machine.
            echo                Install Python 3.11 or newer from https://www.python.org/downloads/
            echo                and tick "Add python.exe to PATH" during setup.
        )
        popd
        pause
        exit /b 1
    )

    echo [neurowhisper] Creating virtual environment...
    echo [neurowhisper] Venv location: !VENV_DIR!
    if exist "!STAMP!" del /f /q "!STAMP!" >nul 2>&1
    !PYTHON_CMD! -m venv --clear "!VENV_DIR!"
    if !errorlevel! neq 0 (
        echo [neurowhisper] ERROR: Failed to create the virtual environment.
        popd
        pause
        exit /b 1
    )
)

:: ---------------------------------------------------------------- dependencies
set "NEED_INSTALL=0"
if not exist "!STAMP!" set "NEED_INSTALL=1"
if "%NEED_INSTALL%"=="0" (
    fc /b "!STAMP!" "!REQ!" >nul 2>&1
    if !errorlevel! neq 0 set "NEED_INSTALL=1"
)

if "%NEED_INSTALL%"=="1" (
    echo [neurowhisper] Installing/updating dependencies ^(this can take a few minutes^)...
    "!VENV_DIR!\Scripts\python.exe" -m pip install --upgrade pip --quiet
    "!VENV_DIR!\Scripts\python.exe" -m pip install -r "!REQ!"
    if !errorlevel! neq 0 (
        echo [neurowhisper] ERROR: Failed to install dependencies.
        popd
        pause
        exit /b 1
    )
    copy /y "!REQ!" "!STAMP!" >nul
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
echo The first launch downloads your selected local speech model ^(default Small ~460 MB,
echo or Parakeet TDT v3 / Large v3 Turbo in Configuration^).
echo.
echo Optional: for NVIDIA GPU acceleration, drop the CUDA/cuDNN DLLs into
echo this folder. Without them the app runs on CPU or ONNX Runtime.
echo.
popd
pause
endlocal
exit /b 0
