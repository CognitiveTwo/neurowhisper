@echo off
:: neurowhisper launcher (Windows)
:: Creates/updates a private virtual environment, then starts the app.
:: The venv lives in %LOCALAPPDATA% so it is never synced or shared between machines.
setlocal EnableDelayedExpansion
pushd "%~dp0"

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

:: ---------------------------------------------------------------- launch
start "" "%VENV_DIR%\Scripts\pythonw.exe" "%~dp0whisper_gui.pyw"
popd
endlocal
exit /b 0
