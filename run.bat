:: WhisperTyper Launcher
cd /d "%~dp0"

if not exist venv (
    echo Virtual environment not found!
    echo Please run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
start "" venv\Scripts\pythonw.exe whisper_gui.pyw
