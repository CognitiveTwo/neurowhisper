#!/usr/bin/env bash
# neurowhisper launcher for macOS / Linux.
#
# EXPERIMENTAL: neurowhisper is developed and tested on Windows 10/11. This
# script exists so the app can be started on macOS and Linux, but global
# hotkeys, clipboard typing and GPU acceleration are not verified there.
# Expect rough edges; please report what breaks.
#
# Linux additionally needs system packages for Tk and PortAudio, e.g.:
#   sudo apt-get install -y python3-tk python3-venv libportaudio2
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HOME}/.local/share/neurowhisper/venv"
REQ="${APP_DIR}/requirements.txt"
STAMP="${VENV_DIR}/requirements.installed"

# ---------------------------------------------------------------- interpreter
PYTHON_CMD=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_CMD="$candidate"
        break
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo "[neurowhisper] No Python interpreter found."
    echo "               Install Python 3.11 or newer from https://www.python.org/downloads/"
    exit 1
fi

# ---------------------------------------------------------------- version gate
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    if ! "$PYTHON_CMD" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
        found="$("$PYTHON_CMD" -c "import sys; print(sys.version.split()[0])" 2>/dev/null || echo unknown)"
        echo "[neurowhisper] neurowhisper needs Python 3.11 or newer (found ${found})"
        echo "               Download a newer Python from https://www.python.org/downloads/"
        exit 1
    fi
    echo "[neurowhisper] First run on this machine - creating virtual environment..."
    echo "[neurowhisper] Venv location: ${VENV_DIR}"
    mkdir -p "$(dirname "$VENV_DIR")"
    # Debian/Ubuntu ship python3 without the venv module; the stock error
    # ("ensurepip is not available") does not say which package to install.
    "$PYTHON_CMD" -m venv "$VENV_DIR" || {
        echo "[neurowhisper] venv creation failed - on Debian/Ubuntu install python3-venv:"
        echo "               sudo apt-get install -y python3-venv"
        exit 1
    }
fi

VENV_PY="${VENV_DIR}/bin/python"

# ---------------------------------------------------------------- dependencies
need_install=0
if [ ! -f "$STAMP" ] || ! cmp -s "$STAMP" "$REQ"; then
    need_install=1
fi

if [ "$need_install" -eq 1 ]; then
    echo "[neurowhisper] Installing/updating dependencies (this can take a few minutes)..."
    "$VENV_PY" -m pip install --upgrade pip --quiet
    "$VENV_PY" -m pip install -r "$REQ"
    cp "$REQ" "$STAMP"
    echo "[neurowhisper] Dependencies are up to date."
else
    echo "[neurowhisper] Dependencies already up to date."
fi

# ---------------------------------------------------------------- launch
cd "$APP_DIR"
exec "$VENV_PY" "${APP_DIR}/whisper_gui.pyw"
