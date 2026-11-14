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

export PYTHONUTF8=1

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HOME}/.local/share/neurowhisper/venv"
VENV_PY="${VENV_DIR}/bin/python"
REQ="${APP_DIR}/requirements.txt"
STAMP="${VENV_DIR}/requirements.installed"

# ---------------------------------------------------------------- fast path
if [ -x "$VENV_PY" ] && [ -f "$STAMP" ] && cmp -s "$STAMP" "$REQ"; then
    if "$VENV_PY" -V >/dev/null 2>&1; then
        cd "$APP_DIR"
        exec "$VENV_PY" "${APP_DIR}/whisper_gui.pyw"
    fi
fi

# ---------------------------------------------------------------- venv / version gate
venv_ok=0
if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then
    venv_ok=1
fi

if [ "$venv_ok" -eq 0 ]; then
    PYTHON_CMD=""
    FOUND_ANY=""
    for candidate in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            [ -z "$FOUND_ANY" ] && FOUND_ANY="$candidate"
            if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then
                PYTHON_CMD="$candidate"
                break
            fi
        fi
    done

    if [ -z "$PYTHON_CMD" ]; then
        if [ -n "$FOUND_ANY" ]; then
            found="$("$FOUND_ANY" -c "import sys; print(sys.version.split()[0])" 2>/dev/null || echo unknown)"
            echo "[neurowhisper] neurowhisper needs Python 3.11 or newer (found ${found})"
            echo "               Download a newer Python from https://www.python.org/downloads/"
        else
            echo "[neurowhisper] No Python interpreter found."
            echo "               Install Python 3.11 or newer from https://www.python.org/downloads/"
        fi
        exit 1
    fi

    echo "[neurowhisper] First run on this machine - creating virtual environment..."
    echo "[neurowhisper] Venv location: ${VENV_DIR}"
    mkdir -p "$(dirname "$VENV_DIR")"
    rm -f "$STAMP"
    # Debian/Ubuntu ship python3 without the venv module; the stock error
    # ("ensurepip is not available") does not say which package to install.
    "$PYTHON_CMD" -m venv --clear "$VENV_DIR" || {
        echo "[neurowhisper] venv creation failed - on Debian/Ubuntu install python3-venv:"
        echo "               sudo apt-get install -y python3-venv"
        exit 1
    }
fi

# ---------------------------------------------------------------- dependencies
need_install=0
if [ ! -f "$STAMP" ] || ! cmp -s "$STAMP" "$REQ"; then
    need_install=1
fi

if [ "$need_install" -eq 1 ]; then
    echo "[neurowhisper] Installing/updating dependencies (this can take a few minutes)..."
    "$VENV_PY" -m pip install --upgrade pip --quiet || true
    "$VENV_PY" -m pip install -r "$REQ"
    cp "$REQ" "$STAMP"
    echo "[neurowhisper] Dependencies are up to date."
else
    echo "[neurowhisper] Dependencies already up to date."
fi

# ---------------------------------------------------------------- launch
cd "$APP_DIR"
exec "$VENV_PY" "${APP_DIR}/whisper_gui.pyw"
