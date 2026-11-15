# neurowhisper for Linux

Standalone Linux desktop application (`.AppImage` + Python package) for local & cloud AI voice dictation, real-time typing, Parakeet ONNX acceleration, and bilingual (EN/DE) Speech Insights.

Completely self-contained inside `linux/` — zero runtime dependencies on the Windows codebase and **no root-only `keyboard` daemon required**.

---

## 1. Quick Start (`.AppImage` — Zero Python Required)

```bash
chmod +x neurowhisper-linux-x86_64.AppImage

# Optional: install ~/.local/share/applications/neurowhisper.desktop + icon
./neurowhisper-linux-x86_64.AppImage --install-desktop

# Launch neurowhisper
./neurowhisper-linux-x86_64.AppImage
```

> **Ubuntu 22.04 / 24.04 FUSE Note:** If AppImages complain about `libfuse.so.2`, install `sudo apt install libfuse2` (or `libfuse2t64` on Ubuntu 24.04+), or run with `APPIMAGE_EXTRACT_AND_RUN=1 ./neurowhisper-linux-x86_64.AppImage`.

---

## 2. X11 & Wayland Global Shortcuts, Paste & Live Typing

`neurowhisper` auto-detects your active display server (`$WAYLAND_DISPLAY`, `$XDG_SESSION_TYPE`, `$DISPLAY`):

### Global Hotkeys (Default: `F9` Live Dictation, `Ctrl+Alt+S` Batch Recording)
- **X11 Sessions:** Works out of the box via `pynput` with zero root privileges.
- **Wayland Sessions (Dual-Stack):**
  1. **XDG Desktop Portal `GlobalShortcuts`:** Automatically binds via session D-Bus (`org.freedesktop.portal.GlobalShortcuts`) on KDE Plasma, Hyprland, and GNOME 47+.
  2. **Universal Wayland CLI IPC (`100%` of Wayland compositors):** Because Wayland isolates global keystrokes by design, you can bind native desktop shortcuts on *any* compositor (GNOME, KDE, Sway, Hyprland, Cosmic, Niri) to:
     ```bash
     neurowhisper --toggle-live
     neurowhisper --toggle-batch
     neurowhisper --stop
     neurowhisper --show
     ```
     *(Or `/path/to/neurowhisper-linux-x86_64.AppImage --toggle-live` — 1-click copy buttons are built directly into **Configuration → Global Shortcuts**.)*

#### Example Wayland Compositor Bindings
- **GNOME:** *Settings → Keyboard → View and Customize Shortcuts → Custom Shortcuts* → Command: `/path/to/neurowhisper-linux-x86_64.AppImage --toggle-live`, Shortcut: `F9`.
- **Sway (`~/.config/sway/config`):**
  ```text
  bindsym F9 exec ~/Applications/neurowhisper-linux-x86_64.AppImage --toggle-live
  bindsym Ctrl+Mod1+s exec ~/Applications/neurowhisper-linux-x86_64.AppImage --toggle-batch
  ```
- **Hyprland (`~/.config/hypr/hyprland.conf`):**
  ```text
  bind = , F9, exec, ~/Applications/neurowhisper-linux-x86_64.AppImage --toggle-live
  bind = CTRL ALT, S, exec, ~/Applications/neurowhisper-linux-x86_64.AppImage --toggle-batch
  ```

### Recommended Optional Host Tools for Clipboard & Auto-Paste
- **X11:** Install `xclip` (or `xsel`) and `xdotool` for terminal-aware auto-paste (`neurowhisper` automatically detects `gnome-terminal`, `kitty`, `alacritty`, `konsole`, `wezterm`, `ghostty`, `xterm`, etc. and sends **`Ctrl+Shift+V`** in terminals vs **`Ctrl+V`** in standard apps!).
  ```bash
  # Debian / Ubuntu
  sudo apt install xclip xdotool libportaudio2
  ```
- **Wayland:** Install `wl-clipboard` (`wl-copy`) and `wtype` (wlroots/Sway/Hyprland) or `ydotool` for virtual keyboard injection. If virtual keyboard injection is blocked by your compositor, transcriptions are still copied immediately to the Wayland clipboard with an audio chime and desktop notification.
  ```bash
  # Debian / Ubuntu
  sudo apt install wl-clipboard wtype libnotify-bin libportaudio2
  ```

---

## 3. Hardware Acceleration & Optional CUDA 12 GPU Add-On

Out of the box, `neurowhisper` ships lean (~220 MB) with fast **CPU INT8** (`faster-whisper` + **Parakeet TDT 0.6B v3** via `onnx-asr`).

- **NVIDIA CUDA 12 GPU Acceleration:**
  - Automatically uses `/usr/local/cuda/lib64` if installed on your host, **or** open **Configuration → CUDA 12 GPU Add-on...** to install `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` directly into user space (`~/.local/share/neurowhisper/cuda-libs`) with 1 click:
    ```bash
    python3 -m pip install --upgrade --target ~/.local/share/neurowhisper/cuda-libs \
        nvidia-cublas-cu12 nvidia-cudnn-cu12
    ```
- **Intel Arc / Iris Xe / Core Ultra NPU (OpenVINO):**
  - Install `openvino-genai` when running from source to unlock `OpenVINO (Intel GPU)`.

---

## 4. XDG Storage Layout & Multi-Machine Cloud Sync

| Purpose | Default XDG Path |
|---|---|
| Configuration & `storage.json` | `~/.config/neurowhisper/` |
| Transcriptions, Stats & AI Insights (`cuda-libs/`) | `~/.local/share/neurowhisper/` |
| Downloaded Speech Models & Rotating Logs | `~/.cache/neurowhisper/{models,logs}/` |
| Single-Instance & CLI Control Socket | `$XDG_RUNTIME_DIR/neurowhisper/control.sock` |

**Multi-Machine Sync (Dropbox / Syncthing / Nextcloud):** Pick a shared folder under **Configuration → Custom Data / Cloud Sync Folder**. `neurowhisper` stores machine-local hardware/API settings in `whisper_config.<HOST>.json` and per-host stats/transcripts in `whisper_stats.<HOST>.json` and `transcriptions/YYYY-MM.<HOST>.txt` — 100% compatible with Windows and macOS `neurowhisper` installs!

---

## 5. Running from Source, Testing & Building `.AppImage`

```bash
# Run from source (inside an isolated Python 3.10–3.12 virtualenv)
pip install -r linux/requirements-linux.txt
PYTHONPATH=linux/src python3 -m neurowhisper_linux.main

# Run unit + Xvfb GUI smoke tests
PYTHONPATH=linux/src python3 linux/tests/smoke_test.py
xvfb-run -a env PYTHONPATH=linux/src python3 linux/tests/smoke_test.py

# Build standalone SquashFS AppImage (linux/dist/neurowhisper-linux-x86_64.AppImage)
./linux/packaging/build_appimage.sh
```
