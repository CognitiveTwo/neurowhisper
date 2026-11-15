# neurowhisper for macOS (Standalone `.app` & `.dmg`)

A self-contained, native dark-mode macOS speech-to-text dictation & Speech Insights application completely decoupled from the Windows build. Ships as a standalone `neurowhisper.app` inside a compressed `.dmg`—**no Python or Homebrew installation required on end-user Macs**.

---

## Key macOS Features

- **Hardware-Accelerated On-Device Inference:**
  - **Apple Metal (`mlx-whisper`):** Native Metal GPU acceleration on Apple Silicon (`M1` / `M2` / `M3` / `M4`) for `Large v3 Turbo`, `Distil Large v3.5`, `Large v3`, `Medium`, `Small`, and `Base`.
  - **ONNX ASR (`CoreMLExecutionProvider` + CPU):** Hardware-accelerated **Parakeet TDT 0.6B v3** (25 languages) via Apple CoreML / Neural Engine with automatic CPU fallback.
  - **Faster-Whisper (CTranslate2 INT8):** Universal ARM64 & Intel x86_64 CPU fallback.
  - **OpenAI Cloud (+AI Copy-Editor):** Optional cloud transcription (`gpt-transcribe`, `gpt-4o-transcribe`, `whisper-1`) with `<transcript>`-guarded AI copy editing.
- **Sonoma / Sequoia Crash-Free Input (`platform_mac.py`):**
  - Uses PyObjC `Quartz` `CGEventTap` with pure integer Carbon virtual keycodes (`kVK_*`)—avoiding `TISCopyCurrentKeyboardInputSource` off-main-thread crashes on German/QWERTZ and international keyboard layouts.
  - Direct Unicode live typing (`CGEventKeyboardSetUnicodeString`) supporting umlauts (`äöüß`), accents, and emoji natively, plus zero-focus-steal `NSPasteboard` clipboard + `Cmd+V` batch pasting.
- **Native Aqua Dark UI & Menu Bar:**
  - 100% `CustomTkinter` dark controls (`SF Pro Text` & `Menlo` typography scaled for 72-DPI Aqua), macOS Menu Bar microphone status icon, Spaces-friendly floating mini-pill, and all **10 Speech Insights & AI Daily Brief** tabs.
- **Cross-Platform Cloud Sync Compatible:**
  - Optional **Data & Cloud Sync Folder** picker (iCloud Drive, Dropbox, Syncthing) shares `transcriptions/YYYY-MM-DD.txt`, `whisper_stats.<HOST>.json`, and `ai_insights.json` seamlessly across macOS, Windows, and Linux while keeping multi-hundred-MB model weights in local `~/Library/Caches/neurowhisper/models`.

---

## Installing from `.dmg` (First-Run Gatekeeper & TCC Guide)

1. Open `neurowhisper-macos-arm64.dmg` (Apple Silicon) or `neurowhisper-macos-x86_64.dmg` (Intel).
2. Drag **`neurowhisper.app`** onto the **`Applications`** folder shortcut.
3. **Clear Gatekeeper Quarantine (Ad-Hoc Signed Builds):**
   - Double-click **`Unquarantine.command`** inside the mounted DMG window (or run `xattr -cr /Applications/neurowhisper.app` in Terminal). This clears the browser download quarantine flag so macOS opens community ad-hoc builds cleanly.
4. **Grant macOS Permissions:**
   - **Microphone:** Click **Allow** when macOS prompts on your first recording.
   - **Accessibility** *(required for global hotkeys & live typing into other apps)*: Click **Grant Accessibility** on the banner inside `neurowhisper` (opens *System Settings → Privacy & Security → Accessibility*), enable `neurowhisper`, then click **Re-check**.
   - **Updating from an older `.dmg`:** Because ad-hoc builds compute a fresh CDHash per build, toggle `neurowhisper` **OFF** and back **ON** once in *Accessibility* settings after replacing `/Applications/neurowhisper.app`.

---

## CLI Control & macOS Shortcuts / Raycast Integration

A running `neurowhisper` instance listens on a private `0600` Unix domain socket (`~/Library/Application Support/neurowhisper/control.sock`). You can trigger dictation from **Apple Shortcuts**, **Raycast**, **Alfred**, **Karabiner-Elements**, or Terminal:

```bash
/Applications/neurowhisper.app/Contents/MacOS/neurowhisper --toggle-live
/Applications/neurowhisper.app/Contents/MacOS/neurowhisper --toggle-batch
/Applications/neurowhisper.app/Contents/MacOS/neurowhisper --stop
/Applications/neurowhisper.app/Contents/MacOS/neurowhisper --show
/Applications/neurowhisper.app/Contents/MacOS/neurowhisper --status
```

---

## Directory Layout on macOS

| Purpose | Default Path | Cloud-Synced? |
|---|---|---|
| Config, Stats, Transcripts | `~/Library/Application Support/neurowhisper/` *(or custom Sync Folder)* | Optional (`sync_dir`) |
| Downloaded STT Models | `~/Library/Caches/neurowhisper/models/` | Never (local cache) |
| Rotating App & Crash Logs | `~/Library/Logs/neurowhisper/` | Never (local logs) |

---

## Building the `.dmg` from Source on macOS

Requires macOS 13+ and [`uv`](https://docs.astral.sh/uv/) (which automatically downloads a relocatable Python 3.12 with modern Tcl/Tk 8.6.14+):

```bash
cd macos
./packaging/build_dmg.sh
```

Output artifacts are written to `macos/dist/neurowhisper-macos-<arch>.dmg` (plus `.sha256`).

### Running Tests

```bash
PYTHONPATH=macos/src python3 -m unittest discover -s macos/tests -p "test_*.py" -v
PYTHONPATH=macos/src python3 macos/tests/smoke_test.py
```
