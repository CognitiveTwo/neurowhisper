# neurowhisper

[![CI](https://github.com/CognitiveTwo/neurowhisper/actions/workflows/ci.yml/badge.svg)](https://github.com/CognitiveTwo/neurowhisper/actions/workflows/ci.yml)

A local voice-dictation app for the desktop. Hold a hotkey, speak, and the text is
typed into whatever window you were already in — your editor, your browser, your
chat client. Transcription runs on your own machine with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), so nothing leaves the
computer unless you explicitly turn on cloud mode.

## What you get

- **Live mode** — streams your speech and types it as you talk.
- **Batch mode** — record a longer stretch, then transcribe it in one go.
- **Automatic GPU/CPU selection** — uses CUDA when it can, falls back to CPU otherwise.
- **Customizable hotkeys** — any key combination, changed from the UI.
- **Statistics** — words dictated, speaking time, and usage over time.
- **Optional cloud mode** — send audio to the OpenAI API instead, if you prefer.
- **Optional AI Insights** — daily summaries of what you dictated, and an
  [`analysis/` toolkit](analysis/README.md) that reports on them over time.

## Requirements

- **Windows 10 or 11** — the supported platform.
- **macOS / Linux — experimental.** `run.sh` starts the app there, but global
  hotkeys, typing into other windows and GPU acceleration are unverified.
- **Python 3.11 or newer.** Install it from
  [python.org](https://www.python.org/downloads/) and tick *Add python.exe to PATH*.
- ~2 GB RAM for the default model, 4 GB+ for the larger ones.
- Optional: an NVIDIA GPU for faster transcription (see [GPU acceleration](#gpu-acceleration)).

## Install and run

**Windows**

1. Download or clone this repository.
2. Double-click **`install.bat`**. It creates a private virtual environment in
   `%LOCALAPPDATA%\neurowhisper\venv` and installs the dependencies.
3. Double-click **`run.bat`** to start the app.

`run.bat` re-checks the dependencies on every start and reinstalls them
automatically if `requirements.txt` has changed, so you can skip straight to
`run.bat` after a `git pull`.

**macOS / Linux (experimental)**

```bash
chmod +x run.sh
./run.sh
```

On Linux, install the system packages for Tk and PortAudio first:

```bash
sudo apt-get install -y python3-tk python3-venv libportaudio2
```

### First launch

The first start downloads the **Small** speech model — about **460 MB**. That
needs an internet connection once and takes a few minutes on a normal line. The
status bar shows the download progress; the app is usable as soon as it reads
*Ready*. Afterwards the model is cached under `models/` and everything works
offline.

### Add your OpenAI API key (optional)

Dictation needs no account and no key. A key of your own unlocks two extra
features:

- **Cloud transcription** — the **Online** mode sends audio to the OpenAI API
  instead of running a model locally.
- **AI Insights** — short daily summaries of what you dictated, plus the
  [`analysis/` toolkit](analysis/README.md) that turns them into a report.

To add one:

1. Create a key at
   [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. Open the **Configuration** section in the app.
3. Paste the key into the **API Key** field and click **Save**. It takes effect
   immediately — no restart.

The key is written to a **per-machine config file** named
`whisper_config.<HOSTNAME>.json` next to the app, never to the shared
`whisper_config.json`. That file is excluded by `.gitignore`. **Never commit or
share it** — it holds your key in plain text.

Cost is on your own OpenAI account: cloud transcription is billed per minute of
audio, and AI Insights costs a few cents per day of dictation. Everything else —
local transcription, hotkeys, statistics, the speech-insight charts — works
without a key and without spending anything.

## Hotkeys

| Action | Default |
|--------|---------|
| Live mode (type while you speak) | **F9** |
| Batch mode (record, then transcribe) | **Ctrl+Alt+S** |

Change them in the app: open the **Configuration** section, click the field for the
hotkey you want to change, press the new combination, and it is saved
immediately. If a hotkey stops responding, click back into the app window or
click the **Rebind** button next to the hotkey fields.

## Models

| Model | Download | RAM | Best for |
|-------|----------|-----|----------|
| Base | ~150 MB | 2 GB | Quick notes, low-spec machines |
| Small (default) | ~460 MB | 2 GB | General use, comfortable on CPU |
| Medium | ~1.5 GB | 4 GB | Better accuracy |
| Large-v3 | ~3 GB | 8 GB | Best accuracy, GPU recommended |

Pick a model in the app; it downloads on first use.

## GPU acceleration

Transcription on an NVIDIA GPU is several times faster than on CPU.

1. Download the CUDA/cuDNN DLLs from
   [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs).
2. Extract them into the application folder (next to `whisper_gui.pyw`).
3. Restart the app and set the backend to **Auto** — it detects CUDA and switches
   to the GPU, and silently falls back to CPU if the DLLs are missing or the
   driver is too old.

The DLLs are large (~1.8 GB) and are deliberately not part of this repository.

**Other backends.** Intel GPUs can run the model through OpenVINO, and NVIDIA
GPUs can alternatively use the Parakeet TDT model via NVIDIA NeMo. Both are
optional and need extra packages installed into the app's virtual environment:

```bash
pip install -r requirements-openvino.txt   # Intel GPU (OpenVINO)
pip install -r requirements-nemo.txt       # Parakeet TDT (NeMo, NVIDIA GPU)
```

Then pick the backend in the app's Configuration section.

## Optional: OpenAI cloud transcription

Instead of running a model locally you can send audio to the OpenAI API: switch
the transcription mode to **Online**. It needs your own API key — see
[Add your OpenAI API key](#add-your-openai-api-key-optional). Online mode is
useful on a machine that is too slow for a local model, or when you want the
larger cloud models; it is the only mode in which audio leaves your computer.

## Analyse your own data

The [`analysis/` toolkit](analysis/README.md) works on the transcripts in
`transcriptions/` and the AI Insights cache. It builds per-day mood and topic
labels, clusters the topics across time, and renders a self-contained HTML
report you can open in a browser. It runs separately from the app and needs an
OpenAI API key for the labelling steps.

## Privacy

Everything stays on your machine by default:

- Audio is transcribed locally and the recordings are temporary files that are
  deleted after use.
- Transcripts are written to `transcriptions/` on your disk. Delete them at any
  time; nothing is uploaded.
- Anything leaves the machine **only** if you turn on the optional OpenAI cloud
  mode (audio) or the opt-in AI Insights feature (transcript text, summarised
  into daily notes). Both are off until you enable them and supply your own API
  key, and both stop the moment you remove it.
- There is no telemetry and no analytics.

## Troubleshooting

Logs live in the `logs/` folder next to the app:

| File | What is in it |
|------|---------------|
| `logs/app.log` | Normal runtime log — start here. |
| `logs/crash.log` | Written when startup fails outright. |
| `logs/hotkey.log` | Hotkey registration and health checks. |

**Hotkeys do nothing.** Global hotkeys need the app to be allowed to install a
keyboard hook. Some antivirus and endpoint-protection tools block that silently
— add the app folder to the exclusions. Applications running as administrator
also swallow hotkeys from non-elevated apps; start neurowhisper as
administrator too if you need it there. As a first step, click back into the app
window or use the **Rebind** button, which re-registers the hooks.

**No microphone found.** Check the input device dropdown in the settings; if it
is empty, Windows has no active recording device, or the app was denied
microphone access under *Settings → Privacy & security → Microphone*.

**Model download fails.** The status bar shows the failure and the details land
in `logs/app.log`. It is almost always a proxy or firewall blocking
huggingface.co.

**The app will not start.** Read `logs/crash.log`. Re-running `install.bat`
rebuilds the virtual environment if a dependency is broken.

## Notes on the implementation

**Hotkey robustness.** Windows drops keyboard hooks under a variety of
conditions (UAC dialogs, lock screen, elevated windows). The app re-registers
hotkeys every 3 minutes, runs a 10-second watchdog on the handlers, refreshes on
window focus and on restore-from-minimize, and offers a manual refresh button.

**Adaptive GUI loop.** The UI refresh runs at 30 ms while transcribing and drops
to 1000 ms when idle, skipping all canvas redraws. The original fixed 30 ms loop
burned about a third of a CPU core continuously and caused system-wide typing
lag; the adaptive loop makes idle CPU usage effectively zero. See
`update_gui_loop()` in `whisper_gui.pyw`.

## License

[MIT](LICENSE) — © 2026 Jonathan Mall.

## Credits

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[CTranslate2](https://github.com/OpenNMT/CTranslate2) and
[customtkinter](https://github.com/TomSchimansky/CustomTkinter).
