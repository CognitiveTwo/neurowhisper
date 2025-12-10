# neurowhisper

A fast, local voice transcription app powered by Whisper AI. Type with your voice using hotkeys - works offline with GPU acceleration or CPU fallback.

![neurowhisper](icon.png)

## Features

- 🎤 **Live Mode** - Real-time transcription, types as you speak
- 📝 **Batch Mode** - Record longer segments, transcribe all at once
- ☁️ **Online Mode** - Optional OpenAI API integration for cloud transcription
- ⚡ **Auto GPU/CPU Detection** - Automatically uses CUDA if available, falls back to CPU
- 🎯 **Customizable Hotkeys** - Set any key combination (Ctrl+Alt+S, F8, etc.)
- 📊 **Statistics Dashboard** - Track words, speaking time, and productivity
- 🌙 **Dark Theme** - Easy on the eyes

## Quick Start

1. **Download** the latest release or clone this repo
2. **Run** `install.bat` to set up the Python environment
3. **Launch** with `run.bat`
4. **Press F8** for Batch mode or **F9** for Live mode

## Requirements

- Windows 10/11
- Python 3.10+ (installed automatically by install.bat)
- ~2GB RAM minimum (4GB+ recommended for larger models)
- Optional: NVIDIA GPU with CUDA for faster transcription

## Models

| Model | Size | RAM | Best For |
|-------|------|-----|----------|
| Base | ~150MB | 2GB | Quick notes, low resources |
| Small | ~500MB | 2GB | General use, CPU-friendly |
| Medium | ~1.5GB | 4GB | Better accuracy |
| Large-v3 | ~3GB | 8GB | Best accuracy (GPU recommended) |

## GPU Acceleration

For faster transcription with NVIDIA GPUs:
1. Download CUDA DLLs from [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs)
2. Extract DLLs to the app folder
3. Restart the app

Without CUDA DLLs, the app automatically uses CPU mode.

## Configuration

Settings are saved in `whisper_config.json`:
- Model selection
- Hotkeys (customizable in the UI)
- Input device
- Pause threshold for live mode

## License

MIT License - feel free to use and modify!

## Credits

- Built with [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- Created by DR.M
