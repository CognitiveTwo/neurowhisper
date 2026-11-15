"""Centralized Linux configuration, model maps, theme palette, and typography."""

from __future__ import annotations

from typing import Optional, Tuple

from . import APP_ID, APP_NAME

# App identity and window defaults
APP_TITLE = APP_NAME
WM_CLASS = APP_NAME
APP_WINDOW_GEOMETRY = "480x830"
APP_WINDOW_MIN_WIDTH = 420
APP_WINDOW_MIN_HEIGHT = 420
CTK_APPEARANCE_MODE = "dark"
CTK_COLOR_THEME = "dark-blue"

# External URLs & Linux add-on instructions
AUTHOR_LINK_URL = "https://www.linkedin.com/in/drjonathanmall/"
OPENAI_KEY_URL = "https://platform.openai.com/api-keys"
CUDA_PIP_PACKAGES = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12")

# Linux typography (anti-aliased Xft / Fontconfig vector fonts — avoids X11 bitmap fallback)
UI_FONT_FAMILY = "DejaVu Sans"
MONO_FONT_FAMILY = "DejaVu Sans Mono"


def ui_font(size: int, weight: Optional[str] = None) -> Tuple:
    """Return a Tk font tuple using ``DejaVu Sans``."""
    if weight:
        return (UI_FONT_FAMILY, int(size), weight)
    return (UI_FONT_FAMILY, int(size))


def mono_font(size: int, weight: Optional[str] = None) -> Tuple:
    """Return a monospace Tk font tuple using ``DejaVu Sans Mono``."""
    if weight:
        return (MONO_FONT_FAMILY, int(size), weight)
    return (MONO_FONT_FAMILY, int(size))


# Core behavior defaults
TYPING_WPM = 40
IDLE_UNLOAD_CHECK_INTERVAL_SECONDS = 15
TRANSCRIPTION_HISTORY_DAYS = 3650

LIVE_PAUSE_RANGE = {
    "min": 0.5,
    "max": 3.0,
    "step": 0.1,
}

MODEL_MAP = {
    "Large v3 Turbo (Fast & accurate, 99 langs, ~800MB)": "large-v3-turbo",
    "Parakeet TDT 0.6B v3 (Ultra-fast ONNX, 25 langs, ~650MB)": "parakeet-tdt-0.6b-v3",
    "Distil Large v3.5 (Ultra-fast English, ~750MB)": "distil-large-v3.5",
    "Large v3 (Best quality, 10GB+ VRAM GPU)": "large-v3",
    "Medium (GPU or fast CPU, 4GB+ RAM)": "medium",
    "Small (CPU-friendly, 2GB+ RAM)": "small",
    "Base (Minimal resources, quick)": "base",
}

MODEL_MAP_REVERSE = {v: k for k, v in MODEL_MAP.items()}

LEGACY_MODEL_ALIASES = {
    "Parakeet TDT 0.6B v3 (Multi-lang, 2GB+ VRAM)": "parakeet-tdt-0.6b-v3",
    "Parakeet TDT 0.6B v3 (Multi-lang, 2GB+ RAM)": "parakeet-tdt-0.6b-v3",
    "Parakeet TDT 0.6B v2 (English, 2GB+ RAM)": "parakeet-tdt-0.6b-v2",
}

MODEL_DOWNLOAD_SIZES = {
    "base": "150 MB",
    "small": "460 MB",
    "parakeet-tdt-0.6b-v3": "650 MB",
    "parakeet-tdt-0.6b-v2": "650 MB",
    "distil-large-v3.5": "750 MB",
    "large-v3-turbo": "800 MB",
    "canary-1b-v2": "1.0 GB",
    "medium": "1.5 GB",
    "large-v3": "3.0 GB",
}

ONNX_ASR_MODEL_IDS = frozenset({
    "parakeet-tdt-0.6b-v3",
    "nemo-parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v2",
    "canary-1b-v2",
    "nemo-canary-1b-v2",
    "whisper-base",
})

OPENVINO_MODEL_MAP = {
    "Large v3 Turbo (Fast & accurate, 99 langs, ~800MB)": "OpenVINO/whisper-large-v3-fp16-ov",
    "Distil Large v3.5 (Ultra-fast English, ~750MB)": "OpenVINO/distil-whisper-large-v3-fp16-ov",
    "Large v3 (Best quality, 10GB+ VRAM GPU)": "OpenVINO/whisper-large-v3-fp16-ov",
    "Medium (GPU or fast CPU, 4GB+ RAM)": "OpenVINO/whisper-medium-fp16-ov",
    "Small (CPU-friendly, 2GB+ RAM)": "OpenVINO/whisper-small-fp16-ov",
    "Base (Minimal resources, quick)": "OpenVINO/whisper-base-fp16-ov",
}

DEFAULT_CONFIG = {
    "model_key": "Small (CPU-friendly, 2GB+ RAM)",
    "backend": "auto",
    "device": "auto",
    "compute_type": "auto",
    "silence_threshold": 0.015,
    "live_pause": 0.8,
    "hotkey_live": "f9",
    "hotkey_batch": "ctrl+alt+s",
    "input_device": None,
    "always_on_top": True,
    "transcription_mode": "local",
    "local_language": "auto",
    "openai_api_key": "",
    "openai_transcription_model": "gpt-transcribe",
    "openai_edit_model": "gpt-4o-mini",
    "openai_edit_prompt": "",
    "openai_language": "auto",
    "idle_unload_minutes": 5,
    "worker_process": True,
    "insights_range": "30 days",
    "ai_insights_auto": False,
    "welcome_shown": False,
}

# Machine-local keys kept in whisper_config.<HOST>.json rather than shared whisper_config.json
MACHINE_LOCAL_KEYS = frozenset({
    "model_key",
    "backend",
    "device",
    "compute_type",
    "input_device",
    "transcription_mode",
    "openai_api_key",
})

AUTO_DEVICE_CONFIG = {
    "cuda_device": "cuda",
    "cuda_compute_type": "float16",
    "cpu_device": "cpu",
    "cpu_compute_type": "int8",
}

THEME_COLORS = {
    "bg": "#0f1419",
    "fg": "#ffffff",
    "accent_bg": "#1a2129",
    "text_bg": "#0f1419",
    "teal": "#2dd4bf",
    "teal_dim": "#134e4a",
    "border": "#2a3441",
    "live": "#ef4444",
    "batch": "#3b82f6",
    "idle": "#374151",
    "online": "#a855f7",
    "transcribing": "#c9a962",
    "ready": "#7cb886",
    "text_muted": "#9ca3af",
    "text_faint": "#6b7280",
    "stopping_live": "#500000",
    "stopping_batch": "#002050",
}

HOTKEY_RUNTIME = {
    "action_queue_maxsize": 32,
    "debounce_seconds": 0.2,
    "refresh_interval_seconds": 180,
    "watchdog_interval_ms": 5000,
    "focus_refresh_min_interval_seconds": 5,
    "focus_registration_age_seconds": 30,
    "shutdown_unregister_sleep_seconds": 0.05,
    "capture_finish_delay_ms": 10,
    "capture_preview_delay_ms": 16,
}

AUDIO_RUNTIME = {
    "sample_rate": 16000,
    "channels": 1,
    "blocksize": 1600,
    "queue_timeout_seconds": 0.1,
    "chunk_duration_seconds": 0.1,
    "live_max_chunk_seconds": 15.0,
    "live_min_chunk_seconds": 0.5,
    "stop_delay_seconds": 0.6,
    "batch_min_segment_seconds": 10.0,
    "batch_long_segment_seconds": 20.0,
    "batch_max_segment_seconds": 60.0,
    "pause_threshold_short_seconds": 2.0,
    "pause_threshold_long_seconds": 1.5,
    "live_buffer_gain": 20.0,
    "mini_vu_multiplier": 3.0,
    "online_executor_workers": 2,
    "batch_executor_workers_cpu": 2,
    "batch_executor_workers_gpu": 1,
}

UI_RUNTIME = {
    "main_loop_interval_ms": 30,
    "initial_histogram_delay_ms": 200,
    "initial_gui_loop_delay_ms": 50,
    "mini_window_default_geometry": "260x48+50+50",
    "mini_window_right_margin_px": 20,
    "mini_window_bottom_margin_px": 60,
    "mini_window_width": 260,
    "mini_window_height": 48,
    "dialog_geometry_cuda": "540x360",
    "ready_state_delay_ms": 2000,
}

HOTKEY_LABELS = {
    "batch": "Ctrl+Alt+S",
    "live": "F9",
}

CUDA_ERROR_KEYWORDS = [
    "cudnn",
    "cublas",
    "cuda",
    "nvrtc",
    "could not load library",
    "cannot open shared object file",
    "libcublas",
    "libcudnn",
]

CUDA_DETECT_KEYWORDS = [
    "cuda",
    "cudnn",
    "cublas",
    "shared object",
    "library",
]

CUDA_DETECT_KEYWORDS_BACKEND = [
    "cuda",
    "cudnn",
    "cublas",
    "shared object",
    "library",
    "driver",
    "runtime",
]

BACKEND_FACTORY_DEFAULTS = {
    "cpu_fallback_backend": "faster-whisper",
    "cpu_fallback_device": "cpu",
    "cpu_fallback_reason": "Using CPU (universal fallback)",
}

BACKEND_DISPLAY_NAMES = {
    "faster-whisper": "Faster Whisper (CPU/CUDA)",
    "onnx-asr": "ONNX ASR (CPU/GPU)",
    "openvino": "OpenVINO (Intel GPU)",
    "openai": "OpenAI Cloud (Online)",
}

ONLINE_BACKEND_NAME = "openai"

# Linux acceleration options (omits legacy 8 GB NeMo toolkit; Parakeet runs 10x lighter on ONNX ASR)
BACKEND_OPTIONS = [
    "Auto",
    "ONNX ASR (CPU/GPU)",
    "CUDA (NVIDIA GPU)",
    "OpenVINO (Intel GPU)",
    "CPU only",
]

BACKEND_DISPLAY_TO_INTERNAL = {
    "Auto": "auto",
    "ONNX ASR (CPU/GPU)": "onnx-asr",
    "CUDA (NVIDIA GPU)": "cuda",
    "OpenVINO (Intel GPU)": "openvino",
    "CPU only": "faster-whisper",
}

BACKEND_DISPLAY_TO_BACKEND_DEVICE = {
    "Auto": ("auto", "auto"),
    "ONNX ASR (CPU/GPU)": ("onnx-asr", "auto"),
    "CUDA (NVIDIA GPU)": ("faster-whisper", "cuda"),
    "OpenVINO (Intel GPU)": ("openvino", "auto"),
    "CPU only": ("faster-whisper", "cpu"),
}


def resolve_backend_display(backend: str, device: str = "auto") -> str:
    """Map persisted ``(backend, device)`` config values back to a Linux ``BACKEND_OPTIONS`` label."""
    b = (backend or "auto").strip().lower()
    d = (device or "auto").strip().lower()
    if b == "auto":
        return "Auto"
    if b in ("onnx-asr", "parakeet"):
        return "ONNX ASR (CPU/GPU)"
    if b == "openvino":
        return "OpenVINO (Intel GPU)"
    if b == "cuda" or (b == "faster-whisper" and d == "cuda"):
        return "CUDA (NVIDIA GPU)"
    if b == "faster-whisper" and d == "auto":
        return "Auto"
    return "CPU only"


OPENAI_DEFAULT_EDIT_PROMPT = """You are a copy editor. Your job is to clean up transcribed speech.

Rules:
- Fix grammar, punctuation, and spelling errors
- Remove filler words (um, uh, like, you know)
- Remove duplicate phrases and repetitions (when someone says the same thing twice)
- Keep the original meaning and tone
- Format as clean, readable text
- Do NOT add any commentary or explanations
- Output ONLY the edited transcription

Here is the transcription to clean up:"""

OPENAI_DEFAULTS = {
    "transcription_model": "gpt-transcribe",
    "edit_model": "gpt-4o-mini",
    "language": "auto",
    "device": "cloud",
    "response_format_gpt": "text",
    "response_format_whisper": "verbose_json",
    "chat_max_tokens": 4000,
    "chat_temperature": 0.3,
}

OPENAI_TRANSCRIPTION_MODELS = {
    "GPT Transcribe (Newest, recommended)": "gpt-transcribe",
    "GPT-4o Transcribe": "gpt-4o-transcribe",
    "GPT-4o Mini Transcribe (Fast, low cost)": "gpt-4o-mini-transcribe",
    "Whisper-1 (Fast, reliable)": "whisper-1",
}

OPENAI_EDIT_MODELS = {
    "GPT-4o Mini (Fast, cheap)": "gpt-4o-mini",
    "GPT-4o (Best quality)": "gpt-4o",
    "GPT-4 Turbo": "gpt-4-turbo",
}

OPENAI_TRANSCRIPTION_MODEL_OPTIONS = list(OPENAI_TRANSCRIPTION_MODELS.values())
OPENAI_EDIT_MODEL_OPTIONS = list(OPENAI_EDIT_MODELS.values())

# Speech insights peer-proxy profile and heuristic benchmark bands.
PEER_PROFILE = {
    "label": "generic heuristic bands",
}

PEER_BENCHMARK_NOTE = "Heuristic proxy benchmark (trend context, not diagnostic norm)."

PEER_BENCHMARKS = {
    "pace_avg_words_entry": {"low": 16.0, "high": 55.0},
    "pace_stability": {"low": 55.0, "high": 100.0},
    "pause_avg_seconds": {"low": 1.2, "high": 4.8},
    "pause_long_share": {"low": 0.05, "high": 0.35},
    "clarity_fillers_per_100": {"low": 0.0, "high": 2.5},
    "clarity_repetition_per_1k": {"low": 0.0, "high": 8.0},
    "lexical_diversity_percent": {"low": 22.0, "high": 48.0},
    "lexical_long_word_percent": {"low": 12.0, "high": 36.0},
    "intervention_open_question_per_100": {"low": 3.0, "high": 18.0},
    "intervention_reflection_ratio": {"low": 0.4, "high": 2.5},
    "intervention_affirmation_per_100": {"low": 1.0, "high": 12.0},
    "intervention_summary_per_100": {"low": 0.3, "high": 6.0},
}

INSIGHTS_RANGE_OPTIONS = {
    "Today": 0,
    "7 days": 7,
    "30 days": 30,
    "All": None,
}
INSIGHTS_RANGE_DEFAULT = "30 days"

PERSONAL_BASELINE_DAYS = 90
PERSONAL_BASELINE_MIN_DAYS = 14
PERSONAL_BASELINE_BAND_FRACTION = 0.25

ENGLISH_FILLER_PATTERNS = {
    "um/uh": r"\b(um|uh)\b",
    "like": r"\blike\b",
    "you know": r"\byou know\b",
    "kind of": r"\bkind of\b",
    "sort of": r"\bsort of\b",
}

GERMAN_FILLER_PATTERNS = {
    "äh/ähm": r"\b(äh|ähm|öhm|öh)\b",
    "hm": r"\b(hm|hmm|mhm)\b",
    "halt/irgendwie": r"\b(halt|irgendwie)\b",
    "quasi/sozusagen": r"\b(quasi|sozusagen)\b",
    "sag ich mal": r"\b(sag ich mal|sage ich mal)\b",
}

ENGLISH_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "just", "your", "you",
    "for", "are", "was", "were", "but", "not", "can", "could", "would", "should",
    "about", "into", "what", "when", "where", "which", "then", "than", "them",
    "they", "there", "their", "been", "being", "because", "also", "it's", "its",
    "i", "me", "my",
}

GERMAN_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "einer", "eines", "und", "oder", "aber", "auch", "noch", "schon", "nur",
    "nicht", "kein", "keine", "ist", "sind", "war", "waren", "bin", "bist",
    "sein", "seine", "hat", "habe", "haben", "hatte", "hatten", "wird",
    "werden", "wurde", "wurden", "kann", "können", "muss", "müssen", "soll",
    "sollen", "will", "wollen", "für", "mit", "von", "vom", "aus", "auf", "an",
    "am", "im", "in", "ins", "zu", "zum", "zur", "bei", "nach", "über", "unter",
    "vor", "durch", "gegen", "ohne", "um", "als", "wie", "wenn", "weil",
    "dass", "was", "wer", "wo", "welche", "welcher", "welches", "man", "ich",
    "mir", "mich", "mein", "meine", "du", "dir", "dich", "er", "sie", "es",
    "wir", "uns", "unser", "ihr", "ihnen", "sich", "hier", "dann", "doch",
    "mal", "sehr", "mehr", "so", "es", "dieser", "diese", "dieses", "einfach",
}

GERMAN_FUNCTION_WORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "und", "oder", "aber", "auch", "noch", "schon", "nur", "nicht", "ist",
    "sind", "war", "haben", "hat", "wird", "werden", "kann", "muss", "soll",
    "mit", "von", "aus", "auf", "im", "zum", "zur", "bei", "nach", "vor",
    "durch", "gegen", "ohne", "als", "wie", "wenn", "weil", "dass", "was",
    "wer", "wo", "man", "ich", "mir", "mich", "wir", "uns", "sich", "dann",
    "doch", "mal", "sehr", "mehr", "für", "über", "unter", "nicht", "ja",
    "also", "hier", "jetzt", "immer", "wieder", "machen", "gemacht", "gut",
}

GERMAN_DETECT_THRESHOLD = 0.08
