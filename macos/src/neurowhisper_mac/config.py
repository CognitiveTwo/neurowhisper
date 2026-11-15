"""Centralized configuration, model maps, Aqua typography, and Speech Insights constants for macOS."""

from __future__ import annotations

import sys
from typing import Optional, Tuple

from . import APP_BUNDLE_ID, APP_NAME

# App identity and window defaults
APP_TITLE = APP_NAME
APP_WINDOW_GEOMETRY = "480x840"
APP_WINDOW_MIN_WIDTH = 420
APP_WINDOW_MIN_HEIGHT = 420
CTK_APPEARANCE_MODE = "dark"
CTK_COLOR_THEME = "dark-blue"

# Standard filenames inside data_dir / resource_dir
CONFIG_FILE = "whisper_config.json"
STATS_FILE = "whisper_stats.json"
AI_INSIGHTS_FILE = "ai_insights.json"
TRANSCRIPTIONS_DIR = "transcriptions"
ICON_PNG = "icon.png"
ICON_ICNS = "icon.icns"

# External URLs
AUTHOR_LINK_URL = "https://www.linkedin.com/in/drjonathanmall/"
OPENAI_KEY_URL = "https://platform.openai.com/api-keys"

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

# Model IDs natively served by OnnxAsrBackend when backend is "auto"
ONNX_ASR_MODEL_IDS = frozenset({
    "parakeet-tdt-0.6b-v3",
    "nemo-parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v2",
    "canary-1b-v2",
    "nemo-canary-1b-v2",
    "whisper-base",
})

# Apple Silicon MLX Whisper HuggingFace repositories
MLX_WHISPER_MODEL_MAP = {
    "Large v3 Turbo (Fast & accurate, 99 langs, ~800MB)": "mlx-community/whisper-large-v3-turbo",
    "Distil Large v3.5 (Ultra-fast English, ~750MB)": "mlx-community/distil-whisper-large-v3.5",
    "Large v3 (Best quality, 10GB+ VRAM GPU)": "mlx-community/whisper-large-v3-mlx",
    "Medium (GPU or fast CPU, 4GB+ RAM)": "mlx-community/whisper-medium-mlx",
    "Small (CPU-friendly, 2GB+ RAM)": "mlx-community/whisper-small-mlx",
    "Base (Minimal resources, quick)": "mlx-community/whisper-base-mlx",
}

MLX_WHISPER_SHORT_ID_MAP = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "distil-large-v3.5": "mlx-community/distil-whisper-large-v3.5",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
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
    "worker_process": False,  # No CUDA on macOS; inference runs cleanly in-process
    "insights_range": "30 days",
    "ai_insights_auto": False,
    "welcome_shown": False,
}

# Dark theme palette
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

PLATFORM_SOUND = {
    "mac_start_sound": "Tink",
    "mac_stop_sound": "Pop",
    "mac_start_sound_file": "/System/Library/Sounds/Tink.aiff",
    "mac_stop_sound_file": "/System/Library/Sounds/Pop.aiff",
    "synth_start_freq": 800,
    "synth_stop_freq": 440,
    "synth_duration_s": 0.12,
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
    "ready_state_delay_ms": 2000,
}

HOTKEY_LABELS = {
    "batch": "Ctrl+Alt+S",
    "live": "F9",
}

# macOS Backend & Acceleration configuration (zero Windows/NVIDIA/Intel clutter)
ONLINE_BACKEND_NAME = "openai"

BACKEND_FACTORY_DEFAULTS = {
    "cpu_fallback_backend": "faster-whisper",
    "cpu_fallback_device": "cpu",
    "cpu_fallback_reason": "CPU INT8 (CTranslate2)",
}

BACKEND_DISPLAY_NAMES = {
    "mlx-whisper": "Apple Metal (MLX)",
    "onnx-asr": "ONNX ASR (CoreML/CPU)",
    "faster-whisper": "Faster Whisper (CPU INT8)",
    "openai": "OpenAI Cloud (Online)",
}

BACKEND_OPTIONS = [
    "Auto",
    "Apple Metal (MLX)",
    "ONNX ASR (CoreML/CPU)",
    "CPU only",
]

BACKEND_DISPLAY_TO_INTERNAL = {
    "Auto": "auto",
    "Apple Metal (MLX)": "mlx-whisper",
    "ONNX ASR (CoreML/CPU)": "onnx-asr",
    "CPU only": "faster-whisper",
}

BACKEND_DISPLAY_TO_BACKEND_DEVICE = {
    "Auto": ("auto", "auto"),
    "Apple Metal (MLX)": ("mlx-whisper", "mps"),
    "ONNX ASR (CoreML/CPU)": ("onnx-asr", "auto"),
    "CPU only": ("faster-whisper", "cpu"),
}


def resolve_backend_display(backend: str, device: str = "auto") -> str:
    """Map persisted ``(backend, device)`` config values back to a ``BACKEND_OPTIONS`` label."""
    b = (backend or "auto").strip().lower()
    d = (device or "auto").strip().lower()
    if b == "auto":
        return "Auto"
    if b in ("mlx-whisper", "mlx", "metal"):
        return "Apple Metal (MLX)"
    if b == "onnx-asr":
        return "ONNX ASR (CoreML/CPU)"
    if b == "faster-whisper" and d == "auto":
        return "Auto"
    return "CPU only"


# Typography helpers (+1pt on macOS Aqua 72-DPI to match Windows 96-DPI legibility)
_CACHED_MAC_UI_FAMILY: Optional[str] = None
_CACHED_MAC_MONO_FAMILY: Optional[str] = None


def _resolve_mac_ui_family() -> str:
    global _CACHED_MAC_UI_FAMILY
    if _CACHED_MAC_UI_FAMILY is not None:
        return _CACHED_MAC_UI_FAMILY
    try:
        import tkinter as tk
        import tkinter.font as tkfont

        if getattr(tk, "_default_root", None) is not None:
            fams = set(tkfont.families())
            for candidate in ("SF Pro Text", ".AppleSystemUIFont", "Helvetica Neue", "Helvetica"):
                if candidate in fams:
                    _CACHED_MAC_UI_FAMILY = candidate
                    return candidate
    except Exception:
        pass
    return "SF Pro Text"


def _resolve_mac_mono_family() -> str:
    global _CACHED_MAC_MONO_FAMILY
    if _CACHED_MAC_MONO_FAMILY is not None:
        return _CACHED_MAC_MONO_FAMILY
    try:
        import tkinter as tk
        import tkinter.font as tkfont

        if getattr(tk, "_default_root", None) is not None:
            fams = set(tkfont.families())
            for candidate in ("Menlo", "SF Mono", "Monaco", "Courier"):
                if candidate in fams:
                    _CACHED_MAC_MONO_FAMILY = candidate
                    return candidate
    except Exception:
        pass
    return "Menlo"


def scaled_pt(size: int) -> int:
    """Adjust point size by +1 on Darwin to compensate for Aqua 72-DPI metrics."""
    base = max(6, int(size))
    return base + 1 if sys.platform == "darwin" else base


def ui_font(size: int, weight: Optional[str] = None) -> Tuple:
    """Return a Tk font tuple using SF Pro Text (with fallback) and +1pt Darwin scaling."""
    pt = scaled_pt(size)
    family = _resolve_mac_ui_family() if sys.platform == "darwin" else "Helvetica"
    if weight:
        return (family, pt, weight)
    return (family, pt)


def mono_font(size: int, weight: Optional[str] = None) -> Tuple:
    """Return a Tk monospace font tuple using Menlo on macOS and +1pt Darwin scaling."""
    pt = scaled_pt(size)
    family = _resolve_mac_mono_family() if sys.platform == "darwin" else "Monospace"
    if weight:
        return (family, pt, weight)
    return (family, pt)


# OpenAI Cloud defaults and models
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

# Speech insights peer-proxy profile and heuristic benchmark bands
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
    "mal", "sehr", "mehr", "so", "dieser", "diese", "dieses", "einfach",
}

GERMAN_FUNCTION_WORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "und", "oder", "aber", "auch", "noch", "schon", "nur", "nicht", "ist",
    "sind", "war", "haben", "hat", "wird", "werden", "kann", "muss", "soll",
    "mit", "von", "aus", "auf", "im", "zum", "zur", "bei", "nach", "vor",
    "durch", "gegen", "ohne", "als", "wie", "wenn", "weil", "dass", "was",
    "wer", "wo", "man", "ich", "mir", "mich", "wir", "uns", "sich", "dann",
    "doch", "mal", "sehr", "mehr", "für", "über", "unter", "ja",
    "also", "hier", "jetzt", "immer", "wieder", "machen", "gemacht", "gut",
}

GERMAN_DETECT_THRESHOLD = 0.08

__all__ = [
    "APP_BUNDLE_ID",
    "APP_TITLE",
    "APP_WINDOW_GEOMETRY",
    "APP_WINDOW_MIN_WIDTH",
    "APP_WINDOW_MIN_HEIGHT",
    "CTK_APPEARANCE_MODE",
    "CTK_COLOR_THEME",
    "CONFIG_FILE",
    "STATS_FILE",
    "AI_INSIGHTS_FILE",
    "TRANSCRIPTIONS_DIR",
    "ICON_PNG",
    "ICON_ICNS",
    "AUTHOR_LINK_URL",
    "OPENAI_KEY_URL",
    "TYPING_WPM",
    "IDLE_UNLOAD_CHECK_INTERVAL_SECONDS",
    "TRANSCRIPTION_HISTORY_DAYS",
    "LIVE_PAUSE_RANGE",
    "MODEL_MAP",
    "MODEL_MAP_REVERSE",
    "LEGACY_MODEL_ALIASES",
    "MODEL_DOWNLOAD_SIZES",
    "ONNX_ASR_MODEL_IDS",
    "MLX_WHISPER_MODEL_MAP",
    "MLX_WHISPER_SHORT_ID_MAP",
    "DEFAULT_CONFIG",
    "THEME_COLORS",
    "PLATFORM_SOUND",
    "HOTKEY_RUNTIME",
    "AUDIO_RUNTIME",
    "UI_RUNTIME",
    "HOTKEY_LABELS",
    "ONLINE_BACKEND_NAME",
    "BACKEND_FACTORY_DEFAULTS",
    "BACKEND_DISPLAY_NAMES",
    "BACKEND_OPTIONS",
    "BACKEND_DISPLAY_TO_INTERNAL",
    "BACKEND_DISPLAY_TO_BACKEND_DEVICE",
    "resolve_backend_display",
    "scaled_pt",
    "ui_font",
    "mono_font",
    "OPENAI_DEFAULT_EDIT_PROMPT",
    "OPENAI_DEFAULTS",
    "OPENAI_TRANSCRIPTION_MODELS",
    "OPENAI_EDIT_MODELS",
    "OPENAI_TRANSCRIPTION_MODEL_OPTIONS",
    "OPENAI_EDIT_MODEL_OPTIONS",
    "PEER_PROFILE",
    "PEER_BENCHMARK_NOTE",
    "PEER_BENCHMARKS",
    "INSIGHTS_RANGE_OPTIONS",
    "INSIGHTS_RANGE_DEFAULT",
    "PERSONAL_BASELINE_DAYS",
    "PERSONAL_BASELINE_MIN_DAYS",
    "PERSONAL_BASELINE_BAND_FRACTION",
    "ENGLISH_FILLER_PATTERNS",
    "GERMAN_FILLER_PATTERNS",
    "ENGLISH_STOPWORDS",
    "GERMAN_STOPWORDS",
    "GERMAN_FUNCTION_WORDS",
    "GERMAN_DETECT_THRESHOLD",
]
