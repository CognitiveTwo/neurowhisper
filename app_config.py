"""Centralized configuration values for WhisperTyper/neurowhisper."""

# App identity and window defaults
APP_USER_MODEL_ID = "neuroflash.neurowhisper.1.0"
APP_TITLE = "neurowhisper"
APP_WINDOW_GEOMETRY = "480x830"  # fits all three accordions collapsed without scrolling
APP_WINDOW_MIN_WIDTH = 420
APP_WINDOW_MIN_HEIGHT = 420
CTK_APPEARANCE_MODE = "dark"
CTK_COLOR_THEME = "dark-blue"

# Files and directories
CONFIG_FILE = "whisper_config.json"
STATS_FILE = "whisper_stats.json"
AI_INSIGHTS_FILE = "ai_insights.json"
TRANSCRIPTIONS_DIR = "transcriptions"
ICON_ICO = "icon.ico"
ICON_PNG = "icon.png"
ICON_ICNS = "icon.icns"

# External URLs
AUTHOR_LINK_URL = "https://www.linkedin.com/in/drjonathanmall/"
CUDA_DLL_URL = "https://github.com/Purfview/whisper-standalone-win/releases/tag/libs"
OPENAI_KEY_URL = "https://platform.openai.com/api-keys"

# Core behavior defaults
TYPING_WPM = 40
IDLE_UNLOAD_CHECK_INTERVAL_SECONDS = 15

MODEL_MAP = {
    "Large v3 (Best quality, 10GB+ VRAM GPU)": "large-v3",
    "Medium (GPU or fast CPU, 4GB+ RAM)": "medium",
    "Small (CPU-friendly, 2GB+ RAM)": "small",
    "Base (Minimal resources, quick)": "base",
    "Parakeet TDT 0.6B v3 (Multi-lang, 2GB+ VRAM)": "nvidia/parakeet-tdt-0.6b-v3",
}

DEFAULT_CONFIG = {
    "model_key": "Small (CPU-friendly, 2GB+ RAM)",
    # "auto" for all three: a fresh install detects the hardware on first run
    # and writes the resolved values into the per-machine config overlay.
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
    "openai_api_key": "",
    "openai_transcription_model": "gpt-transcribe",
    "openai_edit_model": "gpt-4o-mini",
    "openai_edit_prompt": "",
    "openai_language": "auto",
    "idle_unload_minutes": 5,
    "worker_process": True,
    "insights_range": "30 days",
    # Opt-in only: when True (and an API key is set) the app sends transcript
    # text to OpenAI in the background to refresh the AI insight tabs.
    "ai_insights_auto": False,
    # False only on a genuinely fresh install: load_config() flips it to True
    # for any config file that already exists, so the welcome dialog never
    # appears for an existing user upgrading to this version.
    "welcome_shown": False,
}

AUTO_DEVICE_CONFIG = {
    "cuda_device": "cuda",
    "cuda_compute_type": "float16",
    "cpu_device": "cpu",
    "cpu_compute_type": "int8",
}

# Theme colors
THEME_COLORS = {
    "bg": "#0f1419",
    "fg": "#ffffff",
    "accent_bg": "#1a2129",
    "text_bg": "#0f1419",   # inner "well" inside a card (must differ from accent_bg)
    "teal": "#2dd4bf",
    "teal_dim": "#134e4a",
    "border": "#2a3441",
    "live": "#ef4444",
    "batch": "#3b82f6",
    "idle": "#374151",
    "online": "#a855f7",
    "transcribing": "#c9a962",
    "ready": "#7cb886",
    "text_muted": "#9ca3af",   # secondary label text
    "text_faint": "#6b7280",   # small print / captions
    "stopping_live": "#500000",
    "stopping_batch": "#002050",
}

# Platform abstraction defaults
PLATFORM_SHORTCUTS = {
    "mac_paste_modifier": "command",
    "other_paste_modifier": "ctrl",
    "paste_key": "v",
}

PLATFORM_SOUND = {
    "windows_start_freq": 800,
    "windows_stop_freq": 400,
    "windows_duration_ms": 150,
    "mac_start_sound": "Tink",
    "mac_stop_sound": "Pop",
    "mac_start_sound_file": "/System/Library/Sounds/Tink.aiff",
    "mac_stop_sound_file": "/System/Library/Sounds/Pop.aiff",
}

# Hotkey and queue timings
HOTKEY_RUNTIME = {
    "action_queue_maxsize": 32,
    "debounce_seconds": 0.2,
    "refresh_interval_seconds": 180,
    "watchdog_interval_ms": 5000,
    "focus_refresh_min_interval_seconds": 5,
    "focus_registration_age_seconds": 30,
    "shutdown_unregister_sleep_seconds": 0.1,
    "capture_finish_delay_ms": 10,
    "capture_preview_delay_ms": 16,
}

# Audio and transcription timing
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
    "dialog_geometry_cuda": "500x320",
    "ready_state_delay_ms": 2000,
}

# Legacy static hotkey labels used in button text paths
# Fallback hotkey labels used when the config has no value yet.
# Must stay in sync with DEFAULT_CONFIG's hotkey_batch / hotkey_live.
HOTKEY_LABELS = {
    "batch": "Ctrl+Alt+S",
    "live": "F9",
}

# CUDA detection keywords
CUDA_ERROR_KEYWORDS = [
    "cudnn",
    "cublas",
    "cuda",
    "nvrtc",
    "could not load library",
    "dll",
]

CUDA_DETECT_KEYWORDS = [
    "cuda",
    "cudnn",
    "cublas",
    "dll",
    "library",
]

CUDA_DETECT_KEYWORDS_BACKEND = [
    "cuda",
    "cudnn",
    "cublas",
    "dll",
    "library",
    "driver",
    "runtime",
]

# Backend defaults
BACKEND_FACTORY_DEFAULTS = {
    "cpu_fallback_backend": "faster-whisper",
    "cpu_fallback_device": "cpu",
    "cpu_fallback_reason": "Using CPU (universal fallback)",
}

BACKEND_DISPLAY_NAMES = {
    "faster-whisper": "Faster Whisper (CPU/CUDA)",
    "openvino": "OpenVINO (Intel GPU)",
    "openai": "OpenAI Cloud (Online)",
    "parakeet": "Parakeet TDT (NVIDIA NeMo)",
}

ONLINE_BACKEND_NAME = "openai"

BACKEND_OPTIONS = [
    "Auto",
    "Parakeet (NVIDIA NeMo)",
    "OpenVINO (Intel GPU)",
    "CUDA (NVIDIA GPU)",
    "CPU only",
]

BACKEND_DISPLAY_TO_INTERNAL = {
    "Auto": "auto",
    "Parakeet (NVIDIA NeMo)": "parakeet",
    "OpenVINO (Intel GPU)": "openvino",
    "CUDA (NVIDIA GPU)": "cuda",
    "CPU only": "faster-whisper",
}

# Keep behavior exactly as current GUI logic (Parakeet option does not map here yet).
BACKEND_DISPLAY_TO_BACKEND_DEVICE = {
    # "auto" (not "faster-whisper") so the dropdown still reads "Auto" after a
    # restart and detect_best_backend() stays in charge of the choice.
    "Auto": ("auto", "auto"),
    "OpenVINO (Intel GPU)": ("openvino", "auto"),
    "CUDA (NVIDIA GPU)": ("faster-whisper", "cuda"),
    "CPU only": ("faster-whisper", "cpu"),
}

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
    "Whisper-1 (Fast, reliable)": "whisper-1",
}

OPENAI_EDIT_MODELS = {
    "GPT-4o Mini (Fast, cheap)": "gpt-4o-mini",
    "GPT-4o (Best quality)": "gpt-4o",
    "GPT-4 Turbo": "gpt-4-turbo",
}

OPENAI_TRANSCRIPTION_MODEL_OPTIONS = list(OPENAI_TRANSCRIPTION_MODELS.values())
OPENAI_EDIT_MODEL_OPTIONS = list(OPENAI_EDIT_MODELS.values())

LIVE_PAUSE_RANGE = {
    "min": 0.5,
    "max": 3.0,
    "step": 0.1,
}

TRANSCRIPTION_HISTORY_DAYS = 3650

# OpenVINO model map and audio defaults
OPENVINO_MODEL_MAP = {
    "Large v3 (Best quality, 10GB+ VRAM GPU)": "OpenVINO/whisper-large-v3-fp16-ov",
    "Medium (GPU or fast CPU, 4GB+ RAM)": "OpenVINO/whisper-medium-fp16-ov",
    "Small (CPU-friendly, 2GB+ RAM)": "OpenVINO/whisper-small-fp16-ov",
    "Base (Minimal resources, quick)": "OpenVINO/whisper-base-fp16-ov",
}

# Parakeet model map and defaults
PARAKEET_MODEL_MAP = {
    "Parakeet TDT 0.6B v3 (Multi-lang, 2GB+ RAM)": "nvidia/parakeet-tdt-0.6b-v3",
    "Parakeet TDT 0.6B v2 (English, 2GB+ RAM)": "nvidia/parakeet-tdt-0.6b-v2",
}

PARAKEET_DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

# Speech insights peer-proxy profile and heuristic benchmark bands.
# These are non-clinical reference bands intended for trend framing only.
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

# --- Speech insights: rolling window, personal baseline, language lists ---

# Rolling window options for the non-AI insight tabs. Value is the number of
# trailing days; 0 means "today only" (calendar day) and None means "all time".
#
# Known deviation from the InsightsExpanded artboard, which shows three chips
# (Today / 7 d / All) with Today highlighted: "30 days" is kept and is the
# default because the personal baseline compares the current window against a
# trailing 90-day average, and a one-day window is too noisy to read against
# it. The artboard should be updated to four chips rather than the code cut.
INSIGHTS_RANGE_OPTIONS = {
    "Today": 0,
    "7 days": 7,
    "30 days": 30,
    "All": None,
}
INSIGHTS_RANGE_DEFAULT = "30 days"

# Personal baseline: metrics are compared against the user's own trailing
# average instead of a hand-set peer band. PEER_BENCHMARKS is only used as a
# fallback while fewer than PERSONAL_BASELINE_MIN_DAYS of data exist.
PERSONAL_BASELINE_DAYS = 90
PERSONAL_BASELINE_MIN_DAYS = 14
# Half-width of the "normal" band drawn around the personal baseline value.
PERSONAL_BASELINE_BAND_FRACTION = 0.25

# --- Language-aware filler / stopword lists ---

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
    "also/halt": r"\b(also|halt)\b",
    "quasi/sozusagen": r"\b(quasi|sozusagen)\b",
    "genau/ne/ja": r"\b(genau|ne|ja)\b",
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

# Short, high-frequency German function words used by the language heuristic.
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

# Minimum share of tokens that must be German function words for a
# umlaut-free entry to be classified as German.
GERMAN_DETECT_THRESHOLD = 0.08
