# Platform detection and cross-platform compatibility
import sys
import os

from app_config import (
    APP_USER_MODEL_ID,
    APP_TITLE,
    APP_WINDOW_GEOMETRY,
    APP_WINDOW_MIN_WIDTH,
    APP_WINDOW_MIN_HEIGHT,
    AUTO_DEVICE_CONFIG,
    AUDIO_RUNTIME,
    AUTHOR_LINK_URL,
    BACKEND_DISPLAY_TO_BACKEND_DEVICE,
    BACKEND_DISPLAY_TO_INTERNAL,
    BACKEND_OPTIONS,
    CONFIG_FILE,
    CUDA_DETECT_KEYWORDS,
    CUDA_ERROR_KEYWORDS,
    CUDA_DLL_URL,
    CTK_APPEARANCE_MODE,
    CTK_COLOR_THEME,
    DEFAULT_CONFIG,
    HOTKEY_LABELS,
    HOTKEY_RUNTIME,
    ICON_ICO,
    ICON_PNG,
    IDLE_UNLOAD_CHECK_INTERVAL_SECONDS,
    LIVE_PAUSE_RANGE,
    ONLINE_BACKEND_NAME,
    MODEL_MAP,
    OPENAI_EDIT_MODEL_OPTIONS,
    OPENAI_KEY_URL,
    OPENAI_TRANSCRIPTION_MODEL_OPTIONS,
    PEER_BENCHMARK_NOTE,
    PEER_BENCHMARKS,
    PEER_PROFILE,
    INSIGHTS_RANGE_OPTIONS,
    INSIGHTS_RANGE_DEFAULT,
    PERSONAL_BASELINE_DAYS,
    PERSONAL_BASELINE_MIN_DAYS,
    PERSONAL_BASELINE_BAND_FRACTION,
    ENGLISH_FILLER_PATTERNS,
    GERMAN_FILLER_PATTERNS,
    ENGLISH_STOPWORDS,
    GERMAN_STOPWORDS,
    GERMAN_FUNCTION_WORDS,
    GERMAN_DETECT_THRESHOLD,
    STATS_FILE,
    AI_INSIGHTS_FILE,
    THEME_COLORS,
    TRANSCRIPTION_HISTORY_DAYS,
    TRANSCRIPTIONS_DIR,
    TYPING_WPM,
    UI_RUNTIME,
)

# Resolve app directory from script location (works from any CWD / Dropbox)
app_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(app_dir)  # Anchor CWD so relative paths in config constants resolve correctly

# --- Application logging -------------------------------------------------
# Under pythonw.exe there is no console: print() is a silent no-op and an
# unhandled exception vanishes without a window or a message. Everything the
# app wants to say therefore goes to logs/app.log, and a fatal startup error
# additionally lands in logs/crash.log plus a message box (see __main__).
LOG_DIR = os.path.join(app_dir, "logs")

# Set NEUROWHISPER_NO_DIALOGS=1 to suppress modal dialogs (CI / smoke tests).
NO_DIALOGS = os.environ.get("NEUROWHISPER_NO_DIALOGS", "").strip().lower() not in ("", "0", "false", "no")


def _init_app_logger():
    import logging
    from logging.handlers import RotatingFileHandler

    # The handler goes on the ROOT logger, not on a private "neurowhisper"
    # one: platform_utils and every backend log through
    # logging.getLogger(__name__), and those records only reach a file if the
    # root logger has the handler. A private logger with propagate=False
    # captured this module's records and dropped everyone else's.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if any(getattr(h, "_neurowhisper_handler", False) for h in logger.handlers):
        return logging.getLogger("neurowhisper")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(LOG_DIR, "app.log"),
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        handler._neurowhisper_handler = True
        logger.addHandler(handler)
    except Exception:
        logger.addHandler(logging.NullHandler())
    # Mirror to stderr only when there actually is one (never under pythonw).
    if getattr(sys, "stderr", None) is not None:
        try:
            # Log lines contain arrows and warning emoji; a legacy cp1252
            # console would raise UnicodeEncodeError on every one of them.
            try:
                sys.stderr.reconfigure(errors="backslashreplace")
            except Exception:
                pass
            stream = logging.StreamHandler(sys.stderr)
            stream.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            stream._neurowhisper_handler = True
            logger.addHandler(stream)
        except Exception:
            pass
    # Chatty third-party loggers would otherwise flood logs/app.log.
    for noisy in ("urllib3", "httpx", "httpcore", "huggingface_hub", "hf_xet", "filelock",
                  "matplotlib", "PIL", "openai", "asyncio", "numba", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("neurowhisper")


log = _init_app_logger()


def write_crash_log(text):
    """Best-effort append of a fatal error to logs/crash.log. Returns its path."""
    path = os.path.join(LOG_DIR, "crash.log")
    try:
        import time as _time

        os.makedirs(LOG_DIR, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n===== " + _time.strftime("%Y-%m-%d %H:%M:%S") + " =====\n")
            f.write(text)
            f.write("\n")
    except Exception:
        pass
    return path


def show_fatal_message(title, text):
    """Show a native error box. Works before (and after) Tk exists, and is a
    no-op when NEUROWHISPER_NO_DIALOGS is set."""
    if NO_DIALOGS:
        return
    if sys.platform == "win32":
        try:
            import ctypes as _ct

            _ct.windll.user32.MessageBoxW(0, str(text), str(title), 0x10)
            return
        except Exception:
            pass
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb

        _r = _tk.Tk()
        _r.withdraw()
        _mb.showerror(title, text)
        _r.destroy()
        return
    except Exception:
        pass
    if getattr(sys, "stderr", None) is not None:
        try:
            sys.stderr.write("%s: %s\n" % (title, text))
        except Exception:
            pass


def _excepthook(exc_type, exc_value, exc_tb):
    import traceback

    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    log.error("Unhandled exception:\n%s", text)
    write_crash_log(text)


sys.excepthook = _excepthook

# Add app directory to PATH so Windows can find local CUDA DLLs
# This must be done BEFORE importing ctranslate2/faster_whisper
if sys.platform == 'win32':
    os.environ['PATH'] = app_dir + os.pathsep + os.environ.get('PATH', '')

# Set Windows App User Model ID for proper taskbar icon (Windows only)
if sys.platform == 'win32':
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import customtkinter as ctk
import sounddevice as sd
import numpy as np
class _LazyWav:
    """Defers `import scipy.io.wavfile` (~1.2s) until the first .write/.read."""
    def __getattr__(self, name):
        import scipy.io.wavfile as _wav
        return getattr(_wav, name)
wav = _LazyWav()

# Platform abstraction layer for keyboard/sound/clipboard
try:
    from platform_utils import (
        keyboard_module as keyboard,
        play_system_sound,
        play_feedback_sound_async,
        paste_from_clipboard,
        type_text,
        get_paste_shortcut,
        IS_MAC,
        IS_WINDOWS,
        PLATFORM_NAME,
        KEYBOARD_ERROR,
    )
    PLATFORM_UTILS_AVAILABLE = True
except ImportError:
    # Fallback to direct imports if platform_utils not available
    import keyboard
    IS_MAC = False
    IS_WINDOWS = True
    PLATFORM_NAME = 'windows'
    KEYBOARD_ERROR = None
    PLATFORM_UTILS_AVAILABLE = False
    try:
        import winsound
    except ImportError:
        winsound = None

# faster_whisper is imported lazily inside load_model (legacy path) - saves ~4s cold start.

# Backend abstraction layer
try:
    from backends import create_backend, get_available_backends, detect_best_backend
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import json
import time
import shutil
import datetime
import webbrowser
import re
from collections import Counter, deque

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

# --- CONFIGURATION ---
MODEL_MAP_REVERSE = {v: k for k, v in MODEL_MAP.items()}

class _HotkeyFileLogger:
    """Rotating file logger for hotkey diagnostics."""

    def __init__(self, log_dir, max_bytes=512_000, backup_count=2):
        import logging
        from logging.handlers import RotatingFileHandler
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "hotkey.log")
        self._logger = logging.getLogger("whispertyper.hotkey")
        self._logger.setLevel(logging.DEBUG)
        if not self._logger.handlers:
            handler = RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            self._logger.addHandler(handler)

    def __call__(self, msg):
        self._logger.info(msg)


# Win32 helpers for detecting dead keyboard hooks
if sys.platform == "win32":
    import ctypes as _ctypes

    class _LASTINPUTINFO(_ctypes.Structure):
        _fields_ = [("cbSize", _ctypes.c_uint), ("dwTime", _ctypes.c_uint)]

    _ctypes.windll.kernel32.GetTickCount64.restype = _ctypes.c_ulonglong

    def _get_system_idle_ms():
        """Milliseconds since the last physical input event (keyboard or mouse)."""
        lii = _LASTINPUTINFO()
        lii.cbSize = _ctypes.sizeof(_LASTINPUTINFO)
        if _ctypes.windll.user32.GetLastInputInfo(_ctypes.byref(lii)):
            # Use GetTickCount64 to avoid 49-day DWORD wrap-around.
            # dwTime is still 32-bit, but the difference is correct as long
            # as the idle period is < 49 days (always true in practice).
            tick = _ctypes.windll.kernel32.GetTickCount64()
            return tick - (lii.dwTime & 0xFFFFFFFF)
        # On failure, report a large idle time so we don't false-positive.
        return 999_999
else:
    def _get_system_idle_ms():
        return 0


class HotkeyManager:
    """Robust hotkey manager that bypasses keyboard.add_hotkey() entirely.

    Previous approach used keyboard.add_hotkey() which relies on the library's
    internal nonblocking_hotkeys dict keyed by scan-code tuples.  That state
    gets corrupted after multiple add/remove cycles or when the processing
    thread is duplicated (old thread steals events from shared Queue).

    New approach: a single keyboard.hook() callback receives ALL key events
    and matches by key NAME (not scan-code tuple).  This is immune to the
    library's internal state corruption because:
    1. We never call add_hotkey/remove_hotkey (no scan-code dict mutation)
    2. The handlers list used by hook() is a simple append-based list
    3. Key-name matching is done entirely in our code, not the library's

    Dead-hook detection (via GetLastInputInfo) is kept as a safety net
    for the case where Windows silently removes the WH_KEYBOARD_LL hook.
    """

    # GetLastInputInfo() counts mouse movement as input, so "system active +
    # hook silent" is the *normal* state when the user is using the mouse but
    # not typing. The old 30s threshold produced ~800 force-restarts per
    # session (every restart tears down WH_KEYBOARD_LL — every keystroke on
    # the machine flows through that hook, so churn = system-wide input lag).
    # Bump the threshold so the silence heuristic only fires for genuinely
    # dead hooks; _listener_thread_health() is the reliable signal.
    HOOK_DEAD_THRESHOLD_S = 300
    HOOK_DEAD_CONFIRM_CYCLES = 4
    HOOK_GRACE_PERIOD_S = 10
    DEBOUNCE_S = HOTKEY_RUNTIME["debounce_seconds"]

    def __init__(self, keyboard_module, logger):
        self._keyboard = keyboard_module
        self._ui_log = logger
        self._lock = threading.Lock()

        # Registered hotkeys: name -> {"keys": frozenset of lowercase key names,
        #                               "callback": callable,
        #                               "hotkey_str": original string}
        self._bindings = {}
        self._last_registration = 0.0

        # File logger for persistent diagnostics
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        self._flog = _HotkeyFileLogger(log_dir)

        # Heartbeat: updated on every key event received by our global hook
        self._last_hook_event = time.monotonic()
        # Per-hotkey debounce timestamps
        self._last_trigger = {}  # name -> monotonic timestamp

        # The single hook that does everything: heartbeat + hotkey matching
        self._hook_ref = None
        # Set of currently held key names (lowercase)
        self._pressed_keys = set()
        self._install_hook()

        # Dead-hook detection
        self._consecutive_dead_checks = 0
        self._total_listener_restarts = 0
        # Text of the last hook-install failure, surfaced by _register_hotkeys.
        self._install_error = None

    def _log(self, msg):
        self._ui_log(msg)
        self._flog(msg)

    @property
    def last_registration_time(self):
        return self._last_registration

    @property
    def total_listener_restarts(self):
        return self._total_listener_restarts

    # ------------------------------------------------------------------
    # Parse hotkey strings into sets of lowercase key names
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_hotkey(hotkey_str):
        """Parse 'ctrl+alt+s' or 'F9' into a frozenset of lowercase names.

        Splits on '+' and strips each token individually so that multi-word
        key names like 'right ctrl' are preserved.
        """
        parts = [p.strip().lower() for p in hotkey_str.split("+")]
        return frozenset(p for p in parts if p)

    # ------------------------------------------------------------------
    # The single global hook – replaces both heartbeat and add_hotkey
    # ------------------------------------------------------------------
    def _install_hook(self):
        self._remove_hook()
        try:
            def _on_key(event):
                now = time.monotonic()
                self._last_hook_event = now

                name = getattr(event, "name", None)
                if not name:
                    return
                name_lower = name.lower()
                event_type = getattr(event, "event_type", None)

                to_fire = []
                if event_type == "down":
                    with self._lock:
                        self._pressed_keys.add(name_lower)
                        for hk_name, binding in self._bindings.items():
                            if binding["keys"].issubset(self._pressed_keys):
                                last = self._last_trigger.get(hk_name, 0.0)
                                if now - last >= self.DEBOUNCE_S:
                                    self._last_trigger[hk_name] = now
                                    to_fire.append(
                                        (hk_name, binding["callback"])
                                    )
                    # Fire callbacks outside the lock
                    for hk_name, cb in to_fire:
                        try:
                            cb()
                        except Exception as exc:
                            self._flog(
                                f"Hotkey callback error "
                                f"({hk_name}): {exc}"
                            )
                elif event_type == "up":
                    with self._lock:
                        self._pressed_keys.discard(name_lower)

            self._hook_ref = self._keyboard.hook(_on_key, suppress=False)
            self._flog("Hook-based hotkey handler installed")
        except Exception as e:
            # Through _log, not _flog: a failed hook means no hotkeys at all,
            # which the user has to see in the System Log, not only in
            # logs/hotkey.log.
            self._install_error = str(e)
            self._log(f"Hook install failed: {e}")
            log.error("Keyboard hook install failed: %s", e)

    def _remove_hook(self):
        if self._hook_ref is not None:
            try:
                self._keyboard.unhook(self._hook_ref)
            except Exception:
                pass
            self._hook_ref = None

    # ------------------------------------------------------------------
    # Hook-alive detection via GetLastInputInfo comparison
    # ------------------------------------------------------------------
    def _is_hook_alive(self):
        if sys.platform != "win32":
            return True, "non-windows platform"

        now = time.monotonic()
        if now - self._last_registration < self.HOOK_GRACE_PERIOD_S:
            return True, "grace period"

        time_since_hook = now - self._last_hook_event
        system_idle_s = _get_system_idle_ms() / 1000.0

        if system_idle_s > self.HOOK_DEAD_THRESHOLD_S:
            return True, "system idle"

        if time_since_hook > self.HOOK_DEAD_THRESHOLD_S:
            return False, (
                f"hook silent {time_since_hook:.0f}s "
                f"but system idle only {system_idle_s:.1f}s"
            )
        return True, "ok"

    # ------------------------------------------------------------------
    # Listener thread health
    # ------------------------------------------------------------------
    def _listener_thread_health(self):
        listener = getattr(self._keyboard, "_listener", None)
        if listener is None:
            return True, "listener state unavailable"
        if getattr(listener, "listening", False) is False:
            return False, "keyboard listener not listening"
        lt = getattr(listener, "listening_thread", None)
        if lt is not None and hasattr(lt, "is_alive") and not lt.is_alive():
            return False, "keyboard listening thread stopped"
        pt = getattr(listener, "processing_thread", None)
        if pt is not None and hasattr(pt, "is_alive") and not pt.is_alive():
            return False, "keyboard processing thread stopped"
        return True, "ok"

    # ------------------------------------------------------------------
    # Force-restart: reinstall the Windows hook from scratch
    # ------------------------------------------------------------------
    def _force_restart_listener(self):
        if sys.platform != "win32":
            return

        self._flog(">>> FORCE RESTART of keyboard listener <<<")
        listener = getattr(self._keyboard, "_listener", None)
        if listener is None:
            self._flog("No _listener attribute found – cannot restart")
            return

        try:
            # 1. Remove our hook
            self._remove_hook()

            # 2. Reset internal listener so start_if_necessary() creates
            #    new threads with a fresh Windows hook.
            with listener.lock:
                listener.listening = False

            # 3. Clear key state before new events can arrive
            with self._lock:
                self._pressed_keys.clear()

            # 4. Reinstall our hook (triggers start_if_necessary)
            self._install_hook()

            self._total_listener_restarts += 1
            self._consecutive_dead_checks = 0
            self._last_hook_event = time.monotonic()
            self._flog(
                f"Listener restarted (total: {self._total_listener_restarts})"
            )
        except Exception as e:
            self._flog(f"Force restart failed: {e}")

    # ------------------------------------------------------------------
    # Register / unregister hotkeys (just updates our dict, no library calls)
    # ------------------------------------------------------------------
    def register_all(self, bindings):
        """Register hotkeys.  bindings = {name: (hotkey_str, callback)}"""
        with self._lock:
            self._bindings.clear()
            for name, (hotkey_str, callback) in bindings.items():
                self._bindings[name] = {
                    "keys": self._parse_hotkey(hotkey_str),
                    "callback": callback,
                    "hotkey_str": hotkey_str,
                }
                self._flog(
                    f"Hotkey registered: mode={name}, "
                    f"hotkey={hotkey_str}, "
                    f"keys={self._bindings[name]['keys']}"
                )
            self._last_registration = time.monotonic()
            self._last_hook_event = time.monotonic()

    def unregister_all(self):
        with self._lock:
            for name in list(self._bindings):
                self._flog(f"Hotkey unregistered: mode={name}")
            self._bindings.clear()

    # ------------------------------------------------------------------
    # Health status
    # ------------------------------------------------------------------
    def health_status(self, expected_hotkeys):
        with self._lock:
            registered = {
                name: b["hotkey_str"] for name, b in self._bindings.items()
            }
            missing = [n for n in expected_hotkeys if n not in registered]
            mismatched = [
                n for n, hk in expected_hotkeys.items()
                if registered.get(n) != hk
            ]

        thread_ok, thread_reason = self._listener_thread_health()
        hook_alive, hook_reason = self._is_hook_alive()

        reason_parts = []
        if missing:
            reason_parts.append("handlers missing: " + ", ".join(missing))
        if mismatched:
            reason_parts.append("hotkey config changed: " + ", ".join(mismatched))
        if not thread_ok:
            reason_parts.append(thread_reason)

        if not hook_alive:
            self._consecutive_dead_checks += 1
            if self._consecutive_dead_checks >= self.HOOK_DEAD_CONFIRM_CYCLES:
                reason_parts.append("HOOK DEAD: " + hook_reason)
            else:
                self._flog(
                    f"Hook possibly dead "
                    f"({self._consecutive_dead_checks}/"
                    f"{self.HOOK_DEAD_CONFIRM_CYCLES}): {hook_reason}"
                )
        else:
            self._consecutive_dead_checks = 0

        hook_confirmed_dead = (
            not hook_alive
            and self._consecutive_dead_checks >= self.HOOK_DEAD_CONFIRM_CYCLES
        )

        return {
            "healthy": not reason_parts,
            "reason": "; ".join(reason_parts) if reason_parts else "ok",
            "hook_dead": hook_confirmed_dead,
            # No longer needed since we don't use add_hotkey, but kept
            # for API compatibility with the watchdog.
            "hotkey_corrupted": False,
        }

    def cleanup(self):
        """Clean up before shutdown."""
        self._remove_hook()
        with self._lock:
            self._bindings.clear()

def detect_cuda_available():
    """Check if CUDA is available and usable for inference.
    Returns tuple: (is_available: bool, reason: str)
    """
    try:
        import ctranslate2
        # Check if CUDA is supported by ctranslate2
        supported_devices = ctranslate2.get_supported_compute_types("cuda")
        if supported_devices:
            return True, "CUDA detected and available"
        else:
            return False, "CUDA not supported by ctranslate2"
    except Exception as e:
        error_str = str(e).lower()
        if any(kw in error_str for kw in CUDA_DETECT_KEYWORDS):
            return False, f"CUDA libraries not found: {e}"
        return False, f"CUDA detection error: {e}"

def get_optimal_device_config():
    """Detect and return optimal device and compute_type settings."""
    cuda_available, reason = detect_cuda_available()
    if cuda_available:
        return AUTO_DEVICE_CONFIG["cuda_device"], AUTO_DEVICE_CONFIG["cuda_compute_type"], reason
    else:
        return AUTO_DEVICE_CONFIG["cpu_device"], AUTO_DEVICE_CONFIG["cpu_compute_type"], reason

# --- THEME COLORS ---
COLOR_BG = THEME_COLORS["bg"]           # Darker background
COLOR_FG = THEME_COLORS["fg"]           # White text
COLOR_ACCENT_BG = THEME_COLORS["accent_bg"]    # Elevated panels
COLOR_TEXT_BG = THEME_COLORS["text_bg"]      # Card backgrounds
COLOR_TEAL = THEME_COLORS["teal"]         # Primary accent (bars, highlights)
COLOR_TEAL_DIM = THEME_COLORS["teal_dim"]     # Subtle teal for backgrounds
COLOR_BORDER = THEME_COLORS["border"]       # Subtle borders
COLOR_LIVE = THEME_COLORS["live"]         # Red for live mode
COLOR_BATCH = THEME_COLORS["batch"]        # Blue for batch mode
COLOR_IDLE = THEME_COLORS["idle"]         # Gray
COLOR_ONLINE = THEME_COLORS["online"]       # Purple for online mode
COLOR_TRANSCRIBING = THEME_COLORS["transcribing"] # Muted gold for transcribing
COLOR_READY = THEME_COLORS["ready"]        # Sage green for ready to paste
COLOR_TEXT_MUTED = THEME_COLORS["text_muted"]   # Secondary label text
COLOR_TEXT_FAINT = THEME_COLORS["text_faint"]   # Small print / captions
COLOR_STOPPING_LIVE = THEME_COLORS["stopping_live"]
COLOR_STOPPING_BATCH = THEME_COLORS["stopping_batch"]

# Configure CustomTkinter
ctk.set_appearance_mode(CTK_APPEARANCE_MODE)
ctk.set_default_color_theme(CTK_COLOR_THEME)


# --- Speech-insights pure helpers (module level so they can be unit-tested
# without instantiating WhisperApp) ---

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")

_GERMAN_UMLAUTS = set("äöüßÄÖÜ")


def tokenize_words(text):
    """Unicode-aware word tokenizer (keeps umlauts, drops digits/punctuation)."""
    return _WORD_RE.findall((text or "").lower())


def detect_entry_language(text, tokens=None):
    """Cheap per-entry language heuristic -> 'de' or 'en'.

    German if the text contains an umlaut/eszett, or if the share of German
    function words among the tokens clears GERMAN_DETECT_THRESHOLD.
    """
    if not text:
        return "en"
    if any(ch in _GERMAN_UMLAUTS for ch in text):
        return "de"
    if tokens is None:
        tokens = tokenize_words(text)
    if not tokens:
        return "en"
    hits = sum(1 for t in tokens if t in GERMAN_FUNCTION_WORDS)
    return "de" if (hits / len(tokens)) >= GERMAN_DETECT_THRESHOLD else "en"


def filler_patterns_for(lang):
    return GERMAN_FILLER_PATTERNS if lang == "de" else ENGLISH_FILLER_PATTERNS


def stopwords_for(lang):
    return GERMAN_STOPWORDS if lang == "de" else ENGLISH_STOPWORDS


def count_fillers(text, lang):
    """Return {pattern_label: count} for the given language's filler list."""
    lowered = (text or "").lower()
    return {k: len(re.findall(p, lowered)) for k, p in filler_patterns_for(lang).items()}


def filter_entries_by_days(entries, days, now=None):
    """Rolling-window filter over analytics entries.

    days is None -> everything; 0 -> the current calendar day; N -> the
    trailing N days. Returns the list of matching indices.
    """
    if days is None:
        return list(range(len(entries)))
    now = now or datetime.datetime.now()
    if days == 0:
        today = now.date()
        return [i for i, e in enumerate(entries) if e["timestamp"].date() == today]
    cutoff = now - datetime.timedelta(days=days)
    return [i for i, e in enumerate(entries) if e["timestamp"] >= cutoff]


class WhisperApp:
    def __init__(self, root):
        self.root = root
        self._shutting_down = False
        # Any exception raised inside a Tk callback would otherwise be printed
        # to a stdout that does not exist under pythonw.exe.
        try:
            self.root.report_callback_exception = self._report_tk_exception
        except Exception:
            pass
        self.root.title(APP_TITLE)
        # Clamp the default height to the screen: 830px does not fit on a
        # 768p laptop, and an off-screen window looks like "nothing happened".
        try:
            _w, _h = (int(part) for part in APP_WINDOW_GEOMETRY.split("x", 1))
        except Exception:
            _w, _h = 480, 830
        _h = min(_h, max(APP_WINDOW_MIN_HEIGHT, self.root.winfo_screenheight() - 80))
        self.root.geometry(f"{_w}x{_h}")  # Single 480px column; accordions closed fit without scrolling
        self.root.minsize(APP_WINDOW_MIN_WIDTH, APP_WINDOW_MIN_HEIGHT)
        
        # Set window icon (both title bar and taskbar)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ICON_ICO)
        png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ICON_PNG)
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception:
                pass
        # Also try wm_iconphoto for taskbar (works better on some Windows versions)
        if os.path.exists(png_path):
            try:
                from PIL import Image, ImageTk
                icon_img = Image.open(png_path)
                icon_photo = ImageTk.PhotoImage(icon_img)
                self.root.wm_iconphoto(True, icon_photo)
                self._icon_photo = icon_photo  # Keep reference to prevent garbage collection
            except Exception:
                pass
        
        # Apply Dark Theme (keep ttk style for some legacy widgets)
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        self.root.configure(bg=COLOR_BG)
        
        # Created before load_config(): a corrupt-config quarantine logs through
        # log_internal(), which needs this queue, and an AttributeError there
        # would swallow the very message the user has to see.
        self.msg_queue = queue.Queue()
        # Set when an unreadable config file could not be renamed aside; blocks
        # every config write for the session (see _quarantine_corrupt_file).
        self._config_write_blocked = False

        self.config = self.load_config()

        # Plain-attribute mirror of live_pause_var. The audio thread reads the
        # pause threshold every chunk, and reading a Tk variable off the main
        # thread is not safe; the settings trace keeps this in sync.
        try:
            self._live_pause = float(self.config['live_pause'])
        except (TypeError, ValueError, KeyError):
            self._live_pause = DEFAULT_CONFIG['live_pause']

        # State
        self.mode = None 
        self.stopping = False
        self.model = None
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.hotkey_action_queue = queue.Queue(maxsize=HOTKEY_RUNTIME["action_queue_maxsize"])
        self._last_hotkey_event_time = {'batch': 0.0, 'live': 0.0}
        self._capture_hooks = None
        self.running = True
        
        # Stats State - Load persistent stats (own machine) + read-only peer merge
        self.stats = self.load_stats()
        self._peer_stats = self._load_peer_stats()
        self.total_words = self.stats.get('total_words', 0)
        self.total_audio_duration = self.stats.get('total_audio_duration', 0.0)
        
        # Today's stats - persist across restarts within the same day
        today_str = datetime.datetime.now().date().isoformat()
        today_data = self.stats.get('today_data', {})
        if today_data.get('date') == today_str:
            # Same day - restore today's stats
            self.today_words = today_data.get('words', 0)
            self.today_audio_duration = today_data.get('audio_duration', 0.0)
        else:
            # New day - reset today's stats
            self.today_words = 0
            self.today_audio_duration = 0.0
        self.session_date = today_str
        
        # Transcription history
        self.history = []  # List of (timestamp, text, latency_ms) tuples
        self.history_index = -1  # -1 = showing latest live transcription
        self.current_transcription = ""  # Currently displayed transcription
        self.current_latency = 0  # Latency in ms for current transcription
        self.last_transcription_latency = 0  # Last measured transcription latency
        
        # VU Meter Smoothing State
        self.current_display_volume = 0.0
        
        # Buffers
        self.batch_total_samples = 0  # Running sample count for duration tracking
        self.live_buffer = [] # Class var to fix "last bit" bug
        self.live_backup_buffer = []
        self.live_buffer_lock = threading.Lock()  # Prevent race conditions on buffer

        # Transcription backend (set by load_model on its worker thread)
        self.backend = None
        self.current_backend_name = None
        self.current_device = None

        # Idle GPU unload: the model auto-unloads after idle_unload_minutes of
        # no dictation activity (freeing GPU memory held by faster-whisper's
        # CUDA context), then transparently reloads on the next transcription.
        # _model_lock guards load/unload/transcribe races between the idle
        # timer thread and the transcription worker threads.
        self._model_lock = threading.RLock()
        self._model_idle_unloaded = False
        self._last_activity_time = time.time()

        # Analytics cache: parsed transcript entries, appended incrementally so
        # we don't re-read the whole archive after every transcription
        self._analytics_entries_cache = None
        # Tokens/language per entry, kept parallel to _analytics_entries_cache so
        # a refresh doesn't re-tokenize the whole archive
        self._analytics_tokens_cache = None
        self._analytics_langs_cache = None
        self._insights_refresh_job = None
        # Personal trailing-baseline metrics (see _compute_metric_snapshot)
        self._personal_baseline = {}
        self._personal_baseline_days = 0
        # AI insight crunch guards (manual button + once-per-start auto run)
        self._ai_crunch_lock = threading.Lock()
        self._ai_crunch_running = False
        self._ai_auto_crunch_done = False

        # Debounced stats persistence: the stats file lives in a Dropbox-synced
        # folder, so rewriting it after every segment causes constant sync churn
        self._stats_save_job = None
        self._stats_dirty = False
        # Last date seen by record_hourly_words(); a rollover forces an
        # immediate flush so a crash can never lose a whole day's bucket.
        self._stats_last_day = None

        # Debounced histogram redraw + cached own/peer hourly merge
        self._histogram_job = None
        self._hourly_merge_cache = None

        # Audio input stream: opened when a recording starts, closed when it
        # ends, so the microphone is released while idle
        self.stream = None

        # Shutdown guard (on_close is idempotent)
        self._shutting_down = False
        
        # Mini Window Reference
        self.mini_window = None
        self.mini_vu = None
        self.mini_canvas = None
        
        # Streaming Batch Transcription
        self.batch_segments = []           # Ordered list of {seq, audio, result, status}
        self.batch_segment_lock = threading.Lock()
        self.batch_segment_seq = 0         # Sequence counter for ordering
        self.batch_executor = None         # ThreadPoolExecutor for parallel transcription
        self.batch_pending_audio = []      # Current audio buffer not yet segmented
        self.batch_silence_count = 0       # Silence detection counter

        # Online Streaming Transcription (similar to batch but uses OpenAI API)
        self.online_segments = []          # Ordered list of {seq, audio, result, status}
        self.online_segment_lock = threading.Lock()
        self.online_segment_seq = 0        # Sequence counter for ordering
        self.online_executor = None        # ThreadPoolExecutor for parallel API calls
        self.online_pending_audio = []     # Current audio buffer for online mode
        self.online_silence_count = 0      # Silence detection counter for online
        self.online_with_edit = False      # Whether current online session uses GPT editing
        self.online_backend = None         # Configured OpenAI backend reused for all segments of a session

        # Latency tracking for performance metrics
        self.session_latencies = []  # List of latency values (ms) for this session
        self.total_latencies = self.stats.get('total_latencies', [])  # All-time latencies

        self.setup_ui()
        self.load_recent_transcriptions()  # Load history from last 2 days
        self.refresh_insights_tabs()  # Populate tabbed analytics from full transcript archive
        # Refresh stale AI insights once per app start, well after the UI settles
        self.root.after(30000, self._maybe_auto_crunch_ai_insights)
        self.update_transcription_display()  # Show initial display
        self.root.attributes('-topmost', self.config['always_on_top'])
        
        # Threads
        threading.Thread(target=self.processing_loop, daemon=True).start()
        threading.Thread(target=self.load_model, daemon=True).start()
        threading.Thread(target=self._idle_unload_loop, daemon=True).start()
        
        # Audio - the input stream is opened on demand when a recording starts
        # and closed when it ends, so the microphone is free while idle.

        # Hotkeys - mode-aware (work in both Local and Online mode)
        # trigger_on_release=False prevents double-firing on key-down AND key-up
        # Store registration state in a dedicated manager for safer lifecycle handling.
        self._hotkey_manager = HotkeyManager(keyboard, self.log_internal)
        self._hotkey_auto_rebind_count = 0
        self._hotkey_refresh_interval = HOTKEY_RUNTIME["refresh_interval_seconds"]  # Force re-registration every 3 minutes (was 30 min)
        self._hotkey_watchdog_interval = HOTKEY_RUNTIME["watchdog_interval_ms"]  # Check every 5 seconds
        self._last_focus_refresh = 0  # Track focus-based refreshes to avoid spamming
        self._register_hotkeys()

        # Periodic hotkey health check
        self._start_hotkey_watchdog()

        # Fast hotkey pump: drains the action queue every 50ms independently of
        # the adaptive GUI loop (which idles at 1000ms - too slow for a hotkey
        # press, and recording only starts once the action is processed)
        self.root.after(50, self._hotkey_pump)

        # Window focus handler - refresh hotkeys when app regains focus
        self.root.bind("<FocusIn>", self._on_focus_in)
        
        # Minimization Binding
        self.root.bind("<Unmap>", self.on_minimize)
        self.root.bind("<Map>", self.on_restore)
        
        # UI Update Loop
        self.root.after(UI_RUNTIME["initial_gui_loop_delay_ms"], self.update_gui_loop)
        
        # Delayed histogram update (after canvas is fully sized)
        self.root.after(UI_RUNTIME["initial_histogram_delay_ms"], self.update_histogram)

        # Orderly shutdown when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # First-run welcome. Scheduled on the Tk loop (never on the model
        # thread started above) and shown at most once per install.
        self.root.after(700, self._maybe_show_welcome)

    # --- Logging / Tk-safety helpers -------------------------------------

    def _report_tk_exception(self, exc, val, tb):
        """Tk callback errors go to logs/app.log instead of a missing stdout."""
        import traceback

        text = "".join(traceback.format_exception(exc, val, tb))
        log.error("Tk callback exception:\n%s", text)
        try:
            self.log_internal(f"Internal error: {val}")
        except Exception:
            pass

    def _log_once(self, key, message, level="error"):
        """Log a repeating message only the first time it is seen, so a loop
        that fails every tick cannot fill logs/app.log."""
        seen = getattr(self, "_logged_once", None)
        if seen is None:
            seen = self._logged_once = set()
        token = f"{key}:{message}"
        if token in seen:
            return False
        seen.add(token)
        getattr(log, level, log.error)("%s", message)
        return True

    def _ui_after(self, delay_ms, fn, *args):
        """root.after() that is safe to call from a worker thread.

        Returns without scheduling once shutdown has started, and swallows the
        TclError/RuntimeError raised when the interpreter is already gone -
        otherwise a background thread can keep a half-destroyed Tk alive (or
        raise into nowhere) while the window is closing.
        """
        if self._shutting_down:
            return None
        try:
            return self.root.after(delay_ms, fn, *args)
        except (tk.TclError, RuntimeError):
            return None

    @staticmethod
    def _write_json_atomic(path, data, indent=None):
        """Write JSON to `path` without ever leaving a truncated file behind:
        write a private sibling temp file, fsync it, then os.replace() over
        the target.

        The temp name carries pid + random suffix because this folder may be
        synced: a fixed "<path>.tmp" is the same name on every machine, and
        two of them writing at once would corrupt each other's temp file.

        os.replace is retried because on Windows a sync client or an
        antivirus scanner briefly holds the target open, which surfaces as
        PermissionError on an otherwise perfectly fine write.
        """
        import uuid

        tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
                f.flush()
                os.fsync(f.fileno())
            last_error = None
            for attempt in range(3):
                try:
                    os.replace(tmp_path, path)
                    return
                except PermissionError as e:
                    last_error = e
                    if attempt < 2:
                        time.sleep(0.2)
            raise last_error
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _quarantine_corrupt_file(self, path, exc):
        """Rename an unreadable JSON file out of the way instead of silently
        overwriting it with defaults (a corrupt stats file still holds the
        user's history and may be recoverable by hand)."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = f"{path}.corrupt-{stamp}"
        name = os.path.basename(path)
        is_local_overlay = os.path.abspath(path) == os.path.abspath(self._local_config_file())

        try:
            os.replace(path, target)
        except Exception as move_error:
            # The file is unreadable AND we could not move it aside. Continuing
            # with defaults is fine, but writing over it is not: that would
            # destroy the only copy of settings (and, for the per-machine
            # overlay, the OpenAI API key) that might still be recoverable.
            self._config_write_blocked = True
            log.error(
                "Could not read %s (%s) and could not move it aside (%s). "
                "Config writes are disabled for this session so the file is not overwritten.",
                name, exc, move_error,
            )
            try:
                self.log_internal(
                    f"⚠️ {name} is unreadable and could not be renamed ({move_error}). "
                    "Settings will not be saved this session so the file stays intact."
                )
                self.status_var.set("Config unreadable - settings will not be saved")
            except Exception:
                pass
            return None

        log.error("%s was unreadable (%s); moved to %s and continuing with defaults.",
                  name, exc, os.path.basename(target))
        message = (
            f"{name} was corrupt - renamed to {os.path.basename(target)} and defaults restored."
        )
        if is_local_overlay:
            # That overlay is where the OpenAI key lives (MACHINE_LOCAL_KEYS).
            message += (
                " That file held your OpenAI API key: it is still inside the renamed"
                " copy, so delete it once you have re-entered the key (and never share it)."
            )
            log.error(
                "%s held the OpenAI API key; it now lives in %s - delete that file after"
                " re-entering the key.", name, os.path.basename(target),
            )
        try:
            self.log_internal(message)
        except Exception:
            pass
        return target

    def configure_styles(self):
        # Every remaining ttk.Frame/ttk.Label in the main column lives inside
        # the Configuration card, so the ttk defaults have to match the card
        # fill (COLOR_ACCENT_BG) rather than the window ground (COLOR_BG) -
        # otherwise the expanded panel reads as a dark inset inside a lighter
        # card. The CUDA modal, which does sit on COLOR_BG, uses the explicit
        # "Dialog.*" styles below.
        self.style.configure("TFrame", background=COLOR_ACCENT_BG)
        self.style.configure("TLabel", background=COLOR_ACCENT_BG, foreground=COLOR_FG)
        self.style.configure("TLabelframe", background=COLOR_ACCENT_BG, foreground=COLOR_FG)
        self.style.configure("TLabelframe.Label", background=COLOR_ACCENT_BG, foreground=COLOR_FG)
        self.style.configure("Dialog.TFrame", background=COLOR_BG)
        self.style.configure("Dialog.TLabel", background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Dialog.TLabelframe", background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Dialog.TLabelframe.Label", background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("TButton", background=COLOR_ACCENT_BG, foreground=COLOR_FG, borderwidth=1)
        self.style.map("TButton", background=[('active', '#505050')])
        self.style.configure("TCheckbutton", background=COLOR_ACCENT_BG, foreground=COLOR_FG)
        self.style.configure("Horizontal.TProgressbar", troughcolor=COLOR_ACCENT_BG, background=COLOR_TEAL, bordercolor=COLOR_BG, lightcolor=COLOR_TEAL, darkcolor=COLOR_TEAL)

    def _create_tooltip(self, widget, text):
        """Create a hover tooltip for a widget"""
        tooltip = None
        
        def show_tooltip(event):
            nonlocal tooltip
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.configure(bg="#333")
            
            label = tk.Label(tooltip, text=text, justify="left",
                           bg="#333", fg="#fff", font=("Arial", 9),
                           relief="solid", borderwidth=1, padx=8, pady=4)
            label.pack()
        
        def hide_tooltip(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None
        
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def play_feedback_sound(self, start=True):
        # Use platform-aware sound function
        if PLATFORM_UTILS_AVAILABLE:
            play_feedback_sound_async(start=start)
        elif IS_WINDOWS:
            try:
                freq = 800 if start else 400
                dur = 150
                threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()
            except Exception:
                pass


    # Keys that differ per machine (GPU vs cloud, audio device, model selection).
    # 'openai_api_key' lives here so the secret stays in the unsynced per-machine
    # overlay instead of the Dropbox-shared whisper_config.json.
    MACHINE_LOCAL_KEYS = {
        'model_key', 'backend', 'device', 'compute_type',
        'input_device', 'transcription_mode', 'openai_api_key',
    }

    @staticmethod
    def _hostname():
        import platform
        return platform.node().replace('.', '_') or 'machine'

    def _local_config_file(self):
        """Per-machine config overlay: whisper_config.{COMPUTERNAME}.json"""
        return os.path.join(app_dir, f"whisper_config.{self._hostname()}.json")

    def _stats_file(self):
        """Per-machine stats file. Each machine only ever writes its own file,
        so Dropbox never produces 'conflicted copy' duplicates of the stats."""
        return os.path.join(app_dir, f"whisper_stats.{self._hostname()}.json")

    def load_config(self):
        loaded = {}
        config = DEFAULT_CONFIG.copy()
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if "model_size" in loaded:
                    old_val = loaded.pop("model_size")
                    if old_val in MODEL_MAP_REVERSE:
                        loaded["model_key"] = MODEL_MAP_REVERSE[old_val]
                    else:
                        loaded["model_key"] = DEFAULT_CONFIG["model_key"]
                config = {**DEFAULT_CONFIG, **loaded}
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # Never silently overwrite a file we could not parse.
                self._quarantine_corrupt_file(os.path.join(app_dir, CONFIG_FILE), e)
                loaded = {}
                config = DEFAULT_CONFIG.copy()
            except Exception as e:
                log.error("Failed to read %s (%s); continuing with defaults.", CONFIG_FILE, e)
                loaded = {}
                config = DEFAULT_CONFIG.copy()

        # Load per-machine overrides (device, backend, model selection, mic)
        local_file = self._local_config_file()
        if os.path.exists(local_file):
            try:
                with open(local_file, 'r', encoding='utf-8') as f:
                    local = json.load(f)
                config.update({k: v for k, v in local.items() if k in self.MACHINE_LOCAL_KEYS})
                log.info("[Config] Loaded machine-local overrides from %s", os.path.basename(local_file))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._quarantine_corrupt_file(local_file, e)
            except Exception as e:
                log.error("[Config] Warning: failed to load local config: %s", e)

        # Migration: older versions wrote machine-local/sensitive keys (incl. the
        # OpenAI API key) into the Dropbox-synced shared file. Move them into the
        # unsynced per-machine overlay and rewrite the shared file without them.
        leaked = [k for k in self.MACHINE_LOCAL_KEYS if k in loaded]
        needs_rewrite = bool(leaked)
        if leaked:
            log.info("[Config] Migrating %s out of shared config into local overlay", leaked)

        # Migration: a config file that already exists belongs to someone who
        # has been running the app, not to a first-time user. Mark the welcome
        # dialog as already seen so upgrading never pops it up.
        if loaded and not loaded.get("welcome_shown", False):
            config["welcome_shown"] = True
            needs_rewrite = True
            log.info("[Config] Existing config found - marking the first-run welcome as already shown")

        if needs_rewrite:
            self._persist_config(config)

        # Auto-detect device/compute_type on a fresh install (both default to
        # "auto"). The resolved values are written to the per-machine overlay,
        # so this probe runs once per machine and not on every launch.
        # 'backend' is deliberately NOT part of this: it stays "auto" and is
        # resolved by _load_model_locked() on the model thread, against the
        # live backend registry that also knows OpenVINO/Parakeet.
        if config.get('device') == 'auto' or config.get('compute_type') == 'auto':
            try:
                device, compute_type, reason = get_optimal_device_config()
            except Exception as e:
                # Detection itself blew up: land on the universally safe combo.
                device, compute_type = 'cpu', 'int8'
                reason = f"hardware detection failed ({e}) - defaulting to CPU"
                log.error("[Auto-detect] %s", reason)
            if config.get('device') == 'auto':
                config['device'] = device
            if config.get('compute_type') == 'auto':
                config['compute_type'] = compute_type
            # 'backend' stays "auto" on purpose: the Backend dropdown shows
            # "Auto" and _load_model_locked() resolves it against the live
            # backend registry (which knows about OpenVINO/Parakeet too).
            log.info("[Auto-detect] %s -> Using %s mode", reason, str(config['device']).upper())
            # Save the detected config so it persists
            self._save_local_config(config)

        return config

    def load_stats(self):
        """Load persistent statistics (per-machine file, see _stats_file)."""
        stats_path = self._stats_file()
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data.setdefault('hourly_data', {})
                return data
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # Move it aside rather than starting from zero on top of it.
                self._quarantine_corrupt_file(stats_path, e)
            except Exception as e:
                log.error("[Stats] Failed to read %s: %s", os.path.basename(stats_path), e)

        # One-time migration: claim the legacy shared whisper_stats.json as this
        # machine's baseline. The 'claimed_by' marker (synced via Dropbox) stops
        # the other machine from also adopting it, which would double-count.
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r', encoding='utf-8') as f:
                    legacy = json.load(f)
                claimer = legacy.get('claimed_by')
                if claimer is None or claimer == self._hostname():
                    if claimer is None:
                        legacy['claimed_by'] = self._hostname()
                        self._write_json_atomic(os.path.join(app_dir, STATS_FILE), legacy)
                    data = {k: v for k, v in legacy.items() if k != 'claimed_by'}
                    data.setdefault('hourly_data', {})
                    # Persist the baseline immediately so a crash before the
                    # first save_stats() can't lose it
                    self._write_json_atomic(stats_path, data)
                    log.info("[Stats] Claimed legacy stats file as baseline for %s", self._hostname())
                    return data
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._quarantine_corrupt_file(os.path.join(app_dir, STATS_FILE), e)
            except Exception as e:
                log.error("[Stats] Legacy stats migration skipped: %s", e)
        return {'total_words': 0, 'total_audio_duration': 0.0, 'hourly_data': {}}

    def _load_peer_stats(self):
        """Read-only merge of the other machines' stats files (synced via
        Dropbox) so the All-Time numbers stay combined across machines."""
        peers = {'total_words': 0, 'total_audio_duration': 0.0, 'hourly_data': {}}
        own_name = os.path.basename(self._stats_file())
        try:
            for filename in os.listdir(app_dir):
                if not (filename.startswith('whisper_stats.') and filename.endswith('.json')):
                    continue
                # Skip our own file, the legacy shared file, and any sync debris
                if filename in (own_name, STATS_FILE) or '(' in filename or 'conflicted' in filename.lower():
                    continue
                with open(os.path.join(app_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                peers['total_words'] += data.get('total_words', 0)
                peers['total_audio_duration'] += data.get('total_audio_duration', 0.0)
                for date_str, day_data in data.get('hourly_data', {}).items():
                    day = peers['hourly_data'].setdefault(date_str, {})
                    for hour_str, count in day_data.items():
                        day[hour_str] = day.get(hour_str, 0) + count
        except Exception as e:
            log.error("[Stats] Warning: failed to merge peer stats: %s", e)
        return peers

    def save_stats(self):
        """Mark stats dirty and schedule a debounced write (trailing 10s).

        The stats file sits in a Dropbox-synced folder, so writing it after
        every transcription/segment causes constant sync churn. Bursts are
        coalesced into one write; on_close() flushes immediately, and
        record_hourly_words() flushes on a day rollover.

        Known exposure: a termination that never delivers WM_DELETE_WINDOW
        (Task Manager kill, forced logoff) loses up to the debounce window of
        the current day's stats.
        """
        self._stats_dirty = True
        if self._stats_save_job is not None:
            try:
                self.root.after_cancel(self._stats_save_job)
            except Exception:
                pass
        self._stats_save_job = self.root.after(10000, self._run_scheduled_stats_save)

    def _run_scheduled_stats_save(self):
        self._stats_save_job = None
        self.flush_stats()

    def flush_stats(self):
        """Write any pending stats to disk now (no-op if nothing changed)."""
        if self._stats_save_job is not None:
            try:
                self.root.after_cancel(self._stats_save_job)
            except Exception:
                pass
            self._stats_save_job = None
        if not self._stats_dirty:
            return
        self._write_stats()

    def _write_stats(self):
        """Save statistics to this machine's stats file."""
        self.stats['total_words'] = self.total_words
        self.stats['total_audio_duration'] = self.total_audio_duration
        # Keep the all-time latency list bounded (display only uses the tail)
        self.total_latencies = self.total_latencies[-500:]
        self.stats['total_latencies'] = self.total_latencies
        # Save today's data for persistence across restarts
        self.stats['today_data'] = {
            'date': self.session_date,
            'words': self.today_words,
            'audio_duration': self.today_audio_duration
        }
        try:
            self._write_json_atomic(self._stats_file(), self.stats)
            self._stats_dirty = False
        except Exception as e:
            self._log_once("write_stats", f"[Stats] Failed to write stats file: {e}")

    def record_hourly_words(self, word_count):
        """Record words for the current hour in stats"""
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%H")
        
        # Day rollover: persist the finished day before starting a new bucket,
        # so an unclean exit can only ever lose the debounce window of the
        # current day, never a completed one.
        if self._stats_last_day is not None and self._stats_last_day != date_str:
            try:
                self.flush_stats()
            except Exception:
                pass
        self._stats_last_day = date_str

        if 'hourly_data' not in self.stats:
            self.stats['hourly_data'] = {}
        if date_str not in self.stats['hourly_data']:
            self.stats['hourly_data'][date_str] = {}
        
        current = self.stats['hourly_data'][date_str].get(hour_str, 0)
        self.stats['hourly_data'][date_str][hour_str] = current + word_count
        self._hourly_merge_cache = None  # histogram merge must be recomputed

    def format_speaking_time(self, seconds):
        """Format speaking duration in d/h/m format"""
        minutes = seconds / 60.0
        if minutes < 60:
            return f"{minutes:.1f}m"
        
        total_minutes = int(minutes)
        hours = total_minutes // 60
        mins = total_minutes % 60
        
        if hours < 24:
            return f"{hours}h {mins}m"
        
        days = hours // 24
        remaining_hours = hours % 24
        return f"{days}d {remaining_hours}h {mins}m"

    def format_time_saved(self, minutes):
        """
        Format time saved intelligently:
        - Under 60 minutes: show minutes (e.g., "45m")
        - 60+ minutes: show hours + minutes (e.g., "2h 15m")
        - 24+ hours: show days + hours + minutes (e.g., "3d 5h 30m")
        """
        if minutes < 60:
            return f"{minutes:.1f}m"
        
        total_minutes = int(minutes)
        hours = total_minutes // 60
        mins = total_minutes % 60
        
        if hours < 24:
            return f"{hours}h {mins}m"
        
        days = hours // 24
        remaining_hours = hours % 24
        return f"{days}d {remaining_hours}h {mins}m"

    def get_safe_model_path(self, friendly_name):
        safe_name = friendly_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        local_models_root = os.path.join(app_dir, "models")
        return os.path.join(local_models_root, safe_name)

    # Approximate on-disk download sizes, shown in the first-run status text.
    MODEL_DOWNLOAD_SIZES = {
        "base": "150 MB",
        "small": "460 MB",
        "medium": "1.5 GB",
        "large-v3": "3 GB",
    }

    def is_model_downloaded(self, friendly_name):
        """True only when a real model weight file is present.

        A merely non-empty directory is not enough: an aborted first download
        leaves config/tokenizer JSONs behind, and treating that as "downloaded"
        made the app skip the download and then fail to load, forever.
        """
        target_dir = self.get_safe_model_path(friendly_name)
        if not os.path.isdir(target_dir):
            return False
        for _root, _dirs, files in os.walk(target_dir):
            if "model.bin" in files:
                return True
            # NeMo/Parakeet ships a single .nemo archive instead of model.bin,
            # and OpenVINO ships an IR pair (openvino_*.xml + .bin). Without
            # these an OpenVINO/Parakeet user sees "Downloading..." on every
            # single start even though the model is already on disk.
            if any(name.endswith(".nemo") or name.endswith(".xml") for name in files):
                return True
        return False

    @staticmethod
    def _temp_audio_path(tag):
        """Scratch WAV path in the OS temp dir (keeps churn out of the Dropbox folder)."""
        import tempfile
        return os.path.join(tempfile.gettempdir(), f"neurowhisper_{tag}_{os.getpid()}.wav")

    def _save_local_config(self, config=None):
        """Save machine-specific keys to per-machine config file."""
        if config is None:
            config = self.config
        if getattr(self, "_config_write_blocked", False):
            return
        local_data = {k: config[k] for k in self.MACHINE_LOCAL_KEYS if k in config}
        try:
            self._write_json_atomic(self._local_config_file(), local_data)
        except Exception as e:
            log.error("[Config] Warning: failed to save local config: %s", e)

    def _persist_config(self, config=None):
        """Single write path for config: shared keys go to the synced file,
        machine-local/sensitive keys (incl. the API key) go to the local overlay."""
        if config is None:
            config = self.config
        if getattr(self, "_config_write_blocked", False):
            self._log_once(
                "config_write_blocked",
                "Config writes are disabled this session: an unreadable config file could not "
                "be renamed out of the way, so it is left untouched.",
                level="warning",
            )
            return
        shared = {k: v for k, v in config.items() if k not in self.MACHINE_LOCAL_KEYS}
        try:
            self._write_json_atomic(os.path.join(app_dir, CONFIG_FILE), shared)
        except Exception as e:
            log.error("[Config] Warning: failed to save shared config: %s", e)
        self._save_local_config(config)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling for the main content area."""
        # Don't scroll canvas when mouse is over a text widget with its own scrollbar
        widget = event.widget
        if isinstance(widget, (tk.Text, scrolledtext.ScrolledText)):
            return
        self._scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _update_scroll_region(self, event=None):
        """Update the scroll region when content changes size, and show the
        scrollbar only while the content actually overflows the viewport."""
        self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
        try:
            req = self.content_frame.winfo_reqheight()
            avail = self._scroll_canvas.winfo_height()
            mapped = bool(self._scroll_vsb.winfo_ismapped())
            # Hysteresis: hiding the bar widens the canvas, which can re-wrap
            # content and change reqheight, so the hide threshold sits well
            # below the show threshold to stop the two from oscillating.
            if not mapped and req > avail:
                self._scroll_vsb.pack(side="right", fill="y",
                                      before=self._scroll_canvas)
            elif mapped and req <= avail - 24:
                self._scroll_vsb.pack_forget()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shared card / accordion chrome (one style everywhere)
    # ------------------------------------------------------------------
    CARD_RADIUS = 14
    ROW_RADIUS = 11
    CARD_PAD = 13
    CARD_GAP = 10

    def _make_card(self, parent, **kw):
        opts = dict(fg_color=COLOR_ACCENT_BG, corner_radius=self.CARD_RADIUS,
                    border_width=1, border_color=COLOR_BORDER)
        opts.update(kw)
        return ctk.CTkFrame(parent, **opts)

    def _make_accordion(self, parent, title, summary_var, command, collapsed=True):
        """A 42px accordion header row plus its (unpacked) body card.

        Returns (shell, toggle_button, body). The body is packed/unpacked by
        the caller's toggle handler.
        """
        shell = tk.Frame(parent, bg=COLOR_BG)
        shell.pack(fill="x", pady=(0, 6))

        header = ctk.CTkFrame(shell, fg_color=COLOR_ACCENT_BG, corner_radius=self.ROW_RADIUS,
                              border_width=1, border_color=COLOR_BORDER, height=42)
        header.pack(fill="x")
        header.pack_propagate(False)

        btn = tk.Button(header, text="▶" if collapsed else "▼",
                        bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, font=("Arial", 9, "bold"),
                        relief="flat", bd=0, highlightthickness=0, cursor="hand2",
                        activebackground=COLOR_ACCENT_BG, activeforeground=COLOR_TEAL,
                        command=command)
        btn.pack(side="left", padx=(self.CARD_PAD - 2, 2))

        title_lbl = tk.Label(header, text=title, bg=COLOR_ACCENT_BG, fg=COLOR_FG,
                             font=("Arial", 11, "bold"), cursor="hand2")
        title_lbl.pack(side="left")
        title_lbl.bind("<Button-1>", lambda _e: command())

        tk.Label(header, textvariable=summary_var, bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_MUTED,
                 font=("Consolas", 8), anchor="e").pack(side="right", padx=(8, self.CARD_PAD))

        body = self._make_card(shell)
        return shell, btn, body

    def _set_state_dot(self, color):
        canvas = getattr(self, "state_dot_canvas", None)
        if canvas is None:
            return
        try:
            canvas.itemconfigure("dot", fill=color, outline=color)
        except Exception:
            pass

    # Two-line mode-button captions: bold mode word, then hotkey + what the
    # mode does. Centralised so every configure(text=...) site stays in sync.
    MODE_BUTTON_BLURB = {
        "live": "types live",
        "batch": "paste at end",
        "transcribe": "cloud · paste",
        "transcribe_edit": "cloud · edited",
    }

    def _mode_button_text(self, kind):
        hk_key = "hotkey_live" if kind in ("live", "transcribe_edit") else "hotkey_batch"
        label_key = "live" if hk_key == "hotkey_live" else "batch"
        hk = self.config.get(hk_key, HOTKEY_LABELS[label_key].lower()).upper()
        word = {"live": "LIVE", "batch": "BATCH",
                "transcribe": "TRANSCRIBE", "transcribe_edit": "+EDIT"}[kind]
        return word + "\n" + hk + " · " + self.MODE_BUTTON_BLURB[kind]

    def _is_online_mode(self):
        """True while a cloud recording is running. self.mode carries the
        concrete strings set by _start_online_recording(), never 'online'."""
        return self.mode in ("online", "online_transcribe", "online_transcribe_edit")

    def _vu_color(self):
        if self.mode == "live":
            return COLOR_LIVE
        if self.mode == "batch":
            return COLOR_BATCH
        if self._is_online_mode():
            return COLOR_ONLINE
        return COLOR_TEAL

    def _update_device_model_labels(self):
        """Footer line of the state card: live mic + live model/backend."""
        if not hasattr(self, "device_info_var"):
            return
        try:
            dev_name = (self.device_var.get() or "").strip()
        except Exception:
            dev_name = ""
        if len(dev_name) > 26:
            dev_name = dev_name[:25].rstrip() + "…"
        self.device_info_var.set(f"Mic · {dev_name}" if dev_name else "Mic · default")

        if self.transcription_mode.get() == 'online':
            model = self.config.get('openai_transcription_model',
                                    DEFAULT_CONFIG['openai_transcription_model'])
            self.model_info_var.set(f"{model} · cloud")
        else:
            model_key = self.config.get('model_key', '')
            short = model_key.split('(')[0].strip() or model_key
            device = self.config.get('device', 'cpu')
            compute = self.config.get('compute_type', 'int8')
            self.model_info_var.set(f"{short} · {device} · {compute}")

    def _config_summary_text(self):
        model_key = self.config.get('model_key', '')
        short = model_key.split('(')[0].strip() or model_key
        device = self.config.get('device', 'cpu')
        try:
            pause = float(self.live_pause_var.get())
        except Exception:
            pause = float(self.config.get('live_pause', 0.8))
        return f"{short} · {device} · {pause:.1f}s"

    def _update_config_summary(self):
        if hasattr(self, "config_summary_var"):
            self.config_summary_var.set(self._config_summary_text())

    def setup_ui(self):
        # --- STATUS STRIP (pack first at bottom so it stays fixed) ---
        self.status_var = tk.StringVar(value="Loading...")
        self.status_stats_var = tk.StringVar(value="")
        status_bar = tk.Frame(self.root, bg=COLOR_ACCENT_BG)
        status_bar.pack(fill="x", side="bottom")
        tk.Frame(status_bar, bg=COLOR_BORDER, height=1).pack(fill="x", side="top")
        status_inner = tk.Frame(status_bar, bg=COLOR_ACCENT_BG)
        status_inner.pack(fill="x")
        tk.Label(status_inner, textvariable=self.status_var, anchor="w",
                 bg=COLOR_ACCENT_BG, fg=COLOR_FG, font=("Arial", 9)).pack(
            side="left", padx=(14, 4), pady=4)
        tk.Label(status_inner, textvariable=self.status_stats_var, anchor="e",
                 bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_FAINT, font=("Consolas", 8)).pack(
            side="right", padx=(4, 14), pady=4)

        # --- SCROLLABLE CONTENT AREA (fallback for expanded accordions) ---
        self._scroll_canvas = tk.Canvas(self.root, bg=COLOR_BG, highlightthickness=0)
        self._scroll_vsb = ttk.Scrollbar(self.root, orient="vertical",
                                          command=self._scroll_canvas.yview)
        self.content_frame = tk.Frame(self._scroll_canvas, bg=COLOR_BG)

        self.content_frame.bind("<Configure>", self._update_scroll_region)
        self._scroll_canvas_window = self._scroll_canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )
        self._scroll_canvas.configure(yscrollcommand=self._scroll_vsb.set)

        self._scroll_vsb.pack(side="right", fill="y")
        self._scroll_canvas.pack(side="left", fill="both", expand=True)

        # Keep content frame width in sync with canvas viewport
        self._scroll_canvas.bind("<Configure>",
            lambda e: self._scroll_canvas.itemconfig(self._scroll_canvas_window, width=e.width))

        # Mousewheel scrolling
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

        # --- HEADER: wordmark + three icon buttons ---
        header_frame = tk.Frame(self.content_frame, bg=COLOR_BG)
        header_frame.pack(fill="x", padx=14, pady=(10, 4))

        title_frame = tk.Frame(header_frame, bg=COLOR_BG)
        title_frame.pack(side="left")
        tk.Label(title_frame, text="neuro", bg=COLOR_BG, fg=COLOR_FG,
                 font=("Arial", 14, "bold")).pack(side="left")
        tk.Label(title_frame, text="whisper", bg=COLOR_BG, fg=COLOR_BATCH,
                 font=("Arial", 14, "bold")).pack(side="left")

        self.transcription_mode = tk.StringVar(value=self.config.get('transcription_mode', 'local'))
        self.top_var = tk.BooleanVar(value=self.config.get('always_on_top', True))

        # Configuration (rightmost), minimise-to-mini, always-on-top pin
        self.btn_header_config = ctk.CTkButton(
            header_frame, text="CFG", width=40, height=24, corner_radius=8,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=COLOR_ACCENT_BG, hover_color=COLOR_BORDER,
            border_width=1, border_color=COLOR_BORDER, text_color=COLOR_TEXT_MUTED,
            command=self._focus_configuration)
        self.btn_header_config.pack(side="right", padx=(6, 0))
        self._create_tooltip(self.btn_header_config, "Open Configuration")

        self.btn_header_mini = ctk.CTkButton(
            header_frame, text="MINI", width=44, height=24, corner_radius=8,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=COLOR_ACCENT_BG, hover_color=COLOR_BORDER,
            border_width=1, border_color=COLOR_BORDER, text_color=COLOR_TEXT_MUTED,
            command=lambda: self.root.iconify())
        self.btn_header_mini.pack(side="right", padx=(6, 0))
        self._create_tooltip(self.btn_header_mini, "Minimise to the floating mini window")

        self.btn_header_top = ctk.CTkButton(
            header_frame, text="TOP", width=40, height=24, corner_radius=8,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=COLOR_TEAL_DIM if self.top_var.get() else COLOR_ACCENT_BG,
            hover_color=COLOR_BORDER,
            border_width=1,
            border_color=COLOR_TEAL if self.top_var.get() else COLOR_BORDER,
            text_color=COLOR_TEAL if self.top_var.get() else COLOR_TEXT_MUTED,
            command=self._toggle_always_on_top)
        self.btn_header_top.pack(side="right")
        self._create_tooltip(self.btn_header_top, "Keep the window always on top")

        # --- BODY COLUMN ---
        body = tk.Frame(self.content_frame, bg=COLOR_BG)
        body.pack(fill="both", expand=True, padx=14, pady=(4, 12))

        # ==============================================================
        # 1. DICTATION STATE CARD
        # ==============================================================
        state_card = self._make_card(body)
        state_card.pack(fill="x", pady=(0, self.CARD_GAP))
        state_inner = tk.Frame(state_card, bg=COLOR_ACCENT_BG)
        state_inner.pack(fill="x", padx=self.CARD_PAD, pady=(self.CARD_PAD, 11))

        # Row 1: state dot + state word + elapsed | LOCAL/ONLINE toggle
        row1 = tk.Frame(state_inner, bg=COLOR_ACCENT_BG)
        row1.pack(fill="x")

        self.state_dot_canvas = tk.Canvas(row1, width=11, height=11, bg=COLOR_ACCENT_BG,
                                          highlightthickness=0)
        self.state_dot_canvas.create_oval(1, 1, 10, 10, fill=COLOR_IDLE,
                                          outline=COLOR_IDLE, tags="dot")
        self.state_dot_canvas.pack(side="left", padx=(0, 8))

        self.state_word_var = tk.StringVar(value="Loading")
        tk.Label(row1, textvariable=self.state_word_var, bg=COLOR_ACCENT_BG, fg=COLOR_FG,
                 font=("Arial", 12, "bold")).pack(side="left")

        self.state_elapsed_var = tk.StringVar(value="")
        tk.Label(row1, textvariable=self.state_elapsed_var, bg=COLOR_ACCENT_BG,
                 fg=COLOR_TEXT_FAINT, font=("Arial", 9)).pack(side="left", padx=(8, 0))

        mode_container = ctk.CTkFrame(row1, fg_color=COLOR_BG, corner_radius=12,
                                      border_width=1, border_color=COLOR_BORDER)
        mode_container.pack(side="right")
        mode_inner = ctk.CTkFrame(mode_container, fg_color="transparent")
        mode_inner.pack(padx=2, pady=2)

        is_local = self.transcription_mode.get() == 'local'
        self.btn_local = ctk.CTkButton(mode_inner, text="LOCAL",
                                       fg_color=COLOR_TEAL if is_local else COLOR_BG,
                                       hover_color=COLOR_TEAL,
                                       text_color=COLOR_BG if is_local else COLOR_TEXT_FAINT,
                                       font=ctk.CTkFont(size=10, weight="bold"),
                                       corner_radius=10, width=58, height=20,
                                       command=lambda: self.switch_transcription_mode('local'))
        self.btn_local.pack(side="left", padx=1)

        self.btn_online = ctk.CTkButton(mode_inner, text="ONLINE",
                                        fg_color=COLOR_ONLINE if not is_local else COLOR_BG,
                                        hover_color=COLOR_ONLINE,
                                        text_color=COLOR_FG if not is_local else COLOR_TEXT_FAINT,
                                        font=ctk.CTkFont(size=10, weight="bold"),
                                        corner_radius=10, width=62, height=20,
                                        command=lambda: self.switch_transcription_mode('online'))
        self.btn_online.pack(side="left", padx=1)

        self._create_tooltip(self.btn_local, "LOCAL: Uses your computer's hardware\nfor transcription. No internet needed.")
        self._create_tooltip(self.btn_online, "ONLINE: Uses OpenAI's cloud API.\nRequires API key and internet connection.")

        # Row 2: LIVE / BATCH as two equal-width 52px buttons
        self.toggle_container = tk.Frame(state_inner, bg=COLOR_ACCENT_BG)
        self.toggle_container.pack(fill="x", pady=(11, 0))
        self.toggle_inner = self.toggle_container


        self.btn_live = ctk.CTkButton(self.toggle_inner, text=self._mode_button_text("live"),
                                      fg_color=COLOR_BG, hover_color=COLOR_LIVE,
                                      text_color=COLOR_TEXT_MUTED,
                                      font=ctk.CTkFont(size=12, weight="bold"),
                                      corner_radius=self.ROW_RADIUS, height=52, width=190,
                                      border_width=2, border_color=COLOR_LIVE,
                                      command=self.toggle_live_mode)

        self.separator = ctk.CTkFrame(self.toggle_inner, fg_color=COLOR_ACCENT_BG,
                                      width=8, height=52)

        self.btn_batch = ctk.CTkButton(self.toggle_inner, text=self._mode_button_text("batch"),
                                       fg_color=COLOR_BG, hover_color=COLOR_BATCH,
                                       text_color=COLOR_TEXT_MUTED,
                                       font=ctk.CTkFont(size=12, weight="bold"),
                                       corner_radius=self.ROW_RADIUS, height=52, width=190,
                                       border_width=2, border_color=COLOR_BATCH,
                                       command=self.toggle_batch_mode)

        self.btn_transcribe = ctk.CTkButton(self.toggle_inner, text=self._mode_button_text("transcribe"),
                                            fg_color=COLOR_BG, hover_color=COLOR_ONLINE,
                                            text_color=COLOR_TEXT_MUTED,
                                            font=ctk.CTkFont(size=12, weight="bold"),
                                            corner_radius=self.ROW_RADIUS, height=52, width=190,
                                            border_width=2, border_color=COLOR_ONLINE,
                                            command=self.toggle_online_transcribe)

        self.btn_transcribe_edit = ctk.CTkButton(self.toggle_inner, text=self._mode_button_text("transcribe_edit"),
                                                 fg_color=COLOR_BG, hover_color=COLOR_ONLINE,
                                                 text_color=COLOR_TEXT_MUTED,
                                                 font=ctk.CTkFont(size=12, weight="bold"),
                                                 corner_radius=self.ROW_RADIUS, height=52, width=190,
                                                 border_width=2, border_color=COLOR_ONLINE,
                                                 command=self.toggle_online_transcribe_edit)

        self._update_mode_buttons()

        # Legacy aliases (older call sites expect these names)
        self.batch_badge = self.btn_batch
        self.live_badge = self.btn_live
        self.lbl_batch = self.btn_batch
        self.lbl_live = self.btn_live
        self.batch_segment = self.btn_batch
        self.live_segment = self.btn_live

        self._create_tooltip(self.btn_batch, "BATCH MODE: Records until you press again.\nTranscribes everything at once and pastes result.\nBest for longer dictation.")
        self._create_tooltip(self.btn_live, "LIVE MODE: Types as you speak in real-time.\nWords appear directly where your cursor is.\nBest for quick notes and immediate input.")
        self._create_tooltip(self.btn_transcribe, "TRANSCRIBE: Records audio and sends to\nOpenAI Whisper API for transcription.")
        self._create_tooltip(self.btn_transcribe_edit, "TRANSCRIBE+EDIT: Transcribes with OpenAI,\nthen uses GPT to clean up and copy-edit the text.")

        # Row 3: VU meter (colour follows the active mode)
        self.vu_meter = ctk.CTkProgressBar(state_inner, progress_color=COLOR_TEAL,
                                           fg_color=COLOR_BG, corner_radius=4, height=8)
        self.vu_meter.pack(fill="x", pady=(11, 0))
        self.vu_meter.set(0)

        # Row 4: live device + model footer
        foot = tk.Frame(state_inner, bg=COLOR_ACCENT_BG)
        foot.pack(fill="x", pady=(6, 0))
        self.device_info_var = tk.StringVar(value="Mic · default")
        self.model_info_var = tk.StringVar(value="")
        tk.Label(foot, textvariable=self.device_info_var, bg=COLOR_ACCENT_BG,
                 fg=COLOR_TEXT_FAINT, font=("Arial", 8)).pack(side="left")
        tk.Label(foot, textvariable=self.model_info_var, bg=COLOR_ACCENT_BG,
                 fg=COLOR_TEXT_FAINT, font=("Arial", 8)).pack(side="right")

        # ==============================================================
        # 2. LAST TRANSCRIPTION CARD
        # ==============================================================
        transcription_frame = self._make_card(body)
        transcription_frame.pack(fill="x", pady=(0, self.CARD_GAP))
        trans_inner = tk.Frame(transcription_frame, bg=COLOR_ACCENT_BG)
        trans_inner.pack(fill="x", padx=self.CARD_PAD, pady=(11, self.CARD_PAD))

        trans_header = tk.Frame(trans_inner, bg=COLOR_ACCENT_BG)
        trans_header.pack(fill="x", pady=(0, 9))

        self.btn_prev = tk.Button(trans_header, text="◀", bg=COLOR_BG, fg=COLOR_TEXT_MUTED,
                                  font=("Arial", 8), width=2, cursor="hand2",
                                  relief="flat", bd=0, highlightthickness=0,
                                  activebackground=COLOR_BORDER, activeforeground=COLOR_FG,
                                  command=lambda: self.navigate_history(-1))
        self.btn_prev.pack(side="left")

        self.history_label = tk.Label(trans_header, text="0 / 0", bg=COLOR_ACCENT_BG,
                                      fg=COLOR_TEXT_MUTED, font=("Consolas", 9))
        self.history_label.pack(side="left", padx=6)

        self.btn_next = tk.Button(trans_header, text="▶", bg=COLOR_BG, fg=COLOR_TEXT_MUTED,
                                  font=("Arial", 8), width=2, cursor="hand2",
                                  relief="flat", bd=0, highlightthickness=0,
                                  activebackground=COLOR_BORDER, activeforeground=COLOR_FG,
                                  command=lambda: self.navigate_history(1))
        self.btn_next.pack(side="left")

        # timestamp / words / latency (one meta line)
        self.word_count_label = tk.Label(trans_header, text="0 words", bg=COLOR_ACCENT_BG,
                                         fg=COLOR_TEAL, font=("Arial", 9))
        self.word_count_label.pack(side="left", padx=(10, 0))

        self.btn_copy = ctk.CTkButton(trans_header, text="Copy",
                                      fg_color=COLOR_TEAL_DIM, hover_color=COLOR_TEAL_DIM,
                                      border_width=1, border_color=COLOR_TEAL,
                                      text_color=COLOR_TEAL,
                                      font=ctk.CTkFont(size=10, weight="bold"),
                                      corner_radius=9, width=62, height=26,
                                      command=self.copy_current_transcription)
        self.btn_copy.pack(side="right")

        self.transcription_text = tk.Text(trans_inner, height=6, font=("Consolas", 11),
                                          bg=COLOR_TEXT_BG, fg=COLOR_FG, insertbackground='white',
                                          wrap="word", padx=10, pady=9, relief="flat",
                                          highlightthickness=1, highlightbackground=COLOR_BORDER,
                                          highlightcolor=COLOR_BORDER)
        self.transcription_text.pack(fill="x")
        self.transcription_text.insert("1.0", "No transcriptions yet...")
        self.transcription_text.config(state='disabled')

        # ==============================================================
        # 3. SPEECH INSIGHTS ACCORDION
        # ==============================================================
        self.insights_collapsed = tk.BooleanVar(value=self.config.get('insights_collapsed', True))
        self._insights_dirty = True
        self.insights_summary_var = tk.StringVar(value="collapsed")
        _, self.insights_toggle_btn, self.insights_body = self._make_accordion(
            body, "Speech Insights", self.insights_summary_var,
            self.toggle_insights, self.insights_collapsed.get())

        insights_top = tk.Frame(self.insights_body, bg=COLOR_ACCENT_BG)
        insights_top.pack(fill="x", padx=self.CARD_PAD, pady=(11, 8))

        # Range control (Today / 7 d / 30 d / All), top-right of the body
        range_value = self.config.get('insights_range', INSIGHTS_RANGE_DEFAULT)
        if range_value not in INSIGHTS_RANGE_OPTIONS:
            range_value = INSIGHTS_RANGE_DEFAULT
        self.insights_range_var = tk.StringVar(value=range_value)
        range_container = ctk.CTkFrame(insights_top, fg_color=COLOR_BG, corner_radius=9,
                                       border_width=1, border_color=COLOR_BORDER)
        range_container.pack(side="right")
        range_inner = ctk.CTkFrame(range_container, fg_color="transparent")
        range_inner.pack(padx=2, pady=2)
        self.insights_range_buttons = {}
        for key, short in (("Today", "Today"), ("7 days", "7 d"), ("30 days", "30 d"), ("All", "All")):
            if key not in INSIGHTS_RANGE_OPTIONS:
                continue
            btn = ctk.CTkButton(range_inner, text=short, width=40, height=18,
                                corner_radius=7, font=ctk.CTkFont(size=10, weight="bold"),
                                fg_color=COLOR_BG, hover_color=COLOR_TEAL_DIM,
                                text_color=COLOR_TEXT_FAINT,
                                command=lambda k=key: self._select_insights_range(k))
            btn.pack(side="left", padx=1)
            self.insights_range_buttons[key] = btn

        tk.Label(insights_top, text="Window", bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_MUTED,
                 font=("Arial", 9)).pack(side="left")

        # Coverage / baseline / freshness / profile notes
        notes = tk.Frame(self.insights_body, bg=COLOR_ACCENT_BG)
        notes.pack(fill="x", padx=self.CARD_PAD, pady=(0, 8))

        self.insights_coverage_var = tk.StringVar(value="Coverage: scanning transcript archive...")
        tk.Label(notes, textvariable=self.insights_coverage_var, bg=COLOR_ACCENT_BG,
                 fg=COLOR_TEXT_MUTED, font=("Arial", 8), anchor="w", justify="left",
                 wraplength=420).pack(fill="x")

        self.insights_ai_age_var = tk.StringVar(value="AI insights: not crunched yet.")
        tk.Label(notes, textvariable=self.insights_ai_age_var, bg=COLOR_ACCENT_BG,
                 fg=COLOR_TEXT_MUTED, font=("Arial", 8), anchor="w", justify="left",
                 wraplength=420).pack(fill="x")

        self.insights_peer_var = tk.StringVar(
            value=f"Baseline: your trailing {PERSONAL_BASELINE_DAYS}-day average "
                  f"(falls back to reference bands under {PERSONAL_BASELINE_MIN_DAYS} days of data)."
        )
        tk.Label(notes, textvariable=self.insights_peer_var, bg=COLOR_ACCENT_BG,
                 fg=COLOR_TEXT_FAINT, font=("Arial", 8), anchor="w", justify="left",
                 wraplength=420).pack(fill="x")

        self.insights_profile_summary_var = tk.StringVar(
            value="Personal profile vs peer midpoint: waiting for enough data."
        )
        tk.Label(notes, textvariable=self.insights_profile_summary_var, bg=COLOR_ACCENT_BG,
                 fg=COLOR_TEAL, font=("Arial", 9, "bold"), anchor="w", justify="left",
                 wraplength=420).pack(fill="x", pady=(4, 0))

        # All-time totals + both histograms (moved out of the main column)
        stats_card = ctk.CTkFrame(self.insights_body, fg_color=COLOR_BG, corner_radius=10,
                                  border_width=1, border_color=COLOR_BORDER)
        stats_card.pack(fill="x", padx=self.CARD_PAD, pady=(0, 8))

        alltime_card = tk.Frame(stats_card, bg=COLOR_BG)
        alltime_card.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(alltime_card, text="ALL-TIME", bg=COLOR_BG, fg=COLOR_TEXT_MUTED,
                 font=("Arial", 8, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
        for col, (cap, attr) in enumerate((
                ("Words", "lbl_alltime_words"),
                ("Spoken", "lbl_alltime_duration"),
                ("Saved", "lbl_alltime_time"),
                ("Speed", "lbl_alltime_speed"))):
            tk.Label(alltime_card, text=cap, bg=COLOR_BG, fg=COLOR_TEXT_FAINT,
                     font=("Arial", 8)).grid(row=1, column=col, sticky="w", padx=(0, 14))
            lbl = tk.Label(alltime_card, text="0", bg=COLOR_BG, fg=COLOR_TEAL,
                           font=("Arial", 13, "bold"))
            lbl.grid(row=2, column=col, sticky="w", padx=(0, 14))
            setattr(self, attr, lbl)

        self.histogram_canvas_today = tk.Canvas(stats_card, height=86, bg=COLOR_BG,
                                                highlightthickness=0)
        self.histogram_canvas_today.pack(fill="x", padx=10, pady=(4, 2))
        self.histogram_bars_today = []

        self.histogram_canvas_alltime = tk.Canvas(stats_card, height=86, bg=COLOR_BG,
                                                  highlightthickness=0)
        self.histogram_canvas_alltime.pack(fill="x", padx=10, pady=(2, 8))
        self.histogram_bars_alltime = []

        # Tab chips (replaces ttk.Notebook - ten ttk tabs overflow at 480px)
        self.insight_tab_names = [
            "Pace",
            "Pauses",
            "Clarity",
            "Lexical",
            "Interventions",
            "Timeline",
            "Mood",
            "Activity",
            "Profile",
            "Briefs",
        ]
        chip_row = tk.Frame(self.insights_body, bg=COLOR_ACCENT_BG)
        chip_row.pack(fill="x", padx=self.CARD_PAD, pady=(0, 8))
        self.insight_chip_buttons = []
        chip_line = None
        for i, name in enumerate(self.insight_tab_names):
            if i % 4 == 0:
                chip_line = tk.Frame(chip_row, bg=COLOR_ACCENT_BG)
                chip_line.pack(fill="x", pady=1)
            # Content-width chips, not expand/fill: the last line only holds
            # two chips and would otherwise stretch them to half the window.
            chip = ctk.CTkButton(chip_line, text=name, height=22, corner_radius=8,
                                 width=max(52, 8 * len(name) + 16),
                                 font=ctk.CTkFont(size=10),
                                 fg_color=COLOR_BG, hover_color=COLOR_TEAL_DIM,
                                 border_width=1, border_color=COLOR_BORDER,
                                 text_color=COLOR_TEXT_MUTED,
                                 command=lambda idx=i: self._select_insight_tab(idx))
            chip.pack(side="left", padx=(0, 5))
            self.insight_chip_buttons.append(chip)

        # One container; only the selected tab frame is packed.
        self.insights_tab_container = tk.Frame(self.insights_body, bg=COLOR_ACCENT_BG,
                                               highlightthickness=1,
                                               highlightbackground=COLOR_BORDER)
        self.insights_tab_container.pack(fill="x", padx=self.CARD_PAD, pady=(0, self.CARD_PAD))

        self.pace_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.pause_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.clarity_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.lexical_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.intervention_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.ai_timeline_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.ai_mood_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.activity_matrix_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.speaker_profile_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.daily_brief_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)

        self.insight_tab_frames = [
            self.pace_tab, self.pause_tab, self.clarity_tab, self.lexical_tab,
            self.intervention_tab, self.ai_timeline_tab, self.ai_mood_tab,
            self.activity_matrix_tab, self.speaker_profile_tab, self.daily_brief_tab,
        ]

        # Tab bodies are built lazily on first selection (see
        # _ensure_insight_tab_built) so setup_ui only creates the shells.
        self._insight_tab_builders = [
            self._setup_pace_tab,
            self._setup_pause_tab,
            self._setup_clarity_tab,
            self._setup_lexical_tab,
            self._setup_intervention_tab,
            self._setup_ai_timeline_tab,
            self._setup_ai_mood_tab,
            self._setup_activity_matrix_tab,
            self._setup_speaker_profile_tab,
            self._setup_daily_brief_tab,
        ]
        self._insight_tabs_built = set()
        self.insights_tab_status = tk.StringVar(value="1/10")
        self.insight_tab_index = 0
        self.insight_tab_frames[0].pack(fill="x")
        self._update_chip_styles()
        self._update_range_button_styles()
        self._update_insight_tab_status()

        if not self.insights_collapsed.get():
            self.insights_body.pack(fill="x")
            # Restored expanded from config: build the first tab body now, or
            # the startup refresh_insights_tabs() early-returns on every tab
            # and the panel renders empty until the user clicks a chip.
            self._ensure_insight_tab_built(0)

        # ==============================================================
        # 4. CONFIGURATION ACCORDION
        # ==============================================================
        is_first_run = 'config_collapsed' not in self.config
        self.config_collapsed = tk.BooleanVar(value=not is_first_run)
        self.config_summary_var = tk.StringVar(value="")
        _, self.settings_toggle_btn, self.settings_content_card = self._make_accordion(
            body, "Configuration", self.config_summary_var,
            self.toggle_settings, self.config_collapsed.get())

        self.settings_content = ttk.Frame(self.settings_content_card)

        r1 = ttk.Frame(self.settings_content)
        r1.pack(fill="x", padx=5, pady=5)
        ttk.Label(r1, text="Model:").pack(side="left")

        # --- DYNAMIC MODEL LIST ---
        self.model_map_display = {}
        display_values = []
        current_display = self.config['model_key']

        for friendly in MODEL_MAP.keys():
            is_down = self.is_model_downloaded(friendly)
            prefix = "" if is_down else "[get] "
            display_name = f"{prefix}{friendly}"
            self.model_map_display[display_name] = friendly
            display_values.append(display_name)

            if friendly == self.config['model_key']:
                current_display = display_name

        self.model_var = tk.StringVar(value=current_display)
        self.combo = ttk.Combobox(r1, textvariable=self.model_var, values=display_values, width=28, state="readonly")
        self.combo.pack(side="left", padx=5)
        self.combo.bind("<<ComboboxSelected>>", self.on_model_select)

        # Recovery affordance after a failed load (the error dialog names it).
        self.btn_reload_model = tk.Button(
            r1, text="Reload Model",
            bg=COLOR_ACCENT_BG, fg=COLOR_TEAL,
            font=("Arial", 8, "bold"),
            cursor="hand2", relief="groove",
            command=self._manual_model_reload,
        )
        self.btn_reload_model.pack(side="left", padx=(6, 0))
        self._create_tooltip(self.btn_reload_model,
                             "Retry loading the speech model (e.g. after the first\n"
                             "download failed because there was no internet).")

        r1b = ttk.Frame(self.settings_content)
        r1b.pack(fill="x", padx=5, pady=5)
        ttk.Label(r1b, text="Pause (s):").pack(side="left")
        # Question mark with tooltip for pause explanation
        pause_help = ttk.Label(r1b, text="?", foreground=COLOR_TEXT_FAINT,
                               font=("Arial", 9, "bold"), cursor="question_arrow")
        pause_help.pack(side="left", padx=(0, 2))
        self._create_tooltip(pause_help, "Pause Threshold (seconds):\nHow long to wait after you stop speaking\nbefore processing the audio in LIVE mode.\n\nShorter = faster response but may cut you off\nLonger = waits for natural pauses")

        self.live_pause_var = tk.DoubleVar(value=self.config['live_pause'])
        pause_spin = ttk.Spinbox(
            r1b,
            from_=LIVE_PAUSE_RANGE["min"],
            to=LIVE_PAUSE_RANGE["max"],
            increment=LIVE_PAUSE_RANGE["step"],
            textvariable=self.live_pause_var,
            width=5,
        )
        pause_spin.pack(side="left", padx=5)
        # Auto-save on pause change
        self.live_pause_var.trace_add("write", self.on_setting_change)

        cb = ttk.Checkbutton(r1b, text="On Top", variable=self.top_var, command=self.on_top_change)
        cb.pack(side="left", padx=10)

        r2 = ttk.Frame(self.settings_content)
        r2.pack(fill="x", padx=5, pady=5)
        ttk.Label(r2, text="Mic:").pack(side="left")

        # A machine with no sound card (or a broken PortAudio install) must not
        # take the whole window down - query_devices() raises there.
        try:
            devices = list(sd.query_devices())
        except Exception as e:
            devices = []
            log.error("Could not enumerate audio devices: %s", e)
            self.log_internal(f"Could not enumerate audio devices: {e}")
        input_devices = [d['name'] for d in devices if d.get('max_input_channels', 0) > 0]

        # A saved input_device index only survives if it still exists AND is
        # still an input; otherwise fall back to the system default (None).
        current_dev_id = self.config.get('input_device')
        if current_dev_id is not None:
            valid = (
                isinstance(current_dev_id, int)
                and 0 <= current_dev_id < len(devices)
                and devices[current_dev_id].get('max_input_channels', 0) > 0
            )
            if not valid:
                log.warning("Saved input device %r is gone - using the system default.", current_dev_id)
                self.log_internal(f"Saved microphone (device {current_dev_id}) is gone - using the system default.")
                self.config['input_device'] = None
                current_dev_id = None
                self._save_local_config()

        default_dev = ""
        if current_dev_id is not None:
            default_dev = devices[current_dev_id]['name']
        elif input_devices:
            default_dev = input_devices[0]

        if not input_devices:
            self.log_internal("No microphone found - connect an input device and press Rebind.")
            self.device_var = tk.StringVar(value="No microphone found")
            mic_combo = ttk.Combobox(r2, textvariable=self.device_var, values=[], state="disabled")
        else:
            self.device_var = tk.StringVar(value=default_dev)
            mic_combo = ttk.Combobox(r2, textvariable=self.device_var, values=input_devices, state="readonly")
        mic_combo.pack(side="left", fill="x", expand=True, padx=5)
        mic_combo.bind("<<ComboboxSelected>>", self.on_mic_change)

        # --- BACKEND/ACCELERATION SELECTOR ---
        r_backend = ttk.Frame(self.settings_content)
        r_backend.pack(fill="x", padx=5, pady=5)

        ttk.Label(r_backend, text="Acceleration:").pack(side="left")

        # Backend options: Auto detects best, or force specific
        backend_options = BACKEND_OPTIONS
        backend_map = BACKEND_DISPLAY_TO_INTERNAL
        self.backend_map_display = {v: k for k, v in backend_map.items()}

        current_backend = self.config.get('backend', 'auto')
        current_backend_display = self.backend_map_display.get(current_backend, "Auto")

        self.backend_var = tk.StringVar(value=current_backend_display)
        backend_combo = ttk.Combobox(r_backend, textvariable=self.backend_var,
                                      values=backend_options, width=18, state="readonly")
        backend_combo.pack(side="left", padx=5)
        backend_combo.bind("<<ComboboxSelected>>", self.on_backend_change)

        self._create_tooltip(backend_combo, "Acceleration Mode:\n• Auto: Detects best option\n• Parakeet: NVIDIA NeMo (multi-lang)\n• OpenVINO: Intel GPU/NPU\n• CUDA: NVIDIA GPUs\n• CPU only: Works everywhere")

        # Show current detected backend
        self.backend_status_label = ttk.Label(r_backend, text="", foreground=COLOR_TEAL)
        self.backend_status_label.pack(side="left", padx=10)

        # Update status to show current config backend
        current_be = self.config.get('backend', 'faster-whisper')
        current_dev = self.config.get('device', 'cpu')
        self.backend_status_label.config(text=f"[{current_be} / {current_dev}]")

        # --- HOTKEY CONFIGURATION ROW ---
        r3 = ttk.Frame(self.settings_content)
        r3.pack(fill="x", padx=5, pady=5)

        ttk.Label(r3, text="Batch:").pack(side="left")
        self.batch_hotkey_var = tk.StringVar(value=self.config['hotkey_batch'])
        self.batch_hotkey_btn = tk.Button(r3, textvariable=self.batch_hotkey_var,
                                          bg=COLOR_ACCENT_BG, fg=COLOR_TEAL,
                                          font=("Arial", 9, "bold"), width=12,
                                          cursor="hand2", relief="groove",
                                          command=lambda: self.start_hotkey_capture('batch'))
        self.batch_hotkey_btn.pack(side="left", padx=2)
        self._create_tooltip(self.batch_hotkey_btn, "Click and press your desired shortcut for BATCH mode")

        ttk.Label(r3, text="Live:").pack(side="left", padx=(10, 0))
        self.live_hotkey_var = tk.StringVar(value=self.config['hotkey_live'])
        self.live_hotkey_btn = tk.Button(r3, textvariable=self.live_hotkey_var,
                                         bg=COLOR_ACCENT_BG, fg=COLOR_TEAL,
                                         font=("Arial", 9, "bold"), width=12,
                                         cursor="hand2", relief="groove",
                                         command=lambda: self.start_hotkey_capture('live'))
        self.live_hotkey_btn.pack(side="left", padx=2)
        self._create_tooltip(self.live_hotkey_btn, "Click and press your desired shortcut for LIVE mode")

        # Manual recovery action when OS hooks become stale.
        self.refresh_hotkeys_btn = tk.Button(r3, text="Rebind",
                                              bg=COLOR_ACCENT_BG, fg=COLOR_TEAL,
                                              font=("Arial", 9, "bold"), width=8,
                                              cursor="hand2", relief="groove",
                                              command=self._manual_hotkey_refresh)
        self.refresh_hotkeys_btn.pack(side="left", padx=(10, 2))
        self._create_tooltip(self.refresh_hotkeys_btn, "Rebind global hotkeys (use if hotkeys stop working)")

        r3b = ttk.Frame(self.settings_content)
        r3b.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(r3b, text="Health:").pack(side="left")
        self.hotkey_health_var = tk.StringVar(value="Initializing...")
        self.hotkey_health_label = tk.Label(
            r3b,
            textvariable=self.hotkey_health_var,
            bg=COLOR_BG,
            fg=COLOR_IDLE,
            font=("Arial", 9, "bold"),
            anchor="w",
        )
        self.hotkey_health_label.pack(side="left", padx=(4, 2))
        self._create_tooltip(self.hotkey_health_label, "Global hotkey hook health")

        self.hotkey_auto_rebinds_var = tk.StringVar(value="Auto-Rebinds: 0")
        self.hotkey_auto_rebinds_label = tk.Label(
            r3b,
            textvariable=self.hotkey_auto_rebinds_var,
            bg=COLOR_BG,
            fg=COLOR_TEXT_MUTED,
            font=("Arial", 9),
            anchor="w",
        )
        self.hotkey_auto_rebinds_label.pack(side="left", padx=(10, 2))
        self._create_tooltip(self.hotkey_auto_rebinds_label, "Automatic hotkey rebind attempts since startup")
        self._set_hotkey_auto_rebind_indicator()

        # State for hotkey capture
        self.capturing_hotkey = None  # 'batch' or 'live' or None

        # --- OPENAI SETTINGS (for Online mode) ---
        openai_frame = ctk.CTkFrame(self.settings_content, fg_color=COLOR_BG, corner_radius=10,
                                    border_width=1, border_color=COLOR_BORDER)
        openai_frame.pack(fill="x", padx=5, pady=(10, 5))

        openai_header = ctk.CTkLabel(openai_frame, text="OpenAI Settings (Online Mode)", text_color=COLOR_FG,
                                     font=ctk.CTkFont(size=11, weight="bold"))
        openai_header.pack(anchor="w", padx=10, pady=(8, 5))

        # API Key row
        api_row = ttk.Frame(openai_frame)
        api_row.pack(fill="x", padx=10, pady=5)

        ttk.Label(api_row, text="API Key:").pack(side="left")
        self.api_key_var = tk.StringVar(value=self.config.get('openai_api_key', ''))
        self.api_key_entry = ttk.Entry(api_row, textvariable=self.api_key_var, width=24, show="•")
        self.api_key_entry.pack(side="left", padx=5)

        # Show/Hide toggle
        self.api_key_visible = tk.BooleanVar(value=False)
        def toggle_api_visibility():
            if self.api_key_visible.get():
                self.api_key_entry.config(show="")
            else:
                self.api_key_entry.config(show="•")

        show_btn = tk.Button(api_row, text="Show", bg=COLOR_ACCENT_BG, fg=COLOR_FG, font=("Arial", 8),
                            cursor="hand2", relief="flat",
                            command=lambda: (self.api_key_visible.set(not self.api_key_visible.get()), toggle_api_visibility()))
        show_btn.pack(side="left", padx=2)
        self._create_tooltip(show_btn, "Show/hide API key")

        save_key_btn = tk.Button(api_row, text="Save", bg=COLOR_TEAL, fg=COLOR_BG, font=("Arial", 9, "bold"),
                                cursor="hand2", command=self._save_openai_settings)
        save_key_btn.pack(side="left", padx=5)

        test_btn = tk.Button(api_row, text="Test", bg=COLOR_ACCENT_BG, fg=COLOR_FG, font=("Arial", 9),
                            cursor="hand2", command=self._test_openai_connection)
        test_btn.pack(side="left", padx=2)

        # Model selection row
        model_row = ttk.Frame(openai_frame)
        model_row.pack(fill="x", padx=10, pady=5)

        ttk.Label(model_row, text="Transcription:").pack(side="left")
        self.openai_trans_model_var = tk.StringVar(
            value=self.config.get('openai_transcription_model', DEFAULT_CONFIG['openai_transcription_model'])
        )
        trans_model_combo = ttk.Combobox(model_row, textvariable=self.openai_trans_model_var,
                                         values=OPENAI_TRANSCRIPTION_MODEL_OPTIONS, width=16, state="readonly")
        trans_model_combo.pack(side="left", padx=5)
        trans_model_combo.bind("<<ComboboxSelected>>", lambda e: self._auto_save_openai_settings())

        ttk.Label(model_row, text="Lang:").pack(side="left", padx=(8, 0))
        self.openai_lang_var = tk.StringVar(value=self.config.get('openai_language', DEFAULT_CONFIG['openai_language']))
        lang_combo = ttk.Combobox(model_row, textvariable=self.openai_lang_var,
                                  values=["auto", "en", "de", "es", "fr", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"],
                                  width=6, state="readonly")
        lang_combo.pack(side="left", padx=5)
        lang_combo.bind("<<ComboboxSelected>>", lambda e: self._auto_save_openai_settings())

        model_row2 = ttk.Frame(openai_frame)
        model_row2.pack(fill="x", padx=10, pady=5)
        ttk.Label(model_row2, text="Edit Model:").pack(side="left")
        self.openai_edit_model_var = tk.StringVar(
            value=self.config.get('openai_edit_model', DEFAULT_CONFIG['openai_edit_model'])
        )
        edit_model_combo = ttk.Combobox(model_row2, textvariable=self.openai_edit_model_var,
                                        values=OPENAI_EDIT_MODEL_OPTIONS, width=15, state="readonly")
        edit_model_combo.pack(side="left", padx=5)
        edit_model_combo.bind("<<ComboboxSelected>>", lambda e: self._auto_save_openai_settings())

        # Edit prompt row
        prompt_row = ttk.Frame(openai_frame)
        prompt_row.pack(fill="x", padx=10, pady=(5, 10))

        ttk.Label(prompt_row, text="Edit Prompt:").pack(side="left")
        prompt_help = ttk.Label(prompt_row, text="?", foreground=COLOR_TEXT_FAINT,
                               font=("Arial", 9, "bold"), cursor="question_arrow")
        prompt_help.pack(side="left", padx=(0, 5))
        self._create_tooltip(prompt_help, "Custom system prompt for GPT editing.\nLeave empty to use default prompt.\nThe transcribed text will be appended to this prompt.")

        self.openai_prompt_var = tk.StringVar(value=self.config.get('openai_edit_prompt', ''))
        prompt_entry = ttk.Entry(prompt_row, textvariable=self.openai_prompt_var, width=30)
        prompt_entry.pack(side="left", fill="x", expand=True, padx=5)

        # --- DOWNLOAD PROGRESS ---
        self.progress_frame = ttk.Frame(self.settings_content)
        self.progress_frame.pack(fill="x", padx=5, pady=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", side="left", expand=True)
        self.progress_label = ttk.Label(self.progress_frame, text="0%", width=6, anchor="e", font=("Arial", 8))
        self.progress_label.pack(side="right", padx=5)
        self.progress_frame.pack_forget()  # Hide by default

        # --- INSTRUCTIONS / HELP SECTION ---
        help_frame = ctk.CTkFrame(self.settings_content, fg_color=COLOR_BG, corner_radius=10,
                                  border_width=1, border_color=COLOR_BORDER)
        help_frame.pack(fill="x", padx=5, pady=(10, 5))

        help_header = ctk.CTkLabel(help_frame, text="Instructions", text_color=COLOR_FG,
                                   font=ctk.CTkFont(size=11, weight="bold"))
        help_header.pack(anchor="w", padx=10, pady=(8, 5))

        # Build help text with current hotkeys
        batch_hk = self.config['hotkey_batch'].upper()
        live_hk = self.config['hotkey_live'].upper()
        help_text = f"""How to Use:
• Press {batch_hk} for BATCH mode - records until pressed again, then transcribes all at once
• Press {live_hk} for LIVE mode - types words as you speak in real-time
• Hotkeys are customizable above - click the button and press your desired shortcut
• Use the mini floating window (drag-able) for quick access

Troubleshooting:
• Model not loading? Try a smaller model (Medium or Small)
• Slow transcription? Use 'cuda' device if you have an NVIDIA GPU
• No audio? Check your microphone selection above
• First run downloads the model (~1-3GB) - this is normal"""

        help_label = ctk.CTkLabel(help_frame, text=help_text, text_color=COLOR_TEXT_MUTED,
                                  font=ctk.CTkFont(size=10), justify="left", anchor="w")
        help_label.pack(anchor="w", padx=10, pady=(0, 10))

        # Branding footer (moved out of the header)
        branding_frame = tk.Frame(self.settings_content, bg=COLOR_BG)
        branding_frame.pack(fill="x", padx=5, pady=(6, 4))
        self.author_label = tk.Label(branding_frame, text="Created by ", bg=COLOR_BG,
                                     fg=COLOR_TEXT_FAINT, font=("Arial", 9))
        self.author_label.pack(side="left")
        self.author_link = tk.Label(branding_frame, text="DR.M", bg=COLOR_BG,
                                    fg=COLOR_TEAL, font=("Arial", 9, "bold"), cursor="hand2")
        self.author_link.pack(side="left")
        self.author_link.bind("<Button-1>", lambda e: webbrowser.open(AUTHOR_LINK_URL))

        if not self.config_collapsed.get():
            self.settings_content_card.pack(fill="x")
            self.settings_content.pack(fill="both", expand=True, padx=8, pady=8)

        # Save the collapsed state for next session
        if is_first_run:
            self.config['config_collapsed'] = True
            self._persist_config()

        # ==============================================================
        # 5. SYSTEM LOG ACCORDION (ScrolledText built lazily)
        # ==============================================================
        self.log_collapsed = tk.BooleanVar(value=True)
        self.log_summary_var = tk.StringVar(value="no warnings · 0 lines")
        self.text_area = None
        # 300 == the line cap _append_log_line() trims the Text widget back to,
        # so a first expand can never exceed the documented cap.
        self._log_buffer = deque(maxlen=300)
        self._log_line_count = 0
        self._log_warn_count = 0
        _, self.log_toggle_btn, self.log_body = self._make_accordion(
            body, "System Log", self.log_summary_var, self.toggle_log, True)

        # --- final label priming ---
        self._update_device_model_labels()
        self._update_config_summary()
        self._update_state_card()

    # ------------------------------------------------------------------
    # Insight tab skeleton: intro, metric cell, band track, chart, fine print
    # ------------------------------------------------------------------
    INSIGHT_BAND_METRICS = {
        "pace": "pace_avg_words_entry",
        "pause": "pause_avg_seconds",
        "clarity": "clarity_fillers_per_100",
        "lexical": "lexical_diversity_percent",
        "intervention": "intervention_open_question_per_100",
    }

    # Known deviation from the InsightsExpanded artboard: only the five
    # heuristic tabs (Pace/Pauses/Clarity/Lexical/Interventions) use this
    # skeleton. Timeline, Mood, Activity, Profile and Briefs keep their own
    # layouts because none of them owns a single scalar metric with a
    # comparison band, and inventing one would fabricate data. The metric row
    # is likewise one cell rather than the artboard's 3-up.
    def _build_metric_tab(self, parent, prefix, label, intro,
                          summary_var, detail_var, peer_var, canvas_height=110):
        """Shared body for every heuristic insight tab."""
        tk.Label(parent, text=intro, bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_MUTED,
                 font=("Arial", 8), anchor="w", justify="left",
                 wraplength=400).pack(fill="x", padx=10, pady=(8, 6))

        cell = ctk.CTkFrame(parent, fg_color=COLOR_BG, corner_radius=10,
                            border_width=1, border_color=COLOR_BORDER)
        cell.pack(fill="x", padx=10)
        tk.Label(cell, text=label.upper(), bg=COLOR_BG, fg=COLOR_TEXT_MUTED,
                 font=("Arial", 8)).pack(anchor="w", padx=10, pady=(8, 0))
        tk.Label(cell, textvariable=summary_var, bg=COLOR_BG, fg=COLOR_TEAL,
                 font=("Arial", 12, "bold"), anchor="w", justify="left",
                 wraplength=370).pack(fill="x", padx=10)
        tk.Label(cell, textvariable=detail_var, bg=COLOR_BG, fg=COLOR_TEXT_FAINT,
                 font=("Arial", 8), anchor="w", justify="left",
                 wraplength=370).pack(fill="x", padx=10, pady=(0, 8))

        band = tk.Canvas(parent, height=12, bg=COLOR_ACCENT_BG, highlightthickness=0)
        band.pack(fill="x", padx=10, pady=(7, 7))
        setattr(self, f"{prefix}_band_canvas", band)

        canvas = tk.Canvas(parent, height=canvas_height, bg=COLOR_TEXT_BG,
                           highlightthickness=1, highlightbackground=COLOR_BORDER)
        canvas.pack(fill="x", padx=10, pady=(0, 6))

        tk.Label(parent, textvariable=peer_var, bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_FAINT,
                 font=("Arial", 8), anchor="w", justify="left",
                 wraplength=400).pack(fill="x", padx=10, pady=(0, 8))
        return canvas

    def _draw_metric_band(self, canvas, value, metric_key):
        """10px track: baseline/reference low-high filled COLOR_TEAL_DIM,
        the current value as a 3px teal marker."""
        canvas.delete("all")
        try:
            w = canvas.winfo_width()
        except Exception:
            w = 0
        if w < 20:
            w = 400
        try:
            _status, low, high = self._metric_band_status(value, metric_key)
        except Exception:
            return
        canvas.create_rectangle(0, 1, w, 11, fill=COLOR_BG, outline=COLOR_BORDER)
        finite = all(v == v and abs(v) != float("inf") for v in (low, high, value))
        if not finite or high <= low:
            return
        span = high - low
        d0, d1 = low - span * 0.6, high + span * 0.6

        def px(v):
            return max(0.0, min(float(w), (v - d0) / (d1 - d0) * w))

        canvas.create_rectangle(px(low), 1, px(high), 11,
                                fill=COLOR_TEAL_DIM, outline=COLOR_TEAL_DIM)
        x = px(value)
        canvas.create_rectangle(x - 1.5, 0, x + 1.5, 12,
                                fill=COLOR_TEAL, outline=COLOR_TEAL)

    def _update_metric_bands(self, snapshot):
        for prefix, key in self.INSIGHT_BAND_METRICS.items():
            canvas = getattr(self, f"{prefix}_band_canvas", None)
            if canvas is None or key not in snapshot:
                continue
            self._draw_metric_band(canvas, snapshot[key], key)

    def _setup_pace_tab(self):
        self.pace_summary_var = tk.StringVar(value="No pace data yet")
        self.pace_detail_var = tk.StringVar(value="Speak a bit and this will populate.")
        self.pace_peer_var = tk.StringVar(value="Comparison pending.")
        self.pace_canvas = self._build_metric_tab(
            self.pace_tab, "pace", "Words / entry",
            "Pace Stability: tracks how your words per entry vary over recent sessions.",
            self.pace_summary_var, self.pace_detail_var, self.pace_peer_var)

    def _setup_pause_tab(self):
        self.pause_summary_var = tk.StringVar(value="No pause data yet")
        self.pause_detail_var = tk.StringVar(value="Needs consecutive transcribed entries.")
        self.pause_peer_var = tk.StringVar(value="Comparison pending.")
        self.pause_canvas = self._build_metric_tab(
            self.pause_tab, "pause", "Avg pause",
            "Pause Profile: estimates short/medium/long pauses from timestamp gaps between entries.",
            self.pause_summary_var, self.pause_detail_var, self.pause_peer_var)

    def _setup_clarity_tab(self):
        self.clarity_summary_var = tk.StringVar(value="No clarity data yet")
        self.clarity_detail_var = tk.StringVar(value="Detected from the full transcript archive.")
        self.clarity_peer_var = tk.StringVar(value="Comparison pending.")
        self.clarity_canvas = self._build_metric_tab(
            self.clarity_tab, "clarity", "Fillers / 100 w",
            "Clarity Signals: filler words, hedges, and immediate repetition markers.",
            self.clarity_summary_var, self.clarity_detail_var, self.clarity_peer_var)

    def _setup_lexical_tab(self):
        self.lexical_summary_var = tk.StringVar(value="No lexical data yet")
        self.lexical_detail_var = tk.StringVar(value="Uses transcript vocabulary from all archived sessions.")
        self.lexical_peer_var = tk.StringVar(value="Comparison pending.")
        self.lexical_canvas = self._build_metric_tab(
            self.lexical_tab, "lexical", "Diversity",
            "Lexical Diversity: vocabulary range, unique-word ratio, and frequent content words.",
            self.lexical_summary_var, self.lexical_detail_var, self.lexical_peer_var)

    def _setup_intervention_tab(self):
        self.intervention_summary_var = tk.StringVar(value="No intervention data yet")
        self.intervention_detail_var = tk.StringVar(value="Heuristic only; useful for trends, not diagnosis.")
        self.intervention_peer_var = tk.StringVar(value="Comparison pending.")
        self.intervention_canvas = self._build_metric_tab(
            self.intervention_tab, "intervention", "Open questions / 100 w",
            "Intervention Mix: heuristic counts for open questions, reflections, affirmations, and summaries.",
            self.intervention_summary_var, self.intervention_detail_var,
            self.intervention_peer_var)

    def _setup_ai_timeline_tab(self):
        # Inline notice: this tab does nothing without a key, so say so here
        # instead of only failing when the user clicks 'Crunch History'.
        self.ai_key_notice_frame = ctk.CTkFrame(
            self.ai_timeline_tab, fg_color=COLOR_BG, corner_radius=8,
            border_width=1, border_color=COLOR_BORDER)
        ctk.CTkLabel(
            self.ai_key_notice_frame,
            text="AI Insights needs an OpenAI API key. Add it in Configuration.",
            text_color=COLOR_TEXT_MUTED, font=ctk.CTkFont(size=11),
            anchor="w", justify="left", wraplength=280,
        ).pack(side="left", fill="x", expand=True, padx=(10, 6), pady=8)
        ctk.CTkButton(
            self.ai_key_notice_frame, text="Add key", width=76, height=24,
            corner_radius=8, fg_color=COLOR_TEAL, hover_color=COLOR_TEAL_DIM,
            text_color=COLOR_BG, font=ctk.CTkFont(size=11, weight="bold"),
            command=self._focus_api_key,
        ).pack(side="right", padx=(0, 10), pady=8)
        self.ai_key_notice_frame.pack(fill="x", padx=10, pady=(8, 0))

        self.ai_timeline_desc_label = tk.Label(
            self.ai_timeline_tab,
            text="AI Timeline: Tracks shifting focuses, mood, projects, and themes over days.",
            bg=COLOR_ACCENT_BG,
            fg=COLOR_TEXT_MUTED,
            font=("Arial", 8),
            anchor="w",
            justify="left",
            wraplength=400,
        )
        self.ai_timeline_desc_label.pack(fill="x", padx=10, pady=(8, 4))
        self._update_api_key_notice()
        
        self.ai_timeline_summary_var = tk.StringVar(value="Waiting to evaluate history...")
        self.ai_timeline_detail_var = tk.StringVar(value="Click 'Crunch History' to backfill daily summaries using OpenAI.")
        
        header_frame = tk.Frame(self.ai_timeline_tab, bg=COLOR_ACCENT_BG)
        header_frame.pack(fill="x", padx=10, pady=0)
        
        tk.Label(header_frame, textvariable=self.ai_timeline_summary_var, bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, font=("Arial", 10, "bold"), anchor="w").pack(side="left")
        
        self.ai_timeline_temporal_var = tk.StringVar(value="Last 30 Days")
        filter_options = ["Last 7 Days", "Last 30 Days", "Previous Month", "All Time"]
        self.ai_timeline_dropdown = tk.OptionMenu(header_frame, self.ai_timeline_temporal_var, *filter_options, command=lambda _: self.refresh_insights_tabs())
        self.ai_timeline_dropdown.config(bg=COLOR_BG, fg=COLOR_FG, font=("Arial", 9), highlightthickness=1)
        self.ai_timeline_dropdown.pack(side="left", padx=15)
        
        self.btn_crunch_history = tk.Button(
            header_frame,
            text="Crunch History",
            bg=COLOR_ACCENT_BG, fg=COLOR_ONLINE,
            font=("Arial", 8, "bold"), width=12,
            cursor="hand2", relief="groove",
            command=self._trigger_ai_history_crunch
        )
        self.btn_crunch_history.pack(side="right")

        # Explicit consent for the background upload of transcript text.
        self.ai_auto_var = tk.BooleanVar(value=bool(self.config.get('ai_insights_auto', False)))
        auto_cb = tk.Checkbutton(
            self.ai_timeline_tab,
            text="Auto-analyse my transcripts with OpenAI (sends transcript text to OpenAI)",
            variable=self.ai_auto_var,
            command=self._on_ai_auto_change,
            bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_MUTED,
            activebackground=COLOR_ACCENT_BG, activeforeground=COLOR_FG,
            selectcolor=COLOR_BG, font=("Arial", 8),
            anchor="w", justify="left", wraplength=400,
        )
        auto_cb.pack(fill="x", padx=10, pady=(2, 0))

        tk.Label(self.ai_timeline_tab, textvariable=self.ai_timeline_detail_var, bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_FAINT, font=("Arial", 8), anchor="w", justify="left", wraplength=400).pack(fill="x", padx=10, pady=(4, 6))
        
        self.ai_timeline_canvas = tk.Canvas(self.ai_timeline_tab, height=150, bg=COLOR_TEXT_BG, highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.ai_timeline_canvas.pack(fill="x", padx=10, pady=(0, 8))

    def _on_ai_auto_change(self):
        """Persist the AI-insights opt-in checkbox."""
        self.config['ai_insights_auto'] = bool(self.ai_auto_var.get())
        self._persist_config()
        self.log_internal(
            "Automatic OpenAI transcript analysis "
            + ("enabled." if self.config['ai_insights_auto'] else "disabled.")
        )

    def _setup_ai_mood_tab(self):
        tk.Label(
            self.ai_mood_tab,
            text="AI Mood Tracker: Emotional polarity over time, scored from 0-100 (Neutral=50).",
            bg=COLOR_ACCENT_BG,
            fg=COLOR_TEXT_MUTED,
            font=("Arial", 8),
            anchor="w",
            justify="left",
            wraplength=400,
        ).pack(fill="x", padx=10, pady=(8, 4))
        
        self.ai_mood_summary_var = tk.StringVar(value="Waiting to evaluate history...")
        
        header_frame = tk.Frame(self.ai_mood_tab, bg=COLOR_ACCENT_BG)
        header_frame.pack(fill="x", padx=10, pady=0)
        
        tk.Label(header_frame, textvariable=self.ai_mood_summary_var, bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, font=("Arial", 10, "bold"), anchor="w").pack(side="left")
        
        self.ai_mood_temporal_var = tk.StringVar(value="Last 30 Days")
        filter_options = ["Last 7 Days", "Last 30 Days", "Previous Month", "All Time"]
        self.ai_mood_dropdown = tk.OptionMenu(header_frame, self.ai_mood_temporal_var, *filter_options, command=lambda _: self.refresh_insights_tabs())
        self.ai_mood_dropdown.config(bg=COLOR_BG, fg=COLOR_FG, font=("Arial", 9), highlightthickness=1)
        self.ai_mood_dropdown.pack(side="left", padx=15)
        
        self.ai_mood_canvas = tk.Canvas(self.ai_mood_tab, height=140, bg=COLOR_TEXT_BG, highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.ai_mood_canvas.pack(fill="x", padx=10, pady=(10, 8))

    def _setup_activity_matrix_tab(self):
        tk.Label(
            self.activity_matrix_tab,
            text="Activity Matrix: GitHub-style history of your dictation intensity.",
            bg=COLOR_ACCENT_BG,
            fg=COLOR_TEXT_MUTED,
            font=("Arial", 8),
            anchor="w",
            justify="left",
            wraplength=400,
        ).pack(fill="x", padx=10, pady=(8, 4))
        
        header_frame = tk.Frame(self.activity_matrix_tab, bg=COLOR_ACCENT_BG)
        header_frame.pack(fill="x", padx=10, pady=0)
        
        self.activity_metric_var = tk.StringVar(value="Words (Volume)")
        # No per-entry audio duration is recorded (stats only keep aggregate
        # totals and processing latencies), so a real WPM cannot be computed;
        # the previously mocked "Speed (WPM)" option was removed.
        metric_options = ["Words (Volume)", "Mood"]
        self.activity_metric_dropdown = tk.OptionMenu(header_frame, self.activity_metric_var, *metric_options, command=lambda _: self.refresh_insights_tabs())
        self.activity_metric_dropdown.config(bg=COLOR_BG, fg=COLOR_FG, font=("Arial", 9), highlightthickness=1)
        self.activity_metric_dropdown.pack(side="left")
        
        self.activity_canvas = tk.Canvas(self.activity_matrix_tab, height=150, bg=COLOR_TEXT_BG, highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.activity_canvas.pack(fill="x", padx=10, pady=(10, 8))

    def _setup_speaker_profile_tab(self):
        tk.Label(
            self.speaker_profile_tab,
            text="Speaker Profile: A radar map of your aggregate vocal and psychological traits.",
            bg=COLOR_ACCENT_BG,
            fg=COLOR_TEXT_MUTED,
            font=("Arial", 8),
            anchor="w",
            justify="left",
            wraplength=400,
        ).pack(fill="x", padx=10, pady=(8, 4))
        
        self.speaker_profile_canvas = tk.Canvas(self.speaker_profile_tab, height=200, bg=COLOR_TEXT_BG, highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.speaker_profile_canvas.pack(fill="x", padx=10, pady=(10, 8))

    def _setup_daily_brief_tab(self):
        # Left/right split. A plain tk.Frame pair rather than a
        # ttk.PanedWindow: the ttk widget draws in the light theme and reads as
        # a pale block inside the dark card.
        paned = tk.Frame(self.daily_brief_tab, bg=COLOR_ACCENT_BG)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        # Left frame: Listbox of dates
        list_frame = tk.Frame(paned, bg=COLOR_ACCENT_BG, width=96)
        list_frame.pack(side="left", fill="both")
        list_frame.pack_propagate(False)
        
        tk.Label(list_frame, text="History", bg=COLOR_ACCENT_BG, fg=COLOR_FG, font=("Arial", 9, "bold")).pack(anchor="w", padx=2)
        
        self.brief_listbox = tk.Listbox(list_frame, bg=COLOR_TEXT_BG, fg=COLOR_FG, selectbackground=COLOR_TEAL, selectforeground=COLOR_BG, font=("Arial", 10), bd=0, highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.brief_listbox.pack(side="left", fill="both", expand=True)
        
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.brief_listbox.yview)
        scroll.pack(side="right", fill="y")
        self.brief_listbox.config(yscrollcommand=scroll.set)
        
        self.brief_listbox.bind("<<ListboxSelect>>", self._on_brief_date_select)
        
        # Right frame: Text details
        detail_frame = tk.Frame(paned, bg=COLOR_ACCENT_BG)
        detail_frame.pack(side="left", fill="both", expand=True)
        
        tk.Label(detail_frame, text="AI Daily Brief", bg=COLOR_ACCENT_BG, fg=COLOR_FG, font=("Arial", 9, "bold")).pack(anchor="w", padx=4)
        
        self.brief_text = tk.Text(detail_frame, wrap="word", width=24, height=9, bg=COLOR_TEXT_BG, fg=COLOR_FG, font=("Georgia", 10), bd=0, highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.brief_text.pack(fill="both", expand=True, padx=(4, 0))
        self.brief_text.insert("1.0", "Select a date to view its daily brief.")
        self.brief_text.config(state="disabled")

    def _update_insight_tab_status(self):
        if not hasattr(self, "insights_tab_status"):
            return
        total = len(getattr(self, "insight_tab_frames", []) or [])
        if total == 0:
            return
        self.insights_tab_status.set(f"{self.insight_tab_index + 1}/{total}")

    def _update_chip_styles(self):
        """Highlight the chip of the selected insight tab."""
        for i, chip in enumerate(getattr(self, "insight_chip_buttons", [])):
            if i == self.insight_tab_index:
                chip.configure(fg_color=COLOR_TEAL, text_color=COLOR_BG,
                               border_color=COLOR_TEAL)
            else:
                chip.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED,
                               border_color=COLOR_BORDER)

    def _select_insight_tab(self, idx):
        """Chip click: swap the visible tab frame (the notebook replacement)."""
        frames = getattr(self, "insight_tab_frames", None)
        if not frames or idx is None or idx < 0 or idx >= len(frames):
            return
        if idx != self.insight_tab_index:
            frames[self.insight_tab_index].pack_forget()
            self.insight_tab_index = idx
            frames[idx].pack(fill="x")
        self._update_chip_styles()
        self._on_insight_tab_changed()
        self.root.after(50, self._update_scroll_region)

    def _select_insights_range(self, key):
        if key not in INSIGHTS_RANGE_OPTIONS:
            return
        self.insights_range_var.set(key)
        self._update_range_button_styles()
        self._on_insights_range_change(key)

    def _update_range_button_styles(self):
        current = self.insights_range_var.get()
        for key, btn in getattr(self, "insights_range_buttons", {}).items():
            if key == current:
                btn.configure(fg_color=COLOR_TEAL, text_color=COLOR_BG)
            else:
                btn.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_FAINT)

    def _update_insights_summary(self, today_words, session_speed):
        """One-line summary on the collapsed Speech Insights header."""
        if not hasattr(self, "insights_summary_var"):
            return
        parts = [f"{today_words:,} w", f"{session_speed} WPM"]
        snap = getattr(self, "_insights_snapshot", None)
        if snap and "clarity_fillers_per_100" in snap:
            parts.append(f"{snap['clarity_fillers_per_100']:.1f} fill")
        self.insights_summary_var.set(" \u00b7 ".join(parts))

    def _ensure_insight_tab_built(self, idx):
        """Build one tab body on demand. Returns True if it was just built."""
        builders = getattr(self, "_insight_tab_builders", None)
        if not builders or idx is None or idx < 0 or idx >= len(builders):
            return False
        if idx in self._insight_tabs_built:
            return False
        self._insight_tabs_built.add(idx)
        try:
            builders[idx]()
        except Exception as e:
            self._insight_tabs_built.discard(idx)
            self.log_internal(f"Failed to build insight tab {idx}: {e}")
            return False
        return True

    def _on_insight_tab_changed(self, event=None):
        self._update_insight_tab_status()
        if not hasattr(self, "insight_tab_frames"):
            return
        collapsed = getattr(self, "insights_collapsed", None)
        if collapsed is not None and collapsed.get():
            return  # nothing visible - build on expand instead
        if self._ensure_insight_tab_built(self.insight_tab_index):
            # Newly built tab has no data in it yet - populate it now.
            self.refresh_insights_tabs()

    def _on_insights_range_change(self, _value=None):
        value = self.insights_range_var.get()
        if value not in INSIGHTS_RANGE_OPTIONS:
            value = INSIGHTS_RANGE_DEFAULT
            self.insights_range_var.set(value)
        self.config['insights_range'] = value
        self._persist_config()
        self.refresh_insights_tabs()

    def _insights_range_days(self):
        var = getattr(self, "insights_range_var", None)
        label = var.get() if var is not None else INSIGHTS_RANGE_DEFAULT
        if label not in INSIGHTS_RANGE_OPTIONS:
            label = INSIGHTS_RANGE_DEFAULT
        return INSIGHTS_RANGE_OPTIONS[label]

    def _cycle_insight_tab(self, step):
        frames = getattr(self, "insight_tab_frames", None)
        if not frames:
            return
        self._select_insight_tab((self.insight_tab_index + step) % len(frames))

    def _tokenize_words(self, text):
        return tokenize_words(text)

    def _baseline_active(self):
        """True once enough history exists to use the personal trailing average."""
        return (
            getattr(self, "_personal_baseline_days", 0) >= PERSONAL_BASELINE_MIN_DAYS
            and bool(getattr(self, "_personal_baseline", None))
        )

    def _comparison_prefix(self):
        if self._baseline_active():
            return f"vs your {PERSONAL_BASELINE_DAYS}-day average"
        return "vs reference band"

    def _metric_band_status(self, value, metric_key):
        """Classify value against the personal baseline band (or the fallback
        reference band while there isn't enough history)."""
        baseline = self._baseline_value(metric_key)
        if self._baseline_active() and metric_key in self._personal_baseline:
            spread = abs(baseline) * PERSONAL_BASELINE_BAND_FRACTION
            low, high = baseline - spread, baseline + spread
        else:
            band = PEER_BENCHMARKS.get(metric_key, {"low": float("-inf"), "high": float("inf")})
            low = band.get("low", float("-inf"))
            high = band.get("high", float("inf"))
        if value < low:
            return "below", low, high
        if value > high:
            return "above", low, high
        return "within", low, high

    def _fmt_band(self, low, high, unit=""):
        return f"{low:.1f}-{high:.1f}{unit}"

    def _peer_midpoint(self, metric_key):
        band = PEER_BENCHMARKS.get(metric_key, {"low": 0.0, "high": 0.0})
        return (band.get("low", 0.0) + band.get("high", 0.0)) / 2.0

    def _baseline_value(self, metric_key):
        """Reference value for a metric: the user's own trailing average when
        available, otherwise the hand-set PEER_BENCHMARKS midpoint."""
        if self._baseline_active():
            baseline = self._personal_baseline
            if metric_key in baseline:
                return baseline[metric_key]
        return self._peer_midpoint(metric_key)

    def _relative_delta_pct(self, value, metric_key):
        reference = self._baseline_value(metric_key)
        if abs(reference) < 1e-9:
            return 0.0
        return ((value - reference) / reference) * 100.0

    def _more_less_phrase(self, value, metric_key):
        delta = self._relative_delta_pct(value, metric_key)
        direction = "more" if delta >= 0 else "less"
        return f"{abs(delta):.0f}% {direction}", delta

    def _robust_variability_band(self, values):
        """Return capped IQR/2 to show relative variability without outlier spikes."""
        if not values:
            return 0.0
        arr = np.array(values, dtype=float)
        if arr.size < 2:
            return 0.0
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr_half = max(0.0, (q3 - q1) / 2.0)
        median = float(np.median(arr))
        cap = max(2.0, median * 0.75)
        return min(iqr_half, cap)

    def _update_insights_profile_summary(self):
        if not hasattr(self, "insights_profile_summary_var"):
            return
        label = f"Current window {self._comparison_prefix()}"
        snapshot = getattr(self, "_peer_snapshot", {})
        if not snapshot:
            self.insights_profile_summary_var.set(f"{label}: waiting for enough data.")
            return
        ordered = []
        for key in ("pace", "pause", "clarity", "lexical", "interventions"):
            if key in snapshot:
                delta = snapshot[key]
                sign = "+" if delta >= 0 else "-"
                ordered.append(f"{key.capitalize()} {sign}{abs(delta):.0f}%")
        if not ordered:
            self.insights_profile_summary_var.set(f"{label}: waiting for enough data.")
            return
        self.insights_profile_summary_var.set(f"{label}: " + " | ".join(ordered))

    def _draw_empty_insight_canvas(self, canvas, msg):
        canvas.delete("all")
        canvas.update_idletasks()
        width = canvas.winfo_width() or 300
        height = canvas.winfo_height() or 100
        canvas.create_text(width / 2, height / 2, text=msg, fill="#7b8794", font=("Arial", 9))

    def _draw_vertical_bars(self, canvas, items, colors=None):
        canvas.delete("all")
        if not items:
            self._draw_empty_insight_canvas(canvas, "No data")
            return

        canvas.update_idletasks()
        width = canvas.winfo_width() or 300
        height = canvas.winfo_height() or 110
        left, right, top, bottom = 10, 10, 10, 20
        chart_w = width - left - right
        chart_h = height - top - bottom
        if chart_w <= 0 or chart_h <= 0:
            return

        max_val = max(v for _, v in items) if items else 1
        max_val = max(max_val, 1)
        bar_slot = chart_w / max(len(items), 1)
        bar_w = max(10, bar_slot - 12)

        def hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            
        def rgb_to_hex(r, g, b):
            return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"

        for i, (label, value) in enumerate(items):
            x1 = left + i * bar_slot + (bar_slot - bar_w) / 2
            x2 = x1 + bar_w
            bar_h = (value / max_val) * chart_h if max_val > 0 else 0
            if value > 0:
                bar_h = max(2, bar_h)
            y2 = top + chart_h
            y1 = y2 - bar_h
            
            base_color_hex = colors[i % len(colors)] if colors else COLOR_TEAL
            r, g, b = hex_to_rgb(base_color_hex)
            
            # Create a darker shadow color (-40% luminance approx)
            shadow_hex = rgb_to_hex(int(r * 0.6), int(g * 0.6), int(b * 0.6))
            # Create a lighter highlight color (+30% luminance approx)
            highlight_hex = rgb_to_hex(int(r + (255 - r) * 0.3), int(g + (255 - g) * 0.3), int(b + (255 - b) * 0.3))

            # 1. Shadow (offset bottom right)
            canvas.create_rectangle(x1 + 2, y1 + 2, x2 + 2, y2 + 2, fill=shadow_hex, outline="")
            # 2. Main Bar
            canvas.create_rectangle(x1, y1, x2, y2, fill=base_color_hex, outline="")
            # 3. Top Highlight Edge
            canvas.create_line(x1, y1, x2, y1, fill=highlight_hex, width=1)

            if isinstance(value, float):
                if abs(value - round(value)) < 0.05:
                    value_text = f"{value:.0f}"
                else:
                    value_text = f"{value:.1f}"
            else:
                value_text = str(value)

            if bar_h >= 18:
                value_y = y1 + 9
                value_fill = COLOR_BG
            else:
                value_y = max(top + 8, y1 - 8)
                value_fill = "#d1d5db"
            canvas.create_text((x1 + x2) / 2, value_y, text=value_text, fill=value_fill, font=("Arial", 8, "bold"))

            short_label = label if len(label) <= 10 else label[:9] + "…"
            canvas.create_text((x1 + x2) / 2, y2 + 10, text=short_label, fill="#94a3b8", font=("Arial", 8))

    def _month_keys_between(self, start_date, end_date):
        keys = []
        y, m = start_date.year, start_date.month
        while (y < end_date.year) or (y == end_date.year and m <= end_date.month):
            keys.append(f"{y:04d}-{m:02d}")
            m += 1
            if m > 12:
                y += 1
                m = 1
        return keys

    def _load_analytics_entries(self, days=None):
        now = datetime.datetime.now()
        cutoff = None if days is None else (now - datetime.timedelta(days=days))
        entries = []
        scanned_files = 0

        if not os.path.exists(TRANSCRIPTIONS_DIR):
            self._analytics_scanned_files = 0
            return entries

        for filename in sorted(os.listdir(TRANSCRIPTIONS_DIR)):
            if not filename.lower().endswith(".txt"):
                continue
            if "conflicted copy" in filename.lower():
                continue  # Dropbox sync debris duplicates entries - skip
            path = os.path.join(TRANSCRIPTIONS_DIR, filename)
            if not os.path.isfile(path):
                continue
            scanned_files += 1
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or not line.startswith("[") or "] " not in line:
                            continue
                        try:
                            ts_end = line.index("] ")
                            ts = datetime.datetime.strptime(line[1:ts_end], "%Y-%m-%d %H:%M:%S")
                            if cutoff is not None and ts < cutoff:
                                continue
                            text = line[ts_end + 2:].strip()
                            if text:
                                entries.append({"timestamp": ts, "text": text})
                        except Exception:
                            continue
            except Exception:
                continue

        entries.sort(key=lambda x: x["timestamp"])
        self._analytics_scanned_files = scanned_files
        return entries

    def _build_analytics_coverage_text(self, entries, archive_total=None):
        file_count = getattr(self, "_analytics_scanned_files", 0)
        window_label = getattr(self, "insights_range_var", None)
        window_label = window_label.get() if window_label is not None else INSIGHTS_RANGE_DEFAULT
        if not entries:
            if file_count:
                return (
                    f"Coverage: window '{window_label}' has no entries "
                    f"({file_count} transcript file(s) scanned)."
                )
            return "Coverage: no transcript files found yet."

        first_ts = entries[0]["timestamp"].strftime("%Y-%m-%d")
        last_ts = entries[-1]["timestamp"].strftime("%Y-%m-%d")
        total_note = ""
        if archive_total is not None and archive_total != len(entries):
            total_note = f" of {archive_total:,} archived"
        return (
            f"Coverage: window '{window_label}' - {len(entries):,} entries{total_note} "
            f"from {first_ts} to {last_ts} across {file_count} file(s)."
        )

    def _get_analytics_entries(self, force_reload=False):
        """Cached transcript entries; loaded from disk once, then appended
        incrementally by save_transcription(). Tokens and the detected language
        are cached alongside so a refresh never re-tokenizes the archive."""
        if force_reload or self._analytics_entries_cache is None:
            self._analytics_entries_cache = self._load_analytics_entries(days=None)
            self._analytics_tokens_cache = None
            self._analytics_langs_cache = None
        entries = self._analytics_entries_cache
        if (
            self._analytics_tokens_cache is None
            or self._analytics_langs_cache is None
            or len(self._analytics_tokens_cache) != len(entries)
            or len(self._analytics_langs_cache) != len(entries)
        ):
            self._analytics_tokens_cache = [tokenize_words(e["text"]) for e in entries]
            self._analytics_langs_cache = [
                detect_entry_language(e["text"], toks)
                for e, toks in zip(entries, self._analytics_tokens_cache)
            ]
        return entries

    # --- Metric snapshot / personal baseline -----------------------------

    def _filler_stats(self, entries, tokens_per_entry, langs):
        """Per-language filler counts over the given slice.

        Returns (combined_counts, per_lang) where combined_counts maps a
        display label to a count and per_lang maps 'de'/'en' to
        {'words': int, 'fillers': int, 'counts': {label: int}}.
        """
        per_lang = {}
        combined = {}
        for entry, toks, lang in zip(entries, tokens_per_entry, langs):
            bucket = per_lang.setdefault(lang, {"words": 0, "fillers": 0, "counts": {}})
            bucket["words"] += len(toks)
            for label, count in count_fillers(entry["text"], lang).items():
                if not count:
                    continue
                bucket["counts"][label] = bucket["counts"].get(label, 0) + count
                bucket["fillers"] += count
                key = f"{label} ({lang})"
                combined[key] = combined.get(key, 0) + count
        return combined, per_lang

    def _compute_metric_snapshot(self, entries, tokens_per_entry, langs, sentences=None):
        """Compute every benchmarked metric for one slice of entries.

        Used both for the live window and for the trailing personal baseline,
        so the two are always calculated identically.
        """
        snap = {}
        if not entries:
            return snap

        counts = [len(t) for t in tokens_per_entry if t]
        if counts:
            avg_words = float(np.mean(counts))
            variability = self._robust_variability_band(counts)
            snap["pace_avg_words_entry"] = avg_words
            snap["pace_stability"] = (
                max(0.0, 100.0 - min(100.0, (variability / avg_words) * 100.0 * 2.0))
                if avg_words > 0 else 0.0
            )

        gaps = []
        prev_ts = entries[0]["timestamp"]
        for entry in entries[1:]:
            delta = (entry["timestamp"] - prev_ts).total_seconds()
            prev_ts = entry["timestamp"]
            if 0 < delta <= 120:
                gaps.append(delta)
        if gaps:
            snap["pause_avg_seconds"] = float(np.mean(gaps))
            snap["pause_long_share"] = sum(1 for g in gaps if g >= 5.0) / len(gaps)

        all_tokens = [t for toks in tokens_per_entry for t in toks]
        total_words = len(all_tokens)
        if total_words:
            _, per_lang = self._filler_stats(entries, tokens_per_entry, langs)
            filler_total = sum(b["fillers"] for b in per_lang.values())
            snap["clarity_fillers_per_100"] = (filler_total * 100.0) / total_words
            repeats = sum(
                1 for i in range(len(all_tokens) - 1) if all_tokens[i] == all_tokens[i + 1]
            )
            snap["clarity_repetition_per_1k"] = (repeats * 1000.0) / total_words
            snap["lexical_diversity_percent"] = (len(set(all_tokens)) / total_words) * 100.0
            long_words = sum(1 for w in all_tokens if len(w) >= 7)
            snap["lexical_long_word_percent"] = (long_words / total_words) * 100.0

        if sentences is None:
            joined = " ".join(e["text"] for e in entries).strip()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if s.strip()]
            if not sentences and joined:
                sentences = [joined]
        if sentences:
            oq, refl, affirm, summ = self._count_intervention_markers(sentences)
            total = len(sentences)
            snap["intervention_open_question_per_100"] = (oq * 100.0) / total
            snap["intervention_reflection_ratio"] = refl / max(oq, 1)
            snap["intervention_affirmation_per_100"] = (affirm * 100.0) / total
            snap["intervention_summary_per_100"] = (summ * 100.0) / total
        return snap

    def _update_personal_baseline(self, entries_all, tokens_all, langs_all):
        """Recompute the trailing-N-day personal averages used as the
        comparison reference. Falls back to PEER_BENCHMARKS when fewer than
        PERSONAL_BASELINE_MIN_DAYS distinct days of speech exist."""
        idx = filter_entries_by_days(entries_all, PERSONAL_BASELINE_DAYS)
        entries = [entries_all[i] for i in idx]
        distinct_days = len({e["timestamp"].date() for e in entries})
        self._personal_baseline_days = distinct_days
        if distinct_days < PERSONAL_BASELINE_MIN_DAYS:
            self._personal_baseline = {}
            return
        self._personal_baseline = self._compute_metric_snapshot(
            entries,
            [tokens_all[i] for i in idx],
            [langs_all[i] for i in idx],
        )

    def _update_baseline_label(self):
        if not hasattr(self, "insights_peer_var"):
            return
        days = getattr(self, "_personal_baseline_days", 0)
        if self._baseline_active():
            self.insights_peer_var.set(
                f"Reference: your own trailing {PERSONAL_BASELINE_DAYS}-day average "
                f"({days} day(s) of speech); band = +/-{int(PERSONAL_BASELINE_BAND_FRACTION * 100)}%."
            )
        else:
            self.insights_peer_var.set(
                f"Reference: {PEER_PROFILE['label']} until you have "
                f"{PERSONAL_BASELINE_MIN_DAYS} days of your own speech "
                f"(you have {days}). {PEER_BENCHMARK_NOTE}"
            )

    def _update_ai_freshness_label(self):
        if not hasattr(self, "insights_ai_age_var"):
            return
        cache = self._get_daily_insights_cache()
        newest = self._newest_ai_insight_date(cache)
        if newest is None:
            self.insights_ai_age_var.set(
                "AI insights last crunched: never - use 'Crunch History' on the AI Timeline tab."
            )
            return
        age = (datetime.date.today() - newest).days
        if age <= 0:
            self.insights_ai_age_var.set(f"AI insights last crunched: today ({newest.isoformat()}).")
        else:
            self.insights_ai_age_var.set(
                f"AI insights last crunched: {age} day(s) ago ({newest.isoformat()})."
            )

    @staticmethod
    def _newest_ai_insight_date(cache):
        newest = None
        for key in (cache or {}):
            try:
                d = datetime.datetime.strptime(key, "%Y-%m-%d").date()
            except Exception:
                continue
            if newest is None or d > newest:
                newest = d
        return newest

    def _schedule_insights_refresh(self, delay_ms=2000):
        """Debounced insights refresh: coalesces bursts of transcriptions and
        defers the (expensive) full-archive tokenization until recording stops."""
        if self._insights_refresh_job is not None:
            try:
                self.root.after_cancel(self._insights_refresh_job)
            except Exception:
                pass
        self._insights_refresh_job = self.root.after(delay_ms, self._run_scheduled_insights_refresh)

    def _run_scheduled_insights_refresh(self):
        self._insights_refresh_job = None
        if self.mode is not None:
            # Still recording/typing - don't burn the main thread now
            self._schedule_insights_refresh(5000)
            return
        self.refresh_insights_tabs()

    def refresh_insights_tabs(self, force_reload=False):
        if getattr(self, "insights_collapsed", None) is not None and self.insights_collapsed.get():
            # Panel hidden: defer the full-archive tokenization until it is expanded.
            self._insights_dirty = True
            if force_reload:
                self._analytics_entries_cache = None
            return
        self._insights_dirty = False
        self._peer_snapshot = {}
        entries_all = self._get_analytics_entries(force_reload)
        tokens_all = self._analytics_tokens_cache or []
        langs_all = self._analytics_langs_cache or []

        # Personal reference values come from the trailing baseline window,
        # independent of the (usually shorter) display window.
        self._update_personal_baseline(entries_all, tokens_all, langs_all)

        # Rolling display window for the non-AI tabs. The AI tabs keep their
        # own temporal dropdowns, so they still see the whole archive.
        window_idx = filter_entries_by_days(entries_all, self._insights_range_days())
        entries = [entries_all[i] for i in window_idx]
        tokens_per_entry = [tokens_all[i] for i in window_idx]
        langs = [langs_all[i] for i in window_idx]

        if hasattr(self, "insights_coverage_var"):
            self.insights_coverage_var.set(
                self._build_analytics_coverage_text(entries, len(entries_all))
            )
        self._update_baseline_label()
        self._update_ai_freshness_label()

        all_tokens = [t for entry_tokens in tokens_per_entry for t in entry_tokens]
        all_text = " ".join(e["text"] for e in entries).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", all_text) if s.strip()]
        if not sentences and all_text:
            sentences = [all_text]

        # Headline deltas are computed centrally so the summary stays correct
        # even while some tab bodies have not been built yet.
        current = self._compute_metric_snapshot(entries, tokens_per_entry, langs, sentences)
        self._insights_snapshot = current
        for group, keys in (
            ("pace", ("pace_avg_words_entry", "pace_stability")),
            ("pause", ("pause_avg_seconds", "pause_long_share")),
            ("clarity", ("clarity_fillers_per_100", "clarity_repetition_per_1k")),
            ("lexical", ("lexical_diversity_percent", "lexical_long_word_percent")),
            ("interventions", (
                "intervention_open_question_per_100",
                "intervention_reflection_ratio",
                "intervention_affirmation_per_100",
                "intervention_summary_per_100",
            )),
        ):
            deltas = [self._relative_delta_pct(current[k], k) for k in keys if k in current]
            if deltas:
                self._peer_snapshot[group] = sum(deltas) / len(deltas)

        self._refresh_pace_tab(entries, tokens_per_entry)
        self._refresh_pause_tab(entries)
        self._refresh_clarity_tab(entries, tokens_per_entry, langs)
        self._refresh_lexical_tab(all_tokens, langs)
        self._refresh_intervention_tab(sentences)
        self._refresh_ai_timeline_tab(entries_all)
        self._refresh_ai_mood_tab(entries_all)
        self._refresh_activity_matrix_tab(entries, all_text)
        self._refresh_speaker_profile_tab(entries, all_tokens, sentences)
        self._refresh_daily_brief_tab(entries_all)
        self._update_insights_profile_summary()
        self._update_metric_bands(current)

    def _refresh_pace_tab(self, entries, tokens_per_entry=None):
        if not hasattr(self, "pace_summary_var"):
            return  # tab body not built yet
        if not entries:
            self.pace_summary_var.set("No pace data yet")
            self.pace_detail_var.set("Record a few transcriptions to view pace patterns.")
            self.pace_peer_var.set("Comparison unavailable (no transcript entries).")
            self._draw_empty_insight_canvas(self.pace_canvas, "No transcripts found")
            return

        if tokens_per_entry is None:
            tokens_per_entry = [self._tokenize_words(e["text"]) for e in entries]

        counts = [len(t) for t in tokens_per_entry]
        counts = [c for c in counts if c > 0]
        if not counts:
            self.pace_summary_var.set("No pace data yet")
            self.pace_detail_var.set("Record a few transcriptions to view pace patterns.")
            self.pace_peer_var.set("Comparison unavailable (no tokenized entries).")
            self._draw_empty_insight_canvas(self.pace_canvas, "No word counts yet")
            return

        avg_words = float(np.mean(counts))
        variability_words = self._robust_variability_band(counts)
        if avg_words > 0:
            stability = max(0.0, 100.0 - min(100.0, (variability_words / avg_words) * 100.0 * 2.0))
        else:
            stability = 0.0

        today = datetime.datetime.now().date()
        today_words = sum(len(tokens_per_entry[i]) for i, e in enumerate(entries) if e["timestamp"].date() == today)
        self.pace_summary_var.set(f"Avg {avg_words:.1f} words/entry | Stability {stability:.0f}%")
        self.pace_detail_var.set(f"Today {today_words:,} words | Variability band +/-{variability_words:.1f} words/entry")
        _, avg_low, avg_high = self._metric_band_status(avg_words, "pace_avg_words_entry")
        _, stab_low, stab_high = self._metric_band_status(stability, "pace_stability")
        avg_phrase, avg_delta = self._more_less_phrase(avg_words, "pace_avg_words_entry")
        stab_phrase, stab_delta = self._more_less_phrase(stability, "pace_stability")
        self.pace_peer_var.set(
            f"{self._comparison_prefix()}: {avg_phrase} words/entry and {stab_phrase} stability "
            f"({self._fmt_band(avg_low, avg_high)}; {self._fmt_band(stab_low, stab_high, '%')})."
        )
        self._peer_snapshot["pace"] = (avg_delta + stab_delta) / 2.0

        series = counts[-20:]
        self.pace_canvas.delete("all")
        self.pace_canvas.update_idletasks()
        w = self.pace_canvas.winfo_width() or 300
        h = self.pace_canvas.winfo_height() or 110
        left, right, top, bottom = 10, 10, 10, 20
        cw = w - left - right
        ch = h - top - bottom
        if len(series) < 2:
            self._draw_empty_insight_canvas(self.pace_canvas, "Need at least 2 entries for trend")
            return
        max_v = max(series) if series else 1
        max_v = max(max_v, 1)
        step = cw / (len(series) - 1)
        pts = []
        for i, v in enumerate(series):
            x = left + i * step
            y = top + ch - (v / max_v) * ch
            pts.extend([x, y])
        self.pace_canvas.create_line(*pts, fill=COLOR_TEAL, width=2, smooth=True)
        for i, v in enumerate(series):
            x = left + i * step
            y = top + ch - (v / max_v) * ch
            self.pace_canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#99f6e4", outline="")
        self.pace_canvas.create_text(left, h - 10, text="oldest", anchor="w", fill="#94a3b8", font=("Arial", 7))
        self.pace_canvas.create_text(w - right, h - 10, text="latest", anchor="e", fill="#94a3b8", font=("Arial", 7))

    def _refresh_pause_tab(self, entries):
        if not hasattr(self, "pause_summary_var"):
            return  # tab body not built yet
        if len(entries) < 2:
            self.pause_summary_var.set("No pause data yet")
            self.pause_detail_var.set("Need at least two timestamped entries.")
            self.pause_peer_var.set("Comparison unavailable (need at least 2 entries).")
            self._draw_empty_insight_canvas(self.pause_canvas, "No pause intervals")
            return

        gaps = []
        prev_ts = entries[0]["timestamp"]
        for entry in entries[1:]:
            ts = entry["timestamp"]
            delta = (ts - prev_ts).total_seconds()
            prev_ts = ts
            if 0 < delta <= 120:
                gaps.append(delta)

        if not gaps:
            self.pause_summary_var.set("No pause profile yet")
            self.pause_detail_var.set("No short/medium/long gaps detected in range.")
            self.pause_peer_var.set("Comparison unavailable (no comparable pause gaps).")
            self._draw_empty_insight_canvas(self.pause_canvas, "No pause gaps under 120s")
            return

        short = sum(1 for g in gaps if g < 2.0)
        medium = sum(1 for g in gaps if 2.0 <= g < 5.0)
        long = sum(1 for g in gaps if g >= 5.0)
        avg_gap = float(np.mean(gaps))
        self.pause_summary_var.set(f"Avg pause {avg_gap:.1f}s | Short {short} / Medium {medium} / Long {long}")
        self.pause_detail_var.set("Short <2s, Medium 2-5s, Long >=5s (from transcription gaps).")
        long_share = long / len(gaps) if gaps else 0.0
        _, avg_low, avg_high = self._metric_band_status(avg_gap, "pause_avg_seconds")
        _, long_low, long_high = self._metric_band_status(long_share, "pause_long_share")
        avg_phrase, avg_delta = self._more_less_phrase(avg_gap, "pause_avg_seconds")
        long_phrase, long_delta = self._more_less_phrase(long_share, "pause_long_share")
        self.pause_peer_var.set(
            f"{self._comparison_prefix()}: {avg_phrase} pause duration and {long_phrase} long-pause share "
            f"({self._fmt_band(avg_low, avg_high, 's')}; {self._fmt_band(long_low * 100, long_high * 100, '%')})."
        )
        self._peer_snapshot["pause"] = (avg_delta + long_delta) / 2.0
        self._draw_vertical_bars(
            self.pause_canvas,
            [("Short", short), ("Medium", medium), ("Long", long)],
            colors=["#22c55e", "#f59e0b", "#ef4444"],
        )

    def _refresh_clarity_tab(self, entries=None, tokens_per_entry=None, langs=None):
        """Filler/repetition signals, with the filler list chosen per entry
        from its detected language."""
        if not hasattr(self, "clarity_summary_var"):
            return  # tab body not built yet
        entries = entries or []
        tokens_per_entry = tokens_per_entry or []
        langs = langs or []
        tokens = [t for toks in tokens_per_entry for t in toks]

        total_words = len(tokens)
        if total_words == 0:
            self.clarity_summary_var.set("No clarity data yet")
            self.clarity_detail_var.set("Transcribe more speech to estimate clarity signals.")
            self.clarity_peer_var.set("Comparison unavailable (no words detected).")
            self._draw_empty_insight_canvas(self.clarity_canvas, "No tokens")
            return

        counts, per_lang = self._filler_stats(entries, tokens_per_entry, langs)
        filler_total = sum(b["fillers"] for b in per_lang.values())
        filler_per_100 = (filler_total * 100.0) / total_words
        repeated_word_pairs = sum(1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i + 1])
        repetition_per_1k = (repeated_word_pairs * 1000.0) / total_words if total_words else 0.0

        per_lang_bits = []
        for lang in ("de", "en"):
            bucket = per_lang.get(lang)
            if not bucket or not bucket["words"]:
                continue
            rate = (bucket["fillers"] * 100.0) / bucket["words"]
            per_lang_bits.append(f"{lang.upper()} {rate:.1f}/100w ({bucket['words']:,} words)")
        lang_note = " | ".join(per_lang_bits) if per_lang_bits else "n/a"

        self.clarity_summary_var.set(
            f"Fillers {filler_per_100:.1f}/100 words combined | Repetitions {repeated_word_pairs}"
        )
        self.clarity_detail_var.set(
            f"Total words analyzed: {total_words:,} | By language: {lang_note}"
        )
        _, filler_low, filler_high = self._metric_band_status(filler_per_100, "clarity_fillers_per_100")
        _, rep_low, rep_high = self._metric_band_status(repetition_per_1k, "clarity_repetition_per_1k")
        filler_phrase, filler_delta = self._more_less_phrase(filler_per_100, "clarity_fillers_per_100")
        rep_phrase, rep_delta = self._more_less_phrase(repetition_per_1k, "clarity_repetition_per_1k")
        self.clarity_peer_var.set(
            f"{self._comparison_prefix()}: {filler_phrase} fillers and {rep_phrase} repetition density "
            f"({self._fmt_band(filler_low, filler_high, '/100w')}; {self._fmt_band(rep_low, rep_high, '/1k')})."
        )
        self._peer_snapshot["clarity"] = (filler_delta + rep_delta) / 2.0
        items = Counter(counts).most_common(5)
        if not items:
            self._draw_empty_insight_canvas(self.clarity_canvas, "No fillers detected in range")
            return
        self._draw_vertical_bars(
            self.clarity_canvas,
            items,
            colors=["#38bdf8", "#22d3ee", "#2dd4bf", "#f59e0b", "#f97316"],
        )

    def _refresh_lexical_tab(self, tokens=None, langs=None):
        if not hasattr(self, "lexical_summary_var"):
            return  # tab body not built yet
        if tokens is None:
            tokens = []
        if not tokens:
            self.lexical_summary_var.set("No vocabulary data yet")
            self.lexical_detail_var.set("Start dictating to populate vocabulary metrics.")
            self.lexical_peer_var.set("Comparison unavailable (no vocabulary sample).")
            self._draw_empty_insight_canvas(self.lexical_canvas, "No vocabulary data")
            return

        unique = len(set(tokens))
        total = len(tokens)
        diversity = (unique / total) * 100.0 if total > 0 else 0.0
        
        counts = Counter(tokens)
        long_word_count = sum(c for w, c in counts.items() if len(w) >= 7)
        long_ratio = (long_word_count / total) * 100.0 if total > 0 else 0.0
        
        self.lexical_summary_var.set(f"Vocabulary Richness {diversity:.1f}% | Expressive Depth {long_ratio:.1f}%")
        _, div_low, div_high = self._metric_band_status(diversity, "lexical_diversity_percent")
        _, long_low, long_high = self._metric_band_status(long_ratio, "lexical_long_word_percent")
        div_phrase, div_delta = self._more_less_phrase(diversity, "lexical_diversity_percent")
        long_phrase, long_delta = self._more_less_phrase(long_ratio, "lexical_long_word_percent")
        self.lexical_peer_var.set(
            f"{self._comparison_prefix()}: {div_phrase} vocabulary richness and {long_phrase} expressive depth "
            f"({self._fmt_band(div_low, div_high, '%')}; {self._fmt_band(long_low, long_high, '%')})."
        )
        self._peer_snapshot["lexical"] = (div_delta + long_delta) / 2.0

        # Core themes: strip the stopwords of every language present in the
        # window, so a mixed DE/EN archive doesn't surface "der"/"the".
        langs_present = set(langs or [])
        if not langs_present:
            langs_present = {"en"}
        stop_words = set()
        for lang in langs_present:
            stop_words |= stopwords_for(lang)
        content_freq = {w: c for w, c in counts.items() if len(w) > 2 and w not in stop_words}
        top = Counter(content_freq).most_common(5)
        self.lexical_detail_var.set("Core Themes: " + (", ".join(w for w, _ in top) if top else "n/a"))
        self._draw_vertical_bars(
            self.lexical_canvas,
            [(w, c) for w, c in top],
            colors=["#34d399", "#2dd4bf", "#14b8a6", "#0ea5e9", "#22d3ee"],
        )

    @staticmethod
    def _count_intervention_markers(sentences):
        """Heuristic counts of (open questions, reflections, affirmations,
        summaries) across sentences. Shared by the tab and the baseline."""
        open_question_pattern = re.compile(
            r"\b(what|how|when|where|who|could|would|can|tell me)\b", re.IGNORECASE
        )
        reflection_phrases = ("you feel", "it sounds like", "i hear", "you're feeling", "what i'm hearing")
        affirmation_phrases = ("that makes sense", "good job", "great", "i appreciate", "thank you", "important work")
        summary_phrases = ("to summarize", "so far", "in summary", "what we covered")

        open_questions = reflections = affirmations = summaries = 0
        for sentence in sentences:
            s_lower = sentence.lower()
            if "?" in sentence and open_question_pattern.search(sentence):
                open_questions += 1
            if any(p in s_lower for p in reflection_phrases):
                reflections += 1
            if any(p in s_lower for p in affirmation_phrases):
                affirmations += 1
            if any(p in s_lower for p in summary_phrases):
                summaries += 1
        return open_questions, reflections, affirmations, summaries

    def _refresh_intervention_tab(self, sentences=None):
        if not hasattr(self, "intervention_summary_var"):
            return  # tab body not built yet
        if not sentences:
            self.intervention_summary_var.set("No intervention data yet")
            self.intervention_detail_var.set("Heuristic language tag counts will appear here.")
            self.intervention_peer_var.set("Comparison unavailable (no sentence samples).")
            self._draw_empty_insight_canvas(self.intervention_canvas, "No language samples")
            return

        open_questions, reflections, affirmations, summaries = \
            self._count_intervention_markers(sentences)

        self_words = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
        collective_words = re.compile(r"\b(we|us|our|ours|ourselves)\b", re.IGNORECASE)
        self_count = 0
        collective_count = 0
        for sentence in sentences:
            self_count += len(self_words.findall(sentence))
            collective_count += len(collective_words.findall(sentence))

        total = len(sentences)
        self.intervention_summary_var.set(
            f"Open Q {open_questions} | Reflections {reflections} | Affirmations {affirmations} | Summaries {summaries}"
        )
        
        focus_ratio = f"{self_count} vs {collective_count}"
        if collective_count > 0:
            focus_ratio = f"{(self_count/collective_count):.1f}:1"
            
        self.intervention_detail_var.set(f"Across {total} sentence(s) | Self vs Collective Focus: ({focus_ratio})")
        open_q_per_100 = (open_questions * 100.0) / total if total else 0.0
        reflect_ratio = reflections / max(open_questions, 1)
        affirm_per_100 = (affirmations * 100.0) / total if total else 0.0
        summary_per_100 = (summaries * 100.0) / total if total else 0.0

        oq_phrase, oq_delta = self._more_less_phrase(open_q_per_100, "intervention_open_question_per_100")
        rr_phrase, rr_delta = self._more_less_phrase(reflect_ratio, "intervention_reflection_ratio")
        af_phrase, af_delta = self._more_less_phrase(affirm_per_100, "intervention_affirmation_per_100")
        su_phrase, su_delta = self._more_less_phrase(summary_per_100, "intervention_summary_per_100")
        self.intervention_peer_var.set(
            f"{self._comparison_prefix()}: {oq_phrase} open-Q rate, {rr_phrase} reflection ratio, "
            f"{af_phrase} affirmations, {su_phrase} summaries."
        )
        self._peer_snapshot["interventions"] = (oq_delta + rr_delta + af_delta + su_delta) / 4.0
        
        # Determine the 5th bar (Self vs Collective focus)
        focus_val = self_count - collective_count
        focus_label = "Focus: I" if focus_val >= 0 else "Focus: We"
        
        self._draw_vertical_bars(
            self.intervention_canvas,
            [("Open Q", open_questions), ("Reflect", reflections), ("Affirm", affirmations), ("Summary", summaries), (focus_label, abs(focus_val))],
            colors=["#60a5fa", "#818cf8", "#34d399", "#f59e0b", "#ec4899"],
        )

    def _get_daily_insights_cache(self):
        try:
            if os.path.exists(AI_INSIGHTS_FILE):
                with open(AI_INSIGHTS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._quarantine_corrupt_file(os.path.join(app_dir, AI_INSIGHTS_FILE), e)
        except Exception as e:
            self.log_internal(f"Failed to load AI Insights cache: {e}")
        return {}

    def _save_daily_insights_cache(self, cache):
        try:
            self._write_json_atomic(os.path.join(app_dir, AI_INSIGHTS_FILE), cache, indent=4)
        except Exception as e:
            self.log_internal(f"Failed to save AI Insights cache: {e}")

    def _claim_ai_crunch(self):
        """Take the single-run guard; returns False if a crunch is already in
        flight (manual or automatic)."""
        with self._ai_crunch_lock:
            if self._ai_crunch_running:
                return False
            self._ai_crunch_running = True
            return True

    def _trigger_ai_history_crunch(self):
        api_key = self.config.get('openai_api_key', '').strip()
        if not api_key:
            messagebox.showwarning("OpenAI API Key Missing", "To process insights, please set your OpenAI API key in the configuration panel.")
            return
        if not self._claim_ai_crunch():
            self.log_internal("AI insight crunch already running.")
            return

        if hasattr(self, "btn_crunch_history"):
            self.btn_crunch_history.config(state="disabled", text="Working...")
        if hasattr(self, "ai_timeline_detail_var"):
            self.ai_timeline_detail_var.set("Batch processing transcripts through OpenAI... See System Log.")
        threading.Thread(target=self._batch_process_ai_insights, args=(api_key,), daemon=True).start()

    def _maybe_auto_crunch_ai_insights(self):
        """Once per app start, refresh the AI insight cache in the background if
        the newest cached day is more than a day old. Never runs twice and never
        runs while a recording is in progress."""
        if self._ai_auto_crunch_done:
            return
        # Opt-in only: this uploads transcript text to OpenAI, so it must never
        # happen just because an API key happens to be configured.
        if not self.config.get('ai_insights_auto', False):
            return
        api_key = self.config.get('openai_api_key', '').strip()
        if not api_key:
            return
        if self.mode is not None:
            # Recording/typing right now - retry in a few minutes
            self.root.after(300000, self._maybe_auto_crunch_ai_insights)
            return
        newest = self._newest_ai_insight_date(self._get_daily_insights_cache())
        if newest is not None and (datetime.date.today() - newest).days < 1:
            self._ai_auto_crunch_done = True
            return
        if not self._claim_ai_crunch():
            return
        self._ai_auto_crunch_done = True
        self.log_internal("AI insights are stale - running an automatic history crunch in the background.")
        if hasattr(self, "btn_crunch_history"):
            self.btn_crunch_history.config(state="disabled", text="Working...")
        threading.Thread(target=self._batch_process_ai_insights, args=(api_key,), daemon=True).start()

    def _batch_process_ai_insights(self, api_key):
        try:
            self._batch_process_ai_insights_inner(api_key)
        except Exception as e:
            # Never leave the single-run guard stuck on an unexpected failure
            msg = f"AI insight crunch failed: {e}"
            self._ui_after(0, lambda: self._on_crunch_finished(msg))

    def _batch_process_ai_insights_inner(self, api_key):
        import openai
        client = openai.OpenAI(api_key=api_key)

        entries = self._load_analytics_entries(days=None)
        if not entries:
            self._ui_after(0, lambda: self._on_crunch_finished("No transcript entries found to process."))
            return

        # Group entries by day
        from collections import defaultdict
        daily_text = defaultdict(list)
        for e in entries:
            day_str = e["timestamp"].strftime("%Y-%m-%d")
            daily_text[day_str].append(e["text"])

        cache = self._get_daily_insights_cache()
        processed_days = 0

        prompt = (
            "You are a psycho-linguistic AI analyst. Analyze the user's combined transcribed speech for this day.\n"
            "Identify what they are focused on. Provide output EXACTLY as valid JSON with the following schema:\n"
            "{\n"
            '  "mood": "Short 2 word summary of tone",\n'
            '  "mood_score": 50, // 0 to 100 rating (0=Severe Negative/Depressed, 100=Ecstatic Positive/Manic, 50=Neutral)\n'
            '  "daily_brief": "A balanced 3-5 sentence summary of the day focusing on productivity accomplishments, underlying mindset, and current tasks.",\n'
            '  "themes": ["List", "of", "Abstract", "Themes"],\n'
            '  "projects": ["List", "of", "Concrete", "Projects"]\n'
            "}"
        )

        for day_str, texts in sorted(daily_text.items()):
            if day_str in cache:
                continue

            combined_text = " ".join(texts)
            if len(combined_text) < 50: # Skip very empty days
                continue
                
            self.log_internal(f"Analyzing AI Insights for {day_str}...")
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": combined_text[:12000]} # Limit to roughly last hours
                    ],
                    response_format={ "type": "json_object" },
                    temperature=0.2
                )
                result = json.loads(response.choices[0].message.content)
                cache[day_str] = result
                self._save_daily_insights_cache(cache)
                processed_days += 1
                time.sleep(1) # Rate limiting
            except Exception as e:
                self.log_internal(f"Failed AI Analysis for {day_str}: {e}")
                
        self._ui_after(0, lambda: self._on_crunch_finished(f"Processed {processed_days} new days of insights."))

    def _on_crunch_finished(self, message):
        with self._ai_crunch_lock:
            self._ai_crunch_running = False
        self.log_internal(message)
        if hasattr(self, "btn_crunch_history"):
            self.btn_crunch_history.config(state="normal", text="Crunch History")
        if hasattr(self, "ai_timeline_detail_var"):
            self.ai_timeline_detail_var.set(message)
        self._update_ai_freshness_label()
        self.refresh_insights_tabs()

    def _filter_ai_dates(self, dates, filter_str):
        if not dates: return []
        today = datetime.date.today()
        filtered = []
        for d in dates:
            try:
                dt = datetime.datetime.strptime(d, "%Y-%m-%d").date()
                if filter_str == "Last 7 Days":
                    if (today - dt).days <= 7: filtered.append(d)
                elif filter_str == "Last 30 Days":
                    if (today - dt).days <= 30: filtered.append(d)
                elif filter_str == "Previous Month":
                    prev_month = (today.replace(day=1) - datetime.timedelta(days=1)).month
                    if dt.month == prev_month: filtered.append(d)
                else: # All Time
                    filtered.append(d)
            except Exception:
                pass
        return filtered

    def _refresh_ai_timeline_tab(self, entries):
        if not hasattr(self, "ai_timeline_summary_var"):
            return  # tab body not built yet
        self.ai_timeline_summary_var.set("Loading timeline cache...")
        self.ai_timeline_canvas.update_idletasks()
        
        cache = self._get_daily_insights_cache()
        if not cache:
            self._draw_empty_insight_canvas(self.ai_timeline_canvas, "No AI history to plot yet.")
            return

        dates = sorted(cache.keys())
        dates = self._filter_ai_dates(dates, self.ai_timeline_temporal_var.get())
        
        if not dates:
            self.ai_timeline_summary_var.set("No data for selected time frame.")
            self._draw_empty_insight_canvas(self.ai_timeline_canvas, "No valid AI history data in timeframe.")
            return

        self.ai_timeline_summary_var.set(f"Loaded {len(dates)} days of psychological insights.")
        self._draw_timeline_gantt(self.ai_timeline_canvas, cache, dates)

    def _refresh_ai_mood_tab(self, entries):
        if not hasattr(self, "ai_mood_summary_var"):
            return  # tab body not built yet
        self.ai_mood_summary_var.set("Loading mood cache...")
        self.ai_mood_canvas.update_idletasks()
        
        cache = self._get_daily_insights_cache()
        if not cache:
            self._draw_empty_insight_canvas(self.ai_mood_canvas, "No AI history to plot yet.")
            return

        dates = sorted(cache.keys())
        dates = self._filter_ai_dates(dates, self.ai_mood_temporal_var.get())
        
        valid_dates = [d for d in dates if "mood_score" in cache[d]]
        if len(valid_dates) < 2:
            self.ai_mood_summary_var.set("Not enough scored days.")
            self._draw_empty_insight_canvas(self.ai_mood_canvas, "Need at least 2 days of scored Moods.")
            return

        self.ai_mood_summary_var.set(f"Loaded {len(valid_dates)} successfully scored days.")
        self._draw_mood_sparkline(self.ai_mood_canvas, cache, valid_dates)

    def _draw_timeline_gantt(self, canvas, cache, dates):
        canvas.delete("all")
        canvas.update_idletasks()
        w = canvas.winfo_width()
        w = w if w > 10 else 600
        h = canvas.winfo_height()
        h = h if h > 10 else 180
        left, right, top, bottom = 100, 10, 15, 20
        cw = w - left - right
        ch = h - top - bottom
        
        if cw <= 0 or ch <= 0:
            return

        # Extract top 4 recurring themes/projects
        from collections import Counter
        all_tags = []
        for d in dates:
            all_tags.extend(cache[d].get("projects", []))
            all_tags.extend(cache[d].get("themes", []))
            
        top_tags = [tag.title() for tag, _ in Counter(tag.lower() for tag in all_tags if len(tag)>2).most_common(4)]
        
        if not top_tags:
            self._draw_empty_insight_canvas(canvas, "No recurring projects/themes detected.")
            return

        # Draw Y-axis labels
        row_h = ch / len(top_tags)
        for i, tag in enumerate(top_tags):
            y_center = top + i * row_h + (row_h / 2)
            short_tag = tag if len(tag) <= 20 else tag[:19] + "…"
            canvas.create_text(left - 8, y_center, text=short_tag, fill=COLOR_FG, font=("Arial", 9), anchor="e")
            canvas.create_line(left, y_center, left + cw, y_center, fill=COLOR_BORDER, dash=(2, 2))

        # Draw Gantt blocks on X-axis (days)
        day_w = cw / max(len(dates), 1)
        colors = ["#2dd4bf", "#818cf8", "#f472b6", "#fbbf24"]

        for d_i, day_str in enumerate(dates):
            day_data = cache[day_str]
            day_tags = [t.lower() for t in day_data.get("projects", [])] + [t.lower() for t in day_data.get("themes", [])]
            
            x1 = left + d_i * day_w
            x2 = x1 + day_w - 1 if day_w > 2 else x1 + day_w
            
            # Hover text tooltip transparent trigger column
            mood = day_data.get("mood", "Neutral")
            canvas.create_rectangle(x1, top, x2, top + ch, fill="", outline="", tags=("ai_col", f"ai_col_{d_i}"))
            
            def bind_hover(tag_id, hover_text):
                canvas.tag_bind(tag_id, "<Enter>", lambda e, text=hover_text: self._on_hist_hover_simple(canvas, e, text))
                canvas.tag_bind(tag_id, "<Leave>", lambda e: canvas.delete("tooltip", "tooltip_bg"))

            bind_hover(f"ai_col_{d_i}", f"{day_str}\nMood: {mood}")

            for i, target_tag in enumerate(top_tags):
                if target_tag.lower() in day_tags:
                    y1 = top + i * row_h + 4
                    y2 = top + (i + 1) * row_h - 4
                    col = colors[i % len(colors)]
                    
                    canvas.create_rectangle(x1, y1, x2, y2, fill=col, outline="", tags=("ai_col", f"ai_col_{d_i}"))

        # X-axis labels (first and last date)
        canvas.create_text(left, h - 5, text=dates[0][-5:], fill="#94a3b8", font=("Arial", 7), anchor="sw")
        canvas.create_text(left + cw, h - 5, text=dates[-1][-5:], fill="#94a3b8", font=("Arial", 7), anchor="se")

    def _draw_mood_sparkline(self, canvas, cache, dates):
        canvas.delete("all")
        canvas.update_idletasks()
        w = canvas.winfo_width()
        w = w if w > 10 else 600
        h = canvas.winfo_height()
        h = h if h > 10 else 180
        left, right, top, bottom = 40, 10, 15, 20
        cw = w - left - right
        ch = h - top - bottom
        
        if cw <= 0 or ch <= 0: return
        
        # Guidelines (Y-axis 0-100)
        canvas.create_text(left-5, top, text="100", fill=COLOR_FG, font=("Arial", 8), anchor="e")
        canvas.create_line(left, top, left+cw, top, fill=COLOR_BORDER, dash=(2,2))
        
        mid_y = top + ch/2
        canvas.create_text(left-5, mid_y, text="50", fill="#94a3b8", font=("Arial", 8), anchor="e")
        canvas.create_line(left, mid_y, left+cw, mid_y, fill=COLOR_BORDER, dash=(2,2))
        
        canvas.create_text(left-5, top+ch, text="0", fill=COLOR_FG, font=("Arial", 8), anchor="e")
        canvas.create_line(left, top+ch, left+cw, top+ch, fill=COLOR_BORDER, dash=(2,2))
        
        # Plot points
        step = cw / max(len(dates) - 1, 1)
        pts = []
        point_data = [] # for tooltips
        
        for i, d in enumerate(dates):
            score = cache[d].get("mood_score", 50)
            score = max(0, min(100, score))
            x = left + i * step
            y = top + ch - (score / 100.0) * ch
            pts.extend([x, y])
            point_data.append((x, y, d, score, cache[d].get("mood", "")))
            
        # Draw line
        canvas.create_line(*pts, fill="#ec4899", width=2, smooth=True)
        
        # Draw interactive points
        for x, y, d, score, mood in point_data:
            pt_id = canvas.create_oval(x-3, y-3, x+3, y+3, fill="#fbcfe8", outline="#be185d", tags="mood_pt")
            
            def bind_hover(tag_id, px, py, p_date, p_score, p_mood):
                canvas.tag_bind(tag_id, "<Enter>", lambda e, text=f"{p_date}\n{p_mood} ({p_score})": self._on_hist_hover_simple(canvas, type('obj', (object,), {'x': px, 'y': py}), text))
                canvas.tag_bind(tag_id, "<Leave>", lambda e: canvas.delete("tooltip", "tooltip_bg"))

            bind_hover(pt_id, x, y, d, score, mood)
            
        # X-axis labels
        canvas.create_text(left, h-5, text=dates[0][-5:], fill="#94a3b8", font=("Arial", 7), anchor="sw")
        canvas.create_text(left+cw, h-5, text=dates[-1][-5:], fill="#94a3b8", font=("Arial", 7), anchor="se")

    def _on_hist_hover_simple(self, canvas, event, text):
        canvas.delete("tooltip", "tooltip_bg")
        x, y = event.x, event.y
        padding = 5
        canvas.create_text(x, y - 10 - padding, text=text, fill=COLOR_FG, font=("Arial", 9, "bold"), anchor="s", tags="tooltip")
        bbox = canvas.bbox("tooltip")
        if bbox:
            canvas.create_rectangle(bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding,
                                    fill=COLOR_ACCENT_BG, outline=COLOR_BORDER, tags="tooltip_bg")
            canvas.tag_raise("tooltip")

    def _refresh_activity_matrix_tab(self, entries, all_text):
        if not hasattr(self, "activity_canvas"):
            return  # tab body not built yet
        self.activity_canvas.delete("all")
        self.activity_canvas.update_idletasks()
        if not entries:
            self._draw_empty_insight_canvas(self.activity_canvas, "No data yet.")
            return

        from collections import defaultdict
        import datetime
        
        # Group entries by day
        metric_type = self.activity_metric_var.get()
        daily_vals = defaultdict(float)
        
        if metric_type == "Mood":
            cache = self._get_daily_insights_cache()
            for d, data in cache.items():
                if "mood_score" in data:
                    daily_vals[d] = float(data["mood_score"])
        else: # Words (Volume)
            for e in entries:
                d = e["timestamp"].strftime("%Y-%m-%d")
                daily_vals[d] += len(self._tokenize_words(e["text"]))
                
        self._draw_github_heatmap(self.activity_canvas, daily_vals, metric_type)

    def _refresh_speaker_profile_tab(self, entries, all_tokens, sentences):
        if not hasattr(self, "speaker_profile_canvas"):
            return  # tab body not built yet
        self.speaker_profile_canvas.delete("all")
        self.speaker_profile_canvas.update_idletasks()
        if not entries or len(all_tokens) < 10:
            self._draw_empty_insight_canvas(self.speaker_profile_canvas, "Need more spoken data to build profile.")
            return

        import math
        # Axis 1: Lexical Diversity
        unique_words = len(set(t.lower() for t in all_tokens))
        diversity_score = min(100, (unique_words / max(len(all_tokens), 1)) * 300) # heuristic
        
        # Axis 2: Engagement Volume
        today_words = self.today_words
        volume_score = min(100, (today_words / 2000.0) * 100) # baseline 2000 words a day = 100%
        
        # Axis 3: Positivity (Mood)
        cache = self._get_daily_insights_cache()
        mood_scores = [d.get("mood_score", 50) for d in cache.values() if "mood_score" in d]
        positivity_score = sum(mood_scores) / len(mood_scores) if mood_scores else 50
        
        # Axis 4: Inquiry (Questions)
        questions = sum(1 for s in sentences if s.strip().endswith('?'))
        inquiry_score = min(100, (questions / max(len(sentences), 1)) * 500) # heuristic
        
        # Axis 5: Sustained Thought (Sentence Length)
        avg_len = len(all_tokens) / max(len(sentences), 1)
        thought_score = min(100, (avg_len / 25.0) * 100)
        
        axes_data = {
            "Lexical Diversity": max(10, diversity_score),
            "Daily Engagement": max(10, volume_score),
            "Positivity": max(10, positivity_score),
            "Inquiry Drive": max(10, inquiry_score),
            "Sustained Thought": max(10, thought_score)
        }
        
        self._draw_radar_chart(self.speaker_profile_canvas, axes_data)

    def _refresh_daily_brief_tab(self, entries):
        if not hasattr(self, "brief_listbox"):
            return  # tab body not built yet
        cache = self._get_daily_insights_cache()
        self.brief_listbox.delete(0, tk.END)
        if not cache:
            self.brief_listbox.insert(tk.END, "No AI history available.")
            return
            
        dates = sorted(cache.keys(), reverse=True)
        for d in dates:
            score = cache[d].get("mood_score", "?")
            self.brief_listbox.insert(tk.END, f"{d}  [M: {score}]")
            
        # Select first by default if available
        if dates:
            self.brief_listbox.selection_set(0)
            self._on_brief_date_select()

    def _on_brief_date_select(self, event=None):
        sel = self.brief_listbox.curselection()
        if not sel: return
        
        item_text = self.brief_listbox.get(sel[0])
        if "available" in item_text: return
        
        date_str = item_text.split()[0]
        cache = self._get_daily_insights_cache()
        data = cache.get(date_str, {})
        
        self.brief_text.config(state="normal")
        self.brief_text.delete("1.0", tk.END)
        
        if not data:
            self.brief_text.insert(tk.END, "Data missing for this date.")
            self.brief_text.config(state="disabled")
            return
            
        mood = data.get("mood", "Neutral")
        score = data.get("mood_score", "N/A")
        brief = data.get("daily_brief", "No brief generated for this date. Delete the ai_insights.json file and crunch history again to generate briefs.")
        projects = ", ".join(data.get("projects", []))
        themes = ", ".join(data.get("themes", []))
        
        display_text = f"DATE: {date_str}\n"
        display_text += f"{'='*30}\n\n"
        display_text += f"MOOD: {mood} (Score: {score}/100)\n\n"
        display_text += f"DAILY BRIEF:\n{brief}\n\n"
        display_text += f"{'-'*30}\n"
        display_text += f"PROJECTS: {projects}\n"
        display_text += f"THEMES: {themes}\n"
        
        self.brief_text.insert(tk.END, display_text)
        self.brief_text.config(state="disabled")

    def _draw_github_heatmap(self, canvas, data_dict, metric_type):
        import datetime
        w = canvas.winfo_width()
        w = w if w > 10 else 600
        h = canvas.winfo_height()
        h = h if h > 10 else 180
        left, top = 40, 20
        sq_size = 12
        pad = 3
        
        if not data_dict: return
        
        dates = sorted(data_dict.keys())
        try:
            start_date = datetime.datetime.strptime(dates[0], "%Y-%m-%d").date()
            end_date = datetime.date.today()
        except Exception:
            return
            
        # Limit to max 52 weeks (approx 1 year backward from end_date)
        start_date = max(start_date, end_date - datetime.timedelta(days=365))
        
        # Adjust start_date to a Monday
        while start_date.weekday() != 0:
            start_date -= datetime.timedelta(days=1)
            
        total_days = (end_date - start_date).days + 1
        num_cols = (total_days // 7) + 1
        
        # Draw Y-Axis (Days)
        day_labels = ["Mon", "", "Wed", "", "Fri", "", "Sun"]
        for i, lbl in enumerate(day_labels):
            y = top + i * (sq_size + pad) + (sq_size/2)
            canvas.create_text(left - 5, y, text=lbl, fill="#94a3b8", font=("Arial", 8), anchor="e")
            
        max_val = max(data_dict.values()) if data_dict else 1
        if max_val == 0: max_val = 1
        
        curr_date = start_date
        col = 0
        point_data = [] # for tooltips
        
        while curr_date <= end_date:
            d_str = curr_date.strftime("%Y-%m-%d")
            val = data_dict.get(d_str, 0)
            
            row = curr_date.weekday() # 0 = Mon, 6 = Sun
            x = left + col * (sq_size + pad)
            y = top + row * (sq_size + pad)
            
            # Colors
            if val == 0:
                color = "#1e293b" # Empty dark
            else:
                ratio = val / max_val
                if metric_type == "Mood":
                    # Red to Green
                    if val < 40: color = "#ef4444" 
                    elif val < 60: color = "#eab308"
                    else: color = "#22c55e"
                    # Add intensity
                    if ratio > 0.8: color = "#16a34a" if val >= 60 else "#dc2626"
                else:
                    # GitHub Greens
                    if ratio < 0.25: color = "#0e4429"
                    elif ratio < 0.5: color = "#006d32"
                    elif ratio < 0.75: color = "#26a641"
                    else: color = "#39d353"
                    
            r_id = canvas.create_rectangle(x, y, x+sq_size, y+sq_size, fill=color, outline="#0f172a")
            
            if val > 0:
                point_data.append((r_id, x, y, d_str, val))
                
            if row == 6: col += 1
            curr_date += datetime.timedelta(days=1)
            
        # Tooltips
        for r_id, x, y, d, v in point_data:
            def bind_hover(tag_id, px, py, p_date, p_val):
                lbl_text = f"{p_date}: {p_val:.1f} {metric_type}"
                canvas.tag_bind(tag_id, "<Enter>", lambda e, text=lbl_text: self._on_hist_hover_simple(canvas, type('obj', (object,), {'x': px, 'y': py}), text))
                canvas.tag_bind(tag_id, "<Leave>", lambda e: canvas.delete("tooltip", "tooltip_bg"))
            bind_hover(r_id, x, y, d, v)

    def _draw_radar_chart(self, canvas, data_dict):
        import math
        w = canvas.winfo_width()
        w = w if w > 10 else 600
        h = canvas.winfo_height()
        h = h if h > 10 else 220
        cx, cy = w/2, h/2
        radius = min(w/2, h/2) - 40
        
        labels = list(data_dict.keys())
        values = list(data_dict.values())
        num_axes = len(labels)
        angle_step = 2 * math.pi / num_axes
        
        # Grid layers (polygons)
        for r_ratio in [0.25, 0.5, 0.75, 1.0]:
            r = radius * r_ratio
            pts = []
            for i in range(num_axes):
                a = i * angle_step - math.pi/2 # Start at top
                pts.extend([cx + r * math.cos(a), cy + r * math.sin(a)])
            canvas.create_polygon(pts, fill="", outline="#334155", dash=(2,2))
            
        # Draw axes and labels
        axis_pts = []
        for i in range(num_axes):
            a = i * angle_step - math.pi/2
            x, y = cx + radius * math.cos(a), cy + radius * math.sin(a)
            canvas.create_line(cx, cy, x, y, fill="#475569")
            
            # Label
            lx, ly = cx + (radius+15) * math.cos(a), cy + (radius+15) * math.sin(a)
            anchor = "center"
            if math.cos(a) > 0.1: anchor = "w"
            elif math.cos(a) < -0.1: anchor = "e"
            canvas.create_text(lx, ly, text=labels[i], fill="#94a3b8", font=("Arial", 8, "bold"), anchor=anchor)
            
            # Data point (cap between 0.1 and 1.0)
            val = max(0.1, min(1.0, values[i] / 100.0))
            dx, dy = cx + (radius * val) * math.cos(a), cy + (radius * val) * math.sin(a)
            axis_pts.extend([dx, dy])
            
        # Data Polygon
        if axis_pts:
            canvas.create_polygon(axis_pts, fill="#2dd4bf", outline="#14b8a6", width=2, stipple="gray50")
            for i in range(0, len(axis_pts), 2):
                canvas.create_oval(axis_pts[i]-3, axis_pts[i+1]-3, axis_pts[i]+3, axis_pts[i+1]+3, fill="#f0fdfa", outline="#0d9488")

    def toggle_insights(self):
        """Toggle the collapsible Speech Insights accordion; compute lazily."""
        if self.insights_collapsed.get():
            self.insights_body.pack(fill="x")
            self.insights_toggle_btn.config(text="\u25bc")
            self.insights_collapsed.set(False)
            built = self._ensure_insight_tab_built(self.insight_tab_index)
            if self._insights_dirty or built:
                self.refresh_insights_tabs()
            # The histograms live in this body, so they were drawn against an
            # unmapped (1px) canvas until now.
            self.root.after(80, self.update_histogram)
            self.root.after(80, lambda: self._update_metric_bands(
                getattr(self, "_insights_snapshot", None) or {}))
        else:
            self.insights_body.pack_forget()
            self.insights_toggle_btn.config(text="\u25b6")
            self.insights_collapsed.set(True)
        self.config['insights_collapsed'] = self.insights_collapsed.get()
        self._persist_config()
        self.root.after(50, self._update_scroll_region)

    def toggle_settings(self):
        """Toggle the collapsible Configuration accordion."""
        if self.config_collapsed.get():
            self.settings_content_card.pack(fill="x")
            self.settings_content.pack(fill="both", expand=True, padx=8, pady=8)
            self.settings_toggle_btn.config(text="\u25bc")
            self.config_collapsed.set(False)
            self._update_config_summary()
        else:
            self.settings_content.pack_forget()
            self.settings_content_card.pack_forget()
            self.settings_toggle_btn.config(text="\u25b6")
            self.config_collapsed.set(True)
        self.config['config_collapsed'] = self.config_collapsed.get()
        self._persist_config()
        # Refresh scroll region after content change
        self.root.after(50, self._update_scroll_region)

    def toggle_log(self):
        """Toggle the collapsible System Log accordion (widget built lazily)."""
        if self.log_collapsed.get():
            self._build_log_widget()
            self.log_body.pack(fill="x")
            self.log_toggle_btn.config(text="\u25bc")
            self.log_collapsed.set(False)
        else:
            self.log_body.pack_forget()
            self.log_toggle_btn.config(text="\u25b6")
            self.log_collapsed.set(True)
        self.root.after(50, self._update_scroll_region)

    def _build_log_widget(self):
        """Create the ScrolledText on first expand and flush the buffer."""
        if self.text_area is not None:
            return
        self.text_area = scrolledtext.ScrolledText(
            self.log_body, state='disabled', font=("Consolas", 9),
            bg=COLOR_TEXT_BG, fg=COLOR_FG, insertbackground='white',
            height=8, relief="flat", highlightthickness=1,
            highlightbackground=COLOR_BORDER)
        self.text_area.pack(fill="x", padx=self.CARD_PAD, pady=self.CARD_PAD)
        if self._log_buffer:
            self.text_area.config(state='normal')
            for line in self._log_buffer:
                self.text_area.insert('1.0', line)
            self.text_area.config(state='disabled')
            self._log_buffer.clear()

    def _append_log_line(self, line):
        """Route one log line to the ScrolledText, buffering it until the
        System Log accordion has been expanded for the first time."""
        self._log_line_count += 1
        low = line.lower()
        if any(m in low for m in ("error", "fail", "warn", "exception")):
            self._log_warn_count += 1
        if self.text_area is None:
            self._log_buffer.append(line)
        else:
            self.text_area.config(state='normal')
            self.text_area.insert('1.0', line)
            last_line = int(self.text_area.index('end-1c').split('.')[0])
            if last_line > 400:
                self.text_area.delete("301.0", "end")
            self.text_area.config(state='disabled')
        self._update_log_summary()

    def _update_log_summary(self):
        warn = self._log_warn_count
        warn_txt = "no warnings" if warn == 0 else f"{warn} warning{'s' if warn != 1 else ''}"
        self.log_summary_var.set(f"{warn_txt} \u00b7 {self._log_line_count} lines")

    def _focus_configuration(self):
        """Header CFG button: expand Configuration and scroll to it."""
        if self.config_collapsed.get():
            self.toggle_settings()
        self.root.after(80, self._scroll_to_configuration)

    def _scroll_to_configuration(self):
        """Scroll the Configuration accordion header into view.

        Not yview_moveto(1.0): the System Log accordion is packed after
        Configuration, so scrolling to the bottom overshoots the panel the
        CFG button just opened.
        """
        try:
            self._scroll_canvas.update_idletasks()
            y = self.settings_toggle_btn.winfo_rooty() - self.content_frame.winfo_rooty()
            total = max(1, self.content_frame.winfo_height())
            self._scroll_canvas.yview_moveto(max(0.0, min(1.0, y / total)))
        except Exception:
            try:
                self._scroll_canvas.yview_moveto(1.0)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # OpenAI API key: first-run welcome and the inline AI-Insights notice
    # ------------------------------------------------------------------

    def _focus_api_key(self):
        """Expand Configuration, scroll to it and put the caret in the API-key
        entry. Shared by the welcome dialog and the AI Insights notice."""
        try:
            self._focus_configuration()
        except Exception:
            pass

        def _focus_entry():
            entry = getattr(self, "api_key_entry", None)
            try:
                if entry is not None and entry.winfo_exists():
                    entry.focus_set()
                    entry.selection_range(0, "end")
            except Exception:
                pass

        self._ui_after(200, _focus_entry)

    def _has_openai_key(self):
        return bool(str(self.config.get('openai_api_key', '') or '').strip())

    def _update_api_key_notice(self):
        """Show the inline 'needs an API key' banner on the AI Insights tab
        only while no key is configured. Safe to call before the tab is built."""
        frame = getattr(self, "ai_key_notice_frame", None)
        if frame is None:
            return
        try:
            if not frame.winfo_exists():
                return
            if self._has_openai_key():
                frame.pack_forget()
            elif not frame.winfo_ismapped():
                frame.pack(fill="x", padx=10, pady=(8, 0),
                           before=self.ai_timeline_desc_label)
        except Exception:
            pass

    def _mark_welcome_shown(self):
        """Persist the 'the user has seen the welcome' flag exactly once."""
        if self.config.get('welcome_shown'):
            return
        self.config['welcome_shown'] = True
        try:
            self._persist_config()
        except Exception as e:
            log.error("[Welcome] Could not persist welcome_shown: %s", e)

    def _maybe_show_welcome(self):
        """First launch only: explain what works now and what a key unlocks."""
        if self.config.get('welcome_shown'):
            return
        if NO_DIALOGS:
            self._mark_welcome_shown()
            return
        try:
            self._show_welcome_dialog()
        except Exception as e:
            # A dialog that cannot open must not leave the flag unset (it would
            # try again on every start), and must never take the app down.
            log.error("[Welcome] Could not show the welcome dialog: %s", e)
            self._mark_welcome_shown()

    def _show_welcome_dialog(self):
        """Small dark-theme customtkinter greeting shown once, on first run."""
        if getattr(self, "_welcome_dialog", None) is not None:
            return
        live_hk = str(self.config.get('hotkey_live', DEFAULT_CONFIG['hotkey_live'])).upper()
        batch_hk = str(self.config.get('hotkey_batch', DEFAULT_CONFIG['hotkey_batch'])).upper()
        local_name = os.path.basename(self._local_config_file())

        dialog = ctk.CTkToplevel(self.root)
        self._welcome_dialog = dialog
        dialog.title("Welcome to neurowhisper")
        dialog.configure(fg_color=COLOR_BG)
        dialog.resizable(False, False)
        try:
            dialog.transient(self.root)
        except Exception:
            pass

        wrap = ctk.CTkFrame(dialog, fg_color=COLOR_BG)
        wrap.pack(fill="both", expand=True, padx=20, pady=18)

        ctk.CTkLabel(wrap, text="Welcome to neurowhisper", text_color=COLOR_TEAL,
                     font=ctk.CTkFont(size=16, weight="bold"),
                     anchor="w").pack(fill="x", pady=(0, 10))

        paragraphs = [
            f"Dictation already works, entirely on this machine. Press {live_hk} to type "
            f"while you speak, or {batch_hk} to record first and transcribe in one go.",
            "Optional: your own OpenAI API key unlocks cloud transcription (Online mode) "
            "and AI Insights - short daily summaries of what you dictated, which the "
            "analysis/ toolkit turns into a report. Create one at "
            f"{OPENAI_KEY_URL.split('//', 1)[-1]} - it is stored only on this "
            f"machine, in {local_name}.",
            "No hurry: you can add or remove the key at any time under Configuration.",
        ]
        for text in paragraphs:
            ctk.CTkLabel(wrap, text=text, text_color=COLOR_FG,
                         font=ctk.CTkFont(size=12), justify="left", anchor="w",
                         wraplength=400).pack(fill="x", pady=(0, 10))

        btn_row = ctk.CTkFrame(wrap, fg_color=COLOR_BG)
        btn_row.pack(fill="x", pady=(4, 0))

        def _close(add_key=False):
            self._mark_welcome_shown()
            self._welcome_dialog = None
            try:
                if dialog.winfo_exists():
                    dialog.grab_release()
                    dialog.destroy()
            except Exception:
                pass
            if add_key:
                self._focus_api_key()

        ctk.CTkButton(btn_row, text="Later", width=90, height=30, corner_radius=8,
                      fg_color=COLOR_BG, hover_color=COLOR_TEAL_DIM,
                      border_width=1, border_color=COLOR_BORDER,
                      text_color=COLOR_TEXT_MUTED,
                      command=lambda: _close(False)).pack(side="right")
        ctk.CTkButton(btn_row, text="Add API key now", width=150, height=30,
                      corner_radius=8, fg_color=COLOR_TEAL, hover_color=COLOR_TEAL_DIM,
                      text_color=COLOR_BG, font=ctk.CTkFont(size=12, weight="bold"),
                      command=lambda: _close(True)).pack(side="right", padx=(0, 8))

        dialog.protocol("WM_DELETE_WINDOW", lambda: _close(False))

        try:
            dialog.update_idletasks()
            w = max(440, dialog.winfo_reqwidth())
            h = max(300, dialog.winfo_reqheight())
            x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - w) // 2)
            y = self.root.winfo_rooty() + 80
            dialog.geometry(f"{w}x{h}+{x}+{y}")
            if self.config.get('always_on_top'):
                dialog.attributes('-topmost', True)
            dialog.lift()
            dialog.focus_force()
        except Exception:
            pass

    def _toggle_always_on_top(self):
        self.top_var.set(not self.top_var.get())
        self.on_top_change()

    def _update_state_card(self):
        """State dot, state word, elapsed timer and the VU meter colour."""
        if not hasattr(self, "state_word_var"):
            return
        if self.mode != getattr(self, "_state_prev_mode", "?"):
            self._state_prev_mode = self.mode
            self._mode_started_at = time.time() if self.mode else None
        if self.mode == "live":
            word, color = "Live typing", COLOR_LIVE
        elif self.mode == "batch":
            word, color = "Batch recording", COLOR_BATCH
        elif self._is_online_mode():
            word = "Recording + edit" if self.online_with_edit else "Recording"
            color = COLOR_ONLINE
        else:
            word, color = "Ready", COLOR_READY
        self.state_word_var.set(word)
        self._set_state_dot(color)
        started = getattr(self, "_mode_started_at", None)
        if self.mode and started:
            elapsed = int(max(0.0, time.time() - started))
            self.state_elapsed_var.set(f"{elapsed // 60}:{elapsed % 60:02d}")
        else:
            self.state_elapsed_var.set("")
        try:
            self.vu_meter.configure(progress_color=self._vu_color())
        except Exception:
            pass

    def on_setting_change(self, *args):
        # Auto-save pause setting (debounced by trace)
        try:
            self.config['live_pause'] = float(self.live_pause_var.get())
            # Mirror for the audio thread (see __init__): it must not touch
            # the Tk variable itself.
            self._live_pause = self.config['live_pause']
            self._persist_config()
            self._update_config_summary()
        except Exception:
            pass  # Ignore errors during typing

    def on_top_change(self):
        self.config['always_on_top'] = self.top_var.get()
        self.root.attributes('-topmost', self.config['always_on_top'])
        self._persist_config()
        if hasattr(self, "btn_header_top"):
            on = self.top_var.get()
            self.btn_header_top.configure(
                fg_color=COLOR_TEAL_DIM if on else COLOR_ACCENT_BG,
                border_color=COLOR_TEAL if on else COLOR_BORDER,
                text_color=COLOR_TEAL if on else COLOR_TEXT_MUTED)

    def on_mic_change(self, event=None):
        device_name = self.device_var.get()
        try:
            devices = list(sd.query_devices())
        except Exception as e:
            log.error("Could not enumerate audio devices: %s", e)
            self.log_internal(f"Could not enumerate audio devices: {e}")
            return
        for i, d in enumerate(devices):
            if d['name'] == device_name and d['max_input_channels'] > 0:
                self.config['input_device'] = i
                break
        self._persist_config()
        self._update_device_model_labels()
        self.restart_audio_stream()

    def on_backend_change(self, event=None):
        """Handle backend/acceleration selection change."""
        display_val = self.backend_var.get()
        
        # Map display names to (backend, device) pairs
        # Note: 'cuda' is NOT a backend, it's a device. Backend is the transcription engine.
        backend_device_map = BACKEND_DISPLAY_TO_BACKEND_DEVICE
        
        new_backend, new_device = backend_device_map.get(display_val, ("faster-whisper", "auto"))
        
        if new_backend != self.config.get('backend') or new_device != self.config.get('device'):
            self.config['backend'] = new_backend
            self.config['device'] = new_device
            # Set appropriate compute type
            if new_device == 'auto':
                self.config['compute_type'] = 'auto'
            else:
                self.config['compute_type'] = 'float16' if new_device == 'cuda' else 'int8'
            self._persist_config()
            
            messagebox.showinfo("Acceleration Changed", 
                f"Acceleration changed to: {display_val}\n\nRestart the app to apply the new mode.")

    def _update_mode_buttons(self):
        """Show/hide appropriate buttons based on transcription mode."""
        # Hide all buttons first
        for widget in [self.btn_batch, self.separator, self.btn_live, 
                       self.btn_transcribe, self.btn_transcribe_edit]:
            widget.grid_forget()

        # grid with uniform columns: both buttons get exactly half the row,
        # regardless of label length (pack would size them by text width).
        self.toggle_inner.columnconfigure(0, weight=1, uniform="mode")
        self.toggle_inner.columnconfigure(1, weight=0)
        self.toggle_inner.columnconfigure(2, weight=1, uniform="mode")
        if self.transcription_mode.get() == 'local':
            left, right = self.btn_live, self.btn_batch
        else:
            left, right = self.btn_transcribe, self.btn_transcribe_edit
        left.grid(row=0, column=0, sticky="ew")
        self.separator.grid(row=0, column=1)
        right.grid(row=0, column=2, sticky="ew")
        self._update_device_model_labels()

    def switch_transcription_mode(self, new_mode):
        """Switch between local and online transcription modes."""
        if new_mode == self.transcription_mode.get():
            return
        
        # Check if online mode requires API key
        if new_mode == 'online':
            if not self.config.get('openai_api_key'):
                messagebox.showwarning("API Key Required", 
                    "Please configure your OpenAI API key in the settings.\n\n"
                    "Go to Configuration > OpenAI Settings to add your key.")
                return
        
        # Stop any active recording first: otherwise self.mode stays set
        # underneath the freshly swapped mode buttons.
        self._stop_any_recording()

        self.transcription_mode.set(new_mode)
        self.config['transcription_mode'] = new_mode

        # Save config
        self._persist_config()
        
        # Update button appearances
        if new_mode == 'local':
            self.btn_local.configure(fg_color=COLOR_TEAL, text_color=COLOR_BG)
            self.btn_online.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_FAINT)
        else:
            self.btn_local.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_FAINT)
            self.btn_online.configure(fg_color=COLOR_ONLINE, text_color=COLOR_FG)
        
        # Show appropriate action buttons
        self._update_mode_buttons()
        
        self._update_device_model_labels()
        self.msg_queue.put(f"Switched to {new_mode.upper()} mode")

    def _queue_hotkey_action(self, action):
        """Queue hotkey actions so UI work always runs on the Tk main thread."""
        if not self.running:
            return

        now = time.monotonic()
        last = self._last_hotkey_event_time.get(action, 0.0)
        if now - last < HOTKEY_RUNTIME["debounce_seconds"]:
            return
        self._last_hotkey_event_time[action] = now

        try:
            self.hotkey_action_queue.put_nowait(action)
        except queue.Full:
            self.log_internal(f"Hotkey queue full, dropped: {action}")

    def _handle_batch_hotkey(self):
        """Mode-aware handler for batch/transcribe hotkey."""
        self._queue_hotkey_action('batch')

    def _handle_live_hotkey(self):
        """Mode-aware handler for live/transcribe+edit hotkey."""
        self._queue_hotkey_action('live')

    def _expected_hotkeys(self):
        return {
            'live': self.config.get('hotkey_live', HOTKEY_LABELS["live"].lower()),
            'batch': self.config.get('hotkey_batch', HOTKEY_LABELS["batch"].lower()),
        }

    def _set_hotkey_health_indicator(self, healthy, reason="ok"):
        """Update the hotkey health label in Settings."""
        if not hasattr(self, "hotkey_health_var") or not hasattr(self, "hotkey_health_label"):
            return

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        if healthy:
            self.hotkey_health_var.set(f"OK ({timestamp})")
            self.hotkey_health_label.config(fg=COLOR_READY)
            return

        short_reason = (reason or "unknown issue").strip()
        if len(short_reason) > 44:
            short_reason = short_reason[:41] + "..."
        self.hotkey_health_var.set(f"Issue: {short_reason}")
        self.hotkey_health_label.config(fg=COLOR_LIVE)

    def _set_hotkey_auto_rebind_indicator(self):
        """Update automatic rebind count label."""
        if hasattr(self, "hotkey_auto_rebinds_var"):
            count = getattr(self, "_hotkey_auto_rebind_count", 0)
            mgr = getattr(self, "_hotkey_manager", None)
            restarts = mgr.total_listener_restarts if mgr else 0
            label = f"Auto-Rebinds: {count}"
            if restarts > 0:
                label += f" | Restarts: {restarts}"
            self.hotkey_auto_rebinds_var.set(label)

    def _register_hotkeys(self):
        """Register global hotkeys using tracked manager state."""
        bindings = {
            'live': (self.config['hotkey_live'], self._handle_live_hotkey),
            'batch': (self.config['hotkey_batch'], self._handle_batch_hotkey),
        }
        self._hotkey_manager.register_all(bindings)
        health = self._hotkey_manager.health_status(self._expected_hotkeys())
        self._set_hotkey_health_indicator(health["healthy"], health["reason"])

        # First registration only: if the hooks did not take, say so out loud.
        # Silent hotkey failure is indistinguishable from "the app is broken".
        if getattr(self, "_hotkey_warning_shown", False):
            return
        reason = None
        if KEYBOARD_ERROR:
            reason = KEYBOARD_ERROR
        elif getattr(self._hotkey_manager, "_install_error", None):
            reason = self._hotkey_manager._install_error
        elif not health.get("healthy", True):
            reason = health.get("reason", "unknown")
        if reason is None:
            return
        self._hotkey_warning_shown = True
        message = (
            f"Hotkeys could not be registered: {reason}. "
            "Try running as administrator or check antivirus settings."
        )
        self.log_internal(f"⚠️ {message}")
        log.error("%s", message)
        if not NO_DIALOGS:
            self._ui_after(0, lambda: messagebox.showwarning("Hotkeys unavailable", message))

    def _unregister_hotkeys(self):
        """Unregister global hotkeys using tracked manager state."""
        self._hotkey_manager.unregister_all()

    def _refresh_hotkeys(self, reason, user_initiated=False):
        """Hard refresh all hotkey hooks and verify registration health."""
        if reason:
            self.log_internal(f"Hotkey refresh requested: {reason}")
        if not user_initiated:
            self._hotkey_auto_rebind_count += 1
            self._set_hotkey_auto_rebind_indicator()
        self._unregister_hotkeys()
        self._register_hotkeys()
        health = self._hotkey_manager.health_status(self._expected_hotkeys())
        self._set_hotkey_health_indicator(health["healthy"], health["reason"])
        if health["healthy"]:
            self.log_internal("Hotkey refresh complete")
            if user_initiated:
                self.msg_queue.put("Hotkeys refreshed")
            return True
        self.log_internal(f"Hotkey refresh incomplete: {health['reason']}")
        if user_initiated:
            self.msg_queue.put("Hotkey refresh failed - check logs")
        return False

    def _start_hotkey_watchdog(self):
        """Start periodic hotkey health check with dead-hook detection."""

        def check_hotkeys():
            if not self.running:
                return

            needs_reregister = False
            needs_listener_restart = False
            reason = ""
            health = self._hotkey_manager.health_status(self._expected_hotkeys())
            self._set_hotkey_health_indicator(health["healthy"], health["reason"])

            if health.get("hook_dead"):
                # Windows silently removed the WH_KEYBOARD_LL hook.
                # Need a full listener restart to reinstall it.
                needs_listener_restart = True
                reason = health["reason"]
            elif not health["healthy"]:
                needs_reregister = True
                reason = health["reason"]

            if needs_listener_restart:
                self.log_internal(f"Hotkey issue detected – restarting listener ({reason})")
                self._hotkey_manager._force_restart_listener()
                self._refresh_hotkeys(f"watchdog listener restart ({reason})")
            elif needs_reregister:
                self._refresh_hotkeys(f"watchdog ({reason})")

            # Schedule next check using configured interval
            if self.running:
                self.root.after(self._hotkey_watchdog_interval, check_hotkeys)

        # Start first check after interval
        self.root.after(self._hotkey_watchdog_interval, check_hotkeys)

    def cleanup_hotkeys(self):
        """Clean up hotkeys before app exit."""
        self.log_internal("Cleaning up hotkeys...")
        self._hotkey_manager.cleanup()
        self._clear_capture_hooks()
        # Also unhook any capture hooks
        if hasattr(self, '_capture_hook') and self._capture_hook:
            try:
                keyboard.unhook(self._capture_hook)
            except Exception:
                pass
            self._capture_hook = None

    def _clear_capture_hooks(self):
        """Remove temporary hooks used during hotkey capture."""
        hooks = getattr(self, '_capture_hooks', None)
        if hooks:
            for hook in hooks:
                try:
                    keyboard.unhook(hook)
                except Exception:
                    pass
            self._capture_hooks = None

    def _on_focus_in(self, event=None):
        """Check hotkey health when window gains focus.

        Instead of blindly re-registering (which was causing corruption),
        just run a health check.  The watchdog will handle any detected issues.
        """
        current_time = time.time()

        # Avoid spamming checks - minimum 5 seconds between focus-based checks
        if current_time - self._last_focus_refresh < HOTKEY_RUNTIME["focus_refresh_min_interval_seconds"]:
            return

        self._last_focus_refresh = current_time

        # Just update the health indicator; the watchdog handles restarts.
        health = self._hotkey_manager.health_status(self._expected_hotkeys())
        self._set_hotkey_health_indicator(health["healthy"], health["reason"])

    def _manual_hotkey_refresh(self):
        """Manual hotkey refresh triggered by user clicking the refresh button.

        Always does a full listener restart to guarantee recovery from
        any state where Windows has silently dropped the hooks.
        """
        self.log_internal("Manual hotkey rebind requested – doing full listener restart")
        self.msg_queue.put("Restarting keyboard listener...")
        self._hotkey_manager._force_restart_listener()
        self._refresh_hotkeys("manual settings action", user_initiated=True)

        # Update timestamp to prevent immediate watchdog refresh
        import time as time_module
        self._last_focus_refresh = time_module.time()

    def _hotkey_pump(self):
        """Cheap 50ms loop that only drains the hotkey queue, so hotkey presses
        respond instantly even while the main GUI loop idles at 1000ms."""
        if not self.running:
            return
        self._process_hotkey_actions()
        self.root.after(50, self._hotkey_pump)

    def _process_hotkey_actions(self):
        """Process queued hotkey actions on the Tk main thread."""
        while not self.hotkey_action_queue.empty():
            try:
                action = self.hotkey_action_queue.get_nowait()
            except queue.Empty:
                break

            try:
                if action == 'batch':
                    if self.transcription_mode.get() == 'online':
                        self.toggle_online_transcribe()
                    else:
                        self.toggle_batch_mode()
                elif action == 'live':
                    if self.transcription_mode.get() == 'online':
                        self.toggle_online_transcribe_edit()
                    else:
                        self.toggle_live_mode()
            except Exception as e:
                self.log_internal(f"Hotkey action failed ({action}): {e}")

    def toggle_online_transcribe(self):
        """Toggle online transcription (record and send to OpenAI)."""
        if self.mode == 'online_transcribe':
            # Stop recording and transcribe
            self._stop_online_recording(with_edit=False)
        elif self.mode is None:
            # Start recording
            self._start_online_recording(with_edit=False)
        else:
            # Already in another mode - stop it (any key = stop)
            self._stop_any_recording()

    def toggle_online_transcribe_edit(self):
        """Toggle online transcription with GPT editing."""
        if self.mode == 'online_transcribe_edit':
            # Stop recording and transcribe+edit
            self._stop_online_recording(with_edit=True)
        elif self.mode is None:
            # Start recording
            self._start_online_recording(with_edit=True)
        else:
            # Already in another mode - stop it (any key = stop)
            self._stop_any_recording()

    def _start_online_recording(self, with_edit=False):
        """Start recording for online transcription with streaming segments."""
        if not self.start_audio_stream():
            return

        mode_name = 'online_transcribe_edit' if with_edit else 'online_transcribe'
        self.mode = mode_name
        self.online_with_edit = with_edit

        # Reset streaming state
        self.online_segments = []
        self.online_segment_seq = 0
        self.online_pending_audio = []
        self.online_silence_count = 0

        # One OpenAI client for the whole session (no TLS handshake per segment)
        try:
            self.online_backend = self._get_online_backend()
        except Exception as e:
            self.online_backend = None
            self.log_internal(f"OpenAI client init failed: {e}")

        # Create thread pool for parallel API calls (2 workers for API rate limits)
        from concurrent.futures import ThreadPoolExecutor
        self.online_executor = ThreadPoolExecutor(max_workers=AUDIO_RUNTIME["online_executor_workers"])
        
        # Update UI
        btn = self.btn_transcribe_edit if with_edit else self.btn_transcribe
        btn.configure(fg_color=COLOR_ONLINE, text_color=COLOR_FG)
        
        self.play_feedback_sound(start=True)
        self.status_var.set(f"Recording{' + GPT edit' if with_edit else ''}...")
        self.msg_queue.put(f"Recording started (online streaming mode)")

    def _stop_online_recording(self, with_edit=False):
        """Stop recording, process final segment, and combine all results."""
        self.play_feedback_sound(start=False)
        
        # Queue any remaining audio
        if self.online_pending_audio:
            self._queue_online_segment()
        
        # Check if we have any segments
        if not self.online_segments:
            self.mode = None
            self.stop_audio_stream()
            self._reset_online_buttons()
            return

        self.mode = None
        self.stop_audio_stream()

        # Show TRANSCRIBING state
        btn = self.btn_transcribe_edit if with_edit else self.btn_transcribe
        btn.configure(fg_color=COLOR_TRANSCRIBING, text_color="white", text="Finishing...")
        self.status_var.set("Waiting for transcriptions...")
        self.update_mini_window_color(COLOR_TRANSCRIBING)
        
        # Wait for all segments to complete in background
        def wait_and_combine():
            try:
                # Wait for executor to finish
                if self.online_executor:
                    self.online_executor.shutdown(wait=True)
                    self.online_executor = None

                # Combine results in order
                with self.online_segment_lock:
                    results = []
                    total_latency = 0
                    total_audio_duration = 0
                    num_segments = len(self.online_segments)
                    for seg in sorted(self.online_segments, key=lambda x: x['seq']):
                        if seg['result'] and not seg['result'].startswith('[Error'):
                            results.append(seg['result'])
                        total_latency += seg.get('latency', 0)
                        total_audio_duration += seg.get('duration_s', 0)

                result_text = " ".join(results).strip()

                if result_text:
                    # Word counting, stats persistence, and archive saving all
                    # happen once on the Tk main thread via result_queue.
                    self.total_audio_duration += total_audio_duration
                    self.today_audio_duration += total_audio_duration
                    self.last_transcription_latency = total_latency
                    self.result_queue.put(f"[ONLINE] {result_text}")

                    word_count = len(result_text.split())
                    self.msg_queue.put(f"☁️ Transcribed {word_count} words from {num_segments} segments ({total_latency:.0f}ms total)")

                    # Copy to clipboard, paste, and show READY state
                    def deliver():
                        self.root.clipboard_clear()
                        self.root.clipboard_append(result_text)
                        self.root.update()
                        # Platform-aware paste (Cmd+V on Mac, Ctrl+V on Windows)
                        paste_key = get_paste_shortcut() if PLATFORM_UTILS_AVAILABLE else 'ctrl+v'
                        keyboard.send(paste_key)

                        if with_edit:
                            self.btn_transcribe_edit.configure(fg_color=COLOR_READY, text_color="white", text="Done")
                        else:
                            self.btn_transcribe.configure(fg_color=COLOR_READY, text_color="white", text="Done")
                        self.status_var.set("Copied & Pasted!")
                        self.update_mini_window_color(COLOR_READY)
                        self._ui_after(UI_RUNTIME["ready_state_delay_ms"], self._reset_online_buttons)
                    self._ui_after(0, deliver)
                else:
                    self.msg_queue.put("No transcription result")
                    self._ui_after(0, self._reset_online_buttons)

            except Exception as e:
                self.msg_queue.put(f"Online transcription failed: {e}")
                self._ui_after(0, self._reset_online_buttons)

        threading.Thread(target=wait_and_combine, daemon=True).start()

    def _reset_online_buttons(self):
        """Reset online mode button colors and text."""
        self.btn_transcribe.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED, text=self._mode_button_text("transcribe"))
        self.btn_transcribe_edit.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED, text=self._mode_button_text("transcribe_edit"))
        self.update_mini_window_color(COLOR_IDLE)
        self.status_var.set("Ready")

    def _save_openai_settings(self):
        """Save OpenAI settings to config."""
        self.config['openai_api_key'] = self.api_key_var.get().strip()
        self.config['openai_transcription_model'] = self.openai_trans_model_var.get()
        self.config['openai_edit_model'] = self.openai_edit_model_var.get()
        self.config['openai_language'] = self.openai_lang_var.get()
        self.config['openai_edit_prompt'] = self.openai_prompt_var.get().strip()

        self._persist_config()
        self._update_api_key_notice()

        self.msg_queue.put("OpenAI settings saved")
        messagebox.showinfo("Saved", "OpenAI settings saved successfully.")

    def _auto_save_openai_settings(self):
        """Silently save OpenAI settings when combo boxes change."""
        self.config['openai_transcription_model'] = self.openai_trans_model_var.get()
        self.config['openai_edit_model'] = self.openai_edit_model_var.get()
        self.config['openai_language'] = self.openai_lang_var.get()
        
        try:
            self._persist_config()
            self._update_api_key_notice()
            self.log_internal(f"Settings saved: {self.openai_trans_model_var.get()}")
        except Exception:
            pass

    def _test_openai_connection(self):
        """Test the OpenAI API connection."""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("No API Key", "Please enter an API key first.")
            return
        
        self.status_var.set("Testing OpenAI connection...")
        
        def test_connection():
            try:
                from backends.openai_backend import OpenAIBackend
                
                backend = OpenAIBackend()
                backend.configure(api_key=api_key)
                success, message = backend.test_connection()
                
                def show_result():
                    self.status_var.set("Ready")
                    if success:
                        messagebox.showinfo("Connection Test", f"✅ {message}")
                    else:
                        messagebox.showerror("Connection Test", f"❌ {message}")
                
                self._ui_after(0, show_result)
                
            except ImportError as e:
                self._ui_after(0, lambda: messagebox.showerror("Error", 
                    f"OpenAI library not installed.\n\nRun: pip install openai"))
                self._ui_after(0, lambda: self.status_var.set("Ready"))
            except Exception as e:
                self._ui_after(0, lambda: messagebox.showerror("Error", f"Test failed: {e}"))
                self._ui_after(0, lambda: self.status_var.set("Ready"))
        
        threading.Thread(target=test_connection, daemon=True).start()

    def start_hotkey_capture(self, hotkey_type):
        """Start capturing a new hotkey for batch or live mode - supports multi-key combos"""
        self._clear_capture_hooks()
        self.capturing_hotkey = hotkey_type
        self._hotkey_parts = set()  # Track currently held keys
        self._hotkey_confirmed = False
        
        # Update button appearance to show capture mode
        if hotkey_type == 'batch':
            self.batch_hotkey_btn.config(bg=COLOR_TEAL, fg=COLOR_BG, text="Press keys... (Enter to confirm)")
        else:
            self.live_hotkey_btn.config(bg=COLOR_TEAL, fg=COLOR_BG, text="Press keys... (Enter to confirm)")
        
        def update_preview():
            """Update the button text to show current key combo"""
            if not self._hotkey_parts:
                preview = "Press keys..."
            else:
                # Order: ctrl, alt, shift, win, then other keys
                ordered = []
                for mod in ['ctrl', 'alt', 'shift', 'win']:
                    if mod in self._hotkey_parts:
                        ordered.append(mod)
                for key in sorted(self._hotkey_parts):
                    if key not in ['ctrl', 'alt', 'shift', 'win']:
                        ordered.append(key)
                preview = '+'.join(ordered)
            
            preview_text = f"{preview} (Enter=OK, Esc=Cancel)"
            if hotkey_type == 'batch':
                self.batch_hotkey_btn.config(text=preview_text)
            else:
                self.live_hotkey_btn.config(text=preview_text)
        
        def on_key_down(event):
            if self.capturing_hotkey is None or self._hotkey_confirmed:
                return
            
            key_name = event.name.lower()
            
            # Enter confirms the hotkey
            if key_name in ('enter', 'return'):
                if self._hotkey_parts:
                    self._hotkey_confirmed = True
                    # Build final hotkey string
                    ordered = []
                    for mod in ['ctrl', 'alt', 'shift', 'win']:
                        if mod in self._hotkey_parts:
                            ordered.append(mod)
                    for key in sorted(self._hotkey_parts):
                        if key not in ['ctrl', 'alt', 'shift', 'win']:
                            ordered.append(key)
                    new_hotkey = '+'.join(ordered)
                    
                    current_type = self.capturing_hotkey
                    self.capturing_hotkey = None
                    self._clear_capture_hooks()
                    self._ui_after(HOTKEY_RUNTIME["capture_finish_delay_ms"], lambda: self.finish_hotkey_capture(current_type, new_hotkey))
                return
            
            # Escape cancels
            if key_name == 'escape':
                self._clear_capture_hooks()
                self._ui_after(HOTKEY_RUNTIME["capture_finish_delay_ms"], self.cancel_hotkey_capture)
                return
            
            # Normalize modifier names
            if key_name in ('left ctrl', 'right ctrl', 'control'):
                key_name = 'ctrl'
            elif key_name in ('left alt', 'right alt'):
                key_name = 'alt'
            elif key_name in ('left shift', 'right shift'):
                key_name = 'shift'
            elif key_name in ('left windows', 'right windows', 'windows'):
                key_name = 'win'
            
            self._hotkey_parts.add(key_name)
            self.root.after(HOTKEY_RUNTIME["capture_preview_delay_ms"], update_preview)
        
        def on_key_up(event):
            if self.capturing_hotkey is None or self._hotkey_confirmed:
                return
            
            key_name = event.name.lower()
            
            # Normalize modifier names
            if key_name in ('left ctrl', 'right ctrl', 'control'):
                key_name = 'ctrl'
            elif key_name in ('left alt', 'right alt'):
                key_name = 'alt'
            elif key_name in ('left shift', 'right shift'):
                key_name = 'shift'
            elif key_name in ('left windows', 'right windows', 'windows'):
                key_name = 'win'
            
            # Don't remove keys - we want to capture all pressed keys
            # Only update if the key was in our set
            # self._hotkey_parts.discard(key_name)
            # self.root.after(1, update_preview)
        
        # Hook both key down and key up
        on_key_down_hook = keyboard.hook(on_key_down, suppress=False)
        on_key_up_hook = keyboard.on_release(on_key_up, suppress=False)
        
        # Store hook reference for cleanup
        self._capture_hooks = (on_key_down_hook, on_key_up_hook)

    def finish_hotkey_capture(self, hotkey_type, new_hotkey):
        """Finish capturing hotkey and save"""
        try:
            self._clear_capture_hooks()
            # Reset button appearance
            if hotkey_type == 'batch':
                self.batch_hotkey_btn.config(bg=COLOR_ACCENT_BG, fg=COLOR_TEAL)
                self.batch_hotkey_var.set(new_hotkey)
                self.config['hotkey_batch'] = new_hotkey
                
                # Update button labels in header (both local and online)
                self.btn_batch.configure(text=self._mode_button_text("batch"))
                self.btn_transcribe.configure(text=self._mode_button_text("transcribe"))
            else:
                self.live_hotkey_btn.config(bg=COLOR_ACCENT_BG, fg=COLOR_TEAL)
                self.live_hotkey_var.set(new_hotkey)
                self.config['hotkey_live'] = new_hotkey
                
                # Update button labels in header (both local and online)
                self.btn_live.configure(text=self._mode_button_text("live"))
                self.btn_transcribe_edit.configure(text=self._mode_button_text("transcribe_edit"))
            
            # Save config
            self._persist_config()

            # Keep tracked hotkey handlers in sync after hotkey changes.
            self._unregister_hotkeys()
            self._register_hotkeys()
            
            self.log_internal(f"Hotkey updated: {hotkey_type.upper()} → {new_hotkey}")
        except Exception as ex:
            self.log_internal(f"Error setting hotkey: {ex}")

    def cancel_hotkey_capture(self):
        """Cancel hotkey capture and reset UI"""
        self._clear_capture_hooks()
        hotkey_type = self.capturing_hotkey
        self.capturing_hotkey = None
        
        if hotkey_type == 'batch':
            self.batch_hotkey_btn.config(bg=COLOR_ACCENT_BG, fg=COLOR_TEAL)
            self.batch_hotkey_var.set(self.config['hotkey_batch'])
        elif hotkey_type == 'live':
            self.live_hotkey_btn.config(bg=COLOR_ACCENT_BG, fg=COLOR_TEAL)
            self.live_hotkey_var.set(self.config['hotkey_live'])

    def on_model_select(self, event=None):
        display_val = self.model_var.get()
        real_key = self.model_map_display.get(display_val, display_val)
        
        # Fallback
        if real_key not in MODEL_MAP:
             for k in MODEL_MAP.keys():
                 if k in display_val:
                     real_key = k
                     break
        
        # If already same as config, ignore (e.g. startup)
        if real_key == self.config['model_key']: return

        # Check download status - only prompt if not downloaded
        if not self.is_model_downloaded(real_key):
             confirm = messagebox.askyesno(
                 "Download Required", 
                 f"The model '{real_key}' is not downloaded yet.\n\nDo you want to download it now?\n(This may take a few minutes depending on your internet speed)\n\nClick Yes to Download & Switch.\nClick No to Cancel."
             )
             if not confirm:
                 # Revert to previous
                 self.refresh_model_list()
                 return
        
        # Auto-save for ALL model changes (downloaded or just confirmed download)
        self.save_config()

    def save_config(self):
        # STRIP DISPLAY PREFIX (see refresh_model_list)
        display_val = self.model_var.get()
        real_key = self.model_map_display.get(display_val, display_val)

        # Fallback if map fail
        if real_key not in MODEL_MAP:
             for k in MODEL_MAP.keys():
                 if k in display_val:
                     real_key = k
                     break

        # Capture old key to check for changes
        old_model_key = self.config.get('model_key', '')

        self.config['model_key'] = real_key
        self.config['live_pause'] = float(self.live_pause_var.get())
        self._live_pause = self.config['live_pause']  # mirror for the audio thread
        self.config['always_on_top'] = self.top_var.get()

        device_name = self.device_var.get()
        try:
            devices = list(sd.query_devices())
        except Exception as e:
            devices = []
            log.error("Could not enumerate audio devices while saving: %s", e)
        for i, d in enumerate(devices):
            if d.get('name') == device_name and d.get('max_input_channels', 0) > 0:
                self.config['input_device'] = i
                break

        # Shared settings to the synced file, machine-local/sensitive keys locally
        self._persist_config()

        self.root.attributes('-topmost', self.config['always_on_top'])

        if real_key != old_model_key:
             # Simply reload model in a thread
             threading.Thread(target=self.load_model, daemon=True).start()
             if not self.is_model_downloaded(real_key):
                  messagebox.showinfo("Downloading", f"Downloading {real_key}...\nPlease wait and check the progress bar.")
             else:
                  messagebox.showinfo("Saved", f"Switched to {real_key}.\nLoading in background...")
        else:
             messagebox.showinfo("Saved", "Settings saved.")

        self._update_device_model_labels()
        self._update_config_summary()
        self.restart_audio_stream()

    def refresh_model_list(self):
        # Re-scan download status
        current_selection = self.model_var.get()
        pure_name = self.model_map_display.get(current_selection, current_selection)
        
        self.model_map_display = {}
        display_values = []
        
        new_selection_display = current_selection

        for friendly in MODEL_MAP.keys():
            is_down = self.is_model_downloaded(friendly)
            prefix = "" if is_down else "[get] "
            display_name = f"{prefix}{friendly}"
            self.model_map_display[display_name] = friendly
            display_values.append(display_name)
            
            if friendly == pure_name:
                new_selection_display = display_name
                
        self.combo['values'] = display_values
        self.model_var.set(new_selection_display)
    def _report_recording_start_failure(self, kind):
        """The microphone could not be opened. Beep and say why, everywhere."""
        reason = getattr(self, "_last_audio_error", None) or "no microphone available"
        try:
            self.play_feedback_sound(start=False)
        except Exception:
            pass
        self.log_internal(f"Could not start {kind} recording - microphone unavailable: {reason}")
        log.error("Could not start %s recording: %s", kind, reason)
        try:
            self.status_var.set(f"No microphone - {reason}")
        except Exception:
            pass

    def toggle_live_mode(self):
        """F9 Toggled"""
        if self.stopping: return

        if self.mode == "batch": 
            # Stop batch mode instead of blocking (any key = stop)
            self._stop_any_recording()
            return

        if self.mode == "live":
            # STOP
            self.play_feedback_sound(start=False)
            self.stopping = True
            self.btn_live.configure(text="Stopping...", fg_color=COLOR_STOPPING_LIVE, text_color="white") 
            self.status_var.set("Catching final words...")
            threading.Thread(target=self._delayed_stop_live, daemon=True).start()
        else:
            # START
            if not self.start_audio_stream():
                # A hotkey-only user never sees the window: the stop beep plus
                # the status line are the only feedback that nothing started.
                self._report_recording_start_failure("live")
                return
            self.play_feedback_sound(start=True)
            self.mode = "live"
            with self.audio_queue.mutex: self.audio_queue.queue.clear()
            self.live_buffer = [] 
            self.live_backup_buffer = [] 
            # Highlight LIVE button, dim BATCH button
            self.btn_live.configure(fg_color=COLOR_LIVE, text_color="white", text=self._mode_button_text("live"))
            self.btn_batch.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED, text=self._mode_button_text("batch"))
            self.status_var.set(f"LIVE TYPING ({self.config['hotkey_live']})")
            self.update_mini_window_color(COLOR_LIVE)

    def _delayed_stop_live(self):
        # Runs on a worker thread: do the heavy lifting here, but marshal every
        # widget/StringVar touch to the Tk main thread via root.after().
        time.sleep(AUDIO_RUNTIME["stop_delay_seconds"])

        if self._shutting_down:
            self.stop_audio_stream()
            return

        # --- FIX: Atomically capture and clear the remaining buffer ---
        # This prevents the race condition where processing_loop could
        # process the same chunk we're about to process here.
        with self.live_buffer_lock:
            remaining_buffer = self.live_buffer.copy()
            self.live_buffer = []

        if remaining_buffer and self.model:
            self.log_internal("Flushing final buffer...")
            self.process_live_chunk(remaining_buffer)
        # -----------------------------------------

        # Close the stream BEFORE releasing the guard: a hotkey press landing
        # between the two would otherwise adopt this still-open stream and then
        # have it closed out from under it.
        self.stop_audio_stream()
        self.mode = None
        self.stopping = False

        if self._shutting_down:
            return

        def ui_idle():
            self._reset_buttons_to_idle()
            self.status_var.set("Ready. (Processing Live Backup...)")
        self._ui_after(0, ui_idle)
        self.finalize_live_backup()

    def _reset_buttons_to_idle(self):
        """Reset both mode buttons to idle state"""
        self.btn_batch.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED, text=self._mode_button_text("batch"))
        self.btn_live.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED, text=self._mode_button_text("live"))
        self.update_mini_window_color(COLOR_IDLE)

    def _stop_any_recording(self):
        """Stop any active recording mode (unified stop behavior)."""
        if self.stopping:
            return
        if self.mode == "live":
            self.toggle_live_mode()  # Will trigger stop
        elif self.mode == "batch":
            self.toggle_batch_mode()  # Will trigger stop
        elif self.mode == 'online_transcribe':
            self._stop_online_recording(with_edit=False)
        elif self.mode == 'online_transcribe_edit':
            self._stop_online_recording(with_edit=True)

    def toggle_batch_mode(self):
        """F8 Toggled"""
        if self.stopping: return

        if self.mode == "live":
            # Stop live mode instead of blocking (any key = stop)
            self._stop_any_recording()
            return

        if self.mode == "batch":
            # STOP
            self.play_feedback_sound(start=False)
            self.stopping = True
            self.btn_batch.configure(text="Stopping...", fg_color=COLOR_STOPPING_BATCH, text_color="white") 
            self.status_var.set("Catching final words...")
            threading.Thread(target=self._delayed_stop_batch, daemon=True).start()
        else:
            # START
            if not self.start_audio_stream():
                self._report_recording_start_failure("batch")
                return
            self.play_feedback_sound(start=True)
            self.mode = "batch"
            self.batch_total_samples = 0
            with self.audio_queue.mutex: self.audio_queue.queue.clear()
            
            # Initialize streaming batch state
            self.batch_segments = []
            self.batch_segment_seq = 0
            self.batch_pending_audio = []
            self.batch_silence_count = 0
            
            # Create thread pool - limit workers to avoid GPU memory issues
            max_workers = (
                AUDIO_RUNTIME["batch_executor_workers_cpu"]
                if self.config.get('device') == 'cpu'
                else AUDIO_RUNTIME["batch_executor_workers_gpu"]
            )
            self.batch_executor = ThreadPoolExecutor(max_workers=max_workers)
            
            # Highlight BATCH button, dim LIVE button
            self.btn_batch.configure(fg_color=COLOR_BATCH, text_color="white", text=self._mode_button_text("batch"))
            self.btn_live.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED, text=self._mode_button_text("live"))
            self.status_var.set(f"BATCH RECORDING ({self.config['hotkey_batch']})...")
            self.update_mini_window_color(COLOR_BATCH)

    def _delayed_stop_batch(self):
        # Runs on a worker thread: marshal UI updates to the Tk main thread.
        time.sleep(AUDIO_RUNTIME["stop_delay_seconds"])
        # Close the stream BEFORE releasing the guard (see _delayed_stop_live).
        self.stop_audio_stream()
        self.mode = None
        self.stopping = False

        if self._shutting_down:
            return

        def ui_transcribing():
            # Show TRANSCRIBING state (amber)
            self.btn_batch.configure(fg_color=COLOR_TRANSCRIBING, text_color="white", text="Transcribing...")
            self.btn_live.configure(
                fg_color=COLOR_BG,
                text_color=COLOR_TEXT_MUTED,
                text=self._mode_button_text("live"),
            )
            self.status_var.set("Processing Batch... Please Wait.")
            self.update_mini_window_color(COLOR_TRANSCRIBING)
        self._ui_after(0, ui_transcribing)
        self.finalize_batch()

    def finalize_batch(self):
        # Runs on a worker thread (waits on the transcription pool); all UI
        # work is marshalled to the Tk main thread.
        def ui_reset():
            self._reset_buttons_to_idle()
            self.status_var.set("Ready.")

        if not self.batch_segments and not self.batch_pending_audio:
            self.log_internal("No audio recorded.")
            self._ui_after(0, ui_reset)
            return

        try:
            # Track duration via the running sample counter (avoids keeping a
            # second full copy of the recording in memory)
            duration = self.batch_total_samples / AUDIO_RUNTIME["sample_rate"]
            self.total_audio_duration += duration
            self.today_audio_duration += duration

            # 1. Queue any remaining pending audio
            if self.batch_pending_audio:
                self._queue_batch_segment()

            # 2. Wait for all segments to complete transcription
            if self.batch_executor:
                self.log_internal("Waiting for transcription threads...")
                self.batch_executor.shutdown(wait=True)
                self.batch_executor = None

            # 3. Stitch results in correct order
            with self.batch_segment_lock:
                ordered = sorted(self.batch_segments, key=lambda x: x['seq'])
                results = [s['result'] for s in ordered if s['result'] and not s['result'].startswith('[Error')]
                total_latency = sum(s.get('latency_ms', 0) for s in ordered)
                full_text = " ".join(results).strip()

            # 4. Output the combined text
            if full_text:
                self.log_internal(f"✅ Batch complete: {len(ordered)} segments stitched")
                self.last_transcription_latency = total_latency
                self.result_queue.put(f"[BATCH] {full_text}")

                def deliver():
                    self.root.clipboard_clear()
                    self.root.clipboard_append(full_text)
                    self.root.update()

                    # Show READY state (green) - text is in clipboard
                    self.btn_batch.configure(fg_color=COLOR_READY, text_color="white", text="Ready to paste")
                    self.status_var.set("Ready! Text copied to clipboard. Pasting...")
                    self.update_mini_window_color(COLOR_READY)

                    # Auto-paste
                    # Platform-aware paste (Cmd+V on Mac, Ctrl+V on Windows)
                    paste_key = get_paste_shortcut() if PLATFORM_UTILS_AVAILABLE else 'ctrl+v'
                    keyboard.send(paste_key)

                    # Return to idle after a delay
                    self._ui_after(UI_RUNTIME["ready_state_delay_ms"], ui_reset)
                self._ui_after(0, deliver)
            else:
                self.log_internal("Batch result was empty.")
                self._ui_after(0, ui_reset)

        except Exception as e:
            self.log_internal(f"Batch Error: {e}")
            self._ui_after(0, ui_reset)

    def _idle_unload_loop(self):
        """Background thread: after idle_unload_minutes of no dictation
        activity, unload the model to free GPU memory. The next transcription
        attempt transparently reloads it via _reload_model_if_needed_locked().
        """
        while self.running:
            time.sleep(IDLE_UNLOAD_CHECK_INTERVAL_SECONDS)
            try:
                idle_minutes = float(self.config.get('idle_unload_minutes', DEFAULT_CONFIG['idle_unload_minutes']))
            except (TypeError, ValueError):
                idle_minutes = DEFAULT_CONFIG['idle_unload_minutes']

            if idle_minutes <= 0:
                continue  # 0 = disabled
            if self.mode is not None:
                continue  # never unload mid-recording
            if self._model_idle_unloaded or not self.backend or not self.backend.is_model_loaded:
                continue
            if self.current_device == 'cloud' or self.current_backend_name == 'openai':
                continue  # nothing to free for the cloud backend
            if (time.time() - self._last_activity_time) < idle_minutes * 60:
                continue

            # Non-blocking: if a transcription is in progress, skip this cycle
            # and try again next tick rather than blocking the timer thread.
            if not self._model_lock.acquire(blocking=False):
                continue
            try:
                if self.backend and self.backend.is_model_loaded:
                    self.log_internal(f"Idle {idle_minutes:.0f}min - unloading model to free GPU memory.")
                    self.backend.unload_model()
                    self._model_idle_unloaded = True
                    self._ui_after(0, lambda: self.status_var.set("Idle - model unloaded (GPU freed)"))
            finally:
                self._model_lock.release()

    def _reload_model_if_needed_locked(self):
        """Reload the model if the idle timer unloaded it, blocking until
        ready. Call only while already holding self._model_lock, so callers
        can keep holding the lock through their transcribe() call - this
        guarantees the idle-unload timer can never fire mid-transcription.
        No-op if the model is already loaded (or hasn't loaded for the first
        time yet - that startup race is unchanged from before this feature).
        """
        if self._model_idle_unloaded:
            self.log_internal("Model was idle-unloaded - reloading before transcribing...")
            self._ui_after(0, lambda: self.status_var.set("Reloading model..."))
            self.load_model()
            if (self.backend and self.backend.is_model_loaded) or (self.backend is None and self.model is not None):
                self._model_idle_unloaded = False
            else:
                self.log_internal("⚠️ Model reload failed - will retry on next transcription.")
                self._ui_after(0, lambda: self.status_var.set("Load Error - retrying on next dictation"))
        self._last_activity_time = time.time()

    def finalize_live_backup(self):
        """Re-transcribe the full live session as a clipboard backup.

        Clipboard-only by design: the live chunks were already typed, counted
        into the stats, and saved to the archive. Pushing the backup through
        result_queue as well would double-count every word and duplicate the
        transcript archive.
        """
        if not self.live_backup_buffer:
            self._ui_after(0, lambda: self.status_var.set("Ready."))
            return
        temp_path = self._temp_audio_path("live_backup")
        try:
            # We don't add duration here because live mode already adds it chunk by chunk
            audio_np = np.concatenate(self.live_backup_buffer, axis=0).flatten()
            wav.write(temp_path, AUDIO_RUNTIME["sample_rate"], audio_np)
            with self._model_lock:
                self._reload_model_if_needed_locked()
                if not self.model: return
                # Use backend abstraction if available
                if self.backend and hasattr(self.backend, 'transcribe'):
                    segments = self.backend.transcribe(temp_path)
                    text = " ".join([s.text for s in segments]).strip()
                else:
                    segments, _ = self.model.transcribe(
                        temp_path, beam_size=5, vad_filter=True, condition_on_previous_text=True
                    )
                    text = " ".join([s.text for s in segments]).strip()

            def deliver(backup_text=text):
                if backup_text:
                    self.root.clipboard_clear()
                    self.root.clipboard_append(backup_text)
                    self.status_var.set("Ready. (Backup in Clipboard)")
                else:
                    self.status_var.set("Ready.")
            self._ui_after(0, deliver)
        except Exception as e:
            self.log_internal(f"Backup Error: {e}")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    # --- MINIMIZATION LOGIC ---
    def on_minimize(self, event):
        if self.root.state() == 'iconic' and self.mini_window is None:
            self.create_mini_window()

    def on_restore(self, event):
        if self.root.state() == 'normal' and self.mini_window is not None:
            self.destroy_mini_window()
        # Refresh hotkeys when restored from minimized state
        self._on_focus_in()

    def create_mini_window(self):
        w = UI_RUNTIME["mini_window_width"]
        h = UI_RUNTIME["mini_window_height"]

        self.mini_window = tk.Toplevel(self.root)
        self.mini_window.overrideredirect(True)
        self.mini_window.attributes('-topmost', True)
        self.mini_window.geometry(UI_RUNTIME["mini_window_default_geometry"])

        state_col = COLOR_IDLE
        if self.mode == "live":
            state_col = COLOR_LIVE
        elif self.mode == "batch":
            state_col = COLOR_BATCH
        elif self._is_online_mode():
            state_col = COLOR_ONLINE

        # Body stays dark; only the left bar, border and dot carry the state
        # colour, so the pill stays legible over any wallpaper.
        self.mini_window.configure(bg=COLOR_ACCENT_BG)
        self.mini_canvas = tk.Canvas(self.mini_window, bg=COLOR_ACCENT_BG,
                                     highlightthickness=0, width=w, height=h)
        self.mini_canvas.pack(fill="both", expand=True)

        self.mini_canvas.create_rectangle(0, 0, w - 1, h - 1, outline=state_col,
                                          width=1, tags="mini_border")
        self.mini_canvas.create_rectangle(0, 0, 4, h, fill=state_col,
                                          outline=state_col, tags="left_bar")
        self.mini_canvas.create_oval(11, h // 2 - 4, 19, h // 2 + 4, fill=state_col,
                                     outline=state_col, tags="state_dot")

        self._mini_vu_x0 = 27
        self._mini_vu_x1 = w - 68
        self.mini_canvas.create_text(self._mini_vu_x0, 17, text="Idle", fill=COLOR_FG,
                                     font=("Arial", 9, "bold"), anchor="w",
                                     tags="status_text")
        self.mini_canvas.create_rectangle(self._mini_vu_x0, 31, self._mini_vu_x1, 35,
                                          fill=COLOR_BG, outline=COLOR_BG,
                                          tags="vu_track")
        # 4px VU bar under the status text (fed by mini_vu_multiplier)
        self.mini_vu = self.mini_canvas.create_rectangle(
            self._mini_vu_x0, 31, self._mini_vu_x0, 35,
            fill=state_col, outline=state_col, tags="vu_bar")

        self.mini_canvas.create_text(w - 11, 17, text="0 w", fill=COLOR_TEXT_MUTED,
                                     font=("Consolas", 7), anchor="e",
                                     tags="stats_text")
        self.mini_canvas.create_text(w - 11, 31, text="0.0 m", fill=COLOR_TEXT_FAINT,
                                     font=("Consolas", 7), anchor="e",
                                     tags="stats_text2")

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.mini_window.geometry(
            f"{w}x{h}+{sw - w - UI_RUNTIME['mini_window_right_margin_px']}+{sh - h - UI_RUNTIME['mini_window_bottom_margin_px']}"
        )

        self.update_mini_window_color(state_col)

        # --- DRAG FUNCTIONALITY ---
        self._mini_drag_data = {"x": 0, "y": 0}

        def start_drag(event):
            self._mini_drag_data["x"] = event.x
            self._mini_drag_data["y"] = event.y

        def do_drag(event):
            x = self.mini_window.winfo_x() + (event.x - self._mini_drag_data["x"])
            y = self.mini_window.winfo_y() + (event.y - self._mini_drag_data["y"])
            self.mini_window.geometry(f"+{x}+{y}")

        def restore(_event=None):
            self.root.deiconify()

        self.mini_canvas.bind("<Button-1>", start_drag)
        self.mini_canvas.bind("<B1-Motion>", do_drag)
        self.mini_canvas.bind("<Double-Button-1>", restore)
        self.mini_window.bind("<Button-1>", start_drag)
        self.mini_window.bind("<B1-Motion>", do_drag)
        self.mini_window.bind("<Double-Button-1>", restore)

    def destroy_mini_window(self):
        if self.mini_window:
            self.mini_window.destroy()
            self.mini_window = None
            self.mini_vu = None
            self.mini_canvas = None

    def update_mini_window_color(self, color):
        """Only the left bar, border, dot and VU bar carry the state colour."""
        self._mini_state_color = color
        if self.mini_window and self.mini_canvas:
            self.mini_canvas.itemconfigure("mini_border", outline=color)
            self.mini_canvas.itemconfigure("left_bar", fill=color, outline=color)
            self.mini_canvas.itemconfigure("state_dot", fill=color, outline=color)
            self.mini_canvas.itemconfigure("vu_bar", fill=color, outline=color)

    def processing_loop(self):
        silent_chunks = 0
        current_live_duration = 0
        
        while self.running:
            try:
                data = self.audio_queue.get(timeout=AUDIO_RUNTIME["queue_timeout_seconds"])
                
                # SMOOTHING
                raw_amp = np.sqrt(np.mean(data**2))
                if raw_amp > self.current_display_volume:
                    self.current_display_volume = raw_amp 
                else:
                    self.current_display_volume *= 0.85 
                
                if self.mode == "batch":
                    self.batch_total_samples += len(data)
                    self.batch_pending_audio.append(data)
                    
                    # VAD: detect pause for streaming segmentation
                    if raw_amp < self.config['silence_threshold']:
                        self.batch_silence_count += 1
                    else:
                        self.batch_silence_count = 0
                    
                    # Calculate pending audio duration (~100ms per chunk at 16kHz, 1600 samples)
                    pending_duration = len(self.batch_pending_audio) * AUDIO_RUNTIME["chunk_duration_seconds"]
                    silence_duration = self.batch_silence_count * AUDIO_RUNTIME["chunk_duration_seconds"]
                    
                    # Progressive pause threshold: encourages ~20s segments
                    # - Under 10s: require 2.0s pause (don't cut too early)
                    # - 10s-20s: linearly decrease from 2.0s to 1.5s (progressively easier to cut)
                    # - Over 20s: use 1.5s pause (eager to segment)
                    # This creates more evenly spaced ~20s segments
                    if pending_duration < AUDIO_RUNTIME["batch_min_segment_seconds"]:
                        pause_threshold_batch = AUDIO_RUNTIME["pause_threshold_short_seconds"]
                    elif pending_duration < AUDIO_RUNTIME["batch_long_segment_seconds"]:
                        # Linear interpolation: 2.0 at 10s → 1.5 at 20s
                        progress = (
                            pending_duration - AUDIO_RUNTIME["batch_min_segment_seconds"]
                        ) / (
                            AUDIO_RUNTIME["batch_long_segment_seconds"] - AUDIO_RUNTIME["batch_min_segment_seconds"]
                        )  # 0 to 1
                        pause_threshold_batch = AUDIO_RUNTIME["pause_threshold_short_seconds"] - (
                            (AUDIO_RUNTIME["pause_threshold_short_seconds"] - AUDIO_RUNTIME["pause_threshold_long_seconds"]) * progress
                        )  # 2.0 → 1.5
                    else:
                        pause_threshold_batch = AUDIO_RUNTIME["pause_threshold_long_seconds"]
                    
                    # Segment when: (pause detected AND 10s+ audio) OR 60s max reached
                    is_natural_break = (
                        silence_duration > pause_threshold_batch
                        and pending_duration > AUDIO_RUNTIME["batch_min_segment_seconds"]
                    )
                    is_max_duration = pending_duration > AUDIO_RUNTIME["batch_max_segment_seconds"]
                    
                    # Queue segment when pause detected or max duration reached
                    if (is_natural_break or is_max_duration) and not self.stopping:
                        self._queue_batch_segment()
                elif self.mode in ("online_transcribe", "online_transcribe_edit"):
                    # Online mode: streaming transcription with pause detection
                    self.online_pending_audio.append(data)
                    online_pending_duration = len(self.online_pending_audio) * AUDIO_RUNTIME["chunk_duration_seconds"]  # ~100ms per chunk
                    
                    # Check for silence
                    if raw_amp < self.config['silence_threshold']:
                        self.online_silence_count += 1
                    else:
                        self.online_silence_count = 0
                    
                    online_silence_duration = self.online_silence_count * AUDIO_RUNTIME["chunk_duration_seconds"]
                    
                    # Same progressive threshold as batch: ~20s target segments
                    if online_pending_duration < AUDIO_RUNTIME["batch_min_segment_seconds"]:
                        online_pause_threshold = AUDIO_RUNTIME["pause_threshold_short_seconds"]
                    elif online_pending_duration < AUDIO_RUNTIME["batch_long_segment_seconds"]:
                        progress = (
                            online_pending_duration - AUDIO_RUNTIME["batch_min_segment_seconds"]
                        ) / (
                            AUDIO_RUNTIME["batch_long_segment_seconds"] - AUDIO_RUNTIME["batch_min_segment_seconds"]
                        )
                        online_pause_threshold = AUDIO_RUNTIME["pause_threshold_short_seconds"] - (
                            (AUDIO_RUNTIME["pause_threshold_short_seconds"] - AUDIO_RUNTIME["pause_threshold_long_seconds"]) * progress
                        )
                    else:
                        online_pause_threshold = AUDIO_RUNTIME["pause_threshold_long_seconds"]
                    
                    is_natural_break = (
                        online_silence_duration > online_pause_threshold
                        and online_pending_duration > AUDIO_RUNTIME["batch_min_segment_seconds"]
                    )
                    is_max_duration = online_pending_duration > AUDIO_RUNTIME["batch_max_segment_seconds"]
                    
                    # Queue segment when pause or max duration
                    if (is_natural_break or is_max_duration) and not self.stopping:
                        self._queue_online_segment()
                elif self.mode == "live":
                    with self.live_buffer_lock:
                        self.live_buffer.append(data)
                    self.live_backup_buffer.append(data) 
                    chunk_dur = len(data) / AUDIO_RUNTIME["sample_rate"]
                    current_live_duration += chunk_dur
            
                if raw_amp < self.config['silence_threshold']:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                
                # Audio thread: read the mirrored attribute, never the Tk var.
                pause_threshold = self._live_pause
                is_silence = (silent_chunks * AUDIO_RUNTIME["chunk_duration_seconds"]) > pause_threshold
                is_too_long = current_live_duration > AUDIO_RUNTIME["live_max_chunk_seconds"]

                if (is_silence or is_too_long) and current_live_duration > AUDIO_RUNTIME["live_min_chunk_seconds"] and not self.stopping:
                    # Atomically capture and clear buffer to prevent race conditions
                    with self.live_buffer_lock:
                        buffer_to_process = self.live_buffer.copy()
                        self.live_buffer = []
                    if self.model and buffer_to_process:
                        self.process_live_chunk(buffer_to_process)
                    current_live_duration = 0
                    silent_chunks = 0
            
            except queue.Empty:
                self.current_display_volume *= 0.85
            except Exception as e:
                # Never die silently - this loop is the audio engine
                self.log_internal(f"Audio processing error: {e}")

    def process_live_chunk(self, audio_data):
        if not audio_data: return
        audio_np = np.concatenate(audio_data, axis=0).flatten()
        
        # --- TRACK DURATION ---
        duration = len(audio_np) / AUDIO_RUNTIME["sample_rate"]
        self.total_audio_duration += duration
        self.today_audio_duration += duration
        # ----------------------
        
        temp_path = self._temp_audio_path("live")
        wav.write(temp_path, AUDIO_RUNTIME["sample_rate"], audio_np)
        try:
            with self._model_lock:
                self._reload_model_if_needed_locked()

                # Track transcription latency
                start_time = time.perf_counter()

                # Use backend abstraction if available
                # beam_size=1 for live chunks: latency matters more than the small
                # accuracy gain of beam 5 (the full-session backup uses beam 5)
                if self.backend and hasattr(self.backend, 'transcribe'):
                    segments = self.backend.transcribe(temp_path, beam_size=1)
                    text = "".join([s.text for s in segments]).strip()
                else:
                    segments, _ = self.model.transcribe(temp_path, beam_size=1, vad_filter=True)
                    text = "".join([s.text for s in segments]).strip()

            # Calculate and record latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.session_latencies.append(latency_ms)
            self.total_latencies.append(latency_ms)

            hallucinations = ["thank you.", "you", ".", "Thank you."]
            if text.lower().strip() in [h.lower() for h in hallucinations] or not text: return

            # Log with latency for performance tracking
            self.log_internal(f"🎤 LIVE ⚡{latency_ms:.0f}ms")
            self.last_transcription_latency = latency_ms  # Store for display
            self.result_queue.put(f"[LIVE] {text}")
            # Type the text directly (uses platform-aware keyboard module)
            keyboard.write(text + " ")
        except Exception as e:
            self.log_internal(f"Transcription error: {e}")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _queue_batch_segment(self):
        """Queue current pending audio for background transcription"""
        with self.batch_segment_lock:
            if not self.batch_pending_audio:
                return
            
            seq = self.batch_segment_seq
            self.batch_segment_seq += 1
            audio_data = self.batch_pending_audio.copy()
            
            segment = {
                'seq': seq,
                'audio': audio_data,
                'result': None,
                'status': 'pending'
            }
            self.batch_segments.append(segment)
            self.batch_pending_audio = []
            self.batch_silence_count = 0
        
        # Submit to thread pool (created on batch start)
        if self.batch_executor:
            self.batch_executor.submit(self._transcribe_segment, seq)
        
        self._update_batch_progress()
        self.log_internal(f"📦 Queued segment {seq + 1} for transcription")

    def _transcribe_segment(self, seq):
        """Transcribe a single segment (runs in thread pool)"""
        segment = None
        with self.batch_segment_lock:
            for s in self.batch_segments:
                if s['seq'] == seq:
                    segment = s
                    s['status'] = 'transcribing'
                    break
        
        if not segment or not self.model:
            return

        temp_file = self._temp_audio_path(f"batch_seg_{seq}")
        try:
            audio_np = np.concatenate(segment['audio'], axis=0).flatten()
            wav.write(temp_file, AUDIO_RUNTIME["sample_rate"], audio_np)

            with self._model_lock:
                self._reload_model_if_needed_locked()

                # Track transcription latency
                start_time = time.perf_counter()

                # Use backend abstraction if available
                if self.backend and hasattr(self.backend, 'transcribe'):
                    segments = self.backend.transcribe(temp_file)
                    text = " ".join([s.text for s in segments]).strip()
                else:
                    segments, _ = self.model.transcribe(
                        temp_file, beam_size=5, vad_filter=True
                    )
                    text = " ".join([s.text for s in segments]).strip()

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            with self.batch_segment_lock:
                segment['result'] = text
                segment['status'] = 'done'
                segment['latency_ms'] = latency_ms
                segment['audio'] = None  # Free the raw audio - no longer needed

            # Log with latency info for performance comparison
            self.log_internal(f"✅ Segment {seq + 1} ⚡{latency_ms:.0f}ms")

        except Exception as e:
            with self.batch_segment_lock:
                segment['status'] = 'error'
                segment['result'] = f"[Error: {e}]"
            self.log_internal(f"❌ Segment {seq + 1} error: {e}")
        finally:
            try:
                os.remove(temp_file)
            except OSError:
                pass

        self._update_batch_progress()

    def _update_batch_progress(self):
        """Update UI with batch transcription progress (called from pool threads)"""
        with self.batch_segment_lock:
            total = len(self.batch_segments)
            done = sum(1 for s in self.batch_segments if s['status'] == 'done')

        if total > 0:
            progress_text = f"BATCH ({done}/{total} chunks)"
            self._ui_after(0, lambda t=progress_text: self.status_var.set(t))

    def _queue_online_segment(self):
        """Queue current pending audio for background OpenAI transcription"""
        with self.online_segment_lock:
            if not self.online_pending_audio:
                return
            
            seq = self.online_segment_seq
            self.online_segment_seq += 1
            audio_data = self.online_pending_audio.copy()
            
            segment = {
                'seq': seq,
                'audio': audio_data,
                'result': None,
                'status': 'pending',
                'latency': 0,
                # Duration is recorded up front because 'audio' is freed
                # after transcription to release memory
                'duration_s': len(audio_data) * AUDIO_RUNTIME["chunk_duration_seconds"]
            }
            self.online_segments.append(segment)
            self.online_pending_audio = []
            self.online_silence_count = 0
        
        # Submit to thread pool (created on online start)
        if self.online_executor:
            self.online_executor.submit(self._transcribe_online_segment, seq)
        
        self._update_online_progress()
        self.log_internal(f"📦 Queued online segment {seq + 1}")

    def _get_online_backend(self):
        """Return a configured OpenAI backend for online segments.

        Reuses the backend load_model() already configured when the app runs
        on the cloud backend, so all segments of a session share one OpenAI
        client (and TLS connection pool) instead of handshaking per segment.
        """
        if self.backend is not None and self.current_backend_name == 'openai':
            return self.backend

        from backends.openai_backend import OpenAIBackend

        backend = OpenAIBackend()
        backend.configure(
            api_key=self.config.get('openai_api_key', ''),
            transcription_model=self.config.get('openai_transcription_model', DEFAULT_CONFIG['openai_transcription_model']),
            edit_model=self.config.get('openai_edit_model', DEFAULT_CONFIG['openai_edit_model']),
            edit_prompt=self.config.get('openai_edit_prompt', '') or None,
            language=self.config.get('openai_language', DEFAULT_CONFIG['openai_language'])
        )
        return backend

    def _transcribe_online_segment(self, seq):
        """Transcribe a single segment via OpenAI API (runs in thread pool)"""
        segment = None
        with self.online_segment_lock:
            for s in self.online_segments:
                if s['seq'] == seq:
                    segment = s
                    s['status'] = 'transcribing'
                    break
        
        if not segment:
            return

        temp_file = self._temp_audio_path(f"online_seg_{seq}")
        try:
            # Concatenate audio and save to temp file
            audio_np = np.concatenate(segment['audio'], axis=0).flatten()
            wav.write(temp_file, AUDIO_RUNTIME["sample_rate"], (audio_np * 32767).astype(np.int16))

            # Track transcription latency
            start_time = time.perf_counter()
            
            # One configured backend (one OpenAI client) per recording
            # session; only build a fresh one if the session backend is gone.
            backend = self.online_backend or self._get_online_backend()

            if self.online_with_edit:
                raw_text, edited_text = backend.transcribe_and_edit(temp_file)
                text = edited_text
            else:
                segments_result = backend.transcribe(temp_file)
                text = " ".join(seg.text for seg in segments_result).strip()
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            with self.online_segment_lock:
                segment['result'] = text
                segment['status'] = 'done'
                segment['latency'] = latency_ms
                segment['audio'] = None  # Free raw audio - duration_s was recorded at queue time

            self.log_internal(f"✅ Online segment {seq + 1} ⚡{latency_ms:.0f}ms")

        except Exception as e:
            with self.online_segment_lock:
                segment['status'] = 'error'
                segment['result'] = f"[Error: {e}]"
            self.log_internal(f"❌ Online segment {seq + 1} error: {e}")
        finally:
            try:
                os.remove(temp_file)
            except OSError:
                pass

        self._update_online_progress()

    def _update_online_progress(self):
        """Update UI with online transcription progress (called from pool threads)"""
        with self.online_segment_lock:
            total = len(self.online_segments)
            done = sum(1 for s in self.online_segments if s['status'] == 'done')

        if total > 0:
            progress_text = f"ONLINE ({done}/{total} chunks)"
            self._ui_after(0, lambda t=progress_text: self.status_var.set(t))

    def _show_progress(self, label_text="Loading..."):
        """Show progress bar in indeterminate mode"""
        self.progress_bar.configure(mode="indeterminate")
        self.progress_frame.pack(fill="x", padx=5, pady=0)
        self.progress_bar.start(15)
        self.progress_label.configure(text=label_text)
    
    def _hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", value=0)
        self.progress_frame.pack_forget()

    def log_internal(self, msg):
        self.msg_queue.put(msg)
    
    def copy_text(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        messagebox.showinfo("Copied", "Text copied to clipboard!")

    def save_transcription(self, text, latency_ms=0):
        """Save transcription to monthly log file and add to history"""
        if not text.strip():
            return
            
        timestamp = datetime.datetime.now()
        
        # Add to in-memory history with latency
        self.history.append((timestamp, text, latency_ms))
        self.current_transcription = text
        self.current_latency = latency_ms
        self.history_index = len(self.history) - 1  # Point to the latest
        
        # Save to per-machine monthly file: each machine appends only to its own
        # file, so simultaneous use on two machines can't produce Dropbox
        # 'conflicted copy' duplicates. Readers merge all month files.
        try:
            os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
            filename = f"{timestamp.strftime('%Y-%m')}.{self._hostname()}.txt"
            filepath = os.path.join(TRANSCRIPTIONS_DIR, filename)
            
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp_str}] {text}\n")
        except Exception as e:
            self.log_internal(f"Error saving transcription: {e}")

        # Keep the analytics cache in sync without re-reading the archive.
        # Tokens/language are appended too so refresh_insights_tabs never has to
        # re-tokenize the whole archive.
        if self._analytics_entries_cache is not None:
            self._analytics_entries_cache.append({"timestamp": timestamp, "text": text})
            if self._analytics_tokens_cache is not None and self._analytics_langs_cache is not None:
                toks = tokenize_words(text)
                self._analytics_tokens_cache.append(toks)
                self._analytics_langs_cache.append(detect_entry_language(text, toks))

        # Update display
        self.update_transcription_display()
        self._schedule_insights_refresh()

    def load_recent_transcriptions(self):
        """Load transcriptions from last 2 calendar days + today"""
        self.history = []
        today = datetime.date.today()
        cutoff_date = today - datetime.timedelta(days=TRANSCRIPTION_HISTORY_DAYS)
        
        try:
            if not os.path.exists(TRANSCRIPTIONS_DIR):
                return
            
            # Get relevant month files (current and possibly previous month)
            months_to_check = set()
            for i in range(TRANSCRIPTION_HISTORY_DAYS + 1):  # Today and recent days might span 2 months
                check_date = today - datetime.timedelta(days=i)
                months_to_check.add(check_date.strftime("%Y-%m"))

            # Merge all month files: legacy shared ("YYYY-MM.txt") and
            # per-machine ("YYYY-MM.HOSTNAME.txt"); skip Dropbox conflict debris
            for filename in sorted(os.listdir(TRANSCRIPTIONS_DIR)):
                if not filename.lower().endswith('.txt'):
                    continue
                if 'conflicted copy' in filename.lower():
                    continue  # avoid double-counting duplicated entries
                if filename[:7] not in months_to_check:
                    continue
                filepath = os.path.join(TRANSCRIPTIONS_DIR, filename)

                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse timestamp and text
                        # Format: [YYYY-MM-DD HH:MM:SS] text
                        if line.startswith('[') and '] ' in line:
                            try:
                                ts_end = line.index('] ')
                                ts_str = line[1:ts_end]
                                text = line[ts_end + 2:]
                                
                                timestamp = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                                
                                # Check if within date range
                                if timestamp.date() >= cutoff_date:
                                    # Archived lines carry no latency: pad to
                                    # the (timestamp, text, latency_ms) shape
                                    self.history.append((timestamp, text, 0))
                            except (ValueError, IndexError):
                                continue
            
            # Sort by timestamp
            self.history.sort(key=lambda x: x[0])
            
            # Recalculate robust Today stats ignoring volatile session data.
            if self.history:
                today_date = datetime.date.today()
                today_text = " ".join([text for dt, text, _lat in self.history if dt.date() == today_date])
                self.today_words = len(self._tokenize_words(today_text))
                self.today_audio_duration = (self.today_words / 130.0) * 60.0 # Estimate duration linearly
            
            if self.history:
                self.history_index = len(self.history) - 1
                self.current_transcription = self.history[-1][1]
                
        except Exception as e:
            self.log_internal(f"Error loading transcription history: {e}")

    def navigate_history(self, direction):
        """Navigate through transcription history. direction: -1 for prev, 1 for next"""
        if not self.history:
            return
        
        new_index = self.history_index + direction
        
        # Clamp to valid range
        if new_index < 0:
            new_index = 0
        elif new_index >= len(self.history):
            new_index = len(self.history) - 1
        
        if new_index != self.history_index:
            self.history_index = new_index
            entry = self.history[new_index]
            self.current_transcription = entry[1]
            self.current_latency = entry[2]
            self.update_transcription_display()

    def copy_current_transcription(self):
        """Copy the currently displayed transcription to clipboard"""
        if self.current_transcription:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.current_transcription)
            self.root.update()
            self.log_internal("Transcription copied to clipboard")

    def update_transcription_display(self):
        """Update the transcription body and the one-line header meta."""
        total = len(self.history)
        current = self.history_index + 1 if total > 0 else 0
        self.history_label.config(text=f"{current} / {total}")

        word_count = len(self.current_transcription.split()) if self.current_transcription else 0
        parts = []
        if 0 <= self.history_index < total:
            ts = self.history[self.history_index][0]
            try:
                parts.append(ts.strftime("%H:%M"))
            except Exception:
                pass
        parts.append(f"{word_count} words")
        if self.current_latency > 0:
            parts.append(f"{self.current_latency / 1000.0:.1f} s")
        self.word_count_label.config(text=(" · ".join(parts)))

        self.transcription_text.config(state='normal')
        self.transcription_text.delete("1.0", "end")

        if self.current_transcription:
            self.transcription_text.insert("1.0", self.current_transcription)
        else:
            self.transcription_text.insert("1.0", "No transcriptions yet...")

        self.transcription_text.config(state='disabled')

    def _merged_hourly_data(self):
        """Own hourly data combined with the other machines' (read-only).

        Cached: update_histogram() needs it twice per redraw and it only
        changes when record_hourly_words() mutates our own hourly stats.
        """
        if self._hourly_merge_cache is not None:
            return self._hourly_merge_cache
        merged = {}
        for source in (self.stats.get('hourly_data', {}), self._peer_stats.get('hourly_data', {})):
            for date_str, day_data in source.items():
                day = merged.setdefault(date_str, {})
                for hour_str, count in day_data.items():
                    day[hour_str] = day.get(hour_str, 0) + count
        self._hourly_merge_cache = merged
        return merged

    def get_histogram_data(self, for_today=True):
        """Get hourly word counts for histogram display"""
        hourly_data = self._merged_hourly_data()

        if for_today:
            # Get today's data only
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            day_data = hourly_data.get(today_str, {})
            return {int(h): count for h, count in day_data.items()}
        else:
            # Aggregate all days - return raw totals
            aggregated = {}
            for date_str, day_data in hourly_data.items():
                for hour_str, count in day_data.items():
                    hour = int(hour_str)
                    aggregated[hour] = aggregated.get(hour, 0) + count
            return aggregated

    def get_histogram_stats(self):
        """Get average and robust variability band per hour for all-time data."""
        hourly_data = self._merged_hourly_data()
        
        # Collect values per hour across all days
        hour_values = {h: [] for h in range(24)}
        
        for date_str, day_data in hourly_data.items():
            for hour_str, count in day_data.items():
                hour = int(hour_str)
                hour_values[hour].append(count)
        
        # Calculate statistics for each hour
        stats = {}
        for hour in range(6, 24):  # 6am to midnight
            values = hour_values[hour]
            if values:
                avg = sum(values) / len(values)
                variability = self._robust_variability_band(values)
                stats[hour] = {'avg': avg, 'var': variability, 'count': len(values)}
            else:
                stats[hour] = {'avg': 0, 'var': 0, 'count': 0}
        
        return stats

    def _schedule_histogram_refresh(self, delay_ms=2000):
        """Debounced histogram redraw (cancel-and-reschedule)."""
        if self._histogram_job is not None:
            try:
                self.root.after_cancel(self._histogram_job)
            except Exception:
                pass
        self._histogram_job = self.root.after(delay_ms, self._run_scheduled_histogram_refresh)

    def _run_scheduled_histogram_refresh(self):
        self._histogram_job = None
        self.update_histogram()

    def update_histogram(self):
        """Redraw both histograms"""
        # Draw today's histogram (dynamic hour range)
        today_data = self.get_histogram_data(for_today=True)
        self.draw_histogram_on_canvas(self.histogram_canvas_today, today_data, 
                                       fixed_hour_range=None, bars_list_name='histogram_bars_today')
        
        # Draw all-time histogram with averages and error bars
        alltime_stats = self.get_histogram_stats()
        self.draw_histogram_with_errorbars(self.histogram_canvas_alltime, alltime_stats)

    def draw_histogram_on_canvas(self, canvas, data, fixed_hour_range=None, bars_list_name='histogram_bars'):
        """Draw histogram bars on a specific canvas with teal gradient 3D effect"""
        canvas.delete("all")
        
        # Store bar rectangles for hover detection
        bars_list = []
        setattr(self, bars_list_name, bars_list)
        
        canvas.update_idletasks()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        # The canvases now live inside the collapsed Insights body, where an
        # unmapped widget reports 1px; fall back to a sane drawing size.
        if width < 50:
            width = 400
        if height < 30:
            height = 86
        
        if not data and not fixed_hour_range:
            canvas.create_text(width / 2, height / 2, 
                              text="No activity", fill="#666", font=("Arial", 9))
            return
        
        # Determine hour range - ALWAYS use 6am to 10pm for consistency
        min_hour, max_hour = 6, 22  # Fixed range for uniform bar width
        
        hours = list(range(min_hour, max_hour + 1))
        num_bars = len(hours)
        
        # Padding - extra space at bottom for hour labels
        left_padding = 8
        right_padding = 8
        top_padding = 8
        bottom_padding = 18  # Extra space for hour labels
        
        bar_area_width = width - left_padding - right_padding
        bar_area_height = height - top_padding - bottom_padding
        bar_width = max(8, (bar_area_width / num_bars) - 3)
        bar_spacing = bar_area_width / num_bars
        
        # Get max value for scaling
        max_val = max(data.values()) if data and data.values() else 1
        
        # Draw bars with teal gradient effect
        for i, hour in enumerate(hours):
            count = data.get(hour, 0) if data else 0
            bar_height = (count / max_val) * bar_area_height if max_val > 0 else 0
            
            x = left_padding + i * bar_spacing + bar_spacing / 2 - bar_width / 2
            y_bottom = height - bottom_padding
            y_top = y_bottom - max(bar_height, 2)  # Minimum 2px height for visibility
            
            if count > 0:
                # Teal gradient: lighter top, darker bottom
                intensity = count / max_val if max_val > 0 else 0
                
                # Base teal colors
                r_light = int(45 + intensity * 30)
                g_light = int(212 - intensity * 40)
                b_light = int(191 - intensity * 30)
                
                r_dark = int(19 + intensity * 20)
                g_dark = int(78 - intensity * 20)
                b_dark = int(74 - intensity * 20)
                
                color_light = f"#{r_light:02x}{g_light:02x}{b_light:02x}"
                color_dark = f"#{r_dark:02x}{g_dark:02x}{b_dark:02x}"
                
                # Draw 3D effect - darker base
                canvas.create_rectangle(x + 2, y_top + 2, x + bar_width + 2, y_bottom + 2, 
                                        fill=color_dark, outline="")
                
                # Main bar with teal color
                bar_id = canvas.create_rectangle(x, y_top, x + bar_width, y_bottom, 
                                                fill=color_light, outline="")
                
                # Highlight on top edge
                canvas.create_line(x, y_top, x + bar_width, y_top, 
                                  fill=COLOR_TEAL, width=1)
                
                bars_list.append({
                    'id': bar_id, 'hour': hour, 'count': count, 'max_val': max_val,
                    'x1': x, 'y1': y_top, 'x2': x + bar_width, 'y2': y_bottom
                })
        
        # Draw hour labels at bottom (every 4 hours: 6, 10, 14, 18, 22)
        for i, hour in enumerate(hours):
            if hour % 4 == 2 or hour == 6 or hour == 22:  # Show 6, 10, 14, 18, 22
                x = left_padding + i * bar_spacing + bar_spacing / 2
                canvas.create_text(x, height - 2, text=str(hour), 
                                  fill=COLOR_TEXT_MUTED, font=("Arial", 7), anchor="s")
        
        # Bind hover events for axis labels
        canvas.bind("<Motion>", lambda e, c=canvas, b=bars_list_name, mv=max_val, hrs=hours: self._on_hist_hover(e, c, b))
        canvas.bind("<Leave>", lambda e, c=canvas: c.delete("tooltip"))
    
    def _on_hist_hover(self, event, canvas, bars_list_name):
        """Show tooltip when hovering over a histogram bar"""
        canvas.delete("tooltip")
        
        bars_list = getattr(self, bars_list_name, [])
        
        for bar in bars_list:
            if bar['x1'] <= event.x <= bar['x2'] and bar['y1'] <= event.y <= bar['y2']:
                hour = bar['hour']
                count = bar['count']
                tooltip_text = f"{count}w @ {hour}:00"
                
                tx = (bar['x1'] + bar['x2']) / 2
                ty = max(15, bar['y1'] - 10)
                
                text_id = canvas.create_text(tx, ty, text=tooltip_text, fill="#fff", 
                                            font=("Arial", 8, "bold"), tags="tooltip")
                bbox = canvas.bbox(text_id)
                if bbox:
                    canvas.create_rectangle(bbox[0] - 3, bbox[1] - 2,
                                           bbox[2] + 3, bbox[3] + 2,
                                           fill="#333", outline="#555", tags="tooltip")
                    canvas.tag_raise(text_id)
                break

    def draw_histogram_with_errorbars(self, canvas, stats):
        """Draw histogram with teal gradient bars and robust variability whiskers."""
        canvas.delete("all")
        self.histogram_bars_alltime = []
        
        canvas.update_idletasks()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width < 50:
            width = 400
        if height < 30:
            height = 86
        
        # Use same hour range as today histogram for alignment (6am to 10pm)
        hours = list(range(6, 23))  # 6am to 10pm (matching today)
        num_bars = len(hours)
        
        # Padding - extra space at bottom for hour labels
        left_padding = 8
        right_padding = 8
        top_padding = 8
        bottom_padding = 18  # Extra space for hour labels
        
        bar_area_width = width - left_padding - right_padding
        bar_area_height = height - top_padding - bottom_padding
        bar_width = max(8, (bar_area_width / num_bars) - 3)
        bar_spacing = bar_area_width / num_bars
        
        # Get max value (avg + variability band) for scaling
        max_val = 1
        for hour in hours:
            if hour in stats:
                top = stats[hour]['avg'] + stats[hour]['var']
                if top > max_val:
                    max_val = top
        
        # Draw bars with teal gradient effect
        for i, hour in enumerate(hours):
            stat = stats.get(hour, {'avg': 0, 'var': 0, 'count': 0})
            avg = stat['avg']
            variability = stat['var']
            
            x = left_padding + i * bar_spacing + bar_spacing / 2 - bar_width / 2
            y_bottom = height - bottom_padding
            
            # Calculate bar height based on average
            bar_height = (avg / max_val) * bar_area_height if max_val > 0 else 0
            y_top = y_bottom - max(bar_height, 2) if avg > 0 else y_bottom
            
            if avg > 0:
                # Teal gradient: lighter top, darker bottom
                intensity = avg / max_val if max_val > 0 else 0
                
                # Base teal colors
                r_light = int(45 + intensity * 30)
                g_light = int(212 - intensity * 40)
                b_light = int(191 - intensity * 30)
                
                r_dark = int(19 + intensity * 20)
                g_dark = int(78 - intensity * 20)
                b_dark = int(74 - intensity * 20)
                
                color_light = f"#{r_light:02x}{g_light:02x}{b_light:02x}"
                color_dark = f"#{r_dark:02x}{g_dark:02x}{b_dark:02x}"
                
                # Draw 3D effect - darker base
                canvas.create_rectangle(x + 2, y_top + 2, x + bar_width + 2, y_bottom + 2, 
                                        fill=color_dark, outline="")
                
                # Main bar with teal color
                bar_id = canvas.create_rectangle(x, y_top, x + bar_width, y_bottom, 
                                                fill=color_light, outline="")
                
                # Highlight on top edge
                canvas.create_line(x, y_top, x + bar_width, y_top, 
                                  fill=COLOR_TEAL, width=1)

                # Draw capped robust variability whisker around the mean.
                if variability > 0:
                    x_center = x + (bar_width / 2)
                    upper = avg + variability
                    lower = max(0, avg - variability)
                    y_err_top = y_bottom - (upper / max_val) * bar_area_height
                    y_err_bottom = y_bottom - (lower / max_val) * bar_area_height
                    cap = 3
                    canvas.create_line(x_center, y_err_top, x_center, y_err_bottom, fill="#cbd5e1", width=1)
                    canvas.create_line(x_center - cap, y_err_top, x_center + cap, y_err_top, fill="#cbd5e1", width=1)
                    canvas.create_line(x_center - cap, y_err_bottom, x_center + cap, y_err_bottom, fill="#cbd5e1", width=1)
                
                self.histogram_bars_alltime.append({
                    'id': bar_id, 'hour': hour, 
                    'avg': avg, 'var': variability, 'count': stat['count'], 'max_val': max_val,
                    'x1': x, 'y1': y_top, 'x2': x + bar_width, 'y2': y_bottom
                })
        
        # Draw hour labels at bottom (every 4 hours: 6, 10, 14, 18, 22)
        for i, hour in enumerate(hours):
            if hour % 4 == 2 or hour == 6 or hour == 22:  # Show 6, 10, 14, 18, 22
                x = left_padding + i * bar_spacing + bar_spacing / 2
                canvas.create_text(x, height - 2, text=str(hour), 
                                  fill=COLOR_TEXT_MUTED, font=("Arial", 7), anchor="s")
        
        # Bind hover for stats tooltip
        canvas.bind("<Motion>", self._on_alltime_hist_hover)
        canvas.bind("<Leave>", lambda e: canvas.delete("tooltip"))

    def _on_alltime_hist_hover(self, event):
        """Show tooltip with avg +/- variability band when hovering over all-time histogram bar."""
        canvas = self.histogram_canvas_alltime
        canvas.delete("tooltip")
        
        for bar in self.histogram_bars_alltime:
            if bar['x1'] <= event.x <= bar['x2'] and bar['y1'] <= event.y <= bar['y2']:
                hour = bar['hour']
                avg = bar['avg']
                variability = bar['var']
                count = bar['count']
                tooltip_text = f"{avg:.0f}+/-{variability:.0f} words @ {hour}:00\n({count} days, robust variability)"
                
                tx = (bar['x1'] + bar['x2']) / 2
                ty = max(20, bar['y1'] - 15)
                
                text_id = canvas.create_text(tx, ty, text=tooltip_text, fill="#fff", 
                                            font=("Arial", 7), tags="tooltip", justify="center")
                bbox = canvas.bbox(text_id)
                if bbox:
                    canvas.create_rectangle(bbox[0] - 3, bbox[1] - 2,
                                           bbox[2] + 3, bbox[3] + 2,
                                           fill="#333", outline="#555", tags="tooltip")
                    canvas.tag_raise(text_id)
                break


    def update_gui_loop(self):
        # The whole body is guarded: one unexpected exception here used to kill
        # the loop permanently (no reschedule), freezing every live readout
        # while the app looked fine. The reschedule now lives in `finally`.
        loop_interval = 1000
        try:
            loop_interval = self._update_gui_loop_body()
        except Exception as e:
            self._log_once("update_gui_loop", f"update_gui_loop failed: {e!r}")
        finally:
            if not self._shutting_down:
                try:
                    self.root.after(loop_interval, self.update_gui_loop)
                except (tk.TclError, RuntimeError):
                    pass

    def _update_gui_loop_body(self):
        """One tick of the GUI refresh loop. Returns the next delay in ms."""
        # (Hotkey actions are drained by the dedicated 50ms _hotkey_pump loop)

        # 1. Logs (buffered until the System Log accordion is first expanded)
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self._append_log_line(f"[{timestamp}] {msg}\n")

        # 2. Results
        while not self.result_queue.empty():
            text_entry = self.result_queue.get()
            clean_text = text_entry.replace("[LIVE] ", "").replace("[BATCH] ", "").replace("[ONLINE] ", "")

            # --- COUNT WORDS ---
            if clean_text:
                count = len(clean_text.split())
                self.total_words += count
                self.today_words += count
                self.record_hourly_words(count)  # Track hourly activity
                self.save_stats()  # Persist stats after each transcription
                self.save_transcription(clean_text, self.last_transcription_latency)
                self.last_transcription_latency = 0  # Reset after use
                self._schedule_histogram_refresh()  # Refresh histogram (debounced)
            # -------------------

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            if "[LIVE]" in text_entry:
                mode_tag = "LIVE"
            elif "[BATCH]" in text_entry:
                mode_tag = "BATCH"
            else:
                mode_tag = "ONLINE"
            self._append_log_line(
                f"[{timestamp}] {mode_tag}: {len(clean_text.split())} words transcribed\n")

        # --- STATISTICS (cheap arithmetic; also runs while idle so the status
        # strip and the Insights summary line stay current) ---
        alltime_words = self.total_words + self._peer_stats.get('total_words', 0)
        alltime_audio_duration = self.total_audio_duration + self._peer_stats.get('total_audio_duration', 0.0)
        alltime_mins_saved = alltime_words / TYPING_WPM
        alltime_spoken_mins = alltime_audio_duration / 60.0

        today_mins_saved = self.today_words / TYPING_WPM
        today_spoken_mins = self.today_audio_duration / 60.0

        alltime_speed = int(alltime_words / alltime_spoken_mins) if alltime_spoken_mins > 0 else 0
        session_speed = int(self.today_words / today_spoken_mins) if today_spoken_mins > 0 else 0

        session_avg_latency = sum(self.session_latencies) / len(self.session_latencies) if self.session_latencies else 0
        alltime_avg_latency = sum(self.total_latencies[-100:]) / len(self.total_latencies[-100:]) if self.total_latencies else 0
        avg_latency = session_avg_latency or alltime_avg_latency

        # All-time block now lives inside the expanded Speech Insights body
        self.lbl_alltime_words.config(text=f"{alltime_words:,}")
        self.lbl_alltime_time.config(text=f"{self.format_time_saved(alltime_mins_saved)}")
        self.lbl_alltime_duration.config(text=f"{self.format_speaking_time(alltime_audio_duration)}")
        self.lbl_alltime_speed.config(text=f"{alltime_speed} WPM")

        self.status_stats_var.set(
            f"{avg_latency / 1000.0:.1f} s · {today_spoken_mins:.0f} min today"
            f" · saved {self.format_time_saved(today_mins_saved)}"
        )
        self._update_insights_summary(self.today_words, session_speed)
        self._update_state_card()

        # 3. VU meter (progress colour follows the active mode)
        display_val = min(1.0, self.current_display_volume * AUDIO_RUNTIME["live_buffer_gain"])
        self.vu_meter.set(display_val)

        # 4. Mini window
        if self.mini_window and self.mini_canvas:
            x0 = getattr(self, "_mini_vu_x0", 27)
            x1 = getattr(self, "_mini_vu_x1", UI_RUNTIME["mini_window_width"] - 68)
            frac = min(1.0, display_val * AUDIO_RUNTIME["mini_vu_multiplier"])
            self.mini_canvas.coords(self.mini_vu, x0, 31, x0 + (x1 - x0) * frac, 35)

            elapsed = self.state_elapsed_var.get() or "0:00"
            if self.mode == "live":
                status_txt = f"LIVE · {elapsed}"
            elif self.mode == "batch":
                with self.batch_segment_lock:
                    total = len(self.batch_segments)
                    done = sum(1 for seg in self.batch_segments if seg['status'] == 'done')
                status_txt = (f"Transcribing · {done} of {total}" if total > 0
                              else f"BATCH · {elapsed}")
            elif self._is_online_mode():
                with self.online_segment_lock:
                    total = len(self.online_segments)
                    done = sum(1 for seg in self.online_segments if seg['status'] == 'done')
                status_txt = (f"Transcribing · {done} of {total}" if total > 0
                              else f"ONLINE · {elapsed}")
            elif getattr(self, "_mini_state_color", None) == COLOR_READY and self.current_transcription:
                status_txt = f"Copied · {len(self.current_transcription.split())} words"
            else:
                status_txt = f"Idle · {self.config['hotkey_live'].upper()}"

            self.mini_canvas.itemconfigure("status_text", text=status_txt)
            self.mini_canvas.itemconfigure("stats_text", text=f"{self.today_words:,} w")
            self.mini_canvas.itemconfigure(
                "stats_text2", text=f"{self.format_speaking_time(self.today_audio_duration)}")

        # Adaptive loop speed: fast (30ms) when transcribing, slow (1000ms) when idle
        return UI_RUNTIME["main_loop_interval_ms"] if self.mode else 1000

    def load_model(self):
        """Serialise model loads. The startup thread, save_config(),
        _switch_to_cpu_mode() and the idle-reload path all call this.
        _model_lock is an RLock, so _reload_model_if_needed_locked() (which
        already holds it) re-enters safely instead of deadlocking."""
        with self._model_lock:
            self._load_model_locked()

    def _manual_model_reload(self):
        """'Reload Model' button: retry a load that previously failed."""
        self._model_error_dialog_shown = False
        self._model_idle_unloaded = False
        self.status_var.set("Reloading model...")
        self.log_internal("Manual model reload requested.")
        threading.Thread(target=self.load_model, daemon=True).start()

    def _download_status_text(self, model_id):
        """First-run status line: say what is happening and roughly how big it
        is, so a several-minute wait does not look like a frozen app."""
        short = str(model_id).split("/")[-1]
        size = self.MODEL_DOWNLOAD_SIZES.get(str(model_id).lower())
        if size:
            return f"Downloading model {short} (~{size})… first run only"
        return f"Downloading model {short}… first run only"

    def _report_model_load_failure(self, error):
        """Make a failed model load impossible to miss.

        Before this, a load failure left the status bar reading "Loading..."
        forever with the traceback going to a stdout that does not exist
        under pythonw.exe.
        """
        message = str(error)
        log.error("Model load failed: %s", message, exc_info=True)
        try:
            self.log_internal(f"❌ Model load failed: {message}")
        except Exception:
            pass
        self._ui_after(0, lambda: self.status_var.set("Model load failed – see System Log"))
        self._ui_after(0, lambda: self._set_state_dot(COLOR_LIVE))
        if getattr(self, "_model_error_dialog_shown", False) or NO_DIALOGS:
            return
        self._model_error_dialog_shown = True
        detail = (
            f"neurowhisper could not load the speech model.\n\n{message}\n\n"
            "Check your internet connection for the first download, then use "
            "Reload Model."
        )
        self._ui_after(0, lambda: messagebox.showerror("Model load failed", detail))

    def _load_model_locked(self):
        # Set by the backend-failure path below so the legacy block loads on
        # CPU instead of retrying the device that just failed.
        legacy_device = None
        legacy_compute_type = None
        try:
            friendly_name = self.config['model_key']
            model_id = MODEL_MAP.get(friendly_name, "medium")
            
            # --- 1. REFRESH DROPDOWN INDICATORS ---
            # This ensures we see the arrow/checkmark correctly on startup/reload
            self._ui_after(0, self.refresh_model_list)
            
            self.log_internal(f"Init {friendly_name}...")
            
            safe_name = friendly_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            local_models_root = os.path.join(app_dir, "models")
            target_dir = os.path.join(local_models_root, safe_name)
            
            # --- 2. DETERMINE BACKEND ---
            backend_name = self.config.get('backend', 'auto')
            device = self.config.get('device', 'auto')
            compute_type = self.config.get('compute_type', 'auto')
            
            # Check if this is a Parakeet model - requires Parakeet backend
            is_parakeet_model = 'parakeet' in model_id.lower() or 'nvidia/parakeet' in model_id.lower()
            
            # Use backend abstraction if available
            if BACKENDS_AVAILABLE and backend_name != 'legacy':
                # Force Parakeet backend for Parakeet models
                if is_parakeet_model:
                    backend_name = 'parakeet'
                    self.log_internal(f"[Parakeet model detected] Using Parakeet backend")
                    # Parakeet works best with CUDA
                    if device == 'auto':
                        try:
                            import torch
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        except ImportError:
                            device = 'cpu'
                # Auto-detect best backend if needed (for non-Parakeet models)
                elif backend_name == 'auto' or device == 'auto':
                    detected_backend, detected_device, reason = detect_best_backend()
                    if backend_name == 'auto':
                        # The detected backend brings its own device, and it
                        # overrides the persisted one. load_config() resolves
                        # device with a CUDA-or-CPU probe that knows nothing
                        # about OpenVINO, so a persisted 'cpu' would otherwise
                        # be handed to the OpenVINO backend, which expects
                        # 'GPU'/'NPU'. A device the user pinned explicitly is
                        # untouched, because then backend is not 'auto'.
                        backend_name = detected_backend
                        if detected_device and detected_device != device:
                            self.log_internal(
                                f"[Auto-detect] device {device} -> {detected_device} "
                                f"(follows the auto-selected {detected_backend} backend)"
                            )
                        device = detected_device
                        # The old compute_type belongs to the old device.
                        compute_type = 'auto'
                    elif device == 'auto':
                        device = detected_device
                    self.log_internal(f"[Auto-detect] {reason}")
                
                # Create backend instance
                self.backend = create_backend(backend_name)
                self.current_backend_name = backend_name
                self.current_device = device
                
                # Resolve compute type
                if compute_type == 'auto':
                    if hasattr(self.backend, 'get_optimal_compute_type'):
                        compute_type = self.backend.get_optimal_compute_type(device)
                    else:
                        compute_type = 'int8' if device == 'cpu' else 'float16'
                
                self.log_internal(f"Loading {backend_name} backend ({device}, {compute_type})...")
                
                # Progress bar for download. Never for the cloud backend:
                # it has no local model, so "Downloading model ..." would be a
                # lie and the indeterminate bar would spin forever.
                if backend_name != ONLINE_BACKEND_NAME and not self.is_model_downloaded(friendly_name):
                    download_text = self._download_status_text(model_id)
                    self.log_internal(download_text)
                    self._ui_after(0, lambda: self._show_progress("Downloading..."))
                    self._ui_after(0, lambda t=download_text: self.status_var.set(t))
                
                # Load model via backend - with fallback on failure.
                # configure() is inside the try: without an API key it raises,
                # and that belongs to the "set your OpenAI key" branch below,
                # not to the generic model-load-failure dialog.
                try:
                    if backend_name == 'openai':
                        self.backend.configure(
                            api_key=self.config.get('openai_api_key', ''),
                            transcription_model=self.config.get('openai_transcription_model', DEFAULT_CONFIG['openai_transcription_model']),
                            edit_model=self.config.get('openai_edit_model', DEFAULT_CONFIG['openai_edit_model']),
                            edit_prompt=self.config.get('openai_edit_prompt', '') or None,
                            language=self.config.get('openai_language', DEFAULT_CONFIG['openai_language']),
                        )

                    self.backend.load_model(
                        model_key=friendly_name,
                        device=device,
                        compute_type=compute_type,
                        model_path=target_dir
                    )
                    
                    # For compatibility with existing transcription code
                    self.model = self.backend.model if hasattr(self.backend, 'model') else self.backend
                    
                    self._ui_after(0, self._hide_progress)
                    self._ui_after(0, self.refresh_model_list)
                    self.log_internal(f"Model Ready. [{backend_name.upper()} / {device.upper()}]")
                    self._ui_after(0, lambda: self.status_var.set(f"Ready [{backend_name} / {device}]"))
                    self._ui_after(0, self._update_device_model_labels)
                    return
                except Exception as backend_error:
                    # Backend failed
                    self.log_internal(f"⚠️ {backend_name} failed: {backend_error}")
                    
                    # Don't fall back to faster-whisper for Parakeet models - they're incompatible
                    if is_parakeet_model:
                        error_msg = (
                            f"Parakeet model requires NeMo to be installed.\n\n"
                            f"Error: {backend_error}\n\n"
                            f"To install NeMo, run:\n"
                            f'pip install "nemo_toolkit[asr]"\n\n'
                            f"Then restart the application."
                        )
                        self.log_internal('❌ Parakeet requires NeMo. Run: pip install "nemo_toolkit[asr]"')
                        self._ui_after(0, self._hide_progress)
                        if not NO_DIALOGS:
                            self._ui_after(0, lambda: messagebox.showerror("NeMo Required", error_msg))
                        self._ui_after(0, lambda: self.status_var.set("Error: NeMo not installed"))
                        return
                    
                    if backend_name == 'openai' or device == 'cloud':
                        # Cloud backend has no local fallback (usually a missing API key)
                        self.backend = None
                        self._ui_after(0, self._hide_progress)
                        self._ui_after(0, lambda: self.status_var.set("Online mode - set OpenAI API key in Configuration"))
                        return

                    # For other backends, fall back to faster-whisper on CPU.
                    # The legacy block below must actually USE cpu/int8 - reading
                    # self.config here would retry the very device that just
                    # failed (e.g. cuda/float16) and fail again.
                    self.log_internal("Falling back to CPU (int8)")
                    log.warning("Backend %s failed (%s); falling back to CPU (int8)",
                                backend_name, backend_error)
                    legacy_device = 'cpu'
                    legacy_compute_type = 'int8'
                    self._ui_after(0, self._hide_progress)
                    # Fall through to legacy code below

            # --- FALLBACK: Original WhisperModel code (legacy mode) ---
            if legacy_device is None:
                legacy_device = self.config.get('device', 'cpu')
                legacy_compute_type = self.config.get('compute_type', 'int8')
            if legacy_device == 'auto':
                legacy_device = 'cpu'
            if legacy_compute_type == 'auto':
                legacy_compute_type = 'int8' if legacy_device == 'cpu' else 'float16'
            self.backend = None
            self.current_backend_name = 'faster-whisper'
            self.current_device = legacy_device
            
            # --- 2. PROGRESS BAR SETUP ---
            # Check if likely already downloaded first to avoid flashing progress bar
            if not self.is_model_downloaded(friendly_name) and snapshot_download:
                download_text = self._download_status_text(model_id)
                self.log_internal(download_text)

                # Show progress bar in indeterminate mode (animated)
                def show_progress():
                    self.progress_bar.configure(mode="indeterminate")
                    self.progress_frame.pack(fill="x", padx=5, pady=0)
                    self.progress_bar.start(15)  # Animation speed
                    self.progress_label.configure(text="Downloading...")

                def hide_progress():
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate", value=0)
                    self.progress_frame.pack_forget()

                self._ui_after(0, show_progress)
                self._ui_after(0, lambda t=download_text: self.status_var.set(t))

                try:
                   repo_id = model_id
                   if "/" not in model_id: repo_id = f"systran/faster-whisper-{model_id}"

                   # No local_dir_use_symlinks: huggingface_hub 1.x removed the
                   # kwarg and raises TypeError if it is passed.
                   snapshot_download(repo_id=repo_id, local_dir=target_dir)
                finally:
                   # Hide progress bar
                   self._ui_after(0, hide_progress)

            # ----------------------------------
            
            if not os.path.exists(target_dir) or not os.listdir(target_dir):
                self.migrate_from_cache(model_id, target_dir)

            os.makedirs(target_dir, exist_ok=True)
            
            # Check if model files exist directly in target_dir (from snapshot_download)
            # Look for model.bin which is the main model file
            model_bin_path = os.path.join(target_dir, "model.bin")
            if os.path.exists(model_bin_path):
                # Model was downloaded via snapshot_download directly into target_dir
                # Load from the local path directly
                self.log_internal(f"Loading model from local path...")
                from faster_whisper import WhisperModel
                self.model = WhisperModel(
                    target_dir,  # Use the directory path directly
                    device=legacy_device,
                    compute_type=legacy_compute_type
                )
            else:
                # Fallback: model needs to be downloaded or is in cache format
                from faster_whisper import WhisperModel
                self.model = WhisperModel(
                    model_id,
                    device=legacy_device,
                    compute_type=legacy_compute_type,
                    download_root=target_dir
                )
            
            # Refresh again to show checkmark now that it's downloaded
            self._ui_after(0, self.refresh_model_list)

            self.log_internal("Model Ready.")
            self._ui_after(0, lambda: self.status_var.set("Ready."))
        except Exception as e:
            error_str = str(e).lower()
            self.log_internal(f"Load Error: {e}")
            self._ui_after(0, lambda: self.progress_frame.pack_forget())

            # Check for CUDA/cuDNN specific errors
            is_cuda_error = any(kw in error_str for kw in CUDA_ERROR_KEYWORDS)

            if is_cuda_error and self.config.get('device') == 'cuda':
                # Keep the existing, more specific CUDA dialog for this case,
                # but still make the failure visible in the status bar.
                log.error("Model load failed (CUDA): %s", e, exc_info=True)
                self._ui_after(0, lambda: self.status_var.set("Model load failed – see System Log"))
                self._ui_after(0, lambda: self._set_state_dot(COLOR_LIVE))
                self._ui_after(0, lambda err=e: self._show_cuda_error_dialog(str(err)))
            else:
                self._report_model_load_failure(e)

    def migrate_from_cache(self, model_id, target_dir):
        try:
            repo_id = model_id
            if "/" not in model_id: repo_id = f"systran/faster-whisper-{model_id}"
            cache_name = f"models--{repo_id.replace('/', '--')}"
            default_cache_path = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", cache_name))
            
            if os.path.exists(default_cache_path):
                snapshots_dir = os.path.join(default_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = os.listdir(snapshots_dir)
                    if snapshots:
                        source_dir = os.path.join(snapshots_dir, snapshots[-1])
                        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        except Exception as e:
            self.log_internal(f"Migration skip: {e}")

    def _show_cuda_error_dialog(self, error_msg):
        """Show a helpful dialog when CUDA/cuDNN fails to load"""
        dialog = tk.Toplevel(self.root)
        dialog.title("CUDA Libraries Required")
        dialog.geometry(UI_RUNTIME["dialog_geometry_cuda"])
        dialog.configure(bg=COLOR_BG)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 250
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 160
        dialog.geometry(f"+{x}+{y}")
        
        # Title
        tk.Label(dialog, text="⚠️ CUDA Libraries Not Found", 
                bg=COLOR_BG, fg="#FFA726", font=("Arial", 14, "bold")).pack(pady=(15, 10))
        
        # Explanation
        explanation = (
            "GPU acceleration requires NVIDIA cuDNN and cuBLAS libraries.\n\n"
            "To enable GPU mode:\n"
            "1. Click 'Download CUDA DLLs' below\n"
            "2. Download the archive for your system\n"
            "3. Extract DLL files to the WhisperTyper folder\n"
            "4. Restart the application\n\n"
            "Or use CPU mode (slower but works without GPU)."
        )
        tk.Label(dialog, text=explanation, bg=COLOR_BG, fg=COLOR_FG, 
                font=("Arial", 10), justify="left", wraplength=450).pack(padx=20, pady=5)
        
        # Error details (collapsible)
        error_frame = ttk.LabelFrame(dialog, text="Error Details", style="Dialog.TLabelframe")
        error_frame.pack(fill="x", padx=20, pady=10)
        error_label = tk.Label(error_frame, text=error_msg[:200] + "..." if len(error_msg) > 200 else error_msg,
                              bg=COLOR_BG, fg="#888", font=("Consolas", 8), wraplength=440)
        error_label.pack(padx=10, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(dialog, style="Dialog.TFrame")
        btn_frame.pack(pady=15)
        
        def open_download():
            webbrowser.open(CUDA_DLL_URL)
        
        def use_cpu():
            dialog.destroy()
            self._switch_to_cpu_mode()
        
        tk.Button(btn_frame, text="📥 Download CUDA DLLs", command=open_download,
                 bg="#1976d2", fg="white", font=("Arial", 10, "bold"), 
                 padx=15, pady=8, cursor="hand2").pack(side="left", padx=10)
        
        tk.Button(btn_frame, text="🖥️ Use CPU Mode", command=use_cpu,
                 bg=COLOR_ACCENT_BG, fg="white", font=("Arial", 10), 
                 padx=15, pady=8, cursor="hand2").pack(side="left", padx=10)
        
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy,
                 bg=COLOR_IDLE, fg="white", font=("Arial", 10), 
                 padx=15, pady=8).pack(side="left", padx=10)

    def _switch_to_cpu_mode(self):
        """Switch to CPU mode and reload the model"""
        self.log_internal("Switching to CPU mode...")
        self.status_var.set("Switching to CPU mode...")
        
        # Update config
        self.config['device'] = 'cpu'
        self.root.after(0, self._update_device_model_labels)
        self.config['compute_type'] = 'int8'  # Best for CPU

        # Save config
        self._persist_config()
        
        # Reload model on CPU
        threading.Thread(target=self.load_model, daemon=True).start()

    def audio_callback(self, indata, frames, time, status):
        if self.mode is not None:
            self.audio_queue.put(indata.copy())

    def start_audio_stream(self):
        """Open the input stream for a recording. Returns True on success.

        The microphone is only held while recording; stop_audio_stream()
        releases it again when the recording ends.
        """
        if self.stream is not None:
            return True
        dev_id = self.config.get('input_device')
        try:
            self.stream = sd.InputStream(
                device=dev_id,
                samplerate=AUDIO_RUNTIME["sample_rate"],
                channels=AUDIO_RUNTIME["channels"],
                callback=self.audio_callback,
                blocksize=AUDIO_RUNTIME["blocksize"],
            )
            self.stream.start()
            return True
        except Exception as e:
            self.stream = None
            self._last_audio_error = str(e)
            log.error("Could not open the input stream (device=%r): %s", dev_id, e)
            self.log_internal(f"Audio Device Fail: {e}")
            try:
                self.status_var.set(f"Audio Device Fail: {e}")
            except Exception:
                pass
            return False

    def stop_audio_stream(self):
        """Stop and close the input stream, releasing the microphone."""
        stream = self.stream
        self.stream = None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            self.log_internal(f"Audio stream close failed (continuing): {e}")

    def restart_audio_stream(self):
        """Input device / config changed: drop any open stream so the next
        recording opens the newly selected device.

        Both callers (on_mic_change, save_config) are reachable mid-recording
        via the always-live global hotkeys, so if a recording is in progress the
        stream has to be reopened immediately - otherwise audio_callback never
        fires again and the recording silently ends up empty."""
        was_open = self.stream is not None
        self.stop_audio_stream()
        if was_open and self.mode is not None and not self._shutting_down:
            self.start_audio_stream()

    def on_close(self):
        """Orderly shutdown: stop recording, flush stats, release the mic,
        tear down executors and the model worker, unregister hotkeys.
        Idempotent - the __main__ finally block calls it again."""
        if self._shutting_down:
            return
        self._shutting_down = True

        self.running = False

        # Stop whatever is recording. Done inline rather than via
        # _stop_any_recording(): that spawns the async finalisation threads,
        # which would sleep, transcribe and then touch widgets that this
        # method is about to destroy.
        self.stopping = True
        self.mode = None
        try:
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
        except Exception:
            pass

        # Persist any debounced stats right now
        try:
            self.flush_stats()
        except Exception:
            pass

        # Release the microphone
        try:
            self.stop_audio_stream()
        except Exception:
            pass

        # Executor shutdown and model unload both go on a helper thread with a
        # bounded join. Verified bug: closing the window while the very first
        # model download was still running blocked here forever, because
        # unload_model() waits on the worker that is busy downloading.
        def _release_heavy_resources():
            for attr in ('online_executor', 'batch_executor'):
                executor = getattr(self, attr, None)
                if executor is not None:
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    setattr(self, attr, None)
            try:
                if self.backend is not None:
                    self.backend.unload_model()
            except Exception as e:
                log.warning("unload_model() during shutdown failed: %s", e)

        # Captured before the thread starts: _teardown_worker() waits up to 3 s
        # for a clean shutdown, so when our own 3 s join expires the child may
        # still be alive - and the releaser is a daemon thread, so nobody would
        # ever kill it. An orphaned worker keeps its CUDA context, which is the
        # exact thing worker mode exists to release.
        worker_proc = getattr(self.backend, "_proc", None) if self.backend is not None else None

        releaser = threading.Thread(target=_release_heavy_resources, daemon=True)
        releaser.start()
        releaser.join(timeout=3)
        if releaser.is_alive():
            log.warning("Model/executor teardown did not finish in 3s - closing anyway.")
            if worker_proc is not None and worker_proc.poll() is None:
                log.warning("Killing the model worker process (pid %s) so it cannot be orphaned.",
                            worker_proc.pid)
                try:
                    worker_proc.kill()
                except Exception as e:
                    log.error("Could not kill the worker process: %s", e)

        try:
            self.cleanup_hotkeys()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    # Startup is wrapped because under pythonw.exe a failure here would be
    # completely invisible: no console, no window, no message. The guard
    # covers construction and entering the mainloop ONLY - a failure while
    # shutting down is not a failure to start, and must not raise a dialog
    # that says it was.
    #
    # `except Exception`, not BaseException: SystemExit and KeyboardInterrupt
    # are how this process is meant to end, and swallowing SystemExit here
    # would turn a deliberate exit into a "failed to start" dialog.
    app = None
    try:
        root = tk.Tk()
        app = WhisperApp(root)
        root.mainloop()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        crash_text = traceback.format_exc()
        log.error("neurowhisper failed to start:\n%s", crash_text)
        crash_path = write_crash_log(crash_text)
        show_fatal_message(
            "neurowhisper failed to start",
            "neurowhisper could not start.\n\n"
            + crash_text.strip().splitlines()[-1]
            + "\n\nFull details were written to:\n"
            + crash_path,
        )
        sys.exit(1)
    finally:
        # Idempotent: a window close already ran this via WM_DELETE_WINDOW.
        # Teardown problems are logged, never shown as a startup failure.
        if app is not None:
            try:
                app.on_close()
            except Exception:
                log.error("Error during shutdown (ignored):", exc_info=True)
        # Give hotkeys time to unregister
        try:
            time.sleep(HOTKEY_RUNTIME["shutdown_unregister_sleep_seconds"])
        except Exception:
            pass
