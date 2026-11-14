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
    LEGACY_MODEL_ALIASES,
    LIVE_PAUSE_RANGE,
    MODEL_DOWNLOAD_SIZES,
    ONLINE_BACKEND_NAME,
    ONNX_ASR_MODEL_IDS,
    MODEL_MAP,
    OPENAI_EDIT_MODEL_OPTIONS,
    OPENAI_KEY_URL,
    OPENAI_TRANSCRIPTION_MODEL_OPTIONS,
    PEER_BENCHMARK_NOTE,
    PEER_BENCHMARKS,
    PEER_PROFILE,
    INSIGHTS_RANGE_OPTIONS,
    INSIGHTS_RANGE_DEFAULT,
    INSIGHTS_QUICK_CHIPS,
    INSIGHTS_BUCKET_OPTIONS,
    INSIGHTS_LANG_OPTIONS,
    INSIGHTS_MIN_WORDS_CONFIDENT,
    INSIGHTS_OUTLIER_SIGMA,
    LEXICAL_CACHE_PREFIX,
    ACOUSTIC_META_SUBDIR,
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
    resolve_backend_display,
)

try:
    import lexical_insights
except Exception:
    lexical_insights = None

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
import itertools
import types as _types

SOUNDDEVICE_ERROR = None
try:
    import sounddevice as sd
except OSError as _sd_exc:
    SOUNDDEVICE_ERROR = str(_sd_exc)
    log.warning("Audio library unavailable (%s); microphone capture disabled.", _sd_exc)
    sd = _types.ModuleType("sounddevice")
    sd.default = _types.SimpleNamespace(device=None, samplerate=None, channels=None, dtype=None)
    sd.PortAudioError = type("PortAudioError", (OSError,), {})
    sd.query_devices = lambda *a, **kw: []
    sd.query_hostapis = lambda *a, **kw: []
    sd.InputStream = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError(f"PortAudio library not found ({SOUNDDEVICE_ERROR})")
    )

import numpy as np
from backends.base import write_pcm16_wav


class _FastStdlibWav:
    """Drop-in ``wav.write(path, rate, data)`` backed by stdlib ``wave`` (0ms import, clipped int16)."""

    @staticmethod
    def write(filename, rate, data):
        write_pcm16_wav(filename, data, sample_rate=rate)


wav = _FastStdlibWav()

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
        IS_LINUX,
        PLATFORM_NAME,
        KEYBOARD_ERROR,
    )
    PLATFORM_UTILS_AVAILABLE = True
except ImportError:
    # Fallback to direct imports if platform_utils not available
    import keyboard
    IS_MAC = False
    IS_WINDOWS = True
    IS_LINUX = sys.platform.startswith("linux")
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
    from backends import create_backend, detect_best_backend, unload_all_backends
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    unload_all_backends = None

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

_TEMP_FILE_COUNTER = itertools.count(1)


def _get_snapshot_download():
    """Lazy import of ``huggingface_hub.snapshot_download`` (used only by legacy fallback)."""
    try:
        from huggingface_hub import snapshot_download as _sd
        return _sd
    except ImportError:
        return None


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
        # Set of currently held key names (lowercase, normalized)
        self._pressed_keys = set()
        self._fired_hotkeys = set()
        self.paused = False
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

    _KEY_ALIASES = {
        "left windows": "win",
        "right windows": "win",
        "windows": "win",
        "super": "win",
        "left super": "win",
        "right super": "win",
        "left ctrl": "ctrl",
        "right ctrl": "ctrl",
        "control": "ctrl",
        "left alt": "alt",
        "right alt": "alt",
        "alt gr": "alt",
        "left shift": "shift",
        "right shift": "shift",
        "left command": "command",
        "right command": "command",
        "cmd": "command",
    }

    @classmethod
    def _normalize_key_name(cls, name):
        n = (name or "").strip().lower()
        return cls._KEY_ALIASES.get(n, n)

    # ------------------------------------------------------------------
    # Parse hotkey strings into sets of lowercase key names
    # ------------------------------------------------------------------
    @classmethod
    def _parse_hotkey(cls, hotkey_str):
        """Parse 'ctrl+alt+s' or 'F9' into a frozenset of normalized lowercase names."""
        parts = [cls._normalize_key_name(p) for p in hotkey_str.split("+")]
        return frozenset(p for p in parts if p)

    # ------------------------------------------------------------------
    # The single global hook – replaces both heartbeat and add_hotkey
    # ------------------------------------------------------------------
    def _install_hook(self):
        self._remove_hook()
        if self._keyboard is None:
            self._install_error = "Keyboard backend unavailable"
            self._log("Hook install skipped: keyboard backend unavailable")
            return
        try:
            def _on_key(event):
                now = time.monotonic()
                prev_event = self._last_hook_event
                self._last_hook_event = now

                name = getattr(event, "name", None)
                if not name:
                    return
                name_lower = self._normalize_key_name(name)
                event_type = getattr(event, "event_type", None)

                if self.paused:
                    if event_type == "up":
                        with self._lock:
                            self._pressed_keys.discard(name_lower)
                    return

                cb_to_fire = None
                hk_to_fire = None
                if event_type == "down":
                    with self._lock:
                        # Clear orphaned modifiers left behind by Win+L lock screen / UAC
                        if now - prev_event > 8.0 and self._pressed_keys:
                            self._pressed_keys.clear()
                            self._fired_hotkeys.clear()
                        self._pressed_keys.add(name_lower)

                        matched = []
                        for hk_name, binding in self._bindings.items():
                            req = binding["keys"]
                            if req and req.issubset(self._pressed_keys):
                                matched.append((len(req), hk_name, binding["callback"]))
                            else:
                                self._fired_hotkeys.discard(hk_name)

                        if matched:
                            matched.sort(key=lambda item: item[0], reverse=True)
                            _sz, best_name, best_cb = matched[0]
                            for _s, shadowed_name, _c in matched[1:]:
                                self._fired_hotkeys.add(shadowed_name)
                            if best_name not in self._fired_hotkeys:
                                self._fired_hotkeys.add(best_name)
                                last = self._last_trigger.get(best_name, 0.0)
                                if now - last >= self.DEBOUNCE_S:
                                    self._last_trigger[best_name] = now
                                    hk_to_fire = best_name
                                    cb_to_fire = best_cb

                    if cb_to_fire is not None:
                        try:
                            cb_to_fire()
                        except Exception as exc:
                            self._flog(f"Hotkey callback error ({hk_to_fire}): {exc}")
                elif event_type == "up":
                    with self._lock:
                        self._pressed_keys.discard(name_lower)
                        for hk_name, binding in self._bindings.items():
                            if not binding["keys"].issubset(self._pressed_keys):
                                self._fired_hotkeys.discard(hk_name)

            self._hook_ref = self._keyboard.hook(_on_key, suppress=False)
            self._flog("Hook-based hotkey handler installed")
        except Exception as e:
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
        from backends.faster_whisper_backend import FasterWhisperBackend
        return FasterWhisperBackend().check_cuda()
    except Exception:
        pass
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False, "No CUDA GPU detected (ctranslate2)"
        supported_devices = ctranslate2.get_supported_compute_types("cuda")
        if supported_devices:
            return True, "CUDA detected and available"
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
COLOR_FG_MUTED = THEME_COLORS["text_muted"]
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

_COMPILED_FILLERS = {
    "en": {k: re.compile(p) for k, p in ENGLISH_FILLER_PATTERNS.items()},
    "de": {k: re.compile(p) for k, p in GERMAN_FILLER_PATTERNS.items()},
}


def tokenize_words(text):
    """Unicode-aware word tokenizer (keeps umlauts, drops digits/punctuation)."""
    return _WORD_RE.findall((text or "").lower())


def detect_entry_language(text, tokens=None):
    """Cheap per-entry language heuristic -> 'de' or 'en'."""
    if not text:
        return "en"
    if tokens is None:
        tokens = tokenize_words(text)
    umlaut_count = sum(1 for ch in text if ch in _GERMAN_UMLAUTS)
    if not tokens:
        return "de" if umlaut_count > 0 else "en"
    hits = sum(1 for t in tokens if t in GERMAN_FUNCTION_WORDS)
    if umlaut_count > 0 and (umlaut_count >= 2 or len(tokens) <= 5 or hits >= 1):
        return "de"
    return "de" if (hits / len(tokens)) >= GERMAN_DETECT_THRESHOLD else "en"


def filler_patterns_for(lang):
    return GERMAN_FILLER_PATTERNS if lang == "de" else ENGLISH_FILLER_PATTERNS


def stopwords_for(lang):
    return GERMAN_STOPWORDS if lang == "de" else ENGLISH_STOPWORDS


def count_fillers(text, lang):
    """Return {pattern_label: count} for the given language's precompiled filler list."""
    lowered = (text or "").lower()
    compiled = _COMPILED_FILLERS["de" if lang == "de" else "en"]
    return {k: len(rx.findall(lowered)) for k, rx in compiled.items()}


def filter_entries_by_days(entries, days, now=None):
    """Rolling- or calendar-window filter over analytics entries.

    days is None -> everything; 0 -> the current calendar day; N -> the
    trailing N days; string specs ('ytd', 'prev_6m', 'Year 2025', 'year:2025',
    'YYYY-MM-DD..YYYY-MM-DD') -> inclusive date range. Returns matching indices.
    """
    if days is None:
        return list(range(len(entries)))
    now = now or datetime.datetime.now()
    if days == 0:
        today = now.date()
        return [i for i, e in enumerate(entries) if e["timestamp"].date() == today]
    if isinstance(days, int) and days > 0:
        cutoff = now - datetime.timedelta(days=days)
        return [i for i, e in enumerate(entries) if e["timestamp"] >= cutoff]
    if lexical_insights is not None:
        first_d = entries[0]["timestamp"].date() if entries else None
        start_d, end_d = lexical_insights.resolve_date_bounds(
            days, today=now.date(), first_archive_date=first_d
        )
        return [
            i for i, e in enumerate(entries)
            if (start_d is None or e["timestamp"].date() >= start_d)
            and (end_d is None or e["timestamp"].date() <= end_d)
        ]
    return list(range(len(entries)))


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
        self._live_chunk_queue = queue.Queue()
        self._live_session_texts = []
        self._live_reset_event = threading.Event()

        # Transcription backend (set by load_model on its worker thread)
        self.backend = None
        self.current_backend_name = None
        self.current_device = None
        self._model_loading = True
        self._model_load_failed = False

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
        self._lexical_save_job = None
        self._lexical_index = None
        self._current_lexical_slice = None
        self._pending_acoustic_metas = deque(maxlen=64)
        self._acoustic_meta_lock = threading.Lock()
        self._trends_explorer_win = None
        self._ai_insights_mem_cache = None
        self._ui_render_cache = {}
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
        self.load_recent_transcriptions()  # Load recent history (+ seed analytics cache)
        # Defer full tabbed analytics build/paint slightly so the first window frame
        # appears immediately even when Speech Insights was left expanded.
        if not self.insights_collapsed.get():
            self._schedule_insights_refresh(delay_ms=25)
        else:
            self._insights_dirty = True
        # Refresh stale AI insights once per app start, well after the UI settles
        self._hotkey_pump_job = None
        self._gui_loop_job = None
        self._hotkey_watchdog_job = None
        self._init_histogram_job = None
        self._welcome_job = None
        self._auto_crunch_job = self.root.after(30000, self._maybe_auto_crunch_ai_insights)
        self.update_transcription_display()  # Show initial display
        self.root.attributes('-topmost', self.config['always_on_top'])
        
        # Threads
        threading.Thread(target=self.processing_loop, daemon=True).start()
        threading.Thread(target=self._live_transcribe_loop, daemon=True).start()
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
        self._hotkey_pump_job = self.root.after(50, self._hotkey_pump)

        # Window focus handler - refresh hotkeys when app regains focus
        self.root.bind("<FocusIn>", self._on_focus_in)
        
        # Minimization Binding
        self.root.bind("<Unmap>", self.on_minimize)
        self.root.bind("<Map>", self.on_restore)
        
        # UI Update Loop
        self._gui_loop_job = self.root.after(UI_RUNTIME["initial_gui_loop_delay_ms"], self.update_gui_loop)
        
        # Delayed histogram update (after canvas is fully sized)
        self._init_histogram_job = self.root.after(UI_RUNTIME["initial_histogram_delay_ms"], self.update_histogram)

        # Orderly shutdown when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # First-run welcome. Scheduled on the Tk loop (never on the model
        # thread started above) and shown at most once per install.
        self._welcome_job = self.root.after(700, self._maybe_show_welcome)

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

    def _get_monitor_workarea(self, rx, ry):
        """Return (left, top, right, bottom) of the monitor containing (rx, ry),
        supporting multi-monitor negative/secondary coordinates on Windows."""
        if IS_WINDOWS:
            try:
                import ctypes
                from ctypes import wintypes

                class _MONITORINFO(ctypes.Structure):
                    _fields_ = [
                        ("cbSize", wintypes.DWORD),
                        ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD),
                    ]

                pt = wintypes.POINT(int(rx), int(ry))
                hmon = ctypes.windll.user32.MonitorFromPoint(pt, 2)
                if hmon:
                    mi = _MONITORINFO()
                    mi.cbSize = ctypes.sizeof(_MONITORINFO)
                    if ctypes.windll.user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                        r = mi.rcWork
                        return int(r.left), int(r.top), int(r.right), int(r.bottom)
            except Exception:
                pass
        try:
            sw = int(self.root.winfo_screenwidth())
            sh = int(self.root.winfo_screenheight())
            if 0 <= rx < sw and 0 <= ry < sh:
                return 0, 0, sw, sh
        except Exception:
            pass
        return None

    def _position_tooltip_window(self, tip, anchor_x, avoid_y_min, avoid_y_max, ref_x=None, ref_y=None):
        """Place tooltip cleanly below avoid_y_max (or above avoid_y_min if near
        screen bottom) without ever covering the cursor or spilling off-monitor."""
        try:
            tip.update_idletasks()
            tip_w = max(1, tip.winfo_reqwidth())
            tip_h = max(1, tip.winfo_reqheight())
        except Exception:
            tip_w, tip_h = 220, 28

        x = int(anchor_x)
        y = int(avoid_y_max) + 6
        px = x if ref_x is None else int(ref_x)
        py = int(avoid_y_max) if ref_y is None else int(ref_y)
        bounds = self._get_monitor_workarea(px, py)
        if bounds is not None:
            ml, mt, mr, mb = bounds
            if x + tip_w > mr - 6:
                x = max(ml + 6, mr - tip_w - 6)
            if x < ml + 6:
                x = ml + 6
            if y + tip_h > mb - 6:
                y = max(mt + 6, int(avoid_y_min) - tip_h - 6)
        tip.wm_geometry(f"+{int(x)}+{int(y)}")

    def _create_tooltip(self, widget, text):
        """Create a topmost dark-theme hover tooltip for ``widget``.

        ``text`` may be either a string or a zero-arg callable returning a string.
        """
        if not hasattr(self, "_active_tooltips"):
            self._active_tooltips = set()
        tooltip = None

        def hide_tooltip(event=None):
            nonlocal tooltip
            if tooltip is not None:
                self._active_tooltips.discard(tooltip)
                try:
                    tooltip.destroy()
                except Exception:
                    pass
                tooltip = None

        def show_tooltip(event=None):
            nonlocal tooltip
            if getattr(self, "_shutting_down", False):
                return
            hide_tooltip()
            try:
                resolved = text() if callable(text) else text
            except Exception:
                resolved = ""
            if not resolved:
                return
            try:
                wx = widget.winfo_rootx()
                wy = widget.winfo_rooty()
                wh = widget.winfo_height()
            except Exception:
                return

            tooltip = tk.Toplevel(self.root)
            self._active_tooltips.add(tooltip)
            tooltip.wm_overrideredirect(True)
            try:
                tooltip.wm_attributes("-topmost", True)
            except Exception:
                pass
            tooltip.configure(bg=COLOR_BORDER)

            label = tk.Label(
                tooltip,
                text=str(resolved),
                justify="left",
                bg=COLOR_ACCENT_BG,
                fg=COLOR_FG,
                font=("Arial", 9),
                padx=8,
                pady=5,
            )
            label.pack(padx=1, pady=1)
            self._position_tooltip_window(
                tooltip,
                anchor_x=wx + 8,
                avoid_y_min=wy,
                avoid_y_max=wy + wh,
                ref_x=wx + 8,
                ref_y=wy + wh // 2,
            )

        widget._nw_hide_tooltip = hide_tooltip
        widget.bind("<Enter>", show_tooltip, add="+")
        widget.bind("<Leave>", hide_tooltip, add="+")
        widget.bind("<ButtonPress>", hide_tooltip, add="+")
        widget.bind("<Unmap>", hide_tooltip, add="+")
        widget.bind("<Destroy>", hide_tooltip, add="+")

    def _enhance_combobox(self, combo, hint_text=None):
        """Add three UX upgrades to a ``ttk.Combobox``:
        1. Full-line hover tooltip on the closed combobox (selected value + hint).
        2. Live per-item full-line hover tooltip inside the open dropdown list.
        3. Auto-widened dropdown popup (via native ``postoffset``) plus a guard
           preventing mouse-wheel page scrolling from mutating closed comboboxes.
        """
        import tkinter.font as tkfont

        if not hasattr(self, "_active_tooltips"):
            self._active_tooltips = set()
        self._combo_style_seq = getattr(self, "_combo_style_seq", 0) + 1
        style_name = f"NWCombo{self._combo_style_seq}.TCombobox"
        try:
            combo.configure(style=style_name)
        except Exception:
            style_name = None

        def _closed_tooltip_text():
            try:
                current = (combo.get() or "").strip()
            except Exception:
                current = ""
            if current and hint_text:
                return f"{current}\n\n{hint_text}"
            return current or (hint_text or "")

        self._create_tooltip(combo, _closed_tooltip_text)

        item_tip = None
        item_label = None
        last_idx = None
        popdown_bound = False
        last_postoffset = (0, 0, 0, 0)

        def _hide_item_tip(*_args):
            nonlocal item_tip, item_label, last_idx
            last_idx = None
            if item_tip is not None:
                self._active_tooltips.discard(item_tip)
                try:
                    item_tip.destroy()
                except Exception:
                    pass
                item_tip = None
                item_label = None

        def _show_item_tip(full_text, anchor_x, row_top_y, row_bot_y, ref_x, ref_y):
            nonlocal item_tip, item_label
            if getattr(self, "_shutting_down", False) or not full_text:
                _hide_item_tip()
                return
            if item_tip is None or not item_tip.winfo_exists():
                item_tip = tk.Toplevel(self.root)
                self._active_tooltips.add(item_tip)
                item_tip.wm_overrideredirect(True)
                try:
                    item_tip.wm_attributes("-topmost", True)
                except Exception:
                    pass
                item_tip.configure(bg=COLOR_TEAL)
                item_label = tk.Label(
                    item_tip,
                    text=full_text,
                    justify="left",
                    bg=COLOR_BG,
                    fg=COLOR_FG,
                    font=("Arial", 9),
                    padx=9,
                    pady=4,
                )
                item_label.pack(padx=1, pady=1)
            else:
                item_label.configure(text=full_text)
            self._position_tooltip_window(
                item_tip,
                anchor_x=anchor_x,
                avoid_y_min=row_top_y,
                avoid_y_max=row_bot_y,
                ref_x=ref_x,
                ref_y=ref_y,
            )

        def _ensure_popdown_bound():
            nonlocal popdown_bound
            if popdown_bound:
                return
            try:
                popdown = combo.tk.call("ttk::combobox::PopdownWindow", combo._w)
                lb_path = f"{popdown}.f.l"
            except Exception:
                return

            def _parse_bbox(raw_bbox):
                if not raw_bbox:
                    return None
                if isinstance(raw_bbox, str):
                    parts = raw_bbox.split()
                    return [int(v) for v in parts] if len(parts) == 4 else None
                if len(raw_bbox) == 4:
                    return [int(v) for v in raw_bbox]
                return None

            def _on_lb_motion(x_s, y_s, rx_s, ry_s):
                nonlocal last_idx
                try:
                    y = int(float(y_s))
                    rx = int(float(rx_s))
                    ry = int(float(ry_s))
                    size = int(combo.tk.call(lb_path, "size"))
                    if size <= 0:
                        _hide_item_tip()
                        return
                    idx = int(combo.tk.call(lb_path, "nearest", y))
                    bbox = _parse_bbox(combo.tk.call(lb_path, "bbox", idx))
                    if not bbox:
                        _hide_item_tip()
                        return
                    _bx, by, _bw, bh = bbox
                    if y < by - 2 or y > by + bh + 2:
                        _hide_item_tip()
                        return
                    full_line = str(combo.tk.call(lb_path, "get", idx))
                    lb_rooty = int(combo.tk.call("winfo", "rooty", lb_path))
                    row_top_y = lb_rooty + by
                    row_bot_y = row_top_y + bh
                    last_idx = idx
                    _show_item_tip(full_line, rx + 14, row_top_y, row_bot_y, rx, ry)
                except Exception:
                    _hide_item_tip()

            def _on_lb_key(*_a):
                nonlocal last_idx
                try:
                    sel = combo.tk.call(lb_path, "curselection")
                    if not sel:
                        return
                    idx = int(sel[0] if not isinstance(sel, str) else sel.split()[0])
                    bbox = _parse_bbox(combo.tk.call(lb_path, "bbox", idx))
                    if not bbox:
                        return
                    _bx, by, _bw, bh = bbox
                    full_line = str(combo.tk.call(lb_path, "get", idx))
                    lb_rootx = int(combo.tk.call("winfo", "rootx", lb_path))
                    lb_rooty = int(combo.tk.call("winfo", "rooty", lb_path))
                    last_idx = idx
                    _show_item_tip(
                        full_line,
                        lb_rootx + 12,
                        lb_rooty + by,
                        lb_rooty + by + bh,
                        lb_rootx + 12,
                        lb_rooty + by + bh // 2,
                    )
                except Exception:
                    pass

            try:
                motion_cmd = combo.register(_on_lb_motion)
                key_cmd = combo.register(_on_lb_key)
                hide_cmd = combo.register(_hide_item_tip)
                combo._nw_popdown_callbacks = (
                    _on_lb_motion,
                    _on_lb_key,
                    _hide_item_tip,
                    motion_cmd,
                    key_cmd,
                    hide_cmd,
                )
                combo.tk.call("bind", lb_path, "<Motion>", f"+{motion_cmd} %x %y %X %Y")
                combo.tk.call("bind", lb_path, "<KeyRelease-Up>", f"+{key_cmd}")
                combo.tk.call("bind", lb_path, "<KeyRelease-Down>", f"+{key_cmd}")
                combo.tk.call("bind", lb_path, "<Leave>", f"+{hide_cmd}")
                combo.tk.call("bind", popdown, "<Unmap>", f"+{hide_cmd}")
                popdown_bound = True
            except Exception:
                pass

        def _on_post():
            nonlocal last_postoffset
            if hasattr(combo, "_nw_hide_tooltip"):
                combo._nw_hide_tooltip()
            _ensure_popdown_bound()
            if not style_name:
                return
            try:
                vals = list(combo["values"] or ())
                if not vals:
                    return
                try:
                    fnt = tkfont.nametofont(str(combo.cget("font") or "TkTextFont"))
                except Exception:
                    fnt = tkfont.nametofont("TkTextFont")
                max_px = max(fnt.measure(str(v)) for v in vals) + 34
                cb_w = max(1, combo.winfo_width())
                cb_rx = combo.winfo_rootx()
                cb_ry = combo.winfo_rooty()
                bounds = self._get_monitor_workarea(cb_rx, cb_ry)
                if bounds is not None:
                    ml, _mt, mr, _mb = bounds
                    max_px = min(max_px, max(cb_w, (mr - ml) - 20))
                extra_w = max(0, int(max_px - cb_w))
                dx = 0
                if extra_w > 0 and bounds is not None:
                    ml, _mt, mr, _mb = bounds
                    overflow_r = (cb_rx + cb_w + extra_w) - (mr - 8)
                    if overflow_r > 0:
                        dx = -min(overflow_r, max(0, cb_rx - (ml + 8)))
                new_offset = (int(dx), 0, int(extra_w), 0)
                if new_offset != last_postoffset:
                    last_postoffset = new_offset
                    self.style.configure(style_name, postoffset=new_offset)
                    combo.update_idletasks()
            except Exception:
                pass

        try:
            combo.configure(postcommand=_on_post)
        except Exception:
            pass

        def _on_closed_combo_wheel(event):
            try:
                popdown = combo.tk.call("ttk::combobox::PopdownWindow", combo._w)
                if int(combo.tk.call("winfo", "ismapped", popdown)):
                    return None
            except Exception:
                pass
            try:
                if combo.winfo_toplevel() is self.root:
                    self._on_mousewheel(event, force_canvas=True)
            except Exception:
                pass
            return "break"

        combo.bind("<MouseWheel>", _on_closed_combo_wheel)
        if IS_LINUX:
            combo.bind("<Button-4>", _on_closed_combo_wheel)
            combo.bind("<Button-5>", _on_closed_combo_wheel)
        combo.bind("<<ComboboxSelected>>", lambda _e: _hide_item_tip(), add="+")
        combo.bind("<Destroy>", lambda _e: _hide_item_tip(), add="+")
        combo._nw_hide_item_tip = _hide_item_tip

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
        cfg_path = os.path.join(app_dir, CONFIG_FILE)
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
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
                self._quarantine_corrupt_file(cfg_path, e)
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

        # Normalize legacy/renamed model_key values so upgraded configs resolve cleanly.
        raw_model_key = config.get("model_key", "")
        if raw_model_key not in MODEL_MAP:
            alias_id = LEGACY_MODEL_ALIASES.get(raw_model_key, raw_model_key)
            if alias_id == "parakeet-tdt-0.6b-v2" and alias_id not in MODEL_MAP_REVERSE:
                alias_id = "parakeet-tdt-0.6b-v3"
            mapped_key = alias_id if alias_id in MODEL_MAP else MODEL_MAP_REVERSE.get(alias_id)
            config["model_key"] = mapped_key if mapped_key in MODEL_MAP else DEFAULT_CONFIG["model_key"]

        # Normalize legacy 'backend': 'cuda' configs to ('faster-whisper', 'cuda').
        if config.get("backend") == "cuda":
            config["backend"] = "faster-whisper"
            if config.get("device") in ("auto", "", None):
                config["device"] = "cuda"

        if needs_rewrite:
            self._persist_config(config)

        # Keep 'backend', 'device', and 'compute_type' as 'auto' during load_config():
        # hardware probing (ctranslate2 / onnxruntime / openvino) runs off the Tk main
        # thread inside _load_model_locked(), which makes cold window startup instant
        # and avoids clobbering 'auto' to 'cpu' before OpenVINO/ONNX backends are checked.
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
        legacy_stats_path = os.path.join(app_dir, STATS_FILE)
        if os.path.exists(legacy_stats_path):
            try:
                with open(legacy_stats_path, 'r', encoding='utf-8') as f:
                    legacy = json.load(f)
                claimer = legacy.get('claimed_by')
                if claimer is None or claimer == self._hostname():
                    if claimer is None:
                        legacy['claimed_by'] = self._hostname()
                        self._write_json_atomic(legacy_stats_path, legacy)
                    data = {k: v for k, v in legacy.items() if k != 'claimed_by'}
                    data.setdefault('hourly_data', {})
                    # Persist the baseline immediately so a crash before the
                    # first save_stats() can't lose it
                    self._write_json_atomic(stats_path, data)
                    log.info("[Stats] Claimed legacy stats file as baseline for %s", self._hostname())
                    return data
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._quarantine_corrupt_file(legacy_stats_path, e)
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
    MODEL_DOWNLOAD_SIZES = MODEL_DOWNLOAD_SIZES

    def is_model_downloaded(self, friendly_name):
        """True only when a real model weight file is present.

        A merely non-empty directory is not enough: an aborted first download
        leaves config/tokenizer JSONs behind, and treating that as "downloaded"
        made the app skip the download and then fail to load, forever.
        """
        candidate_dirs = [self.get_safe_model_path(friendly_name)]
        if self.config.get("backend") == "onnx-asr":
            model_id = MODEL_MAP.get(friendly_name, friendly_name)
            fallback_map = {
                "base": "whisper-base",
                "distil-large-v3.5": "nemo-parakeet-tdt-0.6b-v2",
                "small": "nemo-parakeet-tdt-0.6b-v3",
                "medium": "nemo-parakeet-tdt-0.6b-v3",
                "large-v3": "nemo-parakeet-tdt-0.6b-v3",
                "large-v3-turbo": "nemo-parakeet-tdt-0.6b-v3",
            }
            if model_id in fallback_map:
                candidate_dirs = [
                    os.path.join(app_dir, "models", f"onnx_{fallback_map[model_id]}")
                ]

        for target_dir in candidate_dirs:
            if not os.path.isdir(target_dir):
                continue
            for root, _dirs, files in os.walk(target_dir):
                # Prune hidden metadata dirs (.cache, .locks) left by huggingface_hub
                _dirs[:] = [d for d in _dirs if not d.startswith(".")]
                if "model.bin" in files:
                    return True
                if any(name.endswith(".nemo") for name in files):
                    return True
                xml_stems = [os.path.splitext(name)[0] for name in files if name.endswith(".xml")]
                if xml_stems:
                    file_set = set(files)
                    if all(
                        f"{stem}.bin" in file_set
                        and os.path.getsize(os.path.join(root, f"{stem}.bin")) > 0
                        for stem in xml_stems
                    ):
                        return True
                if any(name.endswith((".onnx", ".ort")) for name in files) and (
                    "vocab.txt" in files or "config.json" in files
                ):
                    return True
        return False

    def _local_language_arg(self):
        """Return configured local transcription language code, or None for auto-detect."""
        lang = str(self.config.get("local_language", "auto") or "auto").strip().lower()
        return None if (not lang or lang == "auto") else lang

    @staticmethod
    def _temp_audio_path(tag):
        """Collision-free scratch WAV path in the OS temp dir (legacy fallback only)."""
        import tempfile
        return os.path.join(
            tempfile.gettempdir(),
            f"neurowhisper_{tag}_{os.getpid()}_{next(_TEMP_FILE_COUNTER)}.wav",
        )

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

    def _on_canvas_configure(self, event):
        """Sync content frame width and re-clamp scroll bounds when window resizes."""
        try:
            self._scroll_canvas.itemconfig(self._scroll_canvas_window, width=event.width)
        except Exception:
            pass
        self._update_scroll_region()

    def _on_scrollbar_yview(self, *args):
        """Scrollbar command wrapper that keeps canvasy(0) strictly within [0, max_y]."""
        try:
            req_h = self.content_frame.winfo_reqheight()
            avail_h = max(1, self._scroll_canvas.winfo_height())
            if req_h <= avail_h:
                self._scroll_canvas.yview_moveto(0.0)
                return
            self._scroll_canvas.yview(*args)
            self._clamp_scroll_position(req_h, avail_h)
        except Exception:
            pass

    def _clamp_scroll_position(self, req_h=None, avail_h=None):
        """Ensure the content frame never scrolls into negative space (top gap)
        or past its bottom edge."""
        if req_h is None:
            req_h = self.content_frame.winfo_reqheight()
        if avail_h is None:
            avail_h = max(1, self._scroll_canvas.winfo_height())
        region_h = max(req_h, avail_h, 1)
        cur_y = float(self._scroll_canvas.canvasy(0))
        if req_h <= avail_h:
            if abs(cur_y) > 0.5:
                self._scroll_canvas.yview_moveto(0.0)
            return
        max_y = float(req_h - avail_h)
        clamped_y = max(0.0, min(max_y, cur_y))
        if abs(clamped_y - cur_y) > 0.5:
            self._scroll_canvas.yview_moveto(clamped_y / float(region_h))

    def _schedule_scroll_region_sync(self):
        """Refresh and clamp scroll bounds immediately and once layout settles."""
        if getattr(self, "_shutting_down", False):
            return
        try:
            self.content_frame.update_idletasks()
            self._update_scroll_region()
        except Exception:
            pass
        prev = getattr(self, "_scroll_sync_job", None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except Exception:
                pass

        def _settled():
            self._scroll_sync_job = None
            if not getattr(self, "_shutting_down", False):
                self._update_scroll_region()

        self._scroll_sync_job = self.root.after(50, _settled)

    def _on_mousewheel(self, event, force_canvas=False):
        """Handle mousewheel/touchpad scrolling with strict [0, max_y] clamping."""
        if not force_canvas:
            widget = getattr(event, "widget", None)
            # Internal Tcl popup widgets (such as open ttk.Combobox popdown listboxes)
            # arrive as raw str paths: let them scroll their own list without moving root.
            if isinstance(widget, str):
                return
            if isinstance(widget, (tk.Text, scrolledtext.ScrolledText, tk.Listbox)):
                return
            if widget is not None:
                try:
                    if widget.winfo_toplevel() is not self.root:
                        return
                except Exception:
                    return

        try:
            req_h = self.content_frame.winfo_reqheight()
            avail_h = max(1, self._scroll_canvas.winfo_height())
            if req_h <= avail_h:
                if abs(float(self._scroll_canvas.canvasy(0))) > 0.5:
                    self._scroll_canvas.yview_moveto(0.0)
                return

            num = getattr(event, "num", None)
            delta = getattr(event, "delta", 0)
            if num == 4:
                step_px = -48.0
            elif num == 5:
                step_px = 48.0
            elif delta:
                # Windows/X11 wheel notch = ±120 -> 48 px; Precision Touchpad
                # sends smaller smooth deltas (e.g. 15..60); macOS sends small ints.
                if IS_MAC and abs(delta) < 20:
                    step_px = float(-delta) * 14.0
                else:
                    step_px = -(float(delta) / 120.0) * 48.0
                    if 0.0 < abs(step_px) < 4.0:
                        step_px = -4.0 if step_px < 0 else 4.0
            else:
                return

            region_h = float(max(req_h, avail_h))
            cur_y = float(self._scroll_canvas.canvasy(0))
            max_y = float(req_h - avail_h)
            new_y = max(0.0, min(max_y, cur_y + step_px))
            if abs(new_y - cur_y) >= 0.5:
                self._scroll_canvas.yview_moveto(new_y / region_h)
        except Exception:
            pass

    def _update_scroll_region(self, event=None):
        """Update the scroll region when content or viewport changes size,
        clamp scrollregion height >= viewport height so content can never drift
        into negative space, and show the scrollbar only when content overflows."""
        try:
            req_w = max(1, self._scroll_canvas.winfo_width())
            req_h = self.content_frame.winfo_reqheight()
            avail_h = max(1, self._scroll_canvas.winfo_height())
            region_h = max(req_h, avail_h)
            self._scroll_canvas.configure(scrollregion=(0, 0, req_w, region_h))
            self._clamp_scroll_position(req_h, avail_h)

            mapped = bool(self._scroll_vsb.winfo_ismapped())
            # Hysteresis: hiding the bar widens the canvas, which can re-wrap
            # content and change reqheight, so the hide threshold sits below
            # the show threshold to stop the two from oscillating.
            if not mapped and req_h > avail_h + 2:
                self._scroll_vsb.pack(side="right", fill="y",
                                      before=self._scroll_canvas)
            elif mapped and req_h <= avail_h - 18:
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

        live_be = self.current_backend_name or self.config.get('backend', 'auto')
        live_dev = self.current_device or self.config.get('device', 'auto')
        if hasattr(self, "backend_status_label"):
            try:
                self.backend_status_label.config(text=f"[{live_be} / {live_dev}]")
            except Exception:
                pass

        if self.transcription_mode.get() == 'online':
            model = self.config.get('openai_transcription_model',
                                    DEFAULT_CONFIG['openai_transcription_model'])
            self.model_info_var.set(f"{model} · cloud")
        else:
            model_key = self.config.get('model_key', '')
            short = model_key.split('(')[0].strip() or model_key
            compute = self.config.get('compute_type', 'auto')
            if compute == 'auto' and live_dev != 'auto':
                compute = 'float16' if str(live_dev).lower() == 'cuda' else 'int8'
            self.model_info_var.set(f"{short} · {live_dev} · {compute}")
        self._update_config_summary()

    def _config_summary_text(self):
        model_key = self.config.get('model_key', '')
        short = model_key.split('(')[0].strip() or model_key
        device = self.current_device or self.config.get('device', 'auto')
        try:
            pause = float(self.live_pause_var.get())
        except Exception:
            pause = float(self.config.get('live_pause', 0.8))
        return f"{short} · {device} · {pause:.1f}s"

    def _update_config_summary(self):
        if hasattr(self, "config_summary_var"):
            self.config_summary_var.set(self._config_summary_text())

    def _on_local_lang_change(self, event=None):
        if not hasattr(self, "local_lang_var"):
            return
        self.config['local_language'] = (self.local_lang_var.get() or "auto").strip().lower()
        self._persist_config()
        self.log_internal(f"Local language set to: {self.config['local_language']}")

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
        self._scroll_canvas = tk.Canvas(
            self.root, bg=COLOR_BG, highlightthickness=0, yscrollincrement=1
        )
        self._scroll_vsb = ttk.Scrollbar(
            self.root, orient="vertical", command=self._on_scrollbar_yview
        )
        self.content_frame = tk.Frame(self._scroll_canvas, bg=COLOR_BG)

        self.content_frame.bind("<Configure>", self._update_scroll_region)
        self._scroll_canvas_window = self._scroll_canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )
        self._scroll_canvas.configure(yscrollcommand=self._scroll_vsb.set)

        self._scroll_canvas.pack(side="left", fill="both", expand=True)

        # Keep content frame width AND clamped scrollregion in sync with viewport
        self._scroll_canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel scrolling (Windows/macOS <MouseWheel>, Linux X11 <Button-4>/<Button-5>)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        if IS_LINUX:
            self.root.bind_all("<Button-4>", self._on_mousewheel)
            self.root.bind_all("<Button-5>", self._on_mousewheel)

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
        mic_footer_lbl = tk.Label(foot, textvariable=self.device_info_var, bg=COLOR_ACCENT_BG,
                                  fg=COLOR_TEXT_FAINT, font=("Arial", 8))
        mic_footer_lbl.pack(side="left")
        model_footer_lbl = tk.Label(foot, textvariable=self.model_info_var, bg=COLOR_ACCENT_BG,
                                    fg=COLOR_TEXT_FAINT, font=("Arial", 8))
        model_footer_lbl.pack(side="right")
        self._create_tooltip(
            mic_footer_lbl,
            lambda: f"Mic · {self.device_var.get()}"
            if hasattr(self, "device_var") and self.device_var.get()
            else self.device_info_var.get(),
        )
        self._create_tooltip(
            model_footer_lbl,
            lambda: (
                f"{self.config.get('model_key', '')}\n({self.model_info_var.get()})"
                if self.transcription_mode.get() == "local"
                else f"Cloud · {self.model_info_var.get()}"
            ),
        )

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
        insights_top.pack(fill="x", padx=self.CARD_PAD, pady=(11, 4))

        # Row 1: Quick Horizon Chips (1d / 7d / 30d / 90d / 6m / YTD / 1y / All)
        range_value = self.config.get('insights_range', INSIGHTS_RANGE_DEFAULT)
        if not self._is_valid_insights_range(range_value):
            range_value = INSIGHTS_RANGE_DEFAULT
        self.insights_range_var = tk.StringVar(value=range_value)
        range_container = ctk.CTkFrame(insights_top, fg_color=COLOR_BG, corner_radius=9,
                                       border_width=1, border_color=COLOR_BORDER)
        range_container.pack(side="right")
        range_inner = ctk.CTkFrame(range_container, fg_color="transparent")
        range_inner.pack(padx=2, pady=2)
        self.insights_range_buttons = {}
        chip_tooltips = {
            "Today": "Today only (intra-day hourly view)",
            "7 days": "Last 7 days (daily view)",
            "30 days": "Last 30 days (daily + 7d trend)",
            "90 days": "Last 90 days (daily + 14d trend)",
            "6 months": "Last 6 months (trailing 182 days)",
            "This year": "This calendar year (Jan 1 – Today)",
            "12 months": "Last 12 months (trailing 365 days)",
            "All": "Full multi-year transcript archive",
        }
        for key, short in INSIGHTS_QUICK_CHIPS:
            w_chip = max(28, 7 * len(short) + 12)
            btn = ctk.CTkButton(range_inner, text=short, width=w_chip, height=19,
                                corner_radius=7, font=ctk.CTkFont(size=10, weight="bold"),
                                fg_color=COLOR_BG, hover_color=COLOR_TEAL_DIM,
                                text_color=COLOR_TEXT_FAINT,
                                command=lambda k=key: self._select_insights_range(k))
            btn.pack(side="left", padx=1)
            self.insights_range_buttons[key] = btn
            if key in chip_tooltips:
                self._create_tooltip(btn, chip_tooltips[key])

        tk.Label(insights_top, text="Horizon", bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_MUTED,
                 font=("Arial", 9)).pack(side="left")

        # Row 2: Custom Period/Year + Bucket Granularity + Language Slice + Wide Explorer
        insights_subbar = tk.Frame(self.insights_body, bg=COLOR_ACCENT_BG)
        insights_subbar.pack(fill="x", padx=self.CARD_PAD, pady=(0, 8))

        tk.Label(insights_subbar, text="Slice:", bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_FAINT,
                 font=("Arial", 8)).pack(side="left", padx=(0, 3))
        period_values = list(INSIGHTS_RANGE_OPTIONS.keys())
        if range_value not in period_values:
            period_values.append(range_value)
        self.insights_period_combo = ttk.Combobox(
            insights_subbar, textvariable=self.insights_range_var,
            values=period_values, state="readonly", width=10,
        )
        self.insights_period_combo.pack(side="left", padx=(0, 4))
        self.insights_period_combo.bind("<<ComboboxSelected>>", self._on_insights_period_combo_select)
        self._enhance_combobox(
            self.insights_period_combo,
            "Time slice (includes calendar years once archive is scanned)",
        )

        bucket_val = self.config.get('insights_bucket', 'Auto')
        if bucket_val not in INSIGHTS_BUCKET_OPTIONS:
            bucket_val = 'Auto'
        self.insights_bucket_var = tk.StringVar(value=bucket_val)
        self.insights_bucket_combo = ttk.Combobox(
            insights_subbar, textvariable=self.insights_bucket_var,
            values=list(INSIGHTS_BUCKET_OPTIONS), state="readonly", width=6,
        )
        self.insights_bucket_combo.pack(side="left", padx=(0, 4))
        self.insights_bucket_combo.bind("<<ComboboxSelected>>", self._on_insights_filter_change)
        self._enhance_combobox(
            self.insights_bucket_combo,
            "Time bucket resolution (Auto adapts Daily / Weekly / Monthly)",
        )

        lang_val = self.config.get('insights_lang', 'All')
        if lang_val not in INSIGHTS_LANG_OPTIONS:
            lang_val = 'All'
        self.insights_lang_var = tk.StringVar(value=lang_val)
        self.insights_lang_combo = ttk.Combobox(
            insights_subbar, textvariable=self.insights_lang_var,
            values=list(INSIGHTS_LANG_OPTIONS), state="readonly", width=4,
        )
        self.insights_lang_combo.pack(side="left", padx=(0, 4))
        self.insights_lang_combo.bind("<<ComboboxSelected>>", self._on_insights_filter_change)
        self._enhance_combobox(
            self.insights_lang_combo,
            "Language filter: All (EN+DE normalized), EN only, or DE only",
        )

        self.insights_trim_outliers_var = tk.BooleanVar(
            value=bool(self.config.get("insights_trim_outliers", False))
        )

        self.btn_expand_insights = ctk.CTkButton(
            insights_subbar, text="⤢ Expand", width=66, height=22,
            corner_radius=7, font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=COLOR_BG, hover_color=COLOR_TEAL_DIM,
            border_width=1, border_color=COLOR_TEAL,
            text_color=COLOR_TEAL,
            command=self.open_trends_explorer,
        )
        self.btn_expand_insights.pack(side="right")
        self._create_tooltip(
            self.btn_expand_insights,
            "Open wide resizable Longitudinal Trends Explorer & HTML Report export",
        )

        self.btn_trim_outliers = ctk.CTkButton(
            insights_subbar, text="✂ >2.5σ", width=68, height=22,
            corner_radius=7, font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=COLOR_BG, hover_color=COLOR_TEAL_DIM,
            border_width=1, border_color=COLOR_BORDER,
            text_color=COLOR_TEXT_MUTED,
            command=self._toggle_trim_outliers,
        )
        self.btn_trim_outliers.pack(side="right", padx=(0, 4))
        self._create_tooltip(
            self.btn_trim_outliers,
            "Kick out >2.5σ outliers (non-destructive):\n"
            "Excludes accidental open-mic pushes, extreme stall spikes,\n"
            "and runaway utterances from charts & averages without\n"
            "modifying any transcript files on disk.",
        )

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
            value=f"Baseline: your trailing {PERSONAL_BASELINE_DAYS}-day median & MAD band "
                  f"(non-diagnostic personal self-tracking)."
        )
        tk.Label(notes, textvariable=self.insights_peer_var, bg=COLOR_ACCENT_BG,
                 fg=COLOR_TEXT_FAINT, font=("Arial", 8), anchor="w", justify="left",
                 wraplength=420).pack(fill="x")

        self.insights_profile_summary_var = tk.StringVar(
            value="Personal profile vs baseline: waiting for enough data."
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

        # Tab chips (12 tabs in a balanced 3x4 grid)
        self.insight_tab_names = [
            "FogWatch",
            "Pace",
            "Pauses",
            "Clarity",
            "Lexical",
            "Mind",
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
            chip = ctk.CTkButton(chip_line, text=name, height=22, corner_radius=8,
                                 width=max(50, 7 * len(name) + 16),
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

        self.fogwatch_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.pace_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.pause_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.clarity_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.lexical_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.mind_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.intervention_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.ai_timeline_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.ai_mood_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.activity_matrix_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.speaker_profile_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)
        self.daily_brief_tab = tk.Frame(self.insights_tab_container, bg=COLOR_ACCENT_BG)

        self.insight_tab_frames = [
            self.fogwatch_tab, self.pace_tab, self.pause_tab, self.clarity_tab,
            self.lexical_tab, self.mind_tab, self.intervention_tab,
            self.ai_timeline_tab, self.ai_mood_tab, self.activity_matrix_tab,
            self.speaker_profile_tab, self.daily_brief_tab,
        ]

        # Tab bodies are built lazily on first selection
        self._insight_tab_builders = [
            self._setup_fogwatch_tab,
            self._setup_pace_tab,
            self._setup_pause_tab,
            self._setup_clarity_tab,
            self._setup_lexical_tab,
            self._setup_mind_tab,
            self._setup_intervention_tab,
            self._setup_ai_timeline_tab,
            self._setup_ai_mood_tab,
            self._setup_activity_matrix_tab,
            self._setup_speaker_profile_tab,
            self._setup_daily_brief_tab,
        ]
        self._insight_tabs_built = set()
        self.insights_tab_status = tk.StringVar(value="1/12")
        self.insight_tab_index = 0
        self.insight_tab_frames[0].pack(fill="x")
        self._update_chip_styles()
        self._update_range_button_styles()
        self._update_insight_tab_status()

        if not self.insights_collapsed.get():
            self.insights_body.pack(fill="x")
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
        self._enhance_combobox(
            self.combo,
            "Local speech-to-text model ('[get]' = downloads once on first selection, then runs 100% offline)",
        )

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

        ttk.Label(r1b, text="Lang:").pack(side="left", padx=(6, 0))
        self.local_lang_var = tk.StringVar(value=self.config.get('local_language', DEFAULT_CONFIG['local_language']))
        local_lang_combo = ttk.Combobox(
            r1b,
            textvariable=self.local_lang_var,
            values=["auto", "en", "de", "es", "fr", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"],
            width=6,
            state="readonly",
        )
        local_lang_combo.pack(side="left", padx=4)
        local_lang_combo.bind("<<ComboboxSelected>>", self._on_local_lang_change)
        self._enhance_combobox(
            local_lang_combo,
            "Spoken language for local dictation ('auto' detects automatically;\npinning 'en' or 'de' avoids short-chunk language flips).",
        )

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
        self.mic_combo = mic_combo
        mic_combo.pack(side="left", fill="x", expand=True, padx=5)
        mic_combo.bind("<<ComboboxSelected>>", self.on_mic_change)
        self._enhance_combobox(mic_combo, "Active audio input device")

        # --- BACKEND/ACCELERATION SELECTOR ---
        r_backend = ttk.Frame(self.settings_content)
        r_backend.pack(fill="x", padx=5, pady=5)

        ttk.Label(r_backend, text="Acceleration:").pack(side="left")

        # Backend options: Auto detects best, or force specific (backend, device) pair
        backend_options = BACKEND_OPTIONS
        backend_map = BACKEND_DISPLAY_TO_INTERNAL
        self.backend_map_display = {v: k for k, v in backend_map.items()}

        current_be = self.config.get('backend', 'auto')
        current_dev = self.config.get('device', 'auto')
        current_backend_display = resolve_backend_display(current_be, current_dev)

        self.backend_var = tk.StringVar(value=current_backend_display)
        backend_combo = ttk.Combobox(r_backend, textvariable=self.backend_var,
                                      values=backend_options, width=21, state="readonly")
        self.backend_combo = backend_combo
        backend_combo.pack(side="left", padx=5)
        backend_combo.bind("<<ComboboxSelected>>", self.on_backend_change)

        self._enhance_combobox(
            backend_combo,
            "Acceleration Mode:\n"
            "• Auto: Detects best local engine\n"
            "• CUDA: NVIDIA GPUs (Faster-Whisper)\n"
            "• ONNX ASR (CPU/GPU): Lightweight ONNX Runtime (Parakeet TDT v3)\n"
            "• OpenVINO (Intel): Intel GPU/NPU\n"
            "• Parakeet (NVIDIA NeMo): NVIDIA NeMo toolkit\n"
            "• CPU only: Works everywhere",
        )

        # Show current detected backend
        self.backend_status_label = ttk.Label(r_backend, text="", foreground=COLOR_TEAL)
        self.backend_status_label.pack(side="left", padx=6)
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
        self._enhance_combobox(trans_model_combo, "OpenAI speech-to-text model")

        ttk.Label(model_row, text="Lang:").pack(side="left", padx=(8, 0))
        self.openai_lang_var = tk.StringVar(value=self.config.get('openai_language', DEFAULT_CONFIG['openai_language']))
        lang_combo = ttk.Combobox(model_row, textvariable=self.openai_lang_var,
                                  values=["auto", "en", "de", "es", "fr", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"],
                                  width=6, state="readonly")
        lang_combo.pack(side="left", padx=5)
        lang_combo.bind("<<ComboboxSelected>>", lambda e: self._auto_save_openai_settings())
        self._enhance_combobox(lang_combo, "Spoken language hint for OpenAI transcription ('auto' detects automatically)")

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
        self._enhance_combobox(edit_model_combo, "OpenAI model used by +EDIT mode")

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
        prompt_entry.bind("<FocusOut>", lambda e: self._auto_save_openai_settings())
        prompt_entry.bind("<Return>", lambda e: self._auto_save_openai_settings())

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
        "fogwatch": "clarity_index",
        "pace": "pace_avg_words_entry",
        "pause": "pause_long_pct",
        "clarity": "clarity_fillers_per_100",
        "lexical": "mattr_diversity",
        "mind": "vitality_balance",
        "intervention": "intervention_open_question_per_100",
    }

    def _build_metric_tab(self, parent, prefix, label, intro,
                          summary_var, detail_var, peer_var,
                          canvas_height=144, toolbar_builder=None):
        """Shared body for heuristic & longitudinal insight tabs."""
        tk.Label(parent, text=intro, bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_MUTED,
                 font=("Arial", 8), anchor="w", justify="left",
                 wraplength=400).pack(fill="x", padx=10, pady=(8, 6))

        cell = ctk.CTkFrame(parent, fg_color=COLOR_BG, corner_radius=10,
                            border_width=1, border_color=COLOR_BORDER)
        cell.pack(fill="x", padx=10)
        hdr_lbl = tk.Label(cell, text=label.upper(), bg=COLOR_BG, fg=COLOR_TEXT_MUTED,
                           font=("Arial", 8))
        hdr_lbl.pack(anchor="w", padx=10, pady=(8, 0))
        setattr(self, f"{prefix}_header_label", hdr_lbl)
        tk.Label(cell, textvariable=summary_var, bg=COLOR_BG, fg=COLOR_TEAL,
                 font=("Arial", 12, "bold"), anchor="w", justify="left",
                 wraplength=370).pack(fill="x", padx=10)
        tk.Label(cell, textvariable=detail_var, bg=COLOR_BG, fg=COLOR_TEXT_FAINT,
                 font=("Arial", 8), anchor="w", justify="left",
                 wraplength=370).pack(fill="x", padx=10, pady=(0, 8))

        band = tk.Canvas(parent, height=12, bg=COLOR_ACCENT_BG, highlightthickness=0)
        band.pack(fill="x", padx=10, pady=(6, 5))
        setattr(self, f"{prefix}_band_canvas", band)

        if toolbar_builder is not None:
            tb = tk.Frame(parent, bg=COLOR_ACCENT_BG)
            tb.pack(fill="x", padx=10, pady=(0, 4))
            tb_res = toolbar_builder(tb)
            if isinstance(tb_res, tuple) and len(tb_res) >= 2:
                setattr(self, f"{prefix}_pill_buttons", tb_res[0])
                setattr(self, f"{prefix}_sync_pills", tb_res[1])

        canvas = tk.Canvas(parent, height=canvas_height, bg=COLOR_TEXT_BG,
                           highlightthickness=1, highlightbackground=COLOR_BORDER)
        canvas.pack(fill="x", padx=10, pady=(0, 6))

        self._build_insight_guide_box(parent, prefix)

        tk.Label(parent, textvariable=peer_var, bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_FAINT,
                 font=("Arial", 8), anchor="w", justify="left",
                 wraplength=400).pack(fill="x", padx=10, pady=(0, 8))
        return canvas

    def _build_insight_guide_box(self, parent, prefix):
        """Compact zero-LLM dimension explanation + window pattern reading card."""
        box = ctk.CTkFrame(
            parent, fg_color=COLOR_BG, corner_radius=8,
            border_width=1, border_color=COLOR_BORDER,
        )
        box.pack(fill="x", padx=10, pady=(0, 6))
        exp_var = tk.StringVar(value="")
        pat_var = tk.StringVar(value="")
        setattr(self, f"{prefix}_explain_var", exp_var)
        setattr(self, f"{prefix}_pattern_var", pat_var)
        tk.Label(
            box, textvariable=exp_var, bg=COLOR_BG, fg=COLOR_TEXT_MUTED,
            font=("Arial", 8), anchor="w", justify="left", wraplength=386,
        ).pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(
            box, textvariable=pat_var, bg=COLOR_BG, fg="#99f6e4",
            font=("Arial", 8, "bold"), anchor="w", justify="left", wraplength=386,
        ).pack(fill="x", padx=8, pady=(0, 6))
        return box

    def _set_tab_guide(self, prefix, guide):
        """Populate ``{prefix}_explain_var`` and ``{prefix}_pattern_var`` from a guide dict."""
        if not isinstance(guide, dict):
            return
        exp_var = getattr(self, f"{prefix}_explain_var", None)
        pat_var = getattr(self, f"{prefix}_pattern_var", None)
        if exp_var is not None:
            expl = str(guide.get("explanation", "")).strip()
            exp_var.set(f"What it measures: {expl}" if expl else "")
        if pat_var is not None:
            pat = str(guide.get("pattern", "")).strip()
            corr = str(guide.get("correlation", "")).strip()
            full_pat = f"{pat}  {corr}".strip() if corr else pat
            pat_var.set(f"Window pattern: {full_pat}" if full_pat else "")

    def _make_pill_bar(self, parent, options, var, on_change):
        """Render a row of compact CTk pill buttons bound to ``var``."""
        btns = {}

        def _sync_styles():
            cur = var.get()
            for k, b in btns.items():
                if k == cur:
                    b.configure(fg_color=COLOR_TEAL, text_color=COLOR_BG, border_color=COLOR_TEAL)
                else:
                    b.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED, border_color=COLOR_BORDER)

        def _pick(key):
            if var.get() == key:
                return
            var.set(key)
            _sync_styles()
            on_change()

        for key, caption in options:
            w_btn = max(42, 6 * len(caption) + 14)
            btn = ctk.CTkButton(
                parent, text=caption, width=w_btn, height=19, corner_radius=7,
                font=ctk.CTkFont(size=9, weight="bold"),
                fg_color=COLOR_BG, hover_color=COLOR_TEAL_DIM,
                border_width=1, border_color=COLOR_BORDER,
                text_color=COLOR_TEXT_MUTED,
                command=lambda k=key: _pick(k),
            )
            btn.pack(side="left", padx=(0, 4))
            btns[key] = btn

        _sync_styles()
        return btns, _sync_styles

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
        if value is None:
            return
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
            if canvas is None or key not in snapshot or snapshot[key] is None:
                continue
            self._draw_metric_band(canvas, snapshot[key], key)

    def _setup_fogwatch_tab(self):
        self.fogwatch_summary_var = tk.StringVar(value="Scanning cognitive clarity signals...")
        self.fogwatch_detail_var = tk.StringVar(
            value="Personal self-tracking proxy (non-diagnostic): MATTR-50 + specificity + fluency + pause stalls."
        )
        self.fogwatch_peer_var = tk.StringVar(value="Comparison pending.")
        self.fogwatch_metric_var = tk.StringVar(value="clarity_index")
        opts = [
            ("clarity_index", "Clarity Index"),
            ("mattr_diversity", "MATTR-50"),
            ("vague_per_100", "Vague Words"),
            ("pause_long_pct", "Pauses ≥5s"),
        ]
        self.fogwatch_canvas = self._build_metric_tab(
            self.fogwatch_tab,
            "fogwatch",
            "Cognitive Clarity Index",
            "FogWatch: longitudinal self-tracking of lexical sharpness, word-finding load, and mid-burst stalls (non-diagnostic).",
            self.fogwatch_summary_var,
            self.fogwatch_detail_var,
            self.fogwatch_peer_var,
            canvas_height=154,
            toolbar_builder=lambda tb: self._make_pill_bar(
                tb, opts, self.fogwatch_metric_var,
                lambda: self._refresh_fogwatch_tab(self._current_lexical_slice),
            ),
        )

    def _setup_pace_tab(self):
        self.pace_summary_var = tk.StringVar(value="No pace data yet")
        self.pace_detail_var = tk.StringVar(value="Speak a bit and this will populate.")
        self.pace_peer_var = tk.StringVar(value="Comparison pending.")
        self.pace_metric_var = tk.StringVar(value="pace_words_entry")
        opts = [
            ("pace_words_entry", "Words / Entry"),
            ("words_per_sentence", "Words / Sent"),
            ("acoustic_wpm", "Acoustic WPM"),
        ]
        self.pace_canvas = self._build_metric_tab(
            self.pace_tab,
            "pace",
            "Words / entry",
            "Pace & Elaboration: tracks sustained utterance length, sentence length, and local acoustic WPM over time.",
            self.pace_summary_var,
            self.pace_detail_var,
            self.pace_peer_var,
            canvas_height=146,
            toolbar_builder=lambda tb: self._make_pill_bar(
                tb, opts, self.pace_metric_var,
                lambda: self.refresh_insights_tabs(),
            ),
        )

    def _setup_pause_tab(self):
        self.pause_summary_var = tk.StringVar(value="No pause data yet")
        self.pause_detail_var = tk.StringVar(value="Needs consecutive transcribed entries.")
        self.pause_peer_var = tk.StringVar(value="Comparison pending.")
        self.pause_view_var = tk.StringVar(value="pause_long_pct")
        opts = [
            ("pause_long_pct", "Long ≥5s %"),
            ("pause_p90_s", "P90 Tail (s)"),
            ("pause_avg_s", "Avg Pause"),
            ("acoustic_silence_pct", "VAD Silence"),
            ("breakdown", "Breakdown"),
        ]
        self.pause_canvas = self._build_metric_tab(
            self.pause_tab,
            "pause",
            "Long pauses (≥5s)",
            "Pause Profile: longitudinal tail stalls (≥5s share & P90 gap), forward VAD silence, or Short/Med/Long breakdown.",
            self.pause_summary_var,
            self.pause_detail_var,
            self.pause_peer_var,
            canvas_height=146,
            toolbar_builder=lambda tb: self._make_pill_bar(
                tb, opts, self.pause_view_var,
                lambda: self.refresh_insights_tabs(),
            ),
        )

    def _setup_clarity_tab(self):
        self.clarity_summary_var = tk.StringVar(value="No clarity data yet")
        self.clarity_detail_var = tk.StringVar(value="Detected from the full transcript archive.")
        self.clarity_peer_var = tk.StringVar(value="Comparison pending.")
        self.clarity_view_var = tk.StringVar(value="fillers_per_100")
        opts = [
            ("fillers_per_100", "Fillers"),
            ("disfluency_per_1k", "Repairs / 1k"),
            ("vague_per_100", "Vague / 100w"),
            ("breakdown", "Top Fillers"),
        ]
        self.clarity_canvas = self._build_metric_tab(
            self.clarity_tab,
            "clarity",
            "Fillers / 100 w",
            "Fluency & Retrieval: longitudinal filler rate, self-repairs/restarts, vague placeholders, or top filler breakdown.",
            self.clarity_summary_var,
            self.clarity_detail_var,
            self.clarity_peer_var,
            canvas_height=146,
            toolbar_builder=lambda tb: self._make_pill_bar(
                tb, opts, self.clarity_view_var,
                lambda: self.refresh_insights_tabs(),
            ),
        )

    def _setup_lexical_tab(self):
        self.lexical_summary_var = tk.StringVar(value="No lexical data yet")
        self.lexical_detail_var = tk.StringVar(value="Uses transcript vocabulary from all archived sessions.")
        self.lexical_peer_var = tk.StringVar(value="Comparison pending.")
        self.lexical_view_var = tk.StringVar(value="mattr_diversity")
        opts = [
            ("mattr_diversity", "MATTR-50"),
            ("content_density_pct", "Content %"),
            ("lexical_long_word_percent", "7+ Chars %"),
            ("breakdown", "Core Words"),
        ]
        self.lexical_canvas = self._build_metric_tab(
            self.lexical_tab,
            "lexical",
            "MATTR-50 Diversity",
            "Lexical Richness: length-invariant MATTR-50 diversity (no sample-size distortion), content density, and core themes.",
            self.lexical_summary_var,
            self.lexical_detail_var,
            self.lexical_peer_var,
            canvas_height=146,
            toolbar_builder=lambda tb: self._make_pill_bar(
                tb, opts, self.lexical_view_var,
                lambda: self.refresh_insights_tabs(),
            ),
        )

    def _setup_mind_tab(self):
        self.mind_summary_var = tk.StringVar(value="Scanning local well-being & style markers...")
        self.mind_detail_var = tk.StringVar(
            value="100% offline lexical proxies: Vitality vs Strain, 1st-person focus, absolutist words, hedges, late-night %."
        )
        self.mind_peer_var = tk.StringVar(value="Comparison pending.")
        self.mind_metric_var = tk.StringVar(value="vitality_balance")
        opts = [
            ("vitality_balance", "Vitality"),
            ("self_focus_per_100", "I-Talk"),
            ("absolutist_per_100", "Absolutist"),
            ("hedges_per_100", "Hedges"),
            ("late_night_pct", "Night %"),
        ]
        self.mind_canvas = self._build_metric_tab(
            self.mind_tab,
            "mind",
            "Lexical Vitality vs Strain",
            "Mind & Circadian (Local, Non-Diagnostic): tracks subtle shifts in affect words, self-focus, rigidity, and night speech.",
            self.mind_summary_var,
            self.mind_detail_var,
            self.mind_peer_var,
            canvas_height=154,
            toolbar_builder=lambda tb: self._make_pill_bar(
                tb, opts, self.mind_metric_var,
                lambda: self._refresh_mind_tab(self._current_lexical_slice),
            ),
        )

    def _setup_intervention_tab(self):
        self.intervention_summary_var = tk.StringVar(value="No intervention data yet")
        self.intervention_detail_var = tk.StringVar(value="Heuristic only; useful for trends, not diagnosis.")
        self.intervention_peer_var = tk.StringVar(value="Comparison pending.")
        self.intervention_canvas = self._build_metric_tab(
            self.intervention_tab, "intervention", "Open questions / 100 w",
            "Intervention Mix: heuristic counts for open questions, reflections, affirmations, and summaries.",
            self.intervention_summary_var, self.intervention_detail_var,
            self.intervention_peer_var, canvas_height=130)

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

        self.ai_timeline_temporal_var = tk.StringVar(value="Follow Horizon")
        filter_options = ["Follow Horizon", "Last 7 Days", "Last 30 Days", "Previous Month", "All Time"]
        self.ai_timeline_dropdown = tk.OptionMenu(header_frame, self.ai_timeline_temporal_var, *filter_options, command=lambda _: self.refresh_insights_tabs())
        self.ai_timeline_dropdown.config(bg=COLOR_BG, fg=COLOR_FG, font=("Arial", 9), highlightthickness=1)
        self.ai_timeline_dropdown.pack(side="left", padx=8)

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

        self.ai_mood_temporal_var = tk.StringVar(value="Follow Horizon")
        filter_options = ["Follow Horizon", "Last 7 Days", "Last 30 Days", "Previous Month", "All Time"]
        self.ai_mood_dropdown = tk.OptionMenu(header_frame, self.ai_mood_temporal_var, *filter_options, command=lambda _: self.refresh_insights_tabs())
        self.ai_mood_dropdown.config(bg=COLOR_BG, fg=COLOR_FG, font=("Arial", 9), highlightthickness=1)
        self.ai_mood_dropdown.pack(side="left", padx=10)

        self.ai_mood_canvas = tk.Canvas(self.ai_mood_tab, height=140, bg=COLOR_TEXT_BG, highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.ai_mood_canvas.pack(fill="x", padx=10, pady=(10, 6))
        self._build_insight_guide_box(self.ai_mood_tab, "mood")

    def _setup_activity_matrix_tab(self):
        tk.Label(
            self.activity_matrix_tab,
            text="Activity Matrix: GitHub-style calendar heatmap of volume, cognitive clarity, diversity, pauses, or mood.",
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
        metric_options = [
            "Words (Volume)",
            "Clarity Index",
            "Lexical Diversity",
            "Long Pauses %",
            "Vitality",
            "Mood",
        ]
        self.activity_metric_dropdown = tk.OptionMenu(
            header_frame, self.activity_metric_var, *metric_options,
            command=lambda _: self.refresh_insights_tabs(),
        )
        self.activity_metric_dropdown.config(bg=COLOR_BG, fg=COLOR_FG, font=("Arial", 9), highlightthickness=1)
        self.activity_metric_dropdown.pack(side="left")

        self.activity_canvas = tk.Canvas(self.activity_matrix_tab, height=150, bg=COLOR_TEXT_BG, highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.activity_canvas.pack(fill="x", padx=10, pady=(10, 6))
        self._build_insight_guide_box(self.activity_matrix_tab, "activity")

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
        self.speaker_profile_canvas.pack(fill="x", padx=10, pady=(10, 6))
        self._build_insight_guide_box(self.speaker_profile_tab, "profile")

    def _setup_daily_brief_tab(self):
        paned = tk.Frame(self.daily_brief_tab, bg=COLOR_ACCENT_BG)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

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

    @staticmethod
    def _is_valid_insights_range(val):
        if not isinstance(val, str) or not val.strip():
            return False
        s = val.strip()
        low = s.lower()
        if s in INSIGHTS_RANGE_OPTIONS:
            return True
        if low.startswith("year ") or low.startswith("year:") or low.startswith("month:"):
            return True
        if ".." in s or (len(s) == 4 and s.isdigit()):
            return True
        return low in ("ytd", "prev_6m", "last_quarter", "last quarter", "previous month")

    def _sync_period_combo_values(self, daily_map):
        if lexical_insights is None:
            return
        opts = lexical_insights.available_period_options(daily_map or {})
        cur = self.insights_range_var.get() if hasattr(self, "insights_range_var") else INSIGHTS_RANGE_DEFAULT
        if cur and cur not in opts:
            opts.append(cur)
        if hasattr(self, "insights_period_combo"):
            try:
                self.insights_period_combo["values"] = opts
            except Exception:
                pass
        exp_combo = getattr(self, "_explorer_period_combo", None)
        if exp_combo is not None:
            try:
                exp_combo["values"] = opts
            except Exception:
                pass

    def _select_insights_range(self, key):
        if not self._is_valid_insights_range(key):
            return
        self.insights_range_var.set(key)
        self._update_range_button_styles()
        self._on_insights_range_change(key)

    def _on_insights_period_combo_select(self, _event=None):
        self._update_range_button_styles()
        self._on_insights_range_change()

    def _on_insights_filter_change(self, _event=None):
        if hasattr(self, "insights_bucket_var"):
            b_val = self.insights_bucket_var.get()
            if b_val in INSIGHTS_BUCKET_OPTIONS:
                self.config["insights_bucket"] = b_val
        if hasattr(self, "insights_lang_var"):
            l_val = self.insights_lang_var.get()
            if l_val in INSIGHTS_LANG_OPTIONS:
                self.config["insights_lang"] = l_val
        self._persist_config()
        self.refresh_insights_tabs()

    def _update_range_button_styles(self):
        current = self.insights_range_var.get() if hasattr(self, "insights_range_var") else ""
        for btn_map in (
            getattr(self, "insights_range_buttons", {}),
            getattr(self, "_explorer_range_buttons", {}),
        ):
            for key, btn in (btn_map or {}).items():
                try:
                    if key == current:
                        btn.configure(fg_color=COLOR_TEAL, text_color=COLOR_BG)
                    else:
                        btn.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_FAINT)
                except Exception:
                    pass
        self._update_trim_outlier_button_styles()

    def _is_trim_outliers_enabled(self):
        var = getattr(self, "insights_trim_outliers_var", None)
        if var is not None:
            return bool(var.get())
        return bool(self.config.get("insights_trim_outliers", False))

    def _toggle_trim_outliers(self):
        """Toggle non-destructive >2.5σ outlier filtering across all Speech Insights views."""
        new_val = not self._is_trim_outliers_enabled()
        if hasattr(self, "insights_trim_outliers_var"):
            self.insights_trim_outliers_var.set(new_val)
        self.config["insights_trim_outliers"] = new_val
        self._persist_config()
        self._current_lexical_slice = None
        self._update_trim_outlier_button_styles()
        self.refresh_insights_tabs()

    def _update_trim_outlier_button_styles(self):
        trim_on = self._is_trim_outliers_enabled()
        sl = getattr(self, "_current_lexical_slice", None)
        n_out = int(sl.get("total_outlier_buckets", 0)) if isinstance(sl, dict) else 0

        main_btn = getattr(self, "btn_trim_outliers", None)
        if main_btn is not None:
            try:
                if trim_on:
                    lbl = f"✂ >2.5σ ({n_out})" if n_out > 0 else "✂ >2.5σ ✓"
                    main_btn.configure(
                        text=lbl,
                        fg_color="#78350f",
                        hover_color="#92400e",
                        border_color="#f59e0b",
                        text_color="#fef3c7",
                    )
                else:
                    lbl = f"✂ >2.5σ·{n_out}" if n_out > 0 else "✂ >2.5σ"
                    main_btn.configure(
                        text=lbl,
                        fg_color=COLOR_BG,
                        hover_color=COLOR_TEAL_DIM,
                        border_color="#f59e0b" if n_out > 0 else COLOR_BORDER,
                        text_color="#fbbf24" if n_out > 0 else COLOR_TEXT_MUTED,
                    )
            except Exception:
                pass

        exp_btn = getattr(self, "_explorer_trim_btn", None)
        if exp_btn is not None:
            try:
                if trim_on:
                    lbl = f"✂ Kick >2.5σ ({n_out})" if n_out > 0 else "✂ Kick >2.5σ ✓"
                    exp_btn.configure(
                        text=lbl,
                        fg_color="#78350f",
                        hover_color="#92400e",
                        border_color="#f59e0b",
                        text_color="#fef3c7",
                    )
                else:
                    lbl = f"✂ Kick >2.5σ ({n_out} found)" if n_out > 0 else "✂ Kick >2.5σ"
                    exp_btn.configure(
                        text=lbl,
                        fg_color=COLOR_BG,
                        hover_color=COLOR_TEAL_DIM,
                        border_color="#f59e0b" if n_out > 0 else COLOR_BORDER,
                        text_color="#fbbf24" if n_out > 0 else COLOR_TEXT_MUTED,
                    )
            except Exception:
                pass

    def _update_insights_summary(self, today_words, session_speed):
        """One-line summary on the collapsed Speech Insights header."""
        if not hasattr(self, "insights_summary_var"):
            return
        parts = [f"{today_words:,} w", f"{session_speed} WPM"]
        snap = getattr(self, "_insights_snapshot", None)
        if snap and snap.get("clarity_index") is not None:
            parts.append(f"Clarity {snap['clarity_index']:.0f}")
        elif snap and "clarity_fillers_per_100" in snap:
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
        if not self._is_valid_insights_range(value):
            value = INSIGHTS_RANGE_DEFAULT
            self.insights_range_var.set(value)
        self._update_range_button_styles()
        self.config['insights_range'] = value
        self._persist_config()
        self.refresh_insights_tabs()

    def _insights_range_days(self):
        var = getattr(self, "insights_range_var", None)
        label = var.get() if var is not None else INSIGHTS_RANGE_DEFAULT
        if label in INSIGHTS_RANGE_OPTIONS:
            return INSIGHTS_RANGE_OPTIONS[label]
        if self._is_valid_insights_range(label):
            return label
        return INSIGHTS_RANGE_OPTIONS[INSIGHTS_RANGE_DEFAULT]

    def _get_lexical_index(self, force_rebuild=False):
        """Return the incremental per-machine LexicalDiskIndex (refreshing changed months)."""
        if lexical_insights is None:
            return None
        trans_dir = os.path.join(app_dir, TRANSCRIPTIONS_DIR)
        cache_path = os.path.join(app_dir, f"{LEXICAL_CACHE_PREFIX}.{self._hostname()}.json")
        idx = getattr(self, "_lexical_index", None)
        if idx is None or os.path.abspath(idx.transcriptions_dir) != os.path.abspath(trans_dir):
            idx = lexical_insights.LexicalDiskIndex(
                transcriptions_dir=trans_dir,
                cache_path=cache_path,
                hostname=self._hostname(),
            )
            idx.refresh(force_rebuild=force_rebuild, persist=True)
            self._lexical_index = idx
        elif force_rebuild or not idx._loaded_from_disk:
            idx.refresh(force_rebuild=force_rebuild, persist=True)
        return idx

    def _schedule_lexical_index_save(self, delay_ms=4000):
        if self._shutting_down:
            return
        if self._lexical_save_job is not None:
            try:
                self.root.after_cancel(self._lexical_save_job)
            except Exception:
                pass
        self._lexical_save_job = self._ui_after(delay_ms, self._flush_lexical_index)

    def _flush_lexical_index(self):
        self._lexical_save_job = None
        idx = getattr(self, "_lexical_index", None)
        if idx is not None:
            try:
                idx.save()
            except Exception as exc:
                self.log_internal(f"Lexical cache save skipped: {exc}")

    def _cycle_insight_tab(self, step):
        frames = getattr(self, "insight_tab_frames", None)
        if not frames:
            return
        self._select_insight_tab((self.insight_tab_index + step) % len(frames))

    def _tokenize_words(self, text):
        return tokenize_words(text)

    def _baseline_active(self):
        """True once enough history exists to use the personal trailing baseline."""
        return (
            getattr(self, "_personal_baseline_days", 0) >= PERSONAL_BASELINE_MIN_DAYS
            and bool(getattr(self, "_personal_baseline", None))
        )

    def _comparison_prefix(self):
        if self._baseline_active():
            return f"vs your {PERSONAL_BASELINE_DAYS}-day baseline"
        return "vs reference band"

    def _metric_band_status(self, value, metric_key):
        """Classify value against the personal MAD baseline band (or fallback reference band)."""
        lex_base = (getattr(self, "_lexical_baselines", None) or {}).get(metric_key)
        if lex_base and lex_base.get("is_personal"):
            low = float(lex_base["low"])
            high = float(lex_base["high"])
        elif self._baseline_active() and metric_key in self._personal_baseline:
            baseline = self._baseline_value(metric_key)
            spread = max(0.25, abs(baseline) * PERSONAL_BASELINE_BAND_FRACTION)
            low, high = baseline - spread, baseline + spread
        else:
            band = PEER_BENCHMARKS.get(metric_key, {"low": float("-inf"), "high": float("inf")})
            low = band.get("low", float("-inf"))
            high = band.get("high", float("inf"))
        if value is None:
            return "within", low, high
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
        """Reference value for a metric: personal trailing median/mean when available."""
        lex_base = (getattr(self, "_lexical_baselines", None) or {}).get(metric_key)
        if lex_base and lex_base.get("is_personal"):
            return float(lex_base["median"])
        if self._baseline_active():
            baseline = self._personal_baseline
            if metric_key in baseline:
                return baseline[metric_key]
        return self._peer_midpoint(metric_key)

    def _relative_delta_pct(self, value, metric_key):
        if value is None:
            return 0.0
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
        sl = getattr(self, "_current_lexical_slice", None)
        if sl and sl.get("callout") and int((sl.get("summary") or {}).get("words", 0)) >= INSIGHTS_MIN_WORDS_CONFIDENT:
            self.insights_profile_summary_var.set(sl["callout"])
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

    def _draw_timeseries_chart(
        self,
        canvas,
        slice_data,
        primary_key,
        secondary_key=None,
        show_baseline=True,
        empty_msg=None,
    ):
        """Render a longitudinal development line graph with personal baseline band,
        confidence-weighted raw bucket points, smooth rolling trajectory, optional
        dual-metric overlay, and clamped interactive crosshair HUD."""
        canvas._nw_ts_args = (slice_data, primary_key, secondary_key, show_baseline, empty_msg)
        if not getattr(canvas, "_nw_ts_resize_bound", False):
            canvas._nw_ts_resize_bound = True

            def _on_cfg(event, c=canvas):
                prev_wh = getattr(c, "_nw_ts_last_wh", None)
                cur_wh = (event.width, event.height)
                if cur_wh != prev_wh and event.width > 30 and event.height > 30:
                    c._nw_ts_last_wh = cur_wh
                    args = getattr(c, "_nw_ts_args", None)
                    if args is not None:
                        self._draw_timeseries_chart(c, *args)

            canvas.bind("<Configure>", _on_cfg, add="+")

        canvas.delete("all")
        if not slice_data or lexical_insights is None:
            self._draw_empty_insight_canvas(canvas, empty_msg or "No longitudinal data yet", clear_ts=False)
            return

        points = slice_data.get("points") or []
        rolling_map = slice_data.get("rolling") or {}
        baselines = slice_data.get("baselines") or {}
        spec = lexical_insights.METRIC_SPECS.get(primary_key, {
            "label": primary_key,
            "short": primary_key,
            "unit": "",
            "decimals": 1,
            "color": COLOR_TEAL,
        })
        sec_spec = (
            lexical_insights.METRIC_SPECS.get(secondary_key)
            if secondary_key and secondary_key != "None" and secondary_key != primary_key
            else None
        )

        valid_pairs = [(i, float(p[primary_key])) for i, p in enumerate(points) if p.get(primary_key) is not None]
        if not valid_pairs:
            default_empty = (
                "Recorded automatically on new local dictations"
                if primary_key in ("acoustic_wpm", "acoustic_silence_pct")
                else "No data in selected horizon"
            )
            self._draw_empty_insight_canvas(canvas, empty_msg or default_empty, clear_ts=False)
            return

        canvas.update_idletasks()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 40:
            w = int(canvas.cget("width") or 380)
        if h < 40:
            h = int(canvas.cget("height") or 146)
        canvas._nw_ts_last_wh = (w, h)

        sec_pairs = []
        if sec_spec is not None:
            sec_pairs = [(i, float(p[secondary_key])) for i, p in enumerate(points) if p.get(secondary_key) is not None]
        has_sec = len(sec_pairs) >= 1

        left = 42
        right = 44 if has_sec else 12
        top = 20
        bottom = 20
        cw = max(20, w - left - right)
        ch = max(20, h - top - bottom)

        raw_vals = [v for _, v in valid_pairs]
        roll_series = rolling_map.get(primary_key) or []
        roll_vals = [float(v) for v in roll_series if v is not None]
        all_v = raw_vals + roll_vals
        base_info = baselines.get(primary_key) if show_baseline else None
        if base_info:
            all_v.extend([float(base_info["low"]), float(base_info["high"])])

        y_min = min(all_v)
        y_max = max(all_v)
        span = max(y_max - y_min, 1e-3)
        y_min -= span * 0.12
        y_max += span * 0.14
        if min(raw_vals) >= 0.0 and primary_key != "vitality_balance":
            y_min = max(0.0, y_min)
        if y_max <= y_min:
            y_max = y_min + 1.0

        n_pts = len(points)

        def x_of(idx):
            if n_pts <= 1:
                return left + cw * 0.5
            return left + (idx / float(n_pts - 1)) * cw

        def y_of(val):
            t = (float(val) - y_min) / (y_max - y_min)
            return top + ch - max(0.0, min(1.0, t)) * ch

        decimals = int(spec.get("decimals", 1))
        unit = spec.get("unit", "")
        prim_color = spec.get("color", COLOR_TEAL)

        def fmt_v(v, dec=decimals, u=unit):
            return f"{v:.{dec}f}{u}"

        # 1. Personal baseline band (drawn underneath gridlines and data)
        if base_info:
            by_hi = y_of(base_info["high"])
            by_lo = y_of(base_info["low"])
            by_med = y_of(base_info["median"])
            canvas.create_rectangle(
                left, by_hi, left + cw, by_lo,
                fill="#112829", outline="",
            )
            canvas.create_line(left, by_hi, left + cw, by_hi, fill="#1b4341")
            canvas.create_line(left, by_lo, left + cw, by_lo, fill="#1b4341")
            canvas.create_line(left, by_med, left + cw, by_med, fill="#24635d", dash=(3, 3))

        # 2. Horizontal gridlines + primary Y-axis labels
        for frac in (0.0, 0.5, 1.0):
            gy = top + frac * ch
            gval = y_max - frac * (y_max - y_min)
            canvas.create_line(left, gy, left + cw, gy, fill="#1e293b", dash=(2, 3))
            lbl_txt = f"{gval:.0f}" if (y_max - y_min) >= 12 or decimals == 0 else f"{gval:.1f}"
            canvas.create_text(
                left - 5, gy, text=lbl_txt, fill="#94a3b8",
                font=("Consolas", 7), anchor="e",
            )

        # 3. Optional secondary overlay (independent right-hand scale)
        sec_y_min, sec_y_max = 0.0, 1.0
        if has_sec and sec_spec is not None:
            s_raw = [v for _, v in sec_pairs]
            s_roll = rolling_map.get(secondary_key) or []
            s_all = s_raw + [float(v) for v in s_roll if v is not None]
            sec_y_min = min(s_all)
            sec_y_max = max(s_all)
            s_span = max(sec_y_max - sec_y_min, 1e-3)
            sec_y_min -= s_span * 0.12
            sec_y_max += s_span * 0.14
            if min(s_raw) >= 0.0:
                sec_y_min = max(0.0, sec_y_min)
            if sec_y_max <= sec_y_min:
                sec_y_max = sec_y_min + 1.0

            sec_color = sec_spec.get("color", "#f59e0b")
            sec_dec = int(sec_spec.get("decimals", 1))

            def y_sec(val):
                t = (float(val) - sec_y_min) / (sec_y_max - sec_y_min)
                return top + ch - max(0.0, min(1.0, t)) * ch

            for frac in (0.0, 1.0):
                gy = top + frac * ch
                gval = sec_y_max - frac * (sec_y_max - sec_y_min)
                s_lbl = f"{gval:.0f}" if (sec_y_max - sec_y_min) >= 12 or sec_dec == 0 else f"{gval:.1f}"
                canvas.create_text(
                    left + cw + 4, gy, text=s_lbl, fill=sec_color,
                    font=("Consolas", 7), anchor="w",
                )

            sec_plot = [(i, float(s_roll[i])) for i in range(min(len(points), len(s_roll))) if s_roll[i] is not None]
            if len(sec_plot) < 2:
                sec_plot = sec_pairs
            if len(sec_plot) >= 2:
                s_coords = []
                for idx, sval in sec_plot:
                    s_coords.extend([x_of(idx), y_sec(sval)])
                canvas.create_line(
                    *s_coords, fill=sec_color, width=1.6, dash=(4, 2),
                    smooth=(len(sec_plot) >= 3),
                )

        # 4. Raw bucket connectors + confidence-styled dots
        if 2 <= len(valid_pairs) <= 100:
            raw_coords = []
            for idx, val in valid_pairs:
                raw_coords.extend([x_of(idx), y_of(val)])
            canvas.create_line(*raw_coords, fill="#243547", width=1)

        if len(valid_pairs) == 1:
            idx0, val0 = valid_pairs[0]
            px0, py0 = x_of(idx0), y_of(val0)
            canvas.create_oval(px0 - 4, py0 - 4, px0 + 4, py0 + 4, fill=prim_color, outline="#f0fdfa")
            canvas.create_text(
                px0, max(top + 8, py0 - 10), text=fmt_v(val0),
                fill=prim_color, font=("Arial", 8, "bold"),
            )
        elif n_pts <= 140:
            for idx, val in valid_pairs:
                px_i, py_i = x_of(idx), y_of(val)
                if points[idx].get("low_confidence"):
                    canvas.create_oval(px_i - 1.6, py_i - 1.6, px_i + 1.6, py_i + 1.6, fill="#475569", outline="")
                else:
                    canvas.create_oval(px_i - 2.3, py_i - 2.3, px_i + 2.3, py_i + 2.3, fill=prim_color, outline="")

        # 4b. Subtle top-edge amber '×' ticks for >2.5σ buckets kicked out by trim_outliers
        if n_pts <= 180:
            for idx, pt_obj in enumerate(points):
                trimmed_map = pt_obj.get("_trimmed_values")
                if isinstance(trimmed_map, dict) and primary_key in trimmed_map:
                    tx_m = x_of(idx)
                    ty_m = top + 4
                    canvas.create_line(tx_m - 2.5, ty_m - 2.5, tx_m + 2.5, ty_m + 2.5, fill="#f59e0b", width=1.4)
                    canvas.create_line(tx_m - 2.5, ty_m + 2.5, tx_m + 2.5, ty_m - 2.5, fill="#f59e0b", width=1.4)

        # 5. Bold rolling trajectory curve
        roll_pairs = [
            (i, float(roll_series[i]))
            for i in range(min(n_pts, len(roll_series)))
            if roll_series[i] is not None
        ]
        if len(roll_pairs) >= 2:
            r_coords = []
            for idx, rval in roll_pairs:
                r_coords.extend([x_of(idx), y_of(rval)])
            canvas.create_line(
                *r_coords, fill=prim_color, width=2.3,
                smooth=(len(roll_pairs) >= 3),
            )

        # 6. Compact top legend strip
        roll_lbl = slice_data.get("rolling_label", "trend")
        legend_bits = [f"━ {spec.get('short', primary_key)} ({roll_lbl})"]
        if base_info:
            legend_bits.append("░ 90d norm")
        if has_sec and sec_spec is not None:
            legend_bits.append(f"╌ {sec_spec.get('short', secondary_key)}")
        n_trim_prim = int((slice_data.get("trimmed_points_by_metric") or {}).get(primary_key, 0))
        if n_trim_prim > 0:
            legend_bits.append(f"✂ {n_trim_prim} >2.5σ")
        canvas.create_text(
            left + cw, 8, text="   ".join(legend_bits),
            fill="#94a3b8", font=("Arial", 7), anchor="e",
        )

        # 7. Smart non-overlapping X-axis date ticks
        max_ticks = max(2, min(6, cw // 82))
        if n_pts == 1:
            tick_indices = [0]
        elif n_pts <= max_ticks:
            tick_indices = list(range(n_pts))
        else:
            tick_indices = sorted({round(i * (n_pts - 1) / float(max_ticks - 1)) for i in range(max_ticks)})
        for tidx in tick_indices:
            tx = x_of(tidx)
            anchor = "w" if tidx == 0 and n_pts > 1 else ("e" if tidx == n_pts - 1 and n_pts > 1 else "center")
            t_lbl = points[tidx].get("short_date") or points[tidx].get("key", "")
            canvas.create_text(tx, h - 4, text=t_lbl, fill="#94a3b8", font=("Consolas", 7), anchor=anchor)

        # 8. Interactive <Motion> crosshair & clamped HUD tooltip
        def _on_motion(event):
            canvas.delete("ts_hover")
            if not points:
                return
            if n_pts == 1:
                idx = 0
            else:
                rel = (event.x - left) / float(max(1, cw))
                idx = max(0, min(n_pts - 1, int(round(rel * (n_pts - 1)))))

            pt = points[idx]
            hx = x_of(idx)
            canvas.create_line(hx, top, hx, top + ch, fill="#64748b", dash=(2, 2), tags="ts_hover")

            pval = pt.get(primary_key)
            trimmed_pt = pt.get("_trimmed_values") if isinstance(pt.get("_trimmed_values"), dict) else {}
            rval = roll_series[idx] if idx < len(roll_series) else None
            if pval is not None:
                hy = y_of(pval)
                canvas.create_oval(
                    hx - 4, hy - 4, hx + 4, hy + 4,
                    fill=prim_color, outline="#ffffff", width=1.2, tags="ts_hover",
                )
            elif rval is not None:
                hy = y_of(rval)
            else:
                hy = top + ch * 0.4

            lines = [str(pt.get("label") or pt.get("key", ""))]
            if pval is not None:
                r_note = f" · trend {fmt_v(rval)}" if rval is not None else ""
                lines.append(f"{spec.get('short', primary_key)}: {fmt_v(pval)}{r_note}")
            elif primary_key in trimmed_pt:
                lines.append(
                    f"{spec.get('short', primary_key)}: ✂ {fmt_v(float(trimmed_pt[primary_key]))} (>2.5σ excluded)"
                )
            else:
                lines.append(f"{spec.get('short', primary_key)}: n/a")

            if has_sec and sec_spec is not None:
                s_dec = int(sec_spec.get("decimals", 1))
                s_unit = sec_spec.get("unit", "")
                if pt.get(secondary_key) is not None:
                    lines.append(f"{sec_spec.get('short', secondary_key)}: {pt[secondary_key]:.{s_dec}f}{s_unit}")
                elif secondary_key in trimmed_pt:
                    lines.append(
                        f"{sec_spec.get('short', secondary_key)}: ✂ {float(trimmed_pt[secondary_key]):.{s_dec}f}{s_unit}"
                    )

            meta_line = f"{int(pt.get('words', 0)):,} w · {int(pt.get('entries', 0))} ent"
            if pt.get("de_share_pct") is not None:
                meta_line += f" · {pt['de_share_pct']:.0f}% DE"
            if pt.get("low_confidence"):
                meta_line += " (light)"
            lines.append(meta_line)

            hud_text = "\n".join(lines)
            tid = canvas.create_text(
                hx, hy, text=hud_text, fill=COLOR_FG,
                font=("Arial", 8), justify="left", anchor="nw", tags="ts_hover",
            )
            bbox = canvas.bbox(tid)
            if not bbox:
                return
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pad = 5
            box_x = hx + 10 if (hx + 10 + tw + pad * 2) <= (w - 4) else max(6, hx - tw - 12)
            box_y = max(4 + pad, min(h - th - pad - 4, hy - th // 2))
            canvas.coords(tid, box_x, box_y)
            bbox = canvas.bbox(tid)
            if bbox:
                bg_id = canvas.create_rectangle(
                    bbox[0] - pad, bbox[1] - pad + 1, bbox[2] + pad, bbox[3] + pad - 1,
                    fill="#0b1117", outline=COLOR_TEAL, width=1, tags="ts_hover",
                )
                canvas.tag_raise(tid, bg_id)

        _on_leave = lambda _e: canvas.delete("ts_hover")
        canvas._nw_ts_motion_fn = _on_motion
        canvas._nw_ts_leave_fn = _on_leave
        canvas.bind("<Motion>", _on_motion)
        canvas.bind("<Leave>", _on_leave)

    def _draw_empty_insight_canvas(self, canvas, msg, clear_ts=True):
        if clear_ts:
            canvas._nw_ts_args = None
        canvas._nw_ts_motion_fn = None
        canvas._nw_ts_leave_fn = None
        canvas.unbind("<Motion>")
        canvas.unbind("<Leave>")
        canvas.delete("all")
        canvas.update_idletasks()
        width = canvas.winfo_width() or 300
        height = canvas.winfo_height() or 100
        canvas.create_text(width / 2, height / 2, text=msg, fill="#7b8794", font=("Arial", 9))

    def _draw_vertical_bars(self, canvas, items, colors=None):
        canvas._nw_ts_args = None
        canvas._nw_ts_motion_fn = None
        canvas._nw_ts_leave_fn = None
        canvas.unbind("<Motion>")
        canvas.unbind("<Leave>")
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

    _TRANSCRIPT_LINE_RX = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.*)$")

    @classmethod
    def _parse_transcript_file(cls, filepath, cutoff_dt=None):
        """Parse a monthly transcript file into ``[{"timestamp": dt, "text": str}]``.

        Supports multi-paragraph entries (e.g. from Online +EDIT): lines that do
        not begin with a ``[YYYY-MM-DD HH:MM:SS] `` header are appended with a
        newline to the preceding entry.
        """
        parsed = []
        current = None
        current_excluded = False
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                stripped = raw_line.rstrip("\r\n")
                m = cls._TRANSCRIPT_LINE_RX.match(stripped)
                if m:
                    if current is not None and current["text"].strip():
                        current["text"] = current["text"].strip()
                        parsed.append(current)
                    current = None
                    current_excluded = False
                    try:
                        ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                    if cutoff_dt is not None and ts < cutoff_dt:
                        current_excluded = True
                        continue
                    current = {"timestamp": ts, "text": m.group(2)}
                elif current is not None and not current_excluded:
                    if stripped or current["text"]:
                        current["text"] += "\n" + stripped
        if current is not None and current["text"].strip():
            current["text"] = current["text"].strip()
            parsed.append(current)
        return parsed

    def _load_analytics_entries(self, days=None):
        now = datetime.datetime.now()
        cutoff = None if days is None else (now - datetime.timedelta(days=days))
        cutoff_month = cutoff.strftime("%Y-%m") if cutoff is not None else None
        entries = []
        scanned_files = 0

        trans_dir = os.path.join(app_dir, TRANSCRIPTIONS_DIR)
        if not os.path.exists(trans_dir):
            self._analytics_scanned_files = 0
            return entries

        for filename in sorted(os.listdir(trans_dir)):
            if not filename.lower().endswith(".txt"):
                continue
            if "conflicted copy" in filename.lower():
                continue  # Dropbox sync debris duplicates entries - skip
            path = os.path.join(trans_dir, filename)
            if not os.path.isfile(path):
                continue
            scanned_files += 1
            if cutoff_month is not None and len(filename) >= 7 and filename[:4].isdigit() and filename[4] == "-":
                if filename[:7] < cutoff_month:
                    continue
            try:
                entries.extend(self._parse_transcript_file(path, cutoff_dt=cutoff))
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
                f"Baseline: your own trailing {PERSONAL_BASELINE_DAYS}-day median & MAD band "
                f"({days} active day(s) of speech; personal self-tracking, non-diagnostic)."
            )
        else:
            self.insights_peer_var.set(
                f"Baseline: {PEER_PROFILE['label']} until you reach "
                f"{PERSONAL_BASELINE_MIN_DAYS} active days (currently {days}). {PEER_BENCHMARK_NOTE}"
            )

    def _update_ai_freshness_label(self):
        if not hasattr(self, "insights_ai_age_var"):
            return
        cache = self._get_daily_insights_cache()
        newest = self._newest_ai_insight_date(cache)
        if newest is None:
            self.insights_ai_age_var.set(
                "AI insights (optional): not crunched yet — all lexical & pause graphs run 100% locally."
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
        defers archive refresh until recording stops."""
        if self._insights_refresh_job is not None:
            try:
                self.root.after_cancel(self._insights_refresh_job)
            except Exception:
                pass
        self._insights_refresh_job = self.root.after(delay_ms, self._run_scheduled_insights_refresh)

    def _run_scheduled_insights_refresh(self):
        self._insights_refresh_job = None
        if self._shutting_down:
            return
        if self.mode is not None:
            # Still recording/typing - don't burn the main thread now
            self._schedule_insights_refresh(5000)
            return
        self.refresh_insights_tabs()

    def _explorer_is_open(self):
        win = getattr(self, "_trends_explorer_win", None)
        if win is None:
            return False
        try:
            return bool(win.winfo_exists())
        except Exception:
            return False

    def refresh_insights_tabs(self, force_reload=False):
        panel_collapsed = (
            getattr(self, "insights_collapsed", None) is not None
            and self.insights_collapsed.get()
        )
        explorer_open = self._explorer_is_open()
        if panel_collapsed and not explorer_open:
            # Both accordion and wide explorer hidden: defer refresh until opened.
            self._insights_dirty = True
            if force_reload:
                self._analytics_entries_cache = None
                self._lexical_index = None
            return

        # 1. Fast incremental on-disk lexical & pause slice (<5 ms warm)
        lex_idx = self._get_lexical_index(force_rebuild=force_reload)
        sl = None
        range_label = (
            self.insights_range_var.get()
            if hasattr(self, "insights_range_var")
            else INSIGHTS_RANGE_DEFAULT
        )
        bucket_mode = (
            self.insights_bucket_var.get()
            if hasattr(self, "insights_bucket_var")
            else "Auto"
        )
        lang_mode = (
            self.insights_lang_var.get()
            if hasattr(self, "insights_lang_var")
            else "All"
        )

        trim_on = self._is_trim_outliers_enabled()

        if lex_idx is not None and lexical_insights is not None:
            self._analytics_scanned_files = lex_idx.scanned_files
            self._sync_period_combo_values(lex_idx.daily)
            sl = lexical_insights.query_slice(
                lex_idx,
                range_spec=range_label,
                bucket_mode=bucket_mode,
                lang_mode=lang_mode,
                trim_outliers=trim_on,
                outlier_sigma=INSIGHTS_OUTLIER_SIGMA,
            )
            self._current_lexical_slice = sl
            self._lexical_baselines = sl.get("baselines") or {}
            self._personal_baseline_days = sl.get("baseline_days", 0)
            if self._personal_baseline_days >= PERSONAL_BASELINE_MIN_DAYS:
                # Populate _personal_baseline medians so _baseline_active() is True
                for mk, bdict in self._lexical_baselines.items():
                    if bdict.get("is_personal"):
                        self._personal_baseline[mk] = float(bdict["median"])
            self._update_trim_outlier_button_styles()

        if panel_collapsed and explorer_open:
            self._insights_dirty = True
            self._refresh_trends_explorer()
            return

        self._insights_dirty = False
        self._peer_snapshot = {}
        entries_all = self._get_analytics_entries(force_reload)
        tokens_all = self._analytics_tokens_cache or []
        langs_all = self._analytics_langs_cache or []

        # Personal reference values for legacy heuristic keys (e.g. interventions)
        self._update_personal_baseline(entries_all, tokens_all, langs_all)
        if sl is not None and self._lexical_baselines:
            self._personal_baseline_days = max(
                self._personal_baseline_days, sl.get("baseline_days", 0)
            )
            for mk, bdict in self._lexical_baselines.items():
                if bdict.get("is_personal"):
                    self._personal_baseline[mk] = float(bdict["median"])

        # Rolling display window + optional EN/DE language filter + optional >2.5σ utterance trim
        window_idx = filter_entries_by_days(entries_all, self._insights_range_days())
        if lang_mode in ("EN", "DE"):
            target_l = lang_mode.lower()
            window_idx = [i for i in window_idx if i < len(langs_all) and langs_all[i] == target_l]
        entries = [entries_all[i] for i in window_idx]
        tokens_per_entry = [tokens_all[i] for i in window_idx]
        langs = [langs_all[i] for i in window_idx]

        removed_entry_outliers = 0
        if trim_on and lexical_insights is not None and entries:
            kept_rel, removed_entry_outliers = lexical_insights.filter_outlier_entries(
                entries, tokens_per_entry, INSIGHTS_OUTLIER_SIGMA
            )
            if removed_entry_outliers > 0:
                entries = [entries[k] for k in kept_rel]
                tokens_per_entry = [tokens_per_entry[k] for k in kept_rel]
                langs = [langs[k] for k in kept_rel]

        if hasattr(self, "insights_coverage_var"):
            cov_txt = self._build_analytics_coverage_text(entries, len(entries_all))
            if sl is not None:
                cov_txt += f" · Bucket: {sl.get('bucket_mode', 'daily')} · Lang: {lang_mode}"
            if removed_entry_outliers > 0:
                cov_txt += f" · ✂ {removed_entry_outliers} entry outlier(s) excluded"
            self.insights_coverage_var.set(cov_txt)
        self._update_baseline_label()
        self._update_ai_freshness_label()

        all_tokens = [t for entry_tokens in tokens_per_entry for t in entry_tokens]
        all_text = " ".join(e["text"] for e in entries).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", all_text) if s.strip()]
        if not sentences and all_text:
            sentences = [all_text]

        current = self._compute_metric_snapshot(entries, tokens_per_entry, langs, sentences)
        if sl is not None:
            for mk, mv in (sl.get("summary") or {}).items():
                if mv is not None:
                    current[mk] = mv
            if sl["summary"].get("mattr_diversity") is not None:
                # Replace sample-length-biased raw TTR with length-invariant MATTR-50
                current["lexical_diversity_percent"] = sl["summary"]["mattr_diversity"]
        self._insights_snapshot = current

        for group, keys in (
            ("pace", ("pace_avg_words_entry", "pace_stability")),
            ("pause", ("pause_avg_seconds", "pause_long_pct")),
            ("clarity", ("clarity_fillers_per_100", "clarity_repetition_per_1k")),
            ("lexical", ("mattr_diversity", "lexical_long_word_percent")),
            ("interventions", (
                "intervention_open_question_per_100",
                "intervention_reflection_ratio",
                "intervention_affirmation_per_100",
                "intervention_summary_per_100",
            )),
        ):
            deltas = [self._relative_delta_pct(current[k], k) for k in keys if k in current and current[k] is not None]
            if deltas:
                self._peer_snapshot[group] = sum(deltas) / len(deltas)

        self._refresh_fogwatch_tab(sl)
        self._refresh_pace_tab(entries, tokens_per_entry, sl)
        self._refresh_pause_tab(entries, sl)
        self._refresh_clarity_tab(entries, tokens_per_entry, langs, sl)
        self._refresh_lexical_tab(all_tokens, langs, sl)
        self._refresh_mind_tab(sl)
        self._refresh_intervention_tab(sentences)
        self._refresh_ai_timeline_tab(entries_all)
        self._refresh_ai_mood_tab(entries_all)
        self._refresh_activity_matrix_tab(entries, all_text, tokens_per_entry)
        self._refresh_speaker_profile_tab(entries, all_tokens, sentences, sl)
        self._refresh_daily_brief_tab(entries_all)
        self._update_insights_profile_summary()
        self._update_metric_bands(current)
        if explorer_open:
            self._refresh_trends_explorer()

    def _sync_dynamic_tab_band(self, prefix, metric_key, value):
        self.INSIGHT_BAND_METRICS[prefix] = metric_key
        if lexical_insights is not None and metric_key in lexical_insights.METRIC_SPECS:
            hdr = getattr(self, f"{prefix}_header_label", None)
            if hdr is not None:
                hdr.config(text=lexical_insights.METRIC_SPECS[metric_key]["label"].upper())
        band_c = getattr(self, f"{prefix}_band_canvas", None)
        if band_c is not None and value is not None:
            self._draw_metric_band(band_c, value, metric_key)

    def _refresh_fogwatch_tab(self, sl=None):
        if not hasattr(self, "fogwatch_summary_var"):
            return
        sl = sl or getattr(self, "_current_lexical_slice", None)
        mkey = self.fogwatch_metric_var.get()
        sec_key = "pause_long_pct" if mkey == "clarity_index" else None
        if lexical_insights is not None:
            self._set_tab_guide(
                "fogwatch",
                lexical_insights.interpret_metric_window(sl, mkey, secondary_key=sec_key),
            )
        if not sl or int((sl.get("summary") or {}).get("words", 0)) == 0:
            self.fogwatch_summary_var.set("No cognitive clarity data in slice")
            self.fogwatch_detail_var.set("Record or select a broader horizon (e.g. 6m, YTD, All) to inspect trends.")
            self.fogwatch_peer_var.set("Personal self-tracking indicator (non-diagnostic).")
            self._draw_empty_insight_canvas(self.fogwatch_canvas, "No transcripts in selected slice")
            return

        summ = sl["summary"]
        ci = summ.get("clarity_index")
        mattr = summ.get("mattr_diversity")
        vague = summ.get("vague_per_100")
        lp = summ.get("pause_long_pct")

        ci_txt = f"Clarity {ci:.0f}/100" if ci is not None else "Clarity n/a"
        mattr_txt = f"MATTR {mattr:.1f}%" if mattr is not None else "MATTR n/a"
        lp_txt = f"≥5s pauses {lp:.1f}%" if lp is not None else "pauses n/a"
        self.fogwatch_summary_var.set(f"{ci_txt} · {mattr_txt} · {lp_txt}")

        vague_txt = f"Vague {vague:.2f}/100w · " if vague is not None else ""
        self.fogwatch_detail_var.set(f"{vague_txt}{sl.get('callout', '')}")

        spec = lexical_insights.METRIC_SPECS.get(mkey, {}) if lexical_insights else {}
        val = summ.get(mkey)
        self._sync_dynamic_tab_band("fogwatch", mkey, val)
        base = (sl.get("baselines") or {}).get(mkey)
        if base and val is not None:
            u = spec.get("unit", "")
            dec = int(spec.get("decimals", 1))
            self.fogwatch_peer_var.set(
                f"{spec.get('short', mkey)}: {val:.{dec}f}{u} vs your 90d norm "
                f"{base['median']:.{dec}f}{u} (band {base['low']:.{dec}f}–{base['high']:.{dec}f}{u}) · non-diagnostic."
            )
        else:
            self.fogwatch_peer_var.set("Personal self-tracking trend (non-diagnostic).")

        self._draw_timeseries_chart(self.fogwatch_canvas, sl, mkey, secondary_key=sec_key)

    def _refresh_mind_tab(self, sl=None):
        if not hasattr(self, "mind_summary_var"):
            return
        sl = sl or getattr(self, "_current_lexical_slice", None)
        mkey = self.mind_metric_var.get()
        if lexical_insights is not None:
            self._set_tab_guide("mind", lexical_insights.interpret_metric_window(sl, mkey))
        if not sl or int((sl.get("summary") or {}).get("words", 0)) == 0:
            self.mind_summary_var.set("No local affect/style data in slice")
            self.mind_detail_var.set("100% local lexical markers populate automatically from your transcripts.")
            self.mind_peer_var.set("Personal self-tracking only — not a clinical or medical measure.")
            self._draw_empty_insight_canvas(self.mind_canvas, "No transcripts in selected slice")
            return

        summ = sl["summary"]
        vit = summ.get("vitality_balance")
        italk = summ.get("self_focus_per_100")
        absw = summ.get("absolutist_per_100")
        hedg = summ.get("hedges_per_100")
        night = summ.get("late_night_pct")

        vit_txt = f"Vitality {vit:.0f}/100" if vit is not None else "Vitality n/a"
        it_txt = f"I-Talk {italk:.1f}/100w" if italk is not None else "I-Talk n/a"
        nt_txt = f"Night {night:.0f}%" if night is not None else "Night 0%"
        self.mind_summary_var.set(f"{vit_txt} · {it_txt} · {nt_txt}")

        abs_txt = f"{absw:.2f}/100w" if absw is not None else "0.00"
        hdg_txt = f"{hedg:.2f}/100w" if hedg is not None else "0.00"
        we_val = summ.get("we_focus_per_100") or 0.0
        self.mind_detail_var.set(
            f"Absolutist {abs_txt} · Hedges {hdg_txt} · Collective 'we/wir' {we_val:.2f}/100w"
        )

        spec = lexical_insights.METRIC_SPECS.get(mkey, {}) if lexical_insights else {}
        val = summ.get(mkey)
        self._sync_dynamic_tab_band("mind", mkey, val)
        base = (sl.get("baselines") or {}).get(mkey)
        if base and val is not None:
            u = spec.get("unit", "")
            dec = int(spec.get("decimals", 1))
            self.mind_peer_var.set(
                f"{spec.get('short', mkey)}: {val:.{dec}f}{u} vs your 90d norm "
                f"{base['median']:.{dec}f}{u} (band {base['low']:.{dec}f}–{base['high']:.{dec}f}{u}) · non-diagnostic."
            )
        else:
            self.mind_peer_var.set("Local lexical self-tracking (non-diagnostic).")

        self._draw_timeseries_chart(self.mind_canvas, sl, mkey)

    def _refresh_pace_tab(self, entries, tokens_per_entry=None, sl=None):
        if not hasattr(self, "pace_summary_var"):
            return
        sl = sl or getattr(self, "_current_lexical_slice", None)
        mkey = getattr(self, "pace_metric_var", None)
        mkey = mkey.get() if mkey is not None else "pace_words_entry"
        if lexical_insights is not None:
            self._set_tab_guide("pace", lexical_insights.interpret_metric_window(sl, mkey))
        if not entries:
            self.pace_summary_var.set("No pace data yet")
            self.pace_detail_var.set("Record a few transcriptions to view pace patterns.")
            self.pace_peer_var.set("Comparison unavailable (no transcript entries).")
            self._draw_empty_insight_canvas(self.pace_canvas, "No transcripts found")
            return

        if tokens_per_entry is None:
            tokens_per_entry = [self._tokenize_words(e["text"]) for e in entries]

        counts = [len(t) for t in tokens_per_entry if len(t) > 0]
        if not counts:
            self.pace_summary_var.set("No pace data yet")
            self.pace_detail_var.set("Record a few transcriptions to view pace patterns.")
            self.pace_peer_var.set("Comparison unavailable (no tokenized entries).")
            self._draw_empty_insight_canvas(self.pace_canvas, "No word counts yet")
            return

        avg_words = float(np.mean(counts))
        variability_words = self._robust_variability_band(counts)
        stability = (
            max(0.0, 100.0 - min(100.0, (variability_words / avg_words) * 100.0 * 2.0))
            if avg_words > 0
            else 0.0
        )

        summ = (sl or {}).get("summary") or {}
        ac_wpm = summ.get("acoustic_wpm")
        wps = summ.get("words_per_sentence")
        wpm_bit = f" · {ac_wpm:.0f} WPM (acoustic)" if ac_wpm is not None else ""
        wps_bit = f" · {wps:.1f} w/sentence" if wps is not None else ""
        self.pace_summary_var.set(f"Avg {avg_words:.1f} w/entry · Stability {stability:.0f}%{wpm_bit}")
        self.pace_detail_var.set(
            f"Variability ±{variability_words:.1f} w/entry{wps_bit} · {len(counts):,} entries in slice"
        )
        _, avg_low, avg_high = self._metric_band_status(avg_words, "pace_avg_words_entry")
        _, stab_low, stab_high = self._metric_band_status(stability, "pace_stability")
        avg_phrase, avg_delta = self._more_less_phrase(avg_words, "pace_avg_words_entry")
        stab_phrase, stab_delta = self._more_less_phrase(stability, "pace_stability")
        self.pace_peer_var.set(
            f"{self._comparison_prefix()}: {avg_phrase} words/entry and {stab_phrase} stability "
            f"({self._fmt_band(avg_low, avg_high)}; {self._fmt_band(stab_low, stab_high, '%')})."
        )
        self._peer_snapshot["pace"] = (avg_delta + stab_delta) / 2.0

        if sl is not None:
            self._sync_dynamic_tab_band("pace", mkey, summ.get(mkey, avg_words))
            self._draw_timeseries_chart(self.pace_canvas, sl, mkey)
        else:
            self._draw_empty_insight_canvas(self.pace_canvas, "No longitudinal slice")

    def _refresh_pause_tab(self, entries, sl=None):
        if not hasattr(self, "pause_summary_var"):
            return
        sl = sl or getattr(self, "_current_lexical_slice", None)
        summ = (sl or {}).get("summary") or {}
        gap_count = int(summ.get("gap_count", 0))

        # Fall back to direct entry delta scan if lexical index was bypassed
        if gap_count == 0 and len(entries) >= 2:
            gaps = []
            prev_ts = entries[0]["timestamp"]
            for entry in entries[1:]:
                delta = (entry["timestamp"] - prev_ts).total_seconds()
                prev_ts = entry["timestamp"]
                if 0 < delta <= 120:
                    gaps.append(delta)
            short = sum(1 for g in gaps if g < 2.0)
            medium = sum(1 for g in gaps if 2.0 <= g < 5.0)
            long_c = sum(1 for g in gaps if g >= 5.0)
            gap_count = len(gaps)
            avg_gap = float(np.mean(gaps)) if gaps else 0.0
            p90_gap = float(np.percentile(gaps, 90)) if gaps else 0.0
            long_pct = (long_c / gap_count * 100.0) if gap_count else 0.0
        else:
            short = int(summ.get("gap_short", 0))
            medium = int(summ.get("gap_med", 0))
            long_c = int(summ.get("gap_long", 0))
            avg_gap = float(summ.get("pause_avg_s") or 0.0)
            p90_gap = float(summ.get("pause_p90_s") or 0.0)
            long_pct = float(summ.get("pause_long_pct") or 0.0)

        view_mode = getattr(self, "pause_view_var", None)
        view_mode = view_mode.get() if view_mode is not None else "pause_long_pct"
        if lexical_insights is not None:
            if view_mode == "breakdown":
                self._set_tab_guide(
                    "pause",
                    lexical_insights.interpret_categorical_view(
                        "pause_breakdown", {"short": short, "medium": medium, "long": long_c}, sl
                    ),
                )
            else:
                self._set_tab_guide("pause", lexical_insights.interpret_metric_window(sl, view_mode))

        if gap_count == 0 and view_mode != "acoustic_silence_pct":
            self.pause_summary_var.set("No burst pause intervals in slice")
            self.pause_detail_var.set("Needs consecutive utterances within 120 seconds.")
            self.pause_peer_var.set("Comparison unavailable (no burst gaps ≤120s in slice).")
            self._draw_empty_insight_canvas(self.pause_canvas, "No pause gaps ≤120s in slice")
            return

        vad_sil = summ.get("acoustic_silence_pct")
        vad_bit = f" · VAD silence {vad_sil:.1f}%" if vad_sil is not None else ""
        self.pause_summary_var.set(
            f"Long ≥5s: {long_pct:.1f}% · P90 {p90_gap:.1f}s · Avg {avg_gap:.1f}s"
        )
        self.pause_detail_var.set(
            f"Short <2s: {short} · Med 2–5s: {medium} · Long ≥5s: {long_c}{vad_bit}"
        )

        _, avg_low, avg_high = self._metric_band_status(avg_gap, "pause_avg_seconds")
        _, long_low, long_high = self._metric_band_status(long_pct, "pause_long_pct")
        avg_phrase, avg_delta = self._more_less_phrase(avg_gap, "pause_avg_seconds")
        long_phrase, long_delta = self._more_less_phrase(long_pct, "pause_long_pct")
        self.pause_peer_var.set(
            f"{self._comparison_prefix()}: {long_phrase} ≥5s stall share and {avg_phrase} mean gap "
            f"({self._fmt_band(long_low, long_high, '%')}; {self._fmt_band(avg_low, avg_high, 's')})."
        )
        self._peer_snapshot["pause"] = (avg_delta + long_delta) / 2.0

        if view_mode == "breakdown":
            self._sync_dynamic_tab_band("pause", "pause_long_pct", long_pct)
            self._draw_vertical_bars(
                self.pause_canvas,
                [("Short <2s", short), ("Med 2-5s", medium), ("Long ≥5s", long_c)],
                colors=["#22c55e", "#f59e0b", "#ef4444"],
            )
        else:
            self._sync_dynamic_tab_band("pause", view_mode, summ.get(view_mode, long_pct))
            self._draw_timeseries_chart(self.pause_canvas, sl, view_mode)

    def _refresh_clarity_tab(self, entries=None, tokens_per_entry=None, langs=None, sl=None):
        """Fluency, self-repairs & vague placeholder trends + per-language filler breakdown."""
        if not hasattr(self, "clarity_summary_var"):
            return
        sl = sl or getattr(self, "_current_lexical_slice", None)
        entries = entries or []
        tokens_per_entry = tokens_per_entry or []
        langs = langs or []
        tokens = [t for toks in tokens_per_entry for t in toks]
        view_mode = getattr(self, "clarity_view_var", None)
        view_mode = view_mode.get() if view_mode is not None else "fillers_per_100"

        total_words = len(tokens)
        if total_words == 0:
            if lexical_insights is not None:
                if view_mode == "breakdown":
                    self._set_tab_guide(
                        "clarity",
                        lexical_insights.interpret_categorical_view("clarity_breakdown", {"top_items": []}, sl),
                    )
                else:
                    self._set_tab_guide("clarity", lexical_insights.interpret_metric_window(sl, view_mode))
            self.clarity_summary_var.set("No clarity data yet")
            self.clarity_detail_var.set("Transcribe more speech to estimate fluency & retrieval signals.")
            self.clarity_peer_var.set("Comparison unavailable (no words detected).")
            self._draw_empty_insight_canvas(self.clarity_canvas, "No words in selected slice")
            return

        counts, per_lang = self._filler_stats(entries, tokens_per_entry, langs)
        summ = (sl or {}).get("summary") or {}
        filler_total = sum(b["fillers"] for b in per_lang.values())
        filler_per_100 = (
            float(summ["fillers_per_100"])
            if summ.get("fillers_per_100") is not None
            else (filler_total * 100.0) / total_words
        )
        repairs_per_1k = (
            float(summ["disfluency_per_1k"])
            if summ.get("disfluency_per_1k") is not None
            else (sum(1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i + 1]) * 1000.0) / total_words
        )
        vague_per_100 = float(summ.get("vague_per_100") or 0.0)

        per_lang_bits = []
        for lang in ("de", "en"):
            bucket = per_lang.get(lang)
            if not bucket or not bucket["words"]:
                continue
            rate = (bucket["fillers"] * 100.0) / bucket["words"]
            per_lang_bits.append(f"{lang.upper()} {rate:.1f}/100w ({bucket['words']:,}w)")
        lang_note = " · ".join(per_lang_bits) if per_lang_bits else f"{total_words:,} words"

        self.clarity_summary_var.set(
            f"Fillers {filler_per_100:.2f}/100w · Repairs {repairs_per_1k:.1f}/1k · Vague {vague_per_100:.2f}/100w"
        )
        self.clarity_detail_var.set(f"By language: {lang_note}")
        _, filler_low, filler_high = self._metric_band_status(filler_per_100, "clarity_fillers_per_100")
        _, rep_low, rep_high = self._metric_band_status(repairs_per_1k, "clarity_repetition_per_1k")
        filler_phrase, filler_delta = self._more_less_phrase(filler_per_100, "clarity_fillers_per_100")
        rep_phrase, rep_delta = self._more_less_phrase(repairs_per_1k, "clarity_repetition_per_1k")
        self.clarity_peer_var.set(
            f"{self._comparison_prefix()}: {filler_phrase} fillers and {rep_phrase} repair density "
            f"({self._fmt_band(filler_low, filler_high, '/100w')}; {self._fmt_band(rep_low, rep_high, '/1k')})."
        )
        self._peer_snapshot["clarity"] = (filler_delta + rep_delta) / 2.0

        if view_mode == "breakdown":
            self._sync_dynamic_tab_band("clarity", "clarity_fillers_per_100", filler_per_100)
            items = Counter(counts).most_common(5)
            if lexical_insights is not None:
                self._set_tab_guide(
                    "clarity",
                    lexical_insights.interpret_categorical_view("clarity_breakdown", {"top_items": items}, sl),
                )
            if not items:
                self._draw_empty_insight_canvas(self.clarity_canvas, "No fillers detected in slice")
                return
            self._draw_vertical_bars(
                self.clarity_canvas,
                items,
                colors=["#38bdf8", "#22d3ee", "#2dd4bf", "#f59e0b", "#f97316"],
            )
        else:
            if lexical_insights is not None:
                self._set_tab_guide("clarity", lexical_insights.interpret_metric_window(sl, view_mode))
            self._sync_dynamic_tab_band("clarity", view_mode, summ.get(view_mode, filler_per_100))
            self._draw_timeseries_chart(self.clarity_canvas, sl, view_mode)

    def _refresh_lexical_tab(self, tokens=None, langs=None, sl=None):
        if not hasattr(self, "lexical_summary_var"):
            return
        sl = sl or getattr(self, "_current_lexical_slice", None)
        tokens = tokens or []
        view_mode = getattr(self, "lexical_view_var", None)
        view_mode = view_mode.get() if view_mode is not None else "mattr_diversity"
        if not tokens:
            if lexical_insights is not None:
                if view_mode == "breakdown":
                    self._set_tab_guide(
                        "lexical",
                        lexical_insights.interpret_categorical_view("lexical_breakdown", {"top_words": []}, sl),
                    )
                else:
                    self._set_tab_guide("lexical", lexical_insights.interpret_metric_window(sl, view_mode))
            self.lexical_summary_var.set("No vocabulary data yet")
            self.lexical_detail_var.set("Start dictating to populate vocabulary metrics.")
            self.lexical_peer_var.set("Comparison unavailable (no vocabulary sample).")
            self._draw_empty_insight_canvas(self.lexical_canvas, "No vocabulary data")
            return

        total = len(tokens)
        summ = (sl or {}).get("summary") or {}
        if summ.get("mattr_diversity") is not None:
            diversity = float(summ["mattr_diversity"])
        elif lexical_insights is not None:
            m_sum, m_w = lexical_insights.compute_mattr(tokens)
            diversity = (m_sum / m_w) * 100.0 if m_w > 0 else (len(set(tokens)) / total * 100.0)
        else:
            diversity = (len(set(tokens)) / total) * 100.0

        counts = Counter(tokens)
        long_word_count = sum(c for w, c in counts.items() if len(w) >= 7)
        long_ratio = (
            float(summ["lexical_long_word_percent"])
            if summ.get("lexical_long_word_percent") is not None
            else (long_word_count / total) * 100.0
        )
        content_pct = summ.get("content_density_pct")
        cont_bit = f" · Content {content_pct:.1f}%" if content_pct is not None else ""

        self.lexical_summary_var.set(
            f"MATTR-50 {diversity:.1f}%{cont_bit} · Depth {long_ratio:.1f}%"
        )
        _, div_low, div_high = self._metric_band_status(diversity, "mattr_diversity")
        _, long_low, long_high = self._metric_band_status(long_ratio, "lexical_long_word_percent")
        div_phrase, div_delta = self._more_less_phrase(diversity, "mattr_diversity")
        long_phrase, long_delta = self._more_less_phrase(long_ratio, "lexical_long_word_percent")
        self.lexical_peer_var.set(
            f"{self._comparison_prefix()}: {div_phrase} MATTR-50 diversity and {long_phrase} 7+ char depth "
            f"({self._fmt_band(div_low, div_high, '%')}; {self._fmt_band(long_low, long_high, '%')})."
        )
        self._peer_snapshot["lexical"] = (div_delta + long_delta) / 2.0

        langs_present = set(langs or []) or {"en"}
        stop_words = set()
        for lang in langs_present:
            stop_words |= stopwords_for(lang)
        content_freq = {w: c for w, c in counts.items() if len(w) > 2 and w not in stop_words}
        top = Counter(content_freq).most_common(5)
        self.lexical_detail_var.set(
            "Core Themes: " + (", ".join(w for w, _ in top) if top else "n/a")
            + f" ({total:,} words)"
        )

        if view_mode == "breakdown":
            if lexical_insights is not None:
                self._set_tab_guide(
                    "lexical",
                    lexical_insights.interpret_categorical_view("lexical_breakdown", {"top_words": top}, sl),
                )
            self._sync_dynamic_tab_band("lexical", "mattr_diversity", diversity)
            self._draw_vertical_bars(
                self.lexical_canvas,
                [(w, c) for w, c in top],
                colors=["#34d399", "#2dd4bf", "#14b8a6", "#0ea5e9", "#22d3ee"],
            )
        else:
            if lexical_insights is not None:
                self._set_tab_guide("lexical", lexical_insights.interpret_metric_window(sl, view_mode))
            self._sync_dynamic_tab_band("lexical", view_mode, summ.get(view_mode, diversity))
            self._draw_timeseries_chart(self.lexical_canvas, sl, view_mode)

    _OPEN_QUESTION_RX = re.compile(
        r"\b(what|how|when|where|who|why|could|would|can|tell me|"
        r"was|wie|wann|wo|wer|warum|wieso|weshalb|inwiefern|könnten|würden|erzähl)\b",
        re.IGNORECASE,
    )
    _SELF_WORDS_RX = re.compile(
        r"\b(i|me|my|mine|myself|ich|mir|mich|mein|meine|meinem|meinen|meiner|meines)\b",
        re.IGNORECASE,
    )
    _COLLECTIVE_WORDS_RX = re.compile(
        r"\b(we|us|our|ours|ourselves|wir|uns|unser|unsere|unserem|unseren|unserer|unseres)\b",
        re.IGNORECASE,
    )
    _REFLECTION_PHRASES = (
        "you feel", "it sounds like", "i hear", "you're feeling", "what i'm hearing",
        "du fühlst", "sie fühlen", "es klingt", "ich höre", "wenn ich dich richtig verstehe",
    )
    _AFFIRMATION_PHRASES = (
        "that makes sense", "good job", "great", "i appreciate", "thank you", "important work",
        "das macht sinn", "das ergibt sinn", "super", "gut gemacht", "danke", "vielen dank", "ich schätze",
    )
    _SUMMARY_PHRASES = (
        "to summarize", "so far", "in summary", "what we covered",
        "zusammenfassend", "bisher", "kurz zusammengefasst", "insgesamt",
    )

    @classmethod
    def _count_intervention_markers(cls, sentences):
        """Heuristic counts of (open questions, reflections, affirmations,
        summaries) across English and German sentences."""
        open_questions = reflections = affirmations = summaries = 0
        for sentence in sentences:
            s_lower = sentence.lower()
            if "?" in sentence and cls._OPEN_QUESTION_RX.search(sentence):
                open_questions += 1
            if any(p in s_lower for p in cls._REFLECTION_PHRASES):
                reflections += 1
            if any(p in s_lower for p in cls._AFFIRMATION_PHRASES):
                affirmations += 1
            if any(p in s_lower for p in cls._SUMMARY_PHRASES):
                summaries += 1
        return open_questions, reflections, affirmations, summaries

    def _refresh_intervention_tab(self, sentences=None):
        if not hasattr(self, "intervention_summary_var"):
            return  # tab body not built yet
        sl = getattr(self, "_current_lexical_slice", None)
        if not sentences:
            if lexical_insights is not None:
                self._set_tab_guide(
                    "intervention",
                    {
                        "explanation": (
                            "Counts conversational stance markers — open questions, empathetic reflections, "
                            "affirmations, summaries, and individual ('I/ich') vs. collective ('we/wir') framing."
                        ),
                        "pattern": "No sentence samples in this slice yet — widen horizon to inspect stance.",
                    },
                )
            self.intervention_summary_var.set("No intervention data yet")
            self.intervention_detail_var.set("Heuristic language tag counts will appear here.")
            self.intervention_peer_var.set("Comparison unavailable (no sentence samples).")
            self._draw_empty_insight_canvas(self.intervention_canvas, "No language samples")
            return

        open_questions, reflections, affirmations, summaries = \
            self._count_intervention_markers(sentences)

        self_count = 0
        collective_count = 0
        for sentence in sentences:
            self_count += len(self._SELF_WORDS_RX.findall(sentence))
            collective_count += len(self._COLLECTIVE_WORDS_RX.findall(sentence))

        if lexical_insights is not None:
            self._set_tab_guide(
                "intervention",
                lexical_insights.interpret_categorical_view(
                    "interventions",
                    {
                        "open_q": open_questions,
                        "reflect": reflections,
                        "affirm": affirmations,
                        "summary": summaries,
                        "self_count": self_count,
                        "we_count": collective_count,
                    },
                    sl,
                ),
            )

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

        focus_val = self_count - collective_count
        focus_label = "Focus: I" if focus_val >= 0 else "Focus: We"

        self._draw_vertical_bars(
            self.intervention_canvas,
            [("Open Q", open_questions), ("Reflect", reflections), ("Affirm", affirmations), ("Summary", summaries), (focus_label, abs(focus_val))],
            colors=["#60a5fa", "#818cf8", "#34d399", "#f59e0b", "#ec4899"],
        )

    def _get_daily_insights_cache(self):
        if self._ai_insights_mem_cache is not None:
            return self._ai_insights_mem_cache
        data = {}
        ai_path = os.path.join(app_dir, AI_INSIGHTS_FILE)
        try:
            if os.path.exists(ai_path):
                with open(ai_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._quarantine_corrupt_file(ai_path, e)
        except Exception as e:
            self.log_internal(f"Failed to load AI Insights cache: {e}")
        self._ai_insights_mem_cache = data
        return data

    def _save_daily_insights_cache(self, cache):
        self._ai_insights_mem_cache = dict(cache) if isinstance(cache, dict) else {}
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
        entries_snapshot = list(self._get_analytics_entries())
        threading.Thread(
            target=self._batch_process_ai_insights,
            args=(api_key, entries_snapshot),
            daemon=True,
        ).start()

    def _maybe_auto_crunch_ai_insights(self):
        """Once per app start, refresh the AI insight cache in the background if
        the newest cached day is more than a day old. Never runs twice and never
        runs while a recording is in progress."""
        self._auto_crunch_job = None
        if self._shutting_down or self._ai_auto_crunch_done:
            return
        if not self.config.get('ai_insights_auto', False):
            return
        api_key = self.config.get('openai_api_key', '').strip()
        if not api_key:
            return
        if self.mode is not None:
            if not self._shutting_down:
                self._auto_crunch_job = self.root.after(300000, self._maybe_auto_crunch_ai_insights)
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
        entries_snapshot = list(self._get_analytics_entries())
        threading.Thread(
            target=self._batch_process_ai_insights,
            args=(api_key, entries_snapshot),
            daemon=True,
        ).start()

    def _batch_process_ai_insights(self, api_key, entries_snapshot=None):
        try:
            self._batch_process_ai_insights_inner(api_key, entries_snapshot=entries_snapshot)
        except Exception as e:
            msg = f"AI insight crunch failed: {e}"
            self._ui_after(0, lambda: self._on_crunch_finished(msg))

    def _batch_process_ai_insights_inner(self, api_key, entries_snapshot=None):
        import openai
        client = openai.OpenAI(api_key=api_key, timeout=45.0)

        entries = list(entries_snapshot) if entries_snapshot is not None else self._load_analytics_entries()
        if not entries:
            self._ui_after(0, lambda: self._on_crunch_finished("No transcript entries found to process."))
            return

        from collections import defaultdict
        daily_text = defaultdict(list)
        for e in entries:
            day_str = e["timestamp"].strftime("%Y-%m-%d")
            daily_text[day_str].append(e["text"])

        cache = dict(self._get_daily_insights_cache())
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
            if self._shutting_down:
                break
            if day_str in cache:
                continue

            combined_text = " ".join(texts)
            if len(combined_text) < 50:
                continue

            self.log_internal(f"Analyzing AI Insights for {day_str}...")

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": combined_text[:12000]}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                result = json.loads(response.choices[0].message.content)
                cache[day_str] = result
                self._save_daily_insights_cache(cache)
                processed_days += 1
                time.sleep(1)
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
        if not dates:
            return []
        today = datetime.date.today()
        if filter_str == "Follow Horizon" and lexical_insights is not None:
            first_d = None
            try:
                first_d = datetime.date.fromisoformat(sorted(dates)[0])
            except Exception:
                pass
            range_lbl = (
                self.insights_range_var.get()
                if hasattr(self, "insights_range_var")
                else INSIGHTS_RANGE_DEFAULT
            )
            start_d, end_d = lexical_insights.resolve_date_bounds(
                range_lbl, today=today, first_archive_date=first_d
            )
            s_iso = start_d.isoformat() if start_d else "0000-01-01"
            e_iso = end_d.isoformat() if end_d else "9999-12-31"
            return [d for d in dates if s_iso <= d <= e_iso]

        prev_month_date = today.replace(day=1) - datetime.timedelta(days=1)
        filtered = []
        for d in dates:
            try:
                dt = datetime.datetime.strptime(d, "%Y-%m-%d").date()
                delta_days = (today - dt).days
                if filter_str == "Last 7 Days":
                    if 0 <= delta_days <= 7:
                        filtered.append(d)
                elif filter_str == "Last 30 Days":
                    if 0 <= delta_days <= 30:
                        filtered.append(d)
                elif filter_str == "Previous Month":
                    if dt.year == prev_month_date.year and dt.month == prev_month_date.month:
                        filtered.append(d)
                else:  # All Time
                    filtered.append(d)
            except Exception:
                pass
        return filtered

    def _refresh_ai_timeline_tab(self, entries):
        if not hasattr(self, "ai_timeline_summary_var"):
            return
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
            return
        self.ai_mood_summary_var.set("Loading mood cache...")
        self.ai_mood_canvas.update_idletasks()
        sl = getattr(self, "_current_lexical_slice", None)

        cache = self._get_daily_insights_cache()
        if not cache:
            if lexical_insights is not None:
                self._set_tab_guide(
                    "mood",
                    lexical_insights.interpret_categorical_view("mood", {"scores": []}, sl),
                )
            self._draw_empty_insight_canvas(self.ai_mood_canvas, "No AI history to plot yet.")
            return

        dates = sorted(cache.keys())
        dates = self._filter_ai_dates(dates, self.ai_mood_temporal_var.get())

        valid_dates = [d for d in dates if "mood_score" in cache[d]]
        scores = [float(cache[d].get("mood_score", 50)) for d in valid_dates]
        if lexical_insights is not None:
            self._set_tab_guide(
                "mood",
                lexical_insights.interpret_categorical_view("mood", {"scores": scores}, sl),
            )
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

        from collections import Counter
        all_tags = []
        for d in dates:
            all_tags.extend(cache[d].get("projects", []))
            all_tags.extend(cache[d].get("themes", []))

        top_tags = [tag.title() for tag, _ in Counter(tag.lower() for tag in all_tags if len(tag) > 2).most_common(4)]

        if not top_tags:
            self._draw_empty_insight_canvas(canvas, "No recurring projects/themes detected.")
            return

        row_h = ch / len(top_tags)
        for i, tag in enumerate(top_tags):
            y_center = top + i * row_h + (row_h / 2)
            short_tag = tag if len(tag) <= 20 else tag[:19] + "…"
            canvas.create_text(left - 8, y_center, text=short_tag, fill=COLOR_FG, font=("Arial", 9), anchor="e")
            canvas.create_line(left, y_center, left + cw, y_center, fill=COLOR_BORDER, dash=(2, 2))

        day_w = cw / max(len(dates), 1)
        colors = ["#2dd4bf", "#818cf8", "#f472b6", "#fbbf24"]

        for d_i, day_str in enumerate(dates):
            day_data = cache[day_str]
            day_tags = [t.lower() for t in day_data.get("projects", [])] + [t.lower() for t in day_data.get("themes", [])]

            x1 = left + d_i * day_w
            x2 = x1 + day_w - 1 if day_w > 2 else x1 + day_w

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

        if cw <= 0 or ch <= 0:
            return

        canvas.create_text(left - 5, top, text="100", fill=COLOR_FG, font=("Arial", 8), anchor="e")
        canvas.create_line(left, top, left + cw, top, fill=COLOR_BORDER, dash=(2, 2))

        mid_y = top + ch / 2
        canvas.create_text(left - 5, mid_y, text="50", fill="#94a3b8", font=("Arial", 8), anchor="e")
        canvas.create_line(left, mid_y, left + cw, mid_y, fill=COLOR_BORDER, dash=(2, 2))

        canvas.create_text(left - 5, top + ch, text="0", fill=COLOR_FG, font=("Arial", 8), anchor="e")
        canvas.create_line(left, top + ch, left + cw, top + ch, fill=COLOR_BORDER, dash=(2, 2))

        step = cw / max(len(dates) - 1, 1)
        pts = []
        point_data = []

        for i, d in enumerate(dates):
            score = cache[d].get("mood_score", 50)
            score = max(0, min(100, score))
            x = left + i * step
            y = top + ch - (score / 100.0) * ch
            pts.extend([x, y])
            point_data.append((x, y, d, score, cache[d].get("mood", "")))

        canvas.create_line(*pts, fill="#ec4899", width=2, smooth=True)

        for x, y, d, score, mood in point_data:
            pt_id = canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#fbcfe8", outline="#be185d", tags="mood_pt")

            def bind_hover(tag_id, px, py, p_date, p_score, p_mood):
                canvas.tag_bind(tag_id, "<Enter>", lambda e, text=f"{p_date}\n{p_mood} ({p_score})": self._on_hist_hover_simple(canvas, type('obj', (object,), {'x': px, 'y': py}), text))
                canvas.tag_bind(tag_id, "<Leave>", lambda e: canvas.delete("tooltip", "tooltip_bg"))

            bind_hover(pt_id, x, y, d, score, mood)

        canvas.create_text(left, h - 5, text=dates[0][-5:], fill="#94a3b8", font=("Arial", 7), anchor="sw")
        canvas.create_text(left + cw, h - 5, text=dates[-1][-5:], fill="#94a3b8", font=("Arial", 7), anchor="se")

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

    def _refresh_activity_matrix_tab(self, entries, all_text, tokens_per_entry=None):
        if not hasattr(self, "activity_canvas"):
            return
        self.activity_canvas.delete("all")
        self.activity_canvas.update_idletasks()

        from collections import defaultdict
        metric_type = self.activity_metric_var.get()
        daily_vals = defaultdict(float)

        lex_idx = self._get_lexical_index() if lexical_insights is not None else None
        lang_mode = self.insights_lang_var.get() if hasattr(self, "insights_lang_var") else "All"
        sl = getattr(self, "_current_lexical_slice", None)
        key_map = {
            "Clarity Index": "clarity_index",
            "Lexical Diversity": "mattr_diversity",
            "Long Pauses %": "pause_long_pct",
            "Vitality": "vitality_balance",
        }
        mapped_mkey = key_map.get(metric_type)

        if metric_type == "Mood":
            cache = self._get_daily_insights_cache()
            for d, data in cache.items():
                if "mood_score" in data:
                    daily_vals[d] = float(data["mood_score"])
        elif lex_idx is not None and mapped_mkey is not None:
            offsets = lexical_insights._compute_language_offsets(lex_idx.daily)
            baselines = getattr(self, "_lexical_baselines", None) or {}
            for d_iso, rec in lex_idx.daily.items():
                m = lexical_insights.compute_record_metrics(rec, lang_mode=lang_mode, lang_offsets=offsets)
                if m["words"] <= 0:
                    continue
                if mapped_mkey == "clarity_index":
                    val = lexical_insights.score_clarity_index(m, baselines)
                else:
                    val = m.get(mapped_mkey)
                if val is not None:
                    daily_vals[d_iso] = float(val)
        elif tokens_per_entry is not None and len(tokens_per_entry) == len(entries):
            for e, toks in zip(entries, tokens_per_entry):
                d = e["timestamp"].strftime("%Y-%m-%d")
                daily_vals[d] += len(toks)
        else:
            for e in entries:
                d = e["timestamp"].strftime("%Y-%m-%d")
                daily_vals[d] += len(self._tokenize_words(e["text"]))

        if self._is_trim_outliers_enabled() and lexical_insights is not None and len(daily_vals) >= 6:
            cut_key = mapped_mkey or "volume_words"
            base_ref = (getattr(self, "_lexical_baselines", None) or {}).get(cut_key)
            lo_c, hi_c, _, _ = lexical_insights.compute_outlier_cutoff(
                list(daily_vals.values()), base_ref, cut_key, INSIGHTS_OUTLIER_SIGMA
            )
            if lo_c is not None or hi_c is not None:
                for d_k in list(daily_vals.keys()):
                    dv = daily_vals[d_k]
                    if (hi_c is not None and dv > hi_c) or (lo_c is not None and dv < lo_c):
                        del daily_vals[d_k]

        if lexical_insights is not None:
            mean_v = (sum(daily_vals.values()) / len(daily_vals)) if daily_vals else None
            cat_g = lexical_insights.interpret_categorical_view(
                "activity",
                {"metric_type": metric_type, "active_days": len(daily_vals), "mean_val": mean_v},
                sl,
            )
            if mapped_mkey is not None and sl is not None:
                m_g = lexical_insights.interpret_metric_window(sl, mapped_mkey)
                cat_g["explanation"] = f"{cat_g['explanation']} Layer: {m_g['explanation']}"
                cat_g["pattern"] = m_g["pattern"]
            self._set_tab_guide("activity", cat_g)

        if not daily_vals:
            self._draw_empty_insight_canvas(self.activity_canvas, "No data in selected metric yet.")
            return

        self._draw_github_heatmap(self.activity_canvas, daily_vals, metric_type)

    def _refresh_speaker_profile_tab(self, entries, all_tokens, sentences, sl=None):
        if not hasattr(self, "speaker_profile_canvas"):
            return
        self.speaker_profile_canvas.delete("all")
        self.speaker_profile_canvas.update_idletasks()
        sl = sl or getattr(self, "_current_lexical_slice", None)
        if not entries or len(all_tokens) < 10:
            if lexical_insights is not None:
                self._set_tab_guide(
                    "profile",
                    lexical_insights.interpret_categorical_view("profile", {"axes": {}}, sl),
                )
            self._draw_empty_insight_canvas(self.speaker_profile_canvas, "Need more spoken data to build profile.")
            return

        summ = (sl or {}).get("summary") or {}

        # Axis 1: Length-invariant MATTR-50 diversity (scaled 45%->15 .. 80%->95)
        if summ.get("mattr_diversity") is not None:
            mattr = float(summ["mattr_diversity"])
            diversity_score = max(10.0, min(100.0, (mattr - 40.0) * 2.2))
        else:
            unique_words = len(set(t.lower() for t in all_tokens))
            diversity_score = min(100.0, (unique_words / max(len(all_tokens), 1)) * 300.0)

        # Axis 2: Cognitive Clarity / Fluency
        if summ.get("clarity_index") is not None:
            clarity_score = max(10.0, min(100.0, float(summ["clarity_index"])))
        else:
            clarity_score = 50.0

        # Axis 3: Vitality / Mood (AI mood if crunched, else 100% local lexical vitality)
        cache = self._get_daily_insights_cache()
        mood_scores = [d.get("mood_score", 50) for d in cache.values() if "mood_score" in d]
        if mood_scores:
            positivity_score = sum(mood_scores) / len(mood_scores)
        elif summ.get("vitality_balance") is not None:
            positivity_score = float(summ["vitality_balance"])
        else:
            positivity_score = 50.0

        # Axis 4: Inquiry (Questions)
        questions = sum(1 for s in sentences if s.strip().endswith('?'))
        inquiry_score = min(100.0, (questions / max(len(sentences), 1)) * 500.0)

        # Axis 5: Sustained Thought (Sentence Length)
        avg_len = len(all_tokens) / max(len(sentences), 1)
        thought_score = min(100.0, (avg_len / 25.0) * 100.0)

        axes_data = {
            "MATTR Diversity": max(10.0, diversity_score),
            "Cognitive Clarity": max(10.0, clarity_score),
            "Vitality / Mood": max(10.0, positivity_score),
            "Inquiry Drive": max(10.0, inquiry_score),
            "Sustained Thought": max(10.0, thought_score),
        }

        if lexical_insights is not None:
            self._set_tab_guide(
                "profile",
                lexical_insights.interpret_categorical_view("profile", {"axes": axes_data}, sl),
            )

        self._draw_radar_chart(self.speaker_profile_canvas, axes_data)

    def _refresh_daily_brief_tab(self, entries):
        if not hasattr(self, "brief_listbox"):
            return
        cache = self._get_daily_insights_cache()
        self.brief_listbox.delete(0, tk.END)
        if not cache:
            self.brief_listbox.insert(tk.END, "No AI history available.")
            return

        dates = sorted(cache.keys(), reverse=True)
        for d in dates:
            score = cache[d].get("mood_score", "?")
            self.brief_listbox.insert(tk.END, f"{d}  [M: {score}]")

        if dates:
            self.brief_listbox.selection_set(0)
            self._on_brief_date_select()

    def _on_brief_date_select(self, event=None):
        sel = self.brief_listbox.curselection()
        if not sel:
            return

        item_text = self.brief_listbox.get(sel[0])
        if "available" in item_text:
            return

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
        w = canvas.winfo_width()
        w = w if w > 10 else 420
        h = canvas.winfo_height()
        h = h if h > 10 else 150
        left, top = 36, 18
        sq_size = 11
        pad = 2

        if not data_dict:
            return

        dates = sorted(data_dict.keys())
        try:
            first_data_date = datetime.datetime.strptime(dates[0], "%Y-%m-%d").date()
            end_date = datetime.date.today()
            sl = getattr(self, "_current_lexical_slice", None)
            if sl and sl.get("end_date") and sl["end_date"] != "9999-12-31":
                end_date = min(end_date, datetime.date.fromisoformat(sl["end_date"]))
        except Exception:
            return

        start_date = max(first_data_date, end_date - datetime.timedelta(days=364))
        while start_date.weekday() != 0:
            start_date -= datetime.timedelta(days=1)

        day_labels = ["Mon", "", "Wed", "", "Fri", "", "Sun"]
        for i, lbl in enumerate(day_labels):
            y = top + i * (sq_size + pad) + (sq_size / 2)
            canvas.create_text(left - 5, y, text=lbl, fill="#94a3b8", font=("Arial", 7), anchor="e")

        in_window_vals = [
            v for d_s, v in data_dict.items()
            if start_date.isoformat() <= d_s <= end_date.isoformat()
        ]
        max_val = max(in_window_vals) if in_window_vals else (max(data_dict.values()) or 1.0)
        if max_val <= 0:
            max_val = 1.0

        curr_date = start_date
        col = 0
        point_data = []

        while curr_date <= end_date:
            d_str = curr_date.strftime("%Y-%m-%d")
            has_day = d_str in data_dict
            val = data_dict.get(d_str, 0.0)

            row = curr_date.weekday()
            x = left + col * (sq_size + pad)
            y = top + row * (sq_size + pad)

            if not has_day or (metric_type == "Words (Volume)" and val <= 0):
                color = "#1e293b"
            elif metric_type in ("Mood", "Clarity Index", "Vitality"):
                if val < 42:
                    color = "#dc2626" if val < 34 else "#ef4444"
                elif val < 54:
                    color = "#ca8a04" if val < 48 else "#eab308"
                else:
                    color = "#16a34a" if val < 62 else "#22c55e"
            elif metric_type == "Long Pauses %":
                # Lower long-pause % is smoother (green), high is stall-heavy (amber/red)
                if val <= 8.0:
                    color = "#22c55e"
                elif val <= 18.0:
                    color = "#f59e0b"
                else:
                    color = "#ef4444"
            elif metric_type == "Lexical Diversity":
                if val < 58.0:
                    color = "#0e4429"
                elif val < 65.0:
                    color = "#006d32"
                elif val < 72.0:
                    color = "#26a641"
                else:
                    color = "#39d353"
            else:
                ratio = val / max_val
                if ratio < 0.25:
                    color = "#0e4429"
                elif ratio < 0.5:
                    color = "#006d32"
                elif ratio < 0.75:
                    color = "#26a641"
                else:
                    color = "#39d353"

            r_id = canvas.create_rectangle(x, y, x + sq_size, y + sq_size, fill=color, outline="#0f172a")
            if has_day and (val > 0 or metric_type == "Long Pauses %"):
                point_data.append((r_id, x, y, d_str, val))

            if row == 6:
                col += 1
            curr_date += datetime.timedelta(days=1)

        for r_id, x, y, d, v in point_data:
            def bind_hover(tag_id, px, py, p_date, p_val):
                lbl_text = f"{p_date}: {p_val:.1f} ({metric_type})"
                canvas.tag_bind(tag_id, "<Enter>", lambda e, text=lbl_text: self._on_hist_hover_simple(canvas, type('obj', (object,), {'x': px, 'y': py}), text))
                canvas.tag_bind(tag_id, "<Leave>", lambda e: canvas.delete("tooltip", "tooltip_bg"))
            bind_hover(r_id, x, y, d, v)

    def _draw_radar_chart(self, canvas, data_dict):
        import math
        w = canvas.winfo_width()
        w = w if w > 10 else 600
        h = canvas.winfo_height()
        h = h if h > 10 else 220
        cx, cy = w / 2, h / 2
        radius = min(w / 2, h / 2) - 40

        labels = list(data_dict.keys())
        values = list(data_dict.values())
        num_axes = len(labels)
        angle_step = 2 * math.pi / num_axes

        for r_ratio in [0.25, 0.5, 0.75, 1.0]:
            r = radius * r_ratio
            pts = []
            for i in range(num_axes):
                a = i * angle_step - math.pi / 2
                pts.extend([cx + r * math.cos(a), cy + r * math.sin(a)])
            canvas.create_polygon(pts, fill="", outline="#334155", dash=(2, 2))

        axis_pts = []
        for i in range(num_axes):
            a = i * angle_step - math.pi / 2
            x, y = cx + radius * math.cos(a), cy + radius * math.sin(a)
            canvas.create_line(cx, cy, x, y, fill="#475569")

            lx, ly = cx + (radius + 15) * math.cos(a), cy + (radius + 15) * math.sin(a)
            anchor = "center"
            if math.cos(a) > 0.1:
                anchor = "w"
            elif math.cos(a) < -0.1:
                anchor = "e"
            canvas.create_text(lx, ly, text=labels[i], fill="#94a3b8", font=("Arial", 8, "bold"), anchor=anchor)

            val = max(0.1, min(1.0, values[i] / 100.0))
            dx, dy = cx + (radius * val) * math.cos(a), cy + (radius * val) * math.sin(a)
            axis_pts.extend([dx, dy])

        if axis_pts:
            canvas.create_polygon(axis_pts, fill="#2dd4bf", outline="#14b8a6", width=2, stipple="gray50")
            for i in range(0, len(axis_pts), 2):
                canvas.create_oval(axis_pts[i] - 3, axis_pts[i + 1] - 3, axis_pts[i] + 3, axis_pts[i + 1] + 3, fill="#f0fdfa", outline="#0d9488")

    # ------------------------------------------------------------------
    # Wide Resizable Longitudinal Trends Explorer + Offline HTML Report
    # ------------------------------------------------------------------

    def open_trends_explorer(self):
        """Open (or focus) the wide resizable Longitudinal Speech Insights Explorer."""
        if lexical_insights is None:
            return
        if self._explorer_is_open():
            try:
                self._trends_explorer_win.deiconify()
                self._trends_explorer_win.lift()
                self._trends_explorer_win.focus_force()
                self._refresh_trends_explorer()
            except Exception:
                pass
            return

        win = tk.Toplevel(self.root)
        win.title("neurowhisper · Longitudinal Speech & Cognitive Clarity Explorer")
        win.geometry("1060x725")
        win.minsize(780, 540)
        win.configure(bg=COLOR_BG)
        self._trends_explorer_win = win

        def _close_explorer():
            self._trends_explorer_win = None
            self._explorer_range_buttons = {}
            self._explorer_period_combo = None
            self._explorer_main_canvas = None
            self._explorer_vol_canvas = None
            self._explorer_report_btn = None
            self._explorer_trim_btn = None
            try:
                win.destroy()
            except Exception:
                pass

        win.protocol("WM_DELETE_WINDOW", _close_explorer)

        # 1. Top Control Bar (synced with main Speech Insights horizon/bucket/lang/outliers)
        top_bar = tk.Frame(win, bg=COLOR_ACCENT_BG, highlightthickness=1, highlightbackground=COLOR_BORDER)
        top_bar.pack(fill="x", padx=14, pady=(12, 8))

        chip_wrap = tk.Frame(top_bar, bg=COLOR_ACCENT_BG)
        chip_wrap.pack(side="left", padx=(10, 8), pady=8)
        tk.Label(chip_wrap, text="Horizon:", bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_MUTED,
                 font=("Arial", 9, "bold")).pack(side="left", padx=(0, 6))

        self._explorer_range_buttons = {}
        for key, short in INSIGHTS_QUICK_CHIPS:
            w_c = max(32, 7 * len(short) + 14)
            b = ctk.CTkButton(
                chip_wrap, text=short, width=w_c, height=22, corner_radius=7,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=COLOR_BG, hover_color=COLOR_TEAL_DIM,
                text_color=COLOR_TEXT_FAINT,
                command=lambda k=key: self._select_insights_range(k),
            )
            b.pack(side="left", padx=1)
            self._explorer_range_buttons[key] = b

        lex_idx = self._get_lexical_index()
        period_opts = lexical_insights.available_period_options(lex_idx.daily if lex_idx else {})
        if self.insights_range_var.get() not in period_opts:
            period_opts.append(self.insights_range_var.get())

        sel_wrap = tk.Frame(top_bar, bg=COLOR_ACCENT_BG)
        sel_wrap.pack(side="left", padx=4, pady=8)

        self._explorer_period_combo = ttk.Combobox(
            sel_wrap, textvariable=self.insights_range_var,
            values=period_opts, state="readonly", width=13,
        )
        self._explorer_period_combo.pack(side="left", padx=(0, 6))
        self._explorer_period_combo.bind("<<ComboboxSelected>>", self._on_insights_period_combo_select)
        self._enhance_combobox(self._explorer_period_combo, "Select horizon or specific calendar year")

        exp_bucket = ttk.Combobox(
            sel_wrap, textvariable=self.insights_bucket_var,
            values=list(INSIGHTS_BUCKET_OPTIONS), state="readonly", width=8,
        )
        exp_bucket.pack(side="left", padx=(0, 6))
        exp_bucket.bind("<<ComboboxSelected>>", self._on_insights_filter_change)
        self._enhance_combobox(exp_bucket, "Aggregation resolution (Auto / Daily / Weekly / Monthly)")

        exp_lang = ttk.Combobox(
            sel_wrap, textvariable=self.insights_lang_var,
            values=list(INSIGHTS_LANG_OPTIONS), state="readonly", width=5,
        )
        exp_lang.pack(side="left", padx=(0, 6))
        exp_lang.bind("<<ComboboxSelected>>", self._on_insights_filter_change)
        self._enhance_combobox(exp_lang, "Language filter (All = cross-language normalized)")

        self._explorer_report_btn = ctk.CTkButton(
            top_bar, text="📄 HTML Report", width=116, height=24, corner_radius=7,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=COLOR_BG, hover_color=COLOR_TEAL_DIM,
            border_width=1, border_color=COLOR_TEAL, text_color=COLOR_TEAL,
            command=self._export_and_open_html_report,
        )
        self._explorer_report_btn.pack(side="right", padx=(4, 10), pady=8)

        self._explorer_trim_btn = ctk.CTkButton(
            top_bar, text="✂ Kick >2.5σ", width=132, height=24, corner_radius=7,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=COLOR_BG, hover_color=COLOR_TEAL_DIM,
            border_width=1, border_color=COLOR_BORDER, text_color=COLOR_TEXT_MUTED,
            command=self._toggle_trim_outliers,
        )
        self._explorer_trim_btn.pack(side="right", padx=(4, 2), pady=8)
        self._create_tooltip(
            self._explorer_trim_btn,
            "Kick out >2.5σ outlier spikes (e.g. accidental hot-mic / stuck hotkey).\n"
            "100% non-destructive — raw transcript files on disk are never modified.",
        )

        # 2. Callout & Coverage Strip
        callout_frame = tk.Frame(win, bg=COLOR_BG)
        callout_frame.pack(fill="x", padx=16, pady=(0, 6))
        self._explorer_callout_var = tk.StringVar(value="")
        self._explorer_sub_var = tk.StringVar(value="")
        tk.Label(
            callout_frame, textvariable=self._explorer_callout_var,
            bg=COLOR_BG, fg=COLOR_TEAL, font=("Arial", 10, "bold"),
            anchor="w", justify="left",
        ).pack(fill="x")
        tk.Label(
            callout_frame, textvariable=self._explorer_sub_var,
            bg=COLOR_BG, fg=COLOR_TEXT_FAINT, font=("Arial", 8),
            anchor="w", justify="left",
        ).pack(fill="x")

        # 3. 6 Clickable KPI Summary Cards
        kpi_row = tk.Frame(win, bg=COLOR_BG)
        kpi_row.pack(fill="x", padx=14, pady=(0, 8))
        self._explorer_kpi_vars = {}
        kpi_defs = [
            ("clarity_index", "CLARITY INDEX"),
            ("mattr_diversity", "MATTR-50 DIVERSITY"),
            ("vague_per_100", "WORD-FINDING (VAGUE)"),
            ("pause_long_pct", "LONG PAUSES ≥5s"),
            ("fillers_per_100", "FILLERS / REPAIRS"),
            ("vitality_balance", "VITALITY & NIGHT %"),
        ]
        for col_i, (m_key, title_txt) in enumerate(kpi_defs):
            kpi_row.grid_columnconfigure(col_i, weight=1)
            card = tk.Frame(
                kpi_row, bg=COLOR_ACCENT_BG, highlightthickness=1,
                highlightbackground=COLOR_BORDER, cursor="hand2",
            )
            card.grid(row=0, column=col_i, sticky="nsew", padx=3)
            v_main = tk.StringVar(value="—")
            v_sub = tk.StringVar(value="")
            self._explorer_kpi_vars[m_key] = (v_main, v_sub)
            l1 = tk.Label(card, text=title_txt, bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_FAINT, font=("Arial", 7, "bold"))
            l1.pack(anchor="w", padx=8, pady=(6, 0))
            l2 = tk.Label(card, textvariable=v_main, bg=COLOR_ACCENT_BG, fg=COLOR_FG, font=("Arial", 13, "bold"))
            l2.pack(anchor="w", padx=8)
            l3 = tk.Label(card, textvariable=v_sub, bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_MUTED, font=("Arial", 8))
            l3.pack(anchor="w", padx=8, pady=(0, 6))
            for wdg in (card, l1, l2, l3):
                wdg.bind("<Button-1>", lambda _e, mk=m_key: self._select_explorer_primary_metric(mk))

        # 4. Primary & Secondary Metric Selector Row
        metric_bar = tk.Frame(win, bg=COLOR_BG)
        metric_bar.pack(fill="x", padx=16, pady=(0, 4))

        self._explorer_label_to_key = {
            spec["label"]: k for k, spec in lexical_insights.METRIC_SPECS.items()
        }
        self._explorer_key_to_label = {
            k: spec["label"] for k, spec in lexical_insights.METRIC_SPECS.items()
        }
        metric_labels = list(self._explorer_label_to_key.keys())

        tk.Label(metric_bar, text="Primary Curve:", bg=COLOR_BG, fg=COLOR_TEXT_MUTED,
                 font=("Arial", 9, "bold")).pack(side="left", padx=(0, 5))
        self._explorer_primary_var = tk.StringVar(
            value=self._explorer_key_to_label["clarity_index"]
        )
        prim_cb = ttk.Combobox(
            metric_bar, textvariable=self._explorer_primary_var,
            values=metric_labels, state="readonly", width=30,
        )
        prim_cb.pack(side="left", padx=(0, 14))
        prim_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_trends_explorer())
        self._enhance_combobox(prim_cb, "Primary longitudinal metric (left Y-axis + 90-day personal baseline band)")

        tk.Label(metric_bar, text="Overlay (Right Axis):", bg=COLOR_BG, fg=COLOR_TEXT_MUTED,
                 font=("Arial", 9)).pack(side="left", padx=(0, 5))
        self._explorer_secondary_var = tk.StringVar(
            value=self._explorer_key_to_label["pause_long_pct"]
        )
        sec_cb = ttk.Combobox(
            metric_bar, textvariable=self._explorer_secondary_var,
            values=["None"] + metric_labels, state="readonly", width=28,
        )
        sec_cb.pack(side="left", padx=(0, 12))
        sec_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_trends_explorer())
        self._enhance_combobox(sec_cb, "Optional second metric plotted on independent right Y-axis")

        self._explorer_desc_var = tk.StringVar(value="")
        tk.Label(
            metric_bar, textvariable=self._explorer_desc_var,
            bg=COLOR_BG, fg=COLOR_TEXT_FAINT, font=("Arial", 8), anchor="w",
        ).pack(side="left", fill="x", expand=True)

        # 4b. Local Zero-LLM Dimension Explanation + Window Pattern Card
        guide_card = tk.Frame(
            win, bg=COLOR_ACCENT_BG, highlightthickness=1, highlightbackground="#1f2937"
        )
        guide_card.pack(fill="x", padx=14, pady=(0, 6))
        self._explorer_explain_var = tk.StringVar(value="")
        self._explorer_pattern_var = tk.StringVar(value="")
        exp_lbl = tk.Label(
            guide_card, textvariable=self._explorer_explain_var,
            bg=COLOR_ACCENT_BG, fg=COLOR_TEXT_MUTED, font=("Arial", 8),
            anchor="w", justify="left", wraplength=980,
        )
        exp_lbl.pack(fill="x", padx=10, pady=(4, 1))
        pat_lbl = tk.Label(
            guide_card, textvariable=self._explorer_pattern_var,
            bg=COLOR_ACCENT_BG, fg="#99f6e4", font=("Arial", 9, "bold"),
            anchor="w", justify="left", wraplength=980,
        )
        pat_lbl.pack(fill="x", padx=10, pady=(0, 5))

        def _sync_exp_guide_wrap(ev, l_a=exp_lbl, l_b=pat_lbl):
            wl = max(320, int(ev.width) - 24)
            if wl != getattr(guide_card, "_nw_last_wl", None):
                guide_card._nw_last_wl = wl
                l_a.configure(wraplength=wl)
                l_b.configure(wraplength=wl)

        guide_card.bind("<Configure>", _sync_exp_guide_wrap)

        # 5. Main Longitudinal Line Graph Canvas (expands with window)
        self._explorer_main_canvas = tk.Canvas(
            win, height=320, bg=COLOR_TEXT_BG,
            highlightthickness=1, highlightbackground=COLOR_BORDER,
        )
        self._explorer_main_canvas.pack(fill="both", expand=True, padx=14, pady=(0, 6))

        # 6. Bottom Volume + Late-Night Share Context Sub-Chart
        self._explorer_vol_canvas = tk.Canvas(
            win, height=84, bg=COLOR_TEXT_BG,
            highlightthickness=1, highlightbackground=COLOR_BORDER,
        )
        self._explorer_vol_canvas.pack(fill="x", padx=14, pady=(0, 12))
        def _on_vol_cfg(event, c=self._explorer_vol_canvas):
            cur_wh = (event.width, event.height)
            if cur_wh != getattr(c, "_nw_vol_last_wh", None) and event.width > 30 and event.height > 20:
                c._nw_vol_last_wh = cur_wh
                self._draw_explorer_volume_strip(self._current_lexical_slice)

        self._explorer_vol_canvas.bind("<Configure>", _on_vol_cfg)

        self._update_range_button_styles()
        self._update_trim_outlier_button_styles()
        if self._current_lexical_slice is None:
            self.refresh_insights_tabs()
        else:
            self._refresh_trends_explorer()

    def _select_explorer_primary_metric(self, metric_key):
        lbl = (getattr(self, "_explorer_key_to_label", {}) or {}).get(metric_key)
        if lbl and hasattr(self, "_explorer_primary_var"):
            self._explorer_primary_var.set(lbl)
            self._refresh_trends_explorer()

    def _refresh_trends_explorer(self):
        if not self._explorer_is_open() or lexical_insights is None:
            return
        self._update_range_button_styles()
        trim_on = self._is_trim_outliers_enabled()
        sl = self._current_lexical_slice
        if sl is None or bool(sl.get("trim_outliers", False)) != trim_on:
            lex_idx = self._get_lexical_index()
            if lex_idx is not None:
                sl = lexical_insights.query_slice(
                    lex_idx,
                    range_spec=self.insights_range_var.get(),
                    bucket_mode=self.insights_bucket_var.get(),
                    lang_mode=self.insights_lang_var.get(),
                    trim_outliers=trim_on,
                    outlier_sigma=INSIGHTS_OUTLIER_SIGMA,
                )
                self._current_lexical_slice = sl
        self._update_trim_outlier_button_styles()

        if not sl:
            return

        summ = sl.get("summary") or {}
        bases = sl.get("baselines") or {}
        self._explorer_callout_var.set(sl.get("callout", ""))
        self._explorer_sub_var.set(
            f"Slice: {sl.get('start_date')} → {sl.get('end_date')}  ·  "
            f"{sl.get('active_days_in_slice', 0)} active day(s)  ·  "
            f"{int(summ.get('words', 0)):,} words ({int(summ.get('entries', 0)):,} entries, "
            f"{summ.get('de_share_pct', 0.0):.0f}% DE)  ·  "
            f"Personal self-tracking only (non-diagnostic)."
        )

        def _fmt_kpi(mk, fmt_str, suffix=""):
            v = summ.get(mk)
            if v is None:
                return "—", "no sample"
            b = bases.get(mk)
            sub = f"90d norm {b['median']:{fmt_str}}{suffix}" if b else ""
            return f"{v:{fmt_str}}{suffix}", sub

        kpis = getattr(self, "_explorer_kpi_vars", {})
        if "clarity_index" in kpis:
            m_txt, s_txt = _fmt_kpi("clarity_index", ".1f", "")
            kpis["clarity_index"][0].set(m_txt)
            kpis["clarity_index"][1].set(s_txt or "50 = personal norm")
        if "mattr_diversity" in kpis:
            m_txt, s_txt = _fmt_kpi("mattr_diversity", ".1f", "%")
            kpis["mattr_diversity"][0].set(m_txt)
            kpis["mattr_diversity"][1].set(s_txt)
        if "vague_per_100" in kpis:
            m_txt, s_txt = _fmt_kpi("vague_per_100", ".2f", "/100w")
            kpis["vague_per_100"][0].set(m_txt)
            kpis["vague_per_100"][1].set(s_txt)
        if "pause_long_pct" in kpis:
            m_txt, s_txt = _fmt_kpi("pause_long_pct", ".1f", "%")
            p90 = summ.get("pause_p90_s")
            kpis["pause_long_pct"][0].set(m_txt)
            kpis["pause_long_pct"][1].set(f"P90 {p90:.1f}s · {s_txt}" if p90 is not None else s_txt)
        if "fillers_per_100" in kpis:
            m_txt, _ = _fmt_kpi("fillers_per_100", ".2f", "/100w")
            rep = summ.get("disfluency_per_1k")
            kpis["fillers_per_100"][0].set(m_txt)
            kpis["fillers_per_100"][1].set(f"Repairs {rep:.1f}/1k" if rep is not None else "")
        if "vitality_balance" in kpis:
            m_txt, _ = _fmt_kpi("vitality_balance", ".1f", "")
            ln = summ.get("late_night_pct")
            kpis["vitality_balance"][0].set(m_txt)
            kpis["vitality_balance"][1].set(f"Late-night {ln:.1f}%" if ln is not None else "")

        p_key = self._explorer_label_to_key.get(self._explorer_primary_var.get(), "clarity_index")
        s_lbl = self._explorer_secondary_var.get()
        s_key = self._explorer_label_to_key.get(s_lbl) if s_lbl != "None" else None

        guide = lexical_insights.interpret_metric_window(sl, p_key, secondary_key=s_key)
        corr_txt = guide.get("correlation") or ""
        self._explorer_desc_var.set(corr_txt)
        if hasattr(self, "_explorer_explain_var"):
            self._explorer_explain_var.set(f"What it measures: {guide.get('explanation', '')}")
        if hasattr(self, "_explorer_pattern_var"):
            pat_txt = guide.get("pattern", "")
            self._explorer_pattern_var.set(
                f"Window pattern: {pat_txt}" + (f"  ·  {corr_txt}" if corr_txt else "")
            )

        self._draw_timeseries_chart(
            self._explorer_main_canvas, sl, p_key, secondary_key=s_key, show_baseline=True
        )
        self._draw_explorer_volume_strip(sl)

    def _draw_explorer_volume_strip(self, sl):
        canvas = getattr(self, "_explorer_vol_canvas", None)
        if canvas is None:
            return
        try:
            if not canvas.winfo_exists():
                return
        except Exception:
            return
        canvas.delete("all")
        pts = (sl or {}).get("points") or []
        if not pts:
            self._draw_empty_insight_canvas(canvas, "No volume bars in slice")
            return
        canvas.update_idletasks()
        w = canvas.winfo_width() or 980
        h = canvas.winfo_height() or 84
        left, right, top, bottom = 42, 44, 16, 16
        cw = max(20, w - left - right)
        ch = max(16, h - top - bottom)

        kept_vols = [
            float(p.get("words", 0))
            for p in pts
            if "volume_words" not in (p.get("_trimmed_values") or {}) and float(p.get("words", 0)) > 0
        ]
        max_w = max(kept_vols) if kept_vols else 1.0
        n = len(pts)
        slot = cw / float(max(1, n))
        bar_w = max(1.5, min(18.0, slot * 0.72))

        canvas.create_text(
            left, 8,
            text="Dictation Volume per Bucket (purple cap = 22:00–05:00 late-night share)",
            fill="#94a3b8", font=("Arial", 7), anchor="w",
        )
        canvas.create_text(
            left - 5, top, text=f"{int(max_w):,}", fill="#94a3b8",
            font=("Consolas", 7), anchor="e",
        )
        canvas.create_line(left, top + ch, left + cw, top + ch, fill="#1e293b")

        for i, p in enumerate(pts):
            cx = (left + cw * 0.5) if n == 1 else (left + (i / float(n - 1)) * cw)
            if "volume_words" in (p.get("_trimmed_values") or {}):
                canvas.create_text(cx, top + 5, text="×", fill="#f59e0b", font=("Arial", 8, "bold"))
                continue
            words = float(p.get("words", 0))
            if words <= 0:
                continue
            bh = max(2.0, min(1.0, words / max_w) * ch)
            x0, x1 = cx - bar_w / 2.0, cx + bar_w / 2.0
            y1 = top + ch
            y0 = y1 - bh
            canvas.create_rectangle(x0, y0, x1, y1, fill="#1d4e5f", outline="")
            ln_pct = float(p.get("late_night_pct") or 0.0)
            if ln_pct > 0.0:
                nh = max(1.5, bh * min(1.0, ln_pct / 100.0))
                canvas.create_rectangle(x0, y0, x1, y0 + nh, fill="#a855f7", outline="")

    def _generate_html_report_sync(self, out_dir=None):
        """Build standalone offline HTML report (series.json + report.html) and return report path."""
        script_root = os.path.dirname(os.path.abspath(__file__))
        for candidate_dir in (os.path.join(app_dir, "analysis"), os.path.join(script_root, "analysis")):
            if os.path.isdir(candidate_dir) and candidate_dir not in sys.path:
                sys.path.insert(0, candidate_dir)
        from analysis import build_series, report as analysis_report

        target_out = os.path.abspath(out_dir or os.path.join(app_dir, "analysis", "out"))
        os.makedirs(target_out, exist_ok=True)
        trans_dir = os.path.join(app_dir, TRANSCRIPTIONS_DIR)
        ai_path = os.path.join(app_dir, AI_INSIGHTS_FILE)

        # Only touch Tk-scheduled flush directly when invoked on the main thread
        if threading.current_thread() is threading.main_thread():
            self._flush_lexical_index()

        cache_p = (
            self._lexical_cache_file()
            if out_dir is None
            else os.path.join(target_out, f"lexical_series_cache.{self._hostname()}.json")
        )
        series = build_series.build(
            transcripts_dir=trans_dir,
            cache_path=cache_p,
            out_dir=target_out,
            ai_insights_path=ai_path,
            trim_outliers=self._is_trim_outliers_enabled(),
        )
        series_path = os.path.join(target_out, "series.json")
        build_series.common.write_json(series_path, series)

        cal_path = os.path.join(target_out, "calendar_features.json")
        calendar = build_series.common.read_json(cal_path) if os.path.isfile(cal_path) else None
        html_text = analysis_report.render(series, calendar=calendar)
        report_path = os.path.join(target_out, "report.html")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(html_text)
        return report_path

    def _export_and_open_html_report(self):
        btn = getattr(self, "_explorer_report_btn", None)
        if btn is not None:
            try:
                btn.configure(state="disabled", text="Building...")
            except Exception:
                pass
        # Flush lexical index safely on the Tk main thread before starting worker
        try:
            self._flush_lexical_index()
        except Exception:
            pass

        def _worker():
            try:
                import pathlib

                report_path = self._generate_html_report_sync()
                self.log_internal(f"📄 Exported longitudinal HTML report: {report_path}")
                if not NO_DIALOGS:
                    webbrowser.open(pathlib.Path(os.path.abspath(report_path)).as_uri())
            except (Exception, SystemExit) as exc:
                self.log_internal(f"HTML report export failed: {exc}")
            finally:
                def _restore():
                    if btn is not None:
                        try:
                            btn.configure(state="normal", text="📄 HTML Report")
                        except Exception:
                            pass
                self._ui_after(0, _restore)

        threading.Thread(target=_worker, daemon=True).start()

    def toggle_insights(self):
        """Toggle the collapsible Speech Insights accordion; compute lazily."""
        if self.insights_collapsed.get():
            self.insights_body.pack(fill="x")
            self.insights_toggle_btn.config(text="\u25bc")
            self.insights_collapsed.set(False)
            built = self._ensure_insight_tab_built(self.insight_tab_index)
            if self._insights_dirty or built:
                self.refresh_insights_tabs()
            self.root.after(80, self.update_histogram)
            self.root.after(80, lambda: self._update_metric_bands(
                getattr(self, "_insights_snapshot", None) or {}))
        else:
            self.insights_body.pack_forget()
            self.insights_toggle_btn.config(text="\u25b6")
            self.insights_collapsed.set(True)
        self.config['insights_collapsed'] = self.insights_collapsed.get()
        self._persist_config()
        self._schedule_scroll_region_sync()

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
        self._schedule_scroll_region_sync()

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
        self._schedule_scroll_region_sync()

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
        prev = getattr(self, "_focus_cfg_job", None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except Exception:
                pass
        self._focus_cfg_job = self.root.after(60, self._scroll_to_configuration)

    def _scroll_to_configuration(self):
        """Scroll the Configuration accordion header into view without overshooting."""
        self._focus_cfg_job = None
        if getattr(self, "_shutting_down", False):
            return
        try:
            self.content_frame.update_idletasks()
            self._update_scroll_region()
            req_h = self.content_frame.winfo_reqheight()
            avail_h = max(1, self._scroll_canvas.winfo_height())
            if req_h <= avail_h:
                self._scroll_canvas.yview_moveto(0.0)
                return
            y = self.settings_toggle_btn.winfo_rooty() - self.content_frame.winfo_rooty()
            max_y = float(req_h - avail_h)
            target_y = max(0.0, min(max_y, float(y)))
            self._scroll_canvas.yview_moveto(target_y / float(req_h))
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
        self._welcome_job = None
        if self._shutting_down or self.config.get('welcome_shown'):
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

        is_cloud_mode = self.config.get('transcription_mode') == 'online'
        if self.stopping:
            word, color = "Transcribing...", COLOR_TRANSCRIBING
        elif self.mode == "live":
            word, color = "Live typing", COLOR_LIVE
        elif self.mode == "batch":
            word, color = "Batch recording", COLOR_BATCH
        elif self._is_online_mode():
            word = "Recording + edit" if self.online_with_edit else "Recording"
            color = COLOR_ONLINE
        elif not is_cloud_mode and getattr(self, "_model_loading", False):
            status_low = (self.status_var.get() if hasattr(self, "status_var") else "").lower()
            word = "Downloading..." if "download" in status_low else "Loading..."
            color = COLOR_TRANSCRIBING
        elif not is_cloud_mode and getattr(self, "_model_load_failed", False) and self.model is None:
            word, color = "Model error", COLOR_LIVE
        elif not is_cloud_mode and getattr(self, "_model_idle_unloaded", False):
            word, color = "Idle (GPU freed)", COLOR_IDLE
        else:
            word, color = "Ready", COLOR_READY

        cache = self._ui_render_cache
        if cache.get("state_word") != word:
            cache["state_word"] = word
            self.state_word_var.set(word)
        if cache.get("state_dot") != color:
            cache["state_dot"] = color
            self._set_state_dot(color)

        started = getattr(self, "_mode_started_at", None)
        if self.mode and started:
            elapsed = int(max(0.0, time.time() - started))
            elapsed_str = f"{elapsed // 60}:{elapsed % 60:02d}"
        else:
            elapsed_str = ""
        if cache.get("state_elapsed") != elapsed_str:
            cache["state_elapsed"] = elapsed_str
            self.state_elapsed_var.set(elapsed_str)

        vu_col = self._vu_color()
        if cache.get("vu_color") != vu_col:
            cache["vu_color"] = vu_col
            try:
                self.vu_meter.configure(progress_color=vu_col)
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
        """Handle backend/acceleration selection change with live hot-reload."""
        display_val = self.backend_var.get()
        backend_device_map = BACKEND_DISPLAY_TO_BACKEND_DEVICE
        new_backend, new_device = backend_device_map.get(display_val, ("auto", "auto"))

        if new_backend != self.config.get('backend') or new_device != self.config.get('device'):
            self.config['backend'] = new_backend
            self.config['device'] = new_device
            if new_device == 'auto':
                self.config['compute_type'] = 'auto'
            else:
                self.config['compute_type'] = 'float16' if new_device == 'cuda' else 'int8'
            self._persist_config()
            self._update_device_model_labels()
            self.log_internal(f"Acceleration changed to {display_val} -> reloading model in background...")
            self._manual_model_reload()

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
            if self.running and not self._shutting_down:
                try:
                    self._hotkey_watchdog_job = self.root.after(self._hotkey_watchdog_interval, check_hotkeys)
                except (tk.TclError, RuntimeError):
                    pass

        # Start first check after interval
        self._hotkey_watchdog_job = self.root.after(self._hotkey_watchdog_interval, check_hotkeys)

    def cleanup_hotkeys(self):
        """Clean up hotkeys before app exit."""
        self.log_internal("Cleaning up hotkeys...")
        self._hotkey_manager.cleanup()
        self._clear_capture_hooks()
        if hasattr(self, '_capture_hook') and self._capture_hook and keyboard is not None:
            try:
                keyboard.unhook(self._capture_hook)
            except Exception:
                pass
            self._capture_hook = None

    def _clear_capture_hooks(self):
        """Remove temporary hooks used during hotkey capture and unpause hotkeys."""
        mgr = getattr(self, "_hotkey_manager", None)
        if mgr is not None:
            mgr.paused = False
        hooks = getattr(self, '_capture_hooks', None)
        if hooks and keyboard is not None:
            for hook in hooks:
                if hook is None:
                    continue
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
        self._hotkey_pump_job = None
        if not self.running or self._shutting_down:
            return
        self._process_hotkey_actions()
        if self.running and not self._shutting_down:
            try:
                self._hotkey_pump_job = self.root.after(50, self._hotkey_pump)
            except (tk.TclError, RuntimeError):
                pass

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
        if self.stopping:
            return
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
        if self.stopping:
            return
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
            self._report_recording_start_failure("online")
            return

        # Reset streaming state before exposing self.mode to processing_loop
        self.online_with_edit = with_edit
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
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
        self.online_executor = ThreadPoolExecutor(max_workers=AUDIO_RUNTIME["online_executor_workers"])
        mode_name = 'online_transcribe_edit' if with_edit else 'online_transcribe'
        self.mode = mode_name
        
        # Update UI
        btn = self.btn_transcribe_edit if with_edit else self.btn_transcribe
        btn.configure(fg_color=COLOR_ONLINE, text_color=COLOR_FG)
        
        self.play_feedback_sound(start=True)
        self.status_var.set(f"Recording{' + GPT edit' if with_edit else ''}...")
        self.msg_queue.put("Recording started (online streaming mode)")

    def _stop_online_recording(self, with_edit=False):
        """Stop recording, drain remaining mic frames, process final segment, and combine results."""
        self.stopping = True
        self.play_feedback_sound(start=False)

        # Close stream first, then drain any trailing mic frames queued before close
        self.stop_audio_stream()
        while True:
            try:
                self.online_pending_audio.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break

        # Queue any remaining audio before detaching session state
        if self.online_pending_audio:
            self._queue_online_segment()

        with self.online_segment_lock:
            segments_snapshot = list(self.online_segments)
        executor = self.online_executor
        self.online_executor = None
        self.mode = None
        self.stopping = False
        
        # Check if we have any segments
        if not segments_snapshot:
            if executor is not None:
                executor.shutdown(wait=False)
            self._reset_online_buttons()
            return

        # Show TRANSCRIBING state
        btn = self.btn_transcribe_edit if with_edit else self.btn_transcribe
        btn.configure(fg_color=COLOR_TRANSCRIBING, text_color="white", text="Finishing...")
        self.status_var.set("Waiting for transcriptions...")
        self.update_mini_window_color(COLOR_TRANSCRIBING)
        
        # Wait for all segments to complete in background
        def wait_and_combine():
            try:
                if executor is not None:
                    executor.shutdown(wait=True)

                with self.online_segment_lock:
                    results = []
                    errors = []
                    total_latency = 0
                    total_audio_duration = 0
                    num_segments = len(segments_snapshot)
                    for seg in sorted(segments_snapshot, key=lambda x: x['seq']):
                        res = seg.get('result')
                        if res and not res.startswith('[Error'):
                            results.append(res)
                        elif seg.get('status') == 'error' and res:
                            errors.append(res)
                        total_latency += seg.get('latency', 0)
                        total_audio_duration += seg.get('duration_s', 0)

                result_text = " ".join(results).strip()

                if result_text:
                    self.total_audio_duration += total_audio_duration
                    self.today_audio_duration += total_audio_duration
                    self.last_transcription_latency = total_latency
                    self._push_pending_acoustic_meta(result_text, None, latency_ms=total_latency)
                    self.result_queue.put(f"[ONLINE] {result_text}")

                    word_count = len(result_text.split())
                    self.msg_queue.put(f"☁️ Transcribed {word_count} words from {num_segments} segments ({total_latency:.0f}ms total)")

                    def deliver():
                        self.root.clipboard_clear()
                        self.root.clipboard_append(result_text)
                        self.root.update()
                        pasted = False
                        try:
                            if PLATFORM_UTILS_AVAILABLE:
                                pasted = bool(paste_from_clipboard())
                            elif keyboard is not None:
                                keyboard.send('ctrl+v')
                                pasted = True
                        except Exception as paste_err:
                            self.log_internal(f"Auto-paste skipped (text is in clipboard): {paste_err}")

                        if self.mode is None and not self.stopping:
                            if with_edit:
                                self.btn_transcribe_edit.configure(fg_color=COLOR_READY, text_color="white", text="Done")
                            else:
                                self.btn_transcribe.configure(fg_color=COLOR_READY, text_color="white", text="Done")
                            self.status_var.set("Copied & Pasted!" if pasted else "Copied to clipboard!")
                            self.update_mini_window_color(COLOR_READY)
                            self._ui_after(UI_RUNTIME["ready_state_delay_ms"], self._reset_online_buttons)
                    self._ui_after(0, deliver)
                elif errors:
                    first_err = errors[0].strip("[]")
                    self.msg_queue.put(f"Online transcription failed: {first_err}")
                    def show_err(msg=first_err):
                        if self.mode is None and not self.stopping:
                            self._reset_online_buttons(keep_status=True)
                            self.status_var.set(f"Online {msg[:68]}")
                    self._ui_after(0, show_err)
                else:
                    self.msg_queue.put("No speech detected in online recording")
                    self._ui_after(0, self._reset_online_buttons)

            except Exception as e:
                self.msg_queue.put(f"Online transcription failed: {e}")
                self._ui_after(0, self._reset_online_buttons)

        threading.Thread(target=wait_and_combine, daemon=True).start()

    def _reset_online_buttons(self, keep_status=False):
        """Reset online mode button colors and text."""
        if self.mode is not None or self.stopping or self._shutting_down:
            return
        self.btn_transcribe.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED, text=self._mode_button_text("transcribe"))
        self.btn_transcribe_edit.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED, text=self._mode_button_text("transcribe_edit"))
        self.update_mini_window_color(COLOR_IDLE)
        if not keep_status:
            self.status_var.set("Ready")

    def _sync_openai_backend_config(self):
        """Refresh any cached OpenAI backend client after settings change."""
        self.online_backend = None
        if self.backend is not None and self.current_backend_name == 'openai':
            try:
                self.backend.configure(
                    api_key=self.config.get('openai_api_key', ''),
                    transcription_model=self.config.get('openai_transcription_model', DEFAULT_CONFIG['openai_transcription_model']),
                    edit_model=self.config.get('openai_edit_model', DEFAULT_CONFIG['openai_edit_model']),
                    edit_prompt=self.config.get('openai_edit_prompt', '') or None,
                    language=self.config.get('openai_language', DEFAULT_CONFIG['openai_language']),
                )
            except Exception:
                pass

    def _save_openai_settings(self):
        """Save OpenAI settings to config."""
        self.config['openai_api_key'] = self.api_key_var.get().strip()
        self.config['openai_transcription_model'] = self.openai_trans_model_var.get()
        self.config['openai_edit_model'] = self.openai_edit_model_var.get()
        self.config['openai_language'] = self.openai_lang_var.get()
        self.config['openai_edit_prompt'] = self.openai_prompt_var.get().strip()

        self._persist_config()
        self._sync_openai_backend_config()
        self._update_device_model_labels()
        if hasattr(self, "_update_api_key_notice"):
            try:
                self._update_api_key_notice()
            except Exception:
                pass

        self.msg_queue.put("OpenAI settings saved")
        if not NO_DIALOGS:
            messagebox.showinfo("Saved", "OpenAI settings saved successfully.")

    def _auto_save_openai_settings(self):
        """Silently save OpenAI settings when combo boxes or prompt entry change."""
        self.config['openai_transcription_model'] = self.openai_trans_model_var.get()
        self.config['openai_edit_model'] = self.openai_edit_model_var.get()
        self.config['openai_language'] = self.openai_lang_var.get()
        if hasattr(self, "openai_prompt_var"):
            self.config['openai_edit_prompt'] = self.openai_prompt_var.get().strip()
        
        try:
            self._persist_config()
            self._sync_openai_backend_config()
            self._update_device_model_labels()
            if hasattr(self, "_update_api_key_notice"):
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
                
            except ImportError:
                self._ui_after(0, lambda: messagebox.showerror("Error", 
                    "OpenAI library not installed.\n\nRun: pip install openai"))
                self._ui_after(0, lambda: self.status_var.set("Ready"))
            except Exception as e:
                self._ui_after(0, lambda: messagebox.showerror("Error", f"Test failed: {e}"))
                self._ui_after(0, lambda: self.status_var.set("Ready"))
        
        threading.Thread(target=test_connection, daemon=True).start()

    def start_hotkey_capture(self, hotkey_type):
        """Start capturing a new hotkey for batch or live mode - supports multi-key combos"""
        self._clear_capture_hooks()
        if keyboard is None:
            self.log_internal("Cannot capture hotkey: keyboard backend is unavailable on this system.")
            if not NO_DIALOGS:
                messagebox.showwarning("Hotkeys unavailable", "Keyboard hook backend is unavailable on this system.")
            return

        self.capturing_hotkey = hotkey_type
        self._hotkey_parts = set()  # Track currently held keys
        self._hotkey_confirmed = False
        if getattr(self, "_hotkey_manager", None) is not None:
            self._hotkey_manager.paused = True

        mod_order = ['ctrl', 'alt', 'shift', 'win', 'command', 'cmd']
        
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
                ordered = [m for m in mod_order if m in self._hotkey_parts]
                ordered.extend(k for k in sorted(self._hotkey_parts) if k not in mod_order)
                preview = '+'.join(ordered)
            
            preview_text = f"{preview} (Enter=OK, Esc=Cancel)"
            if hotkey_type == 'batch':
                self.batch_hotkey_btn.config(text=preview_text)
            else:
                self.live_hotkey_btn.config(text=preview_text)
        
        def on_key_down(event):
            if getattr(event, "event_type", "down") != "down":
                return
            if self.capturing_hotkey is None or self._hotkey_confirmed:
                return
            
            key_name = HotkeyManager._normalize_key_name(getattr(event, "name", None))
            if not key_name:
                return
            
            # Enter confirms the hotkey
            if key_name in ('enter', 'return'):
                if self._hotkey_parts:
                    self._hotkey_confirmed = True
                    ordered = [m for m in mod_order if m in self._hotkey_parts]
                    ordered.extend(k for k in sorted(self._hotkey_parts) if k not in mod_order)
                    new_hotkey = '+'.join(ordered)
                    
                    current_type = self.capturing_hotkey
                    self.capturing_hotkey = None
                    self._clear_capture_hooks()
                    self._ui_after(HOTKEY_RUNTIME["capture_finish_delay_ms"], lambda: self.finish_hotkey_capture(current_type, new_hotkey))
                return
            
            # Escape cancels
            if key_name in ('esc', 'escape'):
                self._clear_capture_hooks()
                self._ui_after(HOTKEY_RUNTIME["capture_finish_delay_ms"], self.cancel_hotkey_capture)
                return
            
            self._hotkey_parts.add(key_name)
            self._ui_after(HOTKEY_RUNTIME["capture_preview_delay_ms"], update_preview)
        
        def on_key_up(event):
            return
        
        try:
            on_key_down_hook = keyboard.hook(on_key_down, suppress=False)
            on_key_up_hook = keyboard.on_release(on_key_up, suppress=False)
            self._capture_hooks = (on_key_down_hook, on_key_up_hook)
        except Exception as exc:
            self.log_internal(f"Failed to hook keyboard for capture: {exc}")
            self.cancel_hotkey_capture()

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
            self._update_config_summary()

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
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
            while True:
                try:
                    self._live_chunk_queue.get_nowait()
                    self._live_chunk_queue.task_done()
                except queue.Empty:
                    break
            with self.live_buffer_lock:
                self.live_buffer = []
                self.live_backup_buffer = []
            self._live_session_texts = []
            self._live_reset_event.set()
            self.mode = "live"
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

        # Close the stream FIRST so no new audio frames arrive after our drain,
        # and while self.stopping is still True so no hotkey press can adopt a
        # dying stream.
        self.stop_audio_stream()

        # Drain any trailing mic frames queued before stream close
        drained = []
        while True:
            try:
                drained.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break

        with self.live_buffer_lock:
            if drained:
                self.live_buffer.extend(drained)
                self.live_backup_buffer.extend(drained)
                if len(self.live_backup_buffer) > 3000:
                    self.live_backup_buffer = self.live_backup_buffer[-3000:]
            remaining_buffer = self.live_buffer.copy()
            self.live_buffer = []

        if remaining_buffer and (self.model is not None or self.backend is not None):
            self.log_internal("Flushing final buffer...")
            self._live_chunk_queue.put(remaining_buffer)

        # Wait (bounded + shutdown-aware) for _live_transcribe_loop to finish
        # all queued live chunks in strict FIFO order before snapshotting.
        wait_deadline = time.monotonic() + 30.0
        while self._live_chunk_queue.unfinished_tasks > 0 and not self._shutting_down:
            if time.monotonic() > wait_deadline:
                break
            time.sleep(0.02)

        with self.live_buffer_lock:
            backup_chunks = list(self.live_backup_buffer)
        session_texts = list(self._live_session_texts)

        self.mode = None
        self.stopping = False

        if self._shutting_down:
            return

        def ui_idle():
            if self.mode is None and not self.stopping:
                self._reset_buttons_to_idle()
                self.status_var.set("Ready. (Processing Live Backup...)")
        self._ui_after(0, ui_idle)
        self.finalize_live_backup(backup_chunks, session_texts)

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
            self.batch_total_samples = 0
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()

            # Initialize streaming batch state before flipping self.mode
            self.batch_segments = []
            self.batch_segment_seq = 0
            self.batch_pending_audio = []
            self.batch_silence_count = 0

            # Create thread pool - limit workers to avoid GPU memory issues
            effective_dev = self.current_device or self.config.get('device', 'cpu')
            max_workers = (
                AUDIO_RUNTIME["batch_executor_workers_cpu"]
                if effective_dev == 'cpu'
                else AUDIO_RUNTIME["batch_executor_workers_gpu"]
            )
            self.batch_executor = ThreadPoolExecutor(max_workers=max_workers)
            self.mode = "batch"

            # Highlight BATCH button, dim LIVE button
            self.btn_batch.configure(fg_color=COLOR_BATCH, text_color="white", text=self._mode_button_text("batch"))
            self.btn_live.configure(fg_color=COLOR_BG, text_color=COLOR_TEXT_MUTED, text=self._mode_button_text("live"))
            self.status_var.set(f"BATCH RECORDING ({self.config['hotkey_batch']})...")
            self.update_mini_window_color(COLOR_BATCH)

    def _delayed_stop_batch(self):
        # Runs on a worker thread: marshal UI updates to the Tk main thread.
        time.sleep(AUDIO_RUNTIME["stop_delay_seconds"])
        # Close the stream BEFORE draining & releasing the guard (see _delayed_stop_live).
        self.stop_audio_stream()
        while True:
            try:
                chunk = self.audio_queue.get_nowait()
                self.batch_total_samples += len(chunk)
                self.batch_pending_audio.append(chunk)
            except queue.Empty:
                break

        if self._shutting_down:
            self.mode = None
            self.stopping = False
            return

        def ui_transcribing():
            if self.mode is None and not self.stopping:
                # Show TRANSCRIBING state (amber) only if a new session hasn't started
                self.btn_batch.configure(fg_color=COLOR_TRANSCRIBING, text_color="white", text="Transcribing...")
                self.btn_live.configure(
                    fg_color=COLOR_BG,
                    text_color=COLOR_TEXT_MUTED,
                    text=self._mode_button_text("live"),
                )
                self.status_var.set("Processing Batch... Please Wait.")
                self.update_mini_window_color(COLOR_TRANSCRIBING)
        self.finalize_batch(on_transcribing_ui=ui_transcribing)

    def finalize_batch(self, on_transcribing_ui=None):
        # Runs on a worker thread (waits on the transcription pool); all UI
        # work is marshalled to the Tk main thread.
        def ui_reset():
            if self.mode is None and not self.stopping:
                self._reset_buttons_to_idle()
                self.status_var.set("Ready.")

        # 1. Queue any remaining pending audio while self.stopping is still True
        if self.batch_pending_audio:
            self._queue_batch_segment()

        # 2. Snapshot session state and detach self.batch_executor BEFORE releasing
        # self.mode / self.stopping so a subsequent hotkey press cannot clobber us.
        with self.batch_segment_lock:
            segments_snapshot = list(self.batch_segments)
        total_samples = self.batch_total_samples
        executor = self.batch_executor
        self.batch_executor = None
        self.mode = None
        self.stopping = False

        if not segments_snapshot:
            if executor is not None:
                executor.shutdown(wait=False)
            self.log_internal("No audio recorded.")
            self._ui_after(0, ui_reset)
            return

        if on_transcribing_ui is not None:
            self._ui_after(0, on_transcribing_ui)

        try:
            duration = total_samples / AUDIO_RUNTIME["sample_rate"]
            self.total_audio_duration += duration
            self.today_audio_duration += duration

            # 3. Wait for this session's segments to complete transcription
            if executor is not None:
                self.log_internal("Waiting for transcription threads...")
                executor.shutdown(wait=True)

            # 4. Stitch results in correct order
            with self.batch_segment_lock:
                ordered = sorted(segments_snapshot, key=lambda x: x['seq'])
                results = [s['result'] for s in ordered if s['result'] and not s['result'].startswith('[Error')]
                seg_metas = [
                    s.get('acoustic_meta')
                    for s in ordered
                    if s.get('result') and not s['result'].startswith('[Error') and s.get('acoustic_meta')
                ]
                total_latency = sum(s.get('latency_ms', 0) for s in ordered)
                full_text = " ".join(results).strip()

            # 5. Output the combined text
            if full_text:
                self.log_internal(f"✅ Batch complete: {len(ordered)} segments stitched")
                self.last_transcription_latency = total_latency
                w_count = len(tokenize_words(full_text))
                batch_meta = (
                    lexical_insights.merge_batch_acoustic_metas(seg_metas, total_words=w_count)
                    if (lexical_insights is not None and seg_metas)
                    else None
                )
                self._push_pending_acoustic_meta(full_text, batch_meta, latency_ms=total_latency)
                self.result_queue.put(f"[BATCH] {full_text}")

                def deliver():
                    self.root.clipboard_clear()
                    self.root.clipboard_append(full_text)
                    self.root.update()

                    # Show READY state (green) - text is in clipboard
                    self.btn_batch.configure(fg_color=COLOR_READY, text_color="white", text="Ready to paste")
                    self.status_var.set("Ready! Text copied to clipboard. Pasting...")
                    self.update_mini_window_color(COLOR_READY)

                    # Auto-paste (guarded so UI always returns to idle even if keyboard hook is unavailable)
                    paste_key = get_paste_shortcut() if PLATFORM_UTILS_AVAILABLE else 'ctrl+v'
                    if keyboard is not None:
                        try:
                            keyboard.send(paste_key)
                        except Exception as paste_err:
                            self.log_internal(f"Auto-paste skipped: {paste_err}")

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
                    self._ui_after(0, self._update_state_card)
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

    def finalize_live_backup(self, backup_chunks=None, session_texts=None):
        """Produce a full-session clipboard backup at the end of Live mode.

        Clipboard-only by design: the live chunks were already typed, counted
        into the stats, and saved to the archive. For short clips (<= ~60s)
        when idle, we re-transcribe in-memory at beam_size=5 for higher
        coherence; for long sessions (or if a new recording already started),
        we join the session texts directly to avoid stalling the GPU/CPU.
        """
        if backup_chunks is None:
            with self.live_buffer_lock:
                backup_chunks = list(self.live_backup_buffer)
        if session_texts is None:
            session_texts = list(self._live_session_texts)

        if not backup_chunks and not session_texts:
            self._ui_after(
                0,
                lambda: self.status_var.set("Ready.")
                if (self.mode is None and not self.stopping)
                else None,
            )
            return

        temp_path = None
        text = ""
        try:
            if backup_chunks and len(backup_chunks) <= 600 and self.mode is None and not self._shutting_down:
                audio_np = np.concatenate(backup_chunks, axis=0).flatten()
                lang = self._local_language_arg()
                with self._model_lock:
                    self._reload_model_if_needed_locked()
                    if self.backend and hasattr(self.backend, "transcribe_audio"):
                        segments = self.backend.transcribe_audio(
                            audio_np,
                            sample_rate=AUDIO_RUNTIME["sample_rate"],
                            beam_size=5,
                            vad_filter=True,
                            language=lang,
                        )
                        text = " ".join([s.text for s in segments]).strip()
                    elif self.backend and hasattr(self.backend, "transcribe"):
                        temp_path = self._temp_audio_path("live_backup")
                        wav.write(temp_path, AUDIO_RUNTIME["sample_rate"], audio_np)
                        segments = self.backend.transcribe(temp_path)
                        text = " ".join([s.text for s in segments]).strip()
                    elif self.model is not None:
                        temp_path = self._temp_audio_path("live_backup")
                        wav.write(temp_path, AUDIO_RUNTIME["sample_rate"], audio_np)
                        segments, _ = self.model.transcribe(
                            temp_path,
                            beam_size=5,
                            vad_filter=True,
                            condition_on_previous_text=True,
                            language=lang,
                        )
                        text = " ".join([s.text for s in segments]).strip()

            if not text and session_texts:
                text = " ".join(session_texts).strip()

            def deliver(backup_text=text):
                if backup_text:
                    self.root.clipboard_clear()
                    self.root.clipboard_append(backup_text)
                    if self.mode is None and not self.stopping:
                        self.status_var.set("Ready. (Backup in Clipboard)")
                elif self.mode is None and not self.stopping:
                    self.status_var.set("Ready.")
            self._ui_after(0, deliver)
        except Exception as e:
            self.log_internal(f"Backup Error: {e}")
            self._ui_after(
                0,
                lambda: self.status_var.set("Ready.")
                if (self.mode is None and not self.stopping)
                else None,
            )
        finally:
            if temp_path:
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
        current_live_duration = 0.0

        while self.running:
            try:
                if self._live_reset_event.is_set():
                    self._live_reset_event.clear()
                    silent_chunks = 0
                    current_live_duration = 0.0

                timeout = (
                    0.25
                    if (self.mode is None and self.current_display_volume == 0.0)
                    else AUDIO_RUNTIME["queue_timeout_seconds"]
                )
                data = self.audio_queue.get(timeout=timeout)

                # SMOOTHING
                raw_amp = float(np.sqrt(np.mean(data ** 2)))
                if raw_amp > self.current_display_volume:
                    self.current_display_volume = raw_amp
                else:
                    self.current_display_volume *= 0.85
                    if self.current_display_volume < 1e-4:
                        self.current_display_volume = 0.0

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

                    if pending_duration < AUDIO_RUNTIME["batch_min_segment_seconds"]:
                        pause_threshold_batch = AUDIO_RUNTIME["pause_threshold_short_seconds"]
                    elif pending_duration < AUDIO_RUNTIME["batch_long_segment_seconds"]:
                        progress = (
                            pending_duration - AUDIO_RUNTIME["batch_min_segment_seconds"]
                        ) / (
                            AUDIO_RUNTIME["batch_long_segment_seconds"] - AUDIO_RUNTIME["batch_min_segment_seconds"]
                        )
                        pause_threshold_batch = AUDIO_RUNTIME["pause_threshold_short_seconds"] - (
                            (AUDIO_RUNTIME["pause_threshold_short_seconds"] - AUDIO_RUNTIME["pause_threshold_long_seconds"]) * progress
                        )
                    else:
                        pause_threshold_batch = AUDIO_RUNTIME["pause_threshold_long_seconds"]

                    is_natural_break = (
                        silence_duration > pause_threshold_batch
                        and pending_duration > AUDIO_RUNTIME["batch_min_segment_seconds"]
                    )
                    is_max_duration = pending_duration > AUDIO_RUNTIME["batch_max_segment_seconds"]

                    if (is_natural_break or is_max_duration) and not self.stopping:
                        self._queue_batch_segment()
                elif self.mode in ("online_transcribe", "online_transcribe_edit"):
                    self.online_pending_audio.append(data)
                    online_pending_duration = len(self.online_pending_audio) * AUDIO_RUNTIME["chunk_duration_seconds"]

                    if raw_amp < self.config['silence_threshold']:
                        self.online_silence_count += 1
                    else:
                        self.online_silence_count = 0

                    online_silence_duration = self.online_silence_count * AUDIO_RUNTIME["chunk_duration_seconds"]

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

                    if (is_natural_break or is_max_duration) and not self.stopping:
                        self._queue_online_segment()
                elif self.mode == "live":
                    with self.live_buffer_lock:
                        self.live_buffer.append(data)
                        self.live_backup_buffer.append(data)
                        if len(self.live_backup_buffer) > 3000:
                            self.live_backup_buffer = self.live_backup_buffer[-3000:]
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

                    if (
                        (is_silence or is_too_long)
                        and current_live_duration > AUDIO_RUNTIME["live_min_chunk_seconds"]
                        and not self.stopping
                    ):
                        with self.live_buffer_lock:
                            buffer_to_process = self.live_buffer.copy()
                            self.live_buffer = []
                        if (self.model is not None or self.backend is not None) and buffer_to_process:
                            self._live_chunk_queue.put(buffer_to_process)
                        current_live_duration = 0.0
                        silent_chunks = 0

            except queue.Empty:
                self.current_display_volume *= 0.85
                if self.current_display_volume < 1e-4:
                    self.current_display_volume = 0.0
            except Exception as e:
                # Never die silently - this loop is the audio engine
                self.log_internal(f"Audio processing error: {e}")

    def _live_transcribe_loop(self):
        """Dedicated FIFO worker thread for live-mode chunks so processing_loop
        never blocks mic draining or VU smoothing during inference."""
        while self.running and not self._shutting_down:
            try:
                chunk = self._live_chunk_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if chunk is None:
                    return
                self.process_live_chunk(chunk)
            except Exception as e:
                self.log_internal(f"Live worker error: {e}")
            finally:
                self._live_chunk_queue.task_done()

    _LIVE_HALLUCINATIONS = frozenset({
        "thank you.",
        "thank you",
        "thanks for watching.",
        "thanks for watching!",
        "you",
        ".",
        "...",
        "vielen dank.",
        "vielen dank",
        "danke.",
        "tschüss.",
        "untertitel der amara.org-community",
    })

    def process_live_chunk(self, audio_data):
        if not audio_data:
            return
        audio_np = np.concatenate(audio_data, axis=0).flatten()

        # --- TRACK DURATION ---
        duration = len(audio_np) / AUDIO_RUNTIME["sample_rate"]
        self.total_audio_duration += duration
        self.today_audio_duration += duration
        # ----------------------

        temp_path = None
        try:
            lang = self._local_language_arg()
            with self._model_lock:
                self._reload_model_if_needed_locked()
                if not self.model and not self.backend:
                    return

                start_time = time.perf_counter()

                # Zero-disk in-memory inference whenever supported by the backend
                if self.backend and hasattr(self.backend, 'transcribe_audio'):
                    segments = list(self.backend.transcribe_audio(
                        audio_np,
                        sample_rate=AUDIO_RUNTIME["sample_rate"],
                        beam_size=1,
                        vad_filter=True,
                        language=lang,
                    ))
                    text = "".join([s.text for s in segments]).strip()
                elif self.backend and hasattr(self.backend, 'transcribe'):
                    temp_path = self._temp_audio_path("live")
                    wav.write(temp_path, AUDIO_RUNTIME["sample_rate"], audio_np)
                    segments = list(self.backend.transcribe(temp_path, beam_size=1))
                    text = "".join([s.text for s in segments]).strip()
                else:
                    temp_path = self._temp_audio_path("live")
                    wav.write(temp_path, AUDIO_RUNTIME["sample_rate"], audio_np)
                    raw_segments, _ = self.model.transcribe(
                        temp_path, beam_size=1, vad_filter=True, language=lang
                    )
                    segments = list(raw_segments)
                    text = "".join([s.text for s in segments]).strip()

            latency_ms = (time.perf_counter() - start_time) * 1000
            self.session_latencies.append(latency_ms)
            self.total_latencies.append(latency_ms)

            if not text or text.lower().strip() in self._LIVE_HALLUCINATIONS:
                return

            self._live_session_texts.append(text)
            self.log_internal(f"🎤 LIVE ⚡{latency_ms:.0f}ms")
            self.last_transcription_latency = latency_ms
            ac_meta = (
                lexical_insights.extract_acoustic_meta(
                    duration, segments, words_count=len(tokenize_words(text))
                )
                if (lexical_insights is not None and duration > 0)
                else None
            )
            self._push_pending_acoustic_meta(text, ac_meta, latency_ms=latency_ms)
            self.result_queue.put(f"[LIVE] {text}")
            if keyboard is not None:
                try:
                    keyboard.write(text + " ")
                except Exception as type_err:
                    self.log_internal(f"Auto-type skipped: {type_err}")
        except Exception as e:
            self.log_internal(f"Transcription error: {e}")
        finally:
            if temp_path:
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
        executor = self.batch_executor
        if executor is not None:
            try:
                executor.submit(self._transcribe_segment, seq, segment)
            except RuntimeError:
                pass
        
        self._update_batch_progress()
        self.log_internal(f"📦 Queued segment {seq + 1} for transcription")

    def _transcribe_segment(self, seq, segment_ref=None):
        """Transcribe a single segment (runs in thread pool)"""
        segment = segment_ref
        with self.batch_segment_lock:
            if segment is None:
                for s in self.batch_segments:
                    if s['seq'] == seq:
                        segment = s
                        break
            if segment is not None:
                segment['status'] = 'transcribing'

        if not segment or (not self.model and not self.backend):
            return

        temp_file = None
        try:
            audio_np = np.concatenate(segment['audio'], axis=0).flatten()
            seg_duration = len(audio_np) / AUDIO_RUNTIME["sample_rate"]
            lang = self._local_language_arg()

            with self._model_lock:
                self._reload_model_if_needed_locked()

                start_time = time.perf_counter()

                if self.backend and hasattr(self.backend, 'transcribe_audio'):
                    segments = list(self.backend.transcribe_audio(
                        audio_np,
                        sample_rate=AUDIO_RUNTIME["sample_rate"],
                        beam_size=5,
                        vad_filter=True,
                        language=lang,
                    ))
                    text = " ".join([s.text for s in segments]).strip()
                elif self.backend and hasattr(self.backend, 'transcribe'):
                    temp_file = self._temp_audio_path(f"batch_seg_{seq}")
                    wav.write(temp_file, AUDIO_RUNTIME["sample_rate"], audio_np)
                    segments = list(self.backend.transcribe(temp_file))
                    text = " ".join([s.text for s in segments]).strip()
                else:
                    temp_file = self._temp_audio_path(f"batch_seg_{seq}")
                    wav.write(temp_file, AUDIO_RUNTIME["sample_rate"], audio_np)
                    raw_segments, _ = self.model.transcribe(
                        temp_file, beam_size=5, vad_filter=True, language=lang
                    )
                    segments = list(raw_segments)
                    text = " ".join([s.text for s in segments]).strip()

            latency_ms = (time.perf_counter() - start_time) * 1000
            seg_ac_meta = (
                lexical_insights.extract_acoustic_meta(
                    seg_duration, segments, words_count=len(tokenize_words(text))
                )
                if (lexical_insights is not None and text and seg_duration > 0)
                else None
            )

            with self.batch_segment_lock:
                segment['result'] = text
                segment['status'] = 'done'
                segment['latency_ms'] = latency_ms
                segment['acoustic_meta'] = seg_ac_meta
                segment['audio'] = None  # Free the raw audio - no longer needed

            self.log_internal(f"✅ Segment {seq + 1} ⚡{latency_ms:.0f}ms")

        except Exception as e:
            with self.batch_segment_lock:
                segment['status'] = 'error'
                segment['result'] = f"[Error: {e}]"
            self.log_internal(f"❌ Segment {seq + 1} error: {e}")
        finally:
            if temp_file:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

        self._update_batch_progress()

    def _update_batch_progress(self):
        """Update UI with batch transcription progress (called from pool threads)"""
        if self.mode != "batch" and not self.stopping:
            return
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
        executor = self.online_executor
        session_backend = self.online_backend
        session_with_edit = self.online_with_edit
        if executor is not None:
            try:
                executor.submit(
                    self._transcribe_online_segment,
                    seq,
                    segment,
                    session_backend,
                    session_with_edit,
                )
            except RuntimeError:
                pass
        
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

    def _transcribe_online_segment(
        self,
        seq,
        segment_ref=None,
        session_backend=None,
        session_with_edit=None,
    ):
        """Transcribe a single segment via OpenAI API (runs in thread pool)"""
        segment = segment_ref
        with self.online_segment_lock:
            if segment is None:
                for s in self.online_segments:
                    if s['seq'] == seq:
                        segment = s
                        break
            if segment is not None:
                segment['status'] = 'transcribing'

        if not segment:
            return

        temp_file = None
        try:
            audio_np = np.concatenate(segment['audio'], axis=0).flatten()
            start_time = time.perf_counter()

            # One configured backend (one OpenAI client) per recording session
            backend = session_backend or self.online_backend or self._get_online_backend()
            with_edit = self.online_with_edit if session_with_edit is None else session_with_edit

            if with_edit:
                if hasattr(backend, 'transcribe_and_edit_audio'):
                    _raw_text, edited_text = backend.transcribe_and_edit_audio(
                        audio_np, sample_rate=AUDIO_RUNTIME["sample_rate"]
                    )
                    text = edited_text
                else:
                    temp_file = self._temp_audio_path(f"online_seg_{seq}")
                    wav.write(temp_file, AUDIO_RUNTIME["sample_rate"], audio_np)
                    _raw_text, edited_text = backend.transcribe_and_edit(temp_file)
                    text = edited_text
            else:
                if hasattr(backend, 'transcribe_audio'):
                    segments_result = backend.transcribe_audio(
                        audio_np, sample_rate=AUDIO_RUNTIME["sample_rate"]
                    )
                    text = " ".join(seg.text for seg in segments_result).strip()
                else:
                    temp_file = self._temp_audio_path(f"online_seg_{seq}")
                    wav.write(temp_file, AUDIO_RUNTIME["sample_rate"], audio_np)
                    segments_result = backend.transcribe(temp_file)
                    text = " ".join(seg.text for seg in segments_result).strip()

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
            if temp_file:
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

    def _push_pending_acoustic_meta(self, clean_text, meta, latency_ms=None):
        """Thread-safe handoff of per-utterance acoustic metadata & latency from worker threads."""
        if not clean_text or (meta is None and latency_ms is None):
            return
        with self._acoustic_meta_lock:
            self._pending_acoustic_metas.append((clean_text, meta, latency_ms))

    def _pop_pending_acoustic_meta(self, clean_text):
        """Pop the first matching ``(meta, latency_ms)`` tuple for ``clean_text``."""
        if not clean_text:
            return None, None
        with self._acoustic_meta_lock:
            for idx, item in enumerate(self._pending_acoustic_metas):
                ptxt, pmeta, plat = item if len(item) == 3 else (item[0], item[1], None)
                if ptxt == clean_text:
                    del self._pending_acoustic_metas[idx]
                    return pmeta, plat
        return None, None

    def save_transcription(self, text, latency_ms=0, acoustic_meta=None):
        """Save transcription to monthly log file, record optional acoustic sidecar, and update history."""
        if not text.strip():
            return
            
        timestamp = datetime.datetime.now()
        
        # Add to in-memory history with latency
        self.history.append((timestamp, text, latency_ms))
        self.current_transcription = text
        self.current_latency = latency_ms
        self.history_index = len(self.history) - 1  # Point to the latest

        # Update loaded LexicalDiskIndex BEFORE appending to disk so any mid-day
        # tail seed from disk only reads strictly prior entries (zero double-counting).
        if self._lexical_index is not None:
            try:
                self._lexical_index.record_live_entry(timestamp, text, acoustic_meta)
            except Exception as e:
                self.log_internal(f"Lexical index live update skipped: {e}")

        trans_dir = os.path.join(app_dir, TRANSCRIPTIONS_DIR)
        host = self._hostname()

        # Save to per-machine monthly file: each machine appends only to its own
        # file, so simultaneous use on two machines can't produce Dropbox
        # 'conflicted copy' duplicates. Readers merge all month files.
        try:
            os.makedirs(trans_dir, exist_ok=True)
            filename = f"{timestamp.strftime('%Y-%m')}.{host}.txt"
            filepath = os.path.join(trans_dir, filename)
            
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp_str}] {text}\n")
        except Exception as e:
            self.log_internal(f"Error saving transcription: {e}")

        # Forward sidecar for VAD speech ratio, internal pauses, and acoustic WPM
        if acoustic_meta and lexical_insights is not None:
            try:
                lexical_insights.append_acoustic_meta(trans_dir, host, timestamp, acoustic_meta)
            except Exception as e:
                self.log_internal(f"Acoustic metadata sidecar skipped: {e}")

        if self._lexical_index is not None:
            try:
                self._lexical_index.sync_host_watermarks(timestamp)
                self._schedule_lexical_index_save()
            except Exception:
                pass

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
        """Load recent transcription history and seed analytics cache in one pass."""
        self.history = []
        today = datetime.date.today()
        cutoff_date = today - datetime.timedelta(days=TRANSCRIPTION_HISTORY_DAYS)
        cutoff_month = cutoff_date.strftime("%Y-%m")
        cutoff_dt = datetime.datetime.combine(cutoff_date, datetime.time.min)
        trans_dir = os.path.join(app_dir, TRANSCRIPTIONS_DIR)

        try:
            if not os.path.exists(trans_dir):
                return

            all_entries = []
            scanned_files = 0
            parsed_all_files = True

            # Merge all month files: legacy shared ("YYYY-MM.txt") and
            # per-machine ("YYYY-MM.HOSTNAME.txt"); skip Dropbox conflict debris
            for filename in sorted(os.listdir(trans_dir)):
                if not filename.lower().endswith('.txt'):
                    continue
                if 'conflicted copy' in filename.lower():
                    continue
                filepath = os.path.join(trans_dir, filename)
                if not os.path.isfile(filepath):
                    continue
                scanned_files += 1
                if len(filename) >= 7 and filename[:4].isdigit() and filename[4] == '-':
                    if filename[:7] < cutoff_month:
                        parsed_all_files = False
                        continue

                try:
                    file_entries = self._parse_transcript_file(filepath)
                except Exception:
                    continue

                if parsed_all_files:
                    all_entries.extend(file_entries)

                for entry in file_entries:
                    if entry["timestamp"] >= cutoff_dt:
                        self.history.append((entry["timestamp"], entry["text"], 0))

            self.history.sort(key=lambda x: x[0])

            # Seed the analytics cache when every archive file fell within cutoff_month
            # so refresh_insights_tabs() performs zero duplicate disk reads on startup.
            if parsed_all_files:
                all_entries.sort(key=lambda x: x["timestamp"])
                self._analytics_entries_cache = all_entries
                self._analytics_scanned_files = scanned_files
                self._analytics_tokens_cache = None
                self._analytics_langs_cache = None

            # Reconcile Today's word count while preserving genuine recorded audio duration
            if self.history:
                today_text = " ".join([text for dt, text, _lat in self.history if dt.date() == today])
                archive_today_words = len(self._tokenize_words(today_text))
                if archive_today_words > 0:
                    self.today_words = max(self.today_words, archive_today_words)
                    if self.today_audio_duration <= 0.0:
                        self.today_audio_duration = (self.today_words / 130.0) * 60.0

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
        if self._shutting_down:
            return
        self.update_histogram()

    def update_histogram(self):
        """Redraw both histograms"""
        if self._shutting_down:
            return
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
        self._gui_loop_job = None
        if self._shutting_down:
            return
        loop_interval = 1000
        try:
            loop_interval = self._update_gui_loop_body()
        except Exception as e:
            self._log_once("update_gui_loop", f"update_gui_loop failed: {e!r}")
        finally:
            if not self._shutting_down:
                try:
                    self._gui_loop_job = self.root.after(loop_interval, self.update_gui_loop)
                except (tk.TclError, RuntimeError):
                    pass

    def _update_gui_loop_body(self):
        """One tick of the GUI refresh loop. Returns the next delay in ms."""
        # (Hotkey actions are drained by the dedicated 50ms _hotkey_pump loop)

        # 0. Midnight rollover check (if the app stays running overnight)
        today_iso = datetime.date.today().isoformat()
        if today_iso != self.session_date:
            try:
                self.flush_stats()
            except Exception:
                pass
            self.session_date = today_iso
            self.today_words = 0
            self.today_audio_duration = 0.0
            self.session_latencies = []
            self._hourly_merge_cache = None
            self._schedule_histogram_refresh(delay_ms=200)

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
                ac_meta, item_latency = self._pop_pending_acoustic_meta(clean_text)
                eff_latency = self.last_transcription_latency if item_latency is None else item_latency
                self.save_transcription(
                    clean_text,
                    eff_latency,
                    acoustic_meta=ac_meta,
                )
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

        # --- STATISTICS (cached so idle ticks perform zero redundant Tk redraws) ---
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

        cache = self._ui_render_cache

        alltime_tuple = (
            f"{alltime_words:,}",
            f"{self.format_time_saved(alltime_mins_saved)}",
            f"{self.format_speaking_time(alltime_audio_duration)}",
            f"{alltime_speed} WPM",
        )
        if cache.get("alltime_tuple") != alltime_tuple:
            cache["alltime_tuple"] = alltime_tuple
            self.lbl_alltime_words.config(text=alltime_tuple[0])
            self.lbl_alltime_time.config(text=alltime_tuple[1])
            self.lbl_alltime_duration.config(text=alltime_tuple[2])
            self.lbl_alltime_speed.config(text=alltime_tuple[3])

        status_stats_txt = (
            f"{avg_latency / 1000.0:.1f} s · {today_spoken_mins:.0f} min today"
            f" · saved {self.format_time_saved(today_mins_saved)}"
        )
        if cache.get("status_stats") != status_stats_txt:
            cache["status_stats"] = status_stats_txt
            self.status_stats_var.set(status_stats_txt)

        summary_key = (self.today_words, session_speed)
        if cache.get("insights_summary") != summary_key:
            cache["insights_summary"] = summary_key
            self._update_insights_summary(self.today_words, session_speed)

        self._update_state_card()

        # 3. VU meter (quantized to 0.001 so idle ticks never trigger CTk canvas redraws)
        display_val = min(1.0, self.current_display_volume * AUDIO_RUNTIME["live_buffer_gain"])
        vu_val = round(display_val, 3)
        if cache.get("vu_val") != vu_val:
            cache["vu_val"] = vu_val
            self.vu_meter.set(display_val)

        # 4. Mini window
        if self.mini_window and self.mini_canvas:
            x0 = getattr(self, "_mini_vu_x0", 27)
            x1 = getattr(self, "_mini_vu_x1", UI_RUNTIME["mini_window_width"] - 68)
            frac = min(1.0, display_val * AUDIO_RUNTIME["mini_vu_multiplier"])
            frac_q = round(frac, 3)

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

            stats1_txt = f"{self.today_words:,} w"
            stats2_txt = f"{self.format_speaking_time(self.today_audio_duration)}"
            mini_key = (id(self.mini_canvas), frac_q, status_txt, stats1_txt, stats2_txt)
            if cache.get("mini_state") != mini_key:
                cache["mini_state"] = mini_key
                self.mini_canvas.coords(self.mini_vu, x0, 31, x0 + (x1 - x0) * frac, 35)
                self.mini_canvas.itemconfigure("status_text", text=status_txt)
                self.mini_canvas.itemconfigure("stats_text", text=stats1_txt)
                self.mini_canvas.itemconfigure("stats_text2", text=stats2_txt)

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
        self._model_load_failed = False
        self._model_loading = True
        self.status_var.set("Reloading model...")
        self._update_state_card()
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
        """Make a failed model load impossible to miss."""
        self._model_load_failed = True
        message = str(error)
        log.error("Model load failed: %s", message, exc_info=True)
        try:
            self.log_internal(f"❌ Model load failed: {message}")
        except Exception:
            pass

        def _show_err_state():
            self.status_var.set("Model load failed – see System Log")
            self._ui_render_cache["state_dot"] = COLOR_LIVE
            self._set_state_dot(COLOR_LIVE)
            self._update_state_card()

        self._ui_after(0, _show_err_state)
        if getattr(self, "_model_error_dialog_shown", False) or NO_DIALOGS:
            return
        self._model_error_dialog_shown = True
        detail = (
            f"neurowhisper could not load the speech model.\n\n{message}\n\n"
            "Check your internet connection for the first download, then use "
            "Reload Model."
        )
        self._ui_after(0, lambda: messagebox.showerror("Model load failed", detail))

    @staticmethod
    def _sanitize_no_proxy():
        """Strip brackets around bare IPv6 host entries (e.g. ``[::1]`` -> ``::1``)
        so ``httpx`` does not raise ``InvalidURL``."""
        for key in ("NO_PROXY", "no_proxy"):
            raw = os.environ.get(key)
            if not raw or "[" not in raw:
                continue
            parts = []
            seen = set()
            for item in raw.split(","):
                tok = item.strip()
                if not tok:
                    continue
                if "://" not in tok and tok.startswith("[") and tok.endswith("]"):
                    tok = tok[1:-1]
                if tok not in seen:
                    seen.add(tok)
                    parts.append(tok)
            updated = ",".join(parts)
            if updated != raw:
                os.environ[key] = updated

    @staticmethod
    def _resolve_faster_whisper_repo_id(model_id):
        if "/" in model_id:
            return model_id
        try:
            from faster_whisper.utils import _MODELS as fw_models
            if model_id in fw_models:
                return fw_models[model_id]
        except Exception:
            pass
        return f"systran/faster-whisper-{model_id}"

    @staticmethod
    def _resolve_local_ct2_dir(target_dir):
        """Locate ``model.bin`` either directly in ``target_dir`` or inside a HuggingFace snapshot."""
        if not os.path.isdir(target_dir):
            return None
        direct = os.path.join(target_dir, "model.bin")
        if os.path.isfile(direct):
            return target_dir
        best = None
        best_mtime = -1.0
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if "model.bin" in files:
                cand = os.path.join(root, "model.bin")
                try:
                    mt = os.path.getmtime(cand)
                except OSError:
                    mt = 0.0
                if mt >= best_mtime:
                    best_mtime = mt
                    best = root
        return best

    def _load_model_locked(self):
        legacy_device = None
        legacy_compute_type = None
        self._model_loading = True
        self._model_load_failed = False
        self._ui_after(0, self._update_state_card)
        try:
            friendly_name = self.config['model_key']
            model_id = MODEL_MAP.get(friendly_name, "medium")

            # --- 1. REFRESH DROPDOWN INDICATORS ---
            self._ui_after(0, self.refresh_model_list)

            self.log_internal(f"Init {friendly_name}...")

            safe_name = friendly_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            local_models_root = os.path.join(app_dir, "models")
            target_dir = os.path.join(local_models_root, safe_name)

            # --- 2. DETERMINE BACKEND ---
            backend_name = self.config.get('backend', 'auto')
            device = self.config.get('device', 'auto')
            compute_type = self.config.get('compute_type', 'auto')
            if backend_name == 'cuda':
                backend_name = 'faster-whisper'
                if device in ('auto', '', None):
                    device = 'cuda'

            model_id_low = model_id.lower()
            model_id_tail = model_id_low.split("/")[-1]
            is_onnx_asr_model = (model_id_low in ONNX_ASR_MODEL_IDS) or (model_id_tail in ONNX_ASR_MODEL_IDS)
            is_parakeet_only_model = (
                ('parakeet' in model_id_low or 'nvidia/parakeet' in model_id_low)
                and not is_onnx_asr_model
            )

            if BACKENDS_AVAILABLE and backend_name != 'legacy':
                if is_onnx_asr_model:
                    if backend_name == 'parakeet':
                        parakeet_ok = False
                        try:
                            parakeet_ok, _ = create_backend('parakeet').is_available
                        except Exception:
                            parakeet_ok = False
                        if not parakeet_ok:
                            self.log_internal("[Parakeet] NeMo not installed — using lightweight ONNX ASR backend")
                            backend_name = 'onnx-asr'
                    elif backend_name != 'openai':
                        backend_name = 'onnx-asr'
                        self.log_internal("[ONNX ASR model detected] Using ONNX ASR backend")

                    if device == 'auto':
                        if backend_name == 'parakeet':
                            try:
                                import torch
                                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            except ImportError:
                                device = 'cpu'
                        else:
                            _, detected_device, _ = detect_best_backend()
                            device = (
                                detected_device
                                if detected_device in ('cuda', 'dml', 'coreml', 'cpu')
                                else 'cpu'
                            )
                elif is_parakeet_only_model:
                    backend_name = 'parakeet'
                    self.log_internal("[Parakeet model detected] Using Parakeet backend")
                    if device == 'auto':
                        try:
                            import torch
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        except ImportError:
                            device = 'cpu'
                elif backend_name == 'parakeet':
                    parakeet_ok = False
                    try:
                        parakeet_ok, _ = create_backend('parakeet').is_available
                    except Exception:
                        parakeet_ok = False
                    if not parakeet_ok:
                        self.log_internal("[Parakeet] NeMo not installed — falling back to ONNX ASR backend")
                        backend_name = 'onnx-asr'
                        if device == 'auto':
                            _, detected_device, _ = detect_best_backend()
                            device = (
                                detected_device
                                if detected_device in ('cuda', 'dml', 'coreml', 'cpu')
                                else 'cpu'
                            )
                elif backend_name == 'auto' or device == 'auto':
                    detected_backend, detected_device, reason = detect_best_backend()
                    if backend_name == 'auto':
                        backend_name = detected_backend
                        if detected_device and detected_device != device:
                            self.log_internal(
                                f"[Auto-detect] device {device} -> {detected_device} "
                                f"(follows the auto-selected {detected_backend} backend)"
                            )
                        device = detected_device
                        compute_type = 'auto'
                    elif device == 'auto':
                        if backend_name == 'onnx-asr' and detected_device not in ('cuda', 'dml', 'coreml', 'cpu'):
                            device = 'cpu'
                        else:
                            device = detected_device
                    self.log_internal(f"[Auto-detect] {reason}")

                # Release any previously active backend (and worker subprocess / GPU memory)
                if unload_all_backends is not None:
                    try:
                        unload_all_backends(exclude=backend_name)
                    except Exception as unload_err:
                        log.warning("Could not unload previous backends: %s", unload_err)

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

                if backend_name != ONLINE_BACKEND_NAME and not self.is_model_downloaded(friendly_name):
                    download_text = self._download_status_text(model_id)
                    self.log_internal(download_text)
                    self._ui_after(0, lambda: self._show_progress("Downloading..."))
                    self._ui_after(0, lambda t=download_text: self.status_var.set(t))
                    self._ui_after(0, self._update_state_card)

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
                        model_path=target_dir,
                    )

                    # For compatibility with existing transcription code
                    self.model = self.backend.model if hasattr(self.backend, 'model') else self.backend
                    self._model_load_failed = False

                    self._ui_after(0, self._hide_progress)
                    self._ui_after(0, self.refresh_model_list)
                    self.log_internal(f"Model Ready. [{backend_name.upper()} / {device.upper()}]")
                    self._ui_after(0, lambda: self.status_var.set(f"Ready [{backend_name} / {device}]"))
                    self._ui_after(0, self._update_device_model_labels)
                    return
                except Exception as backend_error:
                    self.log_internal(f"⚠️ {backend_name} failed: {backend_error}")
                    err_low = str(backend_error).lower()

                    if is_parakeet_only_model:
                        self._model_load_failed = True
                        self.model = None
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
                        self.backend = None
                        self.model = None
                        self._model_load_failed = False
                        self._ui_after(0, self._hide_progress)
                        self._ui_after(0, lambda: self.status_var.set("Online mode - set OpenAI API key in Configuration"))
                        return

                    # Do not fall through to legacy faster-whisper when loading an ONNX-only model
                    # or when CPU/offline failure already occurred cleanly on the primary backend.
                    is_offline_or_missing = any(
                        phrase in err_low
                        for phrase in (
                            "is not downloaded yet",
                            "no internet connection",
                            "offline mode",
                            "local_files_only",
                        )
                    )
                    if is_onnx_asr_model or backend_name == 'onnx-asr' or is_offline_or_missing or device == 'cpu':
                        self.model = None
                        self._ui_after(0, self._hide_progress)
                        self._report_model_load_failure(backend_error)
                        return

                    self.log_internal("Falling back to CPU (int8)")
                    log.warning(
                        "Backend %s failed (%s); falling back to CPU (int8)",
                        backend_name,
                        backend_error,
                    )
                    legacy_device = 'cpu'
                    legacy_compute_type = 'int8'
                    self._ui_after(0, self._hide_progress)

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

            self._sanitize_no_proxy()
            offline_env = any(
                os.environ.get(var, "").strip().lower() in ("1", "true", "yes")
                for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
            )

            if not self.is_model_downloaded(friendly_name):
                self.migrate_from_cache(model_id, target_dir)

            if not self.is_model_downloaded(friendly_name) and offline_env:
                raise RuntimeError(
                    f"Model '{friendly_name}' is not downloaded yet and offline mode is enabled."
                )

            snapshot_download = _get_snapshot_download()
            if not self.is_model_downloaded(friendly_name) and snapshot_download:
                download_text = self._download_status_text(model_id)
                self.log_internal(download_text)

                def show_progress():
                    self.progress_bar.configure(mode="indeterminate")
                    self.progress_frame.pack(fill="x", padx=5, pady=0)
                    self.progress_bar.start(15)
                    self.progress_label.configure(text="Downloading...")

                def hide_progress():
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate", value=0)
                    self.progress_frame.pack_forget()

                self._ui_after(0, show_progress)
                self._ui_after(0, lambda t=download_text: self.status_var.set(t))
                self._ui_after(0, self._update_state_card)

                try:
                    repo_id = self._resolve_faster_whisper_repo_id(model_id)
                    snapshot_download(repo_id=repo_id, local_dir=target_dir)
                finally:
                    self._ui_after(0, hide_progress)

            os.makedirs(target_dir, exist_ok=True)
            resolved_local_dir = self._resolve_local_ct2_dir(target_dir)
            from faster_whisper import WhisperModel

            if resolved_local_dir is not None:
                self.log_internal("Loading model from local path...")
                self.model = WhisperModel(
                    resolved_local_dir,
                    device=legacy_device,
                    compute_type=legacy_compute_type,
                    local_files_only=True,
                )
            else:
                self.model = WhisperModel(
                    model_id,
                    device=legacy_device,
                    compute_type=legacy_compute_type,
                    download_root=target_dir,
                    local_files_only=offline_env,
                )

            self._model_load_failed = False
            self._ui_after(0, self.refresh_model_list)
            self._ui_after(0, self._update_device_model_labels)

            self.log_internal("Model Ready.")
            self._ui_after(0, lambda: self.status_var.set("Ready."))
        except Exception as e:
            self._model_load_failed = True
            self.model = None
            error_str = str(e).lower()
            self.log_internal(f"Load Error: {e}")
            self._ui_after(0, lambda: self.progress_frame.pack_forget())

            is_cuda_error = any(kw in error_str for kw in CUDA_ERROR_KEYWORDS)

            if is_cuda_error and self.config.get('device') == 'cuda':
                log.error("Model load failed (CUDA): %s", e, exc_info=True)
                self._ui_after(0, lambda: self.status_var.set("Model load failed – see System Log"))
                self._ui_after(0, lambda: self._set_state_dot(COLOR_LIVE))
                self._ui_after(0, lambda err=e: self._show_cuda_error_dialog(str(err)))
            else:
                self._report_model_load_failure(e)
        finally:
            self._model_loading = False
            self._ui_after(0, self._update_state_card)

    def migrate_from_cache(self, model_id, target_dir):
        try:
            repo_id = self._resolve_faster_whisper_repo_id(model_id)
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

        # Cancel all pending Tk after() callbacks so root.destroy() never logs TclErrors
        for job_attr in (
            "_hotkey_pump_job",
            "_gui_loop_job",
            "_hotkey_watchdog_job",
            "_insights_refresh_job",
            "_lexical_save_job",
            "_histogram_job",
            "_init_histogram_job",
            "_welcome_job",
            "_auto_crunch_job",
            "_scroll_sync_job",
            "_focus_cfg_job",
        ):
            job_id = getattr(self, job_attr, None)
            if job_id is not None:
                setattr(self, job_attr, None)
                try:
                    self.root.after_cancel(job_id)
                except Exception:
                    pass

        for tip in list(getattr(self, "_active_tooltips", ())):
            try:
                tip.destroy()
            except Exception:
                pass
        if hasattr(self, "_active_tooltips"):
            self._active_tooltips.clear()

        if getattr(self, "_trends_explorer_win", None) is not None:
            try:
                self._trends_explorer_win.destroy()
            except Exception:
                pass
            self._trends_explorer_win = None
        self._explorer_range_buttons = {}
        self._explorer_period_combo = None
        self._explorer_main_canvas = None
        self._explorer_vol_canvas = None
        self._explorer_report_btn = None

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
        try:
            self._live_chunk_queue.put_nowait(None)
        except Exception:
            pass

        # Persist any debounced stats & lexical index cache right now
        try:
            self.flush_stats()
        except Exception:
            pass
        try:
            self._flush_lexical_index()
        except Exception:
            pass

        # Release the microphone
        try:
            self.stop_audio_stream()
        except Exception:
            pass

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
            if unload_all_backends is not None:
                try:
                    unload_all_backends()
                except Exception as e:
                    log.warning("unload_all_backends() during shutdown failed: %s", e)

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
