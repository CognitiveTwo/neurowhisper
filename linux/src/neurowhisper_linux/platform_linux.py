"""Linux platform integration: X11 & Wayland session detection, dual-stack global hotkeys,
terminal-aware clipboard/paste/live-typing, PortAudio feedback chimes, and XDG .desktop install.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

import numpy as np

from . import APP_ID, APP_NAME, paths
from .config import HOTKEY_RUNTIME

log = logging.getLogger("neurowhisper.platform")

# Known Linux terminal emulator WM_CLASS / app_id identifiers (lowercased)
TERMINAL_WM_CLASSES: FrozenSet[str] = frozenset({
    "gnome-terminal",
    "gnome-terminal-server",
    "org.gnome.terminal",
    "org.gnome.console",
    "kgx",
    "konsole",
    "org.kde.konsole",
    "kitty",
    "alacritty",
    "wezterm",
    "org.wezfurlong.wezterm",
    "xfce4-terminal",
    "xterm",
    "uxterm",
    "tilix",
    "terminator",
    "foot",
    "footclient",
    "ghostty",
    "com.mitchellh.ghostty",
    "sakura",
    "lxterminal",
    "mate-terminal",
    "yakuake",
    "guake",
    "urxvt",
    "rxvt",
    "st",
    "st-256color",
    "cool-retro-term",
    "hyper",
    "tabby",
    "warp",
})

MODIFIER_NAMES: FrozenSet[str] = frozenset({"ctrl", "alt", "shift", "win"})
MODIFIER_ORDER: Tuple[str, ...] = ("ctrl", "alt", "shift", "win")

# Keep a reference to the latest PortAudio chime buffer so GC never collects it mid-playback
_LAST_CHIME_BUFFER: Optional[np.ndarray] = None
_LAST_WAYLAND_CLIPBOARD_TOAST_TS: float = 0.0


# ---------------------------------------------------------------------------
# Display Session Detection & Subprocess Environment Sanitization
# ---------------------------------------------------------------------------

def detect_display_session(env: Optional[Mapping[str, str]] = None) -> str:
    """Detect active Linux display session: ``'wayland'``, ``'x11'``, or ``'headless'``."""
    environ = os.environ if env is None else env
    session_type = environ.get("XDG_SESSION_TYPE", "").strip().lower()
    wayland_display = environ.get("WAYLAND_DISPLAY", "").strip()
    x11_display = environ.get("DISPLAY", "").strip()

    if wayland_display or session_type == "wayland":
        return "wayland"
    if x11_display or session_type == "x11":
        return "x11"
    return "headless"


def _clean_host_subprocess_env() -> Dict[str, str]:
    """Return an ``os.environ`` copy safe for launching external host utilities (``xdg-open``, ``wl-copy``).

    When running inside a PyInstaller / AppImage bundle, ``LD_LIBRARY_PATH`` points at bundled
    shared libraries which can conflict with host system binaries (e.g. Firefox / GTK / Mesa).
    Restoring ``LD_LIBRARY_PATH_ORIG`` avoids ABI mismatches.
    """
    env = os.environ.copy()
    if getattr(sys, "frozen", False) or env.get("APPIMAGE"):
        orig = env.get("LD_LIBRARY_PATH_ORIG")
        if orig is not None:
            if orig:
                env["LD_LIBRARY_PATH"] = orig
            else:
                env.pop("LD_LIBRARY_PATH", None)
    return env


def open_path_or_url(target: str | os.PathLike) -> bool:
    """Open a directory, file, or URL using ``xdg-open`` (with ``webbrowser`` fallback)."""
    target_str = str(target)
    xdg_open = shutil.which("xdg-open")
    if xdg_open:
        try:
            subprocess.Popen(
                [xdg_open, target_str],
                env=_clean_host_subprocess_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except OSError as exc:
            log.debug("xdg-open failed for %s: %s", target_str, exc)
    try:
        return bool(webbrowser.open(target_str))
    except Exception as exc:
        log.warning("Failed to open %s: %s", target_str, exc)
        return False


# ---------------------------------------------------------------------------
# Key & Hotkey Normalization Helpers
# ---------------------------------------------------------------------------

def normalize_key_token(token: str) -> str:
    """Canonicalize a single hotkey token (e.g., ``control`` -> ``ctrl``, ``super`` -> ``win``)."""
    t = (token or "").strip().lower()
    aliases = {
        "control": "ctrl",
        "ctrl_l": "ctrl",
        "ctrl_r": "ctrl",
        "lctrl": "ctrl",
        "rctrl": "ctrl",
        "alt_l": "alt",
        "alt_r": "alt",
        "alt_gr": "alt",
        "iso_level3_shift": "alt",
        "option": "alt",
        "shift_l": "shift",
        "shift_r": "shift",
        "cmd": "win",
        "cmd_l": "win",
        "cmd_r": "win",
        "super": "win",
        "super_l": "win",
        "super_r": "win",
        "meta": "win",
        "windows": "win",
        "return": "enter",
        "escape": "esc",
        "prior": "page_up",
        "next": "page_down",
        "pgup": "page_up",
        "pgdn": "page_down",
    }
    return aliases.get(t, t)


def parse_hotkey_combo(hotkey_str: str) -> FrozenSet[str]:
    """Parse ``'ctrl+alt+s'`` or ``'F9'`` into a normalized ``frozenset`` of key tokens."""
    if not hotkey_str:
        return frozenset()
    parts = [normalize_key_token(p) for p in hotkey_str.split("+") if p.strip()]
    return frozenset(p for p in parts if p)


def format_hotkey_combo(tokens: Set[str] | FrozenSet[str]) -> str:
    """Format a set of normalized tokens as ``'ctrl+alt+s'`` in canonical modifier order."""
    mods = [m for m in MODIFIER_ORDER if m in tokens]
    non_mods = sorted(t for t in tokens if t not in MODIFIER_NAMES)
    return "+".join(mods + non_mods)


def hotkey_to_portal_trigger(hotkey_str: str) -> str:
    """Convert ``'ctrl+alt+s'`` / ``'f9'`` into XDG Desktop Portal trigger syntax (``'<Control><Alt>s'``, ``'F9'``)."""
    tokens = parse_hotkey_combo(hotkey_str)
    if not tokens:
        return ""
    mod_map = {
        "ctrl": "<Control>",
        "alt": "<Alt>",
        "shift": "<Shift>",
        "win": "<Super>",
    }
    prefix = "".join(mod_map[m] for m in MODIFIER_ORDER if m in tokens)
    non_mods = sorted(t for t in tokens if t not in MODIFIER_NAMES)
    if not non_mods:
        return prefix
    key = non_mods[0]
    if re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", key):
        key_part = key.upper()
    elif len(key) == 1:
        key_part = key.lower()
    else:
        special_map = {
            "space": "space",
            "tab": "Tab",
            "enter": "Return",
            "esc": "Escape",
            "backspace": "BackSpace",
            "delete": "Delete",
            "insert": "Insert",
            "home": "Home",
            "end": "End",
            "page_up": "Page_Up",
            "page_down": "Page_Down",
            "up": "Up",
            "down": "Down",
            "left": "Left",
            "right": "Right",
            "pause": "Pause",
        }
        key_part = special_map.get(key, key)
    return f"{prefix}{key_part}"


def normalize_pynput_key(key: object) -> Optional[str]:
    """Normalize a ``pynput.keyboard.Key`` or ``KeyCode`` instance into a lowercase token."""
    if key is None:
        return None

    name = getattr(key, "name", None)
    if isinstance(name, str) and name:
        lower_name = name.lower()
        if lower_name.startswith("ctrl"):
            return "ctrl"
        if lower_name.startswith("alt"):
            return "alt"
        if lower_name.startswith("shift"):
            return "shift"
        if lower_name.startswith(("cmd", "super", "meta")):
            return "win"
        if re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", lower_name):
            return lower_name
        return normalize_key_token(lower_name)

    char = getattr(key, "char", None)
    vk = getattr(key, "vk", None)

    if isinstance(char, str) and len(char) == 1:
        code = ord(char)
        # On X11, pressing Ctrl+A..Ctrl+Z produces ASCII control characters 1..26
        if 1 <= code <= 26:
            return chr(code + 96)
        if code == 32:
            return "space"
        if 33 <= code <= 126 or code >= 160:
            return char.lower()

    if isinstance(vk, int):
        # X11 keysyms XK_F1 (0xFFBE) .. XK_F24 (0xFFD5)
        if 0xFFBE <= vk <= 0xFFD5:
            return f"f{vk - 0xFFBE + 1}"
        # X11 modifier keysyms
        if vk in (0xFFE3, 0xFFE4):
            return "ctrl"
        if vk in (0xFFE9, 0xFFEA, 0xFE03):
            return "alt"
        if vk in (0xFFE1, 0xFFE2):
            return "shift"
        if vk in (0xFFEB, 0xFFEC):
            return "win"
        if vk == 0x0020:
            return "space"
        if 0x0021 <= vk <= 0x007E:
            return chr(vk).lower()

    return None


# ---------------------------------------------------------------------------
# Terminal Detection, Clipboard, Smart Paste & Live Typing
# ---------------------------------------------------------------------------

def matches_terminal_wm_class(wm_class_raw: str) -> bool:
    """Return ``True`` if an ``xprop WM_CLASS`` or ``xdotool`` class string matches a known terminal."""
    if not wm_class_raw:
        return False
    quoted = re.findall(r'"([^"]+)"', wm_class_raw)
    candidates = quoted if quoted else [wm_class_raw.strip()]
    for item in candidates:
        cleaned = item.strip().lower()
        if not cleaned:
            continue
        if cleaned in TERMINAL_WM_CLASSES:
            return True
        # Handle reverse-DNS suffixes or wrapper names like 'gnome-terminal-server'
        base_parts = re.split(r"[\s/,]+", cleaned)
        for part in base_parts:
            if part in TERMINAL_WM_CLASSES:
                return True
    return False


def get_active_x11_wm_class(timeout: float = 0.25) -> str:
    """Query the active X11 window's ``WM_CLASS`` via ``xdotool`` or ``xprop``."""
    if not os.environ.get("DISPLAY"):
        return ""
    env = _clean_host_subprocess_env()

    xdotool = shutil.which("xdotool")
    if xdotool:
        try:
            proc = subprocess.run(
                [xdotool, "getactivewindow", "getwindowclassname"],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    xprop = shutil.which("xprop")
    if xprop:
        try:
            root_proc = subprocess.run(
                [xprop, "-root", "_NET_ACTIVE_WINDOW"],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if root_proc.returncode == 0:
                match = re.search(r"0x[0-9a-fA-F]+", root_proc.stdout)
                if match and match.group(0).lower() != "0x0":
                    win_id = match.group(0)
                    cls_proc = subprocess.run(
                        [xprop, "-id", win_id, "WM_CLASS"],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    if cls_proc.returncode == 0 and "WM_CLASS" in cls_proc.stdout:
                        return cls_proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    return ""


def is_active_window_terminal() -> bool:
    """Return ``True`` when the focused X11/XWayland window is a known Linux terminal emulator."""
    return matches_terminal_wm_class(get_active_x11_wm_class())


def get_paste_shortcut(is_terminal: Optional[bool] = None) -> str:
    """Return ``'ctrl+shift+v'`` for terminals and ``'ctrl+v'`` for standard GUI apps."""
    term = is_active_window_terminal() if is_terminal is None else bool(is_terminal)
    return "ctrl+shift+v" if term else "ctrl+v"


def copy_to_clipboard(text: str, root=None) -> bool:
    """Copy ``text`` to the Wayland/X11 system clipboard (plus Tk clipboard sync)."""
    if text is None:
        return False

    tk_ok = False
    if root is not None:
        try:
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update_idletasks()
            tk_ok = True
        except Exception as exc:
            log.debug("Tk clipboard sync failed: %s", exc)

    env = _clean_host_subprocess_env()
    session = detect_display_session(env)
    payload = text.encode("utf-8")

    if session == "wayland":
        wl_copy = shutil.which("wl-copy")
        if wl_copy:
            try:
                # Redirect stdout/stderr to DEVNULL so wl-copy's forked daemon does not hold pipes open
                proc = subprocess.run(
                    [wl_copy],
                    input=payload,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.5,
                )
                if proc.returncode == 0:
                    return True
            except (OSError, subprocess.SubprocessError) as exc:
                log.debug("wl-copy failed: %s", exc)

    if env.get("DISPLAY"):
        xclip = shutil.which("xclip")
        if xclip:
            try:
                proc = subprocess.run(
                    [xclip, "-selection", "clipboard"],
                    input=payload,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.5,
                )
                if proc.returncode == 0:
                    return True
            except (OSError, subprocess.SubprocessError) as exc:
                log.debug("xclip failed: %s", exc)

        xsel = shutil.which("xsel")
        if xsel:
            try:
                proc = subprocess.run(
                    [xsel, "--clipboard", "--input"],
                    input=payload,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.5,
                )
                if proc.returncode == 0:
                    return True
            except (OSError, subprocess.SubprocessError) as exc:
                log.debug("xsel failed: %s", exc)

    return tk_ok


def send_desktop_notification(title: str, body: str) -> bool:
    """Send a non-blocking desktop notification via ``notify-send`` when available."""
    notify_send = shutil.which("notify-send")
    if not notify_send:
        return False
    try:
        subprocess.Popen(
            [
                notify_send,
                "-a",
                APP_NAME,
                "-i",
                "audio-input-microphone",
                "-t",
                "2500",
                title,
                body,
            ],
            env=_clean_host_subprocess_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except OSError:
        return False


def _maybe_notify_wayland_clipboard_only() -> None:
    global _LAST_WAYLAND_CLIPBOARD_TOAST_TS
    now = time.monotonic()
    if now - _LAST_WAYLAND_CLIPBOARD_TOAST_TS >= 8.0:
        _LAST_WAYLAND_CLIPBOARD_TOAST_TS = now
        send_desktop_notification(
            APP_NAME,
            "Copied transcription to clipboard (press Ctrl+V to paste).",
        )


def _x11_send_paste_keys(use_shift: bool, env: Dict[str, str]) -> bool:
    """Synthesize ``Ctrl+Shift+V`` or ``Ctrl+V`` on X11 via ``pynput`` or ``xdotool``."""
    try:
        from pynput import keyboard as pynput_kb

        ctrl = pynput_kb.Controller()
        ctrl.press(pynput_kb.Key.ctrl)
        if use_shift:
            ctrl.press(pynput_kb.Key.shift)
        ctrl.press("v")
        ctrl.release("v")
        if use_shift:
            ctrl.release(pynput_kb.Key.shift)
        ctrl.release(pynput_kb.Key.ctrl)
        return True
    except Exception as exc:
        log.debug("pynput paste failed: %s", exc)

    xdotool = shutil.which("xdotool")
    if xdotool:
        combo = "ctrl+shift+v" if use_shift else "ctrl+v"
        try:
            proc = subprocess.run(
                [xdotool, "key", "--clearmodifiers", combo],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.5,
            )
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("xdotool paste failed: %s", exc)

    return False


def paste_from_clipboard(root=None) -> bool:
    """Paste clipboard contents into the active window (using ``Ctrl+Shift+V`` for terminals)."""
    time.sleep(0.02)
    env = _clean_host_subprocess_env()
    session = detect_display_session(env)
    use_shift = get_paste_shortcut() == "ctrl+shift+v"

    if session == "wayland":
        wtype = shutil.which("wtype")
        if wtype:
            cmd = (
                [wtype, "-M", "ctrl", "-M", "shift", "-k", "v", "-m", "shift", "-m", "ctrl"]
                if use_shift
                else [wtype, "-M", "ctrl", "-k", "v", "-m", "ctrl"]
            )
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.5,
                )
                if proc.returncode == 0:
                    return True
            except (OSError, subprocess.SubprocessError) as exc:
                log.debug("wtype paste failed: %s", exc)

        ydotool = shutil.which("ydotool")
        if ydotool:
            # Linux input event keycodes: 29=LEFTCTRL, 42=LEFTSHIFT, 47=V
            seq = (
                ["29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]
                if use_shift
                else ["29:1", "47:1", "47:0", "29:0"]
            )
            try:
                proc = subprocess.run(
                    [ydotool, "key", *seq],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.5,
                )
                if proc.returncode == 0:
                    return True
            except (OSError, subprocess.SubprocessError) as exc:
                log.debug("ydotool paste failed: %s", exc)

        # Clean fallback: leave transcription on Wayland clipboard and notify user gently
        _maybe_notify_wayland_clipboard_only()
        return False

    if session == "x11":
        return _x11_send_paste_keys(use_shift=use_shift, env=env)

    return False


def type_text(text: str, root=None) -> bool:
    """Type dictated text (plus trailing space) into the active window on Wayland or X11."""
    if not text:
        return False
    payload = text if text.endswith((" ", "\n", "\t")) else f"{text} "
    env = _clean_host_subprocess_env()
    session = detect_display_session(env)

    if session == "wayland":
        wtype = shutil.which("wtype")
        if wtype:
            try:
                proc = subprocess.run(
                    [wtype, "--", payload],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                )
                if proc.returncode == 0:
                    return True
            except (OSError, subprocess.SubprocessError) as exc:
                log.debug("wtype text failed: %s", exc)

        ydotool = shutil.which("ydotool")
        if ydotool:
            try:
                proc = subprocess.run(
                    [ydotool, "type", "--", payload],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                )
                if proc.returncode == 0:
                    return True
            except (OSError, subprocess.SubprocessError) as exc:
                log.debug("ydotool type failed: %s", exc)

        copy_to_clipboard(payload, root=root)
        return paste_from_clipboard(root=root)

    if session == "x11":
        try:
            from pynput import keyboard as pynput_kb

            pynput_kb.Controller().type(payload)
            return True
        except Exception as exc:
            log.debug("pynput type_text failed: %s", exc)

        xdotool = shutil.which("xdotool")
        if xdotool:
            try:
                proc = subprocess.run(
                    [xdotool, "type", "--clearmodifiers", "--delay", "5", "--", payload],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=4.0,
                )
                if proc.returncode == 0:
                    return True
            except (OSError, subprocess.SubprocessError) as exc:
                log.debug("xdotool type failed: %s", exc)

        copy_to_clipboard(payload, root=root)
        return paste_from_clipboard(root=root)

    return False


# ---------------------------------------------------------------------------
# Linux Audio Feedback Chimes (PortAudio + PipeWire/PulseAudio fallback)
# ---------------------------------------------------------------------------

def synthesize_chime(
    start: bool = True,
    sample_rate: int = 44100,
    duration_ms: int = 120,
) -> np.ndarray:
    """Synthesize a smooth, click-free mono float32 sine-wave chime (800 Hz start / 400 Hz stop)."""
    n_samples = max(64, int(sample_rate * (duration_ms / 1000.0)))
    freq = 800.0 if start else 400.0
    t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
    wave = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

    # Smooth raised-cosine (Hann) fade-in/fade-out over 20% of each edge (eliminates speaker clicks)
    fade_len = max(16, int(n_samples * 0.2))
    envelope = np.ones(n_samples, dtype=np.float32)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len, dtype=np.float32) / float(fade_len)))
    envelope[:fade_len] = ramp
    envelope[-fade_len:] = ramp[::-1]

    return (wave * envelope * 0.22).astype(np.float32)


def play_feedback_sound_async(start: bool = True) -> None:
    """Play a non-blocking start (800 Hz) or stop (400 Hz) audio chime on a background thread."""
    if os.environ.get("NEUROWHISPER_NO_AUDIO_FEEDBACK", "").strip() == "1":
        return

    def _worker() -> None:
        global _LAST_CHIME_BUFFER
        try:
            import sounddevice as sd

            tone = synthesize_chime(start=start, sample_rate=44100, duration_ms=120)
            _LAST_CHIME_BUFFER = tone
            sd.play(tone, samplerate=44100, blocking=False)
            return
        except Exception as exc:
            log.debug("PortAudio feedback chime skipped (%s); trying system player", exc)

        sound_candidates = (
            "/usr/share/sounds/freedesktop/stereo/message.oga",
            "/usr/share/sounds/freedesktop/stereo/audio-volume-change.oga",
        )
        sound_file = next((p for p in sound_candidates if os.path.isfile(p)), None)
        if not sound_file:
            return
        for player in ("pw-play", "paplay"):
            exe = shutil.which(player)
            if exe:
                try:
                    subprocess.run(
                        [exe, sound_file],
                        env=_clean_host_subprocess_env(),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=1.5,
                    )
                    return
                except (OSError, subprocess.SubprocessError):
                    continue

    threading.Thread(target=_worker, name="neurowhisper-chime", daemon=True).start()


# ---------------------------------------------------------------------------
# Dual-Stack Global Hotkey Manager (X11 pynput + Wayland GlobalShortcuts Portal)
# ---------------------------------------------------------------------------

class LinuxHotkeyManager:
    """Thread-safe Linux global hotkey manager supporting both X11 and Wayland sessions.

    - **X11**: Uses a lazy ``pynput.keyboard.Listener`` normalizing modifier variants and X11
      control-character keysyms, plus live key-combination capture for the Settings recorder.
    - **Wayland**: Connects to ``org.freedesktop.portal.Desktop`` / ``GlobalShortcuts`` over the
      session D-Bus via pure-Python ``jeepney`` (calling ``CreateSession`` + ``BindShortcuts`` and
      listening for ``Activated`` signals). Degrades cleanly with zero crashes when the active
      Wayland compositor does not implement ``GlobalShortcuts``, exposing status hints for IPC
      shortcuts (``neurowhisper --toggle-live`` / ``--toggle-batch``) and optional XWayland fallback.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._paused: bool = False
        self._session: str = detect_display_session()
        self._bindings: List[Tuple[str, FrozenSet[str], Callable[[], None]]] = []
        self._action_bindings: Dict[str, Tuple[str, Callable[[], None]]] = {}
        self._pressed_keys: Set[str] = set()
        self._fired_combos: Set[FrozenSet[str]] = set()
        self._last_fire_ts: Dict[str, float] = {}

        # X11 pynput listener state
        self._x11_listener = None
        self._x11_healthy: bool = False

        # Key-capture state (for interactive "Change" hotkey buttons)
        self._capture_active: bool = False
        self._capture_seen_tokens: Set[str] = set()
        self._capture_on_update: Optional[Callable[[str], None]] = None
        self._capture_on_finish: Optional[Callable[[str], None]] = None

        # Wayland XDG GlobalShortcuts portal state
        self._portal_thread: Optional[threading.Thread] = None
        self._portal_stop = threading.Event()
        self._portal_bound: bool = False
        self._portal_status: str = ""
        self._portal_generation: int = 0

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        with self._lock:
            self._paused = bool(value)
            if self._paused:
                self._pressed_keys.clear()
                self._fired_combos.clear()

    def register_all(
        self,
        hotkeys_dict: Dict[str, Callable[[], None]],
        action_names: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Register ``{hotkey_string: callback}`` across X11 and/or Wayland.

        ``action_names`` optionally maps ``hotkey_string -> action_id`` (``'toggle_live'`` / ``'toggle_batch'``)
        for descriptive Wayland XDG GlobalShortcuts portal bindings.
        """
        with self._lock:
            self.unregister_all()
            self._session = detect_display_session()

            parsed_list: List[Tuple[str, FrozenSet[str], Callable[[], None]]] = []
            action_map: Dict[str, Tuple[str, Callable[[], None]]] = {}
            idx = 0
            for raw_hk, cb in (hotkeys_dict or {}).items():
                tokens = parse_hotkey_combo(raw_hk)
                if not tokens or not callable(cb):
                    continue
                norm_hk = format_hotkey_combo(tokens)
                parsed_list.append((norm_hk, tokens, cb))
                act_id = (action_names or {}).get(raw_hk) or (action_names or {}).get(norm_hk)
                if not act_id:
                    act_id = "toggle_live" if idx == 0 else ("toggle_batch" if idx == 1 else f"shortcut_{idx}")
                action_map[act_id] = (norm_hk, cb)
                idx += 1

            self._bindings = parsed_list
            self._action_bindings = action_map
            if not self._bindings:
                return False

            if os.environ.get("NEUROWHISPER_DISABLE_HOTKEYS", "").strip() == "1":
                self._portal_status = "Global hotkeys disabled via NEUROWHISPER_DISABLE_HOTKEYS=1"
                return False

            if self._session == "x11":
                ok = self._start_x11_listener_locked()
                self._portal_status = "X11 global hotkeys active (pynput)" if ok else "X11 hotkey listener unavailable"
                return ok

            if self._session == "wayland":
                # Start Wayland GlobalShortcuts portal client in background daemon thread
                self._portal_status = "Connecting to Wayland GlobalShortcuts portal..."
                self._start_wayland_portal_locked()
                # Also attach best-effort XWayland listener if DISPLAY is available
                if os.environ.get("DISPLAY"):
                    self._start_x11_listener_locked()
                return True

            self._portal_status = "Headless session — use 'neurowhisper --toggle-live' via CLI IPC"
            return False

    def unregister_all(self) -> None:
        """Stop active X11 and Wayland shortcut listeners."""
        with self._lock:
            self._bindings.clear()
            self._action_bindings.clear()
            self._pressed_keys.clear()
            self._fired_combos.clear()
            self._portal_generation += 1
            self._portal_stop.set()
            self._portal_bound = False
            listener = self._x11_listener
            self._x11_listener = None
            self._x11_healthy = False

        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass

    def _force_restart_listener(self) -> bool:
        """Watchdog recovery entrypoint matching the Windows ``HotkeyManager`` API."""
        with self._lock:
            if not self._bindings:
                return False
            saved = {hk: cb for hk, _, cb in self._bindings}
            saved_actions = {hk: act_id for act_id, (hk, _) in self._action_bindings.items()}
        return self.register_all(saved, action_names=saved_actions)

    def health_status(self) -> Dict[str, object]:
        """Return current hotkey backend diagnostics and Wayland shortcut guidance."""
        with self._lock:
            session = self._session
            listener_alive = bool(
                self._x11_listener is not None and getattr(self._x11_listener, "running", False)
            )
            if session == "x11":
                healthy = listener_alive and self._x11_healthy
                backend = "x11-pynput"
                manual_needed = False
            elif session == "wayland":
                healthy = self._portal_bound or listener_alive
                backend = "wayland-portal" if self._portal_bound else (
                    "wayland-xwayland+ipc" if listener_alive else "wayland-ipc"
                )
                manual_needed = not self._portal_bound
            else:
                healthy = False
                backend = "headless-ipc"
                manual_needed = False

            return {
                "session": session,
                "backend": backend,
                "healthy": healthy,
                "registered_count": len(self._bindings),
                "portal_bound": self._portal_bound,
                "portal_status": self._portal_status,
                "wayland_manual_shortcuts_needed": manual_needed,
                "shortcut_hint": (
                    "neurowhisper --toggle-live  |  neurowhisper --toggle-batch"
                ),
            }

    def start_key_capture(
        self,
        on_update: Callable[[str], None],
        on_finish: Callable[[str], None],
    ) -> Callable[[], None]:
        """Begin interactive hotkey capture; returns a ``cancel()`` callable."""
        with self._lock:
            self._capture_active = True
            self._capture_seen_tokens = set()
            self._capture_on_update = on_update
            self._capture_on_finish = on_finish
            if os.environ.get("DISPLAY") and (
                self._x11_listener is None or not getattr(self._x11_listener, "running", False)
            ):
                self._start_x11_listener_locked()

        def _cancel() -> None:
            with self._lock:
                self._capture_active = False
                self._capture_seen_tokens.clear()
                self._capture_on_update = None
                self._capture_on_finish = None

        return _cancel

    def cleanup(self) -> None:
        self.unregister_all()

    # ------------------------------------------------------------------
    # X11 pynput Listener Internals
    # ------------------------------------------------------------------

    def _start_x11_listener_locked(self) -> bool:
        try:
            from pynput import keyboard as pynput_kb

            listener = pynput_kb.Listener(
                on_press=self._on_x11_key_press,
                on_release=self._on_x11_key_release,
            )
            listener.daemon = True
            listener.start()
            self._x11_listener = listener
            self._x11_healthy = True
            return True
        except Exception as exc:
            log.warning("Could not start X11 pynput keyboard listener: %s", exc)
            self._x11_listener = None
            self._x11_healthy = False
            return False

    def _on_x11_key_press(self, key: object) -> None:
        token = normalize_pynput_key(key)
        if not token:
            return

        capture_cb: Optional[Callable[[str], None]] = None
        capture_finish_cb: Optional[Callable[[str], None]] = None
        capture_combo_str = ""
        to_fire: List[Callable[[], None]] = []

        with self._lock:
            if self._capture_active:
                if token == "esc" and not self._capture_seen_tokens:
                    self._capture_active = False
                    capture_finish_cb = self._capture_on_finish
                    self._capture_on_update = None
                    self._capture_on_finish = None
                    capture_combo_str = ""
                else:
                    self._capture_seen_tokens.add(token)
                    capture_combo_str = format_hotkey_combo(self._capture_seen_tokens)
                    capture_cb = self._capture_on_update
                    # Finish immediately once a non-modifier key is pressed!
                    if token not in MODIFIER_NAMES:
                        self._capture_active = False
                        capture_finish_cb = self._capture_on_finish
                        self._capture_on_update = None
                        self._capture_on_finish = None
            else:
                self._pressed_keys.add(token)
                if not self._paused:
                    now = time.monotonic()
                    debounce = float(HOTKEY_RUNTIME.get("debounce_seconds", 0.2))
                    for norm_hk, req_tokens, cb in self._bindings:
                        if req_tokens.issubset(self._pressed_keys):
                            # Ensure no extra modifiers are held beyond those required by the shortcut
                            held_mods = self._pressed_keys & MODIFIER_NAMES
                            req_mods = req_tokens & MODIFIER_NAMES
                            if held_mods != req_mods:
                                continue
                            if req_tokens in self._fired_combos:
                                continue
                            last_ts = self._last_fire_ts.get(norm_hk, 0.0)
                            if now - last_ts >= debounce:
                                self._last_fire_ts[norm_hk] = now
                                self._fired_combos.add(req_tokens)
                                to_fire.append(cb)

        if capture_cb is not None and capture_combo_str:
            try:
                capture_cb(capture_combo_str)
            except Exception:
                pass
        if capture_finish_cb is not None:
            try:
                capture_finish_cb(capture_combo_str)
            except Exception:
                pass
        for cb in to_fire:
            threading.Thread(target=cb, name="neurowhisper-hotkey-cb", daemon=True).start()

    def _on_x11_key_release(self, key: object) -> None:
        token = normalize_pynput_key(key)
        if not token:
            return

        finish_cb: Optional[Callable[[str], None]] = None
        combo_str = ""
        with self._lock:
            if self._capture_active and self._capture_seen_tokens:
                combo_str = format_hotkey_combo(self._capture_seen_tokens)
                self._capture_active = False
                self._capture_seen_tokens.clear()
                finish_cb = self._capture_on_finish
                self._capture_on_update = None
                self._capture_on_finish = None
            self._pressed_keys.discard(token)
            self._fired_combos = {
                combo for combo in self._fired_combos if combo.issubset(self._pressed_keys)
            }

        if finish_cb is not None:
            try:
                finish_cb(combo_str)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Wayland XDG Desktop Portal GlobalShortcuts Internals (jeepney)
    # ------------------------------------------------------------------

    def _start_wayland_portal_locked(self) -> None:
        self._portal_stop = threading.Event()
        stop_event = self._portal_stop
        self._portal_generation += 1
        gen = self._portal_generation
        action_snapshot = dict(self._action_bindings)

        thread = threading.Thread(
            target=self._wayland_portal_worker,
            args=(gen, stop_event, action_snapshot),
            name="neurowhisper-wayland-portal",
            daemon=True,
        )
        self._portal_thread = thread
        thread.start()

    def _set_portal_state(self, gen: int, bound: bool, status: str) -> None:
        with self._lock:
            if gen != self._portal_generation:
                return
            self._portal_bound = bound
            self._portal_status = status

    def _wayland_portal_worker(
        self,
        gen: int,
        stop_event: threading.Event,
        action_snapshot: Dict[str, Tuple[str, Callable[[], None]]],
    ) -> None:
        try:
            from jeepney import DBusAddress, MatchRule, MessageType, new_method_call
            from jeepney.io.blocking import open_dbus_connection
        except Exception as exc:
            self._set_portal_state(
                gen,
                False,
                f"jeepney unavailable ({exc}); bind 'neurowhisper --toggle-live' in Wayland shortcuts",
            )
            return

        portal_addr = DBusAddress(
            "/org/freedesktop/portal/desktop",
            bus_name="org.freedesktop.portal.Desktop",
            interface="org.freedesktop.portal.GlobalShortcuts",
        )
        props_addr = DBusAddress(
            "/org/freedesktop/portal/desktop",
            bus_name="org.freedesktop.portal.Desktop",
            interface="org.freedesktop.DBus.Properties",
        )
        dbus_addr = DBusAddress(
            "/org/freedesktop/DBus",
            bus_name="org.freedesktop.DBus",
            interface="org.freedesktop.DBus",
        )

        conn = None
        try:
            conn = open_dbus_connection("SESSION")

            # 1. Probe whether the active desktop portal implements GlobalShortcuts
            probe = new_method_call(
                props_addr,
                "Get",
                "ss",
                ("org.freedesktop.portal.GlobalShortcuts", "version"),
            )
            probe_reply = conn.send_and_get_reply(probe, timeout=2.0)
            if probe_reply.header.message_type == MessageType.error:
                self._set_portal_state(
                    gen,
                    False,
                    "Wayland compositor does not support GlobalShortcuts portal — bind 'neurowhisper --toggle-live' in system settings",
                )
                return

            # Subscribe to Portal Request::Response and GlobalShortcuts::Activated signals
            resp_rule = MatchRule(
                type="signal",
                interface="org.freedesktop.portal.Request",
                member="Response",
            )
            act_rule = MatchRule(
                type="signal",
                interface="org.freedesktop.portal.GlobalShortcuts",
                member="Activated",
            )
            for rule in (resp_rule, act_rule):
                add_match = new_method_call(dbus_addr, "AddMatch", "s", (rule.serialise(),))
                conn.send_and_get_reply(add_match, timeout=2.0)

            # 2. CreateSession
            req_token = f"nw_req_{uuid.uuid4().hex[:10]}"
            sess_token = f"nw_sess_{uuid.uuid4().hex[:10]}"
            create_msg = new_method_call(
                portal_addr,
                "CreateSession",
                "a{sv}",
                (
                    {
                        "handle_token": ("s", req_token),
                        "session_handle_token": ("s", sess_token),
                    },
                ),
            )
            create_reply = conn.send_and_get_reply(create_msg, timeout=3.0)
            if create_reply.header.message_type == MessageType.error:
                self._set_portal_state(
                    gen,
                    False,
                    "Wayland GlobalShortcuts CreateSession rejected — use 'neurowhisper --toggle-live'",
                )
                return
            req_path = create_reply.body[0]

            session_handle = self._await_portal_response_session_handle(conn, req_path, stop_event, timeout=5.0)
            if not session_handle:
                self._set_portal_state(
                    gen,
                    False,
                    "Wayland GlobalShortcuts session did not return a handle — use 'neurowhisper --toggle-live'",
                )
                return

            # 3. BindShortcuts for toggle_live & toggle_batch
            shortcut_defs = []
            labels = {
                "toggle_live": "Toggle Live Dictation",
                "toggle_batch": "Toggle Batch Recording",
            }
            for act_id, (norm_hk, _) in action_snapshot.items():
                opts: Dict[str, Tuple[str, str]] = {
                    "description": ("s", labels.get(act_id, act_id)),
                }
                trig = hotkey_to_portal_trigger(norm_hk)
                if trig:
                    opts["preferred_trigger"] = ("s", trig)
                shortcut_defs.append((act_id, opts))

            bind_token = f"nw_bind_{uuid.uuid4().hex[:10]}"
            bind_msg = new_method_call(
                portal_addr,
                "BindShortcuts",
                "oa(sa{sv})sa{sv}",
                (
                    session_handle,
                    shortcut_defs,
                    "",
                    {"handle_token": ("s", bind_token)},
                ),
            )
            bind_reply = conn.send_and_get_reply(bind_msg, timeout=5.0)
            if bind_reply.header.message_type == MessageType.error:
                self._set_portal_state(
                    gen,
                    False,
                    "Wayland GlobalShortcuts BindShortcuts unsupported — bind 'neurowhisper --toggle-live' in system settings",
                )
                return

            self._set_portal_state(
                gen,
                True,
                "Wayland GlobalShortcuts portal active",
            )
            log.info("Registered Wayland GlobalShortcuts via XDG Desktop Portal")

            # 4. Event loop watching for GlobalShortcuts::Activated signals
            while not stop_event.is_set():
                try:
                    msg = conn.receive(timeout=0.5)
                except TimeoutError:
                    continue
                except Exception as exc:
                    if "timed out" in str(exc).lower():
                        continue
                    raise

                hdr = msg.header
                if (
                    hdr.message_type == MessageType.signal
                    and hdr.fields.get(2) == "org.freedesktop.portal.GlobalShortcuts"
                    and hdr.fields.get(3) == "Activated"
                ):
                    # Signal body: (o session_handle, s shortcut_id, t timestamp, a{sv} options)
                    if len(msg.body) >= 2:
                        shortcut_id = str(msg.body[1])
                        entry = action_snapshot.get(shortcut_id)
                        if entry is not None and not self.paused:
                            norm_hk, cb = entry
                            now = time.monotonic()
                            debounce = float(HOTKEY_RUNTIME.get("debounce_seconds", 0.2))
                            with self._lock:
                                last_ts = self._last_fire_ts.get(norm_hk, 0.0)
                                if now - last_ts < debounce:
                                    continue
                                self._last_fire_ts[norm_hk] = now
                            threading.Thread(
                                target=cb,
                                name=f"neurowhisper-portal-{shortcut_id}",
                                daemon=True,
                            ).start()
        except Exception as exc:
            log.info("Wayland GlobalShortcuts portal unavailable (%s); IPC CLI fallback ready", exc)
            self._set_portal_state(
                gen,
                False,
                "Wayland portal shortcuts unavailable — bind 'neurowhisper --toggle-live' in desktop settings",
            )
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    @staticmethod
    def _await_portal_response_session_handle(
        conn,
        expected_path: str,
        stop_event: threading.Event,
        timeout: float = 5.0,
    ) -> Optional[str]:
        from jeepney import MessageType

        deadline = time.monotonic() + timeout
        while not stop_event.is_set() and time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                msg = conn.receive(timeout=min(0.5, remaining))
            except TimeoutError:
                continue
            except Exception as exc:
                if "timed out" in str(exc).lower():
                    continue
                return None

            hdr = msg.header
            if (
                hdr.message_type == MessageType.signal
                and hdr.fields.get(2) == "org.freedesktop.portal.Request"
                and hdr.fields.get(3) == "Response"
            ):
                sig_path = str(hdr.fields.get(1, ""))
                if expected_path and sig_path and sig_path != expected_path:
                    continue
                if len(msg.body) >= 2:
                    resp_code = int(msg.body[0])
                    results = msg.body[1]
                    if resp_code != 0:
                        return None
                    if isinstance(results, dict) and "session_handle" in results:
                        val = results["session_handle"]
                        if isinstance(val, tuple) and len(val) == 2:
                            return str(val[1])
                        return str(val)
        return None


# ---------------------------------------------------------------------------
# Desktop Launcher (.desktop) & Icon Integration
# ---------------------------------------------------------------------------

def _resolve_desktop_exec_command(app_exec: Optional[str] = None) -> str:
    if app_exec and app_exec.strip():
        return app_exec.strip()
    appimage = os.environ.get("APPIMAGE", "").strip()
    if appimage:
        return shlex.quote(appimage)
    if getattr(sys, "frozen", False):
        return shlex.quote(str(Path(sys.executable).resolve()))
    which_nw = shutil.which(APP_NAME)
    if which_nw:
        return shlex.quote(which_nw)
    src_dir = Path(__file__).resolve().parents[1]
    return f"env PYTHONPATH={shlex.quote(str(src_dir))} {shlex.quote(sys.executable)} -m neurowhisper_linux.main"


def build_desktop_entry_text(exec_cmd: str, icon_value: str = APP_NAME) -> str:
    """Render a FreeDesktop-compliant ``neurowhisper.desktop`` specification with quick actions."""
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Version=1.0\n"
        f"Name={APP_NAME}\n"
        "GenericName=Voice Dictation & Speech Insights\n"
        "Comment=Local & cloud AI voice dictation with live typing, Parakeet ONNX, and Speech Insights\n"
        f"Exec={exec_cmd}\n"
        f"Icon={icon_value}\n"
        "Terminal=false\n"
        "Categories=AudioVideo;Audio;Utility;\n"
        "Keywords=whisper;dictation;speech;transcription;voice;parakeet;\n"
        f"StartupWMClass={APP_NAME}\n"
        "StartupNotify=true\n"
        "Actions=ToggleLive;ToggleBatch;StopRecording;\n"
        "\n"
        "[Desktop Action ToggleLive]\n"
        "Name=Toggle Live Dictation\n"
        f"Exec={exec_cmd} --toggle-live\n"
        "\n"
        "[Desktop Action ToggleBatch]\n"
        "Name=Toggle Batch Recording\n"
        f"Exec={exec_cmd} --toggle-batch\n"
        "\n"
        "[Desktop Action StopRecording]\n"
        "Name=Stop Recording\n"
        f"Exec={exec_cmd} --stop\n"
    )


def install_desktop_entry(
    app_exec: Optional[str] = None,
    target_data_home: Optional[Path | str] = None,
) -> Tuple[Path, Path]:
    """Install ``neurowhisper.desktop`` and ``neurowhisper.png`` into the user's XDG data directory."""
    data_home = (
        Path(target_data_home).expanduser()
        if target_data_home is not None
        else paths.get_xdg_data_home()
    )
    apps_dir = data_home / "applications"
    icons_dir = data_home / "icons" / "hicolor" / "256x256" / "apps"
    apps_dir.mkdir(parents=True, exist_ok=True)
    icons_dir.mkdir(parents=True, exist_ok=True)

    icon_src = paths.get_icon_png_path()
    icon_dest = icons_dir / f"{APP_NAME}.png"
    if icon_src.is_file():
        try:
            from PIL import Image

            with Image.open(icon_src) as img:
                resized = img.convert("RGBA").resize((256, 256), Image.Resampling.LANCZOS)
                resized.save(icon_dest, format="PNG")
        except Exception:
            shutil.copy2(icon_src, icon_dest)

    exec_cmd = _resolve_desktop_exec_command(app_exec=app_exec)
    desktop_path = apps_dir / f"{APP_NAME}.desktop"
    desktop_content = build_desktop_entry_text(exec_cmd=exec_cmd, icon_value=APP_NAME)
    desktop_path.write_text(desktop_content, encoding="utf-8")
    try:
        os.chmod(desktop_path, 0o755)
    except OSError:
        pass

    update_db = shutil.which("update-desktop-database")
    if update_db:
        try:
            subprocess.run(
                [update_db, str(apps_dir)],
                env=_clean_host_subprocess_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    log.info("Installed desktop entry at %s (icon: %s)", desktop_path, icon_dest)
    return desktop_path, icon_dest
