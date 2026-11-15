"""Native macOS integration layer (PyObjC Quartz / AppKit / ApplicationServices).

Engineered specifically for macOS Sonoma / Sequoia stability:
- Zero third-party keyboard listener libraries that call
  ``TISCopyCurrentKeyboardInputSource`` off the main queue (which crashes on
  German/QWERTZ and non-US input sources).
- Global hotkeys via Quartz ``CGEventTap`` on a dedicated ``CFRunLoop`` thread
  using pure integer virtual keycode tables (``kVK_*``).
- Clipboard via ``AppKit.NSPasteboard`` + ``Cmd+V`` via ``CGEventPost``, and
  direct Unicode typing via ``CGEventKeyboardSetUnicodeString`` chunked safely
  on UTF-16 code-unit boundaries (preserving German umlauts ``äöüß`` & emoji).
- TCC permission checks & System Settings deep-links.
- Degrades gracefully when PyObjC is absent (e.g., cross-platform Linux CI).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .config import HOTKEY_RUNTIME, PLATFORM_SOUND

logger = logging.getLogger(__name__)

try:
    import Quartz  # type: ignore
except ImportError:
    Quartz = None  # type: ignore

try:
    import AppKit  # type: ignore
except ImportError:
    AppKit = None  # type: ignore

try:
    import ApplicationServices  # type: ignore
except ImportError:
    ApplicationServices = None  # type: ignore


def _get_quartz():
    return globals().get("Quartz") or sys.modules.get("Quartz")


def _get_appkit():
    return globals().get("AppKit") or sys.modules.get("AppKit")


def _get_app_services():
    return globals().get("ApplicationServices") or sys.modules.get("ApplicationServices")


# ---------------------------------------------------------------------------
# Carbon / Quartz Virtual Keycodes (<HIToolbox/Events.h>) & Event Constants
# ---------------------------------------------------------------------------

kCGSessionEventTap = 1
kCGHIDEventTap = 0
kCGHeadInsertEventTap = 0
kCGEventTapOptionListenOnly = 1
kCGEventSourceStateHIDSystemState = 1

kCGEventKeyDown = 10
kCGEventKeyUp = 11
kCGEventFlagsChanged = 12
kCGEventTapDisabledByTimeout = 0xFFFFFFFE
kCGEventTapDisabledByUserInput = 0xFFFFFFFF

kCGKeyboardEventAutorepeat = 8

kCGEventFlagMaskShift = 0x00020000
kCGEventFlagMaskControl = 0x00040000
kCGEventFlagMaskAlternate = 0x00080000
kCGEventFlagMaskCommand = 0x00100000
kCGEventFlagMaskSecondaryFn = 0x00800000

kVK_ANSI_V = 0x09
kVK_Return = 0x24

# Canonical key name -> macOS hardware virtual keycode
KEY_NAME_TO_VK: Dict[str, int] = {
    # Letters (ANSI virtual keycodes)
    "a": 0x00,
    "s": 0x01,
    "d": 0x02,
    "f": 0x03,
    "h": 0x04,
    "g": 0x05,
    "z": 0x06,
    "x": 0x07,
    "c": 0x08,
    "v": 0x09,
    "b": 0x0B,
    "q": 0x0C,
    "w": 0x0D,
    "e": 0x0E,
    "r": 0x0F,
    "y": 0x10,
    "t": 0x11,
    "1": 0x12,
    "2": 0x13,
    "3": 0x14,
    "4": 0x15,
    "6": 0x16,
    "5": 0x17,
    "=": 0x18,
    "equal": 0x18,
    "9": 0x19,
    "7": 0x1A,
    "-": 0x1B,
    "minus": 0x1B,
    "8": 0x1C,
    "0": 0x1D,
    "]": 0x1E,
    "bracketright": 0x1E,
    "o": 0x1F,
    "u": 0x20,
    "[": 0x21,
    "bracketleft": 0x21,
    "i": 0x22,
    "p": 0x23,
    "return": 0x24,
    "enter": 0x24,
    "l": 0x25,
    "j": 0x26,
    "'": 0x27,
    "apostrophe": 0x27,
    "k": 0x28,
    ";": 0x29,
    "semicolon": 0x29,
    "\\": 0x2A,
    "backslash": 0x2A,
    ",": 0x2B,
    "comma": 0x2B,
    "/": 0x2C,
    "slash": 0x2C,
    "n": 0x2D,
    "m": 0x2E,
    ".": 0x2F,
    "period": 0x2F,
    "tab": 0x30,
    "space": 0x31,
    "`": 0x32,
    "grave": 0x32,
    "backspace": 0x33,
    "delete": 0x33,
    "escape": 0x35,
    "esc": 0x35,
    # Function keys
    "f1": 0x7A,
    "f2": 0x78,
    "f3": 0x63,
    "f4": 0x76,
    "f5": 0x60,
    "f6": 0x61,
    "f7": 0x62,
    "f8": 0x64,
    "f9": 0x65,
    "f10": 0x6D,
    "f11": 0x67,
    "f12": 0x6F,
    "f13": 0x69,
    "f14": 0x6B,
    "f15": 0x71,
    "f16": 0x6A,
    "f17": 0x40,
    "f18": 0x4F,
    "f19": 0x50,
    "f20": 0x5A,
    # Navigation
    "home": 0x73,
    "page_up": 0x74,
    "prior": 0x74,
    "forward_delete": 0x75,
    "end": 0x77,
    "page_down": 0x79,
    "next": 0x79,
    "left": 0x7B,
    "right": 0x7C,
    "down": 0x7D,
    "up": 0x7E,
    # Standalone modifier keycodes (for single-key hotkeys like Right Option)
    "right_option": 0x3D,
    "ralt": 0x3D,
    "left_option": 0x3A,
    "lalt": 0x3A,
    "right_command": 0x36,
    "rcmd": 0x36,
    "left_command": 0x37,
    "lcmd": 0x37,
    "right_shift": 0x3C,
    "left_shift": 0x38,
    "right_control": 0x3E,
    "left_control": 0x3B,
    "fn_key": 0x3F,
}

# Preferred canonical key name per virtual keycode (for reverse lookup during capture)
VK_TO_KEY_NAME: Dict[int, str] = {}
for _name, _vk in KEY_NAME_TO_VK.items():
    if _vk not in VK_TO_KEY_NAME:
        VK_TO_KEY_NAME[_vk] = _name

STANDALONE_MODIFIER_VKS: Dict[int, Tuple[str, int]] = {
    0x3D: ("right_option", kCGEventFlagMaskAlternate),
    0x3A: ("left_option", kCGEventFlagMaskAlternate),
    0x36: ("right_command", kCGEventFlagMaskCommand),
    0x37: ("left_command", kCGEventFlagMaskCommand),
    0x3C: ("right_shift", kCGEventFlagMaskShift),
    0x38: ("left_shift", kCGEventFlagMaskShift),
    0x3E: ("right_control", kCGEventFlagMaskControl),
    0x3B: ("left_control", kCGEventFlagMaskControl),
    0x3F: ("fn_key", kCGEventFlagMaskSecondaryFn),
}

MODIFIER_ALIASES: Dict[str, str] = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "ctl": "ctrl",
    "alt": "alt",
    "opt": "alt",
    "option": "alt",
    "shift": "shift",
    "cmd": "cmd",
    "command": "cmd",
    "super": "cmd",
    "meta": "cmd",
    "win": "cmd",
    "fn": "fn",
}

MODIFIER_ORDER: Tuple[str, ...] = ("ctrl", "alt", "shift", "cmd", "fn")

MODIFIER_TO_FLAG: Dict[str, int] = {
    "ctrl": kCGEventFlagMaskControl,
    "alt": kCGEventFlagMaskAlternate,
    "shift": kCGEventFlagMaskShift,
    "cmd": kCGEventFlagMaskCommand,
    "fn": kCGEventFlagMaskSecondaryFn,
}

PRIMARY_MODIFIER_MASK = (
    kCGEventFlagMaskControl
    | kCGEventFlagMaskAlternate
    | kCGEventFlagMaskShift
    | kCGEventFlagMaskCommand
)

DISPLAY_MODIFIER_NAMES: Dict[str, str] = {
    "ctrl": "Ctrl",
    "alt": "Opt",
    "shift": "Shift",
    "cmd": "Cmd",
    "fn": "Fn",
}


@dataclass(frozen=True)
class ParsedHotkey:
    """Normalized representation of a macOS global hotkey."""

    canonical: str
    keycode: int
    modifiers: FrozenSet[str]
    required_flags: int
    check_fn_flag: bool
    is_standalone_modifier: bool


def parse_hotkey_combo(hotkey_str: str) -> Optional[ParsedHotkey]:
    """Parse a hotkey string (e.g. ``'ctrl+alt+s'``, ``'f9'``, ``'right_option'``) into keycode + flags."""
    if not hotkey_str or not isinstance(hotkey_str, str):
        return None
    tokens = [tok.strip().lower() for tok in hotkey_str.replace(" ", "").split("+") if tok.strip()]
    if not tokens:
        return None

    mods: set[str] = set()
    key_token: Optional[str] = None
    for tok in tokens:
        if tok in MODIFIER_ALIASES and len(tokens) > 1:
            mods.add(MODIFIER_ALIASES[tok])
        elif tok in KEY_NAME_TO_VK:
            if key_token is not None:
                return None
            key_token = tok
        elif tok in MODIFIER_ALIASES and len(tokens) == 1:
            # Map bare 'fn' -> 'fn_key' if bound as a standalone key
            if tok == "fn":
                key_token = "fn_key"
            else:
                return None
        else:
            return None

    if key_token is None:
        return None

    vk = KEY_NAME_TO_VK[key_token]
    canonical_key = VK_TO_KEY_NAME.get(vk, key_token)
    ordered_mods = [m for m in MODIFIER_ORDER if m in mods]
    canonical = "+".join([*ordered_mods, canonical_key]) if ordered_mods else canonical_key

    req_flags = 0
    for m in ordered_mods:
        req_flags |= MODIFIER_TO_FLAG[m]

    is_mod_only = vk in STANDALONE_MODIFIER_VKS and not ordered_mods
    return ParsedHotkey(
        canonical=canonical,
        keycode=vk,
        modifiers=frozenset(ordered_mods),
        required_flags=req_flags,
        check_fn_flag=("fn" in mods),
        is_standalone_modifier=is_mod_only,
    )


def normalize_hotkey_string(hotkey_str: str) -> str:
    """Normalize a hotkey string to canonical lowercase format, or return stripped input if invalid."""
    parsed = parse_hotkey_combo(hotkey_str)
    if parsed is not None:
        return parsed.canonical
    return (hotkey_str or "").strip().lower()


def format_hotkey_label(hotkey_str: str) -> str:
    """Format a canonical hotkey string into a readable macOS UI badge (e.g. ``Ctrl+Opt+S``)."""
    parsed = parse_hotkey_combo(hotkey_str)
    raw = parsed.canonical if parsed is not None else (hotkey_str or "").strip()
    if not raw:
        return "Unassigned"
    parts = raw.split("+")
    formatted: List[str] = []
    for part in parts:
        if part in DISPLAY_MODIFIER_NAMES:
            formatted.append(DISPLAY_MODIFIER_NAMES[part])
        elif part == "right_option":
            formatted.append("Right Opt")
        elif part == "left_option":
            formatted.append("Left Opt")
        elif part == "right_command":
            formatted.append("Right Cmd")
        elif part == "left_command":
            formatted.append("Left Cmd")
        elif part == "fn_key":
            formatted.append("Fn")
        elif len(part) == 1:
            formatted.append(part.upper())
        elif part.startswith("f") and part[1:].isdigit():
            formatted.append(part.upper())
        else:
            formatted.append(part.replace("_", " ").title())
    return "+".join(formatted)


def format_tk_keypress_combo(event) -> Tuple[Optional[str], List[str]]:
    """Extract ``(canonical_combo_if_complete, preview_parts)`` from a Tk ``<KeyPress>`` event.

    Enables interactive hotkey recording inside the Settings window even before
    macOS Accessibility permission has been granted.
    """
    keysym = str(getattr(event, "keysym", "") or "")
    state = int(getattr(event, "state", 0) or 0)
    keysym_lower = keysym.lower()

    # Decode Tk modifier state bits (portable across Aqua Tk and X11 Tk)
    mods: List[str] = []
    if state & 0x0004:
        mods.append("ctrl")
    if sys.platform == "darwin":
        # Aqua Tk: Mod1 (0x0008) is Command, Mod2 (0x0010) is Option/Alt
        if state & 0x0010:
            mods.append("alt")
        if state & 0x0001:
            mods.append("shift")
        if state & 0x0008:
            mods.append("cmd")
    else:
        if state & 0x0008:
            mods.append("alt")
        if state & 0x0001:
            mods.append("shift")
        if state & 0x0040:
            mods.append("cmd")

    tk_mod_keys = {
        "control_l": "ctrl",
        "control_r": "ctrl",
        "alt_l": "alt",
        "alt_r": "alt",
        "option_l": "alt",
        "option_r": "alt",
        "shift_l": "shift",
        "shift_r": "shift",
        "super_l": "cmd",
        "super_r": "cmd",
        "meta_l": "cmd",
        "meta_r": "cmd",
    }
    if keysym_lower in tk_mod_keys:
        m = tk_mod_keys[keysym_lower]
        if m not in mods:
            mods.append(m)
        ordered = [x for x in MODIFIER_ORDER if x in mods]
        return None, [DISPLAY_MODIFIER_NAMES[x] for x in ordered]

    key_alias_map = {
        "return": "return",
        "kp_enter": "enter",
        "escape": "escape",
        "BackSpace".lower(): "backspace",
        "delete": "forward_delete",
        "space": "space",
        "tab": "tab",
        "prior": "page_up",
        "next": "page_down",
    }
    mapped_key = key_alias_map.get(keysym_lower, keysym_lower)
    if mapped_key not in KEY_NAME_TO_VK:
        char = str(getattr(event, "char", "") or "").lower()
        if char and char in KEY_NAME_TO_VK:
            mapped_key = char
        else:
            ordered = [x for x in MODIFIER_ORDER if x in mods]
            return None, [DISPLAY_MODIFIER_NAMES[x] for x in ordered]

    ordered = [x for x in MODIFIER_ORDER if x in mods]
    combo_raw = "+".join([*ordered, mapped_key]) if ordered else mapped_key
    parsed = parse_hotkey_combo(combo_raw)
    if parsed is None:
        return None, []
    preview = format_hotkey_label(parsed.canonical).split("+")
    return parsed.canonical, preview


# ---------------------------------------------------------------------------
# TCC Permission Helpers (Accessibility & Microphone)
# ---------------------------------------------------------------------------


def is_accessibility_trusted(prompt: bool = False) -> bool:
    """Check whether this process has macOS Accessibility (TCC) trust."""
    app_services = _get_app_services()
    if app_services is not None:
        try:
            if hasattr(app_services, "AXIsProcessTrustedWithOptions"):
                opt_key = getattr(
                    app_services,
                    "kAXTrustedCheckOptionPrompt",
                    "AXTrustedCheckOptionPrompt",
                )
                return bool(app_services.AXIsProcessTrustedWithOptions({opt_key: bool(prompt)}))
            if hasattr(app_services, "AXIsProcessTrusted"):
                return bool(app_services.AXIsProcessTrusted())
        except Exception as exc:
            logger.debug("AXIsProcessTrusted check failed: %s", exc)
            return False
    return sys.platform != "darwin"


def open_accessibility_settings() -> bool:
    """Open macOS System Settings -> Privacy & Security -> Accessibility."""
    url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    try:
        subprocess.Popen(
            ["/usr/bin/open", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as exc:
        logger.warning("Could not open Accessibility settings: %s", exc)
        return False


def open_microphone_settings() -> bool:
    """Open macOS System Settings -> Privacy & Security -> Microphone."""
    url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
    try:
        subprocess.Popen(
            ["/usr/bin/open", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as exc:
        logger.warning("Could not open Microphone settings: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Global Hotkey Manager (Quartz CGEventTap — Zero pynput)
# ---------------------------------------------------------------------------


class MacHotkeyManager:
    """Thread-safe macOS global hotkey manager backed by Quartz ``CGEventTap``."""

    def __init__(self, log_fn: Optional[Callable[[str], None]] = None) -> None:
        self._log_fn = log_fn
        self._lock = threading.RLock()
        self.paused: bool = False
        self._bindings: Dict[str, Tuple[ParsedHotkey, Callable[[], None]]] = {}
        self._last_trigger: Dict[str, float] = {}
        self._tap = None
        self._run_loop = None
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._tap_ok: bool = False
        self.last_error: str = ""
        self._capture_cb: Optional[Callable[[Optional[str]], None]] = None
        self._capture_update_cb: Optional[Callable[[List[str]], None]] = None

    def _log(self, msg: str) -> None:
        logger.debug(msg)
        if self._log_fn is not None:
            try:
                self._log_fn(msg)
            except Exception:
                pass

    @property
    def tap_active(self) -> bool:
        return bool(self._tap_ok and self._thread is not None and self._thread.is_alive())

    def register_all(
        self,
        hotkey_specs: Dict[str, Tuple[str, Callable[[], None]]],
    ) -> List[str]:
        """Register named global hotkeys. Returns list of hotkey names that failed to bind."""
        failed: List[str] = []
        parsed_map: Dict[str, Tuple[ParsedHotkey, Callable[[], None]]] = {}

        for name, (hk_str, callback) in hotkey_specs.items():
            parsed = parse_hotkey_combo(hk_str)
            if parsed is None:
                self._log(f"Invalid hotkey syntax for {name}: {hk_str!r}")
                failed.append(name)
            else:
                parsed_map[name] = (parsed, callback)

        with self._lock:
            self._bindings = parsed_map

        if not parsed_map:
            return failed

        if not self._ensure_tap_running():
            for name in parsed_map:
                if name not in failed:
                    failed.append(name)
        return failed

    def _ensure_tap_running(self) -> bool:
        with self._lock:
            if self.tap_active:
                return True

        quartz = _get_quartz()
        if quartz is None:
            self._tap_ok = False
            self.last_error = "quartz_unavailable"
            return sys.platform != "darwin"

        self._stop_tap_loop()
        self._ready_event.clear()
        thread = threading.Thread(
            target=self._run_event_tap_loop,
            name="neurowhisper-cgeventtap",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        self._ready_event.wait(timeout=1.5)
        return self._tap_ok

    def _run_event_tap_loop(self) -> None:
        quartz = _get_quartz()
        if quartz is None:
            self._tap_ok = False
            self.last_error = "quartz_unavailable"
            self._ready_event.set()
            return

        try:
            mask = (
                (1 << kCGEventKeyDown)
                | (1 << kCGEventKeyUp)
                | (1 << kCGEventFlagsChanged)
            )
            tap = quartz.CGEventTapCreate(
                getattr(quartz, "kCGSessionEventTap", kCGSessionEventTap),
                getattr(quartz, "kCGHeadInsertEventTap", kCGHeadInsertEventTap),
                getattr(quartz, "kCGEventTapOptionListenOnly", kCGEventTapOptionListenOnly),
                mask,
                self._event_tap_callback,
                None,
            )
            if tap is None:
                self._tap_ok = False
                self.last_error = "accessibility_denied"
                self._log("CGEventTapCreate returned None (Accessibility permission required)")
                self._ready_event.set()
                return

            source = quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
            loop = quartz.CFRunLoopGetCurrent()
            mode = getattr(quartz, "kCFRunLoopCommonModes", "kCFRunLoopCommonModes")
            quartz.CFRunLoopAddSource(loop, source, mode)
            quartz.CGEventTapEnable(tap, True)

            with self._lock:
                self._tap = tap
                self._run_loop = loop
                self._tap_ok = True
                self.last_error = ""

            self._ready_event.set()
            quartz.CFRunLoopRun()
        except Exception as exc:
            self._tap_ok = False
            self.last_error = str(exc)
            self._log(f"CGEventTap loop error: {exc}")
            self._ready_event.set()
        finally:
            with self._lock:
                self._tap_ok = False
                self._tap = None
                self._run_loop = None

    def _event_tap_callback(self, proxy, event_type: int, event, refcon):
        quartz = _get_quartz()
        if quartz is None:
            return event

        if event_type in (kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput):
            tap = self._tap
            if tap is not None:
                try:
                    quartz.CGEventTapEnable(tap, True)
                    self._log("Re-enabled CGEventTap after macOS timeout/user disable")
                except Exception:
                    pass
            return event

        try:
            keycode = int(quartz.CGEventGetIntegerValueField(event, 9))  # kCGKeyboardEventKeycode = 9
            flags = int(quartz.CGEventGetFlags(event))
            autorepeat = int(quartz.CGEventGetIntegerValueField(event, kCGKeyboardEventAutorepeat))
        except Exception:
            return event

        # Interactive hotkey capture hook
        capture_cb = self._capture_cb
        if capture_cb is not None:
            self._handle_capture_event(event_type, keycode, flags)
            return event

        if self.paused or autorepeat:
            return event

        self._dispatch_matching_hotkey(event_type, keycode, flags)
        return event

    def _handle_capture_event(self, event_type: int, keycode: int, flags: int) -> None:
        active_mods: List[str] = []
        if flags & kCGEventFlagMaskControl:
            active_mods.append("ctrl")
        if flags & kCGEventFlagMaskAlternate:
            active_mods.append("alt")
        if flags & kCGEventFlagMaskShift:
            active_mods.append("shift")
        if flags & kCGEventFlagMaskCommand:
            active_mods.append("cmd")

        if event_type == kCGEventFlagsChanged:
            upd = self._capture_update_cb
            if upd is not None:
                try:
                    upd([DISPLAY_MODIFIER_NAMES[m] for m in active_mods])
                except Exception:
                    pass
            return

        if event_type != kCGEventKeyDown:
            return

        if keycode == 0x35 and not active_mods:  # Escape cancels capture
            cb = self._capture_cb
            self._capture_cb = None
            self._capture_update_cb = None
            if cb is not None:
                cb(None)
            return

        key_name = VK_TO_KEY_NAME.get(keycode)
        if not key_name or keycode in STANDALONE_MODIFIER_VKS:
            return
        combo = "+".join([*active_mods, key_name]) if active_mods else key_name
        cb = self._capture_cb
        self._capture_cb = None
        self._capture_update_cb = None
        if cb is not None:
            cb(combo)

    def _dispatch_matching_hotkey(self, event_type: int, keycode: int, flags: int) -> None:
        now = time.monotonic()
        debounce = float(HOTKEY_RUNTIME.get("debounce_seconds", 0.2))

        with self._lock:
            items = list(self._bindings.items())

        for name, (parsed, callback) in items:
            if parsed.is_standalone_modifier:
                if event_type != kCGEventFlagsChanged or keycode != parsed.keycode:
                    continue
                _, mod_mask = STANDALONE_MODIFIER_VKS[parsed.keycode]
                if not (flags & mod_mask):
                    continue
            else:
                if event_type != kCGEventKeyDown or keycode != parsed.keycode:
                    continue
                if (flags & PRIMARY_MODIFIER_MASK) != (parsed.required_flags & PRIMARY_MODIFIER_MASK):
                    continue
                if parsed.check_fn_flag and not (flags & kCGEventFlagMaskSecondaryFn):
                    continue

            last = self._last_trigger.get(name, 0.0)
            if (now - last) < debounce:
                continue
            self._last_trigger[name] = now
            try:
                callback()
            except Exception as exc:
                self._log(f"Hotkey callback error ({name}): {exc}")

    def capture_next_hotkey(
        self,
        on_complete: Callable[[Optional[str]], None],
        on_update: Optional[Callable[[List[str]], None]] = None,
    ) -> Callable[[], None]:
        """Capture the next pressed hotkey via CGEventTap if active; returns a cancel function."""
        self._capture_cb = on_complete
        self._capture_update_cb = on_update

        def cancel() -> None:
            if self._capture_cb is on_complete:
                self._capture_cb = None
                self._capture_update_cb = None

        return cancel

    def health_status(self, expected_hotkeys: Dict[str, str]) -> Tuple[bool, str]:
        """Verify that all expected hotkeys are parsed and backed by a live CGEventTap."""
        with self._lock:
            for name, raw in expected_hotkeys.items():
                entry = self._bindings.get(name)
                if entry is None:
                    return False, f"missing_binding:{name}"
                norm = normalize_hotkey_string(raw)
                if entry[0].canonical != norm:
                    return False, f"mismatch:{name}"

        if _get_quartz() is None and sys.platform != "darwin":
            return True, "ok_mock"

        if not self.tap_active:
            return False, self.last_error or "tap_inactive"
        return True, "ok"

    def _force_restart_listener(self) -> bool:
        self._stop_tap_loop()
        return self._ensure_tap_running()

    def unregister_all(self) -> None:
        with self._lock:
            self._bindings.clear()

    def _stop_tap_loop(self) -> None:
        quartz = _get_quartz()
        with self._lock:
            tap = self._tap
            loop = self._run_loop
            thread = self._thread
            self._tap = None
            self._run_loop = None
            self._thread = None
            self._tap_ok = False

        if quartz is not None:
            if tap is not None:
                try:
                    quartz.CGEventTapEnable(tap, False)
                except Exception:
                    pass
            if loop is not None:
                try:
                    quartz.CFRunLoopStop(loop)
                except Exception:
                    pass

        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)

    def cleanup(self) -> None:
        self._capture_cb = None
        self._capture_update_cb = None
        self.unregister_all()
        self._stop_tap_loop()


# ---------------------------------------------------------------------------
# Clipboard, Paste & Unicode Live Typing
# ---------------------------------------------------------------------------


def copy_to_clipboard(text: str, root=None) -> bool:
    """Copy UTF-8/Unicode text to the macOS general pasteboard without stealing focus."""
    if text is None:
        return False

    appkit = _get_appkit()
    if appkit is not None:
        try:
            pb = appkit.NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb_type = getattr(appkit, "NSPasteboardTypeString", "public.utf8-plain-text")
            ok = pb.setString_forType_(str(text), pb_type)
            if ok:
                return True
        except Exception as exc:
            logger.debug("NSPasteboard copy failed: %s", exc)

    if sys.platform == "darwin" and Path("/usr/bin/pbcopy").exists():
        try:
            subprocess.run(
                ["/usr/bin/pbcopy"],
                input=str(text).encode("utf-8"),
                check=True,
                timeout=2.0,
            )
            return True
        except Exception as exc:
            logger.debug("pbcopy fallback failed: %s", exc)

    if root is not None and threading.current_thread() is threading.main_thread():
        try:
            root.clipboard_clear()
            root.clipboard_append(str(text))
            root.update_idletasks()
            return True
        except Exception as exc:
            logger.debug("Tk clipboard fallback failed: %s", exc)

    return False


def paste_from_clipboard(root=None) -> bool:
    """Synthesize ``Cmd+V`` into the active macOS application."""
    quartz = _get_quartz()
    if quartz is not None and is_accessibility_trusted(prompt=False):
        try:
            src = quartz.CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
            cmd_flag = getattr(quartz, "kCGEventFlagMaskCommand", kCGEventFlagMaskCommand)
            hid_tap = getattr(quartz, "kCGHIDEventTap", kCGHIDEventTap)

            down = quartz.CGEventCreateKeyboardEvent(src, kVK_ANSI_V, True)
            quartz.CGEventSetFlags(down, cmd_flag)
            up = quartz.CGEventCreateKeyboardEvent(src, kVK_ANSI_V, False)
            quartz.CGEventSetFlags(up, cmd_flag)

            quartz.CGEventPost(hid_tap, down)
            quartz.CGEventPost(hid_tap, up)
            return True
        except Exception as exc:
            logger.debug("Quartz Cmd+V paste failed: %s", exc)

    if sys.platform == "darwin" and Path("/usr/bin/osascript").exists():
        try:
            subprocess.run(
                [
                    "/usr/bin/osascript",
                    "-e",
                    'tell application "System Events" to keystroke "v" using command down',
                ],
                check=True,
                timeout=2.5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as exc:
            logger.debug("osascript Cmd+V fallback failed: %s", exc)

    return False


def chunk_utf16_safe(text: str, max_units: int = 16) -> List[Tuple[str, int]]:
    """Split ``text`` into chunks of at most ``max_units`` UTF-16 code units.

    Guarantees surrogate pairs (such as emoji > U+FFFF) are never split across
    events, returning ``[(chunk_str, utf16_code_unit_count), ...]``.
    """
    if max_units < 2:
        raise ValueError("max_units must be >= 2 to accommodate UTF-16 surrogate pairs")
    if not text:
        return []

    chunks: List[Tuple[str, int]] = []
    buf: List[str] = []
    units = 0

    for ch in text:
        ch_units = 2 if ord(ch) > 0xFFFF else 1
        if units + ch_units > max_units and buf:
            chunks.append(("".join(buf), units))
            buf = [ch]
            units = ch_units
        else:
            buf.append(ch)
            units += ch_units

    if buf:
        chunks.append(("".join(buf), units))
    return chunks


def type_text(text: str, root=None) -> bool:
    """Type arbitrary Unicode text (including ``äöüß``, accents, emoji) into the focused app.

    Uses ``Quartz.CGEventKeyboardSetUnicodeString`` chunked to <=16 UTF-16 code
    units per event, falling back to clipboard copy + ``Cmd+V`` if Accessibility
    is unavailable.
    """
    if not text:
        return True

    quartz = _get_quartz()
    if quartz is not None and is_accessibility_trusted(prompt=False):
        try:
            src = quartz.CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
            hid_tap = getattr(quartz, "kCGHIDEventTap", kCGHIDEventTap)
            segments = str(text).split("\n")
            for seg_idx, segment in enumerate(segments):
                for chunk_str, utf16_len in chunk_utf16_safe(segment, max_units=16):
                    down = quartz.CGEventCreateKeyboardEvent(src, 0, True)
                    quartz.CGEventSetFlags(down, 0)
                    quartz.CGEventKeyboardSetUnicodeString(down, utf16_len, chunk_str)

                    up = quartz.CGEventCreateKeyboardEvent(src, 0, False)
                    quartz.CGEventSetFlags(up, 0)
                    quartz.CGEventKeyboardSetUnicodeString(up, utf16_len, chunk_str)

                    quartz.CGEventPost(hid_tap, down)
                    quartz.CGEventPost(hid_tap, up)
                    time.sleep(0.002)

                if seg_idx < len(segments) - 1:
                    ret_down = quartz.CGEventCreateKeyboardEvent(src, kVK_Return, True)
                    quartz.CGEventSetFlags(ret_down, 0)
                    ret_up = quartz.CGEventCreateKeyboardEvent(src, kVK_Return, False)
                    quartz.CGEventSetFlags(ret_up, 0)
                    quartz.CGEventPost(hid_tap, ret_down)
                    quartz.CGEventPost(hid_tap, ret_up)
                    time.sleep(0.002)
            return True
        except Exception as exc:
            logger.debug("CGEventKeyboardSetUnicodeString failed, falling back to clipboard: %s", exc)

    copied = copy_to_clipboard(text, root=root)
    pasted = paste_from_clipboard(root=root)
    return bool(copied and pasted)


# ---------------------------------------------------------------------------
# Audio Feedback & Aqua Window Styling
# ---------------------------------------------------------------------------


def play_feedback_sound(start: bool = True) -> bool:
    """Play macOS start ('Tink') or stop ('Pop') cue without blocking."""
    appkit = _get_appkit()
    sound_name = PLATFORM_SOUND["mac_start_sound"] if start else PLATFORM_SOUND["mac_stop_sound"]
    if appkit is not None:
        try:
            ns_sound = appkit.NSSound.soundNamed_(sound_name)
            if ns_sound is not None:
                ns_sound.play()
                return True
        except Exception as exc:
            logger.debug("NSSound failed: %s", exc)

    sound_file = (
        PLATFORM_SOUND["mac_start_sound_file"]
        if start
        else PLATFORM_SOUND["mac_stop_sound_file"]
    )
    if sys.platform == "darwin" and Path("/usr/bin/afplay").exists() and Path(sound_file).exists():
        try:
            subprocess.Popen(
                ["/usr/bin/afplay", sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as exc:
            logger.debug("afplay fallback failed: %s", exc)

    # Synthesized PortAudio sine-wave fallback
    try:
        import numpy as np
        import sounddevice as sd

        sr = 24000
        dur = float(PLATFORM_SOUND.get("synth_duration_s", 0.12))
        freq = float(
            PLATFORM_SOUND["synth_start_freq"] if start else PLATFORM_SOUND["synth_stop_freq"]
        )
        t = np.linspace(0.0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
        envelope = np.sin(np.pi * t / dur).astype(np.float32)
        wave = (0.18 * np.sin(2.0 * np.pi * freq * t) * envelope).astype(np.float32)
        sd.play(wave, samplerate=sr, blocking=False)
        return True
    except Exception:
        return False


def play_feedback_sound_async(start: bool = True) -> None:
    """Trigger recording start/stop sound cue asynchronously."""
    appkit = _get_appkit()
    if appkit is not None:
        # NSSound.play() is native Cocoa async and safest on the caller thread
        try:
            sound_name = (
                PLATFORM_SOUND["mac_start_sound"] if start else PLATFORM_SOUND["mac_stop_sound"]
            )
            ns_sound = appkit.NSSound.soundNamed_(sound_name)
            if ns_sound is not None:
                ns_sound.play()
                return
        except Exception:
            pass
    threading.Thread(target=play_feedback_sound, args=(start,), daemon=True).start()


def apply_floating_pill_style(window) -> bool:
    """Configure a Tk/CTk Toplevel as a Spaces-friendly floating utility panel on macOS."""
    applied = False
    try:
        window.attributes("-topmost", True)
        applied = True
    except Exception:
        pass

    if sys.platform == "darwin":
        try:
            window.tk.call(
                "::tk::unsupported::MacWindowStyle",
                "style",
                window._w,
                "floating",
                "sideTitlebar",
            )
            applied = True
        except Exception as exc:
            logger.debug("MacWindowStyle floating call skipped: %s", exc)
    return applied


def reveal_in_finder(target_path: Path | str) -> bool:
    """Open a directory or file in macOS Finder (with cross-platform fallback)."""
    p = Path(target_path)
    p.mkdir(parents=True, exist_ok=True) if not p.suffix else None
    cmd = ["/usr/bin/open", str(p)] if sys.platform == "darwin" else ["xdg-open", str(p)]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as exc:
        logger.warning("Failed to open %s: %s", p, exc)
        return False
