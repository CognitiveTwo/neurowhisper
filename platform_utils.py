"""
Platform Abstraction Layer for NeuroWhisper
Provides cross-platform support for Windows, macOS, and Linux.
"""

import logging
import os
import platform
import sys
import threading
import time
import types

logger = logging.getLogger(__name__)

from app_config import (
    APP_USER_MODEL_ID,
    ICON_ICNS,
    ICON_ICO,
    ICON_PNG,
    PLATFORM_SHORTCUTS,
    PLATFORM_SOUND,
)

# Platform detection
IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
PLATFORM_NAME = "mac" if IS_MAC else "windows" if IS_WINDOWS else "linux"

# Modifier key for paste operations (Cmd on Mac, Ctrl on Windows/Linux)
PASTE_MODIFIER = (
    PLATFORM_SHORTCUTS["mac_paste_modifier"]
    if IS_MAC
    else PLATFORM_SHORTCUTS["other_paste_modifier"]
)


def get_paste_shortcut():
    """Get the platform-appropriate paste keyboard shortcut."""
    return f"{PASTE_MODIFIER}+{PLATFORM_SHORTCUTS['paste_key']}"


# ============================================================================
# Keyboard/Hotkey Abstraction
# ============================================================================

_keyboard_backend = None
_hotkey_listeners = {}
KEYBOARD_ERROR: "str | None" = None

# macOS Carbon virtual keycode fallback table (ANSI layout) used when holding
# Ctrl/Option turns key.char into control bytes ('\x13') or Option symbols ('ß').
_MAC_VK_TO_NAME = {
    0: "a", 1: "s", 2: "d", 3: "f", 4: "h", 5: "g", 6: "z", 7: "x",
    8: "c", 9: "v", 11: "b", 12: "q", 13: "w", 14: "e", 15: "r",
    16: "y", 17: "t", 18: "1", 19: "2", 20: "3", 21: "4", 22: "6",
    23: "5", 24: "=", 25: "9", 26: "7", 27: "-", 28: "8", 29: "0",
    30: "]", 31: "o", 32: "u", 33: "[", 34: "i", 35: "p", 36: "enter",
    37: "l", 38: "j", 39: "'", 40: "k", 41: ";", 42: "\\", 43: ",",
    44: "/", 45: "n", 46: "m", 47: ".", 48: "tab", 49: "space",
    50: "`", 51: "backspace", 53: "esc",
    96: "f5", 97: "f6", 98: "f7", 99: "f3", 100: "f8", 101: "f9",
    103: "f11", 109: "f10", 111: "f12", 118: "f4", 120: "f2", 122: "f1",
    123: "left", 124: "right", 125: "down", 126: "up",
}


def _build_pynput_wrapper():
    """Construct a thread-safe ``pynput`` keyboard wrapper supporting ``hook``,
    ``unhook``, ``on_release``, ``add_hotkey``, ``write``, and ``send``."""
    from pynput import keyboard as pynput_keyboard
    from pynput.keyboard import Key, KeyCode, Listener

    class PynputKeyboardWrapper:
        """Keyboard-library-compatible API backed by ``pynput`` (macOS & non-root Linux)."""

        def __init__(self):
            self._lock = threading.RLock()
            self._hotkeys = {}
            self._hooks = []
            self._release_hooks = []
            self._current_keys = set()
            self._vk_pressed_names = {}
            self._fired = set()
            self._listener = None
            self._controller = None  # Instantiated lazily on first write/send

        def _get_controller(self):
            if self._controller is None:
                self._controller = pynput_keyboard.Controller()
            return self._controller

        def _normalize_key(self, key):
            if isinstance(key, Key):
                name = getattr(key, "name", None) or str(key)
                name = name.lower()
                mapping = {
                    "cmd": "command",
                    "cmd_l": "command",
                    "cmd_r": "command",
                    "ctrl": "ctrl",
                    "ctrl_l": "ctrl",
                    "ctrl_r": "ctrl",
                    "alt": "alt",
                    "alt_l": "alt",
                    "alt_r": "alt",
                    "alt_gr": "alt",
                    "shift": "shift",
                    "shift_l": "shift",
                    "shift_r": "shift",
                    "super": "win",
                    "super_l": "win",
                    "super_r": "win",
                }
                return mapping.get(name, name)

            if isinstance(key, KeyCode):
                vk = getattr(key, "vk", None)
                ch = getattr(key, "char", None)
                if ch:
                    code = ord(ch[0])
                    # Ctrl+A..Ctrl+Z arrive as KeyCode(char='\x01'..'\x1a'); physical
                    # Tab/Enter/Esc arrive as Key enum instances handled above.
                    if 1 <= code <= 26:
                        return chr(ord("a") + code - 1)
                    if IS_MAC and vk in _MAC_VK_TO_NAME and ord(ch[0]) > 127:
                        # Option+<key> on macOS produces non-ASCII symbols (e.g. Option+S -> 'ß')
                        return _MAC_VK_TO_NAME[vk]
                    return ch.lower()
                if vk is not None:
                    if IS_MAC and vk in _MAC_VK_TO_NAME:
                        return _MAC_VK_TO_NAME[vk]
                    if 65 <= vk <= 90:
                        return chr(vk).lower()
                    if 48 <= vk <= 57:
                        return chr(vk)
                    return str(vk)
            return str(key).lower()

        @staticmethod
        def _parse_hotkey(hotkey_str):
            parts = [p for p in hotkey_str.lower().replace(" ", "").split("+") if p]
            return frozenset(parts)

        def _check_hotkeys_unlocked(self):
            with self._lock:
                items = list(self._hotkeys.items())
                current = set(self._current_keys)

            matched = []
            with self._lock:
                for hotkey_str, callback in items:
                    required_keys = self._parse_hotkey(hotkey_str)
                    if required_keys and required_keys.issubset(current):
                        matched.append((len(required_keys), hotkey_str, callback))
                    else:
                        self._fired.discard(hotkey_str)

                if not matched:
                    return

                matched.sort(key=lambda item: item[0], reverse=True)
                _size, best_hotkey, best_callback = matched[0]
                for _s, hotkey_str, _cb in matched[1:]:
                    self._fired.add(hotkey_str)
                if best_hotkey in self._fired:
                    return
                self._fired.add(best_hotkey)

            try:
                best_callback()
            except Exception as e:
                logger.error("Hotkey callback error: %s", e)

        def _on_press(self, key):
            key_name = self._normalize_key(key)
            vk = getattr(key, "vk", None)
            scan_code = vk or 0
            ev = types.SimpleNamespace(
                name=key_name, event_type="down", scan_code=scan_code, time=time.time()
            )
            with self._lock:
                self._current_keys.add(key_name)
                if vk is not None:
                    self._vk_pressed_names[vk] = key_name
                hooks = list(self._hooks)

            for cb in hooks:
                try:
                    cb(ev)
                except Exception as e:
                    logger.debug("Hook press callback error: %s", e)

            self._check_hotkeys_unlocked()

        def _on_release(self, key):
            key_name = self._normalize_key(key)
            vk = getattr(key, "vk", None)
            scan_code = vk or 0
            prev_name = None
            with self._lock:
                self._current_keys.discard(key_name)
                if vk is not None:
                    prev_name = self._vk_pressed_names.pop(vk, None)
                    if prev_name:
                        self._current_keys.discard(prev_name)
                hooks = list(self._hooks)
                rel_hooks = list(self._release_hooks)
            ev = types.SimpleNamespace(
                name=prev_name or key_name,
                event_type="up",
                scan_code=scan_code,
                time=time.time(),
            )

            for cb in hooks:
                try:
                    cb(ev)
                except Exception as e:
                    logger.debug("Hook release callback error: %s", e)

            for cb in rel_hooks:
                try:
                    cb(ev)
                except Exception as e:
                    logger.debug("Release callback error: %s", e)

            self._check_hotkeys_unlocked()

        def _listener_alive(self) -> bool:
            lst = self._listener
            if lst is None:
                return False
            is_alive_fn = getattr(lst, "is_alive", None)
            alive = is_alive_fn() if callable(is_alive_fn) else True
            running = getattr(lst, "running", True)
            return bool(alive and running)

        def _ensure_listener(self):
            with self._lock:
                if not self._listener_alive():
                    lst = Listener(on_press=self._on_press, on_release=self._on_release)
                    # Compatibility attribute read by HotkeyManager._listener_thread_health()
                    try:
                        lst.listening = True
                    except Exception:
                        pass
                    lst.start()
                    self._listener = lst

        def add_hotkey(self, hotkey, callback, suppress=False):
            with self._lock:
                self._hotkeys[hotkey.lower()] = callback
            self._ensure_listener()
            return hotkey.lower()

        def remove_hotkey(self, hotkey):
            with self._lock:
                self._hotkeys.pop(str(hotkey).lower(), None)

        def hook(self, callback, suppress=False):
            with self._lock:
                if callback not in self._hooks:
                    self._hooks.append(callback)
            self._ensure_listener()
            return callback

        def on_release(self, callback, suppress=False):
            with self._lock:
                if callback not in self._release_hooks:
                    self._release_hooks.append(callback)
            self._ensure_listener()
            return callback

        def unhook(self, callback):
            with self._lock:
                if callback in self._hooks:
                    self._hooks.remove(callback)
                if callback in self._release_hooks:
                    self._release_hooks.remove(callback)

        def unhook_all(self):
            with self._lock:
                self._hooks.clear()
                self._release_hooks.clear()
                self._hotkeys.clear()
                self._current_keys.clear()
                self._vk_pressed_names.clear()
                self._fired.clear()
                lst = self._listener
                self._listener = None
            if lst is not None:
                try:
                    lst.stop()
                except Exception:
                    pass

        def is_pressed(self, key_name):
            with self._lock:
                return str(key_name).lower() in self._current_keys

        def write(self, text):
            if text:
                self._get_controller().type(text)

        def send(self, hotkey):
            tokens = [k.strip() for k in hotkey.lower().split("+") if k.strip()]
            if not tokens:
                return

            key_map = {
                "command": Key.cmd,
                "cmd": Key.cmd,
                "win": getattr(Key, "cmd", None),
                "ctrl": Key.ctrl,
                "alt": Key.alt,
                "shift": Key.shift,
                "enter": Key.enter,
                "return": Key.enter,
                "tab": Key.tab,
                "space": Key.space,
                "backspace": Key.backspace,
                "delete": Key.delete,
                "escape": Key.esc,
                "esc": Key.esc,
                "up": getattr(Key, "up", None),
                "down": getattr(Key, "down", None),
                "left": getattr(Key, "left", None),
                "right": getattr(Key, "right", None),
                "home": getattr(Key, "home", None),
                "end": getattr(Key, "end", None),
                "page_up": getattr(Key, "page_up", None),
                "page_down": getattr(Key, "page_down", None),
            }
            for i in range(1, 21):
                fk = getattr(Key, f"f{i}", None)
                if fk is not None:
                    key_map[f"f{i}"] = fk

            pynput_keys = []
            for k in tokens:
                mapped = key_map.get(k)
                if mapped is not None:
                    pynput_keys.append(mapped)
                elif len(k) == 1:
                    pynput_keys.append(KeyCode.from_char(k))
                else:
                    logger.warning("Unrecognized key token in send(%r): %r", hotkey, k)
                    return

            modifiers = pynput_keys[:-1]
            final_key = pynput_keys[-1]
            ctrl = self._get_controller()
            pressed_mods = []
            try:
                for mod in modifiers:
                    ctrl.press(mod)
                    pressed_mods.append(mod)
                ctrl.press(final_key)
                ctrl.release(final_key)
            finally:
                for mod in reversed(pressed_mods):
                    try:
                        ctrl.release(mod)
                    except Exception:
                        pass

        def press_and_release(self, key):
            self.send(key)

    return PynputKeyboardWrapper


class _AdaptiveLinuxKeyboardProxy:
    """Uses ``keyboard`` first, and transparently falls back to ``pynput`` on X11
    when Linux ``keyboard`` raises ``must be root`` at runtime."""

    def __init__(self, primary):
        self._primary = primary
        self._fallback = None
        self._using_fallback = False
        self._lock = threading.Lock()

    def _activate_fallback(self, orig_exc: BaseException):
        with self._lock:
            if self._using_fallback and self._fallback is not None:
                return self._fallback
            try:
                cls = _build_pynput_wrapper()
                self._fallback = cls()
                self._using_fallback = True
                global _keyboard_backend
                _keyboard_backend = "pynput"
                logger.info(
                    "Switched Linux keyboard backend from 'keyboard' (%s) to non-root 'pynput'.",
                    orig_exc,
                )
                return self._fallback
            except Exception:
                raise orig_exc

    @property
    def _listener(self):
        target = self._fallback if self._using_fallback else self._primary
        return getattr(target, "_listener", None)

    def _call(self, method_name: str, *args, **kwargs):
        if self._using_fallback and self._fallback is not None:
            return getattr(self._fallback, method_name)(*args, **kwargs)
        fn = getattr(self._primary, method_name)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if "root" in str(exc).lower() or isinstance(exc, PermissionError):
                fb = self._activate_fallback(exc)
                return getattr(fb, method_name)(*args, **kwargs)
            raise

    def hook(self, *args, **kwargs):
        return self._call("hook", *args, **kwargs)

    def unhook(self, *args, **kwargs):
        if self._using_fallback and self._fallback is not None:
            return self._fallback.unhook(*args, **kwargs)
        return self._primary.unhook(*args, **kwargs)

    def unhook_all(self, *args, **kwargs):
        if self._using_fallback and self._fallback is not None:
            return self._fallback.unhook_all(*args, **kwargs)
        if hasattr(self._primary, "unhook_all"):
            return self._primary.unhook_all(*args, **kwargs)

    def on_release(self, *args, **kwargs):
        return self._call("on_release", *args, **kwargs)

    def add_hotkey(self, *args, **kwargs):
        return self._call("add_hotkey", *args, **kwargs)

    def remove_hotkey(self, *args, **kwargs):
        if self._using_fallback and self._fallback is not None:
            return self._fallback.remove_hotkey(*args, **kwargs)
        return self._primary.remove_hotkey(*args, **kwargs)

    def write(self, *args, **kwargs):
        return self._call("write", *args, **kwargs)

    def send(self, *args, **kwargs):
        return self._call("send", *args, **kwargs)

    def press_and_release(self, *args, **kwargs):
        return self._call("press_and_release", *args, **kwargs)

    def is_pressed(self, *args, **kwargs):
        if self._using_fallback and self._fallback is not None:
            return self._fallback.is_pressed(*args, **kwargs)
        if hasattr(self._primary, "is_pressed"):
            try:
                return self._primary.is_pressed(*args, **kwargs)
            except Exception:
                return False
        return False


if IS_MAC:
    try:
        MacKeyboardWrapper = _build_pynput_wrapper()
        keyboard_module = MacKeyboardWrapper()
        _keyboard_backend = "pynput"
    except Exception as e:
        KEYBOARD_ERROR = (
            f"pynput not installed or failed to load: {e}. "
            "Hotkeys will not work on Mac. Install with: pip install pynput"
        )
        logger.warning(KEYBOARD_ERROR)
        keyboard_module = None
        _keyboard_backend = None
else:
    try:
        import keyboard as _raw_keyboard

        if IS_LINUX:
            keyboard_module = _AdaptiveLinuxKeyboardProxy(_raw_keyboard)
        else:
            keyboard_module = _raw_keyboard
        _keyboard_backend = "keyboard"
    except Exception as e:
        if IS_LINUX:
            try:
                _LinuxWrapper = _build_pynput_wrapper()
                keyboard_module = _LinuxWrapper()
                _keyboard_backend = "pynput"
            except Exception as e2:
                KEYBOARD_ERROR = f"keyboard/pynput failed to load on Linux: {e}; {e2}"
                logger.warning(KEYBOARD_ERROR)
                keyboard_module = None
                _keyboard_backend = None
        else:
            KEYBOARD_ERROR = f"keyboard library not installed or failed to load: {e}"
            logger.warning(KEYBOARD_ERROR)
            keyboard_module = None
            _keyboard_backend = None


# ============================================================================
# Sound Abstraction
# ============================================================================

def play_system_sound(
    frequency=PLATFORM_SOUND["windows_start_freq"],
    duration_ms=PLATFORM_SOUND["windows_duration_ms"],
    start=True,
):
    """Play a system feedback sound (non-fatal on headless/minimal systems)."""
    if IS_WINDOWS:
        try:
            import winsound
            freq = frequency if start else PLATFORM_SOUND["windows_stop_freq"]
            winsound.Beep(freq, duration_ms)
        except Exception:
            pass
    elif IS_MAC:
        try:
            from AppKit import NSSound
            sound_name = (
                PLATFORM_SOUND["mac_start_sound"]
                if start
                else PLATFORM_SOUND["mac_stop_sound"]
            )
            sound = NSSound.soundNamed_(sound_name)
            if sound:
                sound.play()
        except ImportError:
            try:
                import subprocess
                sound_file = (
                    PLATFORM_SOUND["mac_start_sound_file"]
                    if start
                    else PLATFORM_SOUND["mac_stop_sound_file"]
                )
                subprocess.run(
                    ["afplay", sound_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                )
            except Exception:
                pass
        except Exception:
            pass


def play_feedback_sound_async(start=True):
    """Play feedback sound in a background thread to avoid blocking."""
    threading.Thread(
        target=play_system_sound,
        args=(PLATFORM_SOUND["windows_start_freq"], PLATFORM_SOUND["windows_duration_ms"], start),
        daemon=True,
    ).start()


# ============================================================================
# Clipboard & Text Injection Abstraction
# ============================================================================

def paste_from_clipboard() -> bool:
    """Send the paste keyboard shortcut (Ctrl+V on Windows/Linux, Cmd+V on Mac)."""
    if not keyboard_module:
        return False
    try:
        keyboard_module.send(get_paste_shortcut())
        return True
    except Exception as exc:
        logger.warning("paste_from_clipboard failed: %s", exc)
        return False


def type_text(text: str) -> bool:
    """Type text using the active keyboard backend."""
    if not keyboard_module or not text:
        return False
    try:
        keyboard_module.write(text)
        return True
    except Exception as exc:
        logger.warning("type_text failed: %s", exc)
        return False


# ============================================================================
# Window/Icon Abstraction
# ============================================================================

def set_app_id():
    """Set the Windows App User Model ID for proper taskbar grouping."""
    if IS_WINDOWS:
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                APP_USER_MODEL_ID
            )
        except Exception:
            pass


def get_icon_path(base_path):
    """Get ``(primary_icon_path, png_path)`` for the current platform."""
    png_path = os.path.join(base_path, ICON_PNG)
    if IS_MAC:
        icns_path = os.path.join(base_path, ICON_ICNS)
        if os.path.exists(icns_path):
            return icns_path, png_path
        return png_path, png_path
    ico_path = os.path.join(base_path, ICON_ICO)
    return ico_path, png_path


# ============================================================================
# Utility Functions
# ============================================================================

def get_platform_info():
    """Get a string describing the current platform."""
    return f"{PLATFORM_NAME} ({platform.platform()})"


def print_platform_status():
    """Print platform detection status for debugging."""
    logger.info("Platform: %s", PLATFORM_NAME)
    logger.info("Keyboard backend: %s", _keyboard_backend or "none")
    logger.info("Paste shortcut: %s", get_paste_shortcut())
