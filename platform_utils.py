"""
Platform Abstraction Layer for NeuroWhisper
Provides cross-platform support for Mac and Windows.
"""

import logging
import sys
import platform

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
IS_MAC = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'
IS_LINUX = sys.platform.startswith('linux')
PLATFORM_NAME = 'mac' if IS_MAC else 'windows' if IS_WINDOWS else 'linux'

# Modifier key for paste operations (Cmd on Mac, Ctrl on Windows/Linux)
PASTE_MODIFIER = (
    PLATFORM_SHORTCUTS['mac_paste_modifier']
    if IS_MAC
    else PLATFORM_SHORTCUTS['other_paste_modifier']
)


def get_paste_shortcut():
    """Get the platform-appropriate paste keyboard shortcut."""
    return f"{PASTE_MODIFIER}+{PLATFORM_SHORTCUTS['paste_key']}"


# ============================================================================
# Keyboard/Hotkey Abstraction
# ============================================================================
# On Windows: use 'keyboard' library (full global hotkey support)
# On Mac: use 'pynput' library (doesn't require root for basic functionality)

_keyboard_backend = None
_hotkey_listeners = {}  # Store active hotkey listeners

# Set to the import-failure text when the platform's keyboard backend could not
# be loaded (e.g. Linux "must be root to ..." OSError, or Mac pynput missing).
# Stays None when the backend loaded successfully.
KEYBOARD_ERROR: "str | None" = None

if IS_MAC:
    try:
        from pynput import keyboard as pynput_keyboard
        from pynput.keyboard import Key, KeyCode, Listener
        
        class MacKeyboardWrapper:
            """Wrapper to provide keyboard-library-like API using pynput on Mac."""
            
            def __init__(self):
                self._hotkeys = {}  # hotkey_string -> callback
                self._current_keys = set()
                self._listener = None
                self._controller = pynput_keyboard.Controller()
                # Hotkeys currently "fired" (all their keys are held down), so
                # we only invoke the callback on the transition into a match
                # rather than on every key-repeat event while held.
                self._fired = set()
            
            def _normalize_key(self, key):
                """Normalize a pynput key to a string."""
                if isinstance(key, Key):
                    name = key.name.lower()
                    # Map pynput names to keyboard library names
                    mapping = {
                        'cmd': 'command',
                        'cmd_l': 'command',
                        'cmd_r': 'command',
                        'ctrl': 'ctrl',
                        'ctrl_l': 'ctrl',
                        'ctrl_r': 'ctrl',
                        'alt': 'alt',
                        'alt_l': 'alt',
                        'alt_r': 'alt',
                        'alt_gr': 'alt',
                        'shift': 'shift',
                        'shift_l': 'shift',
                        'shift_r': 'shift',
                    }
                    return mapping.get(name, name)
                elif isinstance(key, KeyCode):
                    if key.char:
                        return key.char.lower()
                    elif key.vk:
                        # Handle function keys by virtual key code
                        # F1-F12 are typically vk 112-123 on some systems
                        # But pynput uses Key.f1, Key.f2, etc. for function keys
                        return str(key.vk)
                return str(key).lower()
            
            def _parse_hotkey(self, hotkey_str):
                """Parse a hotkey string like 'ctrl+alt+s' into a frozenset of keys."""
                parts = hotkey_str.lower().replace(' ', '').split('+')
                return frozenset(parts)
            
            def _check_hotkeys(self):
                """Fire at most one hotkey per key event.

                Two rules matter here:
                - Fire only on the transition INTO a match, tracked in
                  self._fired, so holding the keys down (auto-repeat) does not
                  re-trigger the callback.
                - Among all matching hotkeys fire only the most specific one.
                  With "f9" and "ctrl+alt+f9" both registered, pressing the
                  full combo matches both; releasing ctrl+alt would then leave
                  "f9" still held and fire the shorter binding as a phantom
                  second hotkey.

                self._hotkeys is snapshotted because a callback is free to
                register or remove hotkeys while we are iterating.
                """
                items = list(self._hotkeys.items())

                matched = []
                for hotkey_str, callback in items:
                    required_keys = self._parse_hotkey(hotkey_str)
                    if required_keys.issubset(self._current_keys):
                        matched.append((len(required_keys), hotkey_str, callback))
                    else:
                        self._fired.discard(hotkey_str)

                if not matched:
                    return

                matched.sort(key=lambda item: item[0], reverse=True)
                _size, best_hotkey, best_callback = matched[0]

                # Everything a longer binding shadows counts as "already
                # handled", so it cannot fire on the way back out.
                for _s, hotkey_str, _cb in matched[1:]:
                    self._fired.add(hotkey_str)

                if best_hotkey in self._fired:
                    return
                self._fired.add(best_hotkey)
                try:
                    best_callback()
                except Exception as e:
                    logger.error(f"Hotkey callback error: {e}")

            def _on_press(self, key):
                key_name = self._normalize_key(key)
                self._current_keys.add(key_name)
                self._check_hotkeys()

            def _on_release(self, key):
                key_name = self._normalize_key(key)
                self._current_keys.discard(key_name)
                self._check_hotkeys()
            
            def _ensure_listener(self):
                """Start the keyboard listener if not already running."""
                if self._listener is None or not self._listener.is_alive():
                    self._listener = Listener(
                        on_press=self._on_press,
                        on_release=self._on_release
                    )
                    self._listener.start()
            
            def add_hotkey(self, hotkey, callback, suppress=False):
                """Register a global hotkey."""
                self._hotkeys[hotkey.lower()] = callback
                self._ensure_listener()
            
            def remove_hotkey(self, hotkey):
                """Remove a registered hotkey."""
                hotkey_lower = hotkey.lower()
                if hotkey_lower in self._hotkeys:
                    del self._hotkeys[hotkey_lower]
            
            def write(self, text):
                """Type text using the keyboard."""
                self._controller.type(text)
            
            def send(self, hotkey):
                """Send a key combination (e.g., 'command+v')."""
                keys = hotkey.lower().split('+')
                
                # Map key names to pynput keys
                key_map = {
                    'command': Key.cmd,
                    'cmd': Key.cmd,
                    'ctrl': Key.ctrl,
                    'alt': Key.alt,
                    'shift': Key.shift,
                    'enter': Key.enter,
                    'return': Key.enter,
                    'tab': Key.tab,
                    'space': Key.space,
                    'backspace': Key.backspace,
                    'delete': Key.delete,
                    'escape': Key.esc,
                    'esc': Key.esc,
                }
                
                # Add function keys
                for i in range(1, 13):
                    key_map[f'f{i}'] = getattr(Key, f'f{i}')
                
                # Convert key names to pynput keys
                pynput_keys = []
                for k in keys:
                    if k in key_map:
                        pynput_keys.append(key_map[k])
                    elif len(k) == 1:
                        pynput_keys.append(k)
                    else:
                        pynput_keys.append(KeyCode.from_char(k[0]) if k else None)
                
                # Press all modifier keys, then the final key, then release
                modifiers = pynput_keys[:-1]
                final_key = pynput_keys[-1] if pynput_keys else None
                
                for mod in modifiers:
                    if mod:
                        self._controller.press(mod)
                
                if final_key:
                    if isinstance(final_key, str):
                        self._controller.press(final_key)
                        self._controller.release(final_key)
                    else:
                        self._controller.press(final_key)
                        self._controller.release(final_key)
                
                for mod in reversed(modifiers):
                    if mod:
                        self._controller.release(mod)
            
            def press_and_release(self, key):
                """Press and release a single key."""
                self.send(key)
            
            def hook(self, callback, suppress=False):
                """Hook all keyboard events (limited implementation)."""
                # pynput doesn't support the same hook API
                # This is a simplified version
                self._ensure_listener()
                return callback  # Return for unhook compatibility
            
            def unhook(self, callback):
                """Unhook a keyboard callback (no-op on Mac)."""
                pass
            
            def on_release(self, callback, suppress=False):
                """Register a key release callback (limited implementation)."""
                self._ensure_listener()
                return callback
        
        keyboard_module = MacKeyboardWrapper()
        _keyboard_backend = 'pynput'
        
    except Exception as e:
        KEYBOARD_ERROR = (
            f"pynput not installed or failed to load: {e}. "
            "Hotkeys will not work on Mac. Install with: pip install pynput"
        )
        logger.warning(KEYBOARD_ERROR)
        keyboard_module = None
        _keyboard_backend = None

else:
    # Windows/Linux - use standard keyboard library.
    # On Linux, `import keyboard` can raise ImportError (not installed) OR
    # OSError/other Exception (e.g. "must be root to ..." when not running
    # with sufficient privileges), so catch Exception broadly here.
    try:
        import keyboard
        keyboard_module = keyboard
        _keyboard_backend = 'keyboard'
    except Exception as e:
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
    """
    Play a system feedback sound.
    
    Args:
        frequency: Sound frequency in Hz (Windows only)
        duration_ms: Duration in milliseconds (Windows only)
        start: If True, play start sound; if False, play stop sound
    """
    if IS_WINDOWS:
        try:
            import winsound
            freq = frequency if start else PLATFORM_SOUND["windows_stop_freq"]
            winsound.Beep(freq, duration_ms)
        except Exception:
            pass
    elif IS_MAC:
        try:
            # Use macOS system sounds via AppKit
            from AppKit import NSSound
            # Use built-in system sounds
            sound_name = PLATFORM_SOUND["mac_start_sound"] if start else PLATFORM_SOUND["mac_stop_sound"]
            sound = NSSound.soundNamed_(sound_name)
            if sound:
                sound.play()
        except ImportError:
            # Fallback: try afplay command
            try:
                import subprocess
                sound_file = (
                    PLATFORM_SOUND["mac_start_sound_file"]
                    if start
                    else PLATFORM_SOUND["mac_stop_sound_file"]
                )
                subprocess.Popen(['afplay', sound_file], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            except Exception:
                pass
        except Exception:
            pass


def play_feedback_sound_async(start=True):
    """Play feedback sound in a background thread to avoid blocking."""
    import threading
    threading.Thread(
        target=play_system_sound, 
        args=(PLATFORM_SOUND["windows_start_freq"], PLATFORM_SOUND["windows_duration_ms"], start), 
        daemon=True
    ).start()


# ============================================================================
# Clipboard Abstraction
# ============================================================================

def paste_from_clipboard():
    """
    Send the paste keyboard shortcut (Ctrl+V on Windows, Cmd+V on Mac).
    """
    if keyboard_module:
        keyboard_module.send(get_paste_shortcut())


def type_text(text):
    """
    Type text using the keyboard.
    """
    if keyboard_module:
        keyboard_module.write(text)


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
    """
    Get the appropriate icon path for the current platform.
    
    Args:
        base_path: Directory containing icon files
        
    Returns:
        Tuple of (ico_path, png_path) for Windows, or (icns_path, png_path) for Mac
    """
    import os
    
    if IS_MAC:
        icns_path = os.path.join(base_path, ICON_ICNS)
        png_path = os.path.join(base_path, ICON_PNG)
        return icns_path, png_path
    else:
        ico_path = os.path.join(base_path, ICON_ICO)
        png_path = os.path.join(base_path, ICON_PNG)
        return ico_path, png_path


# ============================================================================
# Utility Functions
# ============================================================================

def get_platform_info():
    """Get a string describing the current platform."""
    return f"{PLATFORM_NAME} ({platform.platform()})"


def print_platform_status():
    """Print platform detection status for debugging."""
    logger.info(f"Platform: {PLATFORM_NAME}")
    logger.info(f"Keyboard backend: {_keyboard_backend or 'none'}")
    logger.info(f"Paste shortcut: {get_paste_shortcut()}")


# Initialize platform-specific settings on import
set_app_id()
