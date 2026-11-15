"""macOS Menu Bar status item controller with dynamically rendered Pillow icons."""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Optional

from PIL import Image, ImageDraw

from . import APP_NAME
from .config import THEME_COLORS

logger = logging.getLogger(__name__)

STATUS_BADGE_COLORS: Dict[str, str] = {
    "idle": "#9ca3af",
    "live": THEME_COLORS["live"],
    "batch": THEME_COLORS["batch"],
    "online": THEME_COLORS["online"],
    "transcribing": THEME_COLORS["transcribing"],
    "ready": THEME_COLORS["ready"],
    "error": "#f97316",
}

STATUS_DEFAULT_LABELS: Dict[str, str] = {
    "idle": "Ready",
    "live": "Recording (Live Dictation)",
    "batch": "Recording (Batch)",
    "online": "Recording (OpenAI Cloud)",
    "transcribing": "Transcribing…",
    "ready": "Copied to Clipboard",
    "error": "Attention Needed",
}


def render_menubar_icon(state: str = "idle", size: int = 64) -> Image.Image:
    """Render a crisp 64x64 RGBA microphone status icon with a colored state indicator."""
    dim = max(32, int(size))
    img = Image.new("RGBA", (dim, dim), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    scale = dim / 64.0
    mic_color = (245, 247, 250, 245) if state == "idle" else (255, 255, 255, 255)
    stroke = max(2, round(4 * scale))

    # Microphone capsule
    cap_left = round(22 * scale)
    cap_top = round(8 * scale)
    cap_right = round(42 * scale)
    cap_bottom = round(36 * scale)
    draw.rounded_rectangle(
        [cap_left, cap_top, cap_right, cap_bottom],
        radius=round(10 * scale),
        fill=mic_color,
    )

    # U-shaped cradle
    arc_box = [
        round(15 * scale),
        round(16 * scale),
        round(49 * scale),
        round(45 * scale),
    ]
    draw.arc(arc_box, start=0, end=180, fill=mic_color, width=stroke)

    # Stem & base
    cx = round(32 * scale)
    draw.line(
        [(cx, round(45 * scale)), (cx, round(55 * scale))],
        fill=mic_color,
        width=stroke,
    )
    draw.line(
        [(round(22 * scale), round(55 * scale)), (round(42 * scale), round(55 * scale))],
        fill=mic_color,
        width=stroke,
    )

    # Status indicator badge in bottom-right corner
    badge_hex = STATUS_BADGE_COLORS.get(state, STATUS_BADGE_COLORS["idle"]).lstrip("#")
    r = int(badge_hex[0:2], 16)
    g = int(badge_hex[2:4], 16)
    b = int(badge_hex[4:6], 16)
    bx0, by0 = round(42 * scale), round(42 * scale)
    bx1, by1 = round(60 * scale), round(60 * scale)
    draw.ellipse(
        [bx0 - round(2 * scale), by0 - round(2 * scale), bx1 + round(2 * scale), by1 + round(2 * scale)],
        fill=(15, 20, 25, 230),
    )
    draw.ellipse([bx0, by0, bx1, by1], fill=(r, g, b, 255))

    return img


class MenuBarController:
    """Non-blocking macOS menu bar icon wrapper around ``pystray.Icon.run_detached``."""

    def __init__(
        self,
        on_toggle_live: Optional[Callable[[], None]] = None,
        on_toggle_batch: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_toggle_window: Optional[Callable[[], None]] = None,
        on_open_transcriptions: Optional[Callable[[], None]] = None,
        on_quit: Optional[Callable[[], None]] = None,
        dispatch_to_ui: Optional[Callable[[Callable[[], None]], None]] = None,
    ) -> None:
        self._on_toggle_live = on_toggle_live
        self._on_toggle_batch = on_toggle_batch
        self._on_stop = on_stop
        self._on_toggle_window = on_toggle_window
        self._on_open_transcriptions = on_open_transcriptions
        self._on_quit = on_quit
        self._dispatch_to_ui = dispatch_to_ui

        self._state: str = "idle"
        self._status_text: str = STATUS_DEFAULT_LABELS["idle"]
        self._icon = None
        self._pystray = None
        self.available: bool = False

    def _invoke(self, fn: Optional[Callable[[], None]]) -> None:
        if fn is None:
            return
        if self._dispatch_to_ui is not None:
            try:
                self._dispatch_to_ui(fn)
                return
            except Exception:
                pass
        try:
            fn()
        except Exception as exc:
            logger.debug("MenuBar action callback error: %s", exc)

    def _build_menu(self):
        pystray = self._pystray
        if pystray is None:
            return None
        Item = pystray.MenuItem
        return pystray.Menu(
            Item(lambda _item: f"Status: {self._status_text}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            Item("Toggle Live Dictation", lambda _icon, _item: self._invoke(self._on_toggle_live)),
            Item("Toggle Batch Recording", lambda _icon, _item: self._invoke(self._on_toggle_batch)),
            Item("Stop Recording", lambda _icon, _item: self._invoke(self._on_stop)),
            pystray.Menu.SEPARATOR,
            Item("Show / Hide Window", lambda _icon, _item: self._invoke(self._on_toggle_window)),
            Item(
                "Open Transcriptions Folder",
                lambda _icon, _item: self._invoke(self._on_open_transcriptions),
            ),
            pystray.Menu.SEPARATOR,
            Item(f"Quit {APP_NAME}", lambda _icon, _item: self._invoke(self._on_quit)),
        )

    def start(self) -> bool:
        """Attach the menu bar icon via ``run_detached()``. Degrades cleanly to no-op if unavailable."""
        if os.environ.get("NEUROWHISPER_NO_TRAY", "").strip().lower() in ("1", "true", "yes"):
            self.available = False
            return False

        if self._icon is not None:
            return self.available

        try:
            import pystray  # Lazy import: avoids headless Xorg DisplayNameError on Linux CI

            self._pystray = pystray
            image = render_menubar_icon(self._state)
            icon = pystray.Icon(
                APP_NAME,
                icon=image,
                title=f"{APP_NAME} — {self._status_text}",
                menu=self._build_menu(),
            )
            icon.run_detached()
            self._icon = icon
            self.available = True
            return True
        except Exception as exc:
            logger.info("Menu bar icon disabled (headless or unsupported environment): %s", exc)
            self._icon = None
            self.available = False
            return False

    def set_state(self, state: str, status_text: Optional[str] = None) -> None:
        """Update the menu bar icon color badge and status label."""
        norm_state = state if state in STATUS_BADGE_COLORS else "idle"
        self._state = norm_state
        self._status_text = status_text or STATUS_DEFAULT_LABELS.get(norm_state, "Ready")

        icon = self._icon
        if icon is None or not self.available:
            return
        try:
            icon.icon = render_menubar_icon(norm_state)
            icon.title = f"{APP_NAME} — {self._status_text}"
            icon.update_menu()
        except Exception as exc:
            logger.debug("MenuBar set_state update skipped: %s", exc)

    def stop(self) -> None:
        """Detach and remove the menu bar status item."""
        icon = self._icon
        self._icon = None
        self.available = False
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
