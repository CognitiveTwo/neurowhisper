"""Non-fatal Linux StatusNotifierItem / AppIndicator / XEmbed system tray controller.

Dynamically renders crisp 64x64 RGBA Pillow status icons reflecting recording state
(``idle``, ``live``, ``batch``, ``online``, ``transcribing``, ``ready``) and degrades
gracefully on headless CI, Wayland compositors without a tray, or when
``NEUROWHISPER_NO_TRAY=1`` is set.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Dict, Optional

from PIL import Image, ImageDraw

from . import APP_NAME
from .config import THEME_COLORS

log = logging.getLogger("neurowhisper.tray")

TRAY_STATE_COLORS: Dict[str, str] = {
    "idle": THEME_COLORS["teal"],
    "ready": THEME_COLORS["ready"],
    "live": THEME_COLORS["live"],
    "batch": THEME_COLORS["batch"],
    "online": THEME_COLORS["online"],
    "transcribing": THEME_COLORS["transcribing"],
}

TRAY_STATE_LABELS: Dict[str, str] = {
    "idle": f"{APP_NAME} — Idle",
    "ready": f"{APP_NAME} — Ready",
    "live": f"{APP_NAME} — Live Dictation Active",
    "batch": f"{APP_NAME} — Batch Recording Active",
    "online": f"{APP_NAME} — Cloud Recording Active",
    "transcribing": f"{APP_NAME} — Transcribing...",
}


def render_tray_icon_image(state: str = "idle", size: int = 64) -> Image.Image:
    """Render a 64x64 anti-aliased RGBA microphone badge colored for ``state``."""
    accent = TRAY_STATE_COLORS.get(state, THEME_COLORS["teal"])
    bg_fill = "#131a22"
    border_fill = "#2a3441"

    scale = 2
    hi = size * scale
    img = Image.new("RGBA", (hi, hi), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Outer dark circular plate
    pad = int(hi * 0.06)
    draw.ellipse([pad, pad, hi - pad, hi - pad], fill=bg_fill, outline=border_fill, width=4)

    # Active recording ring glow when live/batch/online/transcribing
    if state in ("live", "batch", "online", "transcribing"):
        ring_pad = int(hi * 0.11)
        draw.ellipse(
            [ring_pad, ring_pad, hi - ring_pad, hi - ring_pad],
            outline=accent,
            width=6,
        )

    # Stylized microphone capsule in center
    cx = hi // 2
    cap_w = int(hi * 0.22)
    cap_top = int(hi * 0.23)
    cap_bot = int(hi * 0.56)
    draw.rounded_rectangle(
        [cx - cap_w // 2, cap_top, cx + cap_w // 2, cap_bot],
        radius=cap_w // 2,
        fill=accent,
    )

    # Microphone cradle arc + stem + base
    arc_w = int(hi * 0.36)
    arc_top = int(hi * 0.34)
    arc_bot = int(hi * 0.66)
    draw.arc(
        [cx - arc_w // 2, arc_top, cx + arc_w // 2, arc_bot],
        start=0,
        end=180,
        fill="#e5e7eb",
        width=5,
    )
    stem_bot = int(hi * 0.76)
    draw.line([cx, arc_bot, cx, stem_bot], fill="#e5e7eb", width=5)
    base_half = int(hi * 0.13)
    draw.line([cx - base_half, stem_bot, cx + base_half, stem_bot], fill="#e5e7eb", width=5)

    # Status indicator dot at bottom-right
    dot_r = int(hi * 0.11)
    dot_cx = int(hi * 0.77)
    dot_cy = int(hi * 0.77)
    draw.ellipse(
        [dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r],
        fill=accent,
        outline=bg_fill,
        width=4,
    )

    return img.resize((size, size), Image.Resampling.LANCZOS)


class SystemTrayController:
    """Safe wrapper around ``pystray.Icon`` for Linux desktop tray integration."""

    def __init__(
        self,
        on_toggle_live: Optional[Callable[[], None]] = None,
        on_toggle_batch: Optional[Callable[[], None]] = None,
        on_stop_recording: Optional[Callable[[], None]] = None,
        on_toggle_window: Optional[Callable[[], None]] = None,
        on_open_transcriptions: Optional[Callable[[], None]] = None,
        on_quit: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_toggle_live = on_toggle_live
        self._on_toggle_batch = on_toggle_batch
        self._on_stop_recording = on_stop_recording
        self._on_toggle_window = on_toggle_window
        self._on_open_transcriptions = on_open_transcriptions
        self._on_quit = on_quit

        self._lock = threading.Lock()
        self._icon = None
        self._thread: Optional[threading.Thread] = None
        self._state: str = "idle"
        self._icon_cache: Dict[str, Image.Image] = {}
        self._available: bool = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def state(self) -> str:
        return self._state

    def _get_cached_image(self, state: str) -> Image.Image:
        key = state if state in TRAY_STATE_COLORS else "idle"
        img = self._icon_cache.get(key)
        if img is None:
            img = render_tray_icon_image(key, size=64)
            self._icon_cache[key] = img
        return img

    def start(self) -> bool:
        """Start the background ``pystray`` icon thread if supported by the current session."""
        if os.environ.get("NEUROWHISPER_NO_TRAY", "").strip() == "1":
            log.info("System tray disabled via NEUROWHISPER_NO_TRAY=1")
            return False

        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            log.info("Headless environment detected — skipping system tray")
            return False

        with self._lock:
            if self._icon is not None:
                return self._available

            try:
                import pystray

                def _wrap(cb: Optional[Callable[[], None]]):
                    def _handler(_icon, _item):
                        if callable(cb):
                            try:
                                cb()
                            except Exception as exc:
                                log.debug("Tray callback error: %s", exc)
                    return _handler

                menu = pystray.Menu(
                    pystray.MenuItem(
                        "Show / Hide Window",
                        _wrap(self._on_toggle_window),
                        default=True,
                    ),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem("Toggle Live Dictation", _wrap(self._on_toggle_live)),
                    pystray.MenuItem("Toggle Batch Recording", _wrap(self._on_toggle_batch)),
                    pystray.MenuItem("Stop Recording", _wrap(self._on_stop_recording)),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem(
                        "Open Transcriptions Folder",
                        _wrap(self._on_open_transcriptions),
                    ),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem(f"Quit {APP_NAME}", _wrap(self._on_quit)),
                )

                icon = pystray.Icon(
                    APP_NAME,
                    icon=self._get_cached_image(self._state),
                    title=TRAY_STATE_LABELS.get(self._state, APP_NAME),
                    menu=menu,
                )
                self._icon = icon
            except Exception as exc:
                log.info("System tray backend unavailable (%s) — continuing without tray", exc)
                self._icon = None
                self._available = False
                return False

            def _run_tray() -> None:
                try:
                    self._available = True
                    icon.run()
                except Exception as exc:
                    log.info("System tray exited cleanly (%s)", exc)
                    self._available = False

            self._thread = threading.Thread(
                target=_run_tray,
                name="neurowhisper-tray",
                daemon=True,
            )
            self._thread.start()
            return True

    def set_state(self, state: str, tooltip: Optional[str] = None) -> None:
        """Update the tray icon badge color and tooltip non-fatally."""
        norm = (state or "idle").strip().lower()
        if norm not in TRAY_STATE_COLORS:
            norm = "idle"
        with self._lock:
            self._state = norm
            icon = self._icon
            if icon is None or not self._available:
                return
            try:
                icon.icon = self._get_cached_image(norm)
                icon.title = tooltip or TRAY_STATE_LABELS.get(norm, APP_NAME)
            except Exception as exc:
                log.debug("Could not update tray icon state: %s", exc)

    def stop(self) -> None:
        """Stop and remove the system tray icon."""
        with self._lock:
            icon = self._icon
            self._icon = None
            self._available = False
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
