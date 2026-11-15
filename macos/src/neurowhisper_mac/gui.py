"""Full-parity dark-mode macOS CustomTkinter GUI for neurowhisper.

Key macOS engineering guarantees:
- Zero legacy ``tk.Button`` or ``tk.OptionMenu`` widgets (eliminating unreadable
  white Aqua chrome in dark mode).
- ``SF Pro Text`` and ``Menlo`` typography with +1pt Darwin DPI compensation.
- Non-blocking Accessibility TCC permission banner with 1-click System Settings
  deep-link and live ``Re-check`` recovery after ``.dmg`` updates.
- Preserves ``device='auto'`` / ``'coreml'`` / ``'mps'`` across backend loads so
  Apple Metal (MLX) and Parakeet CoreML never get coerced to CPU.
- Full multi-machine config/stats/transcript sync compatibility via ``paths.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple
import webbrowser

import customtkinter as ctk
import numpy as np

from . import APP_NAME, __version__, paths
from .backends import (
    OpenAIBackend,
    WhisperBackend,
    create_backend,
    detect_best_backend,
    get_backend_display_name,
    unload_all_backends,
)
from .config import (
    APP_TITLE,
    APP_WINDOW_GEOMETRY,
    APP_WINDOW_MIN_HEIGHT,
    APP_WINDOW_MIN_WIDTH,
    AUDIO_RUNTIME,
    AUTHOR_LINK_URL,
    BACKEND_DISPLAY_TO_BACKEND_DEVICE,
    BACKEND_OPTIONS,
    CTK_APPEARANCE_MODE,
    CTK_COLOR_THEME,
    DEFAULT_CONFIG,
    HOTKEY_RUNTIME,
    IDLE_UNLOAD_CHECK_INTERVAL_SECONDS,
    LEGACY_MODEL_ALIASES,
    LIVE_PAUSE_RANGE,
    MODEL_DOWNLOAD_SIZES,
    MODEL_MAP,
    MODEL_MAP_REVERSE,
    OPENAI_DEFAULT_EDIT_PROMPT,
    OPENAI_EDIT_MODEL_OPTIONS,
    OPENAI_KEY_URL,
    OPENAI_TRANSCRIPTION_MODEL_OPTIONS,
    THEME_COLORS,
    TYPING_WPM,
    UI_RUNTIME,
    mono_font,
    resolve_backend_display,
    ui_font,
)
from .insights import InsightsMixin, tokenize_words
from .menubar import MenuBarController
from . import platform_mac

logger = logging.getLogger(__name__)

MACHINE_LOCAL_CONFIG_KEYS = frozenset({
    "openai_api_key",
    "backend",
    "device",
    "compute_type",
    "input_device",
    "worker_process",
})

LOCAL_LANGUAGE_OPTIONS = [
    "auto",
    "en",
    "de",
    "fr",
    "es",
    "it",
    "pt",
    "nl",
    "pl",
    "ja",
    "zh",
]


class WhisperApp(InsightsMixin):
    """Main macOS neurowhisper application controller and dark-themed UI."""

    def __init__(
        self,
        root: tk.Misc,
        menubar: Optional[MenuBarController] = None,
        skip_model_load: Optional[bool] = None,
    ) -> None:
        self.root = root
        self.menubar = menubar
        self._shutting_down = False
        self._after_jobs: List[str] = []

        paths.ensure_runtime_dirs()
        self._init_insights_state()

        ctk.set_appearance_mode(CTK_APPEARANCE_MODE)
        ctk.set_default_color_theme(CTK_COLOR_THEME)

        self.root.title(APP_TITLE)
        self.root.geometry(APP_WINDOW_GEOMETRY)
        self.root.minsize(APP_WINDOW_MIN_WIDTH, APP_WINDOW_MIN_HEIGHT)
        self.root.configure(bg=THEME_COLORS["bg"])
        self._apply_window_icon()
        self._configure_dark_ttk_styles()

        # Load config & stats
        self.config: Dict[str, Any] = self._load_config()
        self._init_stats_state()
        self._load_stats()

        # Runtime state
        self.mode: Optional[str] = None  # None | "live" | "batch"
        self._stopping_recording = False
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.ui_queue: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self.hotkey_action_queue: "queue.Queue[str]" = queue.Queue(
            maxsize=int(HOTKEY_RUNTIME["action_queue_maxsize"])
        )
        self.audio_stream = None
        self.current_rms: float = 0.0
        self.last_transcription_text: str = ""
        self.last_activity_ts: float = time.monotonic()

        # Backend state
        self.backend: Optional[WhisperBackend] = None
        self.online_backend: Optional[OpenAIBackend] = None
        self._model_lock = threading.RLock()
        self.current_backend_name: str = str(self.config.get("backend", "auto"))
        self.current_device: str = str(self.config.get("device", "auto"))
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nw-transcribe")

        # Hotkey state
        self.hotkey_manager = platform_mac.MacHotkeyManager(log_fn=self.log_internal)
        self.capturing_hotkey: Optional[str] = None
        self._capture_cancel_fn: Optional[Callable[[], None]] = None
        self._hotkey_auto_rebinds: int = 0
        self._last_hotkey_register_ts: float = 0.0

        # Floating mini-pill state
        self.mini_window: Optional[tk.Toplevel] = None
        self._mini_drag_offset: Tuple[int, int] = (0, 0)

        # Build UI
        self.setup_ui()
        self._apply_always_on_top()
        self._refresh_stats_labels()
        self.root.after(
            int(UI_RUNTIME["initial_histogram_delay_ms"]),
            self._safe_initial_visual_refresh,
        )

        # Bindings & loops
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Unmap>", self._on_root_unmap)
        self.root.bind("<Map>", self._on_root_map)
        self.register_hotkeys()
        self._refresh_tcc_banner()

        self._schedule_recurring(
            int(UI_RUNTIME["main_loop_interval_ms"]),
            self._main_ui_tick,
        )
        self._schedule_recurring(
            int(HOTKEY_RUNTIME["watchdog_interval_ms"]),
            self._hotkey_watchdog_tick,
        )
        self._schedule_recurring(
            int(IDLE_UNLOAD_CHECK_INTERVAL_SECONDS * 1000),
            self._idle_unload_tick,
        )

        if menubar is not None:
            self.attach_menubar(menubar)

        should_skip = (
            skip_model_load
            if skip_model_load is not None
            else os.environ.get("NEUROWHISPER_SKIP_MODEL_LOAD", "").strip().lower()
            in ("1", "true", "yes")
        )
        if should_skip:
            self.status_var.set("Ready (model load deferred)")
            self._update_backend_badge()
        else:
            self.reload_model_async()

    # ------------------------------------------------------------------
    # Window Icon, Dark TTK Theme & Scheduling Helpers
    # ------------------------------------------------------------------

    def _apply_window_icon(self) -> None:
        icon_path = paths.find_resource_file("icon.png")
        if icon_path.exists():
            try:
                from PIL import Image, ImageTk

                img = Image.open(icon_path).resize((64, 64))
                self._window_icon_ref = ImageTk.PhotoImage(img)
                self.root.iconphoto(True, self._window_icon_ref)
            except Exception:
                pass

    def _configure_dark_ttk_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            if "clam" in style.theme_names():
                style.theme_use("clam")
        except Exception:
            pass

        bg = THEME_COLORS["bg"]
        card = THEME_COLORS["accent_bg"]
        fg = THEME_COLORS["fg"]
        border = THEME_COLORS["border"]
        teal = THEME_COLORS["teal"]

        style.configure("TFrame", background=card)
        style.configure("Dark.TFrame", background=bg)
        style.configure("TLabel", background=card, foreground=fg, font=ui_font(9))
        style.configure("Muted.TLabel", background=card, foreground=THEME_COLORS["text_muted"], font=ui_font(8))
        style.configure(
            "TCombobox",
            fieldbackground=bg,
            background=card,
            foreground=fg,
            arrowcolor=teal,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", bg)],
            foreground=[("readonly", fg)],
            selectbackground=[("readonly", THEME_COLORS["teal_dim"])],
            selectforeground=[("readonly", fg)],
        )
        style.configure(
            "TEntry",
            fieldbackground=bg,
            foreground=fg,
            insertcolor=fg,
            bordercolor=border,
        )
        style.configure(
            "Vertical.TScrollbar",
            background=card,
            troughcolor=bg,
            bordercolor=bg,
            arrowcolor=THEME_COLORS["text_muted"],
        )

    def _schedule_after(self, delay_ms: int, callback: Callable[[], None]) -> Optional[str]:
        if self._shutting_down:
            return None
        try:
            job = self.root.after(delay_ms, callback)
            self._after_jobs.append(job)
            return job
        except Exception:
            return None

    def _schedule_recurring(self, interval_ms: int, fn: Callable[[], None]) -> None:
        def wrapper() -> None:
            if self._shutting_down:
                return
            try:
                fn()
            except Exception as exc:
                logger.debug("Recurring UI task error: %s", exc)
            if not self._shutting_down:
                self._schedule_after(interval_ms, wrapper)

        self._schedule_after(interval_ms, wrapper)

    def _post_ui(self, fn: Callable[[], None]) -> None:
        if not self._shutting_down:
            self.ui_queue.put(fn)

    # ------------------------------------------------------------------
    # Configuration & Multi-Machine Peer Stats Persistence
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        merged = dict(DEFAULT_CONFIG)
        shared_path = paths.get_config_file()
        local_path = paths.get_local_config_file()

        for cfg_path in (shared_path, local_path):
            if cfg_path.exists():
                try:
                    with open(cfg_path, "r", encoding="utf-8") as fh:
                        loaded = json.load(fh)
                    if isinstance(loaded, dict):
                        merged.update(loaded)
                except Exception as exc:
                    logger.warning("Failed reading config %s: %s", cfg_path, exc)

        # Normalize legacy model aliases
        mk = str(merged.get("model_key", DEFAULT_CONFIG["model_key"]))
        if mk in LEGACY_MODEL_ALIASES:
            short = LEGACY_MODEL_ALIASES[mk]
            merged["model_key"] = MODEL_MAP_REVERSE.get(short, DEFAULT_CONFIG["model_key"])
        elif mk not in MODEL_MAP:
            merged["model_key"] = MODEL_MAP_REVERSE.get(mk, DEFAULT_CONFIG["model_key"])

        merged["hotkey_live"] = platform_mac.normalize_hotkey_string(
            str(merged.get("hotkey_live") or DEFAULT_CONFIG["hotkey_live"])
        )
        merged["hotkey_batch"] = platform_mac.normalize_hotkey_string(
            str(merged.get("hotkey_batch") or DEFAULT_CONFIG["hotkey_batch"])
        )
        merged["worker_process"] = False
        return merged

    def _persist_config(self) -> None:
        shared_payload = {
            k: v for k, v in self.config.items() if k not in MACHINE_LOCAL_CONFIG_KEYS
        }
        local_payload = {
            k: self.config[k] for k in MACHINE_LOCAL_CONFIG_KEYS if k in self.config
        }
        try:
            paths.write_json_atomic(paths.get_config_file(), shared_payload, indent=2)
            paths.write_json_atomic(paths.get_local_config_file(), local_payload, indent=2)
        except Exception as exc:
            logger.warning("Failed saving config: %s", exc)

    def _init_stats_state(self) -> None:
        self.today_date_str = datetime.date.today().isoformat()
        self.local_today_words: int = 0
        self.local_today_duration: float = 0.0
        self.local_today_hourly: List[int] = [0] * 24
        self.local_alltime_words: int = 0
        self.local_alltime_duration: float = 0.0
        self.local_alltime_hourly: List[int] = [0] * 24
        self.local_active_days: List[str] = []

        self.today_words: int = 0
        self.today_duration: float = 0.0
        self.today_hourly: List[int] = [0] * 24
        self.alltime_words: int = 0
        self.alltime_duration: float = 0.0
        self.alltime_hourly: List[int] = [0] * 24
        self.active_days_count: int = 1

    def _load_stats(self) -> None:
        stats_file = paths.get_stats_file()
        source = stats_file if stats_file.exists() else paths.get_legacy_stats_file()
        today_iso = datetime.date.today().isoformat()
        self.today_date_str = today_iso

        if source.exists():
            try:
                with open(source, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    saved_date = str(data.get("today_date", ""))
                    if saved_date == today_iso:
                        self.local_today_words = int(data.get("today_words", 0) or 0)
                        self.local_today_duration = float(data.get("today_duration", 0.0) or 0.0)
                        hr = list(data.get("today_hourly", [0] * 24))
                        self.local_today_hourly = (hr + [0] * 24)[:24]
                    self.local_alltime_words = int(data.get("alltime_words", 0) or 0)
                    self.local_alltime_duration = float(data.get("alltime_duration", 0.0) or 0.0)
                    ahr = list(data.get("alltime_hourly", [0] * 24))
                    self.local_alltime_hourly = (ahr + [0] * 24)[:24]
                    self.local_active_days = list(data.get("active_days", []) or [])
            except Exception as exc:
                logger.warning("Failed loading stats from %s: %s", source, exc)

        self._merge_peer_stats()

    def _merge_peer_stats(self) -> None:
        """Aggregate local machine stats with any peer ``whisper_stats.*.json`` files in ``data_dir``."""
        today_iso = datetime.date.today().isoformat()
        if self.today_date_str != today_iso:
            self.today_date_str = today_iso
            self.local_today_words = 0
            self.local_today_duration = 0.0
            self.local_today_hourly = [0] * 24

        t_words = self.local_today_words
        t_dur = self.local_today_duration
        t_hourly = list(self.local_today_hourly)
        a_words = self.local_alltime_words
        a_dur = self.local_alltime_duration
        a_hourly = list(self.local_alltime_hourly)
        active_days = set(self.local_active_days)

        own_stats = paths.get_stats_file().resolve()
        data_dir = paths.get_data_dir()
        if data_dir.exists():
            for peer_file in data_dir.glob("whisper_stats.*.json"):
                try:
                    if peer_file.resolve() == own_stats or peer_file.name.endswith(".tmp"):
                        continue
                    with open(peer_file, "r", encoding="utf-8") as fh:
                        pdata = json.load(fh)
                    if not isinstance(pdata, dict):
                        continue
                    if str(pdata.get("today_date", "")) == today_iso:
                        t_words += int(pdata.get("today_words", 0) or 0)
                        t_dur += float(pdata.get("today_duration", 0.0) or 0.0)
                        phr = list(pdata.get("today_hourly", [0] * 24))
                        for i in range(min(24, len(phr))):
                            t_hourly[i] += int(phr[i] or 0)
                    a_words += int(pdata.get("alltime_words", 0) or 0)
                    a_dur += float(pdata.get("alltime_duration", 0.0) or 0.0)
                    pahr = list(pdata.get("alltime_hourly", [0] * 24))
                    for i in range(min(24, len(pahr))):
                        a_hourly[i] += int(pahr[i] or 0)
                    active_days.update(str(d) for d in (pdata.get("active_days") or []))
                except Exception:
                    continue

        self.today_words = t_words
        self.today_duration = t_dur
        self.today_hourly = t_hourly
        self.alltime_words = max(a_words, t_words)
        self.alltime_duration = max(a_dur, t_dur)
        self.alltime_hourly = a_hourly
        self.active_days_count = max(1, len(active_days) or (1 if self.alltime_words > 0 else 1))

    def _save_stats(self) -> None:
        payload = {
            "today_date": self.today_date_str,
            "today_words": self.local_today_words,
            "today_duration": round(self.local_today_duration, 2),
            "today_hourly": self.local_today_hourly,
            "alltime_words": self.local_alltime_words,
            "alltime_duration": round(self.local_alltime_duration, 2),
            "alltime_hourly": self.local_alltime_hourly,
            "active_days": sorted(set(self.local_active_days)),
        }
        try:
            paths.write_json_atomic(paths.get_stats_file(), payload, indent=2)
        except Exception as exc:
            logger.warning("Failed saving stats: %s", exc)

    def _record_utterance(self, text: str, duration_s: float) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        words = tokenize_words(cleaned)
        word_count = len(words)
        now = datetime.datetime.now()
        today_iso = now.date().isoformat()
        if self.today_date_str != today_iso:
            self.today_date_str = today_iso
            self.local_today_words = 0
            self.local_today_duration = 0.0
            self.local_today_hourly = [0] * 24

        hour = now.hour
        self.local_today_words += word_count
        self.local_today_duration += max(0.1, float(duration_s))
        self.local_today_hourly[hour] += word_count
        self.local_alltime_words += word_count
        self.local_alltime_duration += max(0.1, float(duration_s))
        self.local_alltime_hourly[hour] += word_count
        if today_iso not in self.local_active_days:
            self.local_active_days.append(today_iso)

        # Append to daily transcript archive (100% compatible with Windows/Linux)
        trans_dir = paths.get_transcriptions_dir()
        trans_dir.mkdir(parents=True, exist_ok=True)
        day_file = trans_dir / f"{today_iso}.txt"
        stamp = now.strftime("%H:%M:%S")
        single_line = " ".join(cleaned.splitlines())
        try:
            with open(day_file, "a", encoding="utf-8") as fh:
                fh.write(f"[{stamp}] {single_line}\n")
        except OSError as exc:
            logger.warning("Failed appending transcript %s: %s", day_file, exc)

        self.last_transcription_text = cleaned
        self._merge_peer_stats()
        self._save_stats()
        self._post_ui(lambda: self._on_utterance_recorded_ui(stamp, single_line))

    def _on_utterance_recorded_ui(self, stamp: str, text: str) -> None:
        if self._shutting_down:
            return
        if hasattr(self, "history_text"):
            self.history_text.configure(state="normal")
            self.history_text.insert("1.0", f"[{stamp}] {text}\n")
            self.history_text.configure(state="disabled")
        self._refresh_stats_labels()
        self.draw_histograms()
        self._schedule_insights_refresh(1800)

    # ------------------------------------------------------------------
    # UI Construction (100% CTkButton / CTkOptionMenu / Dark TTK)
    # ------------------------------------------------------------------

    def _make_accordion(
        self,
        parent: tk.Widget,
        title: str,
        summary_var: Optional[tk.StringVar],
        toggle_cmd: Callable[[], None],
        collapsed: bool = True,
    ) -> Tuple[ctk.CTkFrame, ctk.CTkButton, tk.Frame]:
        outer = ctk.CTkFrame(
            parent,
            fg_color=THEME_COLORS["accent_bg"],
            corner_radius=10,
            border_width=1,
            border_color=THEME_COLORS["border"],
        )
        outer.pack(fill="x", pady=(0, 8))

        hdr = tk.Frame(outer, bg=THEME_COLORS["accent_bg"])
        hdr.pack(fill="x", padx=8, pady=6)

        arrow = "▸" if collapsed else "▾"
        toggle_btn = ctk.CTkButton(
            hdr,
            text=f"{arrow}  {title}",
            anchor="w",
            height=26,
            corner_radius=6,
            fg_color="transparent",
            hover_color=THEME_COLORS["border"],
            text_color=THEME_COLORS["fg"],
            font=ctk.CTkFont(size=12, weight="bold"),
            command=toggle_cmd,
        )
        toggle_btn.pack(side="left", fill="x", expand=True)

        if summary_var is not None:
            tk.Label(
                hdr,
                textvariable=summary_var,
                bg=THEME_COLORS["accent_bg"],
                fg=THEME_COLORS["text_muted"],
                font=mono_font(8),
                anchor="e",
            ).pack(side="right", padx=(6, 4))

        body = tk.Frame(outer, bg=THEME_COLORS["accent_bg"])
        if not collapsed:
            body.pack(fill="x", padx=6, pady=(0, 8))
        return outer, toggle_btn, body

    def setup_ui(self) -> None:
        bg = THEME_COLORS["bg"]
        card = THEME_COLORS["accent_bg"]
        fg = THEME_COLORS["fg"]
        border = THEME_COLORS["border"]
        teal = THEME_COLORS["teal"]
        muted = THEME_COLORS["text_muted"]

        # Fixed bottom status bar
        self.status_var = tk.StringVar(value="Initializing…")
        self.status_stats_var = tk.StringVar(value="0w today")
        status_bar = tk.Frame(self.root, bg=card)
        status_bar.pack(fill="x", side="bottom")
        tk.Frame(status_bar, bg=border, height=1).pack(fill="x", side="top")
        status_inner = tk.Frame(status_bar, bg=card)
        status_inner.pack(fill="x")
        tk.Label(
            status_inner,
            textvariable=self.status_var,
            anchor="w",
            bg=card,
            fg=fg,
            font=ui_font(9),
        ).pack(side="left", padx=(12, 4), pady=4)
        tk.Label(
            status_inner,
            textvariable=self.status_stats_var,
            anchor="e",
            bg=card,
            fg=THEME_COLORS["text_faint"],
            font=mono_font(8),
        ).pack(side="right", padx=(4, 12), pady=4)

        # Scrollable main viewport
        self._scroll_canvas = tk.Canvas(
            self.root, bg=bg, highlightthickness=0, yscrollincrement=1
        )
        self._scroll_vsb = ttk.Scrollbar(
            self.root, orient="vertical", command=self._scroll_canvas.yview
        )
        self.content_frame = tk.Frame(self._scroll_canvas, bg=bg)
        self._scroll_window = self._scroll_canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )
        self._scroll_canvas.configure(yscrollcommand=self._scroll_vsb.set)
        self._scroll_canvas.pack(side="left", fill="both", expand=True)

        self.content_frame.bind("<Configure>", self._on_content_configure)
        self._scroll_canvas.bind("<Configure>", self._on_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

        # Header wordmark + TOP / MINI / CFG controls
        header = tk.Frame(self.content_frame, bg=bg)
        header.pack(fill="x", padx=14, pady=(10, 4))

        brand = tk.Frame(header, bg=bg)
        brand.pack(side="left")
        tk.Label(brand, text="neuro", bg=bg, fg=fg, font=ui_font(15, "bold")).pack(side="left")
        tk.Label(
            brand, text="whisper", bg=bg, fg=THEME_COLORS["batch"], font=ui_font(15, "bold")
        ).pack(side="left")
        tk.Label(
            brand,
            text=" macOS",
            bg=bg,
            fg=THEME_COLORS["text_faint"],
            font=ui_font(9),
        ).pack(side="left", padx=(4, 0), pady=(3, 0))

        self.top_var = tk.BooleanVar(value=bool(self.config.get("always_on_top", True)))
        self.transcription_mode = tk.StringVar(
            value=str(self.config.get("transcription_mode", "local"))
        )

        self.btn_header_config = ctk.CTkButton(
            header,
            text="CFG",
            width=42,
            height=24,
            corner_radius=8,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=card,
            hover_color=border,
            border_width=1,
            border_color=border,
            text_color=muted,
            command=self._focus_configuration,
        )
        self.btn_header_config.pack(side="right", padx=(6, 0))

        self.btn_header_mini = ctk.CTkButton(
            header,
            text="MINI",
            width=46,
            height=24,
            corner_radius=8,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=card,
            hover_color=border,
            border_width=1,
            border_color=border,
            text_color=muted,
            command=self.enter_mini_mode,
        )
        self.btn_header_mini.pack(side="right", padx=(6, 0))

        self.btn_header_top = ctk.CTkButton(
            header,
            text="TOP",
            width=42,
            height=24,
            corner_radius=8,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=THEME_COLORS["teal_dim"] if self.top_var.get() else card,
            hover_color=border,
            border_width=1,
            border_color=teal if self.top_var.get() else border,
            text_color=teal if self.top_var.get() else muted,
            command=self._toggle_always_on_top,
        )
        self.btn_header_top.pack(side="right")

        self.body = tk.Frame(self.content_frame, bg=bg)
        self.body.pack(fill="both", expand=True, padx=14, pady=(4, 12))

        # 0. Non-blocking macOS Accessibility TCC Banner
        self._build_tcc_banner(self.body)

        # 1. Dictation State & Primary Controls Card
        self._build_dictation_card(self.body)

        # 2. Recent Transcriptions Accordion
        self._build_history_accordion(self.body)

        # 3. Insights & Analytics Accordion
        self._build_insights_accordion(self.body)

        # 4. Configuration Accordion
        self._build_configuration_accordion(self.body)

        # 5. System Log Accordion & Footer
        self._build_log_accordion_and_footer(self.body)

    def _build_tcc_banner(self, parent: tk.Widget) -> None:
        self.tcc_banner = ctk.CTkFrame(
            parent,
            fg_color="#2a1f14",
            corner_radius=10,
            border_width=1,
            border_color="#d97706",
        )
        lbl = tk.Label(
            self.tcc_banner,
            text=(
                "macOS Accessibility permission is needed for global hotkeys & live typing.\n"
                "Tip: After updating from a new .dmg, toggle neurowhisper off & on once in Accessibility."
            ),
            bg="#2a1f14",
            fg="#fde68a",
            font=ui_font(8),
            justify="left",
            anchor="w",
            wraplength=420,
        )
        lbl.pack(fill="x", padx=10, pady=(8, 4))

        btn_row = tk.Frame(self.tcc_banner, bg="#2a1f14")
        btn_row.pack(fill="x", padx=10, pady=(0, 8))

        self.btn_grant_tcc = ctk.CTkButton(
            btn_row,
            text="Grant Accessibility",
            width=136,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#b45309",
            hover_color="#92400e",
            text_color="#ffffff",
            command=self._on_grant_accessibility_clicked,
        )
        self.btn_grant_tcc.pack(side="left", padx=(0, 6))

        self.btn_recheck_tcc = ctk.CTkButton(
            btn_row,
            text="Re-check",
            width=82,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10),
            fg_color=THEME_COLORS["bg"],
            hover_color=THEME_COLORS["border"],
            border_width=1,
            border_color="#d97706",
            text_color="#fde68a",
            command=self._on_recheck_accessibility_clicked,
        )
        self.btn_recheck_tcc.pack(side="left")

    def _refresh_tcc_banner(self) -> None:
        trusted = platform_mac.is_accessibility_trusted(prompt=False)
        if trusted:
            self.tcc_banner.pack_forget()
        else:
            # Pack above the dictation card
            self.tcc_banner.pack(fill="x", pady=(0, 8), before=self.dictation_card)

    def _on_grant_accessibility_clicked(self) -> None:
        platform_mac.is_accessibility_trusted(prompt=True)
        platform_mac.open_accessibility_settings()

    def _on_recheck_accessibility_clicked(self) -> None:
        self._refresh_tcc_banner()
        self._manual_hotkey_refresh()

    def _build_dictation_card(self, parent: tk.Widget) -> None:
        card_bg = THEME_COLORS["accent_bg"]
        bg = THEME_COLORS["bg"]
        border = THEME_COLORS["border"]

        self.dictation_card = ctk.CTkFrame(
            parent,
            fg_color=card_bg,
            corner_radius=12,
            border_width=1,
            border_color=border,
        )
        self.dictation_card.pack(fill="x", pady=(0, 8))

        # Mode strip (Local vs Online + AI Edit) + active model/backend badge
        top_row = tk.Frame(self.dictation_card, bg=card_bg)
        top_row.pack(fill="x", padx=12, pady=(10, 6))

        self.btn_mode_local = ctk.CTkButton(
            top_row,
            text="Local",
            width=62,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10, weight="bold"),
            command=lambda: self._set_transcription_mode("local"),
        )
        self.btn_mode_local.pack(side="left", padx=(0, 4))

        self.btn_mode_online = ctk.CTkButton(
            top_row,
            text="Online",
            width=66,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10, weight="bold"),
            command=lambda: self._set_transcription_mode("online"),
        )
        self.btn_mode_online.pack(side="left", padx=(0, 6))

        self.openai_edit_var = tk.BooleanVar(
            value=bool(self.config.get("openai_edit_enabled", False))
        )
        self.btn_online_edit = ctk.CTkButton(
            top_row,
            text="+Edit",
            width=52,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10),
            command=self._toggle_online_edit,
        )
        self.btn_online_edit.pack(side="left")

        self.model_info_var = tk.StringVar(value="")
        tk.Label(
            top_row,
            textvariable=self.model_info_var,
            bg=card_bg,
            fg=THEME_COLORS["text_muted"],
            font=mono_font(8),
            anchor="e",
        ).pack(side="right")

        # Live / Batch / Stop buttons
        action_row = tk.Frame(self.dictation_card, bg=card_bg)
        action_row.pack(fill="x", padx=12, pady=(2, 6))

        live_lbl = platform_mac.format_hotkey_label(self.config["hotkey_live"])
        batch_lbl = platform_mac.format_hotkey_label(self.config["hotkey_batch"])

        self.btn_live = ctk.CTkButton(
            action_row,
            text=f"Live Dictation ({live_lbl})",
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME_COLORS["live"],
            hover_color="#dc2626",
            text_color="#ffffff",
            command=self.toggle_live,
        )
        self.btn_live.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_batch = ctk.CTkButton(
            action_row,
            text=f"Batch ({batch_lbl})",
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME_COLORS["batch"],
            hover_color="#2563eb",
            text_color="#ffffff",
            command=self.toggle_batch,
        )
        self.btn_batch.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_stop = ctk.CTkButton(
            action_row,
            text="Stop",
            width=60,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=bg,
            hover_color=border,
            border_width=1,
            border_color=border,
            text_color=THEME_COLORS["text_muted"],
            state="disabled",
            command=self.stop_recording,
        )
        self.btn_stop.pack(side="right")

        # VU Level Meter
        self.vu_canvas = tk.Canvas(
            self.dictation_card, height=10, bg=bg, highlightthickness=0
        )
        self.vu_canvas.pack(fill="x", padx=12, pady=(2, 10))

        self._update_mode_buttons()
        self._update_backend_badge()

    def _build_history_accordion(self, parent: tk.Widget) -> None:
        self.history_collapsed = tk.BooleanVar(
            value=bool(self.config.get("history_collapsed", False))
        )
        self.history_summary_var = tk.StringVar(value="Archive")
        _, self.history_toggle_btn, self.history_body = self._make_accordion(
            parent,
            "Recent Transcriptions",
            self.history_summary_var,
            self.toggle_history_section,
            collapsed=self.history_collapsed.get(),
        )

        btn_bar = tk.Frame(self.history_body, bg=THEME_COLORS["accent_bg"])
        btn_bar.pack(fill="x", padx=4, pady=(0, 4))

        ctk.CTkButton(
            btn_bar,
            text="Copy Last",
            width=82,
            height=22,
            corner_radius=6,
            font=ctk.CTkFont(size=10),
            fg_color=THEME_COLORS["bg"],
            hover_color=THEME_COLORS["border"],
            border_width=1,
            border_color=THEME_COLORS["border"],
            text_color=THEME_COLORS["teal"],
            command=self._copy_last_utterance,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_bar,
            text="Open Folder",
            width=92,
            height=22,
            corner_radius=6,
            font=ctk.CTkFont(size=10),
            fg_color=THEME_COLORS["bg"],
            hover_color=THEME_COLORS["border"],
            border_width=1,
            border_color=THEME_COLORS["border"],
            text_color=THEME_COLORS["text_muted"],
            command=self.open_transcriptions_folder,
        ).pack(side="left")

        ctk.CTkButton(
            btn_bar,
            text="Clear View",
            width=76,
            height=22,
            corner_radius=6,
            font=ctk.CTkFont(size=10),
            fg_color=THEME_COLORS["bg"],
            hover_color=THEME_COLORS["border"],
            border_width=1,
            border_color=THEME_COLORS["border"],
            text_color=THEME_COLORS["text_muted"],
            command=self._clear_history_view,
        ).pack(side="right")

        self.history_text = tk.Text(
            self.history_body,
            height=6,
            wrap="word",
            bg=THEME_COLORS["bg"],
            fg=THEME_COLORS["fg"],
            insertbackground=THEME_COLORS["fg"],
            highlightthickness=1,
            highlightbackground=THEME_COLORS["border"],
            relief="flat",
            font=ui_font(9),
            state="disabled",
        )
        self.history_text.pack(fill="x", padx=4, pady=(0, 4))
        self._populate_initial_history()

    def _populate_initial_history(self) -> None:
        today_file = paths.get_transcriptions_dir() / f"{datetime.date.today().isoformat()}.txt"
        if not today_file.exists():
            return
        try:
            lines = today_file.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            if lines:
                self.history_text.configure(state="normal")
                for ln in reversed(lines):
                    self.history_text.insert(tk.END, ln + "\n")
                self.history_text.configure(state="disabled")
                last_ln = lines[-1]
                self.last_transcription_text = (
                    last_ln.split("] ", 1)[1] if "] " in last_ln else last_ln
                )
        except OSError:
            pass

    def _build_insights_accordion(self, parent: tk.Widget) -> None:
        self.insights_collapsed = tk.BooleanVar(
            value=bool(self.config.get("insights_collapsed", True))
        )
        self.insights_summary_var = tk.StringVar(value="0w today")
        _, self.insights_toggle_btn, self.insights_body = self._make_accordion(
            parent,
            "Speech Insights & Activity",
            self.insights_summary_var,
            self.toggle_insights_section,
            collapsed=self.insights_collapsed.get(),
        )

        stats_card = tk.Frame(
            self.insights_body,
            bg=THEME_COLORS["bg"],
            highlightthickness=1,
            highlightbackground=THEME_COLORS["border"],
        )
        stats_card.pack(fill="x", padx=6, pady=(2, 8))

        # Today vs All-Time KPI grids
        kpi_wrap = tk.Frame(stats_card, bg=THEME_COLORS["bg"])
        kpi_wrap.pack(fill="x", padx=10, pady=(8, 4))

        for row_idx, (heading, attrs) in enumerate(
            (
                (
                    "TODAY",
                    (
                        ("Words", "lbl_today_words"),
                        ("Spoken", "lbl_today_duration"),
                        ("Saved", "lbl_today_time"),
                        ("Speed", "lbl_today_speed"),
                    ),
                ),
                (
                    "ALL-TIME",
                    (
                        ("Words", "lbl_alltime_words"),
                        ("Spoken", "lbl_alltime_duration"),
                        ("Saved", "lbl_alltime_time"),
                        ("Speed", "lbl_alltime_speed"),
                    ),
                ),
            )
        ):
            base_r = row_idx * 3
            tk.Label(
                kpi_wrap,
                text=heading,
                bg=THEME_COLORS["bg"],
                fg=THEME_COLORS["text_faint"],
                font=ui_font(7, "bold"),
            ).grid(row=base_r, column=0, columnspan=4, sticky="w", pady=(4 if row_idx else 0, 0))
            for col, (cap, attr) in enumerate(attrs):
                tk.Label(
                    kpi_wrap,
                    text=cap,
                    bg=THEME_COLORS["bg"],
                    fg=THEME_COLORS["text_muted"],
                    font=ui_font(8),
                ).grid(row=base_r + 1, column=col, sticky="w", padx=(0, 16))
                val_lbl = tk.Label(
                    kpi_wrap,
                    text="0",
                    bg=THEME_COLORS["bg"],
                    fg=THEME_COLORS["teal"],
                    font=ui_font(12, "bold"),
                )
                val_lbl.grid(row=base_r + 2, column=col, sticky="w", padx=(0, 16))
                setattr(self, attr, val_lbl)

        self.histogram_canvas_today = tk.Canvas(
            stats_card, height=84, bg=THEME_COLORS["bg"], highlightthickness=0
        )
        self.histogram_canvas_today.pack(fill="x", padx=8, pady=(4, 2))

        self.histogram_canvas_alltime = tk.Canvas(
            stats_card, height=84, bg=THEME_COLORS["bg"], highlightthickness=0
        )
        self.histogram_canvas_alltime.pack(fill="x", padx=8, pady=(2, 8))

        self._build_insights_section(self.insights_body)

    def _build_configuration_accordion(self, parent: tk.Widget) -> None:
        is_first = "config_collapsed" not in self.config
        self.config_collapsed = tk.BooleanVar(
            value=bool(self.config.get("config_collapsed", not is_first))
        )
        self.config_summary_var = tk.StringVar(value="")
        _, self.settings_toggle_btn, self.settings_body = self._make_accordion(
            parent,
            "Configuration",
            self.config_summary_var,
            self.toggle_settings_section,
            collapsed=self.config_collapsed.get(),
        )

        card = THEME_COLORS["accent_bg"]
        bg = THEME_COLORS["bg"]
        border = THEME_COLORS["border"]
        teal = THEME_COLORS["teal"]

        # Row 1: Model + Reload button
        r1 = tk.Frame(self.settings_body, bg=card)
        r1.pack(fill="x", padx=6, pady=4)
        ttk.Label(r1, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=self.config["model_key"])
        self.model_combo = ttk.Combobox(
            r1,
            textvariable=self.model_var,
            values=list(MODEL_MAP.keys()),
            state="readonly",
            width=30,
        )
        self.model_combo.pack(side="left", padx=6, fill="x", expand=True)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        self.btn_reload_model = ctk.CTkButton(
            r1,
            text="Load",
            width=58,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=THEME_COLORS["teal_dim"],
            hover_color=border,
            border_width=1,
            border_color=teal,
            text_color=teal,
            command=self.reload_model_async,
        )
        self.btn_reload_model.pack(side="right")

        # Row 2: Acceleration (macOS native options only) + Language
        r2 = tk.Frame(self.settings_body, bg=card)
        r2.pack(fill="x", padx=6, pady=4)
        ttk.Label(r2, text="Acceleration:").pack(side="left")
        initial_accel = resolve_backend_display(
            str(self.config.get("backend", "auto")),
            str(self.config.get("device", "auto")),
        )
        self.accel_var = tk.StringVar(value=initial_accel)
        self.accel_menu = ctk.CTkOptionMenu(
            r2,
            variable=self.accel_var,
            values=BACKEND_OPTIONS,
            width=175,
            height=24,
            fg_color=bg,
            button_color=border,
            button_hover_color=THEME_COLORS["teal_dim"],
            text_color=THEME_COLORS["fg"],
            font=ctk.CTkFont(size=10),
            command=self._on_acceleration_changed,
        )
        self.accel_menu.pack(side="left", padx=6)

        ttk.Label(r2, text="Lang:").pack(side="left", padx=(6, 2))
        self.local_lang_var = tk.StringVar(
            value=str(self.config.get("local_language", "auto"))
        )
        self.lang_menu = ctk.CTkOptionMenu(
            r2,
            variable=self.local_lang_var,
            values=LOCAL_LANGUAGE_OPTIONS,
            width=74,
            height=24,
            fg_color=bg,
            button_color=border,
            button_hover_color=THEME_COLORS["teal_dim"],
            text_color=THEME_COLORS["fg"],
            font=ctk.CTkFont(size=10),
            command=self._on_local_lang_changed,
        )
        self.lang_menu.pack(side="left")

        # Row 3: Microphone input device
        r_mic = tk.Frame(self.settings_body, bg=card)
        r_mic.pack(fill="x", padx=6, pady=4)
        ttk.Label(r_mic, text="Microphone:").pack(side="left")
        self._mic_map: Dict[str, Optional[int]] = {"System Default": None}
        self._populate_microphones()
        current_dev = self.config.get("input_device")
        selected_mic_lbl = "System Default"
        for lbl, idx in self._mic_map.items():
            if idx is not None and idx == current_dev:
                selected_mic_lbl = lbl
                break
        self.mic_var = tk.StringVar(value=selected_mic_lbl)
        self.mic_combo = ttk.Combobox(
            r_mic,
            textvariable=self.mic_var,
            values=list(self._mic_map.keys()),
            state="readonly",
        )
        self.mic_combo.pack(side="left", padx=6, fill="x", expand=True)
        self.mic_combo.bind("<<ComboboxSelected>>", self._on_mic_selected)

        # Row 4: Live pause & Silence threshold sliders
        r_pause = tk.Frame(self.settings_body, bg=card)
        r_pause.pack(fill="x", padx=6, pady=4)
        ttk.Label(r_pause, text="Live Pause:").pack(side="left")
        self.live_pause_var = tk.DoubleVar(value=float(self.config.get("live_pause", 0.8)))
        self.lbl_pause_val = tk.Label(
            r_pause,
            text=f"{self.live_pause_var.get():.1f}s",
            bg=card,
            fg=teal,
            font=mono_font(8),
            width=5,
        )
        self.lbl_pause_val.pack(side="right")
        self.pause_slider = ctk.CTkSlider(
            r_pause,
            from_=float(LIVE_PAUSE_RANGE["min"]),
            to=float(LIVE_PAUSE_RANGE["max"]),
            number_of_steps=25,
            variable=self.live_pause_var,
            command=self._on_pause_slider_changed,
            height=14,
        )
        self.pause_slider.pack(side="left", fill="x", expand=True, padx=6)

        # Row 5: Global Hotkeys (100% CTkButtons)
        r_hk = tk.Frame(self.settings_body, bg=card)
        r_hk.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(r_hk, text="Batch:").pack(side="left")
        self.batch_hotkey_var = tk.StringVar(
            value=platform_mac.format_hotkey_label(self.config["hotkey_batch"])
        )
        self.batch_hotkey_btn = ctk.CTkButton(
            r_hk,
            textvariable=self.batch_hotkey_var,
            width=102,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=bg,
            hover_color=border,
            border_width=1,
            border_color=border,
            text_color=teal,
            command=lambda: self.start_hotkey_capture("batch"),
        )
        self.batch_hotkey_btn.pack(side="left", padx=4)

        ttk.Label(r_hk, text="Live:").pack(side="left", padx=(6, 0))
        self.live_hotkey_var = tk.StringVar(
            value=platform_mac.format_hotkey_label(self.config["hotkey_live"])
        )
        self.live_hotkey_btn = ctk.CTkButton(
            r_hk,
            textvariable=self.live_hotkey_var,
            width=92,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=bg,
            hover_color=border,
            border_width=1,
            border_color=border,
            text_color=teal,
            command=lambda: self.start_hotkey_capture("live"),
        )
        self.live_hotkey_btn.pack(side="left", padx=4)

        self.refresh_hotkeys_btn = ctk.CTkButton(
            r_hk,
            text="Rebind",
            width=62,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10),
            fg_color=bg,
            hover_color=border,
            border_width=1,
            border_color=border,
            text_color=THEME_COLORS["text_muted"],
            command=self._manual_hotkey_refresh,
        )
        self.refresh_hotkeys_btn.pack(side="right")

        r_hk_health = tk.Frame(self.settings_body, bg=card)
        r_hk_health.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(r_hk_health, text="Hotkey Hook:").pack(side="left")
        self.hotkey_health_var = tk.StringVar(value="Checking…")
        self.hotkey_health_label = tk.Label(
            r_hk_health,
            textvariable=self.hotkey_health_var,
            bg=card,
            fg=THEME_COLORS["text_muted"],
            font=ui_font(8, "bold"),
        )
        self.hotkey_health_label.pack(side="left", padx=6)

        # Row 6: Custom Data / Cloud Sync Folder (Dropbox / iCloud / Syncthing)
        sync_card = ctk.CTkFrame(
            self.settings_body,
            fg_color=bg,
            corner_radius=8,
            border_width=1,
            border_color=border,
        )
        sync_card.pack(fill="x", padx=6, pady=6)
        tk.Label(
            sync_card,
            text="Data & Cloud Sync Folder (iCloud / Dropbox)",
            bg=bg,
            fg=THEME_COLORS["fg"],
            font=ui_font(9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(6, 2))

        self.sync_dir_var = tk.StringVar(value=self._format_sync_dir_display())
        tk.Label(
            sync_card,
            textvariable=self.sync_dir_var,
            bg=bg,
            fg=THEME_COLORS["text_muted"],
            font=mono_font(8),
            anchor="w",
            justify="left",
            wraplength=390,
        ).pack(fill="x", padx=10, pady=(0, 6))

        sync_btns = tk.Frame(sync_card, bg=bg)
        sync_btns.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkButton(
            sync_btns,
            text="Choose Sync Folder…",
            width=142,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10),
            fg_color=card,
            hover_color=border,
            border_width=1,
            border_color=border,
            text_color=teal,
            command=self._choose_sync_folder,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            sync_btns,
            text="Reset to Default",
            width=114,
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=10),
            fg_color=card,
            hover_color=border,
            border_width=1,
            border_color=border,
            text_color=THEME_COLORS["text_muted"],
            command=self._reset_sync_folder,
        ).pack(side="left")

        # Row 7: OpenAI Cloud Settings
        oai_card = ctk.CTkFrame(
            self.settings_body,
            fg_color=bg,
            corner_radius=8,
            border_width=1,
            border_color=border,
        )
        oai_card.pack(fill="x", padx=6, pady=(4, 6))
        tk.Label(
            oai_card,
            text="OpenAI Cloud Settings",
            bg=bg,
            fg=THEME_COLORS["fg"],
            font=ui_font(9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(6, 4))

        key_row = tk.Frame(oai_card, bg=bg)
        key_row.pack(fill="x", padx=10, pady=2)
        tk.Label(key_row, text="API Key:", bg=bg, fg=THEME_COLORS["text_muted"], font=ui_font(8)).pack(
            side="left"
        )
        self.api_key_var = tk.StringVar(value=str(self.config.get("openai_api_key", "")))
        self.api_key_entry = ttk.Entry(
            key_row, textvariable=self.api_key_var, show="•", width=18
        )
        self.api_key_entry.pack(side="left", padx=4, fill="x", expand=True)

        self._api_key_shown = False
        self.btn_show_key = ctk.CTkButton(
            key_row,
            text="Show",
            width=46,
            height=22,
            corner_radius=6,
            font=ctk.CTkFont(size=9),
            fg_color=card,
            hover_color=border,
            text_color=THEME_COLORS["fg"],
            command=self._toggle_api_key_visibility,
        )
        self.btn_show_key.pack(side="left", padx=2)

        self.btn_save_key = ctk.CTkButton(
            key_row,
            text="Save",
            width=46,
            height=22,
            corner_radius=6,
            font=ctk.CTkFont(size=9, weight="bold"),
            fg_color=THEME_COLORS["teal_dim"],
            hover_color=border,
            text_color=teal,
            command=self._save_api_key,
        )
        self.btn_save_key.pack(side="left", padx=2)

        self.btn_test_key = ctk.CTkButton(
            key_row,
            text="Test",
            width=46,
            height=22,
            corner_radius=6,
            font=ctk.CTkFont(size=9),
            fg_color=card,
            hover_color=border,
            text_color=THEME_COLORS["text_muted"],
            command=self._test_api_key_async,
        )
        self.btn_test_key.pack(side="left", padx=(2, 0))

        oai_models_row = tk.Frame(oai_card, bg=bg)
        oai_models_row.pack(fill="x", padx=10, pady=(4, 8))
        tk.Label(
            oai_models_row, text="STT:", bg=bg, fg=THEME_COLORS["text_muted"], font=ui_font(8)
        ).pack(side="left")
        self.oai_stt_var = tk.StringVar(
            value=str(self.config.get("openai_transcription_model", "gpt-transcribe"))
        )
        ctk.CTkOptionMenu(
            oai_models_row,
            variable=self.oai_stt_var,
            values=OPENAI_TRANSCRIPTION_MODEL_OPTIONS,
            width=142,
            height=22,
            fg_color=card,
            button_color=border,
            text_color=THEME_COLORS["fg"],
            font=ctk.CTkFont(size=9),
            command=self._on_oai_model_changed,
        ).pack(side="left", padx=4)

        tk.Label(
            oai_models_row, text="Edit:", bg=bg, fg=THEME_COLORS["text_muted"], font=ui_font(8)
        ).pack(side="left", padx=(6, 0))
        self.oai_edit_model_var = tk.StringVar(
            value=str(self.config.get("openai_edit_model", "gpt-4o-mini"))
        )
        ctk.CTkOptionMenu(
            oai_models_row,
            variable=self.oai_edit_model_var,
            values=OPENAI_EDIT_MODEL_OPTIONS,
            width=116,
            height=22,
            fg_color=card,
            button_color=border,
            text_color=THEME_COLORS["fg"],
            font=ctk.CTkFont(size=9),
            command=self._on_oai_model_changed,
        ).pack(side="left", padx=4)

        self._update_config_summary()

    def _build_log_accordion_and_footer(self, parent: tk.Widget) -> None:
        self.log_collapsed = tk.BooleanVar(value=True)
        _, self.log_toggle_btn, self.log_body = self._make_accordion(
            parent,
            "System Log",
            None,
            self.toggle_log_section,
            collapsed=True,
        )
        self.log_text = tk.Text(
            self.log_body,
            height=5,
            wrap="word",
            bg=THEME_COLORS["bg"],
            fg=THEME_COLORS["text_muted"],
            highlightthickness=1,
            highlightbackground=THEME_COLORS["border"],
            relief="flat",
            font=mono_font(8),
            state="disabled",
        )
        self.log_text.pack(fill="x", padx=4, pady=(0, 4))

        footer = tk.Frame(parent, bg=THEME_COLORS["bg"])
        footer.pack(fill="x", pady=(2, 4))
        author_btn = ctk.CTkButton(
            footer,
            text=f"{APP_NAME} v{__version__} · Dr. Jonathan Mall",
            height=20,
            fg_color="transparent",
            hover_color=THEME_COLORS["accent_bg"],
            text_color=THEME_COLORS["text_faint"],
            font=ctk.CTkFont(size=9),
            command=lambda: webbrowser.open(AUTHOR_LINK_URL),
        )
        author_btn.pack(side="left")

        ctk.CTkButton(
            footer,
            text="OpenAI Keys ↗",
            height=20,
            width=90,
            fg_color="transparent",
            hover_color=THEME_COLORS["accent_bg"],
            text_color=THEME_COLORS["text_faint"],
            font=ctk.CTkFont(size=9),
            command=lambda: webbrowser.open(OPENAI_KEY_URL),
        ).pack(side="right")

    # ------------------------------------------------------------------
    # Scrollable Viewport & Darwin Trackpad MouseWheel
    # ------------------------------------------------------------------

    def _on_content_configure(self, _event=None) -> None:
        self._sync_scrollregion()

    def _on_canvas_configure(self, event) -> None:
        self._scroll_canvas.itemconfigure(self._scroll_window, width=event.width)
        self._sync_scrollregion()

    def _sync_scrollregion(self) -> None:
        bbox = self._scroll_canvas.bbox("all")
        if not bbox:
            return
        cw = self._scroll_canvas.winfo_width()
        ch = self._scroll_canvas.winfo_height()
        x1, y1, x2, y2 = bbox
        self._scroll_canvas.configure(scrollregion=(0, 0, max(cw, x2), max(ch, y2)))

    def _on_mousewheel(self, event) -> None:
        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0:
            return
        if sys.platform == "darwin":
            # Aqua Tk trackpads emit small +/-1..15 deltas rather than multiples of 120
            step = -delta if abs(delta) < 120 else -int(delta / 120)
        else:
            step = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        self._scroll_canvas.yview_scroll(step, "units")

    # ------------------------------------------------------------------
    # Accordion Toggles & Floating Mini Pill
    # ------------------------------------------------------------------

    def _toggle_accordion_generic(
        self,
        flag_var: tk.BooleanVar,
        btn: ctk.CTkButton,
        body: tk.Frame,
        title: str,
        cfg_key: Optional[str] = None,
    ) -> None:
        new_collapsed = not flag_var.get()
        flag_var.set(new_collapsed)
        btn.configure(text=f"{'▸' if new_collapsed else '▾'}  {title}")
        if new_collapsed:
            body.pack_forget()
        else:
            body.pack(fill="x", padx=6, pady=(0, 8))
        if cfg_key:
            self.config[cfg_key] = new_collapsed
            self._persist_config()
        self.root.after(20, self._sync_scrollregion)

    def toggle_history_section(self) -> None:
        self._toggle_accordion_generic(
            self.history_collapsed,
            self.history_toggle_btn,
            self.history_body,
            "Recent Transcriptions",
            "history_collapsed",
        )

    def toggle_insights_section(self) -> None:
        self._toggle_accordion_generic(
            self.insights_collapsed,
            self.insights_toggle_btn,
            self.insights_body,
            "Speech Insights & Activity",
            "insights_collapsed",
        )
        if not self.insights_collapsed.get():
            self.draw_histograms()
            self.refresh_insights_tabs()

    def toggle_settings_section(self) -> None:
        self._toggle_accordion_generic(
            self.config_collapsed,
            self.settings_toggle_btn,
            self.settings_body,
            "Configuration",
            "config_collapsed",
        )

    def toggle_log_section(self) -> None:
        self._toggle_accordion_generic(
            self.log_collapsed,
            self.log_toggle_btn,
            self.log_body,
            "System Log",
            None,
        )

    def _focus_configuration(self) -> None:
        if self.config_collapsed.get():
            self.toggle_settings_section()

    def _toggle_always_on_top(self) -> None:
        val = not self.top_var.get()
        self.top_var.set(val)
        self.config["always_on_top"] = val
        self._apply_always_on_top()
        self._persist_config()
        self.btn_header_top.configure(
            fg_color=THEME_COLORS["teal_dim"] if val else THEME_COLORS["accent_bg"],
            border_color=THEME_COLORS["teal"] if val else THEME_COLORS["border"],
            text_color=THEME_COLORS["teal"] if val else THEME_COLORS["text_muted"],
        )

    def _apply_always_on_top(self) -> None:
        try:
            self.root.attributes("-topmost", bool(self.top_var.get()))
        except Exception:
            pass

    def enter_mini_mode(self) -> None:
        self._ensure_mini_window()
        try:
            self.root.withdraw()
        except Exception:
            pass

    def exit_mini_mode(self) -> None:
        self._destroy_mini_window()
        try:
            self.root.deiconify()
            self.root.lift()
            self._apply_always_on_top()
        except Exception:
            pass

    def _on_root_unmap(self, event=None) -> None:
        if event is not None and event.widget is not self.root:
            return
        if self._shutting_down:
            return
        try:
            if self.root.state() == "iconic":
                self._ensure_mini_window()
        except Exception:
            pass

    def _on_root_map(self, event=None) -> None:
        if event is not None and event.widget is not self.root:
            return
        self._destroy_mini_window()

    def _ensure_mini_window(self) -> None:
        if self.mini_window is not None:
            return
        pill = tk.Toplevel(self.root)
        pill.title(APP_NAME)
        pill.configure(bg=THEME_COLORS["accent_bg"])
        pill.geometry(UI_RUNTIME["mini_window_default_geometry"])
        pill.resizable(False, False)
        platform_mac.apply_floating_pill_style(pill)

        row = tk.Frame(pill, bg=THEME_COLORS["accent_bg"])
        row.pack(fill="both", expand=True, padx=8, pady=6)

        self._mini_dot = tk.Canvas(
            row, width=12, height=12, bg=THEME_COLORS["accent_bg"], highlightthickness=0
        )
        self._mini_dot.pack(side="left", padx=(0, 6))
        self._mini_dot_oval = self._mini_dot.create_oval(
            2, 2, 10, 10, fill=THEME_COLORS["idle"], outline=""
        )

        ctk.CTkButton(
            row,
            text="Live",
            width=54,
            height=26,
            corner_radius=6,
            fg_color=THEME_COLORS["live"],
            text_color="#ffffff",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self.toggle_live,
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            row,
            text="Batch",
            width=58,
            height=26,
            corner_radius=6,
            fg_color=THEME_COLORS["batch"],
            text_color="#ffffff",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self.toggle_batch,
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            row,
            text="Expand",
            width=62,
            height=26,
            corner_radius=6,
            fg_color=THEME_COLORS["bg"],
            border_width=1,
            border_color=THEME_COLORS["border"],
            text_color=THEME_COLORS["fg"],
            font=ctk.CTkFont(size=10),
            command=self.exit_mini_mode,
        ).pack(side="right", padx=(4, 0))

        self.mini_window = pill
        self._sync_mini_pill_state()

    def _destroy_mini_window(self) -> None:
        win = self.mini_window
        self.mini_window = None
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    def _sync_mini_pill_state(self) -> None:
        if self.mini_window is None or not hasattr(self, "_mini_dot"):
            return
        color = (
            THEME_COLORS["live"]
            if self.mode == "live"
            else THEME_COLORS["batch"]
            if self.mode == "batch"
            else THEME_COLORS["idle"]
        )
        try:
            self._mini_dot.itemconfigure(self._mini_dot_oval, fill=color)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Settings Actions, Sync Folder & OpenAI Key Management
    # ------------------------------------------------------------------

    def _populate_microphones(self) -> None:
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if int(dev.get("max_input_channels", 0) or 0) > 0:
                    name = str(dev.get("name", f"Input {idx}")).strip()
                    self._mic_map[f"{name} (#{idx})"] = idx
        except Exception:
            pass

    def _on_mic_selected(self, _event=None) -> None:
        chosen = self.mic_var.get()
        self.config["input_device"] = self._mic_map.get(chosen)
        self._persist_config()
        self.log_internal(f"Selected microphone: {chosen}")

    def _on_pause_slider_changed(self, value: float) -> None:
        rounded = round(float(value), 1)
        self.config["live_pause"] = rounded
        self.lbl_pause_val.configure(text=f"{rounded:.1f}s")
        self._update_config_summary()
        self._persist_config()

    def _on_local_lang_changed(self, value: str) -> None:
        self.config["local_language"] = (value or "auto").strip().lower()
        self._persist_config()

    def _on_model_selected(self, _event=None) -> None:
        mk = self.model_var.get()
        self.config["model_key"] = mk
        self._persist_config()
        self._update_config_summary()
        if self.transcription_mode.get() == "local":
            self.reload_model_async()

    def _on_acceleration_changed(self, display_label: str) -> None:
        backend_name, device_name = BACKEND_DISPLAY_TO_BACKEND_DEVICE.get(
            display_label, ("auto", "auto")
        )
        self.config["backend"] = backend_name
        self.config["device"] = device_name
        self._persist_config()
        self._update_config_summary()
        if self.transcription_mode.get() == "local":
            self.reload_model_async()

    def _set_transcription_mode(self, mode: str) -> None:
        if mode not in ("local", "online"):
            return
        self.transcription_mode.set(mode)
        self.config["transcription_mode"] = mode
        self._persist_config()
        self._update_mode_buttons()
        self._update_backend_badge()
        if mode == "local" and (self.backend is None or not self.backend.is_model_loaded()):
            if os.environ.get("NEUROWHISPER_SKIP_MODEL_LOAD", "").strip() not in ("1", "true"):
                self.reload_model_async()

    def _toggle_online_edit(self) -> None:
        new_val = not self.openai_edit_var.get()
        self.openai_edit_var.set(new_val)
        self.config["openai_edit_enabled"] = new_val
        self._persist_config()
        self._update_mode_buttons()

    def _update_mode_buttons(self) -> None:
        is_online = self.transcription_mode.get() == "online"
        self.btn_mode_local.configure(
            fg_color=THEME_COLORS["teal_dim"] if not is_online else THEME_COLORS["bg"],
            text_color=THEME_COLORS["teal"] if not is_online else THEME_COLORS["text_muted"],
            border_width=1,
            border_color=THEME_COLORS["teal"] if not is_online else THEME_COLORS["border"],
        )
        self.btn_mode_online.configure(
            fg_color="#3b1d5e" if is_online else THEME_COLORS["bg"],
            text_color=THEME_COLORS["online"] if is_online else THEME_COLORS["text_muted"],
            border_width=1,
            border_color=THEME_COLORS["online"] if is_online else THEME_COLORS["border"],
        )
        edit_on = self.openai_edit_var.get()
        self.btn_online_edit.configure(
            fg_color="#3b1d5e" if (is_online and edit_on) else THEME_COLORS["bg"],
            text_color=THEME_COLORS["online"] if edit_on else THEME_COLORS["text_faint"],
            border_width=1,
            border_color=THEME_COLORS["online"] if edit_on else THEME_COLORS["border"],
        )

    def _format_sync_dir_display(self) -> str:
        sync = paths.get_sync_dir()
        if sync is not None:
            return f"Custom Sync: {sync}"
        return f"Default: {paths.get_app_support_dir()}"

    def _choose_sync_folder(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.root,
            title="Select Cloud Sync Folder (iCloud Drive / Dropbox / Syncthing)",
            initialdir=str(paths.get_data_dir()),
        )
        if not selected:
            return
        paths.set_sync_dir(selected)
        paths.ensure_runtime_dirs()
        self._ai_cache_mem = None
        self.sync_dir_var.set(self._format_sync_dir_display())
        self._load_stats()
        self._refresh_stats_labels()
        self.draw_histograms()
        self.refresh_insights_tabs()
        self.status_var.set("Updated cloud sync folder.")

    def _reset_sync_folder(self) -> None:
        paths.set_sync_dir("")
        paths.ensure_runtime_dirs()
        self._ai_cache_mem = None
        self.sync_dir_var.set(self._format_sync_dir_display())
        self._load_stats()
        self._refresh_stats_labels()
        self.draw_histograms()
        self.refresh_insights_tabs()
        self.status_var.set("Reverted to default macOS Application Support storage.")

    def _toggle_api_key_visibility(self) -> None:
        self._api_key_shown = not self._api_key_shown
        self.api_key_entry.configure(show="" if self._api_key_shown else "•")
        self.btn_show_key.configure(text="Hide" if self._api_key_shown else "Show")

    def _save_api_key(self) -> None:
        key = self.api_key_var.get().strip()
        self.config["openai_api_key"] = key
        self._persist_config()
        if self.online_backend is not None:
            self.online_backend.configure(api_key=key)
        self.status_var.set("OpenAI API key saved to local machine overlay.")

    def _on_oai_model_changed(self, _value: str = "") -> None:
        self.config["openai_transcription_model"] = self.oai_stt_var.get()
        self.config["openai_edit_model"] = self.oai_edit_model_var.get()
        self._persist_config()
        self._update_backend_badge()

    def _test_api_key_async(self) -> None:
        key = self.api_key_var.get().strip()
        if not key:
            self.status_var.set("Enter an OpenAI API key first.")
            return
        self.btn_test_key.configure(text="…", state="disabled")

        def worker() -> None:
            be = OpenAIBackend(api_key=key)
            ok, msg = be.test_api_key()
            self._post_ui(
                lambda: (
                    self.btn_test_key.configure(text="Test", state="normal"),
                    self.status_var.set(msg if ok else f"API Key Error: {msg[:60]}"),
                )
            )

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Model Loading & Hardware Acceleration Dispatch (MLX / CoreML / CPU)
    # ------------------------------------------------------------------

    def reload_model_async(self) -> None:
        if self._shutting_down:
            return
        model_key = str(self.config.get("model_key", DEFAULT_CONFIG["model_key"]))
        short_id = MODEL_MAP.get(model_key, model_key)
        dl_size = MODEL_DOWNLOAD_SIZES.get(short_id, "")
        size_hint = f" (~{dl_size} on first download)" if dl_size else ""
        self.status_var.set(f"Loading {model_key.split('(')[0].strip()}{size_hint}…")
        if hasattr(self, "btn_reload_model"):
            self.btn_reload_model.configure(state="disabled", text="…")

        def worker() -> None:
            try:
                with self._model_lock:
                    self._load_model_locked()
                self._post_ui(self._on_model_loaded_ui)
            except Exception as exc:
                logger.exception("Model load failed")
                err_msg = str(exc)
                self._post_ui(lambda: self._on_model_load_failed_ui(err_msg))

        threading.Thread(target=worker, name="neurowhisper-model-load", daemon=True).start()

    def _load_model_locked(self) -> None:
        model_key = str(self.config.get("model_key", DEFAULT_CONFIG["model_key"]))
        cfg_backend = str(self.config.get("backend", "auto") or "auto").strip().lower()
        cfg_device = str(self.config.get("device", "auto") or "auto").strip().lower()
        cfg_compute = str(self.config.get("compute_type", "auto") or "auto").strip().lower()

        if cfg_backend == "auto":
            resolved_backend, resolved_device, reason = detect_best_backend(model_key=model_key)
            self.log_internal(f"Auto-selected {resolved_backend} ({resolved_device}): {reason}")
        else:
            resolved_backend = cfg_backend
            # Preserve 'auto' / 'coreml' / 'mps' — NEVER clobber ONNX ASR or MLX to 'cpu'!
            resolved_device = cfg_device

        if self.backend is not None:
            try:
                self.backend.unload_model()
            except Exception:
                pass
            self.backend = None

        models_dir = str(paths.get_models_dir())
        try:
            be = create_backend(resolved_backend, model_key=model_key, worker_process=False)
            be.load_model(
                model_key,
                device=resolved_device,
                compute_type=cfg_compute,
                download_root=models_dir,
            )
        except Exception as primary_exc:
            # Automatic fallback to CPU Faster-Whisper if MLX fails on non-Parakeet models
            short = MODEL_MAP.get(model_key) or LEGACY_MODEL_ALIASES.get(model_key) or model_key
            if resolved_backend == "mlx-whisper" and short not in ("parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v2"):
                self.log_internal(
                    f"MLX load failed ({primary_exc}); falling back to Faster-Whisper CPU INT8."
                )
                be = create_backend("faster-whisper", model_key=model_key, worker_process=False)
                be.load_model(
                    model_key,
                    device="cpu",
                    compute_type="int8",
                    download_root=models_dir,
                )
            else:
                raise

        self.backend = be
        self.current_backend_name = be.name
        self.current_device = be.current_device
        self.last_activity_ts = time.monotonic()

    def _on_model_loaded_ui(self) -> None:
        if self._shutting_down:
            return
        if hasattr(self, "btn_reload_model"):
            self.btn_reload_model.configure(state="normal", text="Load")
        self._update_backend_badge()
        short = str(self.config.get("model_key", "")).split("(")[0].strip()
        disp = get_backend_display_name(self.current_backend_name)
        self.status_var.set(f"Ready · {short} ({disp})")
        if self.menubar is not None:
            self.menubar.set_state("idle", f"Ready ({short})")

    def _on_model_load_failed_ui(self, err_msg: str) -> None:
        if self._shutting_down:
            return
        if hasattr(self, "btn_reload_model"):
            self.btn_reload_model.configure(state="normal", text="Load")
        self.status_var.set(f"Model load error: {err_msg[:75]}")
        self.log_internal(f"ERROR loading model: {err_msg}")

    def _update_backend_badge(self) -> None:
        if self.transcription_mode.get() == "online":
            stt = self.config.get("openai_transcription_model", "gpt-transcribe")
            self.model_info_var.set(f"{stt} · cloud")
        else:
            short = str(self.config.get("model_key", "")).split("(")[0].strip()
            dev = self.current_device or self.config.get("device", "auto")
            self.model_info_var.set(f"{short} · {self.current_backend_name}/{dev}")
        self._update_config_summary()

    def _update_config_summary(self) -> None:
        if not hasattr(self, "config_summary_var"):
            return
        short = str(self.config.get("model_key", "")).split("(")[0].strip()
        dev = self.current_device or self.config.get("device", "auto")
        pause = float(self.config.get("live_pause", 0.8))
        self.config_summary_var.set(f"{short} · {dev} · {pause:.1f}s")

    def _idle_unload_tick(self) -> None:
        if self.mode is not None or self.backend is None or not self.backend.is_model_loaded():
            return
        mins = float(self.config.get("idle_unload_minutes", 5) or 0)
        if mins <= 0:
            return
        if (time.monotonic() - self.last_activity_ts) >= mins * 60.0:
            with self._model_lock:
                if self.backend is not None and self.mode is None:
                    self.backend.unload_model()
                    self.log_internal("Unloaded idle STT model to free unified memory.")
                    self.status_var.set("Idle (model unloaded — reloads automatically on dictate)")

    # ------------------------------------------------------------------
    # Global Hotkeys & Interactive Capture
    # ------------------------------------------------------------------

    def register_hotkeys(self) -> None:
        specs = {
            "live": (
                self.config["hotkey_live"],
                lambda: self._enqueue_hotkey_action("toggle_live"),
            ),
            "batch": (
                self.config["hotkey_batch"],
                lambda: self._enqueue_hotkey_action("toggle_batch"),
            ),
        }
        failed = self.hotkey_manager.register_all(specs)
        self._last_hotkey_register_ts = time.monotonic()
        self._update_hotkey_health_ui(failed)

    def _enqueue_hotkey_action(self, action: str) -> None:
        try:
            self.hotkey_action_queue.put_nowait(action)
        except queue.Full:
            pass

    def _update_hotkey_health_ui(self, failed: Optional[List[str]] = None) -> None:
        if not hasattr(self, "hotkey_health_var"):
            return
        expected = {
            "live": self.config["hotkey_live"],
            "batch": self.config["hotkey_batch"],
        }
        ok, detail = self.hotkey_manager.health_status(expected)
        if ok and not failed:
            self.hotkey_health_var.set("Active (Quartz CGEventTap)")
            self.hotkey_health_label.configure(fg=THEME_COLORS["ready"])
        elif detail == "accessibility_denied" or not platform_mac.is_accessibility_trusted(False):
            self.hotkey_health_var.set("Waiting for Accessibility permission")
            self.hotkey_health_label.configure(fg=THEME_COLORS["transcribing"])
        else:
            self.hotkey_health_var.set(f"Degraded ({detail})")
            self.hotkey_health_label.configure(fg=THEME_COLORS["live"])

    def _manual_hotkey_refresh(self) -> None:
        self.hotkey_manager._force_restart_listener()
        self.register_hotkeys()
        self._refresh_tcc_banner()

    def _hotkey_watchdog_tick(self) -> None:
        if self.capturing_hotkey is not None:
            return
        expected = {
            "live": self.config["hotkey_live"],
            "batch": self.config["hotkey_batch"],
        }
        ok, _ = self.hotkey_manager.health_status(expected)
        if not ok and platform_mac.is_accessibility_trusted(False):
            self._hotkey_auto_rebinds += 1
            self.hotkey_manager._force_restart_listener()
            self.register_hotkeys()
        else:
            self._update_hotkey_health_ui()

    def start_hotkey_capture(self, target: str) -> None:
        if target not in ("live", "batch"):
            return
        self.capturing_hotkey = target
        self.hotkey_manager.paused = True
        var = self.live_hotkey_var if target == "live" else self.batch_hotkey_var
        var.set("Press keys…")

        self.root.bind("<KeyPress>", self._on_tk_capture_keypress)
        self._capture_cancel_fn = self.hotkey_manager.capture_next_hotkey(
            on_complete=lambda combo: self._post_ui(lambda: self._finish_hotkey_capture(combo)),
            on_update=lambda parts: self._post_ui(
                lambda: var.set("+".join(parts) + "+" if parts else "Press keys…")
            ),
        )

    def _on_tk_capture_keypress(self, event) -> None:
        if self.capturing_hotkey is None:
            return
        if str(getattr(event, "keysym", "")).lower() == "escape":
            self._finish_hotkey_capture(None)
            return
        combo, preview = platform_mac.format_tk_keypress_combo(event)
        target = self.capturing_hotkey
        var = self.live_hotkey_var if target == "live" else self.batch_hotkey_var
        if combo is not None:
            self._finish_hotkey_capture(combo)
        elif preview:
            var.set("+".join(preview) + "+")

    def _finish_hotkey_capture(self, combo: Optional[str]) -> None:
        target = self.capturing_hotkey
        if target is None:
            return
        self.capturing_hotkey = None
        try:
            self.root.unbind("<KeyPress>")
        except Exception:
            pass
        if self._capture_cancel_fn is not None:
            try:
                self._capture_cancel_fn()
            except Exception:
                pass
            self._capture_cancel_fn = None

        cfg_key = f"hotkey_{target}"
        if combo:
            norm = platform_mac.normalize_hotkey_string(combo)
            self.config[cfg_key] = norm
            self._persist_config()

        self.live_hotkey_var.set(platform_mac.format_hotkey_label(self.config["hotkey_live"]))
        self.batch_hotkey_var.set(platform_mac.format_hotkey_label(self.config["hotkey_batch"]))
        self.btn_live.configure(
            text=f"Live Dictation ({platform_mac.format_hotkey_label(self.config['hotkey_live'])})"
        )
        self.btn_batch.configure(
            text=f"Batch ({platform_mac.format_hotkey_label(self.config['hotkey_batch'])})"
        )
        self.hotkey_manager.paused = False
        self.register_hotkeys()

    # ------------------------------------------------------------------
    # Live Dictation & Batch Recording Engine
    # ------------------------------------------------------------------

    def toggle_live(self) -> None:
        if self.mode == "live":
            self.stop_recording()
        elif self.mode == "batch":
            self.stop_recording()
        else:
            self._start_recording("live")

    def toggle_batch(self) -> None:
        if self.mode == "batch":
            self.stop_recording()
        elif self.mode == "live":
            self.stop_recording()
        else:
            self._start_recording("batch")

    def _ensure_active_backend_ready(self) -> WhisperBackend:
        if str(self.config.get("transcription_mode", "local")) == "online":
            key = str(
                self.config.get("openai_api_key", "") or os.environ.get("OPENAI_API_KEY", "")
            ).strip()
            if self.online_backend is None:
                self.online_backend = OpenAIBackend(api_key=key)
            self.online_backend.configure(
                api_key=key,
                transcription_model=str(
                    self.config.get("openai_transcription_model", "gpt-transcribe")
                ),
                edit_enabled=bool(self.config.get("openai_edit_enabled", False)),
                edit_model=str(self.config.get("openai_edit_model", "gpt-4o-mini")),
                edit_prompt=str(self.config.get("openai_edit_prompt", "")),
                language=str(self.config.get("openai_language", "auto")),
            )
            if not self.online_backend.is_model_loaded():
                raise RuntimeError("Configure an OpenAI API key in Configuration first.")
            return self.online_backend

        with self._model_lock:
            if self.backend is None or not self.backend.is_model_loaded():
                self._load_model_locked()
                self._post_ui(self._on_model_loaded_ui)
            assert self.backend is not None
            return self.backend

    def _start_recording(self, target_mode: str) -> None:
        if self.mode is not None or self._stopping_recording:
            return
        try:
            import sounddevice as sd

            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            dev = self.config.get("input_device")
            stream = sd.InputStream(
                samplerate=int(AUDIO_RUNTIME["sample_rate"]),
                channels=int(AUDIO_RUNTIME["channels"]),
                blocksize=int(AUDIO_RUNTIME["blocksize"]),
                dtype="float32",
                device=dev if dev is not None else None,
                callback=self._audio_callback,
            )
            stream.start()
            self.audio_stream = stream
        except Exception as exc:
            self.status_var.set(f"Microphone error: {exc}")
            self.log_internal(f"Microphone open failed: {exc}")
            if sys.platform == "darwin":
                platform_mac.open_microphone_settings()
            return

        self.mode = target_mode
        self.last_activity_ts = time.monotonic()
        platform_mac.play_feedback_sound_async(start=True)
        self._update_recording_controls_ui()

        if target_mode == "live":
            threading.Thread(
                target=self._live_dictation_worker,
                name="neurowhisper-live",
                daemon=True,
            ).start()
        else:
            threading.Thread(
                target=self._batch_recording_worker,
                name="neurowhisper-batch",
                daemon=True,
            ).start()

    def _audio_callback(self, indata: np.ndarray, frames: int, _time_info, status) -> None:
        if status:
            logger.debug("PortAudio status: %s", status)
        chunk = np.asarray(indata[:, 0], dtype=np.float32).copy()
        self.current_rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        self.audio_queue.put(chunk)

    def stop_recording(self) -> None:
        if self.mode is None or self._stopping_recording:
            return
        self._stopping_recording = True
        self.mode = None
        platform_mac.play_feedback_sound_async(start=False)
        self._close_audio_stream()
        self._stopping_recording = False
        self.current_rms = 0.0
        self._update_recording_controls_ui()

    def _close_audio_stream(self) -> None:
        stream = self.audio_stream
        self.audio_stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _live_dictation_worker(self) -> None:
        sr = int(AUDIO_RUNTIME["sample_rate"])
        chunk_s = float(AUDIO_RUNTIME["chunk_duration_seconds"])
        max_chunks = int(float(AUDIO_RUNTIME["live_max_chunk_seconds"]) / chunk_s)
        min_chunks = max(2, int(float(AUDIO_RUNTIME["live_min_chunk_seconds"]) / chunk_s))

        buffer: List[np.ndarray] = []
        silent_chunks = 0
        has_voice = False

        while self.mode == "live" and not self._shutting_down:
            try:
                block = self.audio_queue.get(timeout=float(AUDIO_RUNTIME["queue_timeout_seconds"]))
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
            threshold = float(self.config.get("silence_threshold", 0.015))
            pause_target = max(
                2, int(float(self.config.get("live_pause", 0.8)) / chunk_s)
            )

            buffer.append(block)
            if rms >= threshold:
                has_voice = True
                silent_chunks = 0
            elif has_voice:
                silent_chunks += 1

            should_flush = has_voice and (
                silent_chunks >= pause_target or len(buffer) >= max_chunks
            )
            if should_flush and len(buffer) >= min_chunks:
                audio_arr = np.concatenate(buffer)
                buffer.clear()
                silent_chunks = 0
                has_voice = False
                self._transcribe_and_emit_live(audio_arr, sr)
            elif not has_voice and len(buffer) > 8:
                # Keep small pre-roll before voice onset
                buffer = buffer[-5:]

        if has_voice and len(buffer) >= min_chunks:
            self._transcribe_and_emit_live(np.concatenate(buffer), sr)

    def _transcribe_and_emit_live(self, audio_arr: np.ndarray, sr: int) -> None:
        try:
            be = self._ensure_active_backend_ready()
            lang = (
                self.config.get("openai_language", "auto")
                if str(self.config.get("transcription_mode", "local")) == "online"
                else self.config.get("local_language", "auto")
            )
            segments = be.transcribe(audio_arr, sample_rate=sr, language=lang, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments if s.text and s.text.strip()).strip()
            if text:
                self.last_activity_ts = time.monotonic()
                platform_mac.type_text(text + " ", root=self.root)
                self._record_utterance(text, duration_s=float(audio_arr.size) / sr)
        except Exception as exc:
            logger.warning("Live transcription error: %s", exc)
            self._post_ui(lambda: self.status_var.set(f"Live STT error: {str(exc)[:65]}"))

    def _batch_recording_worker(self) -> None:
        sr = int(AUDIO_RUNTIME["sample_rate"])
        collected: List[np.ndarray] = []

        while self.mode == "batch" and not self._shutting_down:
            try:
                block = self.audio_queue.get(timeout=float(AUDIO_RUNTIME["queue_timeout_seconds"]))
                collected.append(block)
            except queue.Empty:
                continue

        while not self.audio_queue.empty():
            try:
                collected.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break

        if not collected or self._shutting_down:
            return

        full_audio = np.concatenate(collected)
        if full_audio.size < int(sr * 0.25):
            self._post_ui(lambda: self.status_var.set("Recording too short."))
            return

        self._post_ui(self._set_transcribing_ui)
        try:
            be = self._ensure_active_backend_ready()
            lang = (
                self.config.get("openai_language", "auto")
                if str(self.config.get("transcription_mode", "local")) == "online"
                else self.config.get("local_language", "auto")
            )
            segments = be.transcribe(full_audio, sample_rate=sr, language=lang, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments if s.text and s.text.strip()).strip()
            if text:
                self.last_activity_ts = time.monotonic()
                platform_mac.copy_to_clipboard(text, root=self.root)
                platform_mac.paste_from_clipboard(root=self.root)
                self._record_utterance(text, duration_s=float(full_audio.size) / sr)
                self._post_ui(lambda: self._on_batch_ready_ui(len(tokenize_words(text))))
            else:
                self._post_ui(lambda: self.status_var.set("No speech detected."))
                if self.menubar is not None:
                    self.menubar.set_state("idle", "Ready")
        except Exception as exc:
            logger.warning("Batch transcription error: %s", exc)
            err = str(exc)
            self._post_ui(lambda: self.status_var.set(f"Transcription failed: {err[:65]}"))
            if self.menubar is not None:
                self.menubar.set_state("error", "Transcription error")

    def _set_transcribing_ui(self) -> None:
        self.status_var.set("Transcribing…")
        if self.menubar is not None:
            self.menubar.set_state("transcribing", "Transcribing…")

    def _on_batch_ready_ui(self, word_count: int) -> None:
        self.status_var.set(f"Copied & pasted ({word_count} words)")
        if self.menubar is not None:
            self.menubar.set_state("ready", f"Copied ({word_count}w)")
            self._schedule_after(
                int(UI_RUNTIME["ready_state_delay_ms"]),
                lambda: self.menubar.set_state("idle", "Ready") if self.mode is None and self.menubar else None,
            )

    def _update_recording_controls_ui(self) -> None:
        live_lbl = platform_mac.format_hotkey_label(self.config["hotkey_live"])
        batch_lbl = platform_mac.format_hotkey_label(self.config["hotkey_batch"])

        if self.mode == "live":
            self.btn_live.configure(text=f"Stop Live ({live_lbl})", fg_color="#b91c1c")
            self.btn_batch.configure(state="disabled")
            self.btn_stop.configure(state="normal", text_color=THEME_COLORS["fg"])
            self.status_var.set("Recording Live Dictation…")
            if self.menubar is not None:
                st = "online" if self.transcription_mode.get() == "online" else "live"
                self.menubar.set_state(st, "Recording (Live)")
        elif self.mode == "batch":
            self.btn_batch.configure(text=f"Finish Batch ({batch_lbl})", fg_color="#1d4ed8")
            self.btn_live.configure(state="disabled")
            self.btn_stop.configure(state="normal", text_color=THEME_COLORS["fg"])
            self.status_var.set("Recording Batch Audio…")
            if self.menubar is not None:
                st = "online" if self.transcription_mode.get() == "online" else "batch"
                self.menubar.set_state(st, "Recording (Batch)")
        else:
            self.btn_live.configure(
                text=f"Live Dictation ({live_lbl})",
                fg_color=THEME_COLORS["live"],
                state="normal",
            )
            self.btn_batch.configure(
                text=f"Batch ({batch_lbl})",
                fg_color=THEME_COLORS["batch"],
                state="normal",
            )
            self.btn_stop.configure(state="disabled", text_color=THEME_COLORS["text_muted"])
            if self.menubar is not None:
                self.menubar.set_state("idle", "Ready")
        self._sync_mini_pill_state()

    # ------------------------------------------------------------------
    # Main UI Loop, Stats Labels, IPC & MenuBar Dispatch
    # ------------------------------------------------------------------

    def _main_ui_tick(self) -> None:
        # Drain queued UI callbacks
        for _ in range(16):
            try:
                cb = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                cb()
            except Exception as exc:
                logger.debug("Queued UI callback error: %s", exc)

        # Drain hotkey triggers
        while True:
            try:
                action = self.hotkey_action_queue.get_nowait()
            except queue.Empty:
                break
            if action == "toggle_live":
                self.toggle_live()
            elif action == "toggle_batch":
                self.toggle_batch()
            elif action == "stop":
                self.stop_recording()

        # Render live VU bar
        if hasattr(self, "vu_canvas"):
            self.vu_canvas.delete("all")
            w = self.vu_canvas.winfo_width() or 420
            level = min(1.0, self.current_rms * 14.0) if self.mode else 0.0
            if level > 0.01:
                color = THEME_COLORS["live"] if self.mode == "live" else THEME_COLORS["batch"]
                self.vu_canvas.create_rectangle(0, 0, w * level, 10, fill=color, width=0)

    def _safe_initial_visual_refresh(self) -> None:
        if self._shutting_down:
            return
        self.draw_histograms()
        if not self.insights_collapsed.get():
            self.refresh_insights_tabs()

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        s = int(round(max(0.0, seconds)))
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m"
        return f"{s / 3600.0:.1f}h"

    def _refresh_stats_labels(self) -> None:
        t_saved = max(0.0, (self.today_words / float(TYPING_WPM)) * 60.0 - self.today_duration)
        t_wpm = (
            int(round((self.today_words / self.today_duration) * 60.0))
            if self.today_duration > 1.0
            else 0
        )
        a_saved = max(0.0, (self.alltime_words / float(TYPING_WPM)) * 60.0 - self.alltime_duration)
        a_wpm = (
            int(round((self.alltime_words / self.alltime_duration) * 60.0))
            if self.alltime_duration > 1.0
            else 0
        )

        if hasattr(self, "lbl_today_words"):
            self.lbl_today_words.configure(text=f"{self.today_words:,}")
            self.lbl_today_duration.configure(text=self._fmt_duration(self.today_duration))
            self.lbl_today_time.configure(text=self._fmt_duration(t_saved))
            self.lbl_today_speed.configure(text=f"{t_wpm} wpm")

            self.lbl_alltime_words.configure(text=f"{self.alltime_words:,}")
            self.lbl_alltime_duration.configure(text=self._fmt_duration(self.alltime_duration))
            self.lbl_alltime_time.configure(text=self._fmt_duration(a_saved))
            self.lbl_alltime_speed.configure(text=f"{a_wpm} wpm")

        if hasattr(self, "insights_summary_var"):
            self.insights_summary_var.set(f"{self.today_words:,}w today")
        if hasattr(self, "status_stats_var"):
            self.status_stats_var.set(f"{self.today_words:,}w today · {self.alltime_words:,}w total")

    def _copy_last_utterance(self) -> None:
        if self.last_transcription_text:
            platform_mac.copy_to_clipboard(self.last_transcription_text, root=self.root)
            self.status_var.set("Copied last utterance to clipboard.")

    def _clear_history_view(self) -> None:
        if hasattr(self, "history_text"):
            self.history_text.configure(state="normal")
            self.history_text.delete("1.0", tk.END)
            self.history_text.configure(state="disabled")

    def open_transcriptions_folder(self) -> None:
        platform_mac.reveal_in_finder(paths.get_transcriptions_dir())

    def toggle_window_visibility(self) -> None:
        try:
            if self.root.state() in ("withdrawn", "iconic"):
                self.exit_mini_mode()
            else:
                self.enter_mini_mode()
        except Exception:
            self.exit_mini_mode()

    def show_window(self) -> None:
        self.exit_mini_mode()

    def attach_menubar(self, menubar: MenuBarController) -> None:
        self.menubar = menubar
        menubar._on_toggle_live = self.toggle_live
        menubar._on_toggle_batch = self.toggle_batch
        menubar._on_stop = self.stop_recording
        menubar._on_toggle_window = self.toggle_window_visibility
        menubar._on_open_transcriptions = self.open_transcriptions_folder
        menubar._on_quit = self.on_close
        menubar._dispatch_to_ui = self._post_ui

    def handle_ipc_command(
        self, cmd: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Thread-safe IPC command handler invoked by ``IPCServer``."""
        norm = (cmd or "").strip().lower()
        if norm == "toggle_live":
            self._post_ui(self.toggle_live)
        elif norm == "toggle_batch":
            self._post_ui(self.toggle_batch)
        elif norm == "stop":
            self._post_ui(self.stop_recording)
        elif norm == "show":
            self._post_ui(self.show_window)
        elif norm == "quit":
            self._post_ui(self.on_close)

        return {
            "mode": self.mode or "idle",
            "transcription_mode": str(self.config.get("transcription_mode", "local")),
            "backend": self.current_backend_name,
            "device": self.current_device,
            "model": self.config.get("model_key", ""),
            "today_words": self.today_words,
            "alltime_words": self.alltime_words,
        }

    def log_internal(self, message: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        logger.info(message)

        def append() -> None:
            if self._shutting_down or not hasattr(self, "log_text"):
                return
            self.log_text.configure(state="normal")
            self.log_text.insert(tk.END, line + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state="disabled")

        self._post_ui(append)

    # ------------------------------------------------------------------
    # Clean Shutdown
    # ------------------------------------------------------------------

    def on_close(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True

        # Cancel pending Tk timers
        if getattr(self, "_insights_refresh_job", None) is not None:
            try:
                self.root.after_cancel(self._insights_refresh_job)
            except Exception:
                pass
        for job in self._after_jobs:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
        self._after_jobs.clear()

        self._close_audio_stream()
        self._destroy_mini_window()

        try:
            self.hotkey_manager.cleanup()
        except Exception:
            pass

        if self.menubar is not None:
            try:
                self.menubar.stop()
            except Exception:
                pass

        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        try:
            self._persist_config()
            self._save_stats()
        except Exception:
            pass

        try:
            unload_all_backends()
        except Exception:
            pass

        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
