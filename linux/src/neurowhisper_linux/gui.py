"""Standalone Dark-Theme Linux GUI for ``neurowhisper`` (X11 & Wayland).

Features:
- 100% ``CustomTkinter`` buttons and dropdowns (zero ``tk.Button`` or ``tk.OptionMenu`` widgets)
- X11 ``<Button-4>``/``<Button-5>`` + ``<MouseWheel>`` smooth canvas scrolling
- Live dictation (real-time typing) & Batch recording (terminal-aware ``Ctrl+Shift+V`` / ``Ctrl+V``)
- Local Whisper (CPU INT8 / CUDA FP16 / OpenVINO) + Parakeet TDT 0.6B v3 (ONNX) + OpenAI Cloud (+Edit)
- Multi-machine sync-compatible config, stats, and transcript archive via ``paths.py``
- Wayland GlobalShortcuts portal + 1-click CLI IPC shortcut & ``.desktop`` launcher helpers
- User-space CUDA 12 runtime add-on installer dialog (`~/.local/share/neurowhisper/cuda-libs`)
- Full 10-tab Speech Insights + Today/All-Time Histograms via ``InsightsMixin``
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from tkinter import filedialog
from typing import Any, Callable, Dict, List, Optional, Tuple

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from . import APP_NAME, __version__, paths
from .backends import (
    FasterWhisperBackend,
    OpenAIBackend,
    create_backend,
    detect_best_backend,
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
    CUDA_ERROR_KEYWORDS,
    CUDA_PIP_PACKAGES,
    DEFAULT_CONFIG,
    HOTKEY_RUNTIME,
    IDLE_UNLOAD_CHECK_INTERVAL_SECONDS,
    LEGACY_MODEL_ALIASES,
    LIVE_PAUSE_RANGE,
    MACHINE_LOCAL_KEYS,
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
from .insights import InsightsMixin, parse_transcript_file
from .platform_linux import (
    LinuxHotkeyManager,
    copy_to_clipboard,
    detect_display_session,
    format_hotkey_combo,
    install_desktop_entry,
    normalize_key_token,
    open_path_or_url,
    paste_from_clipboard,
    play_feedback_sound_async,
    type_text,
)

log = logging.getLogger("neurowhisper.gui")

LOCAL_LANGUAGES = [
    ("Auto-detect", "auto"),
    ("English", "en"),
    ("German (Deutsch)", "de"),
    ("French (Français)", "fr"),
    ("Spanish (Español)", "es"),
    ("Italian (Italiano)", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
]
LOCAL_LANG_DISPLAY_TO_CODE = dict(LOCAL_LANGUAGES)
LOCAL_LANG_CODE_TO_DISPLAY = {code: disp for disp, code in LOCAL_LANGUAGES}

IDLE_UNLOAD_CHOICES = {
    "Never": 0,
    "2 min": 2,
    "5 min": 5,
    "15 min": 15,
    "30 min": 30,
}
IDLE_UNLOAD_REVERSE = {v: k for k, v in IDLE_UNLOAD_CHOICES.items()}


class WhisperApp(InsightsMixin):
    """Main ``neurowhisper`` Linux application controller and dark-mode UI."""

    def __init__(self, root: tk.Tk, tray_controller=None) -> None:
        self.root = root
        self.tray_controller = tray_controller
        self.colors = dict(THEME_COLORS)
        self.session_type = detect_display_session()

        paths.ensure_user_dirs()
        self.hostname = paths.get_hostname()

        # Runtime state
        self._shutting_down = False
        self.is_recording = False
        self.recording_mode: Optional[str] = None  # "live" | "batch"
        self.current_state = "idle"
        self._last_activity_ts = time.monotonic()
        self._last_idle_check_ts = time.monotonic()

        # Queues & worker executors
        self.ui_queue: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.hotkey_action_queue: "queue.Queue[str]" = queue.Queue(
            maxsize=int(HOTKEY_RUNTIME["action_queue_maxsize"])
        )
        self._audio_stream = None
        self._record_stop_event = threading.Event()
        self._current_vu_level: float = 0.0
        self._backend = None
        self._model_lock = threading.RLock()
        self._model_loading = False
        self._batch_executor: Optional[ThreadPoolExecutor] = None
        self._active_capture_cancel: Optional[Callable[[], None]] = None
        self._tk_capture_tokens: set[str] = set()
        self._mini_win: Optional[tk.Toplevel] = None

        # Load config, multi-machine stats, and archive history
        self.config = self.load_config()
        self.own_stats, self.stats = self.load_stats()
        self.transcription_history: List[Dict[str, str]] = self._load_recent_history()
        self.history_index: int = max(0, len(self.transcription_history) - 1)
        self._init_insights_state()

        # Configure root window
        self.root.title(APP_TITLE)
        self.root.geometry(APP_WINDOW_GEOMETRY)
        self.root.minsize(APP_WINDOW_MIN_WIDTH, APP_WINDOW_MIN_HEIGHT)
        self.root.configure(bg=self.colors["bg"])
        self._apply_window_icon()
        if bool(self.config.get("always_on_top", True)):
            try:
                self.root.attributes("-topmost", True)
            except tk.TclError:
                pass

        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close_button)
        self.root.bind("<Unmap>", self._on_root_unmap)
        self.root.bind("<Map>", self._on_root_map)

        # Build full dark UI
        self._build_ui()
        self._refresh_history_display()
        self._update_stats_cards()

        # Initialize Dual-Stack Linux Global Hotkeys
        self.hotkey_manager = LinuxHotkeyManager()
        self.register_hotkeys()

        # Start main UI event pump & background model warm-up
        self.root.after(int(UI_RUNTIME["initial_gui_loop_delay_ms"]), self._gui_update_loop)
        self.root.after(int(UI_RUNTIME["initial_histogram_delay_ms"]), self.draw_histogram)
        if str(self.config.get("transcription_mode", "local")) == "local":
            self.root.after(350, self.preload_model_async)
        else:
            self._set_app_state("ready", "Online Cloud mode ready")

    # ------------------------------------------------------------------
    # Window Icon & Lifecycle
    # ------------------------------------------------------------------

    def _apply_window_icon(self) -> None:
        icon_path = paths.get_icon_png_path()
        if not icon_path.is_file():
            return
        try:
            with Image.open(icon_path) as img:
                self._app_icon_photo = ImageTk.PhotoImage(img.convert("RGBA").resize((128, 128)))
                self.root.iconphoto(True, self._app_icon_photo)
        except Exception as exc:
            log.debug("Could not set window iconphoto: %s", exc)

    def set_tray_controller(self, tray_controller) -> None:
        self.tray_controller = tray_controller
        if self.tray_controller is not None:
            self.tray_controller.set_state(self.current_state)

    def _on_window_close_button(self) -> None:
        """Minimize to system tray if available; otherwise perform a clean shutdown."""
        if self.tray_controller is not None and self.tray_controller.available:
            self.root.withdraw()
            return
        self.shutdown()

    def toggle_window_visibility(self) -> None:
        """Show/focus or hide the main window (invoked by tray or ``--show``)."""
        self._enqueue_ui(self._toggle_window_visibility_main)

    def show_window(self) -> None:
        self._enqueue_ui(self._show_window_main)

    def _show_window_main(self) -> None:
        self._destroy_mini_window()
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except tk.TclError:
            pass

    def _toggle_window_visibility_main(self) -> None:
        try:
            if self.root.state() in ("withdrawn", "iconic") or not self.root.winfo_viewable():
                self._show_window_main()
            else:
                self.root.withdraw()
        except tk.TclError:
            self._show_window_main()

    def shutdown(self) -> None:
        """Gracefully stop recording, hotkeys, child workers, tray, and Tk mainloop."""
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            if self.is_recording:
                self._record_stop_event.set()
                self.is_recording = False
                self._close_audio_stream()
        except Exception:
            pass
        try:
            self.hotkey_manager.cleanup()
        except Exception:
            pass
        try:
            if self.tray_controller is not None:
                self.tray_controller.stop()
        except Exception:
            pass
        try:
            unload_all_backends()
        except Exception:
            pass
        try:
            self.save_config()
            self.save_stats()
        except Exception:
            pass
        try:
            self._destroy_mini_window()
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Multi-Machine Sync-Compatible Configuration & Statistics
    # ------------------------------------------------------------------

    def load_config(self) -> Dict[str, Any]:
        cfg = dict(DEFAULT_CONFIG)
        for path in (paths.get_shared_config_file(), paths.get_local_config_file(self.hostname)):
            if path.is_file():
                try:
                    import json

                    with open(path, "r", encoding="utf-8") as fh:
                        loaded = json.load(fh)
                    if isinstance(loaded, dict):
                        cfg.update(loaded)
                except Exception as exc:
                    log.warning("Could not read config %s: %s", path, exc)

        # Normalize legacy model display names if synced from an older Windows/macOS install
        mk = str(cfg.get("model_key", DEFAULT_CONFIG["model_key"]))
        if mk in LEGACY_MODEL_ALIASES:
            internal = LEGACY_MODEL_ALIASES[mk]
            cfg["model_key"] = MODEL_MAP_REVERSE.get(internal, DEFAULT_CONFIG["model_key"])
        elif mk not in MODEL_MAP and mk in MODEL_MAP_REVERSE:
            cfg["model_key"] = MODEL_MAP_REVERSE[mk]
        elif mk not in MODEL_MAP:
            cfg["model_key"] = DEFAULT_CONFIG["model_key"]

        return cfg

    def save_config(self) -> None:
        shared_payload = {k: v for k, v in self.config.items() if k not in MACHINE_LOCAL_KEYS}
        local_payload = {k: self.config[k] for k in MACHINE_LOCAL_KEYS if k in self.config}
        try:
            paths.write_json_atomic(paths.get_shared_config_file(), shared_payload, indent=2)
            paths.write_json_atomic(paths.get_local_config_file(self.hostname), local_payload, indent=2)
        except Exception as exc:
            log.warning("Failed to save configuration: %s", exc)

    @staticmethod
    def _empty_stats() -> Dict[str, Any]:
        return {
            "total_words": 0,
            "total_audio_duration": 0.0,
            "total_latencies": [],
            "today_data": {"date": date.today().isoformat(), "words": 0},
            "hourly_data": {},
        }

    def load_stats(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import json

        own = self._empty_stats()
        own_file = paths.get_stats_file(self.hostname)
        legacy_file = paths.get_legacy_stats_file()

        if own_file.is_file():
            try:
                with open(own_file, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    own.update(raw)
            except Exception as exc:
                log.warning("Failed reading %s: %s", own_file, exc)
        elif legacy_file.is_file():
            try:
                with open(legacy_file, "r", encoding="utf-8") as fh:
                    leg = json.load(fh)
                if isinstance(leg, dict) and not leg.get("claimed_by"):
                    own.update({k: v for k, v in leg.items() if k in own})
                    leg["claimed_by"] = self.hostname
                    paths.write_json_atomic(legacy_file, leg, indent=2)
                    paths.write_json_atomic(own_file, own, indent=2)
            except Exception:
                pass

        today_iso = date.today().isoformat()
        if not isinstance(own.get("today_data"), dict) or own["today_data"].get("date") != today_iso:
            own["today_data"] = {"date": today_iso, "words": 0}
        if not isinstance(own.get("hourly_data"), dict):
            own["hourly_data"] = {}

        merged = self._merge_peer_stats(own)
        return own, merged

    def _merge_peer_stats(self, own: Dict[str, Any]) -> Dict[str, Any]:
        import json

        today_iso = date.today().isoformat()
        total_words = int(own.get("total_words", 0))
        total_dur = float(own.get("total_audio_duration", 0.0))
        latencies = list(own.get("total_latencies") or [])[-200:]
        hourly_merged: Dict[str, Dict[str, int]] = {}

        for day_k, hmap in (own.get("hourly_data") or {}).items():
            if isinstance(hmap, dict):
                hourly_merged[day_k] = {str(h): int(c) for h, c in hmap.items()}

        stats_dir = paths.get_stats_dir()
        own_name = paths.get_stats_file(self.hostname).name
        if stats_dir.is_dir():
            for peer_file in sorted(stats_dir.glob("whisper_stats.*.json")):
                if peer_file.name == own_name or "conflicted copy" in peer_file.name.lower():
                    continue
                try:
                    with open(peer_file, "r", encoding="utf-8") as fh:
                        peer = json.load(fh)
                    if not isinstance(peer, dict):
                        continue
                    total_words += int(peer.get("total_words", 0))
                    total_dur += float(peer.get("total_audio_duration", 0.0))
                    for d_key, p_hours in (peer.get("hourly_data") or {}).items():
                        if not isinstance(p_hours, dict):
                            continue
                        dest = hourly_merged.setdefault(d_key, {})
                        for hr_k, cnt in p_hours.items():
                            dest[str(hr_k)] = dest.get(str(hr_k), 0) + int(cnt)
                except Exception:
                    continue

        today_words = sum(hourly_merged.get(today_iso, {}).values())
        own_today = int((own.get("today_data") or {}).get("words", 0))
        today_words = max(today_words, own_today)

        return {
            "total_words": total_words,
            "total_audio_duration": total_dur,
            "total_latencies": latencies,
            "today_data": {"date": today_iso, "words": today_words},
            "hourly_data": hourly_merged,
        }

    def save_stats(self) -> None:
        try:
            paths.write_json_atomic(paths.get_stats_file(self.hostname), self.own_stats, indent=2)
        except Exception as exc:
            log.warning("Could not save stats: %s", exc)

    def _record_utterance_stats_and_archive(
        self,
        text: str,
        audio_duration_sec: float,
        latency_sec: float,
    ) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return

        now = datetime.now()
        today_iso = now.strftime("%Y-%m-%d")
        hour_key = f"{now.hour:02d}"
        month_prefix = now.strftime("%Y-%m")
        word_count = len(cleaned.split())

        # 1. Append to monthly per-host archive file (100% compatible with Windows/macOS sync)
        archive_dir = paths.get_transcriptions_dir()
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_file = archive_dir / f"{month_prefix}.{self.hostname}.txt"
        stamp = now.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(archive_file, "a", encoding="utf-8") as fh:
                fh.write(f"[{stamp}] {cleaned}\n")
        except OSError as exc:
            log.warning("Failed writing transcription archive %s: %s", archive_file, exc)

        # 2. Update machine-local & merged stats
        self.own_stats["total_words"] = int(self.own_stats.get("total_words", 0)) + word_count
        self.own_stats["total_audio_duration"] = float(
            self.own_stats.get("total_audio_duration", 0.0)
        ) + max(0.0, float(audio_duration_sec))
        lats = list(self.own_stats.get("total_latencies") or [])
        lats.append(round(float(latency_sec), 3))
        self.own_stats["total_latencies"] = lats[-200:]

        if (self.own_stats.get("today_data") or {}).get("date") != today_iso:
            self.own_stats["today_data"] = {"date": today_iso, "words": 0}
        self.own_stats["today_data"]["words"] = int(self.own_stats["today_data"].get("words", 0)) + word_count

        hourly = self.own_stats.setdefault("hourly_data", {})
        day_bucket = hourly.setdefault(today_iso, {})
        day_bucket[hour_key] = int(day_bucket.get(hour_key, 0)) + word_count

        self.save_stats()
        self.stats = self._merge_peer_stats(self.own_stats)

        # 3. Append to in-memory history viewer
        self.transcription_history.append(
            {
                "timestamp": stamp,
                "text": cleaned,
            }
        )
        if len(self.transcription_history) > 500:
            self.transcription_history = self.transcription_history[-500:]
        self.history_index = len(self.transcription_history) - 1

        self._enqueue_ui(self._after_utterance_recorded_ui)

    def _after_utterance_recorded_ui(self) -> None:
        self._refresh_history_display()
        self._update_stats_cards()
        self.draw_histogram()

    def _load_recent_history(self) -> List[Dict[str, str]]:
        folder = paths.get_transcriptions_dir()
        if not folder.is_dir():
            return []
        collected: List[Tuple[datetime, str]] = []
        for txt_path in sorted(folder.glob("*.txt"))[-6:]:
            if "conflicted copy" in txt_path.name.lower():
                continue
            for item in parse_transcript_file(txt_path):
                collected.append((item["timestamp"], item["text"]))
        collected.sort(key=lambda pair: pair[0])
        return [
            {"timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"), "text": body}
            for dt, body in collected[-250:]
        ]

    # ------------------------------------------------------------------
    # Dark-Mode User Interface Construction (100% CTk Controls)
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Scrollable outer container (supports X11 Button-4/5 + MouseWheel)
        outer = tk.Frame(self.root, bg=self.colors["bg"])
        outer.pack(fill="both", expand=True)

        self.main_canvas = tk.Canvas(
            outer,
            bg=self.colors["bg"],
            highlightthickness=0,
            bd=0,
        )
        self.scrollbar = ctk.CTkScrollbar(
            outer,
            orientation="vertical",
            command=self.main_canvas.yview,
            fg_color=self.colors["bg"],
            button_color=self.colors["border"],
            button_hover_color=self.colors["teal_dim"],
        )
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)

        self.content_frame = tk.Frame(self.main_canvas, bg=self.colors["bg"], padx=14, pady=10)
        self._canvas_window_id = self.main_canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )

        self.content_frame.bind(
            "<Configure>",
            lambda _e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")),
        )
        self.main_canvas.bind(
            "<Configure>",
            lambda e: self.main_canvas.itemconfigure(self._canvas_window_id, width=e.width),
        )
        self._bind_linux_scroll_events()

        # 1. Header Bar + Display Session Badge
        self._build_header_section(self.content_frame)

        # 2. Primary Recording Controls + VU Meter + Status Line
        self._build_recording_controls(self.content_frame)

        # 3. Quick KPI Summary Row (Today Words / All-Time Words / Time Saved)
        self._build_kpi_bar(self.content_frame)

        # 4. Accordion Sections (100% CTkButton toggles)
        self._accordions: Dict[str, Dict[str, Any]] = {}
        self._create_accordion(
            self.content_frame,
            key="history",
            title="Recent Transcriptions",
            expanded=True,
            builder=self._build_history_section,
        )
        self._create_accordion(
            self.content_frame,
            key="histograms",
            title="Word Volume Histograms",
            expanded=True,
            builder=self.build_histograms_content,
        )
        self._create_accordion(
            self.content_frame,
            key="insights",
            title="Speech Insights & AI Brief",
            expanded=False,
            builder=self.build_speech_insights_content,
        )
        self._create_accordion(
            self.content_frame,
            key="config",
            title="Configuration (Models, Wayland/X11, Sync)",
            expanded=False,
            builder=self._build_configuration_section,
        )

        # 5. Footer Attribution
        footer = tk.Frame(self.content_frame, bg=self.colors["bg"], pady=6)
        footer.pack(fill="x")
        ctk.CTkButton(
            footer,
            text=f"{APP_NAME} v{__version__} for Linux  •  by Dr. Jonathan Mall",
            height=22,
            fg_color="transparent",
            hover_color=self.colors["accent_bg"],
            text_color=self.colors["text_faint"],
            font=ui_font(9),
            command=lambda: open_path_or_url(AUTHOR_LINK_URL),
        ).pack()

    def _bind_linux_scroll_events(self) -> None:
        """Bind both X11 ``<Button-4>``/``<Button-5>`` and ``<MouseWheel>`` across the UI."""
        def _is_scrollable_text_widget(widget) -> bool:
            return isinstance(widget, tk.Text)

        def _on_wheel_up(event) -> None:
            if _is_scrollable_text_widget(event.widget):
                return
            self.main_canvas.yview_scroll(-3, "units")

        def _on_wheel_down(event) -> None:
            if _is_scrollable_text_widget(event.widget):
                return
            self.main_canvas.yview_scroll(3, "units")

        def _on_mousewheel(event) -> None:
            if _is_scrollable_text_widget(event.widget):
                return
            delta = getattr(event, "delta", 0)
            if delta > 0:
                self.main_canvas.yview_scroll(-3, "units")
            elif delta < 0:
                self.main_canvas.yview_scroll(3, "units")

        self.root.bind_all("<Button-4>", _on_wheel_up, add="+")
        self.root.bind_all("<Button-5>", _on_wheel_down, add="+")
        self.root.bind_all("<MouseWheel>", _on_mousewheel, add="+")

    def _build_header_section(self, parent: tk.Frame) -> None:
        hdr = tk.Frame(parent, bg=self.colors["bg"])
        hdr.pack(fill="x", pady=(0, 8))

        left = tk.Frame(hdr, bg=self.colors["bg"])
        left.pack(side="left")

        self.status_dot = tk.Canvas(
            left,
            width=12,
            height=12,
            bg=self.colors["bg"],
            highlightthickness=0,
        )
        self.status_dot.pack(side="left", padx=(0, 7))
        self._status_dot_id = self.status_dot.create_oval(2, 2, 10, 10, fill=self.colors["idle"], outline="")

        tk.Label(
            left,
            text=APP_NAME,
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            font=ui_font(16, "bold"),
        ).pack(side="left")

        sess_badge_text = self.session_type.upper()
        tk.Label(
            hdr,
            text=f" {sess_badge_text} ",
            bg=self.colors["accent_bg"],
            fg=self.colors["teal"],
            font=mono_font(9, "bold"),
            padx=6,
            pady=2,
        ).pack(side="right")

    def _build_recording_controls(self, parent: tk.Frame) -> None:
        card = tk.Frame(
            parent,
            bg=self.colors["accent_bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=10,
        )
        card.pack(fill="x", pady=(0, 8))

        btn_row = tk.Frame(card, bg=self.colors["accent_bg"])
        btn_row.pack(fill="x")

        live_hk = str(self.config.get("hotkey_live", "f9")).upper()
        batch_hk = str(self.config.get("hotkey_batch", "ctrl+alt+s")).upper()

        self.btn_live = ctk.CTkButton(
            btn_row,
            text=f"Live Dictation ({live_hk})",
            height=40,
            corner_radius=8,
            font=ui_font(12, "bold"),
            fg_color=self.colors["teal_dim"],
            hover_color=self.colors["teal"],
            text_color=self.colors["fg"],
            command=self.toggle_live_recording,
        )
        self.btn_live.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.btn_batch = ctk.CTkButton(
            btn_row,
            text=f"Batch Record ({batch_hk})",
            height=40,
            corner_radius=8,
            font=ui_font(12, "bold"),
            fg_color=self.colors["border"],
            hover_color=self.colors["batch"],
            text_color=self.colors["fg"],
            command=self.toggle_batch_recording,
        )
        self.btn_batch.pack(side="left", expand=True, fill="x", padx=(5, 0))

        # Smooth VU Audio Level Bar
        self.vu_canvas = tk.Canvas(
            card,
            height=6,
            bg=self.colors["bg"],
            highlightthickness=0,
        )
        self.vu_canvas.pack(fill="x", pady=(8, 4))
        self._vu_rect = self.vu_canvas.create_rectangle(0, 0, 0, 6, fill=self.colors["teal"], outline="")

        status_row = tk.Frame(card, bg=self.colors["accent_bg"])
        status_row.pack(fill="x")

        self.status_label = tk.Label(
            status_row,
            text="Initializing...",
            bg=self.colors["accent_bg"],
            fg=self.colors["text_muted"],
            font=ui_font(10),
            anchor="w",
        )
        self.status_label.pack(side="left", fill="x", expand=True)

        self.engine_badge_label = tk.Label(
            status_row,
            text="",
            bg=self.colors["accent_bg"],
            fg=self.colors["teal"],
            font=mono_font(9),
            anchor="e",
        )
        self.engine_badge_label.pack(side="right")

    def _build_kpi_bar(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=self.colors["bg"])
        row.pack(fill="x", pady=(0, 8))

        self._kpi_labels: Dict[str, tk.Label] = {}
        specs = [
            ("today", "Today Words"),
            ("total", "Total Words"),
            ("saved", "Time Saved"),
        ]
        for idx, (key, caption) in enumerate(specs):
            box = tk.Frame(
                row,
                bg=self.colors["accent_bg"],
                highlightthickness=1,
                highlightbackground=self.colors["border"],
                padx=8,
                pady=6,
            )
            box.pack(side="left", expand=True, fill="both", padx=(0 if idx == 0 else 6, 0))
            val_lbl = tk.Label(
                box,
                text="0",
                bg=self.colors["accent_bg"],
                fg=self.colors["teal"],
                font=ui_font(13, "bold"),
            )
            val_lbl.pack(anchor="w")
            tk.Label(
                box,
                text=caption,
                bg=self.colors["accent_bg"],
                fg=self.colors["text_muted"],
                font=ui_font(9),
            ).pack(anchor="w")
            self._kpi_labels[key] = val_lbl

    def _update_stats_cards(self) -> None:
        if not hasattr(self, "_kpi_labels"):
            return
        today_w = int((self.stats.get("today_data") or {}).get("words", 0))
        total_w = int(self.stats.get("total_words", 0))
        saved_min = total_w / float(TYPING_WPM)
        saved_str = f"{saved_min / 60.0:.1f} hrs" if saved_min >= 90 else f"{int(round(saved_min))} min"

        self._kpi_labels["today"].configure(text=f"{today_w:,}")
        self._kpi_labels["total"].configure(text=f"{total_w:,}")
        self._kpi_labels["saved"].configure(text=saved_str)

    def _create_accordion(
        self,
        parent: tk.Frame,
        key: str,
        title: str,
        expanded: bool,
        builder: Callable[[tk.Frame], None],
    ) -> None:
        wrapper = tk.Frame(
            parent,
            bg=self.colors["accent_bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        wrapper.pack(fill="x", pady=(0, 8))

        arrow = "▼" if expanded else "▶"
        toggle_btn = ctk.CTkButton(
            wrapper,
            text=f"{arrow}  {title}",
            anchor="w",
            height=32,
            corner_radius=0,
            font=ui_font(11, "bold"),
            fg_color=self.colors["accent_bg"],
            hover_color=self.colors["border"],
            text_color=self.colors["fg"],
            command=lambda k=key: self._toggle_accordion(k),
        )
        toggle_btn.pack(fill="x")

        body = tk.Frame(wrapper, bg=self.colors["accent_bg"], padx=10, pady=8)
        if expanded:
            body.pack(fill="x")

        self._accordions[key] = {
            "title": title,
            "expanded": expanded,
            "button": toggle_btn,
            "body": body,
            "built": expanded,
            "builder": builder,
        }
        if expanded:
            builder(body)

    def _toggle_accordion(self, key: str) -> None:
        acc = self._accordions.get(key)
        if not acc:
            return
        acc["expanded"] = not acc["expanded"]
        arrow = "▼" if acc["expanded"] else "▶"
        acc["button"].configure(text=f"{arrow}  {acc['title']}")
        if acc["expanded"]:
            if not acc["built"]:
                acc["builder"](acc["body"])
                acc["built"] = True
            acc["body"].pack(fill="x")
        else:
            acc["body"].pack_forget()

    # ------------------------------------------------------------------
    # History Section
    # ------------------------------------------------------------------

    def _build_history_section(self, parent: tk.Frame) -> None:
        nav = tk.Frame(parent, bg=self.colors["accent_bg"])
        nav.pack(fill="x", pady=(0, 6))

        self.btn_hist_prev = ctk.CTkButton(
            nav,
            text="◀",
            width=32,
            height=26,
            corner_radius=5,
            font=ui_font(11, "bold"),
            fg_color=self.colors["bg"],
            hover_color=self.colors["border"],
            command=self._history_prev,
        )
        self.btn_hist_prev.pack(side="left")

        self.history_pos_label = tk.Label(
            nav,
            text="0 / 0",
            bg=self.colors["accent_bg"],
            fg=self.colors["text_muted"],
            font=mono_font(9),
            padx=8,
        )
        self.history_pos_label.pack(side="left")

        self.btn_hist_next = ctk.CTkButton(
            nav,
            text="▶",
            width=32,
            height=26,
            corner_radius=5,
            font=ui_font(11, "bold"),
            fg_color=self.colors["bg"],
            hover_color=self.colors["border"],
            command=self._history_next,
        )
        self.btn_hist_next.pack(side="left")

        ctk.CTkButton(
            nav,
            text="Folder",
            width=62,
            height=26,
            corner_radius=5,
            font=ui_font(10),
            fg_color=self.colors["bg"],
            hover_color=self.colors["border"],
            command=lambda: open_path_or_url(paths.get_transcriptions_dir()),
        ).pack(side="right", padx=(4, 0))

        ctk.CTkButton(
            nav,
            text="Copy",
            width=58,
            height=26,
            corner_radius=5,
            font=ui_font(10, "bold"),
            fg_color=self.colors["teal_dim"],
            hover_color=self.colors["teal"],
            command=self._copy_current_history_entry,
        ).pack(side="right")

        self.history_text = tk.Text(
            parent,
            height=5,
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            insertbackground=self.colors["fg"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            font=ui_font(10),
            wrap="word",
            padx=8,
            pady=6,
        )
        self.history_text.pack(fill="x")

    def _refresh_history_display(self) -> None:
        if not hasattr(self, "history_text"):
            return
        total = len(self.transcription_history)
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")

        if total == 0:
            self.history_pos_label.configure(text="0 / 0")
            self.history_text.insert("1.0", "Your transcriptions will appear here.")
        else:
            self.history_index = max(0, min(self.history_index, total - 1))
            rec = self.transcription_history[self.history_index]
            self.history_pos_label.configure(
                text=f"{self.history_index + 1} / {total}  •  {rec['timestamp']}"
            )
            self.history_text.insert("1.0", rec["text"])
        self.history_text.configure(state="disabled")

    def _history_prev(self) -> None:
        if self.history_index > 0:
            self.history_index -= 1
            self._refresh_history_display()

    def _history_next(self) -> None:
        if self.history_index < len(self.transcription_history) - 1:
            self.history_index += 1
            self._refresh_history_display()

    def _copy_current_history_entry(self) -> None:
        if not self.transcription_history:
            return
        rec = self.transcription_history[self.history_index]
        copy_to_clipboard(rec["text"], root=self.root)
        self.set_status("Copied transcription to clipboard")

    # ------------------------------------------------------------------
    # Configuration Section (Local / Online / Audio / Wayland+X11 / Sync)
    # ------------------------------------------------------------------

    def _build_configuration_section(self, parent: tk.Frame) -> None:
        # Mode Toggle: Local vs Online (OpenAI Cloud)
        mode_row = tk.Frame(parent, bg=self.colors["accent_bg"])
        mode_row.pack(fill="x", pady=(0, 8))

        cur_mode = str(self.config.get("transcription_mode", "local"))
        self.btn_mode_local = ctk.CTkButton(
            mode_row,
            text="Local Inference (Offline)",
            height=28,
            corner_radius=6,
            font=ui_font(10, "bold" if cur_mode == "local" else None),
            fg_color=self.colors["teal_dim"] if cur_mode == "local" else self.colors["bg"],
            hover_color=self.colors["teal"],
            command=lambda: self._switch_transcription_mode("local"),
        )
        self.btn_mode_local.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.btn_mode_online = ctk.CTkButton(
            mode_row,
            text="OpenAI Cloud (+GPT Edit)",
            height=28,
            corner_radius=6,
            font=ui_font(10, "bold" if cur_mode != "local" else None),
            fg_color=self.colors["online"] if cur_mode != "local" else self.colors["bg"],
            hover_color="#9333ea",
            command=lambda: self._switch_transcription_mode("online"),
        )
        self.btn_mode_online.pack(side="left", expand=True, fill="x", padx=(4, 0))

        # Local Model Frame
        self.local_cfg_frame = tk.Frame(
            parent,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=8,
        )
        self._build_local_model_controls(self.local_cfg_frame)

        # Online OpenAI Frame
        self.online_cfg_frame = tk.Frame(
            parent,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=8,
        )
        self._build_online_controls(self.online_cfg_frame)

        if cur_mode == "local":
            self.local_cfg_frame.pack(fill="x", pady=(0, 8))
        else:
            self.online_cfg_frame.pack(fill="x", pady=(0, 8))

        # Audio & Timing Controls
        self._build_audio_controls(parent)

        # Global Hotkeys & Wayland / Desktop Integration
        self._build_hotkeys_and_wayland_controls(parent)

        # Custom Data / Sync Folder (`storage.json`)
        self._build_sync_folder_controls(parent)

    def _build_local_model_controls(self, box: tk.Frame) -> None:
        tk.Label(box, text="Speech Model:", bg=self.colors["bg"], fg=self.colors["text_muted"], font=ui_font(10)).pack(
            anchor="w"
        )
        self.model_menu = ctk.CTkOptionMenu(
            box,
            values=list(MODEL_MAP.keys()),
            height=28,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            button_color=self.colors["border"],
            command=self._on_model_selected,
        )
        self.model_menu.set(str(self.config.get("model_key", DEFAULT_CONFIG["model_key"])))
        self.model_menu.pack(fill="x", pady=(2, 6))

        tk.Label(
            box,
            text="Hardware Acceleration:",
            bg=self.colors["bg"],
            fg=self.colors["text_muted"],
            font=ui_font(10),
        ).pack(anchor="w")

        self.backend_menu = ctk.CTkOptionMenu(
            box,
            values=BACKEND_OPTIONS,
            height=28,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            button_color=self.colors["border"],
            command=self._on_backend_selected,
        )
        disp_backend = resolve_backend_display(
            str(self.config.get("backend", "auto")),
            str(self.config.get("device", "auto")),
        )
        self.backend_menu.set(disp_backend)
        self.backend_menu.pack(fill="x", pady=(2, 6))

        lang_idle_row = tk.Frame(box, bg=self.colors["bg"])
        lang_idle_row.pack(fill="x", pady=(2, 6))

        left_col = tk.Frame(lang_idle_row, bg=self.colors["bg"])
        left_col.pack(side="left", expand=True, fill="x", padx=(0, 4))
        tk.Label(left_col, text="Language:", bg=self.colors["bg"], fg=self.colors["text_muted"], font=ui_font(9)).pack(
            anchor="w"
        )
        self.local_lang_menu = ctk.CTkOptionMenu(
            left_col,
            values=[disp for disp, _ in LOCAL_LANGUAGES],
            height=26,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            button_color=self.colors["border"],
            command=self._on_local_lang_selected,
        )
        cur_code = str(self.config.get("local_language", "auto"))
        self.local_lang_menu.set(LOCAL_LANG_CODE_TO_DISPLAY.get(cur_code, "Auto-detect"))
        self.local_lang_menu.pack(fill="x")

        right_col = tk.Frame(lang_idle_row, bg=self.colors["bg"])
        right_col.pack(side="left", expand=True, fill="x", padx=(4, 0))
        tk.Label(
            right_col, text="Idle Unload:", bg=self.colors["bg"], fg=self.colors["text_muted"], font=ui_font(9)
        ).pack(anchor="w")
        self.idle_unload_menu = ctk.CTkOptionMenu(
            right_col,
            values=list(IDLE_UNLOAD_CHOICES.keys()),
            height=26,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            button_color=self.colors["border"],
            command=self._on_idle_unload_selected,
        )
        cur_idle = int(self.config.get("idle_unload_minutes", 5))
        self.idle_unload_menu.set(IDLE_UNLOAD_REVERSE.get(cur_idle, "5 min"))
        self.idle_unload_menu.pack(fill="x")

        act_row = tk.Frame(box, bg=self.colors["bg"])
        act_row.pack(fill="x", pady=(4, 0))

        self.btn_reload_model = ctk.CTkButton(
            act_row,
            text="Load / Reload Model",
            height=28,
            corner_radius=6,
            font=ui_font(10, "bold"),
            fg_color=self.colors["teal_dim"],
            hover_color=self.colors["teal"],
            command=self.preload_model_async,
        )
        self.btn_reload_model.pack(side="left", expand=True, fill="x", padx=(0, 4))

        ctk.CTkButton(
            act_row,
            text="CUDA 12 GPU Add-on...",
            height=28,
            corner_radius=6,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            hover_color=self.colors["border"],
            command=self.show_linux_cuda_addon_dialog,
        ).pack(side="left", expand=True, fill="x", padx=(4, 0))

    def _build_online_controls(self, box: tk.Frame) -> None:
        hdr = tk.Frame(box, bg=self.colors["bg"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="OpenAI API Key:", bg=self.colors["bg"], fg=self.colors["text_muted"], font=ui_font(10)).pack(
            side="left"
        )
        ctk.CTkButton(
            hdr,
            text="Get Key ↗",
            width=74,
            height=22,
            corner_radius=4,
            font=ui_font(9),
            fg_color="transparent",
            hover_color=self.colors["accent_bg"],
            text_color=self.colors["teal"],
            command=lambda: open_path_or_url(OPENAI_KEY_URL),
        ).pack(side="right")

        key_row = tk.Frame(box, bg=self.colors["bg"])
        key_row.pack(fill="x", pady=(2, 6))

        self._api_key_visible = False
        self.api_key_entry = ctk.CTkEntry(
            key_row,
            show="•",
            height=28,
            font=mono_font(10),
            fg_color=self.colors["accent_bg"],
            border_color=self.colors["border"],
        )
        if self.config.get("openai_api_key"):
            self.api_key_entry.insert(0, str(self.config["openai_api_key"]))
        self.api_key_entry.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.btn_show_key = ctk.CTkButton(
            key_row,
            text="Show",
            width=48,
            height=28,
            corner_radius=5,
            font=ui_font(9),
            fg_color=self.colors["accent_bg"],
            hover_color=self.colors["border"],
            command=self._toggle_api_key_visibility,
        )
        self.btn_show_key.pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            key_row,
            text="Save",
            width=48,
            height=28,
            corner_radius=5,
            font=ui_font(9, "bold"),
            fg_color=self.colors["teal_dim"],
            hover_color=self.colors["teal"],
            command=self._save_api_key,
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            key_row,
            text="Test",
            width=48,
            height=28,
            corner_radius=5,
            font=ui_font(9),
            fg_color=self.colors["border"],
            hover_color=self.colors["online"],
            command=self._test_api_key_async,
        ).pack(side="left")

        tk.Label(
            box, text="Cloud Transcription Model:", bg=self.colors["bg"], fg=self.colors["text_muted"], font=ui_font(9)
        ).pack(anchor="w")
        self.openai_transcribe_menu = ctk.CTkOptionMenu(
            box,
            values=OPENAI_TRANSCRIPTION_MODEL_OPTIONS,
            height=26,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            button_color=self.colors["border"],
            command=self._on_openai_transcribe_model_change,
        )
        self.openai_transcribe_menu.set(str(self.config.get("openai_transcription_model", "gpt-transcribe")))
        self.openai_transcribe_menu.pack(fill="x", pady=(2, 6))

        self.cloud_edit_var = tk.BooleanVar(
            value=str(self.config.get("transcription_mode", "local")) == "online_edit"
        )
        ctk.CTkCheckBox(
            box,
            text="Polish grammar & remove fillers with GPT",
            variable=self.cloud_edit_var,
            font=ui_font(10),
            checkbox_width=18,
            checkbox_height=18,
            fg_color=self.colors["online"],
            command=self._on_toggle_cloud_edit,
        ).pack(anchor="w", pady=(0, 4))

        self.openai_edit_menu = ctk.CTkOptionMenu(
            box,
            values=OPENAI_EDIT_MODEL_OPTIONS,
            height=26,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            button_color=self.colors["border"],
            command=self._on_openai_edit_model_change,
        )
        self.openai_edit_menu.set(str(self.config.get("openai_edit_model", "gpt-4o-mini")))
        self.openai_edit_menu.pack(fill="x")

    def _build_audio_controls(self, parent: tk.Frame) -> None:
        box = tk.Frame(
            parent,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=8,
        )
        box.pack(fill="x", pady=(0, 8))

        tk.Label(box, text="Microphone Input:", bg=self.colors["bg"], fg=self.colors["text_muted"], font=ui_font(10)).pack(
            anchor="w"
        )
        self._mic_choices = self._discover_microphones()
        mic_labels = [label for label, _ in self._mic_choices]
        self.mic_menu = ctk.CTkOptionMenu(
            box,
            values=mic_labels,
            height=26,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            button_color=self.colors["border"],
            command=self._on_mic_selected,
        )
        saved_dev = self.config.get("input_device")
        matched_label = next(
            (lbl for lbl, idx in self._mic_choices if idx == saved_dev),
            mic_labels[0],
        )
        self.mic_menu.set(matched_label)
        self.mic_menu.pack(fill="x", pady=(2, 6))

        pause_hdr = tk.Frame(box, bg=self.colors["bg"])
        pause_hdr.pack(fill="x")
        tk.Label(
            pause_hdr,
            text="Live Dictation Pause Threshold:",
            bg=self.colors["bg"],
            fg=self.colors["text_muted"],
            font=ui_font(9),
        ).pack(side="left")
        self.live_pause_val_lbl = tk.Label(
            pause_hdr,
            text=f"{float(self.config.get('live_pause', 0.8)):.1f}s",
            bg=self.colors["bg"],
            fg=self.colors["teal"],
            font=mono_font(9, "bold"),
        )
        self.live_pause_val_lbl.pack(side="right")

        self.live_pause_slider = ctk.CTkSlider(
            box,
            from_=float(LIVE_PAUSE_RANGE["min"]),
            to=float(LIVE_PAUSE_RANGE["max"]),
            number_of_steps=25,
            button_color=self.colors["teal"],
            progress_color=self.colors["teal_dim"],
            command=self._on_live_pause_change,
        )
        self.live_pause_slider.set(float(self.config.get("live_pause", 0.8)))
        self.live_pause_slider.pack(fill="x", pady=(2, 6))

        self.always_on_top_var = tk.BooleanVar(value=bool(self.config.get("always_on_top", True)))
        ctk.CTkCheckBox(
            box,
            text="Keep window always on top",
            variable=self.always_on_top_var,
            font=ui_font(10),
            checkbox_width=18,
            checkbox_height=18,
            fg_color=self.colors["teal"],
            command=self._on_toggle_always_on_top,
        ).pack(anchor="w")

    def _build_hotkeys_and_wayland_controls(self, parent: tk.Frame) -> None:
        box = tk.Frame(
            parent,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=8,
        )
        box.pack(fill="x", pady=(0, 8))

        tk.Label(
            box,
            text="Global Shortcuts & Linux Desktop Integration",
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            font=ui_font(10, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        # Live & Batch Hotkey Recorder Rows (100% CTkButton)
        for mode_key, label_txt in (("hotkey_live", "Live Dictation:"), ("hotkey_batch", "Batch Record:")):
            row = tk.Frame(box, bg=self.colors["bg"])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label_txt, bg=self.colors["bg"], fg=self.colors["text_muted"], font=ui_font(10)).pack(
                side="left"
            )
            btn = ctk.CTkButton(
                row,
                text=str(self.config.get(mode_key, "")).upper(),
                width=145,
                height=26,
                corner_radius=5,
                font=mono_font(10, "bold"),
                fg_color=self.colors["accent_bg"],
                hover_color=self.colors["teal_dim"],
                command=lambda mk=mode_key: self._start_hotkey_capture_ui(mk),
            )
            btn.pack(side="right")
            if mode_key == "hotkey_live":
                self.live_hotkey_btn = btn
            else:
                self.batch_hotkey_btn = btn

        self.hotkey_health_label = tk.Label(
            box,
            text="",
            bg=self.colors["bg"],
            fg=self.colors["text_muted"],
            font=ui_font(9),
            wraplength=410,
            justify="left",
        )
        self.hotkey_health_label.pack(anchor="w", pady=(4, 4))

        # Wayland / Universal CLI IPC Shortcut Copy + .desktop Installer Row
        wl_row = tk.Frame(box, bg=self.colors["bg"])
        wl_row.pack(fill="x", pady=(2, 2))

        ctk.CTkButton(
            wl_row,
            text="Copy --toggle-live",
            height=25,
            corner_radius=5,
            font=ui_font(9),
            fg_color=self.colors["accent_bg"],
            hover_color=self.colors["border"],
            command=lambda: self._copy_cli_shortcut_cmd("--toggle-live"),
        ).pack(side="left", expand=True, fill="x", padx=(0, 3))

        ctk.CTkButton(
            wl_row,
            text="Copy --toggle-batch",
            height=25,
            corner_radius=5,
            font=ui_font(9),
            fg_color=self.colors["accent_bg"],
            hover_color=self.colors["border"],
            command=lambda: self._copy_cli_shortcut_cmd("--toggle-batch"),
        ).pack(side="left", expand=True, fill="x", padx=(3, 3))

        self.refresh_hotkeys_btn = ctk.CTkButton(
            wl_row,
            text="Rebind",
            width=62,
            height=25,
            corner_radius=5,
            font=ui_font(9),
            fg_color=self.colors["border"],
            hover_color=self.colors["teal_dim"],
            command=self.register_hotkeys,
        )
        self.refresh_hotkeys_btn.pack(side="left", padx=(3, 0))

        ctk.CTkButton(
            box,
            text="Install Desktop Launcher (~/.local/share/applications/neurowhisper.desktop)",
            height=27,
            corner_radius=5,
            font=ui_font(9, "bold"),
            fg_color=self.colors["teal_dim"],
            hover_color=self.colors["teal"],
            command=self._install_desktop_launcher_ui,
        ).pack(fill="x", pady=(5, 0))

        self._refresh_hotkey_health_banner()

    def _build_sync_folder_controls(self, parent: tk.Frame) -> None:
        box = tk.Frame(
            parent,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=8,
        )
        box.pack(fill="x")

        tk.Label(
            box,
            text="Custom Data / Cloud Sync Folder (Dropbox / Syncthing / Nextcloud)",
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            font=ui_font(10, "bold"),
        ).pack(anchor="w")

        self.sync_dir_label = tk.Label(
            box,
            text=self._format_sync_dir_status(),
            bg=self.colors["bg"],
            fg=self.colors["text_muted"],
            font=mono_font(9),
            wraplength=410,
            justify="left",
        )
        self.sync_dir_label.pack(anchor="w", pady=(3, 6))

        btn_row = tk.Frame(box, bg=self.colors["bg"])
        btn_row.pack(fill="x")

        ctk.CTkButton(
            btn_row,
            text="Choose Sync Folder...",
            height=26,
            corner_radius=5,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            hover_color=self.colors["teal_dim"],
            command=self._choose_sync_folder_dialog,
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        ctk.CTkButton(
            btn_row,
            text="Reset to XDG Default",
            height=26,
            corner_radius=5,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            hover_color=self.colors["border"],
            command=self._reset_sync_folder,
        ).pack(side="left", expand=True, fill="x", padx=(4, 0))

    # ------------------------------------------------------------------
    # Configuration Callbacks & Linux Helper Dialogs
    # ------------------------------------------------------------------

    def _format_sync_dir_status(self) -> str:
        sd = paths.get_sync_dir()
        if sd is not None:
            return f"Synced folder: {sd}"
        return f"XDG default: {paths.get_data_dir()}"

    def _choose_sync_folder_dialog(self) -> None:
        initial = str(paths.get_sync_dir() or Path.home())
        chosen = filedialog.askdirectory(
            parent=self.root,
            title="Select neurowhisper Sync Folder",
            initialdir=initial,
        )
        if not chosen:
            return
        paths.set_sync_dir(chosen)
        paths.ensure_user_dirs()
        self._reload_after_storage_change()

    def _reset_sync_folder(self) -> None:
        paths.set_sync_dir("")
        paths.ensure_user_dirs()
        self._reload_after_storage_change()

    def _reload_after_storage_change(self) -> None:
        if hasattr(self, "sync_dir_label"):
            self.sync_dir_label.configure(text=self._format_sync_dir_status())
        self.config = self.load_config()
        self.own_stats, self.stats = self.load_stats()
        self.transcription_history = self._load_recent_history()
        self.history_index = max(0, len(self.transcription_history) - 1)
        self._ai_insights_data = self._load_ai_insights_disk()
        self._refresh_history_display()
        self._update_stats_cards()
        self.draw_histogram()
        if hasattr(self, "insights_body"):
            self.refresh_speech_insights()
        self.set_status("Updated data & sync storage directory")

    def _switch_transcription_mode(self, target: str) -> None:
        if target == "local":
            self.config["transcription_mode"] = "local"
            self.btn_mode_local.configure(fg_color=self.colors["teal_dim"], font=ui_font(10, "bold"))
            self.btn_mode_online.configure(fg_color=self.colors["bg"], font=ui_font(10))
            self.online_cfg_frame.pack_forget()
            self.local_cfg_frame.pack(fill="x", pady=(0, 8))
            self.save_config()
            self.preload_model_async()
        else:
            edit = bool(getattr(self, "cloud_edit_var", None) and self.cloud_edit_var.get())
            self.config["transcription_mode"] = "online_edit" if edit else "online"
            self.btn_mode_online.configure(fg_color=self.colors["online"], font=ui_font(10, "bold"))
            self.btn_mode_local.configure(fg_color=self.colors["bg"], font=ui_font(10))
            self.local_cfg_frame.pack_forget()
            self.online_cfg_frame.pack(fill="x", pady=(0, 8))
            self.save_config()
            self._set_app_state("ready", "OpenAI Cloud mode active")

    def _on_model_selected(self, display_name: str) -> None:
        self.config["model_key"] = display_name
        self.save_config()
        if str(self.config.get("transcription_mode", "local")) == "local":
            self.preload_model_async()

    def _on_backend_selected(self, display_label: str) -> None:
        backend, device = BACKEND_DISPLAY_TO_BACKEND_DEVICE.get(display_label, ("auto", "auto"))
        self.config["backend"] = backend
        self.config["device"] = device
        self.save_config()
        if str(self.config.get("transcription_mode", "local")) == "local":
            self.preload_model_async()

    def _on_local_lang_selected(self, display_label: str) -> None:
        self.config["local_language"] = LOCAL_LANG_DISPLAY_TO_CODE.get(display_label, "auto")
        self.save_config()

    def _on_idle_unload_selected(self, label: str) -> None:
        self.config["idle_unload_minutes"] = IDLE_UNLOAD_CHOICES.get(label, 5)
        self.save_config()

    def _toggle_api_key_visibility(self) -> None:
        self._api_key_visible = not self._api_key_visible
        self.api_key_entry.configure(show="" if self._api_key_visible else "•")
        self.btn_show_key.configure(text="Hide" if self._api_key_visible else "Show")

    def _save_api_key(self) -> None:
        self.config["openai_api_key"] = self.api_key_entry.get().strip()
        self.save_config()
        self.set_status("Saved OpenAI API key locally")

    def _test_api_key_async(self) -> None:
        key = self.api_key_entry.get().strip()
        self.config["openai_api_key"] = key
        self.save_config()
        self.set_status("Testing OpenAI API key...")

        def _worker() -> None:
            ok, msg = OpenAIBackend.validate_api_key(key)
            self._enqueue_ui(lambda: self.set_status(msg))

        threading.Thread(target=_worker, name="neurowhisper-key-test", daemon=True).start()

    def _on_openai_transcribe_model_change(self, val: str) -> None:
        self.config["openai_transcription_model"] = val
        self.save_config()

    def _on_toggle_cloud_edit(self) -> None:
        if str(self.config.get("transcription_mode", "local")).startswith("online"):
            self.config["transcription_mode"] = (
                "online_edit" if self.cloud_edit_var.get() else "online"
            )
            self.save_config()

    def _on_openai_edit_model_change(self, val: str) -> None:
        self.config["openai_edit_model"] = val
        self.save_config()

    @staticmethod
    def _discover_microphones() -> List[Tuple[str, Optional[int]]]:
        choices: List[Tuple[str, Optional[int]]] = [("System Default Microphone", None)]
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if int(dev.get("max_input_channels", 0)) > 0:
                    name = str(dev.get("name", f"Input {idx}")).strip()
                    choices.append((f"[{idx}] {name}", idx))
        except Exception as exc:
            log.debug("Microphone enumeration unavailable: %s", exc)
        return choices

    def _on_mic_selected(self, label: str) -> None:
        dev_idx = next((idx for lbl, idx in getattr(self, "_mic_choices", []) if lbl == label), None)
        self.config["input_device"] = dev_idx
        self.save_config()

    def _on_live_pause_change(self, val: float) -> None:
        rounded = round(float(val), 1)
        self.config["live_pause"] = rounded
        if hasattr(self, "live_pause_val_lbl"):
            self.live_pause_val_lbl.configure(text=f"{rounded:.1f}s")
        self.save_config()

    def _on_toggle_always_on_top(self) -> None:
        flag = bool(self.always_on_top_var.get())
        self.config["always_on_top"] = flag
        try:
            self.root.attributes("-topmost", flag)
        except tk.TclError:
            pass
        self.save_config()

    def _copy_cli_shortcut_cmd(self, flag: str) -> None:
        appimage = os.environ.get("APPIMAGE", "").strip()
        if appimage:
            cmd = f'"{appimage}" {flag}'
        elif getattr(sys, "frozen", False):
            cmd = f'"{Path(sys.executable).resolve()}" {flag}'
        else:
            cmd = f"neurowhisper {flag}"
        copy_to_clipboard(cmd, root=self.root)
        self.set_status(f"Copied shortcut command: {cmd}")

    def _install_desktop_launcher_ui(self) -> None:
        try:
            desktop_path, _ = install_desktop_entry()
            self.set_status(f"Installed launcher: {desktop_path.name}")
        except Exception as exc:
            self.set_status(f"Desktop launcher install failed: {exc}")

    def show_linux_cuda_addon_dialog(self) -> None:
        """Open a dark-themed dialog installing CUDA 12 cuBLAS/cuDNN into ``~/.local/share/neurowhisper/cuda-libs``."""
        cuda_dir = paths.get_cuda_libs_dir()
        pkgs = " ".join(CUDA_PIP_PACKAGES)
        pip_cmd = f'python3 -m pip install --upgrade --target "{cuda_dir}" {pkgs}'

        dlg = tk.Toplevel(self.root)
        dlg.title("NVIDIA CUDA 12 Runtime Add-on")
        dlg.geometry(UI_RUNTIME["dialog_geometry_cuda"])
        dlg.configure(bg=self.colors["bg"])
        dlg.transient(self.root)

        body = tk.Frame(dlg, bg=self.colors["bg"], padx=16, pady=14)
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            text="Enable NVIDIA CUDA 12 GPU Acceleration",
            bg=self.colors["bg"],
            fg=self.colors["teal"],
            font=ui_font(13, "bold"),
        ).pack(anchor="w")

        tk.Label(
            body,
            text=(
                "To keep the Linux AppImage compact (~220 MB instead of 2.5 GB), NVIDIA cuBLAS & cuDNN "
                "libraries are loaded dynamically from your system CUDA toolkit or your user profile:\n"
                f"{cuda_dir}"
            ),
            bg=self.colors["bg"],
            fg=self.colors["text_muted"],
            font=ui_font(10),
            wraplength=500,
            justify="left",
        ).pack(anchor="w", pady=(6, 8))

        cmd_box = tk.Text(
            body,
            height=3,
            bg=self.colors["accent_bg"],
            fg=self.colors["fg"],
            font=mono_font(9),
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            wrap="word",
            padx=8,
            pady=6,
        )
        cmd_box.insert("1.0", pip_cmd)
        cmd_box.configure(state="disabled")
        cmd_box.pack(fill="x", pady=(0, 8))

        prog_lbl = tk.Label(
            body,
            text=f"Discovered CUDA dirs: {len(paths.discover_cuda_lib_dirs())}",
            bg=self.colors["bg"],
            fg=self.colors["text_muted"],
            font=ui_font(9),
        )
        prog_lbl.pack(anchor="w", pady=(0, 10))

        btn_row = tk.Frame(body, bg=self.colors["bg"])
        btn_row.pack(fill="x")

        install_btn = ctk.CTkButton(
            btn_row,
            text="Install CUDA 12 Add-on Now",
            height=32,
            corner_radius=6,
            font=ui_font(10, "bold"),
            fg_color=self.colors["teal_dim"],
            hover_color=self.colors["teal"],
        )
        install_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        ctk.CTkButton(
            btn_row,
            text="Copy Command",
            height=32,
            corner_radius=6,
            font=ui_font(10),
            fg_color=self.colors["accent_bg"],
            hover_color=self.colors["border"],
            command=lambda: copy_to_clipboard(pip_cmd, root=self.root),
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            btn_row,
            text="Close",
            width=70,
            height=32,
            corner_radius=6,
            font=ui_font(10),
            fg_color=self.colors["border"],
            command=dlg.destroy,
        ).pack(side="left")

        def _run_install() -> None:
            install_btn.configure(state="disabled", text="Installing (~750 MB)...")
            prog_lbl.configure(text="Downloading nvidia-cublas-cu12 & nvidia-cudnn-cu12...", fg=self.colors["transcribing"])

            def _worker() -> None:
                cuda_dir.mkdir(parents=True, exist_ok=True)
                uv_bin = shutil.which("uv")
                py_bin = shutil.which("python3") or sys.executable
                if uv_bin:
                    cmd = [uv_bin, "pip", "install", "--target", str(cuda_dir), *CUDA_PIP_PACKAGES]
                else:
                    cmd = [py_bin, "-m", "pip", "install", "--upgrade", "--target", str(cuda_dir), *CUDA_PIP_PACKAGES]

                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    ok = proc.returncode == 0
                    msg = (
                        "CUDA 12 runtime installed! Reloading GPU backend..."
                        if ok
                        else f"Install failed: {(proc.stderr or proc.stdout).strip()[-140:]}"
                    )
                except Exception as exc:
                    ok = False
                    msg = f"Install error: {exc}"

                def _done() -> None:
                    if dlg.winfo_exists():
                        install_btn.configure(state="normal", text="Install CUDA 12 Add-on Now")
                        prog_lbl.configure(
                            text=msg,
                            fg=self.colors["ready"] if ok else self.colors["live"],
                        )
                    if ok:
                        self.preload_model_async()

                self._enqueue_ui(_done)

            threading.Thread(target=_worker, name="neurowhisper-cuda-install", daemon=True).start()

        install_btn.configure(command=_run_install)

    # ------------------------------------------------------------------
    # Global Hotkeys & Wayland/X11 Interactive Recorder
    # ------------------------------------------------------------------

    def register_hotkeys(self) -> None:
        live_hk = str(self.config.get("hotkey_live", "f9")).strip().lower()
        batch_hk = str(self.config.get("hotkey_batch", "ctrl+alt+s")).strip().lower()

        bindings = {
            live_hk: lambda: self._enqueue_hotkey_action("toggle_live"),
            batch_hk: lambda: self._enqueue_hotkey_action("toggle_batch"),
        }
        action_names = {
            live_hk: "toggle_live",
            batch_hk: "toggle_batch",
        }
        self.hotkey_manager.register_all(bindings, action_names=action_names)
        self._update_recording_button_labels()
        self.root.after(400, self._refresh_hotkey_health_banner)

    def _refresh_hotkey_health_banner(self) -> None:
        if not hasattr(self, "hotkey_health_label") or not hasattr(self, "hotkey_manager"):
            return
        st = self.hotkey_manager.health_status()
        portal_msg = str(st.get("portal_status") or "")
        if st.get("wayland_manual_shortcuts_needed"):
            text = (
                f"Wayland session: {portal_msg or 'CLI IPC active'}. "
                "Tip: bind a system keyboard shortcut to 'neurowhisper --toggle-live' for universal Wayland support."
            )
            fg = self.colors["transcribing"]
        else:
            text = portal_msg or f"Active shortcut backend: {st.get('backend')}"
            fg = self.colors["ready"] if st.get("healthy") else self.colors["text_muted"]
        self.hotkey_health_label.configure(text=text, fg=fg)

    def _start_hotkey_capture_ui(self, config_key: str) -> None:
        target_btn = self.live_hotkey_btn if config_key == "hotkey_live" else self.batch_hotkey_btn
        target_btn.configure(text="Press keys...", fg_color=self.colors["teal_dim"])
        self.hotkey_manager.paused = True
        self._tk_capture_tokens.clear()

        def _finish(combo: str) -> None:
            self.root.unbind("<KeyPress>")
            if self._active_capture_cancel is not None:
                self._active_capture_cancel()
                self._active_capture_cancel = None
            self.hotkey_manager.paused = False
            if combo:
                self.config[config_key] = combo.lower()
                self.save_config()
                self.register_hotkeys()
            self._update_recording_button_labels()
            if hasattr(self, "live_hotkey_btn"):
                self.live_hotkey_btn.configure(
                    text=str(self.config.get("hotkey_live", "f9")).upper(),
                    fg_color=self.colors["accent_bg"],
                )
            if hasattr(self, "batch_hotkey_btn"):
                self.batch_hotkey_btn.configure(
                    text=str(self.config.get("hotkey_batch", "ctrl+alt+s")).upper(),
                    fg_color=self.colors["accent_bg"],
                )

        # Also bind Tk <KeyPress> so key recording works on pure Wayland when window is focused!
        def _on_tk_key(event) -> None:
            sym = normalize_key_token(getattr(event, "keysym", ""))
            if not sym:
                return
            if sym == "esc":
                self._enqueue_ui(lambda: _finish(""))
                return
            self._tk_capture_tokens.add(sym)
            combo_str = format_hotkey_combo(self._tk_capture_tokens)
            target_btn.configure(text=combo_str.upper())
            if sym not in ("ctrl", "alt", "shift", "win"):
                self._enqueue_ui(lambda c=combo_str: _finish(c))

        self.root.bind("<KeyPress>", _on_tk_key)
        self._active_capture_cancel = self.hotkey_manager.start_key_capture(
            on_update=lambda s: self._enqueue_ui(lambda: target_btn.configure(text=s.upper())),
            on_finish=lambda s: self._enqueue_ui(lambda: _finish(s)),
        )

    def _update_recording_button_labels(self) -> None:
        live_hk = str(self.config.get("hotkey_live", "f9")).upper()
        batch_hk = str(self.config.get("hotkey_batch", "ctrl+alt+s")).upper()
        if not self.is_recording:
            self.btn_live.configure(
                text=f"Live Dictation ({live_hk})",
                fg_color=self.colors["teal_dim"],
            )
            self.btn_batch.configure(
                text=f"Batch Record ({batch_hk})",
                fg_color=self.colors["border"],
            )

    # ------------------------------------------------------------------
    # Backend Model Loading & Idle Unload
    # ------------------------------------------------------------------

    def preload_model_async(self) -> None:
        if self._model_loading:
            return
        self._model_loading = True
        model_display = str(self.config.get("model_key", DEFAULT_CONFIG["model_key"]))
        model_id = MODEL_MAP.get(model_display, "small")
        dl_size = MODEL_DOWNLOAD_SIZES.get(model_id, "~500 MB")
        self._set_app_state("transcribing", f"Loading {model_id} ({dl_size} first run)...")

        def _worker() -> None:
            try:
                with self._model_lock:
                    backend_cfg = str(self.config.get("backend", "auto"))
                    device_cfg = str(self.config.get("device", "auto"))
                    compute_cfg = str(self.config.get("compute_type", "auto"))

                    backend = create_backend(
                        backend_type=backend_cfg,
                        device=device_cfg,
                        model_key=model_id,
                    )
                    eff_dev = device_cfg
                    if backend_cfg == "auto" and device_cfg == "auto":
                        _, eff_dev, _ = detect_best_backend(model_key=model_id)

                    backend.load_model(model_id, device=eff_dev, compute_type=compute_cfg)
                    self._backend = backend
                    self._last_activity_ts = time.monotonic()

                    badge = f"{backend.backend_name} • {model_id}"
                    cuda_fell_back = (
                        isinstance(backend, FasterWhisperBackend)
                        and backend.cuda_failed
                        and device_cfg == "cuda"
                    )

                def _on_ok() -> None:
                    self._model_loading = False
                    self.engine_badge_label.configure(text=badge)
                    if cuda_fell_back:
                        self._set_app_state(
                            "ready",
                            "CUDA 12 libs missing — running on CPU INT8 (see CUDA 12 Add-on)",
                        )
                    else:
                        self._set_app_state("ready", f"Ready ({model_id})")

                self._enqueue_ui(_on_ok)
            except Exception as exc:
                log.exception("Model load failed")
                err_msg = str(exc)

                def _on_err() -> None:
                    self._model_loading = False
                    self._set_app_state("idle", f"Model load error: {err_msg[:90]}")
                    if any(kw in err_msg.lower() for kw in CUDA_ERROR_KEYWORDS):
                        self.show_linux_cuda_addon_dialog()

                self._enqueue_ui(_on_err)

        threading.Thread(target=_worker, name="neurowhisper-model-load", daemon=True).start()

    def _ensure_active_backend(self):
        """Ensure the configured local or OpenAI backend is loaded and ready for inference."""
        mode = str(self.config.get("transcription_mode", "local"))
        if mode.startswith("online"):
            api_key = str(self.config.get("openai_api_key", "") or "").strip()
            if not api_key:
                raise RuntimeError("OpenAI API key is empty — configure it under OpenAI Cloud settings.")
            backend = create_backend(
                backend_type="openai",
                api_key=api_key,
                transcription_model=str(self.config.get("openai_transcription_model", "gpt-transcribe")),
                edit_enabled=(mode == "online_edit"),
                edit_model=str(self.config.get("openai_edit_model", "gpt-4o-mini")),
                edit_prompt=str(self.config.get("openai_edit_prompt", "") or OPENAI_DEFAULT_EDIT_PROMPT),
                language=str(self.config.get("openai_language", "auto")),
            )
            if not backend.is_loaded:
                backend.load_model()
            self._backend = backend
            return backend

        with self._model_lock:
            model_display = str(self.config.get("model_key", DEFAULT_CONFIG["model_key"]))
            model_id = MODEL_MAP.get(model_display, "small")
            backend_cfg = str(self.config.get("backend", "auto"))
            device_cfg = str(self.config.get("device", "auto"))
            compute_cfg = str(self.config.get("compute_type", "auto"))

            if self._backend is None or not self._backend.is_loaded or self._backend.current_model != model_id:
                backend = create_backend(
                    backend_type=backend_cfg,
                    device=device_cfg,
                    model_key=model_id,
                )
                eff_dev = device_cfg
                if backend_cfg == "auto" and device_cfg == "auto":
                    _, eff_dev, _ = detect_best_backend(model_key=model_id)
                backend.load_model(model_id, device=eff_dev, compute_type=compute_cfg)
                self._backend = backend
                self._enqueue_ui(
                    lambda b=f"{backend.backend_name} • {model_id}": self.engine_badge_label.configure(text=b)
                )
            self._last_activity_ts = time.monotonic()
            return self._backend

    def _check_idle_model_unload(self) -> None:
        now = time.monotonic()
        if now - self._last_idle_check_ts < IDLE_UNLOAD_CHECK_INTERVAL_SECONDS:
            return
        self._last_idle_check_ts = now

        idle_mins = int(self.config.get("idle_unload_minutes", 5))
        if idle_mins <= 0 or self.is_recording or self._model_loading:
            return
        if str(self.config.get("transcription_mode", "local")) != "local":
            return
        if self._backend is not None and self._backend.is_loaded:
            if (now - self._last_activity_ts) >= (idle_mins * 60.0):
                log.info("Unloading idle speech model after %d minutes", idle_mins)
                threading.Thread(
                    target=self._unload_idle_worker,
                    name="neurowhisper-idle-unload",
                    daemon=True,
                ).start()

    def _unload_idle_worker(self) -> None:
        with self._model_lock:
            if self.is_recording or self._backend is None:
                return
            unload_all_backends()
            self._backend = None
        self._enqueue_ui(lambda: self._set_app_state("idle", "Idle (model unloaded to free RAM/VRAM)"))

    # ------------------------------------------------------------------
    # Live Dictation & Batch Recording Engine
    # ------------------------------------------------------------------

    def toggle_live_recording(self) -> None:
        if self.is_recording:
            if self.recording_mode == "live":
                self.stop_recording()
            return
        self._start_recording("live")

    def toggle_batch_recording(self) -> None:
        if self.is_recording:
            if self.recording_mode == "batch":
                self.stop_recording()
            return
        self._start_recording("batch")

    def stop_recording(self) -> None:
        if not self.is_recording:
            return
        self._record_stop_event.set()
        play_feedback_sound_async(start=False)
        if self.recording_mode == "live":
            self.btn_live.configure(text="Stopping...", fg_color=self.colors["stopping_live"])
        elif self.recording_mode == "batch":
            self.btn_batch.configure(text="Transcribing...", fg_color=self.colors["stopping_batch"])

    def _start_recording(self, mode: str) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            self.set_status(f"PortAudio / sounddevice unavailable: {exc}")
            return

        self._record_stop_event.clear()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        sr = int(AUDIO_RUNTIME["sample_rate"])
        channels = int(AUDIO_RUNTIME["channels"])
        blocksize = int(AUDIO_RUNTIME["blocksize"])
        input_dev = self.config.get("input_device")

        def _audio_cb(indata, _frames, _time_info, status) -> None:
            if status:
                log.debug("Audio InputStream status: %s", status)
            mono = np.array(indata[:, 0], dtype=np.float32, copy=True)
            rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
            self._current_vu_level = min(1.0, rms * 14.0)
            self.audio_queue.put(mono)

        try:
            stream = sd.InputStream(
                samplerate=sr,
                channels=channels,
                blocksize=blocksize,
                dtype="float32",
                device=input_dev,
                callback=_audio_cb,
            )
            stream.start()
            self._audio_stream = stream
        except Exception as exc:
            log.warning("Could not open microphone stream: %s", exc)
            self.set_status(f"Microphone error: {exc}")
            return

        self.is_recording = True
        self.recording_mode = mode
        self._last_activity_ts = time.monotonic()
        play_feedback_sound_async(start=True)

        is_online = str(self.config.get("transcription_mode", "local")).startswith("online")
        if mode == "live":
            self.btn_live.configure(text="Stop Live (REC)", fg_color=self.colors["live"])
            self.btn_batch.configure(state="disabled")
            self._set_app_state("online" if is_online else "live", "Listening — speak naturally...")
            threading.Thread(target=self._live_recording_loop, name="neurowhisper-live", daemon=True).start()
        else:
            self.btn_batch.configure(text="Finish & Paste (REC)", fg_color=self.colors["batch"])
            self.btn_live.configure(state="disabled")
            self._set_app_state("online" if is_online else "batch", "Recording batch audio...")
            threading.Thread(target=self._batch_recording_loop, name="neurowhisper-batch", daemon=True).start()

    def _close_audio_stream(self) -> None:
        stream = self._audio_stream
        self._audio_stream = None
        self._current_vu_level = 0.0
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _effective_language(self) -> Optional[str]:
        mode = str(self.config.get("transcription_mode", "local"))
        raw = str(
            self.config.get("openai_language" if mode.startswith("online") else "local_language", "auto")
        )
        return None if raw == "auto" else raw

    def _live_recording_loop(self) -> None:
        sr = int(AUDIO_RUNTIME["sample_rate"])
        chunk_sec = float(AUDIO_RUNTIME["chunk_duration_seconds"])
        silence_thresh = float(self.config.get("silence_threshold", 0.015))
        min_sec = float(AUDIO_RUNTIME["live_min_chunk_seconds"])
        max_sec = float(AUDIO_RUNTIME["live_max_chunk_seconds"])

        speech_chunks: List[np.ndarray] = []
        silent_sec = 0.0
        has_voice = False

        try:
            backend = self._ensure_active_backend()
        except Exception as exc:
            self._close_audio_stream()
            self.is_recording = False
            self.recording_mode = None
            self._enqueue_ui(lambda msg=str(exc): self._finish_recording_ui(f"Backend error: {msg[:90]}"))
            return

        while not self._record_stop_event.is_set():
            try:
                block = self.audio_queue.get(timeout=float(AUDIO_RUNTIME["queue_timeout_seconds"]))
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
            speech_chunks.append(block)
            buf_dur = sum(len(c) for c in speech_chunks) / float(sr)

            if rms >= silence_thresh:
                has_voice = True
                silent_sec = 0.0
            else:
                silent_sec += chunk_sec

            pause_limit = float(self.config.get("live_pause", 0.8))
            should_flush = has_voice and (
                (silent_sec >= pause_limit and buf_dur >= min_sec) or (buf_dur >= max_sec)
            )
            if should_flush:
                segment = np.concatenate(speech_chunks)
                speech_chunks = []
                silent_sec = 0.0
                has_voice = False
                self._transcribe_and_type_live_chunk(backend, segment, sr)

        # Drain trailing audio on stop
        self._close_audio_stream()
        while not self.audio_queue.empty():
            try:
                speech_chunks.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        if has_voice and speech_chunks:
            segment = np.concatenate(speech_chunks)
            if (len(segment) / float(sr)) >= 0.35:
                self._transcribe_and_type_live_chunk(backend, segment, sr)

        self.is_recording = False
        self.recording_mode = None
        self._enqueue_ui(lambda: self._finish_recording_ui("Ready"))

    def _transcribe_and_type_live_chunk(self, backend, segment: np.ndarray, sr: int) -> None:
        t0 = time.monotonic()
        try:
            res = backend.transcribe(
                segment,
                sample_rate=sr,
                language=self._effective_language(),
            )
            latency = time.monotonic() - t0
            txt = (res.text or "").strip()
            if txt:
                type_text(txt, root=self.root)
                self._record_utterance_stats_and_archive(txt, len(segment) / float(sr), latency)
        except Exception as exc:
            log.warning("Live chunk transcription failed: %s", exc)
            self._enqueue_ui(lambda m=str(exc): self.set_status(f"Transcription error: {m[:80]}"))

    def _batch_recording_loop(self) -> None:
        sr = int(AUDIO_RUNTIME["sample_rate"])
        chunk_sec = float(AUDIO_RUNTIME["chunk_duration_seconds"])
        silence_thresh = float(self.config.get("silence_threshold", 0.015))
        min_seg = float(AUDIO_RUNTIME["batch_min_segment_seconds"])
        long_seg = float(AUDIO_RUNTIME["batch_long_segment_seconds"])
        max_seg = float(AUDIO_RUNTIME["batch_max_segment_seconds"])
        pause_short = float(AUDIO_RUNTIME["pause_threshold_short_seconds"])
        pause_long = float(AUDIO_RUNTIME["pause_threshold_long_seconds"])

        try:
            backend = self._ensure_active_backend()
        except Exception as exc:
            self._close_audio_stream()
            self.is_recording = False
            self.recording_mode = None
            self._enqueue_ui(lambda msg=str(exc): self._finish_recording_ui(f"Backend error: {msg[:90]}"))
            return

        is_gpu_or_local_worker = backend.current_device in ("cuda", "gpu")
        workers = (
            int(AUDIO_RUNTIME["batch_executor_workers_gpu"])
            if is_gpu_or_local_worker
            else int(AUDIO_RUNTIME["batch_executor_workers_cpu"])
        )
        executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="nw-batch")
        futures: List[Tuple[Future, float]] = []
        current_blocks: List[np.ndarray] = []
        silent_sec = 0.0

        def _submit_segment(blocks: List[np.ndarray]) -> None:
            if not blocks:
                return
            arr = np.concatenate(blocks)
            dur = len(arr) / float(sr)
            if dur < 0.25:
                return
            lang = self._effective_language()
            fut = executor.submit(backend.transcribe, arr, sr, lang)
            futures.append((fut, dur))

        while not self._record_stop_event.is_set():
            try:
                block = self.audio_queue.get(timeout=float(AUDIO_RUNTIME["queue_timeout_seconds"]))
            except queue.Empty:
                continue
            current_blocks.append(block)
            rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
            silent_sec = 0.0 if rms >= silence_thresh else (silent_sec + chunk_sec)

            seg_dur = sum(len(b) for b in current_blocks) / float(sr)
            split_now = (
                seg_dur >= max_seg
                or (seg_dur >= long_seg and silent_sec >= pause_long)
                or (seg_dur >= min_seg and silent_sec >= pause_short)
            )
            if split_now:
                _submit_segment(current_blocks)
                current_blocks = []
                silent_sec = 0.0

        self._close_audio_stream()
        while not self.audio_queue.empty():
            try:
                current_blocks.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        _submit_segment(current_blocks)

        self._enqueue_ui(lambda: self._set_app_state("transcribing", "Transcribing batch recording..."))
        t0 = time.monotonic()
        texts: List[str] = []
        total_dur = 0.0
        error_msg: Optional[str] = None

        for fut, dur in futures:
            total_dur += dur
            try:
                res = fut.result()
                if res.text and res.text.strip():
                    texts.append(res.text.strip())
            except Exception as exc:
                log.warning("Batch segment error: %s", exc)
                error_msg = str(exc)

        executor.shutdown(wait=False)
        latency = time.monotonic() - t0
        full_text = " ".join(texts).strip()

        if full_text:
            copy_to_clipboard(full_text, root=self.root)
            paste_from_clipboard(root=self.root)
            self._record_utterance_stats_and_archive(full_text, total_dur, latency)
            done_msg = f"Copied & pasted ({len(full_text.split())} words)"
        elif error_msg:
            done_msg = f"Batch error: {error_msg[:80]}"
        else:
            done_msg = "No speech detected"

        self.is_recording = False
        self.recording_mode = None
        self._enqueue_ui(lambda msg=done_msg: self._finish_recording_ui(msg))

    def _finish_recording_ui(self, status_msg: str) -> None:
        self.btn_live.configure(state="normal")
        self.btn_batch.configure(state="normal")
        self._update_recording_button_labels()
        self._destroy_mini_window()
        self._set_app_state("ready", status_msg)

    # ------------------------------------------------------------------
    # Wayland-Safe Floating Mini-Pill (X11 Only)
    # ------------------------------------------------------------------

    def _on_root_unmap(self, event) -> None:
        if event.widget is not self.root:
            return
        # On Wayland, overrideredirect Toplevels cannot position themselves and get stranded at (0,0)
        if self.session_type != "x11" or not self.is_recording:
            return
        self._show_x11_mini_window()

    def _on_root_map(self, event) -> None:
        if event.widget is self.root:
            self._destroy_mini_window()

    def _show_x11_mini_window(self) -> None:
        if self._mini_win is not None or self.session_type != "x11":
            return
        try:
            mini = tk.Toplevel(self.root)
            mini.overrideredirect(True)
            mini.attributes("-topmost", True)
            w = int(UI_RUNTIME["mini_window_width"])
            h = int(UI_RUNTIME["mini_window_height"])
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            x = max(20, sw - w - int(UI_RUNTIME["mini_window_right_margin_px"]))
            y = max(20, sh - h - int(UI_RUNTIME["mini_window_bottom_margin_px"]))
            mini.geometry(f"{w}x{h}+{x}+{y}")
            mini.configure(bg=self.colors["accent_bg"])

            inner = tk.Frame(
                mini,
                bg=self.colors["accent_bg"],
                highlightthickness=1,
                highlightbackground=self.colors["teal"],
                padx=10,
                pady=8,
            )
            inner.pack(fill="both", expand=True)
            lbl = tk.Label(
                inner,
                text=f"● REC ({(self.recording_mode or 'live').upper()})",
                bg=self.colors["accent_bg"],
                fg=self.colors["live"] if self.recording_mode == "live" else self.colors["batch"],
                font=ui_font(10, "bold"),
            )
            lbl.pack(side="left")
            ctk.CTkButton(
                inner,
                text="Open",
                width=54,
                height=24,
                corner_radius=5,
                font=ui_font(9, "bold"),
                fg_color=self.colors["teal_dim"],
                command=self._show_window_main,
            ).pack(side="right")
            self._mini_win = mini
        except Exception:
            self._mini_win = None

    def _destroy_mini_window(self) -> None:
        if self._mini_win is not None:
            try:
                self._mini_win.destroy()
            except Exception:
                pass
            self._mini_win = None

    # ------------------------------------------------------------------
    # Thread-Safe Status, IPC Command Dispatch, & 30ms Main Loop
    # ------------------------------------------------------------------

    def _set_app_state(self, state: str, message: Optional[str] = None) -> None:
        self.current_state = state
        dot_color = self.colors.get(state, self.colors["idle"])
        if hasattr(self, "status_dot"):
            self.status_dot.itemconfigure(self._status_dot_id, fill=dot_color)
        if message is not None:
            self.set_status(message)
        if self.tray_controller is not None:
            self.tray_controller.set_state(state)

    def set_status(self, message: str) -> None:
        if hasattr(self, "status_label") and self.status_label.winfo_exists():
            self.status_label.configure(text=message)

    def _enqueue_ui(self, fn: Callable[[], None]) -> None:
        self.ui_queue.put(fn)

    def _enqueue_hotkey_action(self, action: str) -> None:
        try:
            self.hotkey_action_queue.put_nowait(action)
        except queue.Full:
            pass

    def handle_ipc_command(self, cmd: str) -> Dict[str, Any]:
        """Thread-safe handler called by ``IPCServer`` for CLI ``--toggle-*`` / ``--status`` flags."""
        command = (cmd or "").strip().lower()
        if command in ("status", "ping"):
            return {
                "ok": True,
                "pid": os.getpid(),
                "version": __version__,
                "session": self.session_type,
                "state": self.current_state,
                "is_recording": self.is_recording,
                "recording_mode": self.recording_mode,
                "transcription_mode": self.config.get("transcription_mode", "local"),
                "model": MODEL_MAP.get(str(self.config.get("model_key", "")), "small"),
            }

        if command in ("toggle_live", "toggle_batch", "stop", "show"):
            self._enqueue_hotkey_action(command)
            return {"ok": True, "queued": command}

        return {"ok": False, "error": f"Unsupported command: {command}"}

    def _gui_update_loop(self) -> None:
        if self._shutting_down:
            return

        # 1. Drain queued UI callbacks
        while True:
            try:
                cb = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                cb()
            except Exception as exc:
                log.debug("UI callback error: %s", exc)

        # 2. Drain hotkey & CLI IPC actions
        while True:
            try:
                act = self.hotkey_action_queue.get_nowait()
            except queue.Empty:
                break
            if act == "toggle_live":
                self.toggle_live_recording()
            elif act == "toggle_batch":
                self.toggle_batch_recording()
            elif act == "stop":
                self.stop_recording()
            elif act == "show":
                self._show_window_main()

        # 3. Update live VU meter bar
        if hasattr(self, "vu_canvas") and self.vu_canvas.winfo_exists():
            total_w = max(1, self.vu_canvas.winfo_width())
            bar_w = int(total_w * self._current_vu_level) if self.is_recording else 0
            bar_color = (
                self.colors["live"]
                if self.recording_mode == "live"
                else (self.colors["batch"] if self.recording_mode == "batch" else self.colors["teal"])
            )
            self.vu_canvas.coords(self._vu_rect, 0, 0, bar_w, 6)
            self.vu_canvas.itemconfigure(self._vu_rect, fill=bar_color)

        # 4. Periodic idle model unload check
        self._check_idle_model_unload()

        self.root.after(int(UI_RUNTIME["main_loop_interval_ms"]), self._gui_update_loop)
