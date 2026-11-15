"""Frozen-safe entrypoint and CLI controller for the standalone Linux ``neurowhisper`` app.

CRITICAL: ``--fw-worker`` is intercepted BEFORE importing ``tkinter``, ``customtkinter``,
or ``pystray`` so spawning ``[sys.executable, '--fw-worker']`` inside a frozen AppImage
never opens a second GUI window or touches X11/Wayland.
"""

from __future__ import annotations

import multiprocessing
import os as _os
import sys

if __package__ in (None, "") and not getattr(sys, "frozen", False):
    _src_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

# 1. Intercept child faster-whisper worker BEFORE any GUI imports!
if "--fw-worker" in sys.argv:
    multiprocessing.freeze_support()
    from neurowhisper_linux.backends.fw_worker import main as _fw_worker_main

    _fw_worker_main()
    sys.exit(0)

import argparse
import json
import logging
import os
import signal
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List, Optional

from neurowhisper_linux import APP_NAME, __version__, paths
from neurowhisper_linux.ipc import IPCServer, is_instance_running, send_ipc_command


def _setup_logging() -> logging.Logger:
    logs_dir = paths.get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "app.log"

    root_logger = logging.getLogger("neurowhisper")
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        if os.environ.get("NEUROWHISPER_DEBUG", "").strip() == "1":
            root_logger.setLevel(logging.DEBUG)
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setFormatter(formatter)
            root_logger.addHandler(stream_handler)

    def _excepthook(exc_type, exc_value, exc_tb) -> None:
        crash_file = logs_dir / "crash.log"
        try:
            with open(crash_file, "a", encoding="utf-8") as fh:
                fh.write(f"\n--- {datetime.now().isoformat()} ---\n")
                traceback.print_exception(exc_type, exc_value, exc_tb, file=fh)
        except OSError:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook
    return root_logger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="neurowhisper — Standalone Linux AI Voice Dictation & Speech Insights",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{APP_NAME} {__version__}",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--toggle-live",
        action="store_true",
        help="Toggle Live Dictation in the running neurowhisper instance (Wayland/X11 shortcut friendly)",
    )
    group.add_argument(
        "--toggle-batch",
        action="store_true",
        help="Toggle Batch Recording in the running neurowhisper instance (Wayland/X11 shortcut friendly)",
    )
    group.add_argument(
        "--stop",
        action="store_true",
        help="Stop any active recording in the running neurowhisper instance",
    )
    group.add_argument(
        "--show",
        action="store_true",
        help="Show and focus the running neurowhisper window",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Print JSON status of the running neurowhisper instance",
    )
    group.add_argument(
        "--install-desktop",
        action="store_true",
        help="Install ~/.local/share/applications/neurowhisper.desktop and icon",
    )
    return parser


def _handle_cli_ipc_command(cmd: str, print_json: bool = False) -> int:
    resp = send_ipc_command(cmd, timeout=2.5)
    if print_json:
        print(json.dumps(resp, indent=2))
    elif not resp.get("ok"):
        sys.stderr.write(f"{APP_NAME}: {resp.get('error', 'not running')}\n")
    return 0 if resp.get("ok") else 1


def main(argv: Optional[List[str]] = None) -> int:
    multiprocessing.freeze_support()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.install_desktop:
        from neurowhisper_linux.platform_linux import install_desktop_entry

        desktop_path, icon_path = install_desktop_entry()
        print(f"Installed desktop entry: {desktop_path}")
        print(f"Installed icon:          {icon_path}")
        return 0

    if args.toggle_live:
        return _handle_cli_ipc_command("toggle_live")
    if args.toggle_batch:
        return _handle_cli_ipc_command("toggle_batch")
    if args.stop:
        return _handle_cli_ipc_command("stop")
    if args.show:
        return _handle_cli_ipc_command("show")
    if args.status:
        return _handle_cli_ipc_command("status", print_json=True)

    paths.ensure_user_dirs()
    logger = _setup_logging()

    # Single-instance check: focus existing window if already running
    if is_instance_running():
        send_ipc_command("show", timeout=1.5)
        logger.info("Focused existing neurowhisper instance via IPC")
        return 0

    import tkinter as tk
    import customtkinter as ctk

    from neurowhisper_linux.config import CTK_APPEARANCE_MODE, CTK_COLOR_THEME, WM_CLASS
    from neurowhisper_linux.gui import WhisperApp
    from neurowhisper_linux.platform_linux import open_path_or_url
    from neurowhisper_linux.tray import SystemTrayController

    ctk.set_appearance_mode(CTK_APPEARANCE_MODE)
    ctk.set_default_color_theme(CTK_COLOR_THEME)

    # Set className="neurowhisper" so X11 WM_CLASS & Wayland app_id match StartupWMClass=neurowhisper
    root = tk.Tk(className=WM_CLASS)
    app = WhisperApp(root)

    ipc_server = IPCServer(handler=app.handle_ipc_command)
    ipc_server.start()

    tray = SystemTrayController(
        on_toggle_live=lambda: app.handle_ipc_command("toggle_live"),
        on_toggle_batch=lambda: app.handle_ipc_command("toggle_batch"),
        on_stop_recording=lambda: app.handle_ipc_command("stop"),
        on_toggle_window=app.toggle_window_visibility,
        on_open_transcriptions=lambda: open_path_or_url(paths.get_transcriptions_dir()),
        on_quit=lambda: root.after(0, app.shutdown),
    )
    tray.start()
    app.set_tray_controller(tray)

    def _sig_handler(_signum, _frame) -> None:
        try:
            root.after(0, app.shutdown)
        except Exception:
            sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        root.mainloop()
    finally:
        ipc_server.stop()
        tray.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
