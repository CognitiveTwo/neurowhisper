"""Frozen-safe macOS application entrypoint, logging bootstrap, and IPC CLI."""

from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import sys
import traceback
from typing import List, Optional

# Ensure package imports work when executed directly by PyInstaller (__package__ is None)
if __package__ in (None, ""):
    _src_root = str(Path(__file__).resolve().parents[1])
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

from neurowhisper_mac import APP_NAME, __version__, ipc, paths  # noqa: E402


def _configure_logging() -> None:
    logs_dir = paths.get_logs_dir()
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    app_log = logs_dir / "app.log"
    crash_log = logs_dir / "crash.log"

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        try:
            fh = RotatingFileHandler(
                app_log,
                maxBytes=2 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            fh.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            root_logger.addHandler(fh)
        except OSError:
            pass

    def _crash_excepthook(exc_type, exc_value, exc_tb) -> None:
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            with open(crash_log, "a", encoding="utf-8") as cf:
                cf.write(tb_text + "\n")
        except OSError:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _crash_excepthook


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="neurowhisper standalone macOS speech-to-text application.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--toggle-live",
        action="store_true",
        help="Toggle Live Dictation in the running neurowhisper instance.",
    )
    group.add_argument(
        "--toggle-batch",
        action="store_true",
        help="Toggle Batch Recording in the running neurowhisper instance.",
    )
    group.add_argument(
        "--stop",
        action="store_true",
        help="Stop active recording in the running neurowhisper instance.",
    )
    group.add_argument(
        "--show",
        action="store_true",
        help="Reveal and focus the running neurowhisper window.",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Query JSON status from the running neurowhisper instance.",
    )
    return parser


def _filter_macos_finder_args(argv: List[str]) -> List[str]:
    """Strip macOS LaunchServices ``-psn_0_...`` process serial number arguments."""
    return [arg for arg in argv if not arg.startswith("-psn_")]


def _run_cli_ipc_command(cmd: str) -> int:
    resp = ipc.send_ipc_command(cmd)
    print(json.dumps(resp))
    return 0 if resp.get("ok") else 1


def main(argv: Optional[List[str]] = None) -> int:
    multiprocessing.freeze_support()

    raw_args = list(sys.argv[1:] if argv is None else argv)
    clean_args = _filter_macos_finder_args(raw_args)
    parser = build_arg_parser()
    args = parser.parse_args(clean_args)

    if args.version:
        print(f"{APP_NAME} {__version__}")
        return 0

    if args.toggle_live:
        return _run_cli_ipc_command("toggle_live")
    if args.toggle_batch:
        return _run_cli_ipc_command("toggle_batch")
    if args.stop:
        return _run_cli_ipc_command("stop")
    if args.show:
        return _run_cli_ipc_command("show")
    if args.status:
        return _run_cli_ipc_command("status")

    paths.ensure_runtime_dirs()
    _configure_logging()

    allow_multi = os.environ.get("NEUROWHISPER_ALLOW_MULTI_INSTANCE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not allow_multi and ipc.is_instance_running():
        ipc.send_ipc_command("show")
        return 0

    import customtkinter as ctk

    from neurowhisper_mac.gui import WhisperApp
    from neurowhisper_mac.menubar import MenuBarController

    root = ctk.CTk()
    menubar = MenuBarController()
    app = WhisperApp(root, menubar=menubar)

    ipc_server = ipc.IPCServer(handler=app.handle_ipc_command)
    try:
        ipc_server.start(check_existing=not allow_multi)
    except ipc.IPCAlreadyRunningError:
        ipc.send_ipc_command("show")
        app.on_close()
        return 0
    except Exception as exc:
        logging.getLogger(__name__).warning("IPC server failed to bind: %s", exc)

    menubar.start()

    try:
        root.mainloop()
    finally:
        ipc_server.stop()
        menubar.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
