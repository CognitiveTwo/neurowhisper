"""End-to-end headless + Xvfb/X11 GUI smoke test for ``neurowhisper_linux``.

Runs the full unit test suite first, and if ``$DISPLAY`` or ``$WAYLAND_DISPLAY`` is present
(e.g. under ``xvfb-run -a``), boots the real ``WhisperApp`` + ``IPCServer`` inside an isolated
``NEUROWHISPER_HOME``, verifies live IPC ``status`` / ``show`` / tab switching across all 10
Speech Insights tabs, and shuts down cleanly.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

LINUX_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = LINUX_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def run_gui_smoke_test() -> None:
    """Launch ``WhisperApp`` + ``IPCServer`` on an active X11/Xvfb/Wayland display and exercise all tabs."""
    import tkinter as tk
    import customtkinter as ctk

    from neurowhisper_linux import paths
    from neurowhisper_linux.config import CTK_APPEARANCE_MODE, CTK_COLOR_THEME, WM_CLASS
    from neurowhisper_linux.gui import WhisperApp
    from neurowhisper_linux.ipc import IPCServer, send_ipc_command

    with tempfile.TemporaryDirectory(prefix="nw-gui-smoke-") as tmp_home:
        env_overrides = {
            "NEUROWHISPER_HOME": tmp_home,
            "NEUROWHISPER_NO_TRAY": "1",
            "NEUROWHISPER_NO_AUDIO_FEEDBACK": "1",
            "NEUROWHISPER_DISABLE_HOTKEYS": "1",
        }
        old_env = {k: os.environ.get(k) for k in env_overrides}
        os.environ.update(env_overrides)
        try:
            paths.ensure_user_dirs()
            # Seed a sample transcript so all 10 Speech Insights visualizations render real data
            sample_archive = paths.get_transcriptions_dir() / "2026-11.smoke.txt"
            sample_archive.write_text(
                "[2026-11-15 10:00:00] What do we need for the standalone Linux AppImage build?\n"
                "[2026-11-15 10:00:18] Great job verifying X11 and Wayland dual-stack shortcuts!\n",
                encoding="utf-8",
            )
            # Set online mode in seeded config so smoke test doesn't download a 460 MB model from HF
            paths.write_json_atomic(
                paths.get_local_config_file(),
                {"transcription_mode": "online"},
            )

            ctk.set_appearance_mode(CTK_APPEARANCE_MODE)
            ctk.set_default_color_theme(CTK_COLOR_THEME)

            root = tk.Tk(className=WM_CLASS)
            app = WhisperApp(root)
            ipc_server = IPCServer(handler=app.handle_ipc_command)
            assert ipc_server.start(), "IPCServer failed to bind socket in GUI smoke test"

            errors: list[BaseException] = []

            def _exercise_ui() -> None:
                try:
                    # 1. Verify live IPC status roundtrip
                    st = send_ipc_command("status", socket_path=ipc_server.socket_path)
                    assert st.get("ok") is True, f"IPC status failed: {st}"
                    assert st.get("state") in ("ready", "idle"), f"Unexpected app state: {st}"

                    # 2. Expand all accordions (including Insights & Configuration)
                    for acc_key in ("insights", "config"):
                        if not app._accordions[acc_key]["expanded"]:
                            app._toggle_accordion(acc_key)
                    root.update_idletasks()

                    # 3. Switch through both histogram modes and all 10 Speech Insights tabs
                    app.switch_histogram_view("alltime")
                    app.switch_histogram_view("today")
                    for tab_name in app.INSIGHT_TABS:
                        app.switch_insight_tab(tab_name)
                        root.update_idletasks()
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    ipc_server.stop()
                    app.shutdown()

            root.after(180, _exercise_ui)
            root.mainloop()

            if errors:
                raise errors[0]
            print("GUI + IPC smoke test PASSED!")
        finally:
            for k, prev in old_env.items():
                if prev is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(Path(__file__).resolve().parent), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        return 1

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        print("\nDisplay detected — running live Tkinter/CustomTkinter GUI + IPC smoke test...")
        run_gui_smoke_test()
    else:
        print("\nNo DISPLAY/WAYLAND_DISPLAY set — skipped GUI window smoke step (use xvfb-run -a).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
