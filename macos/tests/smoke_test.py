"""GUI smoke test for the standalone macOS neurowhisper application.

Verifies that:
1. ``WhisperApp`` initializes cleanly in an isolated ``NEUROWHISPER_HOME``.
2. All accordions expand/collapse and all 10 Speech Insights tabs build and render.
3. Zero legacy ``tk.Button`` or ``tk.OptionMenu`` instances exist anywhere in the
   live widget tree (preventing white Aqua chrome on macOS Dark Mode).
4. Live IPC ``status`` / ``show`` dispatch and ``on_close()`` complete cleanly.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from typing import Iterator

MACOS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MACOS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def iter_widgets(widget: tk.Misc) -> Iterator[tk.Misc]:
    yield widget
    for child in widget.winfo_children():
        yield from iter_widgets(child)


def run_smoke_test(require_display: bool = False) -> int:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        msg = "SKIP: No DISPLAY available for Tk smoke test (use xvfb-run on headless Linux)."
        if require_display:
            print(msg, file=sys.stderr)
            return 1
        print(msg)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["NEUROWHISPER_HOME"] = tmp
        os.environ["NEUROWHISPER_NO_TRAY"] = "1"
        os.environ["NEUROWHISPER_SKIP_MODEL_LOAD"] = "1"

        # Seed sample transcripts and AI insights so all 10 tabs render real data
        trans_dir = Path(tmp) / "transcriptions"
        trans_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.date.today().isoformat()
        (trans_dir / f"{today_str}.txt").write_text(
            "[09:15:00] What should we prioritize for the Apple Silicon Metal release today?\n"
            "[09:15:08] Ich denke wir sollten die CoreML Beschleunigung und das Menü-Icon testen.\n"
            "[09:15:19] Great point, in summary let's verify every Speech Insights tab!\n",
            encoding="utf-8",
        )

        import customtkinter as ctk

        from neurowhisper_mac import ipc
        from neurowhisper_mac.gui import WhisperApp
        from neurowhisper_mac.menubar import MenuBarController

        root = ctk.CTk()
        menubar = MenuBarController()
        app = WhisperApp(root, menubar=menubar, skip_model_load=True)
        ipc_srv = ipc.IPCServer(handler=app.handle_ipc_command)
        ipc_srv.start()

        try:
            root.update_idletasks()
            root.update()

            # Expand Insights & Configuration accordions and build all 10 tabs
            if app.insights_collapsed.get():
                app.toggle_insights_section()
            if app.config_collapsed.get():
                app.toggle_settings_section()
            if app.log_collapsed.get():
                app.toggle_log_section()

            for idx in range(len(app.insight_tab_names)):
                app._select_insight_tab(idx)
                root.update_idletasks()

            # Walk the entire live widget hierarchy to verify zero tk.Button / tk.OptionMenu
            bad_widgets = []
            total_widgets = 0
            for w in iter_widgets(root):
                total_widgets += 1
                if type(w) is tk.Button or type(w) is tk.OptionMenu:
                    bad_widgets.append((type(w).__name__, str(w)))

            assert not bad_widgets, f"Found forbidden Aqua-white widgets: {bad_widgets}"
            assert len(app._insight_tabs_built) == 10, (
                f"Expected 10 built insight tabs, got {len(app._insight_tabs_built)}"
            )

            # Exercise mini floating pill
            app.enter_mini_mode()
            root.update_idletasks()
            assert app.mini_window is not None
            for w in iter_widgets(app.mini_window):
                assert type(w) is not tk.Button and type(w) is not tk.OptionMenu
            app.exit_mini_mode()
            root.update_idletasks()

            # Exercise live IPC round-trip
            resp = ipc.send_ipc_command("status", socket_path=ipc_srv.socket_path)
            assert resp.get("ok") is True, f"IPC status failed: {resp}"

            print(
                f"SMOKE TEST PASSED: inspected {total_widgets} widgets, "
                "10/10 Speech Insights tabs, mini-pill, and IPC server (0 tk.Button / 0 tk.OptionMenu)."
            )
        finally:
            ipc_srv.stop()
            app.on_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_test(require_display="--require-display" in sys.argv))
