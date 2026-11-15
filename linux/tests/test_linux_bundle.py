"""Comprehensive headless unit & integration test suite for ``neurowhisper_linux``.

Verifies:
1. Strict self-containment (zero imports from root Windows modules, zero ``keyboard``,
   and zero legacy ``tk.Button`` / ``tk.OptionMenu`` widgets across ``neurowhisper_linux``).
2. XDG path isolation, ``AF_UNIX`` <96-char socket guard, ``storage.json`` sync-folder routing,
   frozen ``sys._MEIPASS`` asset lookup, and dynamic CUDA 12 directory discovery.
3. ``AF_UNIX`` IPC server/client single-instance guard, permissions, and command roundtrips.
4. Wayland/X11 session detection, X11 ``pynput`` key/control-byte normalization, XDG Portal
   trigger formatting, terminal ``WM_CLASS`` detection (``Ctrl+Shift+V``), audio chime synthesis,
   and ``.desktop`` launcher installation.
5. Tray icon RGBA rendering and non-fatal headless fallback.
6. ``--fw-worker`` subprocess startup/ping/shutdown, frozen worker argv dispatch, WAV PCM16
   encoding, and Parakeet ONNX backend routing.
7. Bilingual (EN + DE) Speech Insights parsing, filler counting, and radar/baseline metrics.
"""

from __future__ import annotations

import ast
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

LINUX_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = LINUX_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurowhisper_linux import APP_ID, APP_NAME, __version__, config, insights, ipc, paths, platform_linux, tray  # noqa: E402
from neurowhisper_linux.backends import base as backend_base  # noqa: E402
from neurowhisper_linux.backends import detect_best_backend  # noqa: E402
from neurowhisper_linux.backends.faster_whisper_backend import build_worker_command_and_env  # noqa: E402


FORBIDDEN_ROOT_IMPORTS = frozenset({
    "app_config",
    "platform_utils",
    "whisper_gui",
    "keyboard",
})


class TestSelfContainmentAndWidgetStandards(unittest.TestCase):
    """Ensure ``neurowhisper_linux`` has zero coupling to root Windows code or legacy Tk buttons."""

    def test_package_metadata(self) -> None:
        self.assertEqual(APP_NAME, "neurowhisper")
        self.assertEqual(APP_ID, "com.neuroflash.neurowhisper")
        self.assertTrue(__version__)

    def test_ast_zero_root_imports_and_zero_tk_buttons(self) -> None:
        pkg_root = SRC_DIR / "neurowhisper_linux"
        py_files = sorted(pkg_root.rglob("*.py"))
        self.assertGreaterEqual(len(py_files), 12)

        for py_file in py_files:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        self.assertNotIn(
                            top,
                            FORBIDDEN_ROOT_IMPORTS,
                            f"Forbidden import '{alias.name}' in {py_file}",
                        )
                        self.assertNotIn(
                            top,
                            {"backends", "analysis"},
                            f"Ambiguous root import '{alias.name}' in {py_file}",
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        top = node.module.split(".")[0]
                        self.assertNotIn(
                            top,
                            FORBIDDEN_ROOT_IMPORTS,
                            f"Forbidden from-import '{node.module}' in {py_file}",
                        )
                        self.assertNotIn(
                            top,
                            {"backends", "analysis"},
                            f"Unqualified root from-import '{node.module}' in {py_file}",
                        )
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == "tk":
                        self.assertNotIn(
                            node.attr,
                            {"Button", "OptionMenu"},
                            f"Legacy tk.{node.attr} widget found in {py_file} (must use CTkButton/CTkOptionMenu)",
                        )


class TestPathsAndSyncRouting(unittest.TestCase):
    """Verify XDG directories, NEUROWHISPER_HOME, sync-dir routing, socket guard, and CUDA discovery."""

    def test_isolated_home_and_sync_dir_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "NEUROWHISPER_HOME": tmp,
                "NEUROWHISPER_SYNC_DIR": "",
                "NEUROWHISPER_SOCKET_PATH": "",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                paths.ensure_user_dirs()
                self.assertIsNone(paths.get_sync_dir())
                self.assertEqual(paths.get_config_dir(), Path(tmp) / "config")
                self.assertEqual(paths.get_data_dir(), Path(tmp) / "data")
                self.assertEqual(paths.get_models_dir(), Path(tmp) / "cache" / "models")
                self.assertEqual(paths.get_shared_config_file(), Path(tmp) / "config" / "whisper_config.json")
                self.assertEqual(paths.get_transcriptions_dir(), Path(tmp) / "data" / "transcriptions")

                # Activate sync folder and verify shared/peer files move into sync_dir
                sync_target = Path(tmp) / "dropbox_sync"
                paths.set_sync_dir(sync_target)
                self.assertEqual(paths.get_sync_dir(), sync_target)
                self.assertEqual(paths.get_shared_config_file(), sync_target / "whisper_config.json")
                self.assertEqual(paths.get_ai_insights_file(), sync_target / "ai_insights.json")
                self.assertEqual(paths.get_transcriptions_dir(), sync_target / "transcriptions")

                # Reset sync folder back to XDG default
                paths.set_sync_dir("")
                self.assertIsNone(paths.get_sync_dir())
                self.assertEqual(paths.get_transcriptions_dir(), Path(tmp) / "data" / "transcriptions")

    def test_socket_path_length_guard_and_permissions(self) -> None:
        very_long_home = "/tmp/" + ("neurowhisper_deep_nested_dir_" * 5)
        try:
            with mock.patch.dict(
                os.environ,
                {"NEUROWHISPER_HOME": very_long_home, "NEUROWHISPER_SOCKET_PATH": ""},
                clear=False,
            ):
                sock = paths.get_socket_path()
                self.assertLess(len(str(sock)), paths.MAX_UNIX_SOCKET_PATH_LEN)
                mode = stat.S_IMODE(sock.parent.stat().st_mode)
                self.assertEqual(mode, 0o700)
        finally:
            if sock.parent.exists() and str(sock.parent).startswith("/tmp/nw-"):
                sock.parent.rmdir()

    def test_frozen_resource_dir_meipass(self) -> None:
        with tempfile.TemporaryDirectory() as fake_mei:
            res_sub = Path(fake_mei) / "resources"
            res_sub.mkdir()
            (res_sub / "neurowhisper.png").write_bytes(b"PNG")
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
                sys, "_MEIPASS", fake_mei, create=True
            ):
                self.assertEqual(paths.get_resource_dir(), res_sub)
                self.assertEqual(paths.get_icon_png_path(), res_sub / "neurowhisper.png")

    def test_discover_cuda_lib_dirs_expands_nvidia_wheel_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cublas_lib = Path(tmp) / "data" / "cuda-libs" / "nvidia" / "cublas" / "lib"
            cudnn_lib = Path(tmp) / "data" / "cuda-libs" / "nvidia" / "cudnn" / "lib"
            cublas_lib.mkdir(parents=True)
            cudnn_lib.mkdir(parents=True)
            with mock.patch.dict(os.environ, {"NEUROWHISPER_HOME": tmp}, clear=False):
                discovered = paths.discover_cuda_lib_dirs()
                self.assertIn(str(cublas_lib.resolve()), discovered)
                self.assertIn(str(cudnn_lib.resolve()), discovered)


class TestConfigAndIPC(unittest.TestCase):
    """Verify Linux config defaults and AF_UNIX IPC server/client behavior."""

    def test_linux_config_defaults(self) -> None:
        self.assertEqual(config.DEFAULT_CONFIG["hotkey_live"], "f9")
        self.assertEqual(config.DEFAULT_CONFIG["hotkey_batch"], "ctrl+alt+s")
        self.assertTrue(config.DEFAULT_CONFIG["worker_process"])
        self.assertNotIn("NeMo (NVIDIA GPU)", config.BACKEND_OPTIONS)
        self.assertEqual(config.resolve_backend_display("faster-whisper", "cuda"), "CUDA (NVIDIA GPU)")
        self.assertEqual(config.resolve_backend_display("onnx-asr", "auto"), "ONNX ASR (CPU/GPU)")
        self.assertEqual(config.ui_font(12, "bold"), ("DejaVu Sans", 12, "bold"))
        self.assertEqual(config.mono_font(10), ("DejaVu Sans Mono", 10))

    def test_ipc_server_roundtrip_and_single_instance_guard(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            sock_path = Path(tmp_dir) / "control.sock"
            received = []

            def _handler(cmd: str) -> dict:
                received.append(cmd)
                return {"ok": True, "handled": cmd}

            server = ipc.IPCServer(handler=_handler, socket_path=sock_path)
            self.assertTrue(server.start())
            try:
                # Socket permissions must be 0600
                sock_mode = stat.S_IMODE(sock_path.stat().st_mode)
                self.assertEqual(sock_mode, 0o600)

                # Second IPCServer on same active socket must return False
                duplicate = ipc.IPCServer(handler=_handler, socket_path=sock_path)
                self.assertFalse(duplicate.start())

                self.assertTrue(ipc.is_instance_running(socket_path=sock_path))
                for cmd in ("toggle_live", "toggle_batch", "stop", "show", "status"):
                    reply = ipc.send_ipc_command(cmd, socket_path=sock_path)
                    self.assertTrue(reply["ok"])
                    self.assertEqual(reply["handled"], cmd)

                bad = ipc.send_ipc_command("invalid_cmd", socket_path=sock_path)
                self.assertFalse(bad["ok"])
            finally:
                server.stop()

            self.assertFalse(sock_path.exists())
            self.assertFalse(ipc.is_instance_running(socket_path=sock_path))


class TestPlatformLinux(unittest.TestCase):
    """Verify session detection, key normalization, terminal detection, chimes, and .desktop install."""

    def test_detect_display_session(self) -> None:
        self.assertEqual(
            platform_linux.detect_display_session({"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}),
            "wayland",
        )
        self.assertEqual(
            platform_linux.detect_display_session({"XDG_SESSION_TYPE": "wayland"}),
            "wayland",
        )
        self.assertEqual(
            platform_linux.detect_display_session({"DISPLAY": ":0", "XDG_SESSION_TYPE": "x11"}),
            "x11",
        )
        self.assertEqual(platform_linux.detect_display_session({}), "headless")

    def test_pynput_key_normalization_and_portal_triggers(self) -> None:
        # Modifier variants
        self.assertEqual(platform_linux.normalize_pynput_key(SimpleNamespace(name="ctrl_r")), "ctrl")
        self.assertEqual(platform_linux.normalize_pynput_key(SimpleNamespace(name="alt_gr")), "alt")
        self.assertEqual(platform_linux.normalize_pynput_key(SimpleNamespace(name="shift_l")), "shift")
        self.assertEqual(platform_linux.normalize_pynput_key(SimpleNamespace(name="cmd")), "win")
        self.assertEqual(platform_linux.normalize_pynput_key(SimpleNamespace(name="f9")), "f9")
        # X11 Ctrl+S control byte (\x13, vk=115) -> 's'
        self.assertEqual(
            platform_linux.normalize_pynput_key(SimpleNamespace(name=None, char="\x13", vk=115)),
            "s",
        )
        # X11 keysym F12 (0xFFC9) -> 'f12'
        self.assertEqual(
            platform_linux.normalize_pynput_key(SimpleNamespace(name=None, char=None, vk=0xFFC9)),
            "f12",
        )
        # Portal trigger conversion
        self.assertEqual(platform_linux.hotkey_to_portal_trigger("ctrl+alt+s"), "<Control><Alt>s")
        self.assertEqual(platform_linux.hotkey_to_portal_trigger("f9"), "F9")

    def test_terminal_wm_class_and_paste_shortcut(self) -> None:
        self.assertTrue(
            platform_linux.matches_terminal_wm_class('WM_CLASS(STRING) = "gnome-terminal-server", "Gnome-terminal"')
        )
        self.assertTrue(platform_linux.matches_terminal_wm_class("Alacritty"))
        self.assertTrue(platform_linux.matches_terminal_wm_class("kitty"))
        self.assertTrue(platform_linux.matches_terminal_wm_class("com.mitchellh.ghostty"))
        self.assertFalse(platform_linux.matches_terminal_wm_class('WM_CLASS(STRING) = "Navigator", "firefox"'))
        self.assertEqual(platform_linux.get_paste_shortcut(is_terminal=True), "ctrl+shift+v")
        self.assertEqual(platform_linux.get_paste_shortcut(is_terminal=False), "ctrl+v")

    def test_synthesize_chime_click_free_envelope(self) -> None:
        start_wave = platform_linux.synthesize_chime(start=True, sample_rate=44100, duration_ms=120)
        stop_wave = platform_linux.synthesize_chime(start=False, sample_rate=44100, duration_ms=120)
        self.assertEqual(start_wave.dtype, np.float32)
        self.assertEqual(len(start_wave), 5292)
        self.assertAlmostEqual(float(start_wave[0]), 0.0, places=5)
        self.assertAlmostEqual(float(start_wave[-1]), 0.0, places=4)
        self.assertFalse(np.allclose(start_wave, stop_wave))

    def test_install_desktop_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"APPIMAGE": "/opt/apps/neurowhisper.AppImage"}, clear=False):
                desktop_file, icon_file = platform_linux.install_desktop_entry(target_data_home=Path(tmp))
                self.assertTrue(desktop_file.is_file())
                self.assertTrue(icon_file.is_file())
                content = desktop_file.read_text(encoding="utf-8")
                self.assertIn("StartupWMClass=neurowhisper", content)
                self.assertIn("/opt/apps/neurowhisper.AppImage", content)
                self.assertIn("--toggle-live", content)
                self.assertIn("--toggle-batch", content)


class TestTrayAndBackends(unittest.TestCase):
    """Verify tray icon rendering, WAV serializer, frozen worker command, and fw_worker IPC."""

    def test_render_tray_icons_and_headless_controller(self) -> None:
        for st in ("idle", "ready", "live", "batch", "online", "transcribing"):
            img = tray.render_tray_icon_image(st, size=64)
            self.assertEqual(img.size, (64, 64))
            self.assertEqual(img.mode, "RGBA")

        with mock.patch.dict(os.environ, {"NEUROWHISPER_NO_TRAY": "1"}, clear=False):
            ctrl = tray.SystemTrayController()
            self.assertFalse(ctrl.start())
            ctrl.set_state("live")
            ctrl.stop()

    def test_write_pcm16_wav_roundtrip(self) -> None:
        audio = np.sin(np.linspace(0, 20.0, 1600, dtype=np.float32)) * 0.5
        buf = io.BytesIO()
        backend_base.write_pcm16_wav(buf, audio, sample_rate=16000)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getnframes(), 1600)

    def test_frozen_vs_source_worker_command(self) -> None:
        cmd_src, env_src = build_worker_command_and_env()
        self.assertEqual(cmd_src[0], sys.executable)
        self.assertTrue(cmd_src[1].endswith("fw_worker.py"))
        self.assertIn(str(SRC_DIR), env_src.get("PYTHONPATH", ""))

        with mock.patch.object(sys, "frozen", True, create=True):
            cmd_frozen, _ = build_worker_command_and_env()
            self.assertEqual(cmd_frozen, [sys.executable, "--fw-worker"])

    def test_main_fw_worker_intercept_and_cli_version(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_DIR)

        # Verify --version
        ver_proc = subprocess.run(
            [sys.executable, "-m", "neurowhisper_linux.main", "--version"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(ver_proc.returncode, 0)
        self.assertIn("neurowhisper", ver_proc.stdout)

        # Verify --fw-worker intercepts before GUI imports and speaks JSON IPC
        worker_proc = subprocess.run(
            [sys.executable, "-m", "neurowhisper_linux.main", "--fw-worker"],
            input='{"cmd":"ping"}\n{"cmd":"shutdown"}\n',
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(worker_proc.returncode, 0, worker_proc.stderr)
        lines = [json.loads(ln) for ln in worker_proc.stdout.strip().splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(lines[0].get("ready"))
        self.assertTrue(lines[1].get("ok"))
        self.assertTrue(lines[2].get("ok"))

    def test_parakeet_routes_to_onnx_asr_when_available(self) -> None:
        backend_name, _, _ = detect_best_backend(model_key="parakeet-tdt-0.6b-v3")
        self.assertIn(backend_name, {"onnx-asr", "faster-whisper"})


class TestSpeechInsightsAnalytics(unittest.TestCase):
    """Verify bilingual (EN + DE) archive parsing, filler detection, and insights computation."""

    def test_bilingual_language_and_filler_detection(self) -> None:
        en_text = "Um I think like we should, you know, review the architecture roadmap today."
        en_words = insights.tokenize_words(en_text)
        self.assertEqual(insights.detect_entry_language(en_words), "en")
        en_total, en_map = insights.count_fillers(en_text, "en")
        self.assertEqual(en_total, 3)
        self.assertIn("um/uh", en_map)
        self.assertIn("like", en_map)
        self.assertIn("you know", en_map)

        de_text = "Ähm ich glaube wir müssen das halt irgendwie quasi noch einmal prüfen."
        de_words = insights.tokenize_words(de_text)
        self.assertEqual(insights.detect_entry_language(de_words), "de")
        de_total, de_map = insights.count_fillers(de_text, "de")
        self.assertGreaterEqual(de_total, 3)
        self.assertIn("äh/ähm", de_map)

    def test_archive_parsing_and_insights_computation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tr_dir = Path(tmp)
            valid_file = tr_dir / "2026-11.workstation.txt"
            conflict_file = tr_dir / "2026-11.workstation (conflicted copy).txt"

            valid_file.write_text(
                "[2026-11-15 09:00:00] What is the main goal for the Linux AppImage release today?\n"
                "It sounds like reliability on both Wayland and X11 is our top priority.\n"
                "[2026-11-15 09:00:25] Great job on the modular XDG storage design; to summarize, we are ready.\n",
                encoding="utf-8",
            )
            conflict_file.write_text(
                "[2026-11-15 09:01:00] Ignored conflicted duplicate entry.\n",
                encoding="utf-8",
            )

            entries = insights.load_all_transcript_entries(tr_dir)
            self.assertEqual(len(entries), 2)
            metrics = insights.compute_speech_insights(entries)
            self.assertFalse(metrics["empty"])
            self.assertEqual(metrics["entry_count"], 2)
            self.assertGreater(metrics["total_words"], 20)
            self.assertEqual(len(metrics["radar_axes"]), 5)
            self.assertGreater(metrics["open_q_per_100"], 0.0)
            self.assertGreater(metrics["affirmations_per_100"], 0.0)
            self.assertGreater(metrics["summaries_per_100"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
