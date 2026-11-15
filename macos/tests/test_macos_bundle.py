"""Comprehensive unit test suite for the standalone macOS neurowhisper bundle."""

from __future__ import annotations

import ast
import datetime
import io
import json
import os
from pathlib import Path
import plistlib
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

import numpy as np

MACOS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MACOS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import neurowhisper_mac  # noqa: E402
from neurowhisper_mac import config, insights, ipc, main as mac_main, menubar, paths, platform_mac  # noqa: E402
from neurowhisper_mac.backends import (  # noqa: E402
    MlxWhisperBackend,
    OnnxAsrBackend,
    OpenAIBackend,
    detect_best_backend,
    write_pcm16_wav,
)


class TestArchitectureBoundaries(unittest.TestCase):
    """Verify strict isolation, zero pynput imports, and zero white-Aqua tk.Button/OptionMenu."""

    FORBIDDEN_ROOT_IMPORTS = frozenset({
        "app_config",
        "platform_utils",
        "whisper_gui",
        "fw_worker",
        "pynput",
    })

    def test_self_contained_imports_and_zero_legacy_tk_buttons(self) -> None:
        pkg_dir = SRC_DIR / "neurowhisper_mac"
        py_files = sorted(pkg_dir.rglob("*.py"))
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
                            self.FORBIDDEN_ROOT_IMPORTS,
                            f"Forbidden import '{alias.name}' in {py_file}",
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        top = node.module.split(".")[0]
                        self.assertNotIn(
                            top,
                            self.FORBIDDEN_ROOT_IMPORTS,
                            f"Forbidden from-import '{node.module}' in {py_file}",
                        )
                        self.assertNotEqual(
                            top,
                            "backends",
                            f"Root 'backends' import leaked into {py_file}",
                        )
                        self.assertNotEqual(
                            top,
                            "analysis",
                            f"Root 'analysis' import leaked into {py_file}",
                        )
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == "tk":
                        self.assertNotIn(
                            node.attr,
                            ("Button", "OptionMenu"),
                            f"Forbidden Aqua-white tk.{node.attr} found in {py_file}",
                        )


class TestPathsAndSyncStorage(unittest.TestCase):
    """Test macOS ~/Library layout, 104-byte socket guard, and cloud sync bootstrap."""

    def test_default_and_isolated_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"NEUROWHISPER_HOME": tmp}, clear=False):
                os.environ.pop("NEUROWHISPER_CACHE_DIR", None)
                os.environ.pop("NEUROWHISPER_LOG_DIR", None)
                os.environ.pop("NEUROWHISPER_SOCKET_PATH", None)

                paths.ensure_runtime_dirs()
                self.assertEqual(paths.get_app_support_dir(), Path(tmp))
                self.assertEqual(paths.get_models_dir(), Path(tmp) / "cache" / "models")
                self.assertEqual(paths.get_logs_dir(), Path(tmp) / "logs")
                self.assertIsNone(paths.get_sync_dir())
                self.assertEqual(paths.get_data_dir(), Path(tmp))

                sync_target = Path(tmp) / "iCloudSync"
                paths.set_sync_dir(sync_target)
                self.assertEqual(paths.get_sync_dir(), sync_target)
                self.assertEqual(paths.get_data_dir(), sync_target)
                self.assertEqual(
                    paths.get_transcriptions_dir(), sync_target / "transcriptions"
                )
                self.assertEqual(paths.get_config_file(), sync_target / "whisper_config.json")
                # Models and logs MUST stay local even when sync_dir is active
                self.assertEqual(paths.get_models_dir(), Path(tmp) / "cache" / "models")
                self.assertEqual(paths.get_logs_dir(), Path(tmp) / "logs")

                paths.set_sync_dir("")
                self.assertIsNone(paths.get_sync_dir())

    def test_unix_socket_104_byte_fallback(self) -> None:
        deep_home = "/tmp/" + ("very_long_macos_directory_segment_" * 4)
        with mock.patch.dict(os.environ, {"NEUROWHISPER_HOME": deep_home}, clear=False):
            os.environ.pop("NEUROWHISPER_SOCKET_PATH", None)
            sock = paths.get_socket_path()
            self.assertLessEqual(len(os.fsencode(str(sock))), paths.MAX_SAFE_UNIX_SOCKET_LEN)
            self.assertTrue(str(sock).startswith("/tmp/neurowhisper-"))
            mode = stat.S_IMODE(sock.parent.stat().st_mode)
            self.assertEqual(mode, 0o700)

    def test_frozen_meipass_resource_dir(self) -> None:
        with tempfile.TemporaryDirectory() as fake_meipass:
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
                sys, "_MEIPASS", fake_meipass, create=True
            ):
                self.assertEqual(paths.get_resource_dir(), Path(fake_meipass))


class TestConfigAndTypography(unittest.TestCase):
    def test_macos_backend_options_and_typography(self) -> None:
        self.assertEqual(
            config.BACKEND_OPTIONS,
            ["Auto", "Apple Metal (MLX)", "ONNX ASR (CoreML/CPU)", "CPU only"],
        )
        self.assertFalse(config.DEFAULT_CONFIG["worker_process"])
        self.assertEqual(config.resolve_backend_display("mlx-whisper", "mps"), "Apple Metal (MLX)")
        self.assertEqual(
            config.resolve_backend_display("onnx-asr", "coreml"),
            "ONNX ASR (CoreML/CPU)",
        )
        self.assertEqual(config.resolve_backend_display("faster-whisper", "cpu"), "CPU only")

        with mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(config.scaled_pt(10), 11)
            fam, pt = config.ui_font(10)
            self.assertEqual(pt, 11)
            self.assertIsInstance(fam, str)


class TestUnixSocketIPC(unittest.TestCase):
    def test_ipc_roundtrip_duplicate_guard_and_stale_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sock_path = Path(tmp) / "control.sock"
            # Plant a stale socket file to verify automatic cleanup
            sock_path.write_text("stale", encoding="utf-8")

            received = []

            def handler(cmd: str, _payload):
                received.append(cmd)
                return {"mode": "live" if cmd == "toggle_live" else "idle"}

            srv = ipc.IPCServer(handler=handler, socket_path=sock_path)
            srv.start()
            try:
                self.assertTrue(ipc.is_instance_running(sock_path))
                r1 = ipc.send_ipc_command("toggle_live", socket_path=sock_path)
                self.assertTrue(r1["ok"])
                self.assertEqual(r1["mode"], "live")
                self.assertIn("toggle_live", received)

                r_bad = ipc.send_ipc_command("nonexistent_cmd", socket_path=sock_path)
                self.assertFalse(r_bad["ok"])

                # Second server must detect running instance and refuse to clobber
                srv2 = ipc.IPCServer(handler=handler, socket_path=sock_path)
                with self.assertRaises(ipc.IPCAlreadyRunningError):
                    srv2.start(check_existing=True)
            finally:
                srv.stop()

            self.assertFalse(sock_path.exists())


class TestPlatformMac(unittest.TestCase):
    def test_hotkey_parsing_and_formatting(self) -> None:
        p1 = platform_mac.parse_hotkey_combo("Ctrl + Option + S")
        self.assertIsNotNone(p1)
        assert p1 is not None
        self.assertEqual(p1.canonical, "ctrl+alt+s")
        self.assertEqual(p1.keycode, 0x01)
        self.assertEqual(platform_mac.format_hotkey_label("ctrl+alt+s"), "Ctrl+Opt+S")

        p2 = platform_mac.parse_hotkey_combo("F9")
        self.assertIsNotNone(p2)
        assert p2 is not None
        self.assertEqual(p2.canonical, "f9")
        self.assertEqual(p2.keycode, 0x65)
        self.assertFalse(p2.check_fn_flag)

        p3 = platform_mac.parse_hotkey_combo("right_option")
        self.assertIsNotNone(p3)
        assert p3 is not None
        self.assertTrue(p3.is_standalone_modifier)

    def test_utf16_chunking_preserves_umlauts_and_surrogate_pairs(self) -> None:
        sample = "Größe äöüß 😀🎉🚀🔥✨💡🎯🏆🌟🎵!"
        chunks = platform_mac.chunk_utf16_safe(sample, max_units=16)
        self.assertGreater(len(chunks), 1)
        reconstructed = "".join(s for s, _ in chunks)
        self.assertEqual(reconstructed, sample)
        for chunk_str, unit_len in chunks:
            actual_units = len(chunk_str.encode("utf-16-le")) // 2
            self.assertEqual(unit_len, actual_units)
            self.assertLessEqual(unit_len, 16)

    def test_mocked_pyobjc_tcc_clipboard_paste_and_unicode_typing(self) -> None:
        fake_app_services = types.SimpleNamespace(
            kAXTrustedCheckOptionPrompt="AXTrustedCheckOptionPrompt",
            AXIsProcessTrustedWithOptions=mock.MagicMock(return_value=True),
        )
        fake_pb = mock.MagicMock()
        fake_pb.setString_forType_.return_value = True
        fake_appkit = types.SimpleNamespace(
            NSPasteboard=types.SimpleNamespace(generalPasteboard=lambda: fake_pb),
            NSPasteboardTypeString="public.utf8-plain-text",
        )
        posted_events = []
        unicode_calls = []

        def fake_create_kb(src, vk, down):
            return {"vk": vk, "down": down, "flags": 0}

        def fake_set_flags(ev, flags):
            ev["flags"] = flags

        def fake_set_unicode(ev, length, s):
            unicode_calls.append((ev["down"], length, s))

        def fake_post(tap, ev):
            posted_events.append((tap, dict(ev)))

        fake_quartz = types.SimpleNamespace(
            CGEventSourceCreate=lambda state: "src",
            CGEventCreateKeyboardEvent=fake_create_kb,
            CGEventSetFlags=fake_set_flags,
            CGEventKeyboardSetUnicodeString=fake_set_unicode,
            CGEventPost=fake_post,
            kCGEventFlagMaskCommand=platform_mac.kCGEventFlagMaskCommand,
            kCGHIDEventTap=0,
        )

        with mock.patch.object(platform_mac, "ApplicationServices", fake_app_services), \
             mock.patch.object(platform_mac, "AppKit", fake_appkit), \
             mock.patch.object(platform_mac, "Quartz", fake_quartz):
            self.assertTrue(platform_mac.is_accessibility_trusted(prompt=False))
            self.assertTrue(platform_mac.copy_to_clipboard("ÄÖÜ test"))
            fake_pb.clearContents.assert_called_once()
            fake_pb.setString_forType_.assert_called_once_with("ÄÖÜ test", "public.utf8-plain-text")

            self.assertTrue(platform_mac.paste_from_clipboard())
            self.assertEqual(len(posted_events), 2)
            self.assertEqual(posted_events[0][1]["vk"], 0x09)
            self.assertEqual(posted_events[0][1]["flags"], platform_mac.kCGEventFlagMaskCommand)

            posted_events.clear()
            self.assertTrue(platform_mac.type_text("Ümlaut 😀\nZwei"))
            down_strings = [s for is_down, _, s in unicode_calls if is_down]
            self.assertEqual("".join(down_strings), "Ümlaut 😀Zwei")
            # Verify Return keycode (0x24) was posted for the newline
            self.assertTrue(any(ev["vk"] == 0x24 for _, ev in posted_events))

    def test_cgeventtap_callback_ignores_f_key_secondary_fn_flag_and_reenables_timeout(self) -> None:
        mgr = platform_mac.MacHotkeyManager()
        triggered = []
        mgr._bindings["live"] = (
            platform_mac.parse_hotkey_combo("f9"),  # type: ignore[arg-type]
            lambda: triggered.append("live"),
        )

        reenabled = []
        fake_quartz = types.SimpleNamespace(
            CGEventTapEnable=lambda tap, flag: reenabled.append((tap, flag)),
            CGEventGetIntegerValueField=lambda ev, field: ev.get(field, 0),
            CGEventGetFlags=lambda ev: ev.get("flags", 0),
        )
        mgr._tap = "tap_handle"

        with mock.patch.object(platform_mac, "Quartz", fake_quartz):
            # 1. Timeout event re-enables tap
            mgr._event_tap_callback(None, platform_mac.kCGEventTapDisabledByTimeout, {}, None)
            self.assertEqual(reenabled, [("tap_handle", True)])

            # 2. F9 keypress with macOS hardware SecondaryFn bit set still triggers!
            f9_event = {
                9: 0x65,  # kVK_F9
                platform_mac.kCGKeyboardEventAutorepeat: 0,
                "flags": platform_mac.kCGEventFlagMaskSecondaryFn,
            }
            mgr._event_tap_callback(None, platform_mac.kCGEventKeyDown, f9_event, None)
            self.assertEqual(triggered, ["live"])


class TestMenuBarAndBackends(unittest.TestCase):
    def test_menubar_icons_and_headless_guard(self) -> None:
        for state in ("idle", "live", "batch", "online", "transcribing", "ready", "error"):
            img = menubar.render_menubar_icon(state)
            self.assertEqual(img.size, (64, 64))
            self.assertEqual(img.mode, "RGBA")

        with mock.patch.dict(os.environ, {"NEUROWHISPER_NO_TRAY": "1"}):
            ctrl = menubar.MenuBarController()
            self.assertFalse(ctrl.start())
            self.assertFalse(ctrl.available)

    def test_wav_writer(self) -> None:
        tone = (0.25 * np.sin(np.linspace(0, 20, 1600))).astype(np.float32)
        buf = io.BytesIO()
        write_pcm16_wav(buf, tone, sample_rate=16000)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getnframes(), 1600)

    def test_coreml_provider_selection_and_auto_detection(self) -> None:
        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        )
        fake_asr = types.SimpleNamespace(load_model=mock.MagicMock())
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort, "onnx_asr": fake_asr}):
            providers, resolved_dev = OnnxAsrBackend.select_providers("auto")
            self.assertEqual(resolved_dev, "coreml")
            self.assertEqual(providers[0], "CoreMLExecutionProvider")

            be_name, dev, _ = detect_best_backend(
                "Parakeet TDT 0.6B v3 (Ultra-fast ONNX, 25 langs, ~650MB)"
            )
            self.assertEqual((be_name, dev), ("onnx-asr", "coreml"))

    def test_mlx_whisper_backend_on_simulated_apple_silicon(self) -> None:
        fake_mlx = types.SimpleNamespace()
        fake_mlx_whisper = types.SimpleNamespace(
            transcribe=mock.MagicMock(
                return_value={
                    "text": "Hello from Apple Metal",
                    "segments": [{"text": " Hello from Apple Metal ", "start": 0.0, "end": 1.2}],
                }
            )
        )
        with mock.patch.object(sys, "platform", "darwin"), \
             mock.patch("platform.machine", return_value="arm64"), \
             mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx_whisper": fake_mlx_whisper}), \
             mock.patch.object(MlxWhisperBackend, "_ensure_snapshot", return_value="/tmp/fake-mlx-snap"):
            self.assertTrue(MlxWhisperBackend.is_available())
            be_name, dev, _ = detect_best_backend("Small (CPU-friendly, 2GB+ RAM)")
            self.assertEqual((be_name, dev), ("mlx-whisper", "mps"))

            be = MlxWhisperBackend()
            be.load_model("Small (CPU-friendly, 2GB+ RAM)", device="mps")
            self.assertTrue(be.is_model_loaded())
            self.assertEqual(be.current_device, "mps")
            segs = be.transcribe(np.zeros(8000, dtype=np.float32), sample_rate=16000)
            self.assertEqual(len(segs), 1)
            self.assertEqual(segs[0].text, "Hello from Apple Metal")
            be.unload_model()
            self.assertFalse(be.is_model_loaded())

    def test_openai_backend_fallback_and_prompt_guard(self) -> None:
        be = OpenAIBackend(api_key="sk-test", transcription_model="gpt-transcribe", edit_enabled=True)
        calls = []

        def fake_transcribe(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gpt-transcribe":
                raise RuntimeError("404 model gpt-transcribe not found")
            return "um hello world"

        fake_chat_resp = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Hello, world."))]
        )
        fake_client = types.SimpleNamespace(
            audio=types.SimpleNamespace(
                transcriptions=types.SimpleNamespace(create=fake_transcribe)
            ),
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=mock.MagicMock(return_value=fake_chat_resp))
            ),
        )
        be._client = fake_client
        segs = be.transcribe(np.ones(4000, dtype=np.float32) * 0.1)
        self.assertEqual(calls, ["gpt-transcribe", "gpt-4o-transcribe"])
        self.assertEqual(segs[0].text, "Hello, world.")
        chat_call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertIn("<transcript>", chat_call_kwargs["messages"][1]["content"])


class TestInsightsResourcesAndCLI(unittest.TestCase):
    def test_language_fillers_and_transcript_parser(self) -> None:
        de_text = "Ich habe halt irgendwie das Gefühl, ähm dass wir das gut machen."
        en_text = "I think like you know we should um review the architecture."
        self.assertEqual(insights.detect_entry_language(de_text), "de")
        self.assertEqual(insights.detect_entry_language(en_text), "en")
        self.assertGreaterEqual(sum(insights.count_fillers(de_text, "de").values()), 2)
        self.assertGreaterEqual(sum(insights.count_fillers(en_text, "en").values()), 2)

        with tempfile.TemporaryDirectory() as tmp:
            tfile = Path(tmp) / "2026-11-15.txt"
            tfile.write_text(
                f"[09:00:00] {en_text}\n[09:00:10] {de_text}\n",
                encoding="utf-8",
            )
            entries = insights.load_transcript_entries(tmp, max_days=None)
            self.assertEqual(len(entries), 2)
            snap = insights.compute_metric_snapshot(entries)
            self.assertIn("pace_avg_words_entry", snap)
            self.assertIn("pause_avg_seconds", snap)
            self.assertIn("lexical_diversity_percent", snap)

    def test_resources_plists_icns_and_packaging_files(self) -> None:
        res_dir = MACOS_DIR / "resources"
        pkg_dir = MACOS_DIR / "packaging"

        icns_bytes = (res_dir / "icon.icns").read_bytes()
        self.assertEqual(icns_bytes[:4], b"icns")
        self.assertGreater(len(icns_bytes), 1024)

        info = plistlib.loads((res_dir / "Info.plist").read_bytes())
        self.assertEqual(info["CFBundleIdentifier"], neurowhisper_mac.APP_BUNDLE_ID)
        self.assertIn("NSMicrophoneUsageDescription", info)

        ent = plistlib.loads((res_dir / "entitlements.plist").read_bytes())
        self.assertTrue(ent["com.apple.security.device.audio-input"])
        self.assertTrue(ent["com.apple.security.cs.allow-unsigned-executable-memory"])

        spec_src = (pkg_dir / "neurowhisper-macos.spec").read_text(encoding="utf-8")
        compile(spec_src, str(pkg_dir / "neurowhisper-macos.spec"), "exec")
        self.assertIn("COLLECT(", spec_src)
        self.assertIn("BUNDLE(", spec_src)

        for script_name in ("build_dmg.sh", "Unquarantine.command"):
            script_path = pkg_dir / script_name
            self.assertTrue(os.access(script_path, os.X_OK), f"{script_name} is not executable")

    def test_cli_version_psn_filtering_and_status_ipc(self) -> None:
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            rc = mac_main.main(["-psn_0_98765", "--version"])
        self.assertEqual(rc, 0)
        self.assertIn(neurowhisper_mac.__version__, buf.getvalue())

        with tempfile.TemporaryDirectory() as tmp:
            sock = Path(tmp) / "c.sock"
            srv = ipc.IPCServer(handler=lambda c, _p: {"mode": "idle"}, socket_path=sock)
            srv.start()
            try:
                with mock.patch.dict(os.environ, {"NEUROWHISPER_SOCKET_PATH": str(sock)}):
                    out = io.StringIO()
                    with mock.patch("sys.stdout", out):
                        rc2 = mac_main.main(["--status"])
                    self.assertEqual(rc2, 0)
                    parsed = json.loads(out.getvalue().strip())
                    self.assertTrue(parsed["ok"])
                    self.assertEqual(parsed["mode"], "idle")
            finally:
                srv.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
