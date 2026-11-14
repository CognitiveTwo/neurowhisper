#!/usr/bin/env python3
"""Comprehensive unit and integration test suite for neurowhisper.

Covers:
  * app_config model maps, legacy aliases, and backend/device UI round-trips
  * base stdlib WAV writer (clipping, file path + BytesIO, RIFF PCM16 validity)
  * backend registry thread-safety, caching, Intel GPU vs NPU detection, unload_all
  * FasterWhisperBackend local directory resolution (flat + nested HF snapshots),
    ctranslate2 false-positive CUDA guard, and real fw_worker.py subprocess IPC
    (stdout noise isolation, UTF-8 text, zero-disk transcribe_audio, clean exit)
  * OnnxAsrBackend model mapping, quantization/provider selection, >20s chunking,
    language forwarding + fallback, offline error translation, and unload_model
  * OpenAIBackend zero-disk BytesIO WAV upload, 404 gpt-transcribe fallback,
    <transcript> prompt wrapping, and finish_reason=='length' truncation guard
  * PynputKeyboardWrapper modifier cleanup, macOS control-char normalization,
    and HotkeyManager alias normalization, prefix-shadowing suppression,
    stale modifier timeout, and paused capture mode
  * WhisperApp headless/Xvfb integration: Live FIFO queue & stop-drain with zero
    dropped trailing words, Batch stop-drain & auto-paste error resilience,
    multi-paragraph transcript save/reload, same-day WPM duration preservation,
    midnight rollover, and State Card transitions
  * analysis/ toolkit end-to-end: common.py language heuristic & multi-paragraph
    archive parsing, label_days concrete topic selection & per-day API fault
    tolerance, cluster_topics --no-verbose & collision deduplication,
    build_series weekday days & word-weighted German share, calendar_features
    mixed DATE/DATETIME + multi-EXDATE + TRANSPARENT + coverage scoping, and
    report.py path-prefix traversal rejection & SVG bounds
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from importlib.machinery import SourceFileLoader
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from unittest import mock

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "analysis")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if ANALYSIS_DIR not in sys.path:
    sys.path.append(ANALYSIS_DIR)

_SAVED_ENV: dict[str, str | None] = {}


def setUpModule():
    for key, val in (
        ("NEUROWHISPER_NO_DIALOGS", "1"),
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
    ):
        _SAVED_ENV[key] = os.environ.get(key)
        os.environ[key] = val


def tearDownModule():
    for key, prev in _SAVED_ENV.items():
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


# --------------------------------------------------------------------------- #
# 1. Config, WAV writer, and Backend Registry
# --------------------------------------------------------------------------- #

class TestAppConfigAndRegistry(unittest.TestCase):
    def test_modern_models_and_legacy_aliases(self):
        import app_config

        model_ids = set(app_config.MODEL_MAP.values())
        self.assertIn("large-v3-turbo", model_ids)
        self.assertIn("parakeet-tdt-0.6b-v3", model_ids)
        self.assertIn("distil-large-v3.5", model_ids)
        for mid in model_ids:
            self.assertIn(mid, app_config.MODEL_DOWNLOAD_SIZES)

        for legacy_label, target_id in app_config.LEGACY_MODEL_ALIASES.items():
            self.assertTrue(target_id)
            self.assertNotEqual(legacy_label, "")

    def test_backend_options_round_trip(self):
        import app_config

        for option in app_config.BACKEND_OPTIONS:
            self.assertIn(option, app_config.BACKEND_DISPLAY_TO_BACKEND_DEVICE)
            backend, device = app_config.BACKEND_DISPLAY_TO_BACKEND_DEVICE[option]
            resolved = app_config.resolve_backend_display(backend, device)
            self.assertEqual(
                resolved,
                option,
                f"Option {option!r} -> ({backend!r}, {device!r}) resolved to {resolved!r}",
            )
        # Legacy config compatibility
        self.assertEqual(app_config.resolve_backend_display("cuda", "cuda"), "CUDA (NVIDIA GPU)")
        self.assertEqual(app_config.resolve_backend_display("faster-whisper", "auto"), "Auto")
        self.assertEqual(app_config.resolve_backend_display("faster-whisper", "cpu"), "CPU only")

    def test_write_pcm16_wav_file_and_bytesio_with_clipping(self):
        from backends.base import write_pcm16_wav

        samples = np.array([-2.5, -1.0, 0.0, 0.5, 1.0, 3.2], dtype=np.float32)
        buf = io.BytesIO()
        write_pcm16_wav(buf, samples, sample_rate=16000)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getnframes(), len(samples))
            frames = struct.unpack(f"<{len(samples)}h", wf.readframes(len(samples)))

        self.assertEqual(frames[0], -32767)
        self.assertEqual(frames[1], -32767)
        self.assertEqual(frames[2], 0)
        self.assertAlmostEqual(frames[3], 16383, delta=2)
        self.assertEqual(frames[4], 32767)
        self.assertEqual(frames[5], 32767)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "clip.wav")
            write_pcm16_wav(wav_path, samples.reshape(-1, 1), sample_rate=16000)
            with wave.open(wav_path, "rb") as wf:
                self.assertEqual(wf.getnframes(), len(samples))

    def test_backend_registry_reuse_unload_and_intel_device_detection(self):
        import backends

        avail_names = [name for name, _, _ in backends.get_available_backends()]
        self.assertIn("onnx-asr", avail_names)
        b1 = backends.create_backend("onnx-asr")
        b2 = backends.create_backend("onnx-asr")
        self.assertIs(b1, b2)

        # Fake an active loaded backend to verify unload_all_backends(exclude=...)
        sentinel_unloaded = []

        class _DummyBackend:
            is_model_loaded = True

            def unload_model(self):
                sentinel_unloaded.append("unloaded")
                self.is_model_loaded = False

        with backends._registry_lock:
            saved = dict(backends._backend_instances)
            try:
                backends._backend_instances["dummy_a"] = _DummyBackend()
                backends._backend_instances["dummy_b"] = _DummyBackend()
                backends.unload_all_backends(exclude="dummy_a")
                self.assertEqual(len(sentinel_unloaded), 1)
                self.assertIn("dummy_a", backends._backend_instances)
                self.assertNotIn("dummy_b", backends._backend_instances)
            finally:
                backends._backend_instances.clear()
                backends._backend_instances.update(saved)

        # Intel OpenVINO reason string parsing: "Intel GPU: ..." -> "gpu", "Intel NPU: ..." -> "npu"
        class _FakeFW:
            is_available = (True, "faster-whisper available")

            def check_cuda(self):
                return False, "No CUDA"

        class _FakeOVGpu:
            is_available = (True, "OpenVINO available (Intel GPU: Arc)")

        class _FakeOVNpu:
            is_available = (True, "OpenVINO available (Intel NPU)")

        def _factory_for(ov_cls):
            def _create(name):
                if name == "faster-whisper":
                    return _FakeFW()
                if name == "openvino":
                    return ov_cls()
                raise ValueError(name)
            return _create

        with mock.patch.dict(
            backends._backends, {"faster-whisper": object, "openvino": object}
        ):
            with mock.patch.object(backends, "create_backend", side_effect=_factory_for(_FakeOVGpu)):
                b_name, dev, _ = backends.detect_best_backend()
                self.assertEqual((b_name, dev), ("openvino", "gpu"))

            with mock.patch.object(backends, "create_backend", side_effect=_factory_for(_FakeOVNpu)):
                b_name, dev, _ = backends.detect_best_backend()
                self.assertEqual((b_name, dev), ("openvino", "npu"))


# --------------------------------------------------------------------------- #
# 2. FasterWhisperBackend & Real fw_worker.py Subprocess IPC
# --------------------------------------------------------------------------- #

class TestFasterWhisperBackendAndWorker(unittest.TestCase):
    def test_resolve_local_whisper_dir_flat_and_nested_hf_snapshot(self):
        from backends.faster_whisper_backend import _resolve_local_whisper_dir

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_resolve_local_whisper_dir(tmp))

            # Hidden .cache/.locks model.bin should be ignored
            hidden_dir = os.path.join(tmp, ".cache", "huggingface")
            os.makedirs(hidden_dir)
            with open(os.path.join(hidden_dir, "model.bin"), "wb") as fh:
                fh.write(b"ignored")
            self.assertIsNone(_resolve_local_whisper_dir(tmp))

            # Nested HF snapshot layout
            snap = os.path.join(
                tmp,
                "models--Systran--faster-whisper-large-v3-turbo",
                "snapshots",
                "deadbeef1234",
            )
            os.makedirs(snap)
            with open(os.path.join(snap, "model.bin"), "wb") as fh:
                fh.write(b"weights")
            self.assertEqual(_resolve_local_whisper_dir(tmp), snap)

            # Direct model.bin at root takes precedence
            with open(os.path.join(tmp, "model.bin"), "wb") as fh:
                fh.write(b"direct")
            self.assertEqual(_resolve_local_whisper_dir(tmp), tmp)

    def test_ctranslate2_false_positive_cuda_guard(self):
        from backends.faster_whisper_backend import FasterWhisperBackend

        fake_ct2 = types.SimpleNamespace(
            StorageView=object,
            get_cuda_device_count=lambda: 0,
            get_supported_compute_types=lambda dev: {"float16", "int8"} if dev == "cuda" else {"int8"},
        )
        saved_memo = FasterWhisperBackend._cuda_check_result
        FasterWhisperBackend._cuda_check_result = None
        try:
            with mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}):
                fb = FasterWhisperBackend()
                cuda_ok, reason = fb.check_cuda()
                self.assertFalse(cuda_ok)
                self.assertIn("No CUDA GPU", reason)
                self.assertEqual(fb.get_supported_devices(), ["cpu"])
        finally:
            FasterWhisperBackend._cuda_check_result = saved_memo

    def test_real_fw_worker_subprocess_ipc_and_stdout_isolation(self):
        """Spawn the real fw_worker.py process with a stub faster_whisper package on
        PYTHONPATH that writes noise to stdout and returns UTF-8 segments."""
        from backends.faster_whisper_backend import FasterWhisperBackend

        with tempfile.TemporaryDirectory() as tmp:
            stub_pkg = os.path.join(tmp, "stub_site")
            os.makedirs(stub_pkg)
            with open(os.path.join(stub_pkg, "faster_whisper.py"), "w", encoding="utf-8") as fh:
                fh.write(
                    "class _Seg:\n"
                    "    def __init__(self, text, start=0.0, end=1.0):\n"
                    "        self.text = text\n"
                    "        self.start = start\n"
                    "        self.end = end\n"
                    "class _Info:\n"
                    "    language = 'de'\n"
                    "class WhisperModel:\n"
                    "    def __init__(self, model_ref, device='cpu', compute_type='int8', **kw):\n"
                    "        print('NOISY_IMPORT_BAR [####] 100%', flush=True)\n"
                    "        self.model_ref = model_ref\n"
                    "    def transcribe(self, audio, **kwargs):\n"
                    "        print('NOISY_TRANSCRIBE_STDOUT', flush=True)\n"
                    "        lang = kwargs.get('language') or 'auto'\n"
                    "        kind = 'ndarray' if hasattr(audio, 'shape') else 'file'\n"
                    "        return [_Seg(f'Schöne Grüße ({kind},{lang})')], _Info()\n"
                )

            model_dir = os.path.join(
                tmp, "models", "Small", "models--Systran--faster-whisper-small", "snapshots", "abc123"
            )
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "model.bin"), "wb") as fh:
                fh.write(b"fake_ct2_bin")

            old_pp = os.environ.get("PYTHONPATH", "")
            old_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ["PYTHONPATH"] = stub_pkg + (os.pathsep + old_pp if old_pp else "")
            os.environ["HF_HUB_OFFLINE"] = "1"
            backend = FasterWhisperBackend()
            try:
                backend._load_model_via_worker(
                    model_id="small",
                    device="cpu",
                    compute_type="int8",
                    model_path=os.path.join(tmp, "models", "Small"),
                )
                self.assertTrue(backend.is_model_loaded)
                self.assertIsNotNone(backend._proc)

                # Zero-disk numpy array IPC (exercises PCM WAV BytesIO/temp handoff + worker decode)
                pcm = np.zeros(16000, dtype=np.float32)
                segs_mem = backend.transcribe_audio(pcm, sample_rate=16000, language="de")
                self.assertEqual(len(segs_mem), 1)
                self.assertIn("Schöne Grüße", segs_mem[0].text)
                self.assertIn(",de)", segs_mem[0].text)

                # File-path IPC
                wav_path = os.path.join(tmp, "sample.wav")
                from backends.base import write_pcm16_wav
                write_pcm16_wav(wav_path, pcm, sample_rate=16000)
                segs_file = backend.transcribe(wav_path, language="en")
                self.assertEqual(len(segs_file), 1)
                self.assertEqual(segs_file[0].text, "Schöne Grüße (file,en)")

                worker_proc = backend._proc
                backend.unload_model()
                self.assertFalse(backend.is_model_loaded)
                worker_proc.wait(timeout=3.0)
                self.assertIsNotNone(worker_proc.returncode)
            finally:
                backend.unload_model()
                if old_pp:
                    os.environ["PYTHONPATH"] = old_pp
                else:
                    os.environ.pop("PYTHONPATH", None)
                if old_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_offline


# --------------------------------------------------------------------------- #
# 3. OnnxAsrBackend
# --------------------------------------------------------------------------- #

class TestOnnxAsrBackend(unittest.TestCase):
    def test_model_resolution_quantization_and_long_audio_chunking(self):
        import httpx
        from backends.onnx_asr_backend import OnnxAsrBackend, _sanitize_no_proxy_value

        # Verify NO_PROXY sanitization strips '[::1]' -> '::1' and httpx accepts it
        cleaned_proxy = _sanitize_no_proxy_value("localhost, 127.0.0.1, ::1, [::1]")
        self.assertEqual(cleaned_proxy, "localhost,127.0.0.1,::1")
        with mock.patch.dict(os.environ, {"NO_PROXY": cleaned_proxy, "no_proxy": cleaned_proxy}):
            hc = httpx.Client()
            hc.close()

        b = OnnxAsrBackend()
        # Both short IDs AND full GUI friendly display labels must resolve cleanly!
        self.assertEqual(
            b._resolve_asr_name_and_dir("parakeet-tdt-0.6b-v3", "/tmp/models/parakeet")[0],
            "nemo-parakeet-tdt-0.6b-v3",
        )
        self.assertEqual(
            b._resolve_asr_name_and_dir(
                "Parakeet TDT 0.6B v3 (Ultra-fast ONNX, 25 langs, ~650MB)",
                "/tmp/models/parakeet_ui",
            ),
            ("nemo-parakeet-tdt-0.6b-v3", "/tmp/models/parakeet_ui"),
        )
        self.assertEqual(
            b._resolve_asr_name_and_dir("Small (CPU-friendly, 2GB+ RAM)", "/tmp/models/small")[0],
            "nemo-parakeet-tdt-0.6b-v3",
        )
        self.assertEqual(b._resolve_asr_name_and_dir("canary-1b-v2", None)[0], "nemo-canary-1b-v2")
        self.assertEqual(
            b._resolve_asr_name_and_dir(
                "Base (Minimal resources, quick)",
                "/tmp/models/Base_Minimal_resources,_quick",
            ),
            ("whisper-base", "/tmp/models/onnx_whisper-base"),
        )

        calls = []

        class _FakeVadAdapter:
            def recognize(self, waveform, sample_rate=16000, **kwargs):
                # Lazy generator (matching onnx-asr 0.12 SegmentResultsAsrAdapter.recognize)
                def _gen():
                    if kwargs.get("language") == "unsupported_lang":
                        raise KeyError("unsupported_lang")
                    yield types.SimpleNamespace(text="vad_segment_ok", start=0.0, end=1.0)
                return _gen()

        class _FakeModel:
            def recognize(self, waveform, sample_rate=16000, **kwargs):
                if kwargs.get("language") == "unsupported_lang":
                    raise KeyError("unsupported_lang")
                calls.append((len(waveform), sample_rate, dict(kwargs)))
                return f"chunk_{len(calls)}"

            def with_vad(self, vad_model, **kwargs):
                return _FakeVadAdapter()

        b._asr_model = _FakeModel()
        b._vad_model = object()
        b._asr_name = "nemo-canary-1b-v2"
        b._model_loaded = True

        # Lazy VAD generator raising KeyError on unsupported_lang must retry cleanly,
        # and arbitrary sample_rate=22000 must resample to 16000!
        vad_segs = b.transcribe_audio(
            np.ones(22000, dtype=np.float32) * 0.1,
            sample_rate=22000,
            language="unsupported_lang",
        )
        self.assertEqual([s.text for s in vad_segs], ["vad_segment_ok"])

        # Fixed-window fallback when _vad_model is None
        b._vad_model = None
        long_audio = np.ones(42 * 16000, dtype=np.float32) * 0.1
        segs = b.transcribe_audio(long_audio, sample_rate=16000, language="unsupported_lang")
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0].text, "chunk_1")
        self.assertEqual(segs[1].text, "chunk_2")
        # Canary pnc=True must be preserved on KeyError fallback
        self.assertEqual(calls[0][2], {"pnc": True})

        b.unload_model()
        self.assertFalse(b.is_model_loaded)

    def test_offline_error_translation(self):
        from backends.onnx_asr_backend import OnnxAsrBackend

        b = OnnxAsrBackend()
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}):
            with mock.patch(
                "onnx_asr.loader.Manager.create_asr",
                side_effect=RuntimeError("offline mode is enabled"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    b.load_model("parakeet-tdt-0.6b-v3", device="cpu", compute_type="int8")
                self.assertIn("not downloaded yet", str(ctx.exception).lower())


# --------------------------------------------------------------------------- #
# 4. OpenAIBackend
# --------------------------------------------------------------------------- #

class TestOpenAIBackend(unittest.TestCase):
    def test_zero_disk_transcribe_404_fallback_and_edit_truncation_guard(self):
        from backends.openai_backend import OpenAIBackend

        uploaded_files = []
        attempted_models = []

        class _FakeTranscriptions:
            def create(self, **kwargs):
                attempted_models.append(kwargs["model"])
                uploaded_files.append(kwargs["file"])
                if kwargs["model"] == "gpt-transcribe":
                    err = Exception("Error code: 404 - model_not_found")
                    err.status_code = 404
                    raise err
                return "Hello world from memory."

        class _FakeCompletions:
            def __init__(self):
                self.finish_reason = "stop"
                self.reply = "Hello world, cleanly edited."
                self.last_messages = None

            def create(self, **kwargs):
                self.last_messages = kwargs["messages"]
                choice = types.SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=types.SimpleNamespace(content=self.reply),
                )
                return types.SimpleNamespace(choices=[choice])

        completions = _FakeCompletions()
        fake_client = types.SimpleNamespace(
            audio=types.SimpleNamespace(transcriptions=_FakeTranscriptions()),
            chat=types.SimpleNamespace(completions=completions),
            close=mock.MagicMock(),
        )

        backend = OpenAIBackend()
        backend.configure(api_key="sk-test-key", transcription_model="gpt-transcribe")
        backend._client = fake_client

        pcm = np.zeros(8000, dtype=np.float32)
        segs = backend.transcribe_audio(pcm, sample_rate=16000)
        self.assertEqual(attempted_models, ["gpt-transcribe", "gpt-4o-transcribe"])
        self.assertEqual(backend._transcription_model, "gpt-4o-transcribe")
        # Uploaded file is an in-memory ("speech.wav", BytesIO, "audio/wav") tuple
        fname, fbuf, fmime = uploaded_files[0]
        self.assertEqual(fname, "speech.wav")
        self.assertIsInstance(fbuf, io.BytesIO)
        self.assertEqual(fmime, "audio/wav")
        self.assertEqual(segs[0].text, "Hello world from memory.")

        # Prompt-injection wrapping check
        edited = backend._edit_with_gpt("um hello world")
        self.assertEqual(edited, "Hello world, cleanly edited.")
        user_msg = completions.last_messages[-1]["content"]
        self.assertIn("<transcript>", user_msg)
        self.assertIn("</transcript>", user_msg)

        # Truncation guard: finish_reason == 'length' must return raw original text!
        completions.finish_reason = "length"
        completions.reply = "Truncated half..."
        original_long = "Full critical dictation paragraph one and paragraph two."
        self.assertEqual(backend._edit_with_gpt(original_long), original_long)

        backend.unload_model()
        fake_client.close.assert_called_once()


# --------------------------------------------------------------------------- #
# 5. Platform Utils & HotkeyManager
# --------------------------------------------------------------------------- #

class TestPlatformUtilsAndHotkeys(unittest.TestCase):
    def test_pynput_wrapper_modifier_release_on_failure_and_ctrl_char_norm(self):
        from platform_utils import _build_pynput_wrapper

        events = []

        class _FailOnFinalPressController:
            def press(self, k):
                events.append(("press", k))
                if k != "KEY_CTRL":
                    raise RuntimeError("tap failed")

            def release(self, k):
                events.append(("release", k))

            def type(self, txt):
                events.append(("type", txt))

        class _FakeKey:
            ctrl = "KEY_CTRL"
            alt = "KEY_ALT"
            cmd = "KEY_CMD"
            shift = "KEY_SHIFT"
            enter = "KEY_ENTER"
            tab = "KEY_TAB"
            space = "KEY_SPACE"
            backspace = "KEY_BS"
            delete = "KEY_DEL"
            esc = "KEY_ESC"

        class _FakeKeyCode:
            def __init__(self, char=None, vk=None):
                self.char = char
                self.vk = vk

            @staticmethod
            def from_char(ch):
                return f"CHAR_{ch}"

        fake_pynput_kb = types.SimpleNamespace(
            Controller=_FailOnFinalPressController,
            Key=_FakeKey,
            KeyCode=_FakeKeyCode,
            Listener=lambda **kw: types.SimpleNamespace(
                start=lambda: None, stop=lambda: None, is_alive=lambda: True, running=True
            ),
        )

        with mock.patch.dict(
            sys.modules,
            {"pynput": types.SimpleNamespace(keyboard=fake_pynput_kb), "pynput.keyboard": fake_pynput_kb},
        ):
            WrapperCls = _build_pynput_wrapper()
            wrapper = WrapperCls()
            # Control characters '\x13' (Ctrl+S), '\x0d' (Ctrl+M), '\x09' (Ctrl+I) normalize to letters
            self.assertEqual(wrapper._normalize_key(_FakeKeyCode(char="\x13")), "s")
            self.assertEqual(wrapper._normalize_key(_FakeKeyCode(char="\x0d")), "m")
            self.assertEqual(wrapper._normalize_key(_FakeKeyCode(char="\x09")), "i")

            # Pressing Shift+1 ('!' with vk=49), releasing Shift, then releasing '1' (vk=49)
            # must NOT leave '!' stuck in _current_keys and must emit '!' on key-up!
            seen_events = []
            h = wrapper.hook(lambda ev: seen_events.append((ev.event_type, ev.name)))
            wrapper._on_press(_FakeKeyCode(char="!", vk=49))
            self.assertTrue(wrapper.is_pressed("!"))
            wrapper._on_release(_FakeKeyCode(char="1", vk=49))
            self.assertFalse(wrapper.is_pressed("!"))
            self.assertEqual(seen_events, [("down", "!"), ("up", "!")])
            wrapper.unhook(h)

            # unhook_all() must stop an active listener thread
            wrapper.hook(lambda ev: None)
            self.assertIsNotNone(wrapper._listener)
            wrapper.unhook_all()
            self.assertIsNone(wrapper._listener)

            # Pressing 'ctrl+v' where 'v' raises must STILL release 'KEY_CTRL' in finally!
            with self.assertRaises(RuntimeError):
                wrapper.send("ctrl+v")
            self.assertIn(("press", "KEY_CTRL"), events)
            self.assertIn(("release", "KEY_CTRL"), events)

    def test_hotkey_manager_normalization_shadowing_timeout_and_pause(self):
        wg = _get_or_load_whisper_gui_module()

        captured_hook = {}

        class _StubKeyboard:
            def hook(self, fn, suppress=False):
                captured_hook["fn"] = fn
                return fn

            def unhook(self, ref):
                captured_hook.pop("fn", None)

        mgr = wg.HotkeyManager(_StubKeyboard(), lambda msg: None)
        mgr.DEBOUNCE_S = 0.0
        fired = []
        mgr.register_all({
            "short": ("ctrl+s", lambda: fired.append("short")),
            "long": ("ctrl+alt+s", lambda: fired.append("long")),
        })

        send_ev = lambda name, etype: captured_hook["fn"](
            types.SimpleNamespace(name=name, event_type=etype)
        )

        # 1. Alias normalization + prefix shadowing suppression:
        # Holding Left Ctrl + Right Alt + S must ONLY fire 'long' (ctrl+alt+s), NOT 'short' (ctrl+s)!
        send_ev("left ctrl", "down")
        send_ev("right alt", "down")
        send_ev("s", "down")
        self.assertEqual(fired, ["long"])
        send_ev("s", "up")
        send_ev("right alt", "up")
        send_ev("left ctrl", "up")

        # 2. Paused mode (during hotkey capture) suppresses firing
        fired.clear()
        mgr.paused = True
        send_ev("ctrl", "down")
        send_ev("s", "down")
        self.assertEqual(fired, [])
        send_ev("s", "up")
        send_ev("ctrl", "up")
        mgr.paused = False

        # 3. Stale modifier timeout (>8s gap clears orphaned 'alt' from Win+L lock screen)
        send_ev("alt", "down")
        mgr._last_hook_event -= 10.0  # simulate 10s elapsed while locked
        send_ev("ctrl", "down")
        send_ev("s", "down")
        # Because orphaned 'alt' was cleared, 'ctrl+s' ('short') fires instead of 'ctrl+alt+s'!
        self.assertEqual(fired, ["short"])
        mgr.unregister_all()


# --------------------------------------------------------------------------- #
# 6. WhisperApp Headless/Xvfb Integration Tests
# --------------------------------------------------------------------------- #

_CACHED_WG_MODULE = None
_CACHED_WG_BOOT_DIR = None


def _install_safe_hardware_stubs():
    if "keyboard" not in sys.modules or not getattr(sys.modules["keyboard"], "_nw_test_stub", False):
        kb = types.ModuleType("keyboard")
        kb._nw_test_stub = True
        kb.typed_texts = []
        kb.sent_keys = []
        kb.raise_on_send = False

        def _noop(*a, **k):
            return None

        for name in (
            "remove_hotkey", "press_and_release", "hook", "unhook",
            "on_release", "on_press", "unhook_all", "unhook_all_hotkeys",
            "wait", "release", "press", "read_event",
        ):
            setattr(kb, name, _noop)
        kb.add_hotkey = lambda *a, **k: object()
        kb.is_pressed = lambda *a, **k: False
        kb.write = lambda txt, *a, **k: kb.typed_texts.append(txt)

        def _send(shortcut, *a, **k):
            if kb.raise_on_send:
                raise RuntimeError("simulated paste hook error")
            kb.sent_keys.append(shortcut)

        kb.send = _send
        sys.modules["keyboard"] = kb

    try:
        import sounddevice as sd
    except Exception:
        sd = types.ModuleType("sounddevice")
        sd.default = types.SimpleNamespace(device=None, samplerate=None, channels=None, dtype=None)
        sd.PortAudioError = type("PortAudioError", (Exception,), {})
        sd.check_input_settings = lambda *a, **k: None
        sd.sleep = lambda *a, **k: None
        sd.stop = lambda *a, **k: None
        sys.modules["sounddevice"] = sd
    sd.query_devices = lambda *a, **k: [{"name": "Stub Mic", "max_input_channels": 1, "hostapi": 0}]
    sd.query_hostapis = lambda *a, **k: [{"name": "Stub API"}]

    class _DummyStream:
        def __init__(self, *a, **k):
            self.active = True

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            self.active = False

    sd.InputStream = _DummyStream


def _get_or_load_whisper_gui_module():
    global _CACHED_WG_MODULE, _CACHED_WG_BOOT_DIR
    if _CACHED_WG_MODULE is not None:
        return _CACHED_WG_MODULE
    import atexit

    os.environ["NEUROWHISPER_NO_DIALOGS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _install_safe_hardware_stubs()
    if "platform_utils" in sys.modules:
        sys.modules["platform_utils"].keyboard = sys.modules["keyboard"]

    boot_dir = tempfile.mkdtemp(prefix="nw_wg_boot_")
    _CACHED_WG_BOOT_DIR = boot_dir
    atexit.register(lambda: shutil.rmtree(boot_dir, ignore_errors=True))
    for fname in ("whisper_gui.pyw", "app_config.py", "platform_utils.py", "icon.ico", "icon.png"):
        src = os.path.join(REPO_ROOT, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(boot_dir, fname))
    shutil.copytree(
        os.path.join(REPO_ROOT, "backends"),
        os.path.join(boot_dir, "backends"),
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    orig_cwd = os.getcwd()
    try:
        path = os.path.join(boot_dir, "whisper_gui.pyw")
        loader = SourceFileLoader("whisper_gui_testmod", path)
        spec = importlib.util.spec_from_file_location("whisper_gui_testmod", path, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["whisper_gui_testmod"] = mod
        spec.loader.exec_module(mod)
    finally:
        os.chdir(orig_cwd)
    _CACHED_WG_MODULE = mod
    return mod


def _tk_available() -> bool:
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        r.destroy()
        return True
    except Exception:
        return False


class _SlowOrderedFakeBackend:
    def __init__(self, delay_s=0.08):
        self.delay_s = delay_s
        self.is_model_loaded = True
        self.calls = 0
        self._lock = threading.Lock()

    def transcribe_audio(self, audio, sample_rate=16000, **kwargs):
        time.sleep(self.delay_s)
        with self._lock:
            self.calls += 1
            idx = self.calls
        seg = types.SimpleNamespace(text=f"phrase{idx}", start=0.0, end=0.5)
        return [seg]

    def unload_model(self):
        self.is_model_loaded = False


@unittest.skipUnless(_tk_available(), "Tk display unavailable (run under xvfb-run on headless Linux)")
class TestWhisperAppIntegration(unittest.TestCase):
    def setUp(self):
        import tkinter as tk

        self.wg = _get_or_load_whisper_gui_module()
        _install_safe_hardware_stubs()
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="nw_gui_test_")
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir.name)
        self.orig_app_dir = self.wg.app_dir
        self.wg.app_dir = self.tmp_dir.name
        os.makedirs(os.path.join(self.tmp_dir.name, "transcriptions"), exist_ok=True)

        # Bind stub keyboard directly so Linux _AdaptiveLinuxKeyboardProxy does not intercept
        self.orig_kb = self.wg.keyboard
        self.stub_kb = sys.modules["keyboard"]
        self.wg.keyboard = self.stub_kb
        self.stub_kb.typed_texts = []
        self.stub_kb.sent_keys = []
        self.stub_kb.raise_on_send = False

        # Prevent background model download during unit tests; inject deterministic fake backend
        self._load_patch = mock.patch.object(self.wg.WhisperApp, "load_model", lambda self_app: None)
        self._sound_patch = mock.patch.object(self.wg.WhisperApp, "play_feedback_sound", lambda *a, **k: None)
        self._load_patch.start()
        self._sound_patch.start()

        self.tk = tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.tk_errors = []
        self.root.report_callback_exception = lambda exc, val, tb: self.tk_errors.append((exc, val))
        self.app = self.wg.WhisperApp(self.root)
        self.fake_backend = _SlowOrderedFakeBackend(delay_s=0.08)
        self.app.backend = self.fake_backend
        self.app.model = self.fake_backend
        self.app.current_backend_name = "faster-whisper"
        self.app.current_device = "cpu"
        self.app._model_loading = False
        self.app._model_load_failed = False

    def tearDown(self):
        try:
            self.app.on_close()
        except Exception:
            pass
        self._sound_patch.stop()
        self._load_patch.stop()
        self.wg.keyboard = self.orig_kb
        self.wg.app_dir = self.orig_app_dir
        os.chdir(self.orig_cwd)
        self.tmp_dir.cleanup()
        self.assertEqual(self.tk_errors, [])

    def _pump_tk(self, duration_s=0.15):
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)

    def test_live_mode_fifo_queue_and_stop_drain_drops_zero_words(self):
        """Stopping Live mode while chunk 1 is mid-inference and chunk 2 is sitting
        in audio_queue must transcribe and type BOTH chunks in FIFO order."""
        orig_stop_delay = self.wg.AUDIO_RUNTIME["stop_delay_seconds"]
        self.wg.AUDIO_RUNTIME["stop_delay_seconds"] = 0.05
        try:
            self.app.toggle_live_mode()
            self.assertEqual(self.app.mode, "live")

            # Enqueue chunk 1 directly into _live_chunk_queue (starts 80ms inference)
            loud_chunk = np.ones((1600, 1), dtype=np.float32) * 0.2
            self.app._live_chunk_queue.put([loud_chunk] * 6)
            time.sleep(0.02)  # chunk 1 is now mid-inference inside _live_transcribe_loop

            # Put trailing speech directly into audio_queue & immediately stop live mode!
            for _ in range(6):
                self.app.audio_queue.put(loud_chunk)
            self.app.toggle_live_mode()  # triggers _delayed_stop_live

            deadline = time.monotonic() + 3.0
            while (self.app.stopping or self.app.mode is not None or self.fake_backend.calls < 2) and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.02)
            self._pump_tk(0.2)

            # Both chunk 1 and the trailing stop-drained chunk 2 were typed in exact order!
            self.assertGreaterEqual(len(self.stub_kb.typed_texts), 2)
            self.assertEqual(self.stub_kb.typed_texts[:2], ["phrase1 ", "phrase2 "])
        finally:
            self.wg.AUDIO_RUNTIME["stop_delay_seconds"] = orig_stop_delay

    def test_batch_stop_drain_and_paste_exception_resilience(self):
        """Even if keyboard.send('ctrl+v') raises during auto-paste, Batch mode
        still saves the transcript, copies to clipboard, and resets cleanly."""
        orig_stop_delay = self.wg.AUDIO_RUNTIME["stop_delay_seconds"]
        self.wg.AUDIO_RUNTIME["stop_delay_seconds"] = 0.05
        self.stub_kb.raise_on_send = True
        try:
            self.app.toggle_batch_mode()
            self.assertEqual(self.app.mode, "batch")

            loud_chunk = np.ones((1600, 1), dtype=np.float32) * 0.2
            for _ in range(8):
                self.app.audio_queue.put(loud_chunk)

            self.app.toggle_batch_mode()  # stop batch -> drains audio_queue & runs finalize_batch
            deadline = time.monotonic() + 3.0
            while (
                self.app.stopping
                or self.app.mode is not None
                or self.app.result_queue.empty()
            ) and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.02)
            self.app._update_gui_loop_body()
            self._pump_tk(0.15)

            self.assertIsNone(self.app.mode)
            self.assertFalse(self.app.stopping)
            self.assertTrue(any("phrase1" in entry[1] for entry in self.app.history))
        finally:
            self.wg.AUDIO_RUNTIME["stop_delay_seconds"] = orig_stop_delay
            self.stub_kb.raise_on_send = False

    def test_multiparagraph_reload_wpm_duration_preservation_midnight_and_state_card(self):
        # 1. Save a multi-paragraph transcription & real speaking duration
        multi_text = "First spoken paragraph today.\n\nSecond spoken paragraph with details."
        self.app.today_words = 9
        self.app.total_words = 9
        self.app.today_audio_duration = 18.5
        self.app.total_audio_duration = 18.5
        self.app.save_transcription(multi_text, latency_ms=120.0)
        self.app.flush_stats()

        # Reload stats & transcriptions as a same-day restart would
        self.app.load_stats()
        self.app.load_recent_transcriptions()
        self.assertEqual(len(self.app.history), 1)
        self.assertEqual(self.app.history[-1][1], multi_text)
        # Real recorded audio duration (18.5s) must NOT be overwritten by synthetic word estimate!
        self.assertAlmostEqual(self.app.today_audio_duration, 18.5, places=2)

        # 2. State card transitions
        self.app._model_loading = False
        self.app.model = None
        self.app._model_load_failed = True
        self.app._update_state_card()
        self.assertEqual(self.app.state_word_var.get(), "Model error")

        self.app._model_load_failed = False
        self.app._model_idle_unloaded = True
        self.app._update_state_card()
        self.assertEqual(self.app.state_word_var.get(), "Idle (GPU freed)")

        self.app._model_idle_unloaded = False
        self.app.model = self.fake_backend
        self.app._update_state_card()
        self.assertEqual(self.app.state_word_var.get(), "Ready")

        # 3. Midnight rollover resets today's counters cleanly
        self.app.session_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()
        self.app._update_gui_loop_body()
        self.assertEqual(self.app.session_date, dt.date.today().isoformat())
        self.assertEqual(self.app.today_words, 0)
        self.assertEqual(self.app.today_audio_duration, 0.0)

        # 4. Legacy config migration: LEGACY_MODEL_ALIASES & 'backend': 'cuda'
        with open(self.wg.CONFIG_FILE, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_key": "Parakeet TDT 0.6B v2 (English, 2GB+ RAM)",
                    "welcome_shown": True,
                },
                fh,
            )
        local_cfg = self.app._local_config_file()
        with open(local_cfg, "w", encoding="utf-8") as fh:
            json.dump({"backend": "cuda"}, fh)
        migrated = self.app.load_config()
        self.assertEqual(
            migrated["model_key"],
            "Parakeet TDT 0.6B v3 (Ultra-fast ONNX, 25 langs, ~650MB)",
        )
        self.assertEqual(migrated["backend"], "faster-whisper")
        self.assertEqual(migrated["device"], "cuda")


# --------------------------------------------------------------------------- #
# 7. End-to-End Analysis Toolkit Tests
# --------------------------------------------------------------------------- #

class TestAnalysisToolkitPipeline(unittest.TestCase):
    def test_common_language_heuristic_multiparagraph_and_atomic_write(self):
        import common

        # Single loanword/city umlaut in a long English sentence stays English
        en_text = (
            "We scheduled the architecture review meeting with our Zürich engineering "
            "team for Thursday morning after reviewing the deployment checklist."
        )
        self.assertEqual(common.detect_language(en_text), "en")
        self.assertEqual(common.detect_language("Schön!"), "de")
        self.assertEqual(common.detect_language("Das ist ein sehr guter Plan für morgen."), "de")

        with tempfile.TemporaryDirectory() as tmp:
            t_path = os.path.join(tmp, "2026-04.txt")
            with open(t_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "[2026-04-01 09:00:00] Filtered early entry line 1\n"
                    "Filtered continuation line 2 that must not leak\n"
                    "[2026-04-02 10:15:00] Kept paragraph one.\n\n"
                    "Kept paragraph two on next line.\n"
                    "[2026-04-03 11:30:00] Final day entry.\n"
                )

            entries = common.load_entries(tmp, since="2026-04-02", until="2026-04-02")
            self.assertEqual(len(entries), 1)
            self.assertEqual(
                entries[0]["text"],
                "Kept paragraph one.\n\nKept paragraph two on next line.",
            )

            out_json = os.path.join(tmp, "nested", "atomic.json")
            common.write_json(out_json, {"ok": True, "umlaut": "ü"})
            self.assertEqual(common.read_json(out_json), {"ok": True, "umlaut": "ü"})
            leftovers = [f for f in os.listdir(os.path.dirname(out_json)) if f.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_full_analysis_pipeline_series_calendar_clusters_and_report_security(self):
        import common
        import label_days
        import cluster_topics
        import build_series
        import calendar_features
        import report

        with tempfile.TemporaryDirectory() as tmp:
            tx_dir = os.path.join(tmp, "transcriptions")
            out_dir = os.path.join(tmp, "out")
            labels_dir = os.path.join(out_dir, "labels")
            os.makedirs(tx_dir)
            os.makedirs(labels_dir)

            with mock.patch.object(common, "OUT_DIR", out_dir), \
                 mock.patch.object(common, "LABELS_DIR", labels_dir), \
                 mock.patch.object(common, "TRANSCRIPTIONS_DIR", tx_dir):

                # 1. Write transcripts across Mon (with label) and Tue (without label, German + English)
                with open(os.path.join(tx_dir, "2026-04.txt"), "w", encoding="utf-8") as fh:
                    fh.write(
                        "[2026-04-06 09:00:00] Working on the database migration script and API latency benchmark.\n"
                        "[2026-04-07 10:00:00] Das ist ein kurzer deutscher Satz heute Morgen im Büro.\n"
                        "[2026-04-07 14:00:00] Short English note.\n"
                    )

                # 2. Bootstrap label via from_app_insights: verify concrete projects+topics beat abstract themes
                insights_file = os.path.join(tmp, "ai_insights.json")
                common.write_json(insights_file, {
                    "2026-04-06": {
                        "mood": "focused",
                        "mood_score": 78,
                        "daily_brief": (
                            "Worked through the database migration script, verified benchmark numbers, "
                            "and prepared the release checklist for staging."
                        ),
                        "projects": ["database migration script", "API latency benchmark"],
                        "topics": ["staging release checklist"],
                        "themes": ["productivity", "growth"],  # abstract themes must NOT pollute when >=3 concrete exist
                        "activities": ["coding"],
                    }
                })
                rc = label_days.main([
                    "--from-app-insights",
                    "--app-insights", insights_file,
                    "--transcripts", tx_dir,
                ])
                self.assertEqual(rc, 0)
                mon_label = common.read_json(os.path.join(labels_dir, "2026-04-06.json"))
                self.assertEqual(
                    mon_label["topics"],
                    ["database migration script", "API latency benchmark", "staging release checklist"],
                )

                # 3. Verify cluster_topics parser supports --no-verbose
                parsed = cluster_topics.build_parser().parse_args(["--no-verbose"])
                self.assertFalse(parsed.verbose)

                # 4. Run build_series and verify active weekday days vs mood_days & word-weighted German share
                self.assertEqual(build_series.main(["--transcripts", tx_dir, "--labels", labels_dir]), 0)
                series = common.read_json(os.path.join(out_dir, "series.json"))
                tue_row = next(w for w in series["weekdays"] if w["weekday"] == "Tue")
                self.assertEqual(tue_row["days"], 1)
                self.assertEqual(tue_row["mood_days"], 0)
                self.assertIsNone(tue_row["mood_mean"])
                # Tue had 10 German words + 3 English words -> ~10/13 = 0.7692 (not 0.5 entry-unweighted!)
                tue_daily = next(d for d in series["daily"] if d["date"] == "2026-04-07")
                self.assertAlmostEqual(tue_daily["german_share"], round(10 / 13, 4), places=3)

                # 5. Calendar features: mixed all-day DATE + timed DATETIME on same day,
                #    multi-value EXDATE, and TRANSP:TRANSPARENT exclusion
                ics_path = os.path.join(tmp, "cal.ics")
                with open(ics_path, "w", encoding="utf-8") as fh:
                    fh.write(
                        "BEGIN:VCALENDAR\r\n"
                        "VERSION:2.0\r\n"
                        "BEGIN:VEVENT\r\n"
                        "UID:allday-1\r\n"
                        "DTSTART;VALUE=DATE:20260406\r\n"
                        "DTEND;VALUE=DATE:20260407\r\n"
                        "SUMMARY:Company Holiday Banner\r\n"
                        "END:VEVENT\r\n"
                        "BEGIN:VEVENT\r\n"
                        "UID:transparent-1\r\n"
                        "DTSTART:20260406T080000\r\n"
                        "DTEND:20260406T120000\r\n"
                        "TRANSP:TRANSPARENT\r\n"
                        "SUMMARY:Free Reminder Block\r\n"
                        "END:VEVENT\r\n"
                        "BEGIN:VEVENT\r\n"
                        "UID:recurring-standup\r\n"
                        "DTSTART:20260406T100000\r\n"
                        "DTEND:20260406T110000\r\n"
                        "RRULE:FREQ=DAILY;COUNT=2\r\n"
                        "EXDATE:20260407T100000,20260408T100000\r\n"
                        "SUMMARY:Engineering Sync\r\n"
                        "BEGIN:VALARM\r\n"
                        "ACTION:EMAIL\r\n"
                        "SUMMARY:Alarm notification\r\n"
                        "END:VALARM\r\n"
                        "END:VEVENT\r\n"
                        "END:VCALENDAR\r\n"
                    )
                self.assertEqual(calendar_features.main(["--ics", ics_path]), 0)
                cal_data = common.read_json(os.path.join(out_dir, "calendar_features.json"))
                by_day = cal_data["daily"]
                # 2026-04-06 has ONLY the 1.0h timed Engineering Sync (all-day & TRANSPARENT excluded from hours)
                self.assertAlmostEqual(by_day["2026-04-06"]["meeting_hours"], 1.0, places=2)
                self.assertTrue(by_day["2026-04-06"]["all_day"])
                self.assertTrue(by_day["2026-04-06"]["off_or_holiday"])
                # Verify VALARM SUMMARY did not clobber VEVENT SUMMARY
                parsed_events = calendar_features.parse_ics(ics_path, horizon=dt.date(2026, 4, 10))
                summaries = [ev["summary"] for ev in parsed_events]
                self.assertIn("Engineering Sync", summaries)
                self.assertNotIn("Alarm notification", summaries)
                # 2026-04-07 was excluded via multi-value EXDATE -> 0.0h
                self.assertAlmostEqual(by_day.get("2026-04-07", {}).get("meeting_hours", 0.0), 0.0, places=2)

                # 6. Report security: sibling-prefix directory out_evil must be rejected!
                evil_out = out_dir + "_evil"
                os.makedirs(evil_out, exist_ok=True)
                with self.assertRaises(SystemExit):
                    report.main(["--out", os.path.join(evil_out, "pwn.html")])

                # Valid report renders cleanly with 'no data' tooltip on unlabelled Tuesday mood bar
                valid_html_path = os.path.join(out_dir, "report.html")
                self.assertEqual(report.main(["--out", valid_html_path, "--title", "Unit Test Report"]), 0)
                with open(valid_html_path, "r", encoding="utf-8") as fh:
                    html_doc = fh.read()
                self.assertIn("Unit Test Report", html_doc)
                self.assertIn("Tue: no data", html_doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
