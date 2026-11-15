"""Faster-Whisper backend for Linux (CPU INT8 + NVIDIA CUDA FP16 via isolated child worker)."""

from __future__ import annotations

import base64
import gc
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import paths
from ..config import AUTO_DEVICE_CONFIG, CUDA_DETECT_KEYWORDS, CUDA_ERROR_KEYWORDS, DEFAULT_CONFIG
from .base import TranscriptionResult, TranscriptionSegment, WhisperBackend
from .fw_worker import preload_cuda_libraries

log = logging.getLogger("neurowhisper.backends.faster_whisper")

_WORKER_SCRIPT_PATH = Path(__file__).resolve().with_name("fw_worker.py")


class _WorkerCrashed(RuntimeError):
    """Raised when the ``fw_worker`` child process exits unexpectedly."""


def _read_worker_process_setting() -> bool:
    """Read ``worker_process`` from XDG/sync ``whisper_config.json`` (default ``True``)."""
    default_val = bool(DEFAULT_CONFIG.get("worker_process", True))
    merged: dict = {}
    for cfg_file in (paths.get_shared_config_file(), paths.get_local_config_file()):
        if cfg_file.is_file():
            try:
                with open(cfg_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    merged.update(data)
            except Exception:
                continue
    return bool(merged.get("worker_process", default_val))


def build_worker_command_and_env() -> Tuple[List[str], Dict[str, str]]:
    """Build the frozen-safe subprocess command and ``LD_LIBRARY_PATH``-enriched env for ``fw_worker``."""
    env = os.environ.copy()
    cuda_dirs = paths.discover_cuda_lib_dirs()
    if cuda_dirs:
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        existing_parts = [p for p in existing_ld.split(os.pathsep) if p]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys([*cuda_dirs, *existing_parts]))

    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--fw-worker"]
    else:
        cmd = [sys.executable, str(_WORKER_SCRIPT_PATH)]
        src_root = str(Path(__file__).resolve().parents[2])
        existing_pp = env.get("PYTHONPATH", "")
        pp_parts = [p for p in existing_pp.split(os.pathsep) if p]
        if src_root not in pp_parts:
            env["PYTHONPATH"] = os.pathsep.join([src_root, *pp_parts])

    return cmd, env


class FasterWhisperBackend(WhisperBackend):
    """CTranslate2 ``faster-whisper`` backend supporting CPU INT8 and NVIDIA CUDA FP16."""

    def __init__(self, use_worker: Optional[bool] = None) -> None:
        super().__init__()
        self._cuda_tested: bool = False
        self._cuda_works: bool = False
        self._cuda_failed: bool = False

        if use_worker is None:
            use_worker = _read_worker_process_setting()
        self._use_worker: bool = bool(use_worker)
        self._proc: Optional[subprocess.Popen] = None
        self._io_lock = threading.Lock()
        self._stderr_tail: List[str] = []

    @property
    def backend_name(self) -> str:
        if self._device == "cuda":
            return "faster-whisper (CUDA)"
        return "faster-whisper (CPU)"

    @property
    def is_loaded(self) -> bool:
        if self._use_worker:
            return self._proc is not None and self._proc.poll() is None and self._model_name is not None
        return self._model is not None

    @property
    def cuda_failed(self) -> bool:
        return self._cuda_failed

    @classmethod
    def is_available(cls) -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def get_supported_devices(cls) -> List[str]:
        devices = ["cpu"]
        if cls.check_cuda():
            devices.insert(0, "cuda")
        return devices

    @classmethod
    def check_cuda(cls) -> bool:
        """Check whether CTranslate2 sees an NVIDIA GPU (after preloading any user CUDA 12 libs)."""
        try:
            preload_cuda_libraries()
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Child Worker Lifecycle & IPC
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._kill_worker()
        self._stderr_tail = []
        cmd, env = build_worker_command_and_env()

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        self._proc = proc

        def _drain_stderr(pipe) -> None:
            try:
                for line in pipe:
                    stripped = line.rstrip()
                    if stripped:
                        self._stderr_tail.append(stripped)
                        if len(self._stderr_tail) > 40:
                            self._stderr_tail = self._stderr_tail[-40:]
            except Exception:
                pass

        threading.Thread(
            target=_drain_stderr,
            args=(proc.stderr,),
            name="neurowhisper-fw-stderr",
            daemon=True,
        ).start()

        banner = proc.stdout.readline() if proc.stdout is not None else ""
        if not banner:
            tail = "\n".join(self._stderr_tail[-10:])
            self._kill_worker()
            raise _WorkerCrashed(f"fw_worker exited before ready banner.\n{tail}".strip())

        try:
            info = json.loads(banner)
        except json.JSONDecodeError as exc:
            self._kill_worker()
            raise _WorkerCrashed(f"fw_worker emitted invalid banner {banner!r}: {exc}") from exc

        if not info.get("ready"):
            self._kill_worker()
            raise _WorkerCrashed(f"fw_worker failed startup: {info}")

    def _rpc(self, payload: dict) -> dict:
        with self._io_lock:
            if self._proc is None or self._proc.poll() is not None:
                raise _WorkerCrashed("fw_worker is not running")
            try:
                assert self._proc.stdin is not None
                assert self._proc.stdout is not None
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                tail = "\n".join(self._stderr_tail[-10:])
                self._kill_worker()
                raise _WorkerCrashed(f"fw_worker pipe closed: {exc}\n{tail}".strip()) from exc

            if not line:
                rc = self._proc.poll()
                tail = "\n".join(self._stderr_tail[-10:])
                self._kill_worker()
                raise _WorkerCrashed(f"fw_worker exited (code {rc}).\n{tail}".strip())

            return json.loads(line)

    def _kill_worker(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write('{"cmd":"shutdown"}\n')
                    proc.stdin.flush()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Model Loading & Transcription
    # ------------------------------------------------------------------

    def _resolve_device_and_compute(self, device: str, compute_type: str) -> Tuple[str, str]:
        req_dev = (device or "auto").strip().lower()
        req_ct = (compute_type or "auto").strip().lower()

        if req_dev == "auto":
            if not self._cuda_failed and self.check_cuda():
                dev = AUTO_DEVICE_CONFIG["cuda_device"]
                ct = AUTO_DEVICE_CONFIG["cuda_compute_type"] if req_ct == "auto" else compute_type
                return dev, ct
            dev = AUTO_DEVICE_CONFIG["cpu_device"]
            ct = AUTO_DEVICE_CONFIG["cpu_compute_type"] if req_ct == "auto" else compute_type
            return dev, ct

        if req_dev == "cuda" and self._cuda_failed:
            return AUTO_DEVICE_CONFIG["cpu_device"], AUTO_DEVICE_CONFIG["cpu_compute_type"]

        if req_ct == "auto":
            ct = (
                AUTO_DEVICE_CONFIG["cuda_compute_type"]
                if req_dev == "cuda"
                else AUTO_DEVICE_CONFIG["cpu_compute_type"]
            )
            return req_dev, ct
        return req_dev, compute_type

    def load_model(self, model_name: str, device: str = "auto", compute_type: str = "auto") -> None:
        self.unload_model()
        target_dev, target_ct = self._resolve_device_and_compute(device, compute_type)
        models_root = paths.get_models_dir()
        models_root.mkdir(parents=True, exist_ok=True)

        if self._use_worker:
            self._load_in_worker(model_name, target_dev, target_ct, str(models_root))
        else:
            self._load_in_process(model_name, target_dev, target_ct, str(models_root))

    def _load_in_worker(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        download_root: str,
    ) -> None:
        self._start_worker()
        resp = self._rpc(
            {
                "cmd": "load",
                "model": model_name,
                "device": device,
                "compute_type": compute_type,
                "download_root": download_root,
            }
        )
        if not resp.get("ok"):
            err = str(resp.get("error", "unknown worker load error"))
            if device == "cuda":
                log.warning("CUDA load failed (%s); falling back to CPU INT8", err)
                self._cuda_failed = True
                cpu_dev = AUTO_DEVICE_CONFIG["cpu_device"]
                cpu_ct = AUTO_DEVICE_CONFIG["cpu_compute_type"]
                resp2 = self._rpc(
                    {
                        "cmd": "load",
                        "model": model_name,
                        "device": cpu_dev,
                        "compute_type": cpu_ct,
                        "download_root": download_root,
                    }
                )
                if not resp2.get("ok"):
                    self._kill_worker()
                    raise RuntimeError(str(resp2.get("error", "CPU fallback load failed")))
                device, compute_type = cpu_dev, cpu_ct
            else:
                self._kill_worker()
                raise RuntimeError(err)

        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        if device == "cuda":
            self._cuda_tested = False

    def _load_in_process(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        download_root: str,
    ) -> None:
        preload_cuda_libraries()
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            )
            self._model_name = model_name
            self._device = device
            self._compute_type = compute_type
            if device == "cuda":
                self._cuda_tested = False
        except Exception as exc:
            err_lower = str(exc).lower()
            if device == "cuda" or any(kw in err_lower for kw in CUDA_DETECT_KEYWORDS):
                log.warning("In-process CUDA load failed (%s); falling back to CPU INT8", exc)
                self._cuda_failed = True
                self._device = AUTO_DEVICE_CONFIG["cpu_device"]
                self._compute_type = AUTO_DEVICE_CONFIG["cpu_compute_type"]
                self._model = WhisperModel(
                    model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                    download_root=download_root,
                )
                self._model_name = model_name
            else:
                raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
        beam_size: int = 1,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        without_timestamps: bool = True,
    ) -> TranscriptionResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        audio_f32 = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        if self._use_worker:
            return self._transcribe_in_worker(
                audio_f32,
                sample_rate,
                language,
                beam_size,
                vad_filter,
                vad_parameters,
                without_timestamps,
            )
        return self._transcribe_in_process(
            audio_f32,
            sample_rate,
            language,
            beam_size,
            vad_filter,
            vad_parameters,
            without_timestamps,
        )

    def _transcribe_in_worker(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str],
        beam_size: int,
        vad_filter: bool,
        vad_parameters: Optional[dict],
        without_timestamps: bool,
    ) -> TranscriptionResult:
        req = {
            "cmd": "transcribe",
            "audio_b64": base64.b64encode(audio.tobytes()).decode("ascii"),
            "language": language,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "vad_parameters": vad_parameters,
            "without_timestamps": without_timestamps,
        }
        try:
            resp = self._rpc(req)
        except _WorkerCrashed as exc:
            model_name = self._model_name
            was_cuda = self._device == "cuda"
            if was_cuda:
                log.warning("fw_worker crashed on CUDA (%s); respawning on CPU INT8", exc)
                self._cuda_failed = True
                if model_name:
                    self.load_model(
                        model_name,
                        device=AUTO_DEVICE_CONFIG["cpu_device"],
                        compute_type=AUTO_DEVICE_CONFIG["cpu_compute_type"],
                    )
                    resp = self._rpc(req)
                else:
                    raise
            elif model_name:
                log.warning("fw_worker exited unexpectedly (%s); respawning worker", exc)
                self.load_model(model_name, device=self._device, compute_type=self._compute_type)
                resp = self._rpc(req)
            else:
                raise

        if not resp.get("ok"):
            err = str(resp.get("error", "worker transcription error"))
            err_lower = err.lower()
            if self._device == "cuda" and any(kw in err_lower for kw in CUDA_ERROR_KEYWORDS):
                log.warning("CUDA runtime error (%s); reloading worker on CPU INT8", err)
                self._cuda_failed = True
                model_name = self._model_name
                if model_name:
                    self.load_model(
                        model_name,
                        device=AUTO_DEVICE_CONFIG["cpu_device"],
                        compute_type=AUTO_DEVICE_CONFIG["cpu_compute_type"],
                    )
                    resp = self._rpc(req)
            if not resp.get("ok"):
                raise RuntimeError(str(resp.get("error", err)))

        if self._device == "cuda" and not self._cuda_tested:
            self._cuda_tested = True
            self._cuda_works = True

        segments = [
            TranscriptionSegment(start=s["start"], end=s["end"], text=s["text"])
            for s in resp.get("segments", [])
        ]
        return TranscriptionResult(
            text=resp.get("text", ""),
            segments=segments,
            language=resp.get("language") or language or "en",
            language_probability=float(resp.get("language_probability", 1.0)),
            duration=float(resp.get("duration", len(audio) / float(sample_rate))),
        )

    def _transcribe_in_process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str],
        beam_size: int,
        vad_filter: bool,
        vad_parameters: Optional[dict],
        without_timestamps: bool,
    ) -> TranscriptionResult:
        kwargs = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "without_timestamps": without_timestamps,
        }
        if language:
            kwargs["language"] = language
        if vad_parameters:
            kwargs["vad_parameters"] = vad_parameters

        try:
            segments_gen, info = self._model.transcribe(audio, **kwargs)
            segments: List[TranscriptionSegment] = []
            text_parts: List[str] = []
            for seg in segments_gen:
                segments.append(TranscriptionSegment(start=seg.start, end=seg.end, text=seg.text))
                text_parts.append(seg.text.strip())

            if self._device == "cuda" and not self._cuda_tested:
                self._cuda_tested = True
                self._cuda_works = True

            return TranscriptionResult(
                text=" ".join(text_parts),
                segments=segments,
                language=info.language if info else (language or "en"),
                language_probability=info.language_probability if info else 1.0,
                duration=info.duration if info else len(audio) / float(sample_rate),
            )
        except Exception as exc:
            err_lower = str(exc).lower()
            if self._device == "cuda" and any(kw in err_lower for kw in CUDA_ERROR_KEYWORDS):
                self._cuda_failed = True
                model_name = self._model_name
                if model_name:
                    self.load_model(
                        model_name,
                        device=AUTO_DEVICE_CONFIG["cpu_device"],
                        compute_type=AUTO_DEVICE_CONFIG["cpu_compute_type"],
                    )
                    return self._transcribe_in_process(
                        audio,
                        sample_rate,
                        language,
                        beam_size,
                        vad_filter,
                        vad_parameters,
                        without_timestamps,
                    )
            raise

    def unload_model(self) -> None:
        if self._proc is not None:
            self._kill_worker()
            self._model_name = None
            return

        was_cuda = self._device == "cuda"
        super().unload_model()
        gc.collect()
        if was_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
