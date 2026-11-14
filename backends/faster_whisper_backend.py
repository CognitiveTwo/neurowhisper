# Faster Whisper Backend
# Wraps the faster-whisper (CTranslate2) library for CPU and CUDA support

import json
import os
import platform
import queue
import subprocess
import sys
import threading
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np

from app_config import CUDA_DETECT_KEYWORDS_BACKEND
from .base import TranscriptionSegment, WhisperBackend

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CONNECTION_ERROR_MARKERS = (
    "connectionerror",
    "localentrynotfounderror",
    "offlinemodeisenabled",
    "max retries",
    "failed to resolve",
    "failed to establish a new connection",
    "temporary failure in name resolution",
    "name or service not known",
    "getaddrinfo",
    "network is unreachable",
    "offline mode is enabled",
    "outgoing traffic has been disabled",
    "couldn't connect",
)


def _sanitize_no_proxy_env() -> None:
    """Strip brackets around bare IPv6 entries (e.g. ``[::1]`` -> ``::1``)
    that crash ``httpx``'s ``get_environment_proxies()`` with ``InvalidURL``."""
    for var in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(var)
        if not raw or "[" not in raw:
            continue
        parts: List[str] = []
        seen = set()
        for token in raw.split(","):
            item = token.strip()
            if not item:
                continue
            if "://" not in item and item.startswith("[") and item.endswith("]"):
                item = item[1:-1]
            if item not in seen:
                seen.add(item)
                parts.append(item)
        updated = ",".join(parts)
        if updated != raw:
            os.environ[var] = updated


def _is_offline_mode() -> bool:
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _is_connection_error(exc: BaseException) -> bool:
    """Best-effort detection of a network/connection failure raised while
    constructing a WhisperModel (huggingface_hub download path)."""
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        resp = getattr(cur, "response", None)
        if resp is not None and getattr(resp, "status_code", None) is not None:
            return False
        if isinstance(cur, (ConnectionError, TimeoutError)):
            return True
        haystack = f"{type(cur).__name__} {cur}".lower()
        if any(marker in haystack for marker in _CONNECTION_ERROR_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _resolve_local_whisper_dir(model_path: Optional[str]) -> Optional[str]:
    """Return the exact directory containing ``model.bin``, or ``None``.

    Supports both flat CTranslate2 directories (``<model_path>/model.bin``) and
    HuggingFace cache layouts (``<model_path>/models--.../snapshots/<sha>/model.bin``),
    while skipping incomplete ``.cache`` / ``.locks`` folders.
    """
    if not model_path or not os.path.isdir(model_path):
        return None
    top_bin = os.path.join(model_path, "model.bin")
    if os.path.isfile(top_bin):
        return model_path

    candidates = []
    for root, dirs, files in os.walk(model_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if "model.bin" in files:
            bin_path = os.path.join(root, "model.bin")
            if os.path.isfile(bin_path):
                try:
                    mtime = os.path.getmtime(bin_path)
                except OSError:
                    mtime = 0.0
                candidates.append((mtime, root))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _model_present_locally(model_path: Optional[str]) -> bool:
    """Whether usable CTranslate2 ``model.bin`` weights already exist under ``model_path``."""
    return _resolve_local_whisper_dir(model_path) is not None


def _safe_progress(progress_callback: Optional[Callable], status: str) -> None:
    """Call progress_callback(status), swallowing any exception it raises."""
    if not progress_callback:
        return
    try:
        progress_callback(status)
    except Exception:
        pass


# --- Worker-process mode (CUDA only) ----------------------------------------
_WORKER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fw_worker.py")

_WORKER_LOAD_TIMEOUT_SECONDS = 180
_WORKER_LOAD_TIMEOUT_SECONDS_DOWNLOADING = 1800
_WORKER_TRANSCRIBE_TIMEOUT_SECONDS = 300
_WORKER_SHUTDOWN_GRACE_SECONDS = 3.0
_WORKER_STDERR_MAXLEN = 50


class _WorkerModelHandle:
    """Truthy marker stored as ``self._model`` while a worker-process model is loaded."""
    __slots__ = ()


_WORKER_MODEL_HANDLE = _WorkerModelHandle()


class FasterWhisperBackend(WhisperBackend):
    """
    Backend using faster-whisper (CTranslate2) for transcription.
    Supports CPU and NVIDIA CUDA devices, including modern models such as
    ``large-v3-turbo`` and ``distil-large-v3.5``.
    """

    # Process-wide memo for check_cuda(). None = not probed yet.
    _cuda_check_result: Optional[Tuple[bool, str]] = None
    _cuda_check_lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._current_model_key = None
        self._current_device = None

        # Worker-process mode state
        self._worker_mode = False
        self._proc = None
        self._response_queue = None
        self._reader_thread = None
        self._stderr_thread = None
        self._stderr_lines = deque(maxlen=_WORKER_STDERR_MAXLEN)
        self._stderr_lock = threading.Lock()
        self._worker_state_lock = threading.Lock()
        self._worker_request_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "faster-whisper"

    @property
    def is_available(self) -> Tuple[bool, str]:
        """faster-whisper is always available as a core dependency."""
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            return True, "faster-whisper available"
        except ImportError as e:
            return False, f"faster-whisper not installed: {e}"

    def check_cuda(self) -> Tuple[bool, str]:
        """Check if CUDA is actually available and usable for CTranslate2.

        Checks ``get_cuda_device_count() > 0`` BEFORE ``get_supported_compute_types("cuda")``:
        in CTranslate2 4.x, calling ``get_supported_compute_types("cuda")`` a second time
        after an initial driver error returns ``{'float32'}`` due to an internal
        ``std::once_flag``, which previously caused false-positive CUDA detection.
        """
        if FasterWhisperBackend._cuda_check_result is not None:
            return FasterWhisperBackend._cuda_check_result

        with FasterWhisperBackend._cuda_check_lock:
            if FasterWhisperBackend._cuda_check_result is not None:
                return FasterWhisperBackend._cuda_check_result
            try:
                import ctranslate2

                dev_count = ctranslate2.get_cuda_device_count()
                if dev_count <= 0:
                    result = (False, "No CUDA GPU detected (ctranslate2)")
                else:
                    supported = ctranslate2.get_supported_compute_types("cuda")
                    if supported:
                        result = (True, f"CUDA detected ({dev_count} GPU, ctranslate2)")
                    else:
                        result = (False, "CUDA not supported by ctranslate2")
            except Exception as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in CUDA_DETECT_KEYWORDS_BACKEND):
                    result = (False, f"CUDA failed: {e}")
                else:
                    result = (False, f"CUDA detection error: {e}")
            FasterWhisperBackend._cuda_check_result = result
            return result

    def get_supported_devices(self) -> List[str]:
        devices = ["cpu"]
        cuda_available, _ = self.check_cuda()
        if cuda_available:
            devices.append("cuda")
        return devices

    def get_supported_compute_types(self, device: str) -> List[str]:
        try:
            import ctranslate2
            if device == "cuda" and ctranslate2.get_cuda_device_count() <= 0:
                return ["float16", "int8", "float32"]
            return list(ctranslate2.get_supported_compute_types(device))
        except Exception:
            if device == "cpu":
                return ["int8", "float32"]
            elif device == "cuda":
                return ["float16", "int8", "float32"]
            return ["float32"]

    def get_optimal_compute_type(self, device: str) -> str:
        if device == "cuda":
            return "float16"
        return "int8"

    def get_model_map(self) -> dict:
        return {
            "Large v3 Turbo (Fast & accurate, 99 langs, ~800MB)": "large-v3-turbo",
            "Distil Large v3.5 (Ultra-fast English, ~750MB)": "distil-large-v3.5",
            "Large v3 (Best quality, 10GB+ VRAM GPU)": "large-v3",
            "Medium (GPU or fast CPU, 4GB+ RAM)": "medium",
            "Small (CPU-friendly, 2GB+ RAM)": "small",
            "Base (Minimal resources, quick)": "base",
        }

    # --- Model loading -------------------------------------------------

    def load_model(
        self,
        model_key: str,
        device: str,
        compute_type: str,
        model_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Load a faster-whisper model (in-process on CPU, or isolated worker on CUDA)."""
        model_map = self.get_model_map()
        model_id = model_map.get(model_key, model_key)

        if device == "auto":
            cuda_ok, _ = self.check_cuda()
            device = "cuda" if cuda_ok else "cpu"
        if compute_type == "auto":
            compute_type = self.get_optimal_compute_type(device)

        # Always release any previous worker or in-process model first so we
        # never leak a CUDA worker or double peak RAM/VRAM during model switches.
        self.unload_model()

        if device == "cuda" and self._resolve_worker_mode_enabled():
            self._load_model_via_worker(model_id, device, compute_type, model_path, progress_callback)
        else:
            self._load_model_in_process(model_id, device, compute_type, model_path, progress_callback)
            self._worker_mode = False

        self._current_model_key = model_key
        self._current_device = device

    def _load_model_in_process(self, model_id, device, compute_type, model_path, progress_callback=None):
        from faster_whisper import WhisperModel

        _safe_progress(progress_callback, "start")

        resolved_dir = _resolve_local_whisper_dir(model_path)
        try:
            if resolved_dir is not None:
                self._model = WhisperModel(
                    resolved_dir,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=True,
                )
            else:
                if _is_offline_mode():
                    raise RuntimeError(
                        f"Model '{model_id}' is not downloaded yet and there is no internet connection."
                    )
                _sanitize_no_proxy_env()
                _safe_progress(progress_callback, "downloading")
                self._model = WhisperModel(
                    model_id,
                    device=device,
                    compute_type=compute_type,
                    download_root=model_path,
                )
        except Exception as e:
            if resolved_dir is None and _is_connection_error(e):
                raise RuntimeError(
                    f"Model '{model_id}' is not downloaded yet and there is no internet connection."
                ) from e
            raise

        _safe_progress(progress_callback, "loaded")

    def _load_model_via_worker(self, model_id, device, compute_type, model_path, progress_callback=None):
        self._teardown_worker(force=True)
        self._worker_mode = False
        self._model = None

        _safe_progress(progress_callback, "start")

        model_present = _model_present_locally(model_path)
        if not model_present:
            if _is_offline_mode():
                raise RuntimeError(
                    f"Model '{model_id}' is not downloaded yet and there is no internet connection."
                )
            _safe_progress(progress_callback, "downloading")
        load_timeout = (
            _WORKER_LOAD_TIMEOUT_SECONDS if model_present
            else _WORKER_LOAD_TIMEOUT_SECONDS_DOWNLOADING
        )

        self._start_worker()
        request = {
            "op": "load",
            "model_id": model_id,
            "model_path": model_path,
            "device": device,
            "compute_type": compute_type,
        }
        try:
            response = self._send_worker_request(
                request, timeout=load_timeout, progress_callback=progress_callback
            )
        except Exception as e:
            stderr_tail = self._get_stderr_tail()
            self._teardown_worker(force=True)
            if stderr_tail:
                raise RuntimeError(
                    f"{type(e).__name__}: {e}\nWorker stderr (last lines):\n{stderr_tail}"
                ) from e
            raise

        if not response.get("ok"):
            error = response.get("error", "unknown worker error")
            stderr_tail = self._get_stderr_tail()
            self._teardown_worker(force=True)
            if stderr_tail:
                error = f"{error}\nWorker stderr (last lines):\n{stderr_tail}"
            raise RuntimeError(f"Worker failed to load model: {error}")

        self._worker_mode = True
        self._model = _WORKER_MODEL_HANDLE

    def _get_stderr_tail(self) -> str:
        with self._stderr_lock:
            lines = list(self._stderr_lines)
        return "\n".join(lines)

    # --- Transcription ---------------------------------------------------

    @staticmethod
    def _build_transcribe_kwargs(
        beam_size: int, vad_filter: bool, language: Optional[str]
    ) -> dict:
        kwargs = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
        }
        lang = (language or "").strip().lower()
        if lang and lang != "auto":
            kwargs["language"] = lang
        return kwargs

    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        """Transcribe an audio file using faster-whisper (in-process or via worker)."""
        kwargs = self._build_transcribe_kwargs(beam_size, vad_filter, language)

        if self._worker_mode:
            if not self.is_model_loaded:
                raise RuntimeError("No model loaded. Call load_model() first.")
            request = {
                "op": "transcribe",
                "path": audio_path,
                "kwargs": kwargs,
            }
            try:
                response = self._send_worker_request(request, timeout=_WORKER_TRANSCRIBE_TIMEOUT_SECONDS)
            except Exception as e:
                stderr_tail = self._get_stderr_tail()
                if stderr_tail:
                    raise RuntimeError(
                        f"{type(e).__name__}: {e}\nWorker stderr (last lines):\n{stderr_tail}"
                    ) from e
                raise
            if not response.get("ok"):
                error = response.get("error", "unknown error")
                stderr_tail = self._get_stderr_tail()
                if stderr_tail:
                    error = f"{error}\nWorker stderr (last lines):\n{stderr_tail}"
                raise RuntimeError(f"Worker transcription failed: {error}")
            return [
                TranscriptionSegment(text=seg["text"], start=seg["start"], end=seg["end"])
                for seg in response.get("segments", [])
            ]

        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        segments, _ = self._model.transcribe(audio_path, **kwargs)
        return [
            TranscriptionSegment(text=seg.text, start=seg.start, end=seg.end)
            for seg in segments
        ]

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        """Transcribe an in-memory ``float32`` numpy array without disk I/O on CPU/in-process."""
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1).astype(np.float32, copy=False)
        waveform = np.ascontiguousarray(waveform.reshape(-1), dtype=np.float32)
        if waveform.size == 0 or sample_rate <= 0:
            return []

        if self._worker_mode:
            return super().transcribe_audio(
                waveform,
                sample_rate=sample_rate,
                beam_size=beam_size,
                vad_filter=vad_filter,
                language=language,
            )

        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if sample_rate != 16000:
            target_len = max(1, int(round(waveform.size * 16000.0 / float(sample_rate))))
            x_old = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            waveform = np.interp(x_new, x_old, waveform).astype(np.float32)

        kwargs = self._build_transcribe_kwargs(beam_size, vad_filter, language)
        segments, _ = self._model.transcribe(waveform, **kwargs)
        return [
            TranscriptionSegment(text=seg.text, start=seg.start, end=seg.end)
            for seg in segments
        ]

    def unload_model(self) -> None:
        """Unload the current model, freeing CPU/GPU memory."""
        if self._worker_mode or self._proc is not None:
            self._teardown_worker(force=False)
            self._worker_mode = False
        if self._model is not None and self._model is not _WORKER_MODEL_HANDLE:
            ct2_model = getattr(self._model, "model", None)
            if ct2_model is not None and hasattr(ct2_model, "unload_model"):
                try:
                    ct2_model.unload_model()
                except Exception:
                    pass

        self._model = None
        self._current_model_key = None
        self._current_device = None

    @property
    def is_model_loaded(self) -> bool:
        if self._worker_mode:
            with self._worker_state_lock:
                proc = self._proc
                alive = proc is not None and proc.poll() is None
                if not alive:
                    self._model = None
                    self._worker_mode = False
                    self._proc = None
                    self._response_queue = None
                    return False
            return self._model is not None
        return self._model is not None

    @property
    def model(self):
        return self._model

    # --- Worker-process plumbing ------------------------------------------

    def _resolve_worker_mode_enabled(self) -> bool:
        env_override = os.environ.get("NEUROWHISPER_WORKER_PROCESS")
        if env_override is not None:
            return env_override.strip().lower() not in ("0", "false", "no", "")

        try:
            from app_config import CONFIG_FILE
            hostname = platform.node().replace(".", "_") or "machine"
            candidates = [
                os.path.join(_REPO_ROOT, f"whisper_config.{hostname}.json"),
                os.path.join(_REPO_ROOT, CONFIG_FILE),
                CONFIG_FILE,
            ]
            for cfg_path in candidates:
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    if isinstance(cfg, dict) and "worker_process" in cfg:
                        return bool(cfg["worker_process"])
        except Exception:
            pass

        return True

    def _resolve_worker_python_path(self) -> str:
        """Only swap ``pythonw.exe`` -> ``python.exe`` on Windows; preserve ``sys.executable`` elsewhere."""
        if os.name == "nt" and os.path.basename(sys.executable).lower() == "pythonw.exe":
            candidate = os.path.join(os.path.dirname(sys.executable), "python.exe")
            if os.path.exists(candidate):
                return candidate
        return sys.executable

    def _start_worker(self):
        python_path = self._resolve_worker_python_path()
        _sanitize_no_proxy_env()
        worker_env = os.environ.copy()
        worker_env["PYTHONUTF8"] = "1"
        worker_env["PYTHONIOENCODING"] = "utf-8:replace"

        popen_kwargs = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=worker_env,
        )
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.Popen([python_path, _WORKER_SCRIPT_PATH], **popen_kwargs)
        response_queue = queue.Queue()
        with self._stderr_lock:
            self._stderr_lines.clear()
        with self._worker_state_lock:
            self._proc = proc
            self._response_queue = response_queue

        self._reader_thread = threading.Thread(
            target=self._read_worker_stdout, args=(proc, response_queue), daemon=True
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._read_worker_stderr, args=(proc,), daemon=True
        )
        self._stderr_thread.start()

    def _read_worker_stderr(self, proc):
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    with self._stderr_lock:
                        self._stderr_lines.append(line)
        except (ValueError, OSError):
            pass

    def _read_worker_stdout(self, proc, response_queue):
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except (ValueError, TypeError):
                    continue
                response_queue.put(msg)
        except (ValueError, OSError):
            pass
        finally:
            response_queue.put(None)

    def _send_worker_request(
        self, request: dict, timeout: float, progress_callback: Optional[Callable] = None
    ) -> dict:
        with self._worker_request_lock:
            with self._worker_state_lock:
                proc = self._proc
                response_queue = self._response_queue
                if proc is None or proc.poll() is not None:
                    raise RuntimeError("Worker process is not running")

                try:
                    line = json.dumps(request) + "\n"
                    proc.stdin.write(line)
                    proc.stdin.flush()
                except (BrokenPipeError, OSError, ValueError) as e:
                    raise RuntimeError(f"Failed to send request to worker: {e}")

            if response_queue is None:
                raise RuntimeError("Worker process is not running")

            while True:
                try:
                    msg = response_queue.get(timeout=timeout)
                except queue.Empty:
                    self._teardown_worker(force=True)
                    raise TimeoutError(
                        f"Worker did not respond within {timeout}s (op={request.get('op')})"
                    )
                if msg is None:
                    raise RuntimeError("Worker process exited unexpectedly (stdout closed)")
                if isinstance(msg, dict) and msg.get("type") == "progress":
                    _safe_progress(progress_callback, msg.get("status", ""))
                    continue
                return msg

    def _teardown_worker(self, force: bool = False):
        with self._worker_state_lock:
            proc = self._proc
            self._proc = None
            self._response_queue = None
        if proc is None:
            return

        if proc.poll() is None:
            if not force:
                try:
                    proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
            try:
                proc.wait(timeout=_WORKER_SHUTDOWN_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass
