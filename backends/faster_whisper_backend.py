# Faster Whisper Backend
# Wraps the existing faster-whisper library for CPU and CUDA support

import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from typing import List, Tuple, Optional

from app_config import CUDA_DETECT_KEYWORDS_BACKEND
from .base import WhisperBackend, TranscriptionSegment

# Substrings/exception-type names that indicate a model download failed
# because there is no (or a broken) internet connection, as opposed to some
# other failure (bad model id, corrupt cache, etc). Checked against both the
# exception's type name and its message.
_CONNECTION_ERROR_MARKERS = (
    "connectionerror",
    "localentrynotfounderror",
    "hfhubhttperror",
    "offlinemodeisenabled",
    "connection",
    "max retries",
    "failed to resolve",
    "getaddrinfo",
    "offline",
    "couldn't connect",
)


def _is_connection_error(exc: Exception) -> bool:
    """Best-effort detection of a network/connection failure raised while
    constructing a WhisperModel (huggingface_hub download path)."""
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(marker in haystack for marker in _CONNECTION_ERROR_MARKERS)


def _model_present_locally(model_path: Optional[str]) -> bool:
    """Whether usable local model weights already exist under model_path.

    Must walk: huggingface_hub's cache layout puts the weights at
    models--Systran--faster-whisper-small/snapshots/<sha>/model.bin, so a
    top-level check is False even for a fully downloaded model - which used
    to force the 1800 s "downloading" timeout on every single load.
    Mirrors WhisperApp.is_model_downloaded() in whisper_gui.pyw.
    """
    if not model_path or not os.path.isdir(model_path):
        return False
    for _root, _dirs, files in os.walk(model_path):
        if "model.bin" in files:
            return True
        if any(name.endswith(".nemo") or name.endswith(".xml") for name in files):
            return True
    return False


def _safe_progress(progress_callback: Optional[callable], status: str) -> None:
    """Call progress_callback(status), swallowing any exception it raises so
    a bad callback can never break model loading."""
    if not progress_callback:
        return
    try:
        progress_callback(status)
    except Exception:
        pass

# --- Worker-process mode (CUDA only) ----------------------------------------
# ctranslate2's unload_model() frees VRAM but the process's CUDA context
# survives, which keeps the GPU pinned at a higher power state (P2) instead of
# reaching idle (P8). The fix: run the model in a child process and make
# "unload" mean "kill the child" - a dead process can't hold a CUDA context.
# See backends/fw_worker.py for the child script and its wire protocol.
_WORKER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fw_worker.py")

# Generous timeouts: first-time CUDA/cuDNN init plus loading a model already
# on disk can legitimately take tens of seconds. When the model still needs
# to be downloaded from HuggingFace, a much longer timeout is used instead
# (see _load_model_via_worker) since that depends on network speed, not just
# local hardware.
_WORKER_LOAD_TIMEOUT_SECONDS = 180
_WORKER_LOAD_TIMEOUT_SECONDS_DOWNLOADING = 1800
_WORKER_TRANSCRIBE_TIMEOUT_SECONDS = 300
_WORKER_SHUTDOWN_GRACE_SECONDS = 3.0
_WORKER_STDERR_MAXLEN = 50


class _WorkerModelHandle:
    """Truthy marker stored as `self._model` while a worker-process model is
    loaded, so legacy `if self.model:` / `hasattr(backend, 'model')` truthy
    checks in whisper_gui.pyw keep working unchanged. It carries no state -
    all real work happens via IPC with the child process."""
    __slots__ = ()


_WORKER_MODEL_HANDLE = _WorkerModelHandle()


class FasterWhisperBackend(WhisperBackend):
    """
    Backend using faster-whisper (CTranslate2) for transcription.
    Supports CPU and NVIDIA CUDA devices.

    On CUDA, by default the model runs in a child worker process (see
    fw_worker.py) so that unload_model() can fully release the GPU by killing
    the child, rather than merely freeing VRAM while a CUDA context lingers.
    Set the "worker_process" config key (or NEUROWHISPER_WORKER_PROCESS env
    var) to false to fall back to the old in-process behavior.
    """

    # Process-wide memo for check_cuda() (see there). None = not probed yet.
    _cuda_check_result: Optional[Tuple[bool, str]] = None
    _cuda_check_lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._current_model_key = None
        self._current_device = None

        # Worker-process mode state (see _load_model_via_worker below).
        self._worker_mode = False
        self._proc = None
        self._response_queue = None
        self._reader_thread = None
        self._stderr_thread = None
        # Last lines of the worker's stderr, kept so a crash/timeout error can
        # include them for diagnosis.
        self._stderr_lines = deque(maxlen=_WORKER_STDERR_MAXLEN)
        # The drain thread appends while the caller reads; without this lock
        # the read could hit "deque mutated during iteration".
        self._stderr_lock = threading.Lock()
        # Guards mutation of the worker-state fields above (_proc/_response_queue)
        # so a concurrent teardown can't set _proc to None mid-write.
        self._worker_state_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "faster-whisper"

    @property
    def is_available(self) -> Tuple[bool, str]:
        """faster-whisper is always available as it's a core dependency."""
        try:
            from faster_whisper import WhisperModel
            return True, "faster-whisper available"
        except ImportError as e:
            return False, f"faster-whisper not installed: {e}"

    def check_cuda(self) -> Tuple[bool, str]:
        """Check if CUDA is actually available and usable for this backend.

        Memoised on the class: the answer cannot change while the process
        runs, and the probe is not free - it imports ctranslate2, which is
        exactly the import the worker-process design keeps out of the GUI
        process. One probe per process instead of one per call.

        Deliberately ctranslate2-only. The old torch.cuda.is_available()
        follow-up initialised a CUDA context *in the parent process*, which
        is the very thing worker mode exists to avoid, and it could not
        change the answer: ctranslate2 is the library that will actually run
        the model.
        """
        if FasterWhisperBackend._cuda_check_result is not None:
            return FasterWhisperBackend._cuda_check_result

        with FasterWhisperBackend._cuda_check_lock:
            if FasterWhisperBackend._cuda_check_result is not None:
                return FasterWhisperBackend._cuda_check_result
            try:
                import ctranslate2

                supported = ctranslate2.get_supported_compute_types("cuda")
                if supported:
                    result = (True, "CUDA detected (ctranslate2)")
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
        """Return list of supported devices."""
        devices = ["cpu"]
        cuda_available, _ = self.check_cuda()
        if cuda_available:
            devices.append("cuda")
        return devices

    def get_supported_compute_types(self, device: str) -> List[str]:
        """Get compute types for a specific device."""
        try:
            import ctranslate2
            return list(ctranslate2.get_supported_compute_types(device))
        except Exception:
            if device == "cpu":
                return ["int8", "float32"]
            elif device == "cuda":
                return ["float16", "int8", "float32"]
            return ["float32"]

    def get_optimal_compute_type(self, device: str) -> str:
        """Get the optimal compute type for a device."""
        if device == "cuda":
            return "float16"
        else:
            return "int8"

    # --- Model loading -------------------------------------------------

    def load_model(
        self,
        model_key: str,
        device: str,
        compute_type: str,
        model_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> None:
        """Load a faster-whisper model.

        On CUDA (and unless disabled via the "worker_process" config escape
        hatch), the model is loaded inside a child worker process instead of
        this one - see _load_model_via_worker().
        """
        # Map friendly names to model IDs
        model_map = self.get_model_map()
        model_id = model_map.get(model_key, model_key)

        if device == "cuda" and self._resolve_worker_mode_enabled():
            self._load_model_via_worker(model_id, device, compute_type, model_path, progress_callback)
        else:
            self._load_model_in_process(model_id, device, compute_type, model_path, progress_callback)
            self._worker_mode = False

        self._current_model_key = model_key
        self._current_device = device

    def _load_model_in_process(self, model_id, device, compute_type, model_path, progress_callback=None):
        """Original in-process load path (used for CPU, or when worker mode
        is explicitly disabled). faster_whisper is imported lazily here so
        that constructing/using this backend in worker mode never pulls
        faster_whisper/ctranslate2 into THIS (parent) process."""
        from faster_whisper import WhisperModel

        _safe_progress(progress_callback, "start")

        model_present = _model_present_locally(model_path)
        try:
            if model_present:
                # Load from local directory
                self._model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type
                )
            else:
                _safe_progress(progress_callback, "downloading")
                # Download/load from HuggingFace
                self._model = WhisperModel(
                    model_id,
                    device=device,
                    compute_type=compute_type,
                    download_root=model_path
                )
        except Exception as e:
            if not model_present and _is_connection_error(e):
                raise RuntimeError(
                    f"Model '{model_id}' is not downloaded yet and there is no internet connection."
                ) from e
            raise

        _safe_progress(progress_callback, "loaded")

    def _load_model_via_worker(self, model_id, device, compute_type, model_path, progress_callback=None):
        """Start (or restart) the child worker process and have it load the
        model. Raises on failure; the caller (GUI) already has a
        fallback-on-load-failure path."""
        # Tear down any previous worker/in-process model before starting fresh.
        self._teardown_worker(force=True)
        self._model = None

        _safe_progress(progress_callback, "start")

        model_present = _model_present_locally(model_path)
        if not model_present:
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
                # Always RuntimeError: type(e)(msg) blows up with TypeError for
                # any exception whose constructor is not a single string.
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
        # No "loaded" here: the worker emits its own progress message for that
        # (see fw_worker._write_progress), so reporting it again duplicates it.

    def _get_stderr_tail(self) -> str:
        """Return the last captured lines of the worker's stderr, joined."""
        with self._stderr_lock:
            lines = list(self._stderr_lines)
        return "\n".join(lines)

    # --- Transcription ---------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> List[TranscriptionSegment]:
        """Transcribe audio using faster-whisper (in-process or via the
        worker process, depending on how the model was loaded)."""
        if self._worker_mode:
            if not self.is_model_loaded:
                raise RuntimeError("No model loaded. Call load_model() first.")
            request = {
                "op": "transcribe",
                "path": audio_path,
                "kwargs": {"beam_size": beam_size, "vad_filter": vad_filter},
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

        segments, _ = self._model.transcribe(
            audio_path,
            beam_size=beam_size,
            vad_filter=vad_filter
        )

        result = []
        for seg in segments:
            result.append(TranscriptionSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end
            ))

        return result

    def unload_model(self) -> None:
        """Unload the current model, freeing GPU memory.

        Worker-process mode (default on CUDA): unload = shut down (or kill)
        the child process, which destroys its CUDA context outright and lets
        the GPU driver reach its lowest idle power state (P8). This is the
        only reliable way to do that - ctranslate2's own unload_model() (see
        below) frees VRAM but the context survives in-process.

        In-process mode (CPU, or worker mode disabled): empirically verified
        (see tools_test_unload.py) that calling ctranslate2's native
        unload_model() on the underlying model before dropping the Python
        reference reliably releases the bulk of the VRAM the model added,
        whereas a plain `del` alone was not verified to do so.
        """
        if self._worker_mode:
            self._teardown_worker(force=False)
            self._worker_mode = False
        elif self._model is not None:
            ct2_model = getattr(self._model, "model", None)
            if ct2_model is not None and hasattr(ct2_model, "unload_model"):
                ct2_model.unload_model()

        self._model = None
        self._current_model_key = None
        self._current_device = None

    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded.

        In worker mode this also verifies the child process is still alive;
        a crashed/killed child is reported as not-loaded (rather than raising)
        so callers - notably whisper_gui.pyw's idle-unload/reload path - can
        transparently recover by reloading on the next use.
        """
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
        """Direct access to the underlying model (for backwards compatibility).

        In worker mode there is no in-process model object - this returns a
        truthy placeholder (see _WorkerModelHandle) so `if self.model:` style
        checks in whisper_gui.pyw keep working. Real transcription always
        goes through transcribe(), which talks to the child process.
        """
        return self._model

    # --- Worker-process plumbing ------------------------------------------

    def _resolve_worker_mode_enabled(self) -> bool:
        """Config escape hatch. Default True (worker-process mode on CUDA).
        Override order: NEUROWHISPER_WORKER_PROCESS env var, then the
        "worker_process" key in the app's whisper_config.json, then default.
        Set to false/0 to fall back to the old in-process CUDA path."""
        env_override = os.environ.get("NEUROWHISPER_WORKER_PROCESS")
        if env_override is not None:
            return env_override.strip().lower() not in ("0", "false", "no", "")

        try:
            from app_config import CONFIG_FILE
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "worker_process" in cfg:
                    return bool(cfg["worker_process"])
        except Exception:
            pass

        return True

    def _resolve_worker_python_path(self) -> str:
        """The child must run with the SAME environment/site-packages as the
        parent. sys.executable in a venv's pythonw.exe already points at that
        venv's own executable (not a shared/system Python), so we just need
        the console variant (python.exe) next to it - Popen with
        CREATE_NO_WINDOW then launches it with no console flash regardless.
        Falls back to sys.executable itself if that file isn't found."""
        exe_dir = os.path.dirname(sys.executable)
        candidate_name = "python.exe" if os.name == "nt" else "python"
        candidate = os.path.join(exe_dir, candidate_name)
        if os.path.exists(candidate):
            return candidate
        return sys.executable

    def _start_worker(self):
        python_path = self._resolve_worker_python_path()
        popen_kwargs = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if os.name == "nt":
            # Avoid a console window flashing when the parent is pythonw.exe.
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.Popen([python_path, _WORKER_SCRIPT_PATH], **popen_kwargs)
        response_queue = queue.Queue()
        self._stderr_lines.clear()
        with self._worker_state_lock:
            self._proc = proc
            self._response_queue = response_queue
        # Pass proc/queue as local args (not read via self.*) so this thread
        # never races with _teardown_worker() reassigning self._proc /
        # self._response_queue to None on the calling thread while we're
        # still draining the (now-closed) pipe.
        self._reader_thread = threading.Thread(
            target=self._read_worker_stdout, args=(proc, response_queue), daemon=True
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._read_worker_stderr, args=(proc,), daemon=True
        )
        self._stderr_thread.start()

    def _read_worker_stderr(self, proc):
        """Background thread: drain the worker's stderr into a bounded deque
        so the last lines can be surfaced if it dies or times out."""
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
        """Background thread: parse worker stdout lines as JSON and hand them
        to the synchronous request/response caller via a queue. Non-JSON
        lines (e.g. stray library output that ended up on stdout) are
        ignored rather than treated as protocol errors. On EOF (child exited)
        a sentinel (None) unblocks any pending request with a clear error."""
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
        self, request: dict, timeout: float, progress_callback: Optional[callable] = None
    ) -> dict:
        # Hold the worker-state lock across the write so a concurrent
        # _teardown_worker() can't null out _proc between the check and the
        # stdin.write (which would raise AttributeError on None).
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

        # A "load" request may emit interim {"type": "progress", ...} messages
        # before its final {"ok": ...} response - forward those to the
        # caller's progress_callback and keep waiting for the real response,
        # resetting the wait window each time so a slow-but-alive download
        # doesn't time out just because it's taking a while.
        while True:
            try:
                msg = response_queue.get(timeout=timeout)
            except queue.Empty:
                # The reply may still arrive later; if we left it queued, the
                # next request would pop this stale answer and paste the
                # wrong text.
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
        """Shut down the child process. If force=False, ask it to exit
        cleanly first (op=shutdown) and give it a grace period; either way,
        kill it if it's still alive afterward. Safe to call when no worker is
        running."""
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
