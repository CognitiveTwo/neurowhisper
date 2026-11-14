"""
Standalone child-process worker for FasterWhisperBackend's worker-process mode.

Why this exists: on CUDA, faster-whisper/ctranslate2's own unload_model() call
frees the bulk of VRAM but the *CUDA context itself* survives inside the host
process, so the GPU driver never lets the card drop to its lowest idle
performance state (P8) - it idles at a much higher power state instead. The
only reliable way to destroy a CUDA context is to end the process that
created it. This script IS that process: it holds the faster-whisper model,
and "unload" (from the parent's point of view) means "kill/end this process".

Protocol: newline-delimited JSON over stdin/stdout, one request in flight at
a time (half-duplex, synchronous). Anything that is not a protocol response
must never be written to stdout - faster_whisper/huggingface_hub/tqdm already
default their own logging/progress output to stderr, and main() redirects
Python's sys.stdout to sys.stderr as an extra safeguard.

Requests (one JSON object per line on stdin):
  {"op": "load", "model_id": "small", "model_path": "...", "device": "cuda", "compute_type": "float16"}
  {"op": "transcribe", "path": "audio.wav", "kwargs": {"beam_size": 5, "vad_filter": true}}
  {"op": "shutdown"}

Responses (one JSON object per line on stdout):
  {"ok": true}                                                            # load ack / shutdown ack
  {"ok": true, "segments": [{"text": "...", "start": 0.0, "end": 1.2}]}   # transcribe result
  {"ok": false, "error": "..."}                                           # any failure
"""
import json
import os
import sys
from typing import Optional

# Ensure Windows CUDA DLLs placed in the app root are discoverable even when
# FasterWhisperBackend is used outside whisper_gui.pyw.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.name == "nt":
    os.environ["PATH"] = _REPO_ROOT + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_REPO_ROOT)
        except OSError:
            pass

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
        parts = []
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
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # Do not misclassify HTTP 4xx/5xx responses (auth/404/server error) as offline
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
    HuggingFace cache layouts (``<model_path>/models--.../snapshots/<sha>/model.bin``)
    while ignoring incomplete hidden ``.cache`` / ``.locks`` directories.
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
    return _resolve_local_whisper_dir(model_path) is not None


def _reconfigure_utf8() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main():
    _reconfigure_utf8()
    proto_in = sys.stdin
    proto_out = sys.stdout
    # Route stray library print() calls to stderr so stdout stays strictly JSON IPC
    sys.stdout = sys.stderr

    def _write_response(resp):
        # Leading newline ensures a partial native C++ write to fd 1 without a
        # newline cannot corrupt the JSON frame read by the parent.
        proto_out.write("\n" + json.dumps(resp) + "\n")
        proto_out.flush()

    def _write_progress(status):
        try:
            _write_response({"type": "progress", "status": status})
        except Exception:
            pass

    model = None

    for raw_line in proto_in:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            request = json.loads(raw_line)
        except (ValueError, TypeError) as e:
            _write_response({"ok": False, "error": f"bad JSON request: {e}"})
            continue

        op = request.get("op")
        try:
            if op == "load":
                from faster_whisper import WhisperModel

                model_id = request.get("model_id")
                model_path = request.get("model_path")
                device = request.get("device", "cuda")
                compute_type = request.get("compute_type", "float16")

                # Unload any previously loaded model in this worker first
                if model is not None:
                    ct2_prev = getattr(model, "model", None)
                    if ct2_prev is not None and hasattr(ct2_prev, "unload_model"):
                        try:
                            ct2_prev.unload_model()
                        except Exception:
                            pass
                    model = None

                _write_progress("start")

                resolved_dir = _resolve_local_whisper_dir(model_path)
                try:
                    if resolved_dir is not None:
                        model = WhisperModel(
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
                        _write_progress("downloading")
                        model = WhisperModel(
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

                _write_progress("loaded")
                _write_response({"ok": True})

            elif op == "transcribe":
                if model is None:
                    _write_response({"ok": False, "error": "no model loaded"})
                    continue
                path = request.get("path")
                kwargs = dict(request.get("kwargs") or {})
                lang = kwargs.get("language")
                if not lang or str(lang).lower() == "auto":
                    kwargs.pop("language", None)
                segments, _info = model.transcribe(path, **kwargs)
                out_segments = [
                    {"text": seg.text, "start": seg.start, "end": seg.end} for seg in segments
                ]
                _write_response({"ok": True, "segments": out_segments})

            elif op == "shutdown":
                _write_response({"ok": True})
                break

            else:
                _write_response({"ok": False, "error": f"unknown op: {op!r}"})

        except Exception as e:  # noqa: BLE001 - must never crash silently
            _write_response({"ok": False, "error": f"{type(e).__name__}: {e}"})

    sys.exit(0)


if __name__ == "__main__":
    main()
