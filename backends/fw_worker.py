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
default their own logging/progress output to stderr.

Requests (one JSON object per line on stdin):
  {"op": "load", "model_id": "small", "model_path": "...", "device": "cuda", "compute_type": "float16"}
  {"op": "transcribe", "path": "audio.wav", "kwargs": {"beam_size": 5, "vad_filter": true}}
  {"op": "shutdown"}

Responses (one JSON object per line on stdout):
  {"ok": true}                                                            # load ack / shutdown ack
  {"ok": true, "segments": [{"text": "...", "start": 0.0, "end": 1.2}]}   # transcribe result
  {"ok": false, "error": "..."}                                           # any failure

This script must never crash silently: every request is wrapped so failures
are reported as {"ok": false, "error": ...} responses rather than letting an
unhandled exception kill the loop (which would look like a hang to the
parent instead of a clean error).

Exit behavior: the main loop reads until EOF (parent closed its end of our
stdin pipe - e.g. the parent crashed) or until it receives {"op":
"shutdown"}. Either way the process then exits, which is what destroys the
CUDA context. This means a crashed/killed parent can never leave an orphaned
worker pinning the GPU indefinitely.
"""
import json
import os
import sys

# Mirrors faster_whisper_backend._CONNECTION_ERROR_MARKERS / _is_connection_error
# so the worker's error message gets the same friendly translation as the
# in-process load path.
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
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(marker in haystack for marker in _CONNECTION_ERROR_MARKERS)


def _model_present_locally(model_path):
    return bool(model_path) and os.path.exists(os.path.join(model_path, "model.bin"))


def _write_response(resp):
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def _write_progress(status):
    try:
        _write_response({"type": "progress", "status": status})
    except Exception:
        pass


def main():
    model = None

    for raw_line in sys.stdin:
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
                # Imported lazily (and only ever in THIS process) so the
                # parent process never touches faster_whisper/ctranslate2 and
                # therefore never creates a CUDA context of its own.
                from faster_whisper import WhisperModel

                model_id = request.get("model_id")
                model_path = request.get("model_path")
                device = request.get("device", "cuda")
                compute_type = request.get("compute_type", "float16")

                _write_progress("start")

                # Mirrors FasterWhisperBackend._load_model_in_process(): local
                # directory with model.bin wins, else download/load by id.
                model_present = _model_present_locally(model_path)
                try:
                    if model_present:
                        model = WhisperModel(model_path, device=device, compute_type=compute_type)
                    else:
                        _write_progress("downloading")
                        model = WhisperModel(
                            model_id, device=device, compute_type=compute_type, download_root=model_path
                        )
                except Exception as e:
                    if not model_present and _is_connection_error(e):
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
                kwargs = request.get("kwargs") or {}
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

    # EOF (parent pipe closed) or an explicit shutdown - exit so the CUDA
    # context this process may have created is fully torn down by the OS.
    sys.exit(0)


if __name__ == "__main__":
    main()
