"""Out-of-process ``faster-whisper`` worker with dynamic Linux CUDA 12 ``.so`` preloading.

Isolates CTranslate2 / CUDA / OpenMP allocations inside a child process so unloading a
model (`shutdown` / process exit) returns 100% of RAM and VRAM to the Linux kernel, and
any native driver/cuDNN fault cannot crash the GUI process.

Invoked as:
  - ``<AppImage/frozen-binary> --fw-worker`` when frozen via PyInstaller
  - ``python fw_worker.py`` when running directly from source
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import sys
import traceback
from pathlib import Path
from typing import List

# Allow running directly as ``python fw_worker.py`` from source without installed package
if not __package__:
    _src_root = str(Path(__file__).resolve().parents[2])
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

from neurowhisper_linux import paths  # noqa: E402

# Ordered CUDA 12 / cuBLAS / cuDNN shared objects to preload with RTLD_GLOBAL
CUDA_SHARED_LIB_CANDIDATES: tuple[str, ...] = (
    "libcudart.so.12",
    "libnvJitLink.so.12",
    "libnvrtc.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcudnn.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    # CUDA 11 / cuDNN 8 fallbacks for older host toolkits
    "libcudart.so.11.0",
    "libcublasLt.so.11",
    "libcublas.so.11",
    "libcudnn.so.8",
    "libcudnn_ops_infer.so.8",
    "libcudnn_cnn_infer.so.8",
)

_PRELOADED_CUDA_LIBS: List[str] = []
_PRELOAD_RAN: bool = False


def preload_cuda_libraries() -> List[str]:
    """Discover user/venv/system CUDA ``.so`` directories and preload them with ``RTLD_GLOBAL``.

    Must be called BEFORE importing ``ctranslate2``, ``faster_whisper``, or ``onnxruntime`` so the
    dynamic linker satisfies ``libcublas.so.12`` / ``libcudnn.so.9`` symbols even when the user
    installed CUDA wheels into ``~/.local/share/neurowhisper/cuda-libs`` after launching the app.
    """
    global _PRELOAD_RAN
    lib_dirs = paths.discover_cuda_lib_dirs()
    if lib_dirs:
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        existing_parts = [p for p in existing_ld.split(os.pathsep) if p]
        merged = list(dict.fromkeys([*lib_dirs, *existing_parts]))
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(merged)

    if _PRELOAD_RAN and _PRELOADED_CUDA_LIBS:
        return list(_PRELOADED_CUDA_LIBS)
    _PRELOAD_RAN = True

    loaded_basenames = {os.path.basename(p) for p in _PRELOADED_CUDA_LIBS}
    rtld_global = getattr(ctypes, "RTLD_GLOBAL", os.RTLD_GLOBAL if hasattr(os, "RTLD_GLOBAL") else 0)

    for directory in lib_dirs:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            continue
        for lib_name in CUDA_SHARED_LIB_CANDIDATES:
            if lib_name in loaded_basenames:
                continue
            candidate = dir_path / lib_name
            if not candidate.exists():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=rtld_global)
                _PRELOADED_CUDA_LIBS.append(str(candidate))
                loaded_basenames.add(lib_name)
            except OSError:
                continue

    return list(_PRELOADED_CUDA_LIBS)


def _send_json(ipc_out, payload: dict) -> None:
    ipc_out.write(json.dumps(payload) + "\n")
    ipc_out.flush()


def main() -> None:
    """Run the JSON-over-stdio ``faster-whisper`` inference worker loop."""
    # Reserve real stdout exclusively for JSON IPC; route library prints/tqdm to stderr
    ipc_out = sys.stdout
    sys.stdout = sys.stderr

    preloaded = preload_cuda_libraries()

    import numpy as np

    model = None
    loaded_device = "cpu"
    loaded_compute_type = "int8"

    _send_json(
        ipc_out,
        {
            "ready": True,
            "pid": os.getpid(),
            "cuda_libs_count": len(preloaded),
        },
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            _send_json(ipc_out, {"ok": False, "error": f"bad json: {exc}"})
            continue

        cmd = req.get("cmd")

        if cmd == "ping":
            _send_json(ipc_out, {"ok": True, "pid": os.getpid()})

        elif cmd == "shutdown":
            model = None
            _send_json(ipc_out, {"ok": True})
            return

        elif cmd == "load":
            try:
                preload_cuda_libraries()
                from faster_whisper import WhisperModel

                model_name = req["model"]
                device = req.get("device", "cpu")
                compute_type = req.get("compute_type", "int8")
                download_root = req.get("download_root") or str(paths.get_models_dir())
                Path(download_root).mkdir(parents=True, exist_ok=True)

                model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=download_root,
                )
                loaded_device = device
                loaded_compute_type = compute_type
                _send_json(
                    ipc_out,
                    {
                        "ok": True,
                        "device": loaded_device,
                        "compute_type": loaded_compute_type,
                    },
                )
            except Exception as exc:
                model = None
                _send_json(
                    ipc_out,
                    {
                        "ok": False,
                        "error": str(exc),
                        "tb": traceback.format_exc(),
                    },
                )

        elif cmd == "transcribe":
            if model is None:
                _send_json(ipc_out, {"ok": False, "error": "Model not loaded in worker"})
                continue
            try:
                audio_bytes = base64.b64decode(req["audio_b64"])
                audio = np.frombuffer(audio_bytes, dtype=np.float32)

                kwargs = {
                    "beam_size": int(req.get("beam_size", 1)),
                    "vad_filter": bool(req.get("vad_filter", True)),
                    "without_timestamps": bool(req.get("without_timestamps", True)),
                }
                if req.get("language"):
                    kwargs["language"] = req["language"]
                if req.get("vad_parameters"):
                    kwargs["vad_parameters"] = req["vad_parameters"]

                segments_gen, info = model.transcribe(audio, **kwargs)
                segments = []
                text_parts = []
                for seg in segments_gen:
                    segments.append(
                        {
                            "start": float(seg.start),
                            "end": float(seg.end),
                            "text": seg.text,
                        }
                    )
                    text_parts.append(seg.text.strip())

                _send_json(
                    ipc_out,
                    {
                        "ok": True,
                        "text": " ".join(text_parts),
                        "segments": segments,
                        "language": info.language if info else (req.get("language") or "en"),
                        "language_probability": float(info.language_probability) if info else 1.0,
                        "duration": float(info.duration) if info else float(len(audio) / 16000.0),
                    },
                )
            except Exception as exc:
                _send_json(
                    ipc_out,
                    {
                        "ok": False,
                        "error": str(exc),
                        "tb": traceback.format_exc(),
                    },
                )
        else:
            _send_json(ipc_out, {"ok": False, "error": f"unknown cmd: {cmd}"})


if __name__ == "__main__":
    main()
