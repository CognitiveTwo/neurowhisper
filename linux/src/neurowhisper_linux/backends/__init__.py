"""Backend registry, hardware auto-detection, and singleton lifecycle for ``neurowhisper_linux``."""

from __future__ import annotations

import gc
import logging
from typing import Dict, Optional, Tuple, Type

from ..config import (
    BACKEND_DISPLAY_NAMES,
    BACKEND_FACTORY_DEFAULTS,
    CUDA_DETECT_KEYWORDS_BACKEND,
    ONLINE_BACKEND_NAME,
    ONNX_ASR_MODEL_IDS,
)
from .base import TranscriptionResult, TranscriptionSegment, WhisperBackend, write_pcm16_wav
from .faster_whisper_backend import FasterWhisperBackend
from .onnx_asr_backend import OnnxAsrBackend
from .openai_backend import OpenAIBackend
from .openvino_backend import OpenVINOBackend

log = logging.getLogger("neurowhisper.backends")

_BACKENDS: Dict[str, Type[WhisperBackend]] = {
    "faster-whisper": FasterWhisperBackend,
    "onnx-asr": OnnxAsrBackend,
    "openvino": OpenVINOBackend,
    "openai": OpenAIBackend,
}

_backend_instances: Dict[str, WhisperBackend] = {}


def get_available_backends() -> Dict[str, bool]:
    """Return ``{backend_name: is_available}`` for all Linux backends."""
    return {name: cls.is_available() for name, cls in _BACKENDS.items()}


def detect_best_backend(model_key: Optional[str] = None) -> Tuple[str, str, str]:
    """Select ``(backend_name, device, reason)`` for the current Linux hardware & model."""
    if model_key in ONNX_ASR_MODEL_IDS:
        if OnnxAsrBackend.is_available():
            supported = OnnxAsrBackend.get_supported_devices()
            dev = "cuda" if "cuda" in supported else "auto"
            reason = (
                "Parakeet TDT via ONNX Runtime (CUDA)"
                if dev == "cuda"
                else "Parakeet TDT via ONNX Runtime (CPU INT8)"
            )
            return ("onnx-asr", dev, reason)

    if FasterWhisperBackend.is_available():
        try:
            if FasterWhisperBackend.check_cuda():
                return ("faster-whisper", "cuda", "NVIDIA GPU detected (CUDA FP16)")
        except Exception as exc:
            err_str = str(exc).lower()
            if any(kw in err_str for kw in CUDA_DETECT_KEYWORDS_BACKEND):
                log.info("CUDA device visible but runtime libraries missing (%s)", exc)

    if OpenVINOBackend.is_available():
        try:
            if OpenVINOBackend.check_intel_gpu():
                return ("openvino", "gpu", "Intel GPU detected (OpenVINO)")
        except Exception:
            pass

    if FasterWhisperBackend.is_available():
        return (
            BACKEND_FACTORY_DEFAULTS["cpu_fallback_backend"],
            BACKEND_FACTORY_DEFAULTS["cpu_fallback_device"],
            BACKEND_FACTORY_DEFAULTS["cpu_fallback_reason"],
        )

    if OnnxAsrBackend.is_available():
        return ("onnx-asr", "auto", "ONNX Runtime (CPU INT8)")

    if OpenVINOBackend.is_available():
        return ("openvino", "cpu", "OpenVINO (CPU)")

    raise RuntimeError("No local transcription backend available.")


def create_backend(
    backend_type: str = "auto",
    device: str = "auto",
    model_key: Optional[str] = None,
    **kwargs,
) -> WhisperBackend:
    """Create or reuse a singleton ``WhisperBackend`` instance."""
    req = (backend_type or "auto").strip().lower()

    if req == "auto":
        req, auto_dev, reason = detect_best_backend(model_key=model_key)
        if device == "auto":
            device = auto_dev
        log.info("Auto-selected backend: %s (%s) — %s", req, device, reason)
    elif req == "cuda":
        req = "faster-whisper"
        device = "cuda"
    elif req == "parakeet":
        req = "onnx-asr"

    if model_key in ONNX_ASR_MODEL_IDS and req in ("faster-whisper", "openvino"):
        if OnnxAsrBackend.is_available():
            req = "onnx-asr"

    if req not in _BACKENDS:
        raise ValueError(f"Unknown backend: {req}. Available: {list(_BACKENDS.keys())}")

    backend_cls = _BACKENDS[req]
    if not backend_cls.is_available():
        if req != "faster-whisper" and FasterWhisperBackend.is_available():
            log.warning("%s not available; falling back to faster-whisper (CPU)", req)
            req = "faster-whisper"
            backend_cls = FasterWhisperBackend
        else:
            raise RuntimeError(f"Backend '{req}' is not installed or unavailable.")

    for existing_key in list(_backend_instances.keys()):
        if existing_key != req:
            try:
                _backend_instances[existing_key].unload_model()
            except Exception:
                pass
            del _backend_instances[existing_key]

    if req not in _backend_instances:
        _backend_instances[req] = backend_cls(**kwargs) if req == ONLINE_BACKEND_NAME else backend_cls()
    elif req == ONLINE_BACKEND_NAME and kwargs:
        instance = _backend_instances[req]
        if isinstance(instance, OpenAIBackend):
            instance.configure(**kwargs)

    return _backend_instances[req]


def unload_all_backends() -> None:
    """Unload and discard all cached backend instances (including ``fw_worker`` subprocesses)."""
    for key in list(_backend_instances.keys()):
        try:
            _backend_instances[key].unload_model()
        except Exception:
            pass
        del _backend_instances[key]
    gc.collect()


def get_backend_display_name(backend_type: str) -> str:
    return BACKEND_DISPLAY_NAMES.get(backend_type, backend_type)


def is_online_backend(backend_type: str) -> bool:
    return backend_type == ONLINE_BACKEND_NAME


__all__ = [
    "FasterWhisperBackend",
    "OnnxAsrBackend",
    "OpenAIBackend",
    "OpenVINOBackend",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperBackend",
    "create_backend",
    "detect_best_backend",
    "get_available_backends",
    "get_backend_display_name",
    "is_online_backend",
    "unload_all_backends",
    "write_pcm16_wav",
]
