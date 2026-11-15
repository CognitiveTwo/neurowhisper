"""Backend registry and automatic hardware/model dispatcher for macOS."""

from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple, Type

from ..config import (
    BACKEND_DISPLAY_NAMES,
    BACKEND_FACTORY_DEFAULTS,
    LEGACY_MODEL_ALIASES,
    MODEL_MAP,
    ONLINE_BACKEND_NAME,
    ONNX_ASR_MODEL_IDS,
)
from .base import TranscriptionResult, TranscriptionSegment, WhisperBackend, write_pcm16_wav
from .faster_whisper_backend import FasterWhisperBackend
from .mlx_whisper_backend import MlxWhisperBackend
from .onnx_asr_backend import OnnxAsrBackend
from .openai_backend import OpenAIBackend

BACKEND_CLASSES: Dict[str, Type[WhisperBackend]] = {
    "mlx-whisper": MlxWhisperBackend,
    "onnx-asr": OnnxAsrBackend,
    "faster-whisper": FasterWhisperBackend,
    "openai": OpenAIBackend,
}

_ACTIVE_BACKENDS: List[WhisperBackend] = []


def is_online_backend(name: str) -> bool:
    return (name or "").strip().lower() == ONLINE_BACKEND_NAME


def get_backend_display_name(name: str) -> str:
    key = (name or "").strip().lower()
    return BACKEND_DISPLAY_NAMES.get(key, name)


def get_available_backends() -> List[str]:
    return [name for name, cls in BACKEND_CLASSES.items() if cls.is_available()]


def detect_best_backend(model_key: Optional[str] = None) -> Tuple[str, str, str]:
    """Select ``(backend_name, device, reason)`` for macOS hardware and the chosen model."""
    raw = (model_key or "").strip()
    short_id = MODEL_MAP.get(raw) or LEGACY_MODEL_ALIASES.get(raw) or raw

    # 1. Parakeet / ONNX-native models always route to OnnxAsrBackend (CoreML -> CPU)
    if short_id in ONNX_ASR_MODEL_IDS:
        if OnnxAsrBackend.is_available():
            if OnnxAsrBackend.has_coreml_provider():
                return ("onnx-asr", "coreml", "Parakeet ONNX (Apple CoreML)")
            return ("onnx-asr", "cpu", "Parakeet ONNX (CPU)")

    # 2. Apple Silicon Metal GPU via mlx-whisper for Whisper models
    if MlxWhisperBackend.is_available():
        if not short_id or MlxWhisperBackend.supports_model(short_id):
            return ("mlx-whisper", "mps", "Apple Silicon Metal GPU (MLX)")

    # 3. Universal ARM64 / Intel CPU INT8 fallback via CTranslate2
    return (
        BACKEND_FACTORY_DEFAULTS["cpu_fallback_backend"],
        BACKEND_FACTORY_DEFAULTS["cpu_fallback_device"],
        BACKEND_FACTORY_DEFAULTS["cpu_fallback_reason"],
    )


def create_backend(name: str = "auto", model_key: Optional[str] = None, **kwargs) -> WhisperBackend:
    """Instantiate a speech-to-text backend by name (resolving ``auto`` cleanly)."""
    norm = (name or "auto").strip().lower()
    if norm == "auto":
        norm, _, _ = detect_best_backend(model_key=model_key)
    elif norm in ("mlx", "metal"):
        norm = "mlx-whisper"

    cls = BACKEND_CLASSES.get(norm)
    if cls is None:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(BACKEND_CLASSES)}")

    if cls is FasterWhisperBackend:
        instance = FasterWhisperBackend(worker_process=bool(kwargs.get("worker_process", False)))
    elif cls is OpenAIBackend:
        instance = OpenAIBackend(
            api_key=str(kwargs.get("api_key", "") or ""),
            transcription_model=str(kwargs.get("transcription_model", "") or ""),
            edit_enabled=bool(kwargs.get("edit_enabled", False)),
            edit_model=str(kwargs.get("edit_model", "") or ""),
            edit_prompt=str(kwargs.get("edit_prompt", "") or ""),
            language=str(kwargs.get("language", "auto") or "auto"),
        )
    else:
        instance = cls()

    _ACTIVE_BACKENDS.append(instance)
    return instance


def unload_all_backends() -> None:
    """Unload any tracked backend instances and reclaim memory."""
    while _ACTIVE_BACKENDS:
        backend = _ACTIVE_BACKENDS.pop()
        try:
            backend.unload_model()
        except Exception:
            pass
    gc.collect()


__all__ = [
    "FasterWhisperBackend",
    "MlxWhisperBackend",
    "OnnxAsrBackend",
    "OpenAIBackend",
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
