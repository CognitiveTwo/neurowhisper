# Backends Package
# Registry and factory for Whisper transcription backends

import sys
import threading
from typing import Dict, List, Optional, Tuple, Type

from app_config import (
    BACKEND_DISPLAY_NAMES,
    BACKEND_FACTORY_DEFAULTS,
    ONLINE_BACKEND_NAME,
)

# Platform detection
IS_MAC = sys.platform == "darwin"

# Lazy imports guarded by a lock so concurrent threads never see partial state
_registry_lock = threading.RLock()
_backends: Dict[str, Type] = {}
_backend_instances: Dict[str, object] = {}
_register_reasons: List[str] = []


def _register_backends() -> None:
    """Register all available backends lazily and atomically on first access."""
    global _backends, _register_reasons
    if _backends:
        return

    with _registry_lock:
        if _backends:
            return

        registered: Dict[str, Type] = {}
        reasons: List[str] = []

        # Core dependency: faster-whisper
        from .faster_whisper_backend import FasterWhisperBackend
        registered["faster-whisper"] = FasterWhisperBackend

        # Lightweight ONNX Runtime ASR (Parakeet TDT v3/v2, Canary 1B v2)
        try:
            from .onnx_asr_backend import OnnxAsrBackend
            registered["onnx-asr"] = OnnxAsrBackend
        except Exception as e:
            reasons.append(f"onnx-asr: {e}")

        # OpenVINO (Intel GPU/NPU; not supported on macOS)
        if not IS_MAC:
            try:
                from .openvino_backend import OpenVINOBackend
                registered["openvino"] = OpenVINOBackend
            except Exception as e:
                reasons.append(f"openvino: {e}")

        # OpenAI Cloud
        try:
            from .openai_backend import OpenAIBackend
            registered["openai"] = OpenAIBackend
        except Exception as e:
            reasons.append(f"openai: {e}")

        # Optional NVIDIA NeMo Parakeet (.nemo / PyTorch)
        try:
            from .parakeet_backend import ParakeetBackend
            registered["parakeet"] = ParakeetBackend
        except Exception as e:
            reasons.append(f"parakeet: {e}")

        _register_reasons = reasons
        _backends = registered


def get_available_backends() -> List[Tuple[str, bool, str]]:
    """Get list of backends with their availability status."""
    _register_backends()

    results: List[Tuple[str, bool, str]] = []
    with _registry_lock:
        items = list(_backends.items())
        instances = dict(_backend_instances)

    for name, backend_class in items:
        try:
            backend = instances.get(name) or backend_class()
            is_available, reason = backend.is_available
            results.append((name, is_available, reason))
        except Exception as e:
            results.append((name, False, str(e)))

    return results


def detect_best_backend() -> Tuple[str, str, str]:
    """
    Auto-detect the best available backend and device.
    Returns: (backend_name, device, reason)

    Priority order:
    1. CUDA (if NVIDIA GPU available) - fastest for compatible hardware
    2. OpenVINO (if Intel GPU/NPU available) - good for Intel hardware
    3. CPU (always available) - universal fallback
    """
    _register_backends()

    reasons = list(_register_reasons)

    # Check CUDA first (via faster-whisper)
    if "faster-whisper" in _backends:
        try:
            backend = create_backend("faster-whisper")
            cuda_available, cuda_reason = backend.check_cuda()
            if cuda_available:
                return "faster-whisper", "cuda", cuda_reason
            reasons.append(f"faster-whisper cuda: {cuda_reason}")
        except Exception as e:
            reasons.append(f"faster-whisper: {e}")

    # Check OpenVINO next (Intel GPU or NPU)
    if "openvino" in _backends:
        try:
            backend = create_backend("openvino")
            is_available, reason = backend.is_available
            if is_available:
                best_dev = "gpu"
                if hasattr(backend, "_get_best_device"):
                    ov_dev = backend._get_best_device().upper()
                    if "NPU" in ov_dev:
                        best_dev = "npu"
                    elif "GPU" in ov_dev:
                        best_dev = "gpu"
                elif "NPU" in (reason or "").upper():
                    best_dev = "npu"
                return "openvino", best_dev, reason
            reasons.append(f"openvino: {reason}")
        except Exception as e:
            reasons.append(f"openvino: {e}")

    # Fallback to CPU
    cpu_reason = BACKEND_FACTORY_DEFAULTS["cpu_fallback_reason"]
    if reasons:
        cpu_reason = cpu_reason + " (" + "; ".join(reasons) + ")"
    return (
        BACKEND_FACTORY_DEFAULTS["cpu_fallback_backend"],
        BACKEND_FACTORY_DEFAULTS["cpu_fallback_device"],
        cpu_reason,
    )


def create_backend(backend_name: str = "auto"):
    """Factory method to create or retrieve a backend instance."""
    _register_backends()

    if backend_name == "auto":
        backend_name, _, _ = detect_best_backend()

    with _registry_lock:
        if backend_name not in _backends:
            available = list(_backends.keys())
            raise ValueError(f"Unknown backend '{backend_name}'. Available: {available}")

        if backend_name not in _backend_instances:
            _backend_instances[backend_name] = _backends[backend_name]()

        return _backend_instances[backend_name]


def unload_all_backends(exclude: Optional[str] = None) -> None:
    """Unload and evict cached backend instances (optionally keeping ``exclude``)."""
    with _registry_lock:
        to_unload = []
        for name in list(_backend_instances.keys()):
            if exclude and name == exclude:
                continue
            to_unload.append(_backend_instances.pop(name))

    for instance in to_unload:
        try:
            if getattr(instance, "is_model_loaded", False) or hasattr(instance, "unload_model"):
                instance.unload_model()
        except Exception:
            pass


def get_backend_display_name(backend_name: str) -> str:
    """Get human-readable name for a backend."""
    return BACKEND_DISPLAY_NAMES.get(backend_name, backend_name)


def is_online_backend(backend_name: str) -> bool:
    """Check if a backend requires internet connection."""
    return backend_name == ONLINE_BACKEND_NAME
