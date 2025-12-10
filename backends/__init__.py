# Backends Package
# Registry and factory for Whisper transcription backends

from typing import Dict, Type, Optional, List, Tuple

# Lazy imports to avoid loading unused dependencies
_backends: Dict[str, Type] = {}
_backend_instances: Dict[str, object] = {}


def _register_backends():
    """Register all available backends. Called lazily on first access."""
    global _backends
    if _backends:
        return
    
    # Always register faster-whisper (core dependency)
    from .faster_whisper_backend import FasterWhisperBackend
    _backends["faster-whisper"] = FasterWhisperBackend
    
    # Try to register OpenVINO if available
    try:
        from .openvino_backend import OpenVINOBackend
        _backends["openvino"] = OpenVINOBackend
    except ImportError:
        pass  # OpenVINO not installed
    
    # Try to register OpenAI if available
    try:
        from .openai_backend import OpenAIBackend
        _backends["openai"] = OpenAIBackend
    except ImportError:
        pass  # OpenAI not installed


def get_available_backends() -> List[Tuple[str, bool, str]]:
    """
    Get list of backends with their availability status.
    Returns: List of (name, is_available, reason) tuples
    """
    _register_backends()
    
    results = []
    for name, backend_class in _backends.items():
        try:
            backend = backend_class()
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
    2. OpenVINO (if Intel GPU available) - good for Intel hardware
    3. CPU (always available) - universal fallback
    """
    _register_backends()
    
    # Check CUDA first (via faster-whisper)
    if "faster-whisper" in _backends:
        try:
            from .faster_whisper_backend import FasterWhisperBackend
            backend = FasterWhisperBackend()
            cuda_available, cuda_reason = backend.check_cuda()
            if cuda_available:
                return "faster-whisper", "cuda", cuda_reason
        except Exception:
            pass
    
    # Check OpenVINO next
    if "openvino" in _backends:
        try:
            from .openvino_backend import OpenVINOBackend
            backend = OpenVINOBackend()
            is_available, reason = backend.is_available
            if is_available:
                return "openvino", "gpu", reason
        except Exception:
            pass
    
    # Fallback to CPU
    return "faster-whisper", "cpu", "Using CPU (universal fallback)"


def create_backend(backend_name: str = "auto"):
    """
    Factory method to create a backend instance.
    
    Args:
        backend_name: "auto", "faster-whisper", or "openvino"
    
    Returns:
        Backend instance
    """
    _register_backends()
    
    if backend_name == "auto":
        backend_name, _, _ = detect_best_backend()
    
    if backend_name not in _backends:
        available = list(_backends.keys())
        raise ValueError(f"Unknown backend '{backend_name}'. Available: {available}")
    
    # Return cached instance or create new one
    if backend_name not in _backend_instances:
        _backend_instances[backend_name] = _backends[backend_name]()
    
    return _backend_instances[backend_name]


def get_backend_display_name(backend_name: str) -> str:
    """Get human-readable name for a backend."""
    names = {
        "faster-whisper": "Faster Whisper (CPU/CUDA)",
        "openvino": "OpenVINO (Intel GPU)",
        "openai": "OpenAI Cloud (Online)",
    }
    return names.get(backend_name, backend_name)


def is_online_backend(backend_name: str) -> bool:
    """Check if a backend requires internet connection."""
    return backend_name == "openai"
