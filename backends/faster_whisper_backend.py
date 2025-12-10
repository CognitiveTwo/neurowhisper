# Faster Whisper Backend
# Wraps the existing faster-whisper library for CPU and CUDA support

import os
from typing import List, Tuple, Optional

from .base import WhisperBackend, TranscriptionSegment


class FasterWhisperBackend(WhisperBackend):
    """
    Backend using faster-whisper (CTranslate2) for transcription.
    Supports CPU and NVIDIA CUDA devices.
    """
    
    def __init__(self):
        self._model = None
        self._current_model_key = None
        self._current_device = None
    
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
        """Check if CUDA is actually available and usable for this backend."""
        try:
            import ctranslate2
            # Check if CUDA is supported
            supported = ctranslate2.get_supported_compute_types("cuda")
            if not supported:
                return False, "CUDA not supported by ctranslate2"
            
            # Try to actually detect CUDA device count (more reliable)
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    return True, f"CUDA available ({torch.cuda.device_count()} GPU(s))"
            except ImportError:
                pass  # torch not available, fall back to ctranslate2 check
            
            return True, "CUDA detected (ctranslate2)"
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ['cuda', 'cudnn', 'cublas', 'dll', 'library', 'driver', 'runtime']):
                return False, f"CUDA failed: {e}"
            return False, f"CUDA detection error: {e}"
    
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
    
    def load_model(
        self,
        model_key: str,
        device: str,
        compute_type: str,
        model_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> None:
        """Load a faster-whisper model."""
        from faster_whisper import WhisperModel
        
        # Map friendly names to model IDs
        model_map = self.get_model_map()
        model_id = model_map.get(model_key, model_key)
        
        # Determine load path
        if model_path and os.path.exists(os.path.join(model_path, "model.bin")):
            # Load from local directory
            self._model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type
            )
        else:
            # Download/load from HuggingFace
            self._model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute_type,
                download_root=model_path
            )
        
        self._current_model_key = model_key
        self._current_device = device
    
    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> List[TranscriptionSegment]:
        """Transcribe audio using faster-whisper."""
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
        """Unload the current model."""
        self._model = None
        self._current_model_key = None
        self._current_device = None
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._model is not None
    
    @property
    def model(self):
        """Direct access to the underlying model (for backwards compatibility)."""
        return self._model
