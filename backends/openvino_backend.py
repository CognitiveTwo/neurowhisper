# OpenVINO Backend
# Intel GPU acceleration using OpenVINO and Optimum Intel

import os
from typing import List, Tuple, Optional

from app_config import AUDIO_RUNTIME, OPENVINO_MODEL_MAP
from .base import WhisperBackend, TranscriptionSegment


class OpenVINOBackend(WhisperBackend):
    """
    Backend using OpenVINO for Intel GPU acceleration.
    Uses Hugging Face Optimum Intel for optimized Whisper models.
    """
    
    def __init__(self):
        self._model = None
        self._processor = None
        self._current_model_key = None
        self._device = None
    
    @property
    def name(self) -> str:
        return "openvino"
    
    @property
    def is_available(self) -> Tuple[bool, str]:
        """Check if an Intel GPU/NPU is present and the required OpenVINO
        stack (optimum.intel, transformers, librosa) can be imported.
        CPU-only OpenVINO is not considered "available" for this backend."""
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices

            has_gpu = any('GPU' in d for d in devices)
            has_npu = any('NPU' in d for d in devices)

            if not (has_gpu or has_npu):
                return False, "No Intel GPU or NPU detected"

            try:
                import optimum.intel  # noqa: F401
                import transformers  # noqa: F401
                import librosa  # noqa: F401
            except ImportError as e:
                return False, (
                    "OpenVINO dependencies not installed "
                    f"(optimum.intel/transformers/librosa): {e}"
                )

            device_str = []
            if has_gpu:
                device_str.append("Intel GPU")
            if has_npu:
                device_str.append("Intel NPU")
            return True, f"OpenVINO with {', '.join(device_str)} available"

        except ImportError:
            return False, "OpenVINO not installed. Install with: pip install openvino optimum[openvino]"
        except Exception as e:
            return False, f"OpenVINO error: {e}"
    
    def _get_best_device(self) -> str:
        """Determine the best OpenVINO device to use."""
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            
            # Priority: GPU > NPU > CPU
            if any('GPU' in d for d in devices):
                return "GPU"
            elif any('NPU' in d for d in devices):
                return "NPU"
            else:
                return "CPU"
        except Exception:
            return "CPU"
    
    def get_supported_devices(self) -> List[str]:
        """Return list of supported OpenVINO devices."""
        try:
            import openvino as ov
            core = ov.Core()
            # Filter to relevant devices
            relevant = [d for d in core.available_devices if d in ['CPU', 'GPU', 'NPU']]
            return relevant if relevant else ["CPU"]
        except Exception:
            return ["CPU"]
    
    def get_supported_compute_types(self, device: str) -> List[str]:
        """OpenVINO handles precision automatically."""
        # OpenVINO typically uses FP16/INT8 internally
        return ["auto", "fp16", "int8"]
    
    def get_model_map(self) -> dict:
        """OpenVINO uses pre-optimized models from HuggingFace."""
        # Use pre-optimized OpenVINO models (faster loading, no conversion needed)
        return OPENVINO_MODEL_MAP.copy()
    
    def load_model(
        self,
        model_key: str,
        device: str = "auto",
        compute_type: str = "auto",
        model_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> None:
        """Load a pre-optimized OpenVINO Whisper model."""
        try:
            from optimum.intel import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(
                "OpenVINO dependencies not installed. "
                "Install with: pip install openvino optimum[openvino] transformers"
            ) from e
        
        # Map friendly name to model ID
        model_map = self.get_model_map()
        model_id = model_map.get(model_key, "OpenVINO/whisper-small-fp16-ov")
        
        # Determine device
        if device == "auto" or device == "gpu":
            device = self._get_best_device()
        self._device = device
        self._current_device = device  # For compatibility
        
        # Load processor from original OpenAI model (needed for tokenization)
        original_model = model_id.replace("OpenVINO/", "openai/").replace("-fp16-ov", "")
        self._processor = AutoProcessor.from_pretrained(original_model)
        
        # Load pre-optimized OpenVINO model (no export needed)
        self._model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            device=device
        )
        
        self._current_model_key = model_key
    
    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True  # Note: VAD not directly supported, ignored
    ) -> List[TranscriptionSegment]:
        """
        Transcribe audio using OpenVINO.
        Returns a List[TranscriptionSegment] per the WhisperBackend interface.
        """
        if not self._model or not self._processor:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        try:
            import librosa
            import torch
        except ImportError:
            raise ImportError("librosa and torch required for audio processing")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=AUDIO_RUNTIME["sample_rate"])
        
        # Process audio
        inputs = self._processor(
            audio,
            sampling_rate=AUDIO_RUNTIME["sample_rate"],
            return_tensors="pt"
        )
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self._model.generate(
                inputs.input_features,
                max_length=448,
                num_beams=beam_size
            )
        
        # Decode
        transcription = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return [TranscriptionSegment(text=transcription, start=0.0, end=0.0)]
    
    def unload_model(self) -> None:
        """Unload the current model."""
        self._model = None
        self._processor = None
        self._current_model_key = None
        self._device = None
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._model is not None
