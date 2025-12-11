# Parakeet NeMo Backend
# Wraps NVIDIA NeMo ASR for Parakeet TDT model support

import os
import tempfile
from typing import List, Tuple, Optional

from .base import WhisperBackend, TranscriptionSegment


class ParakeetBackend(WhisperBackend):
    """
    Backend using NVIDIA NeMo for Parakeet TDT transcription.
    Supports CPU and NVIDIA CUDA devices.
    
    The Parakeet TDT 0.6B v3 model is a 600M parameter multilingual ASR model
    that supports 25 European languages with automatic language detection.
    """
    
    def __init__(self):
        self._model = None
        self._current_model_key = None
        self._current_device = None
    
    @property
    def name(self) -> str:
        return "parakeet"
    
    @property
    def is_available(self) -> Tuple[bool, str]:
        """Check if NeMo is installed and available."""
        try:
            import nemo.collections.asr as nemo_asr
            return True, "NeMo ASR available"
        except ImportError as e:
            return False, f"NeMo not installed: {e}"
    
    def check_cuda(self) -> Tuple[bool, str]:
        """Check if CUDA is available for NeMo/PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    return True, f"CUDA available: {device_name}"
            return False, "CUDA not available"
        except ImportError:
            return False, "PyTorch not installed"
        except Exception as e:
            return False, f"CUDA check failed: {e}"
    
    def get_supported_devices(self) -> List[str]:
        """Return list of supported devices."""
        devices = ["cpu"]
        cuda_available, _ = self.check_cuda()
        if cuda_available:
            devices.insert(0, "cuda")  # Prefer CUDA
        return devices
    
    def get_supported_compute_types(self, device: str) -> List[str]:
        """Get compute types for a specific device."""
        if device == "cuda":
            return ["float16", "float32"]  # NeMo supports these on GPU
        else:
            return ["float32"]  # CPU typically uses float32
    
    def get_optimal_compute_type(self, device: str) -> str:
        """Get the optimal compute type for a device."""
        if device == "cuda":
            return "float16"
        return "float32"
    
    def get_model_map(self) -> dict:
        """Get mapping of friendly names to model identifiers for Parakeet."""
        return {
            "Parakeet TDT 0.6B v3 (Multi-lang, 2GB+ RAM)": "nvidia/parakeet-tdt-0.6b-v3",
            "Parakeet TDT 0.6B v2 (English, 2GB+ RAM)": "nvidia/parakeet-tdt-0.6b-v2",
        }
    
    def load_model(
        self,
        model_key: str,
        device: str,
        compute_type: str,
        model_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> None:
        """Load a Parakeet model using NeMo."""
        import torch
        import nemo.collections.asr as nemo_asr
        
        # Map friendly names to model IDs
        model_map = self.get_model_map()
        model_id = model_map.get(model_key, model_key)
        
        # If model_key looks like a path or HuggingFace ID, use it directly
        if "/" in model_key and model_key not in model_map.values():
            model_id = model_key
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            # NeMo will use GPU automatically when model is moved to cuda
            map_location = "cuda"
        else:
            map_location = "cpu"
            device = "cpu"  # Fallback to CPU if CUDA requested but not available
        
        if progress_callback:
            progress_callback("Downloading Parakeet model...")
        
        # Load model from HuggingFace or local path
        if model_path and os.path.exists(model_path):
            # Load from local .nemo file
            self._model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=map_location)
        else:
            # Download from HuggingFace
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_id,
                map_location=map_location
            )
        
        # Move to appropriate device
        if device == "cuda":
            self._model = self._model.cuda()
        else:
            self._model = self._model.cpu()
        
        # Set to evaluation mode
        self._model.eval()
        
        # Enable half-precision if requested and on CUDA
        if device == "cuda" and compute_type == "float16":
            self._model = self._model.half()
        
        self._current_model_key = model_key
        self._current_device = device
        
        if progress_callback:
            progress_callback("Parakeet model loaded!")
    
    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> List[TranscriptionSegment]:
        """Transcribe audio using Parakeet/NeMo."""
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # NeMo transcribe method expects a list of audio files
        output = self._model.transcribe([audio_path], timestamps=True)
        
        # Extract segments from the output
        result = []
        
        # The output structure depends on the model, but typically:
        # output[0] contains the transcription result for the first file
        if output and len(output) > 0:
            transcription = output[0]
            
            # Check if we have segment-level timestamps
            if hasattr(transcription, 'timestamp') and transcription.timestamp:
                segments = transcription.timestamp.get('segment', [])
                for seg in segments:
                    result.append(TranscriptionSegment(
                        text=seg.get('segment', seg.get('text', '')),
                        start=seg.get('start', 0.0),
                        end=seg.get('end', 0.0)
                    ))
            else:
                # Fallback: create a single segment with the full text
                text = transcription.text if hasattr(transcription, 'text') else str(transcription)
                result.append(TranscriptionSegment(
                    text=text.strip(),
                    start=0.0,
                    end=0.0
                ))
        
        return result
    
    def transcribe_array(
        self,
        audio_array,
        sample_rate: int = 16000,
        beam_size: int = 5
    ) -> List[TranscriptionSegment]:
        """
        Transcribe audio from numpy array.
        
        Args:
            audio_array: NumPy array of audio samples
            sample_rate: Sample rate of the audio (default 16000)
            beam_size: Beam size for decoding
            
        Returns:
            List of TranscriptionSegment objects
        """
        import numpy as np
        import scipy.io.wavfile as wav
        
        # Save to temporary file (NeMo expects file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            # Ensure proper int16 format
            if audio_array.dtype != np.int16:
                audio_array = (audio_array * 32767).astype(np.int16)
            wav.write(temp_path, sample_rate, audio_array)
        
        try:
            result = self.transcribe(temp_path, beam_size=beam_size)
        finally:
            # Cleanup temp file
            try:
                os.remove(temp_path)
            except:
                pass
        
        return result
    
    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if self._model is not None:
            # Clear CUDA cache if using GPU
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            del self._model
            self._model = None
        
        self._current_model_key = None
        self._current_device = None
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None
    
    @property
    def model(self):
        """Direct access to the underlying NeMo model."""
        return self._model
    
    @property
    def supported_languages(self) -> List[str]:
        """
        Return list of languages supported by Parakeet TDT 0.6B v3.
        The model auto-detects language, so this is informational.
        """
        return [
            "en",  # English
            "de",  # German
            "fr",  # French
            "es",  # Spanish
            "it",  # Italian
            "pt",  # Portuguese
            "nl",  # Dutch
            "pl",  # Polish
            "ru",  # Russian
            "uk",  # Ukrainian
            "cs",  # Czech
            "sk",  # Slovak
            "hu",  # Hungarian
            "ro",  # Romanian
            "bg",  # Bulgarian
            "hr",  # Croatian
            "sl",  # Slovenian
            "el",  # Greek
            "da",  # Danish
            "sv",  # Swedish
            "fi",  # Finnish
            "et",  # Estonian
            "lv",  # Latvian
            "lt",  # Lithuanian
            "mt",  # Maltese
        ]
