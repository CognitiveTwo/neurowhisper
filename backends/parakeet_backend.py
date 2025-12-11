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
        """Load a Parakeet model using NeMo.
        
        Note: model_path is ignored for Parakeet models. NeMo handles its own
        model caching. Only use model_path if it's an explicit .nemo file.
        """
        print(f"[Parakeet] load_model called with:")
        print(f"  model_key: {model_key}")
        print(f"  device: {device}")
        print(f"  compute_type: {compute_type}")
        print(f"  model_path: {model_path}")
        
        print("[Parakeet] Importing torch...")
        import torch
        print(f"[Parakeet] torch imported. CUDA available: {torch.cuda.is_available()}")
        
        print("[Parakeet] Importing nemo.collections.asr...")
        import nemo.collections.asr as nemo_asr
        print("[Parakeet] NeMo ASR imported successfully")
        
        # Map friendly names to model IDs
        model_map = self.get_model_map()
        print(f"[Parakeet] Model map: {model_map}")
        
        # First check if model_key is directly in our map
        if model_key in model_map:
            model_id = model_map[model_key]
            print(f"[Parakeet] Found model_key in map -> model_id: {model_id}")
        # Also check if it's a HuggingFace ID we recognize
        elif model_key in model_map.values():
            model_id = model_key
            print(f"[Parakeet] model_key is a HF ID -> model_id: {model_id}")
        # Check if it's the Parakeet model from GUI's MODEL_MAP
        elif "nvidia/parakeet" in model_key:
            model_id = model_key
            print(f"[Parakeet] model_key contains nvidia/parakeet -> model_id: {model_id}")
        else:
            # Default to the latest v3 model
            model_id = "nvidia/parakeet-tdt-0.6b-v3"
            print(f"[Parakeet] Using default model_id: {model_id}")
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            map_location = "cuda"
            print(f"[Parakeet] Using CUDA device")
        else:
            map_location = "cpu"
            device = "cpu"  # Fallback to CPU if CUDA requested but not available
            print(f"[Parakeet] Using CPU device")
        
        if progress_callback:
            progress_callback("Downloading Parakeet model...")
        
        # Check if model_path is an explicit .nemo file (user-specified local model)
        # Otherwise, let NeMo handle downloading from HuggingFace
        use_local_nemo = False
        if model_path:
            print(f"[Parakeet] Checking model_path: {model_path}")
            # Only use local path if it's explicitly a .nemo file
            if model_path.endswith('.nemo') and os.path.isfile(model_path):
                use_local_nemo = True
                print(f"[Parakeet] Using local .nemo file: {model_path}")
            # Or if it contains a .nemo file directly
            elif os.path.isdir(model_path):
                try:
                    nemo_files = [f for f in os.listdir(model_path) if f.endswith('.nemo')]
                    print(f"[Parakeet] Found .nemo files in dir: {nemo_files}")
                    if nemo_files:
                        model_path = os.path.join(model_path, nemo_files[0])
                        use_local_nemo = True
                        print(f"[Parakeet] Using local .nemo file: {model_path}")
                except Exception as e:
                    print(f"[Parakeet] Error scanning model_path: {e}")
            else:
                print(f"[Parakeet] model_path exists but is not .nemo file or dir with .nemo")
        else:
            print(f"[Parakeet] No model_path provided, will download from HuggingFace")
        
        print(f"[Parakeet] use_local_nemo: {use_local_nemo}")
        
        if use_local_nemo:
            # Load from local .nemo file
            print(f"[Parakeet] Loading from local .nemo file: {model_path}")
            self._model = nemo_asr.models.ASRModel.restore_from(
                model_path, 
                map_location=map_location
            )
        else:
            # Download from HuggingFace using NeMo's native mechanism
            print(f"[Parakeet] Loading from HuggingFace: {model_id}")
            print(f"[Parakeet] Calling ASRModel.from_pretrained(model_name='{model_id}', map_location='{map_location}')")
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_id,
                map_location=map_location
            )
            print(f"[Parakeet] Model loaded from HuggingFace successfully")
        
        # Move to appropriate device
        print(f"[Parakeet] Moving model to device: {device}")
        if device == "cuda":
            self._model = self._model.cuda()
        else:
            self._model = self._model.cpu()
        
        # Set to evaluation mode
        print(f"[Parakeet] Setting model to eval mode")
        self._model.eval()
        
        # Enable half-precision if requested and on CUDA
        if device == "cuda" and compute_type == "float16":
            print(f"[Parakeet] Converting to half precision (float16)")
            self._model = self._model.half()
        
        self._current_model_key = model_key
        self._current_device = device
        
        print(f"[Parakeet] Model loading complete!")
        if progress_callback:
            progress_callback("Parakeet model loaded!")
    
    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> List[TranscriptionSegment]:
        """Transcribe audio using Parakeet/NeMo.
        
        Note: beam_size and vad_filter are accepted for API compatibility with
        other backends but are not used by NeMo's RNNT models.
        """
        print(f"[Parakeet] transcribe() called with audio_path: {audio_path}")
        
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        import os
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"[Parakeet] Audio file exists, size: {os.path.getsize(audio_path)} bytes")
        
        # NeMo's TDT/RNNT models use a simpler transcribe API
        # They don't support beam_size - that's for CTC models
        print(f"[Parakeet] Calling model.transcribe() with timestamps=True")
        try:
            output = self._model.transcribe([audio_path], timestamps=True)
            print(f"[Parakeet] Transcription returned: {type(output)}")
        except TypeError as e:
            # If timestamps parameter is not supported, try without it
            print(f"[Parakeet] timestamps parameter failed: {e}, trying without")
            output = self._model.transcribe([audio_path])
            print(f"[Parakeet] Transcription returned: {type(output)}")
        
        # Extract segments from the output
        result = []
        
        # The output structure depends on the model, but typically:
        # output[0] contains the transcription result for the first file
        if output and len(output) > 0:
            transcription = output[0]
            print(f"[Parakeet] Transcription type: {type(transcription)}")
            print(f"[Parakeet] Transcription value: {transcription}")
            
            # Check if we have segment-level timestamps
            if hasattr(transcription, 'timestamp') and transcription.timestamp:
                print(f"[Parakeet] Has timestamps: {transcription.timestamp}")
                segments = transcription.timestamp.get('segment', [])
                for seg in segments:
                    result.append(TranscriptionSegment(
                        text=seg.get('segment', seg.get('text', '')),
                        start=seg.get('start', 0.0),
                        end=seg.get('end', 0.0)
                    ))
            elif hasattr(transcription, 'text'):
                # Has text attribute
                text = transcription.text.strip()
                print(f"[Parakeet] Using text attribute: {text}")
                result.append(TranscriptionSegment(
                    text=text,
                    start=0.0,
                    end=0.0
                ))
            else:
                # Fallback: treat as string
                text = str(transcription).strip()
                print(f"[Parakeet] Using str() fallback: {text}")
                result.append(TranscriptionSegment(
                    text=text,
                    start=0.0,
                    end=0.0
                ))
        
        print(f"[Parakeet] Returning {len(result)} segments")
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
