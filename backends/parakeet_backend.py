# Parakeet NeMo Backend
# Wraps NVIDIA NeMo ASR for Parakeet TDT model support

import logging
import os
import tempfile
from typing import List, Tuple, Optional

from app_config import AUDIO_RUNTIME, PARAKEET_DEFAULT_MODEL_ID, PARAKEET_MODEL_MAP
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
        return PARAKEET_MODEL_MAP.copy()
    
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
        logging.getLogger(__name__).debug(f"[Parakeet] load_model called with:")
        logging.getLogger(__name__).debug(f"  model_key: {model_key}")
        logging.getLogger(__name__).debug(f"  device: {device}")
        logging.getLogger(__name__).debug(f"  compute_type: {compute_type}")
        logging.getLogger(__name__).debug(f"  model_path: {model_path}")
        
        # Unload any previously loaded model before allocating a new one
        self.unload_model()

        def _safe_progress(status: str) -> None:
            if progress_callback:
                try:
                    progress_callback(status)
                except Exception:
                    pass

        logging.getLogger(__name__).debug("[Parakeet] Importing torch...")
        import torch
        logging.getLogger(__name__).debug(f"[Parakeet] torch imported. CUDA available: {torch.cuda.is_available()}")

        logging.getLogger(__name__).debug("[Parakeet] Importing nemo.collections.asr...")
        import nemo.collections.asr as nemo_asr
        logging.getLogger(__name__).info("[Parakeet] NeMo ASR imported successfully")

        # Map friendly names to model IDs
        model_map = self.get_model_map()
        logging.getLogger(__name__).debug(f"[Parakeet] Model map: {model_map}")

        if model_key in model_map:
            model_id = model_map[model_key]
        elif model_key in model_map.values():
            model_id = model_key
        elif model_key in ("parakeet-tdt-0.6b-v3", "nemo-parakeet-tdt-0.6b-v3"):
            model_id = "nvidia/parakeet-tdt-0.6b-v3"
        elif model_key in ("parakeet-tdt-0.6b-v2", "nemo-parakeet-tdt-0.6b-v2"):
            model_id = "nvidia/parakeet-tdt-0.6b-v2"
        elif "nvidia/parakeet" in model_key:
            model_id = model_key
        else:
            model_id = PARAKEET_DEFAULT_MODEL_ID

        if device in ("cuda", "auto") and torch.cuda.is_available():
            map_location = "cuda"
            device = "cuda"
        else:
            map_location = "cpu"
            device = "cpu"

        _safe_progress("start")

        use_local_nemo = False
        if model_path:
            if model_path.endswith('.nemo') and os.path.isfile(model_path):
                use_local_nemo = True
            elif os.path.isdir(model_path):
                try:
                    nemo_files = [f for f in os.listdir(model_path) if f.endswith('.nemo')]
                    if nemo_files:
                        model_path = os.path.join(model_path, nemo_files[0])
                        use_local_nemo = True
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[Parakeet] Error scanning model_path: {e}")

        if use_local_nemo:
            logging.getLogger(__name__).info(f"[Parakeet] Loading from local .nemo file: {model_path}")
            self._model = nemo_asr.models.ASRModel.restore_from(
                model_path,
                map_location=map_location
            )
        else:
            _safe_progress("downloading")
            logging.getLogger(__name__).info(f"[Parakeet] Loading from HuggingFace: {model_id}")
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_id,
                map_location=map_location
            )

        if device == "cuda":
            self._model = self._model.cuda()
        else:
            self._model = self._model.cpu()

        self._model.eval()

        if device == "cuda" and compute_type in ("float16", "auto"):
            self._model = self._model.half()

        self._current_model_key = model_key
        self._current_device = device

        logging.getLogger(__name__).info("[Parakeet] Model loading complete!")
        _safe_progress("loaded")

    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        """Transcribe audio using Parakeet/NeMo."""
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            output = self._model.transcribe([audio_path], timestamps=True)
        except TypeError:
            output = self._model.transcribe([audio_path])

        result = []
        if output and len(output) > 0:
            transcription = output[0]
            if hasattr(transcription, 'timestamp') and transcription.timestamp:
                segments = transcription.timestamp.get('segment', [])
                for seg in segments:
                    txt = seg.get('segment', seg.get('text', '')).strip()
                    if txt:
                        result.append(TranscriptionSegment(
                            text=txt,
                            start=seg.get('start', 0.0),
                            end=seg.get('end', 0.0)
                        ))
            elif hasattr(transcription, 'text'):
                text = transcription.text.strip()
                if text:
                    result.append(TranscriptionSegment(text=text, start=0.0, end=0.0))
            else:
                text = str(transcription).strip()
                if text:
                    result.append(TranscriptionSegment(text=text, start=0.0, end=0.0))

        return result

    def transcribe_array(
        self,
        audio_array,
        sample_rate: int = AUDIO_RUNTIME["sample_rate"],
        beam_size: int = 5
    ) -> List[TranscriptionSegment]:
        """Transcribe audio from numpy array safely (closed temp file + clipping)."""
        return self.transcribe_audio(audio_array, sample_rate=sample_rate, beam_size=beam_size)

    def unload_model(self) -> None:
        """Unload the current model and release GPU VRAM."""
        if self._model is not None:
            import gc
            model_ref = self._model
            self._model = None
            del model_ref
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        self._current_model_key = None
        self._current_device = None

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
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
