# Abstract Base Class for Whisper Backends

from abc import ABC, abstractmethod
from typing import List, Tuple, NamedTuple, Optional
from dataclasses import dataclass, field
import time


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing information."""
    text: str
    start: float  # Start time in seconds
    end: float    # End time in seconds


@dataclass
class TranscriptionResult:
    """Result of a transcription including performance metrics."""
    segments: List[TranscriptionSegment]
    text: str  # Full concatenated text
    latency_ms: float  # Time from transcribe() call to result (milliseconds)
    backend_name: str  # Which backend produced this result
    device: str  # Which device was used (cpu, cuda, gpu)


class WhisperBackend(ABC):
    """
    Abstract base class defining the interface for Whisper transcription backends.
    
    All backends must implement these methods to be compatible with NeuroWhisper.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'faster-whisper', 'openvino')."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> Tuple[bool, str]:
        """
        Check if this backend is available on the current system.
        
        Returns:
            Tuple of (is_available, reason_message)
        """
        pass
    
    @abstractmethod
    def get_supported_devices(self) -> List[str]:
        """
        Get list of devices this backend supports.
        
        Returns:
            List of device names (e.g., ['cpu', 'cuda'], ['cpu', 'gpu'])
        """
        pass
    
    @abstractmethod
    def get_supported_compute_types(self, device: str) -> List[str]:
        """
        Get list of compute types supported for a specific device.
        
        Args:
            device: Device name (e.g., 'cpu', 'cuda', 'gpu')
            
        Returns:
            List of compute types (e.g., ['int8', 'float16', 'float32'])
        """
        pass
    
    @abstractmethod
    def load_model(
        self,
        model_key: str,
        device: str,
        compute_type: str,
        model_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> None:
        """
        Load a Whisper model.
        
        Args:
            model_key: Model identifier (e.g., 'small', 'medium', 'large-v3')
            device: Target device ('cpu', 'cuda', 'gpu')
            compute_type: Precision ('int8', 'float16', 'float32')
            model_path: Optional local path to model files
            progress_callback: Optional callback for download progress
        """
        pass
    
    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> List[TranscriptionSegment]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            beam_size: Beam size for decoding
            vad_filter: Whether to apply voice activity detection
            
        Returns:
            List of TranscriptionSegment objects
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        pass
    
    @property
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        pass
    
    def get_model_map(self) -> dict:
        """
        Get mapping of friendly names to model identifiers.
        
        Can be overridden by backends that use different model naming.
        """
        return {
            "Large v3 (Best quality, 10GB+ VRAM GPU)": "large-v3",
            "Medium (GPU or fast CPU, 4GB+ RAM)": "medium",
            "Small (CPU-friendly, 2GB+ RAM)": "small",
            "Base (Minimal resources, quick)": "base"
        }
    
    def transcribe_with_timing(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio and measure latency.
        
        This is a convenience wrapper around transcribe() that adds timing.
        
        Args:
            audio_path: Path to the audio file
            beam_size: Beam size for decoding
            vad_filter: Whether to apply voice activity detection
            
        Returns:
            TranscriptionResult with segments, text, and latency_ms
        """
        start_time = time.perf_counter()
        
        segments = self.transcribe(audio_path, beam_size, vad_filter)
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        # Concatenate all segment texts
        full_text = " ".join(seg.text for seg in segments).strip()
        
        return TranscriptionResult(
            segments=segments,
            text=full_text,
            latency_ms=latency_ms,
            backend_name=self.name,
            device=getattr(self, '_current_device', 'unknown')
        )
