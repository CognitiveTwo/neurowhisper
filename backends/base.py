# Abstract Base Class for Whisper Backends

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import tempfile
import time
from typing import Callable, List, Optional, Tuple, Union
import wave

import numpy as np


def write_pcm16_wav(
    target: Union[str, os.PathLike, object],
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> None:
    """Write a mono 16-bit PCM WAV using stdlib ``wave`` (zero ``scipy`` import).

    Clips ``float32``/``float64`` samples cleanly to ``[-1.0, 1.0]`` before
    scaling so peaks above ``1.0`` never wrap around into harsh digital pops.
    ``target`` may be a filesystem path or an open binary file-like object
    (such as ``io.BytesIO``).
    """
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    arr = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.floating):
        clean = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        clipped = np.clip(clean, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2")
    elif arr.dtype == np.int16:
        pcm16 = arr.astype("<i2", copy=False)
    else:
        clean = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        clipped = np.clip(clean, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2")

    target_arg = os.fspath(target) if isinstance(target, (str, os.PathLike)) else target
    with wave.open(target_arg, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


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
        """Return the backend name (e.g., 'faster-whisper', 'onnx-asr', 'openvino')."""
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
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Load a speech-to-text model."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        """Transcribe an audio file on disk."""
        pass

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        """Transcribe an in-memory mono ``float32`` waveform directly.

        Backends that support in-memory arrays override this to avoid temporary
        WAV disk writes and audio decoding overhead. Subclasses that only work
        with files automatically inherit this safe stdlib-``wave`` fallback.
        """
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        arr = arr.reshape(-1)
        if arr.size == 0:
            return []
        fd, temp_path = tempfile.mkstemp(prefix="neurowhisper_buf_", suffix=".wav")
        os.close(fd)
        try:
            write_pcm16_wav(temp_path, arr, sample_rate=sample_rate)
            try:
                return self.transcribe(
                    temp_path, beam_size=beam_size, vad_filter=vad_filter, language=language
                )
            except TypeError:
                # Backwards compatibility with custom subclasses lacking `language`
                return self.transcribe(temp_path, beam_size=beam_size, vad_filter=vad_filter)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
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
        """Get mapping of friendly names to model identifiers."""
        return {
            "Large v3 Turbo (Fast & accurate, 99 langs, ~800MB)": "large-v3-turbo",
            "Distil Large v3.5 (Ultra-fast English, ~750MB)": "distil-large-v3.5",
            "Large v3 (Best quality, 10GB+ VRAM GPU)": "large-v3",
            "Medium (GPU or fast CPU, 4GB+ RAM)": "medium",
            "Small (CPU-friendly, 2GB+ RAM)": "small",
            "Base (Minimal resources, quick)": "base",
        }

    def transcribe_with_timing(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio and measure latency."""
        start_time = time.perf_counter()

        try:
            segments = self.transcribe(
                audio_path, beam_size=beam_size, vad_filter=vad_filter, language=language
            )
        except TypeError:
            segments = self.transcribe(audio_path, beam_size, vad_filter)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        full_text = " ".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip())

        return TranscriptionResult(
            segments=segments,
            text=full_text,
            latency_ms=latency_ms,
            backend_name=self.name,
            device=getattr(self, "_current_device", "unknown"),
        )
