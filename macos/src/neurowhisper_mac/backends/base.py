"""Abstract base interface and PCM16 WAV serializer for speech-to-text backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import io
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union
import wave

import numpy as np


@dataclass
class TranscriptionSegment:
    """Single transcribed speech segment."""

    text: str
    start: float = 0.0
    end: float = 0.0


@dataclass
class TranscriptionResult:
    """Aggregated transcription result."""

    segments: List[TranscriptionSegment]
    language: Optional[str] = None
    language_probability: float = 1.0

    @property
    def text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments if seg.text and seg.text.strip())


def prepare_mono_float32(audio: np.ndarray, sample_rate: int = 16000, target_sr: int = 16000) -> np.ndarray:
    """Normalize input audio to a contiguous 1-D float32 array in [-1.0, 1.0] at ``target_sr``."""
    arr = np.asarray(audio)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)

    if arr.ndim > 1:
        arr = arr.mean(axis=-1)

    if np.issubdtype(arr.dtype, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max(1.0, max_val)
    else:
        arr = arr.astype(np.float32, copy=False)

    arr = np.clip(arr, -1.0, 1.0)

    if sample_rate > 0 and target_sr > 0 and sample_rate != target_sr and arr.size > 1:
        duration = arr.size / float(sample_rate)
        new_len = max(1, int(round(duration * target_sr)))
        old_idx = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
        new_idx = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        arr = np.interp(new_idx, old_idx, arr).astype(np.float32)

    return np.ascontiguousarray(arr, dtype=np.float32)


def write_pcm16_wav(
    target: Union[str, Path, BinaryIO, io.BytesIO],
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> None:
    """Write mono float32/int16 numpy audio as 16-bit PCM WAV to a path or file-like buffer."""
    mono = prepare_mono_float32(audio, sample_rate=sample_rate, target_sr=sample_rate)
    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")

    if isinstance(target, (str, Path)):
        with wave.open(str(target), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm16.tobytes())
    else:
        with wave.open(target, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm16.tobytes())


class WhisperBackend(ABC):
    """Uniform interface across MLX Metal, ONNX ASR CoreML, Faster-Whisper CPU, and OpenAI Cloud."""

    name: str = "base"

    def __init__(self) -> None:
        self._current_model: Optional[str] = None
        self._current_device: str = "cpu"
        self._compute_type: str = "auto"

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model

    @property
    def current_device(self) -> str:
        return self._current_device

    @abstractmethod
    def load_model(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Optional[str] = None,
    ) -> None:
        """Load or prepare model weights."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        vad_filter: bool = True,
        **kwargs,
    ) -> List[TranscriptionSegment]:
        """Transcribe 16kHz mono audio into segments."""

    @abstractmethod
    def unload_model(self) -> None:
        """Release model weights and accelerator memory."""

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Return True if a model is currently resident and ready to transcribe."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this backend's runtime dependencies and hardware are available."""

    @classmethod
    def get_supported_devices(cls) -> List[str]:
        return ["cpu"]

    def get_info(self) -> Tuple[str, str, Optional[str]]:
        return (self.name, self._current_device, self._current_model)
