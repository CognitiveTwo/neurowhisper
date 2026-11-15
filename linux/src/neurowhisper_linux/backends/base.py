"""Abstract base class and PCM WAV helper for all ``neurowhisper_linux`` transcription backends."""

from __future__ import annotations

import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import BinaryIO, List, Optional

import numpy as np


@dataclass
class TranscriptionSegment:
    """A single timestamped segment of transcribed text."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Result returned by ``WhisperBackend.transcribe``."""

    text: str
    segments: List[TranscriptionSegment]
    language: str = "en"
    language_probability: float = 1.0
    duration: float = 0.0


def write_pcm16_wav(file_obj: BinaryIO | str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Write a mono float32 ``[-1.0, 1.0]`` (or int16) numpy array as 16-bit PCM WAV."""
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.dtype != np.int16:
        clipped = np.clip(arr.astype(np.float32, copy=False), -1.0, 1.0)
        audio_int16 = (clipped * 32767.0).astype(np.int16)
    else:
        audio_int16 = arr

    with wave.open(file_obj, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio_int16.tobytes())


class WhisperBackend(ABC):
    """Common interface implemented by local and cloud speech-to-text backends."""

    def __init__(self) -> None:
        self._model = None
        self._model_name: Optional[str] = None
        self._device: str = "cpu"
        self._compute_type: str = "auto"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def current_model(self) -> Optional[str]:
        return self._model_name

    @property
    def current_device(self) -> str:
        return self._device

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @abstractmethod
    def load_model(self, model_name: str, device: str = "auto", compute_type: str = "auto") -> None:
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
        beam_size: int = 1,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        without_timestamps: bool = True,
    ) -> TranscriptionResult:
        pass

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        pass

    @classmethod
    @abstractmethod
    def get_supported_devices(cls) -> List[str]:
        pass
