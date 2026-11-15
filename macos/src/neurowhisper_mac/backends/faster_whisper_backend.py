"""In-process Faster-Whisper (CTranslate2 ARM64/x86_64 CPU INT8) backend for macOS."""

from __future__ import annotations

import gc
import importlib.util
import logging
import sys
import threading
from typing import Dict, List, Optional

import numpy as np

from ..config import LEGACY_MODEL_ALIASES, MODEL_MAP
from .base import TranscriptionSegment, WhisperBackend, prepare_mono_float32

logger = logging.getLogger(__name__)

FASTER_WHISPER_REPO_OVERRIDES: Dict[str, str] = {
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
}


class FasterWhisperBackend(WhisperBackend):
    """CTranslate2 CPU backend running cleanly in-process on macOS (no CUDA worker subprocess)."""

    name = "faster-whisper"

    def __init__(self, worker_process: bool = False) -> None:
        super().__init__()
        self._model = None
        self._resolved_model_id: Optional[str] = None
        self._infer_lock = threading.Lock()

    @classmethod
    def is_available(cls) -> bool:
        if "faster_whisper" in sys.modules:
            return sys.modules["faster_whisper"] is not None
        return importlib.util.find_spec("faster_whisper") is not None

    @classmethod
    def get_supported_devices(cls) -> List[str]:
        return ["cpu"]

    @classmethod
    def resolve_model_id(cls, model_name: str) -> str:
        raw = (model_name or "").strip()
        short = MODEL_MAP.get(raw) or LEGACY_MODEL_ALIASES.get(raw) or raw
        if short in ("parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v2", "canary-1b-v2"):
            raise ValueError(
                f"Model '{model_name}' requires the ONNX ASR backend. "
                "Set Acceleration to 'Auto' or 'ONNX ASR (CoreML/CPU)'."
            )
        return FASTER_WHISPER_REPO_OVERRIDES.get(short, short)

    def load_model(
        self,
        model_name: str,
        device: str = "cpu",
        compute_type: str = "auto",
        download_root: Optional[str] = None,
    ) -> None:
        if not self.is_available():
            raise RuntimeError("faster-whisper is not installed.")

        from faster_whisper import WhisperModel  # type: ignore

        model_id = self.resolve_model_id(model_name)
        resolved_compute = "int8" if compute_type in ("auto", "", None) else compute_type

        # Try offline local cache first to avoid unnecessary HuggingFace Hub pings
        loaded_model = None
        if download_root:
            try:
                loaded_model = WhisperModel(
                    model_id,
                    device="cpu",
                    compute_type=resolved_compute,
                    download_root=str(download_root),
                    local_files_only=True,
                )
            except Exception:
                loaded_model = None

        if loaded_model is None:
            try:
                loaded_model = WhisperModel(
                    model_id,
                    device="cpu",
                    compute_type=resolved_compute,
                    download_root=str(download_root) if download_root else None,
                    local_files_only=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load Faster-Whisper model '{model_id}': {exc}"
                ) from exc

        with self._infer_lock:
            self._model = loaded_model
            self._resolved_model_id = model_id
            self._current_model = model_name
            self._current_device = "cpu"
            self._compute_type = resolved_compute

        logger.info("Loaded Faster-Whisper model %s (%s) on CPU (%s)", model_name, model_id, resolved_compute)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        vad_filter: bool = True,
        **kwargs,
    ) -> List[TranscriptionSegment]:
        if self._model is None:
            raise RuntimeError("Faster-Whisper model is not loaded.")

        mono = prepare_mono_float32(audio, sample_rate=sample_rate, target_sr=16000)
        if mono.size == 0:
            return []

        lang_arg: Optional[str] = None
        if language and language.strip().lower() not in ("auto", ""):
            lang_arg = language.strip().lower()
        elif self._resolved_model_id and "distil-large-v3" in self._resolved_model_id:
            lang_arg = "en"

        with self._infer_lock:
            segments_iter, _info = self._model.transcribe(
                mono,
                language=lang_arg,
                vad_filter=bool(vad_filter),
                beam_size=int(kwargs.get("beam_size", 5)),
                condition_on_previous_text=False,
            )
            results = [
                TranscriptionSegment(
                    text=seg.text.strip(),
                    start=float(seg.start),
                    end=float(seg.end),
                )
                for seg in segments_iter
                if seg.text and seg.text.strip()
            ]
        return results

    def unload_model(self) -> None:
        with self._infer_lock:
            self._model = None
            self._resolved_model_id = None
            self._current_model = None
        gc.collect()

    def is_model_loaded(self) -> bool:
        return self._model is not None
