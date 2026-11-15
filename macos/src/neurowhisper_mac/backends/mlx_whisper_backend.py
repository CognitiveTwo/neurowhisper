"""Apple Silicon Metal GPU Whisper backend powered by ``mlx-whisper``."""

from __future__ import annotations

import gc
import importlib.util
import logging
from pathlib import Path
import platform
import sys
import threading
from typing import Dict, List, Optional

import numpy as np

from ..config import (
    LEGACY_MODEL_ALIASES,
    MLX_WHISPER_MODEL_MAP,
    MLX_WHISPER_SHORT_ID_MAP,
    MODEL_MAP,
)
from .base import TranscriptionSegment, WhisperBackend, prepare_mono_float32

logger = logging.getLogger(__name__)


def _get_mlx_modules():
    mlx_mod = sys.modules.get("mlx")
    mlx_whisper_mod = sys.modules.get("mlx_whisper")
    if mlx_mod is None:
        import mlx as mlx_mod  # type: ignore
    if mlx_whisper_mod is None:
        import mlx_whisper as mlx_whisper_mod  # type: ignore
    return mlx_mod, mlx_whisper_mod


class MlxWhisperBackend(WhisperBackend):
    """Hardware-accelerated Whisper inference on Apple Silicon Metal GPUs via MLX."""

    name = "mlx-whisper"

    def __init__(self) -> None:
        super().__init__()
        self._repo_id: Optional[str] = None
        self._model_path: Optional[str] = None
        self._loaded: bool = False
        self._infer_lock = threading.Lock()

    @classmethod
    def is_apple_silicon(cls) -> bool:
        return sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")

    @classmethod
    def is_available(cls) -> bool:
        if not cls.is_apple_silicon():
            return False
        if "mlx" in sys.modules and "mlx_whisper" in sys.modules:
            return sys.modules["mlx"] is not None and sys.modules["mlx_whisper"] is not None
        return (
            importlib.util.find_spec("mlx") is not None
            and importlib.util.find_spec("mlx_whisper") is not None
        )

    @classmethod
    def get_supported_devices(cls) -> List[str]:
        return ["mps"] if cls.is_available() else []

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        try:
            cls.resolve_repo_id(model_name)
            return True
        except ValueError:
            return False

    @classmethod
    def resolve_repo_id(cls, model_name: str) -> str:
        raw = (model_name or "").strip()
        if raw in MLX_WHISPER_MODEL_MAP:
            return MLX_WHISPER_MODEL_MAP[raw]

        short = MODEL_MAP.get(raw) or LEGACY_MODEL_ALIASES.get(raw) or raw
        if short in MLX_WHISPER_SHORT_ID_MAP:
            return MLX_WHISPER_SHORT_ID_MAP[short]

        if raw.startswith("mlx-community/"):
            return raw

        raise ValueError(
            f"Model '{model_name}' does not have an MLX Whisper repository. "
            "Select a Whisper model or switch Acceleration to 'Auto' / 'ONNX ASR (CoreML/CPU)'."
        )

    def load_model(
        self,
        model_name: str,
        device: str = "mps",
        compute_type: str = "float16",
        download_root: Optional[str] = None,
    ) -> None:
        if not self.is_available():
            raise RuntimeError(
                "Apple Metal (MLX) requires macOS on Apple Silicon (arm64) with mlx-whisper installed."
            )

        _get_mlx_modules()
        repo_id = self.resolve_repo_id(model_name)
        resolved_path = self._ensure_snapshot(repo_id, download_root=download_root)

        with self._infer_lock:
            self._repo_id = repo_id
            self._model_path = resolved_path
            self._current_model = model_name
            self._current_device = "mps"
            self._compute_type = compute_type if compute_type != "auto" else "float16"
            self._loaded = True

        logger.info("Loaded MLX Whisper model %s (%s) on Apple Metal", model_name, resolved_path)

    @staticmethod
    def _ensure_snapshot(repo_id: str, download_root: Optional[str] = None) -> str:
        """Download or resolve HuggingFace snapshot into ``download_root`` cache."""
        if Path(repo_id).exists():
            return str(Path(repo_id))

        cache_dir = str(download_root) if download_root else None
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return repo_id

        # Fast path: check local cache without network traffic
        try:
            return str(
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
            )
        except Exception:
            pass

        # Online download into cache_dir
        try:
            return str(
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download MLX model '{repo_id}'. Check your internet connection "
                f"for the initial download. Details: {exc}"
            ) from exc

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        vad_filter: bool = True,
        **kwargs,
    ) -> List[TranscriptionSegment]:
        if not self._loaded or not self._model_path:
            raise RuntimeError("MLX Whisper model is not loaded.")

        mono = prepare_mono_float32(audio, sample_rate=sample_rate, target_sr=16000)
        if mono.size == 0:
            return []

        lang_arg: Optional[str] = None
        if language and language.strip().lower() not in ("auto", ""):
            lang_arg = language.strip().lower()
        elif self._repo_id and "distil-whisper" in self._repo_id:
            lang_arg = "en"

        _, mlx_whisper = _get_mlx_modules()

        transcribe_kwargs: Dict[str, object] = {
            "path_or_hf_repo": self._model_path,
            "condition_on_previous_text": False,
            "fp16": self._compute_type != "float32",
        }
        if lang_arg is not None:
            transcribe_kwargs["language"] = lang_arg

        with self._infer_lock:
            result = mlx_whisper.transcribe(mono, **transcribe_kwargs)

        if not isinstance(result, dict):
            text_val = str(result or "").strip()
            return [TranscriptionSegment(text=text_val, start=0.0, end=0.0)] if text_val else []

        raw_segments = result.get("segments")
        segments: List[TranscriptionSegment] = []
        if isinstance(raw_segments, list) and raw_segments:
            for seg in raw_segments:
                if isinstance(seg, dict):
                    txt = str(seg.get("text", "") or "").strip()
                    if txt:
                        segments.append(
                            TranscriptionSegment(
                                text=txt,
                                start=float(seg.get("start", 0.0) or 0.0),
                                end=float(seg.get("end", 0.0) or 0.0),
                            )
                        )
        if not segments:
            top_text = str(result.get("text", "") or "").strip()
            if top_text:
                dur = float(mono.size) / 16000.0
                segments.append(TranscriptionSegment(text=top_text, start=0.0, end=dur))
        return segments

    def unload_model(self) -> None:
        with self._infer_lock:
            self._loaded = False
            self._model_path = None
            self._repo_id = None
            self._current_model = None

        try:
            import mlx.core as mx  # type: ignore

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
        gc.collect()

    def is_model_loaded(self) -> bool:
        return self._loaded and self._model_path is not None
