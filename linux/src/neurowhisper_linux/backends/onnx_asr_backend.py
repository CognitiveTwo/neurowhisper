"""ONNX ASR backend for Parakeet TDT 0.6B v3 & Canary on Linux (CPU INT8 + optional CUDA)."""

from __future__ import annotations

import gc
import logging
import os
from typing import List, Optional

import numpy as np

from .. import paths
from .base import TranscriptionResult, TranscriptionSegment, WhisperBackend
from .fw_worker import preload_cuda_libraries

log = logging.getLogger("neurowhisper.backends.onnx_asr")


class OnnxAsrBackend(WhisperBackend):
    """Lightweight ONNX Runtime backend for NVIDIA Parakeet TDT 0.6B v3 (zero PyTorch/NeMo overhead)."""

    MODEL_MAPPING = {
        "parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
        "nemo-parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
        "parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
        "nemo-parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
        "canary-1b-v2": "nemo-canary-1b-v2",
        "nemo-canary-1b-v2": "nemo-canary-1b-v2",
        "whisper-base": "whisper-base",
    }

    DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"

    def __init__(self) -> None:
        super().__init__()
        self._provider: str = "CPUExecutionProvider"

    @property
    def backend_name(self) -> str:
        if "CUDA" in self._provider:
            return "ONNX ASR (CUDA)"
        return "ONNX ASR (CPU)"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import onnx_asr  # noqa: F401
            import onnxruntime  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def get_supported_devices(cls) -> List[str]:
        devices = ["cpu"]
        try:
            preload_cuda_libraries()
            import onnxruntime as ort

            if "CUDAExecutionProvider" in ort.get_available_providers():
                devices.insert(0, "cuda")
        except Exception:
            pass
        return devices

    @staticmethod
    def _ensure_hf_cache_dir() -> None:
        """Route Hugging Face hub downloads into ``~/.cache/neurowhisper/models`` unless user overrode it."""
        if not os.environ.get("HF_HOME") and not os.environ.get("HF_HUB_CACHE"):
            hf_dir = paths.get_models_dir() / "huggingface"
            hf_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HUB_CACHE"] = str(hf_dir)

    def load_model(self, model_name: str, device: str = "auto", compute_type: str = "auto") -> None:
        self.unload_model()
        self._ensure_hf_cache_dir()
        preload_cuda_libraries()

        import onnx_asr
        import onnxruntime as ort

        onnx_model = self.MODEL_MAPPING.get(model_name, self.DEFAULT_MODEL)
        quantization: Optional[str] = "int8" if compute_type in ("auto", "int8") else None
        if compute_type in ("float32", "fp32", "float16", "fp16"):
            quantization = None

        available = ort.get_available_providers()
        req_dev = (device or "auto").strip().lower()
        if req_dev == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._provider = "CUDAExecutionProvider"
            self._device = "cuda"
        elif req_dev == "auto" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._provider = "CUDAExecutionProvider"
            self._device = "cuda"
        else:
            providers = ["CPUExecutionProvider"]
            self._provider = "CPUExecutionProvider"
            self._device = "cpu"

        kwargs: dict = {"providers": providers}
        if quantization:
            kwargs["quantization"] = quantization

        try:
            self._model = onnx_asr.load_model(onnx_model, **kwargs)
        except Exception as exc:
            if quantization:
                log.info("ONNX quantized load (%s) unavailable (%s); loading default weights", quantization, exc)
                kwargs.pop("quantization", None)
                self._model = onnx_asr.load_model(onnx_model, **kwargs)
            else:
                raise

        self._model_name = model_name
        self._compute_type = quantization or "fp32"

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
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        audio_f32 = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        duration = len(audio_f32) / float(sample_rate)

        rec_kwargs = {}
        if language and language != "auto":
            rec_kwargs["language"] = language

        try:
            result = self._model.recognize(audio_f32, sample_rate=sample_rate, **rec_kwargs)
        except TypeError:
            result = self._model.recognize(audio_f32, sample_rate=sample_rate)

        text = result.strip() if isinstance(result, str) else str(result).strip()
        segments = [TranscriptionSegment(start=0.0, end=duration, text=text)] if text else []
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language or "en",
            language_probability=1.0,
            duration=duration,
        )

    def unload_model(self) -> None:
        super().unload_model()
        gc.collect()
