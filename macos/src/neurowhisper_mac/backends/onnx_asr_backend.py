"""ONNX Runtime speech recognition backend (Parakeet TDT 0.6B v3 with Apple CoreML acceleration)."""

from __future__ import annotations

import gc
import importlib.util
import inspect
import logging
import os
import sys
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import LEGACY_MODEL_ALIASES, MODEL_MAP
from .base import TranscriptionSegment, WhisperBackend, prepare_mono_float32

logger = logging.getLogger(__name__)

ONNX_MODEL_NAME_MAP: Dict[str, str] = {
    "parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
    "nemo-parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
    "canary-1b-v2": "nemo-canary-1b-v2",
    "nemo-canary-1b-v2": "nemo-canary-1b-v2",
    "base": "whisper-base",
    "whisper-base": "whisper-base",
}


def _get_ort_and_onnx_asr():
    ort_mod = sys.modules.get("onnxruntime")
    asr_mod = sys.modules.get("onnx_asr")
    if ort_mod is None:
        import onnxruntime as ort_mod  # type: ignore
    if asr_mod is None:
        import onnx_asr as asr_mod  # type: ignore
    return ort_mod, asr_mod


class OnnxAsrBackend(WhisperBackend):
    """ONNX ASR backend prioritizing Apple ``CoreMLExecutionProvider`` when available."""

    name = "onnx-asr"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._providers: List[str] = []
        self._infer_lock = threading.Lock()

    @classmethod
    def is_available(cls) -> bool:
        if "onnxruntime" in sys.modules and "onnx_asr" in sys.modules:
            return sys.modules["onnxruntime"] is not None and sys.modules["onnx_asr"] is not None
        return (
            importlib.util.find_spec("onnxruntime") is not None
            and importlib.util.find_spec("onnx_asr") is not None
        )

    @classmethod
    def has_coreml_provider(cls) -> bool:
        try:
            ort, _ = _get_ort_and_onnx_asr()
            providers = list(ort.get_available_providers() or [])
            return "CoreMLExecutionProvider" in providers
        except Exception:
            return False

    @classmethod
    def get_supported_devices(cls) -> List[str]:
        devices = ["cpu"]
        if cls.has_coreml_provider():
            devices.insert(0, "coreml")
        return devices

    @classmethod
    def resolve_onnx_model_id(cls, model_name: str) -> str:
        raw = (model_name or "").strip()
        short = MODEL_MAP.get(raw) or LEGACY_MODEL_ALIASES.get(raw) or raw
        if short in ONNX_MODEL_NAME_MAP:
            return ONNX_MODEL_NAME_MAP[short]
        raise ValueError(
            f"Model '{model_name}' is not supported by ONNX ASR. "
            "Select 'Parakeet TDT 0.6B v3' or switch Acceleration to 'Auto'."
        )

    @classmethod
    def select_providers(cls, device: str = "auto") -> Tuple[List[str], str]:
        """Return ``(providers_list, resolved_device)`` without clobbering ``auto`` to ``cpu``."""
        req = (device or "auto").strip().lower()
        try:
            ort, _ = _get_ort_and_onnx_asr()
            available = set(ort.get_available_providers() or [])
        except Exception:
            available = {"CPUExecutionProvider"}

        wants_coreml = req in ("auto", "coreml", "mps", "gpu")
        if wants_coreml and "CoreMLExecutionProvider" in available:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"], "coreml"

        return ["CPUExecutionProvider"], "cpu"

    def load_model(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Optional[str] = None,
    ) -> None:
        if not self.is_available():
            raise RuntimeError("onnx-asr and onnxruntime are required for the ONNX ASR backend.")

        _, onnx_asr = _get_ort_and_onnx_asr()
        onnx_id = self.resolve_onnx_model_id(model_name)
        providers, resolved_device = self.select_providers(device)

        load_kwargs: Dict[str, object] = {"providers": providers}
        prev_hf_cache = os.environ.get("HF_HUB_CACHE")

        try:
            sig = inspect.signature(onnx_asr.load_model)
            params = sig.parameters
        except (TypeError, ValueError):
            params = {}

        if download_root:
            os.makedirs(download_root, exist_ok=True)
            if "path" in params and os.path.isdir(os.path.join(download_root, onnx_id)):
                load_kwargs["path"] = os.path.join(download_root, onnx_id)
            os.environ["HF_HUB_CACHE"] = str(download_root)

        try:
            try:
                loaded = onnx_asr.load_model(onnx_id, **load_kwargs)
            except Exception as first_exc:
                # If CoreML compilation failed on a specific op, retry cleanly on CPU
                if resolved_device == "coreml":
                    logger.warning(
                        "CoreMLExecutionProvider failed (%s); falling back to CPUExecutionProvider",
                        first_exc,
                    )
                    providers = ["CPUExecutionProvider"]
                    resolved_device = "cpu"
                    load_kwargs["providers"] = providers
                    loaded = onnx_asr.load_model(onnx_id, **load_kwargs)
                else:
                    raise RuntimeError(
                        f"Failed to load ONNX ASR model '{onnx_id}': {first_exc}"
                    ) from first_exc
        finally:
            if download_root:
                if prev_hf_cache is None:
                    os.environ.pop("HF_HUB_CACHE", None)
                else:
                    os.environ["HF_HUB_CACHE"] = prev_hf_cache

        if hasattr(loaded, "with_timestamps"):
            try:
                loaded = loaded.with_timestamps()
            except Exception:
                pass

        with self._infer_lock:
            self._model = loaded
            self._providers = providers
            self._current_model = model_name
            self._current_device = resolved_device
            self._compute_type = compute_type

        logger.info(
            "Loaded ONNX ASR model %s (%s) on %s (%s)",
            model_name,
            onnx_id,
            resolved_device,
            providers,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        vad_filter: bool = True,
        **kwargs,
    ) -> List[TranscriptionSegment]:
        if self._model is None:
            raise RuntimeError("ONNX ASR model is not loaded.")

        mono = prepare_mono_float32(audio, sample_rate=sample_rate, target_sr=16000)
        if mono.size == 0:
            return []

        rec_kwargs: Dict[str, object] = {"sample_rate": 16000}
        if language and language.strip().lower() not in ("auto", ""):
            try:
                sig = inspect.signature(self._model.recognize)
                if "language" in sig.parameters:
                    rec_kwargs["language"] = language.strip().lower()
            except Exception:
                pass

        with self._infer_lock:
            raw_res = self._model.recognize(mono, **rec_kwargs)

        return self._normalize_output(raw_res, duration=float(mono.size) / 16000.0)

    @staticmethod
    def _normalize_output(raw_res: object, duration: float) -> List[TranscriptionSegment]:
        if raw_res is None:
            return []
        if isinstance(raw_res, str):
            txt = raw_res.strip()
            return [TranscriptionSegment(text=txt, start=0.0, end=duration)] if txt else []

        # Timestamped result object from .with_timestamps()
        if hasattr(raw_res, "segments") and getattr(raw_res, "segments"):
            out: List[TranscriptionSegment] = []
            for seg in getattr(raw_res, "segments"):
                txt = str(getattr(seg, "text", "") or "").strip()
                if txt:
                    out.append(
                        TranscriptionSegment(
                            text=txt,
                            start=float(getattr(seg, "start", 0.0) or 0.0),
                            end=float(getattr(seg, "end", duration) or duration),
                        )
                    )
            if out:
                return out

        if hasattr(raw_res, "text"):
            txt = str(getattr(raw_res, "text") or "").strip()
            return [TranscriptionSegment(text=txt, start=0.0, end=duration)] if txt else []

        if isinstance(raw_res, list):
            out = []
            for item in raw_res:
                if isinstance(item, str) and item.strip():
                    out.append(TranscriptionSegment(text=item.strip(), start=0.0, end=duration))
                elif hasattr(item, "text") and str(item.text).strip():
                    out.append(
                        TranscriptionSegment(
                            text=str(item.text).strip(),
                            start=float(getattr(item, "start", 0.0) or 0.0),
                            end=float(getattr(item, "end", duration) or duration),
                        )
                    )
            return out

        txt = str(raw_res).strip()
        return [TranscriptionSegment(text=txt, start=0.0, end=duration)] if txt else []

    def unload_model(self) -> None:
        with self._infer_lock:
            self._model = None
            self._providers = []
            self._current_model = None
        gc.collect()

    def is_model_loaded(self) -> bool:
        return self._model is not None
