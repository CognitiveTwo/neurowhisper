"""OpenVINO GenAI backend for Intel Arc discrete GPUs, Intel Iris Xe iGPUs, and Intel Core Ultra NPUs on Linux."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .. import paths
from .base import TranscriptionResult, TranscriptionSegment, WhisperBackend

log = logging.getLogger("neurowhisper.backends.openvino")


class OpenVINOBackend(WhisperBackend):
    """Intel OpenVINO ``WhisperPipeline`` backend (``GPU``, ``NPU``, or ``CPU``)."""

    MODEL_MAPPING = {
        "large-v3": "OpenVINO/whisper-large-v3-fp16-ov",
        "large-v3-turbo": "OpenVINO/whisper-large-v3-fp16-ov",
        "distil-large-v3.5": "OpenVINO/distil-whisper-large-v3-fp16-ov",
        "medium": "OpenVINO/whisper-medium-fp16-ov",
        "small": "OpenVINO/whisper-small-fp16-ov",
        "base": "OpenVINO/whisper-base-fp16-ov",
        "tiny": "OpenVINO/whisper-tiny-fp16-ov",
    }

    def __init__(self) -> None:
        super().__init__()
        self._ov_device: str = "CPU"

    @property
    def backend_name(self) -> str:
        return f"OpenVINO ({self._ov_device})"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import openvino_genai  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def get_supported_devices(cls) -> List[str]:
        devices = ["cpu"]
        try:
            import openvino as ov

            core = ov.Core()
            avail = core.available_devices
            if any(d.startswith("GPU") for d in avail):
                devices.insert(0, "gpu")
            if any(d.startswith("NPU") for d in avail):
                devices.append("npu")
        except Exception:
            pass
        return devices

    @classmethod
    def check_intel_gpu(cls) -> bool:
        try:
            import openvino as ov

            core = ov.Core()
            return any(d.startswith("GPU") for d in core.available_devices)
        except Exception:
            return False

    @staticmethod
    def _download_ov_model(hf_repo: str) -> Path:
        from huggingface_hub import snapshot_download

        target_dir = paths.get_models_dir() / "openvino" / hf_repo.replace("/", "_")
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=hf_repo, local_dir=str(target_dir))
        return target_dir

    def load_model(self, model_name: str, device: str = "auto", compute_type: str = "auto") -> None:
        self.unload_model()
        import openvino as ov
        import openvino_genai as ov_genai

        core = ov.Core()
        avail = core.available_devices
        req = (device or "auto").strip().lower()

        if req == "auto":
            if any(d.startswith("GPU") for d in avail):
                self._ov_device = "GPU"
                self._device = "gpu"
            elif any(d.startswith("NPU") for d in avail):
                self._ov_device = "NPU"
                self._device = "npu"
            else:
                self._ov_device = "CPU"
                self._device = "cpu"
        elif req in ("gpu", "intel"):
            self._ov_device = "GPU"
            self._device = "gpu"
        elif req == "npu":
            self._ov_device = "NPU"
            self._device = "npu"
        else:
            self._ov_device = "CPU"
            self._device = "cpu"

        hf_repo = self.MODEL_MAPPING.get(model_name, f"OpenVINO/whisper-{model_name}-fp16-ov")
        model_path = Path(model_name)
        if not model_path.is_dir():
            model_path = self._download_ov_model(hf_repo)

        try:
            self._model = ov_genai.WhisperPipeline(str(model_path), device=self._ov_device)
        except Exception as exc:
            if self._ov_device != "CPU":
                log.warning("OpenVINO %s load failed (%s); falling back to CPU", self._ov_device, exc)
                self._ov_device = "CPU"
                self._device = "cpu"
                self._model = ov_genai.WhisperPipeline(str(model_path), device="CPU")
            else:
                raise

        self._model_name = model_name
        self._compute_type = compute_type

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

        gen_kwargs = {}
        if language and language != "auto":
            gen_kwargs["language"] = f"<|{language}|>"
        if not without_timestamps:
            gen_kwargs["return_timestamps"] = True

        result = self._model.generate(audio_f32.tolist(), **gen_kwargs)
        segments: List[TranscriptionSegment] = []

        if hasattr(result, "texts") and result.texts:
            text = " ".join(t.strip() for t in result.texts if t.strip())
            if hasattr(result, "chunks") and result.chunks:
                for chunk in result.chunks:
                    segments.append(
                        TranscriptionSegment(
                            start=getattr(chunk, "start_ts", 0.0),
                            end=getattr(chunk, "end_ts", duration),
                            text=getattr(chunk, "text", "").strip(),
                        )
                    )
        else:
            text = str(result).strip()

        if not segments and text:
            segments = [TranscriptionSegment(start=0.0, end=duration, text=text)]

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
