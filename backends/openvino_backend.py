# OpenVINO Backend
# Intel GPU/NPU acceleration using OpenVINO and Optimum Intel

import gc
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from app_config import AUDIO_RUNTIME, OPENVINO_MODEL_MAP
from .base import TranscriptionSegment, WhisperBackend

_WINDOW_SECONDS = 30.0


def _is_offline_mode() -> bool:
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _resolve_local_ov_dir(model_path: Optional[str]) -> Optional[str]:
    if not model_path or not os.path.isdir(model_path):
        return None
    for root, dirs, files in os.walk(model_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        xml_stems = [os.path.splitext(name)[0] for name in files if name.endswith(".xml")]
        if xml_stems:
            file_set = set(files)
            if all(
                f"{stem}.bin" in file_set
                and os.path.getsize(os.path.join(root, f"{stem}.bin")) > 0
                for stem in xml_stems
            ):
                return root
    return None


class OpenVINOBackend(WhisperBackend):
    """
    Backend using OpenVINO for Intel GPU/NPU acceleration.
    Uses Hugging Face Optimum Intel for optimized Whisper models.
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._current_model_key = None
        self._device = None
        self._current_device = None

    @property
    def name(self) -> str:
        return "openvino"

    @property
    def is_available(self) -> Tuple[bool, str]:
        """Check if an Intel GPU/NPU is present and the required OpenVINO
        stack (optimum.intel, transformers, librosa, torch) can be imported."""
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices

            has_gpu = any("GPU" in d for d in devices)
            has_npu = any("NPU" in d for d in devices)

            if not (has_gpu or has_npu):
                return False, "No Intel GPU or NPU detected"

            try:
                import optimum.intel  # noqa: F401
                import transformers  # noqa: F401
                import librosa  # noqa: F401
                import torch  # noqa: F401
            except ImportError as e:
                return False, (
                    "OpenVINO dependencies not installed "
                    f"(optimum.intel/transformers/librosa/torch): {e}"
                )

            device_str = []
            if has_gpu:
                device_str.append("Intel GPU")
            if has_npu:
                device_str.append("Intel NPU")
            return True, f"OpenVINO with {', '.join(device_str)} available"

        except ImportError:
            return False, "OpenVINO not installed. Install with: pip install openvino optimum[openvino]"
        except Exception as e:
            return False, f"OpenVINO error: {e}"

    def _get_best_device(self) -> str:
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            if any("GPU" in d for d in devices):
                return "GPU"
            if any("NPU" in d for d in devices):
                return "NPU"
            return "CPU"
        except Exception:
            return "CPU"

    def get_supported_devices(self) -> List[str]:
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            relevant = []
            if any("GPU" in d for d in devices):
                relevant.append("GPU")
            if any("NPU" in d for d in devices):
                relevant.append("NPU")
            if any("CPU" in d for d in devices) or not relevant:
                relevant.append("CPU")
            return relevant
        except Exception:
            return ["CPU"]

    def get_supported_compute_types(self, device: str) -> List[str]:
        return ["auto", "fp16", "int8"]

    def get_model_map(self) -> dict:
        return OPENVINO_MODEL_MAP.copy()

    def load_model(
        self,
        model_key: str,
        device: str = "auto",
        compute_type: str = "auto",
        model_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        self.unload_model()

        def _safe_progress(status: str) -> None:
            if progress_callback:
                try:
                    progress_callback(status)
                except Exception:
                    pass

        try:
            from optimum.intel import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(
                "OpenVINO dependencies not installed. "
                "Install with: pip install openvino optimum[openvino] transformers"
            ) from e

        model_map = self.get_model_map()
        model_id = model_map.get(model_key, "OpenVINO/whisper-small-fp16-ov")

        dev_norm = (device or "auto").strip().upper()
        if dev_norm in ("AUTO", "GPU"):
            ov_device = self._get_best_device()
        elif dev_norm == "NPU":
            ov_device = "NPU"
        elif dev_norm == "CPU":
            ov_device = "CPU"
        else:
            ov_device = dev_norm

        _safe_progress("start")
        local_ov_dir = _resolve_local_ov_dir(model_path)
        offline = _is_offline_mode()

        if local_ov_dir is None and offline:
            raise RuntimeError(
                f"Model '{model_id}' is not downloaded yet and there is no internet connection."
            )
        if local_ov_dir is None:
            _safe_progress("downloading")

        source = local_ov_dir or model_id
        cache_kw = {"cache_dir": model_path} if (model_path and local_ov_dir is None) else {}
        local_only = local_ov_dir is not None or offline

        try:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    source, local_files_only=local_only, **cache_kw
                )
            except Exception:
                # Fallback for older OpenVINO repos missing tokenizer files
                fallback_proc = (
                    model_id.replace("OpenVINO/", "openai/")
                    .replace("-fp16-ov", "")
                    .replace("-int8-ov", "")
                )
                self._processor = AutoProcessor.from_pretrained(
                    fallback_proc, local_files_only=offline, **cache_kw
                )

            self._model = OVModelForSpeechSeq2Seq.from_pretrained(
                source,
                device=ov_device,
                local_files_only=local_only,
                **cache_kw,
            )
        except Exception as exc:
            if local_ov_dir is None and offline:
                raise RuntimeError(
                    f"Model '{model_id}' is not downloaded yet and there is no internet connection."
                ) from exc
            raise

        self._device = ov_device
        self._current_device = ov_device.lower()
        self._current_model_key = model_key
        _safe_progress("loaded")

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = AUDIO_RUNTIME["sample_rate"],
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        if not self._model or not self._processor:
            raise RuntimeError("No model loaded. Call load_model() first.")

        import torch

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1).astype(np.float32, copy=False)
        waveform = np.ascontiguousarray(waveform.reshape(-1), dtype=np.float32)
        if waveform.size == 0 or sample_rate <= 0:
            return []

        target_sr = AUDIO_RUNTIME["sample_rate"]
        if sample_rate != target_sr:
            target_len = max(1, int(round(waveform.size * float(target_sr) / float(sample_rate))))
            x_old = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            waveform = np.interp(x_new, x_old, waveform).astype(np.float32)

        window_samples = int(_WINDOW_SECONDS * target_sr)
        gen_kwargs = {"max_length": 448, "num_beams": beam_size}
        lang = (language or "").strip().lower()
        if lang and lang != "auto":
            gen_kwargs["language"] = lang

        segments: List[TranscriptionSegment] = []
        for offset in range(0, waveform.size, window_samples):
            chunk = waveform[offset : offset + window_samples]
            if chunk.size == 0:
                continue
            inputs = self._processor(chunk, sampling_rate=target_sr, return_tensors="pt")
            with torch.no_grad():
                try:
                    generated_ids = self._model.generate(inputs.input_features, **gen_kwargs)
                except TypeError:
                    generated_ids = self._model.generate(
                        inputs.input_features, max_length=448, num_beams=beam_size
                    )
            text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if text:
                start_s = float(offset) / float(target_sr)
                end_s = float(offset + chunk.size) / float(target_sr)
                segments.append(TranscriptionSegment(text=text, start=start_s, end=end_s))

        return segments

    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        if not self._model or not self._processor:
            raise RuntimeError("No model loaded. Call load_model() first.")

        try:
            import librosa
        except ImportError as e:
            raise ImportError("librosa required for file-based audio loading") from e

        audio, sr = librosa.load(audio_path, sr=AUDIO_RUNTIME["sample_rate"])
        return self.transcribe_audio(
            audio, sample_rate=int(sr), beam_size=beam_size, vad_filter=vad_filter, language=language
        )

    def unload_model(self) -> None:
        if self._model is not None or self._processor is not None:
            self._model = None
            self._processor = None
            gc.collect()
        self._current_model_key = None
        self._device = None
        self._current_device = None

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        return self._model
