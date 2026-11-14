"""ONNX ASR Backend Implementation (`onnx-asr` + `onnxruntime`).

Runs modern speech-to-text models such as NVIDIA Parakeet TDT 0.6B v3
(25 languages incl. English & German, punctuation + capitalization) and
Canary 1B v2 directly via ONNX Runtime without requiring PyTorch or NeMo.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from app_config import LEGACY_MODEL_ALIASES, MODEL_MAP
from .base import TranscriptionSegment, WhisperBackend

log = logging.getLogger("neurowhisper.onnx_asr")

_CONNECTION_ERROR_CLASS_NAMES = frozenset({
    "LocalEntryNotFoundError",
    "OfflineModeIsEnabled",
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "MaxRetryError",
    "NewConnectionError",
    "NameResolutionError",
    "ProxyError",
    "SSLError",
})

_CONNECTION_ERROR_PHRASES = (
    "cannot find the requested files in the local cache",
    "outgoing traffic has been disabled",
    "offline mode is enabled",
    "failed to establish a new connection",
    "max retries exceeded",
    "temporary failure in name resolution",
    "name or service not known",
    "getaddrinfo failed",
    "connection timed out",
    "read timed out",
    "network is unreachable",
    "no address associated with hostname",
)

# Max safe single-window audio length for Conformer/TDT encoders without VAD
_MAX_DIRECT_SECONDS = 20.0
_SUPPORTED_SAMPLE_RATES = frozenset({8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000})


def _sanitize_no_proxy_value(raw: str) -> str:
    """Strip brackets around bare IPv6 host entries (e.g. ``[::1]`` -> ``::1``)
    that crash ``httpx``'s ``get_environment_proxies()`` with ``InvalidURL``."""
    parts: List[str] = []
    seen = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if "://" not in item and item.startswith("[") and item.endswith("]"):
            item = item[1:-1]
        if item not in seen:
            seen.add(item)
            parts.append(item)
    return ",".join(parts)


def _sanitize_no_proxy_env() -> None:
    """Ensure NO_PROXY/no_proxy contain bare IPv6 literals accepted by httpx."""
    for var in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(var)
        if not raw or "[" not in raw:
            continue
        updated = _sanitize_no_proxy_value(raw)
        if updated != raw:
            os.environ[var] = updated


def _is_offline_mode() -> bool:
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _iter_exception_chain(exc: BaseException):
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _is_connection_error(exc: BaseException) -> bool:
    for link in _iter_exception_chain(exc):
        if type(link).__name__ in _CONNECTION_ERROR_CLASS_NAMES:
            return True
        if isinstance(link, (ConnectionError, TimeoutError)):
            return True
        msg = str(link).lower()
        if any(phrase in msg for phrase in _CONNECTION_ERROR_PHRASES):
            return True
    return False


class OnnxAsrBackend(WhisperBackend):
    """Lightweight ONNX Runtime ASR backend powered by ``onnx-asr``.

    Supports NVIDIA Parakeet TDT 0.6B v3 (25 European languages incl. EN & DE),
    Parakeet TDT 0.6B v2 (English), Canary 1B v2, and Whisper Base INT8 on
    CPU, CUDA, DirectML, and CoreML with zero PyTorch/NeMo dependency.
    """

    # Native ONNX ASR models (safe to store directly in their own model_path)
    NATIVE_MODELS = {
        "parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
        "nemo-parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
        "nvidia/parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
        "istupakov/parakeet-tdt-0.6b-v3-onnx": "nemo-parakeet-tdt-0.6b-v3",
        "parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
        "nemo-parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
        "nvidia/parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
        "istupakov/parakeet-tdt-0.6b-v2-onnx": "nemo-parakeet-tdt-0.6b-v2",
        "canary-1b-v2": "nemo-canary-1b-v2",
        "nemo-canary-1b-v2": "nemo-canary-1b-v2",
        "nvidia/canary-1b-v2": "nemo-canary-1b-v2",
        "istupakov/canary-1b-v2-onnx": "nemo-canary-1b-v2",
        "whisper-base": "whisper-base",
    }

    # Fallback mapping when a Whisper model key is paired with the ONNX ASR backend
    FALLBACK_MODELS = {
        "base": "whisper-base",
        "distil-large-v3.5": "nemo-parakeet-tdt-0.6b-v2",
        "small": "nemo-parakeet-tdt-0.6b-v3",
        "medium": "nemo-parakeet-tdt-0.6b-v3",
        "large-v3": "nemo-parakeet-tdt-0.6b-v3",
        "large-v3-turbo": "nemo-parakeet-tdt-0.6b-v3",
    }

    def __init__(self):
        self._asr_model = None
        self._vad_model = None
        self._asr_name: Optional[str] = None
        self._current_device: str = "cpu"
        self._current_compute_type: str = "int8"
        self._current_model_key: Optional[str] = None
        self._model_loaded: bool = False

    @property
    def name(self) -> str:
        return "onnx-asr"

    @property
    def is_available(self) -> Tuple[bool, str]:
        try:
            import onnx_asr  # noqa: F401
            import onnxruntime as ort
            providers = ort.get_available_providers()
            accel = [
                p.replace("ExecutionProvider", "")
                for p in providers
                if p not in ("CPUExecutionProvider", "AzureExecutionProvider")
            ]
            if accel:
                return True, f"ONNX ASR available ({', '.join(accel)}, CPU)"
            return True, "ONNX ASR available (CPU)"
        except ImportError as exc:
            return False, f"onnx-asr not installed ({exc}). Run: pip install onnx-asr"
        except Exception as exc:
            return False, f"ONNX ASR initialization check failed: {exc}"

    def get_supported_devices(self) -> List[str]:
        devices = ["cpu"]
        try:
            import onnxruntime as ort
            providers = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in providers:
                devices.insert(0, "cuda")
            if "DmlExecutionProvider" in providers:
                devices.insert(0, "dml")
            if "CoreMLExecutionProvider" in providers:
                devices.insert(0, "coreml")
        except Exception:
            pass
        return devices

    def get_supported_compute_types(self, device: str) -> List[str]:
        return ["int8", "float32"]

    def get_model_map(self) -> dict:
        return {
            "Parakeet TDT 0.6B v3 (Ultra-fast ONNX, 25 langs, ~650MB)": "parakeet-tdt-0.6b-v3",
            "Parakeet TDT 0.6B v2 (Ultra-fast English, ~650MB)": "parakeet-tdt-0.6b-v2",
            "Canary 1B v2 (Multilingual + translation, ~1GB)": "canary-1b-v2",
            "Base (Minimal resources, quick)": "base",
        }

    def _resolve_asr_name_and_dir(
        self, model_key: str, model_path: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        """Map ``model_key`` (friendly UI label or short ID) to an ``onnx-asr``
        model identifier and cache directory."""
        own_map = self.get_model_map()
        normalized = (
            own_map.get(model_key)
            or MODEL_MAP.get(model_key)
            or LEGACY_MODEL_ALIASES.get(model_key)
            or model_key
        )

        if normalized in self.NATIVE_MODELS:
            return self.NATIVE_MODELS[normalized], model_path

        if normalized in self.FALLBACK_MODELS:
            asr_name = self.FALLBACK_MODELS[normalized]
            log.info(
                "[OnnxAsr] Mapping model '%s' -> '%s' for ONNX ASR backend",
                model_key,
                asr_name,
            )
            if model_path:
                models_root = os.path.dirname(os.path.abspath(model_path))
                return asr_name, os.path.join(models_root, f"onnx_{asr_name}")
            return asr_name, None

        # Pass through custom HuggingFace repo IDs or onnx-asr model names
        return normalized, model_path

    @staticmethod
    def _select_providers(device: str) -> Tuple[List[str], str]:
        """Select ONNX Runtime execution providers and normalized device name."""
        import onnxruntime as ort

        available = list(ort.get_available_providers())
        dev = (device or "auto").lower()

        if dev == "cpu":
            return ["CPUExecutionProvider"], "cpu"

        if dev == "cuda":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
            log.warning("[OnnxAsr] CUDAExecutionProvider unavailable; falling back to CPU.")
            return ["CPUExecutionProvider"], "cpu"

        if dev in ("dml", "directml", "gpu"):
            if "DmlExecutionProvider" in available:
                return ["DmlExecutionProvider", "CPUExecutionProvider"], "dml"
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
            return ["CPUExecutionProvider"], "cpu"

        if dev in ("coreml", "mps"):
            if "CoreMLExecutionProvider" in available:
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"], "coreml"
            return ["CPUExecutionProvider"], "cpu"

        # Auto-select fast ready-to-run providers (skip TensorRT cold engine builds & Azure)
        for provider, norm_dev in (
            ("CUDAExecutionProvider", "cuda"),
            ("DmlExecutionProvider", "dml"),
            ("CoreMLExecutionProvider", "coreml"),
        ):
            if provider in available:
                return [provider, "CPUExecutionProvider"], norm_dev
        return ["CPUExecutionProvider"], "cpu"

    @staticmethod
    def _is_onnx_model_present(local_dir: Optional[str], asr_name: str, quantization: Optional[str]) -> bool:
        if not local_dir or not os.path.isdir(local_dir):
            return False
        try:
            from onnx_asr.loader import create_asr_resolver

            resolver = create_asr_resolver(asr_name, local_dir=local_dir, offline=True)
            resolver._resolve_model_files(Path(local_dir), quantization)
            return True
        except Exception:
            return False

    def _resolve_quantization(
        self, compute_type: str, local_dir: Optional[str], asr_name: str
    ) -> Optional[str]:
        ct = (compute_type or "auto").lower()
        if ct in ("float32", "fp32"):
            return None
        if ct == "int8":
            return "int8"
        # For 'auto' / 'float16' / 'int8_float16': reuse local FP32 if already cached,
        # otherwise prefer compact & fast INT8 (~650 MB vs ~2.5 GB, ~3x faster on CPU).
        if not self._is_onnx_model_present(local_dir, asr_name, "int8") and self._is_onnx_model_present(
            local_dir, asr_name, None
        ):
            return None
        return "int8"

    def load_model(
        self,
        model_key: str,
        device: str = "auto",
        compute_type: str = "auto",
        model_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self._model_loaded:
            self.unload_model()

        def _safe_progress(status: str) -> None:
            if progress_callback:
                try:
                    progress_callback(status)
                except Exception:
                    pass

        _sanitize_no_proxy_env()
        import onnxruntime as ort
        from onnx_asr.loader import Manager

        asr_name, target_dir = self._resolve_asr_name_and_dir(model_key, model_path)
        providers, norm_device = self._select_providers(device)
        quantization = self._resolve_quantization(compute_type, target_dir, asr_name)
        offline = _is_offline_mode()

        _safe_progress("start")
        already_present = self._is_onnx_model_present(target_dir, asr_name, quantization)
        if not already_present:
            if offline and target_dir is not None:
                raise RuntimeError(
                    f"Model '{asr_name}' is not downloaded yet and there is no internet connection."
                )
            if not offline:
                _safe_progress("downloading")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if norm_device == "cpu":
            cpu_count = os.cpu_count() or 4
            sess_options.intra_op_num_threads = max(1, min(cpu_count, 8))
            sess_options.inter_op_num_threads = 1
        elif norm_device == "dml":
            sess_options.enable_mem_pattern = False
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        manager = Manager(sess_options=sess_options, providers=providers)

        try:
            self._asr_model = manager.create_asr(
                asr_name,
                local_dir=target_dir,
                quantization=quantization,
                offline=offline,
            )
        except Exception as exc:
            if offline or _is_connection_error(exc):
                raise RuntimeError(
                    f"Model '{asr_name}' is not downloaded yet and Hugging Face could not "
                    "be reached. Connect to the internet once to download it, or choose an "
                    "already-downloaded model in Settings."
                ) from exc
            raise

        # Load lightweight Silero ONNX VAD (~1.8 MB) for segmenting >20s audio cleanly.
        vad_dir: Optional[str] = None
        if target_dir:
            vad_dir = os.path.join(os.path.dirname(os.path.abspath(target_dir)), "onnx_silero_vad")
        try:
            self._vad_model = manager.create_vad(
                "silero",
                local_dir=vad_dir,
                offline=offline,
            )
        except Exception as vad_exc:
            log.warning(
                "[OnnxAsr] Silero VAD unavailable (%s); falling back to fixed windowing for >20s audio.",
                vad_exc,
            )
            self._vad_model = None

        self._asr_name = asr_name
        self._current_device = norm_device
        self._current_compute_type = quantization or "float32"
        self._current_model_key = model_key
        self._model_loaded = True

        log.info(
            "[OnnxAsr] Loaded '%s' (quant=%s, device=%s, providers=%s)",
            asr_name,
            self._current_compute_type,
            norm_device,
            providers,
        )
        _safe_progress("loaded")

    def _recognize_options(self, language: Optional[str]) -> dict:
        lang = (language or "").strip().lower()
        if not lang or lang == "auto":
            if self._asr_name and "canary" in self._asr_name:
                return {"pnc": True}
            return {}
        if self._asr_name and "canary" in self._asr_name:
            return {"language": lang, "target_language": lang, "pnc": True}
        if self._asr_name and "whisper" in self._asr_name:
            return {"language": lang}
        # Parakeet TDT v2/v3 auto-detects language and punctuation natively
        return {}

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        if not self._model_loaded or self._asr_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1).astype(np.float32, copy=False)
        waveform = np.ascontiguousarray(waveform.reshape(-1), dtype=np.float32)

        if waveform.size == 0 or sample_rate <= 0:
            return []

        if sample_rate not in _SUPPORTED_SAMPLE_RATES:
            new_len = max(1, int(round(waveform.size * 16000.0 / float(sample_rate))))
            waveform = np.interp(
                np.linspace(0.0, 1.0, new_len, endpoint=False),
                np.linspace(0.0, 1.0, waveform.size, endpoint=False),
                waveform,
            ).astype(np.float32)
            sample_rate = 16000

        duration = float(waveform.size) / float(sample_rate)
        rec_kwargs = self._recognize_options(language)

        # Use Silero ONNX VAD when requested OR when audio exceeds encoder window
        use_vad = (vad_filter or duration > _MAX_DIRECT_SECONDS) and self._vad_model is not None
        if use_vad:
            vad_adapter = self._asr_model.with_vad(
                self._vad_model,
                max_speech_duration_s=_MAX_DIRECT_SECONDS,
            )
            try:
                raw_segments = list(
                    vad_adapter.recognize(waveform, sample_rate=sample_rate, **rec_kwargs)
                )
            except KeyError:
                # Unsupported language token -> retry with auto/default options (preserving pnc)
                raw_segments = list(
                    vad_adapter.recognize(
                        waveform, sample_rate=sample_rate, **self._recognize_options(None)
                    )
                )

            out: List[TranscriptionSegment] = []
            for seg in raw_segments:
                txt = (seg.text or "").strip()
                if txt:
                    out.append(
                        TranscriptionSegment(
                            text=txt,
                            start=float(getattr(seg, "start", 0.0)),
                            end=float(getattr(seg, "end", duration)),
                        )
                    )
            return out

        # Fixed-window fallback if >20s and Silero VAD was unavailable offline
        if duration > _MAX_DIRECT_SECONDS:
            window_samples = int(_MAX_DIRECT_SECONDS * sample_rate)
            segments: List[TranscriptionSegment] = []
            for offset in range(0, waveform.size, window_samples):
                chunk = waveform[offset : offset + window_samples]
                if chunk.size == 0:
                    continue
                start_s = float(offset) / float(sample_rate)
                end_s = float(offset + chunk.size) / float(sample_rate)
                txt = self._recognize_single(chunk, sample_rate, rec_kwargs)
                if txt:
                    segments.append(TranscriptionSegment(text=txt, start=start_s, end=end_s))
            return segments

        text = self._recognize_single(waveform, sample_rate, rec_kwargs)
        if not text:
            return []
        return [TranscriptionSegment(text=text, start=0.0, end=duration)]

    def _recognize_single(self, waveform: np.ndarray, sample_rate: int, rec_kwargs: dict) -> str:
        try:
            res = self._asr_model.recognize(waveform, sample_rate=sample_rate, **rec_kwargs)
        except KeyError:
            res = self._asr_model.recognize(
                waveform, sample_rate=sample_rate, **self._recognize_options(None)
            )
        return (res or "").strip() if isinstance(res, str) else ""

    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        if not self._model_loaded or self._asr_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        rec_kwargs = self._recognize_options(language)

        # Try reading integer PCM WAV first (zero ffmpeg/PyAV overhead)
        try:
            from onnx_asr.utils import read_wav

            raw_audio, sr = read_wav(str(audio_path))
            if raw_audio.ndim > 1:
                raw_audio = raw_audio.mean(axis=-1)
            return self.transcribe_audio(
                raw_audio.astype(np.float32, copy=False),
                sample_rate=int(sr),
                beam_size=beam_size,
                vad_filter=vad_filter,
                language=language,
            )
        except Exception:
            pass

        # Decode IEEE-float WAV / MP3 / M4A / FLAC via PyAV (faster_whisper.audio)
        try:
            from faster_whisper.audio import decode_audio

            decoded = decode_audio(str(audio_path), sampling_rate=16000)
            return self.transcribe_audio(
                np.asarray(decoded, dtype=np.float32),
                sample_rate=16000,
                beam_size=beam_size,
                vad_filter=vad_filter,
                language=language,
            )
        except Exception:
            # Direct file fallback via onnx_asr adapter
            if vad_filter and self._vad_model is not None:
                vad_adapter = self._asr_model.with_vad(
                    self._vad_model, max_speech_duration_s=_MAX_DIRECT_SECONDS
                )
                try:
                    segs = list(vad_adapter.recognize(audio_path, channel="mean", **rec_kwargs))
                except KeyError:
                    segs = list(
                        vad_adapter.recognize(
                            audio_path, channel="mean", **self._recognize_options(None)
                        )
                    )
                return [
                    TranscriptionSegment(
                        text=s.text.strip(),
                        start=float(getattr(s, "start", 0.0)),
                        end=float(getattr(s, "end", 0.0)),
                    )
                    for s in segs
                    if s.text and s.text.strip()
                ]
            try:
                res = self._asr_model.recognize(audio_path, channel="mean", **rec_kwargs)
            except KeyError:
                res = self._asr_model.recognize(
                    audio_path, channel="mean", **self._recognize_options(None)
                )
            txt = (res or "").strip() if isinstance(res, str) else ""
            return [TranscriptionSegment(text=txt, start=0.0, end=0.0)] if txt else []

    def unload_model(self) -> None:
        if self._asr_model is not None or self._vad_model is not None:
            self._asr_model = None
            self._vad_model = None
            gc.collect()
            log.info("[OnnxAsr] Model unloaded")
        self._model_loaded = False

    @property
    def is_model_loaded(self) -> bool:
        return self._model_loaded and self._asr_model is not None

    @property
    def model(self):
        return self._asr_model
