# OpenAI Whisper API Backend
# Cloud-based transcription for computers without powerful local hardware

import os
import time
from typing import List, Tuple, Optional

from .base import WhisperBackend, TranscriptionSegment, TranscriptionResult


class OpenAIBackend(WhisperBackend):
    """
    Backend that uses OpenAI's cloud-based Whisper API for transcription.
    
    Supports two modes:
    - Transcription only: Uses whisper-1 or gpt-4o-transcribe
    - Transcription + Edit: Transcribes, then uses GPT to copy-edit the text
    """
    
    # Default edit prompt based on QuickWhisper
    DEFAULT_EDIT_PROMPT = """You are a copy editor. Your job is to clean up transcribed speech.

Rules:
- Fix grammar, punctuation, and spelling errors
- Remove filler words (um, uh, like, you know)
- Remove duplicate phrases and repetitions (when someone says the same thing twice)
- Keep the original meaning and tone
- Format as clean, readable text
- Do NOT add any commentary or explanations
- Output ONLY the edited transcription

Here is the transcription to clean up:"""

    def __init__(self):
        self._client = None
        self._api_key = None
        self._transcription_model = "whisper-1"
        self._edit_model = "gpt-4o-mini"
        self._edit_prompt = self.DEFAULT_EDIT_PROMPT
        self._language = "auto"
        self._current_device = "cloud"
    
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def is_available(self) -> Tuple[bool, str]:
        """Check if OpenAI backend is available (has API key configured)."""
        if self._api_key:
            return True, "OpenAI API configured"
        return False, "OpenAI API key not configured"
    
    def get_supported_devices(self) -> List[str]:
        """OpenAI runs in the cloud."""
        return ["cloud"]
    
    def get_supported_compute_types(self, device: str) -> List[str]:
        """Not applicable for cloud backend."""
        return ["default"]
    
    def configure(self, 
                  api_key: str,
                  transcription_model: str = "whisper-1",
                  edit_model: str = "gpt-4o-mini",
                  edit_prompt: str = None,
                  language: str = "auto") -> None:
        """
        Configure the OpenAI backend with API credentials and model settings.
        
        Args:
            api_key: OpenAI API key
            transcription_model: Model for transcription (whisper-1, gpt-4o-transcribe)
            edit_model: Model for copy-editing (gpt-4o-mini, gpt-4o, etc.)
            edit_prompt: Custom system prompt for editing
            language: Language code or "auto" for auto-detect
        """
        self._api_key = api_key
        self._transcription_model = transcription_model
        self._edit_model = edit_model
        self._edit_prompt = edit_prompt or self.DEFAULT_EDIT_PROMPT
        self._language = language
        
        # Initialize client
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
    
    def load_model(self,
                   model_key: str,
                   device: str,
                   compute_type: str,
                   model_path: Optional[str] = None,
                   progress_callback: Optional[callable] = None) -> None:
        """
        No-op for cloud backend - models are loaded server-side.
        Just validates configuration.
        """
        if not self._api_key:
            raise ValueError("OpenAI API key not configured. Call configure() first.")
        
        if not self._client:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
    
    def transcribe(self,
                   audio_path: str,
                   beam_size: int = 5,
                   vad_filter: bool = True) -> List[TranscriptionSegment]:
        """
        Transcribe audio using OpenAI's Whisper API.
        
        Args:
            audio_path: Path to the audio file
            beam_size: Ignored (API doesn't expose this)
            vad_filter: Ignored (API handles this automatically)
            
        Returns:
            List of TranscriptionSegment objects
        """
        if not self._client:
            raise ValueError("OpenAI client not initialized. Call configure() or load_model() first.")
        
        with open(audio_path, "rb") as audio_file:
            # Determine response format based on model type
            if "gpt" in self._transcription_model.lower():
                # GPT-4o transcribe returns plain text
                transcription = self._client.audio.transcriptions.create(
                    file=audio_file,
                    model=self._transcription_model,
                    language=None if self._language == "auto" else self._language,
                    response_format="text"
                )
                text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
            else:
                # Whisper-1 can return verbose JSON with timestamps
                transcription = self._client.audio.transcriptions.create(
                    file=audio_file,
                    model=self._transcription_model,
                    language=None if self._language == "auto" else self._language,
                    response_format="verbose_json"
                )
                # Extract text from response
                if hasattr(transcription, 'text'):
                    text = transcription.text.strip()
                elif isinstance(transcription, dict):
                    text = transcription.get("text", "").strip()
                else:
                    text = str(transcription).strip()
        
        # Return as single segment (API doesn't provide fine-grained timestamps in simple mode)
        return [TranscriptionSegment(text=text, start=0.0, end=0.0)]
    
    def transcribe_and_edit(self,
                            audio_path: str,
                            beam_size: int = 5,
                            vad_filter: bool = True) -> Tuple[str, str]:
        """
        Transcribe audio and then copy-edit with GPT.
        
        Args:
            audio_path: Path to the audio file
            beam_size: Ignored
            vad_filter: Ignored
            
        Returns:
            Tuple of (raw_transcription, edited_text)
        """
        # First, get the raw transcription
        segments = self.transcribe(audio_path, beam_size, vad_filter)
        raw_text = " ".join(seg.text for seg in segments).strip()
        
        if not raw_text:
            return "", ""
        
        # Then, edit with GPT
        edited_text = self._edit_with_gpt(raw_text)
        
        return raw_text, edited_text
    
    def _edit_with_gpt(self, text: str) -> str:
        """
        Send transcription to GPT for copy-editing.
        
        Args:
            text: Raw transcription text
            
        Returns:
            Edited/cleaned text
        """
        if not self._client:
            raise ValueError("OpenAI client not initialized.")
        
        try:
            response = self._client.chat.completions.create(
                model=self._edit_model,
                messages=[
                    {"role": "system", "content": self._edit_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=4000,
                temperature=0.3  # Lower temperature for more consistent editing
            )
            
            edited = response.choices[0].message.content
            return edited.strip() if edited else text
            
        except Exception as e:
            print(f"GPT editing failed: {e}")
            # Return original text if editing fails
            return text
    
    def transcribe_with_timing(self,
                               audio_path: str,
                               beam_size: int = 5,
                               vad_filter: bool = True) -> TranscriptionResult:
        """
        Transcribe and measure latency for cloud API.
        """
        start_time = time.perf_counter()
        
        segments = self.transcribe(audio_path, beam_size, vad_filter)
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        full_text = " ".join(seg.text for seg in segments).strip()
        
        return TranscriptionResult(
            segments=segments,
            text=full_text,
            latency_ms=latency_ms,
            backend_name=self.name,
            device="cloud"
        )
    
    def unload_model(self) -> None:
        """No-op for cloud backend."""
        pass
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if client is configured."""
        return self._client is not None and self._api_key is not None
    
    def get_model_map(self) -> dict:
        """Return available OpenAI transcription models."""
        return {
            "Whisper-1 (Fast, reliable)": "whisper-1",
            "GPT-4o Transcribe (Newest)": "gpt-4o-transcribe",
        }
    
    def get_edit_model_map(self) -> dict:
        """Return available GPT models for editing."""
        return {
            "GPT-4o Mini (Fast, cheap)": "gpt-4o-mini",
            "GPT-4o (Best quality)": "gpt-4o",
            "GPT-4 Turbo": "gpt-4-turbo",
        }
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the OpenAI API connection.
        
        Returns:
            Tuple of (success, message)
        """
        if not self._client:
            return False, "Client not configured"
        
        try:
            # Simple API call to verify credentials
            self._client.models.list()
            return True, "Connection successful!"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
