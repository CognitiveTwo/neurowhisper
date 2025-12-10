import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import keyboard
from faster_whisper import WhisperModel
import queue
import threading
import time
import sys
import os

# --- CONFIGURATION ---
HOTKEY_TOGGLE = 'F9'       # Key to start/stop
MODEL_SIZE = "large-v3"    # Best accuracy for RTX 3090
DEVICE = "cuda"            # GPU acceleration
COMPUTE_TYPE = "float16"   # Standard for RTX 30 series
SILENCE_THRESHOLD = 0.015  # Volume threshold (adjust if your mic is noisy)

# --- ROBUSTNESS SETTINGS ---
# INCREASED: We now wait 2.5 seconds of silence before finalizing a sentence.
# This prevents it from putting a period (.) every time you take a breath.
SILENCE_DURATION = 2.5     

# INCREASED: We allow up to 30 seconds of continuous speech before forcing a write.
MAX_CHUNK_DURATION = 30.0  
# ---------------------------

print(f"Loading {MODEL_SIZE} model on {DEVICE}...")

try:
    # Attempt to load on GPU
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"SUCCESS: Model '{MODEL_SIZE}' loaded on GPU!")
    
except Exception as e:
    print(f"\n[ERROR] GPU failed to load. Error: {e}")
    print("Please ensure you have the NVIDIA cuDNN 9 DLLs in this folder.")
    print("Press Enter to exit...")
    input()
    sys.exit(1)

# Global queues and flags
audio_queue = queue.Queue()
is_recording = False

def process_audio_chunk(audio_data):
    """Takes the recorded audio data and transcribes it."""
    if not audio_data: 
        return
    
    # Flatten audio data
    audio_np = np.concatenate(audio_data, axis=0).flatten()
    
    # Skip very short noise (< 0.5s) to avoid processing clicks
    if len(audio_np) < 16000 * 0.5:
        return

    temp_filename = "temp_live.wav"
    wav.write(temp_filename, 16000, audio_np)

    try:
        segments, info = model.transcribe(
            temp_filename, 
            beam_size=5, 
            language=None,                  # Auto-detect language (German/English mixed)
            vad_filter=True,                # Filter out silence/hallucinations
            vad_parameters=dict(min_silence_duration_ms=1000), # More aggressive VAD
            condition_on_previous_text=False # Prevent loops
        )
        
        text = "".join([s.text for s in segments]).strip()
        
        # Safety Net: Filter out common Whisper hallucinations
        hallucinations = ["thank you.", "thanks for watching.", "go back to sleep.", "you"]
        if text.lower().strip() in hallucinations:
            return 
        
        if text:
            print(f"Typing: {text}")
            keyboard.write(text + " ")
            
    except Exception as e:
        print(f"Transcription Error: {e}")

def processing_loop():
    """Background logic that decides WHEN to transcribe."""
    buffer = []
    silent_chunks = 0
    current_buffer_length_seconds = 0
    
    print("Background service started.")
    
    while True:
        try:
            # Get data from microphone (non-blocking)
            data = audio_queue.get(timeout=0.1)
            buffer.append(data)
            
            chunk_duration = len(data) / 16000
            current_buffer_length_seconds += chunk_duration
            
            # Detect Silence
            amplitude = np.sqrt(np.mean(data**2))
            if amplitude < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0
            
            # --- TRIGGER LOGIC ---
            # 1. Silence Trigger: Have we been silent for SILENCE_DURATION seconds?
            # (assuming ~0.1s per chunk)
            silence_trigger = (silent_chunks * 0.1) > SILENCE_DURATION
            
            # 2. Max Duration Trigger: Have we been talking too long?
            max_len_trigger = current_buffer_length_seconds > MAX_CHUNK_DURATION

            # Only transcribe if a trigger is met AND we have enough audio (> 1.0s)
            if (silence_trigger or max_len_trigger) and current_buffer_length_seconds > 1.0:
                process_audio_chunk(buffer)
                buffer = [] # Clear buffer
                current_buffer_length_seconds = 0
                silent_chunks = 0
                
        except queue.Empty:
            # If user stops recording manually (F9), process whatever is left
            if not is_recording and buffer:
                process_audio_chunk(buffer)
                buffer = []
                current_buffer_length_seconds = 0

def audio_callback(indata, frames, time, status):
    """Raw audio input stream."""
    if is_recording:
        audio_queue.put(indata.copy())

def toggle_recording():
    global is_recording
    if not is_recording:
        is_recording = True
        print("\n[REC] Listening... (Press F9 to stop)")
        with audio_queue.mutex:
            audio_queue.queue.clear()
    else:
        is_recording = False
        print("\n[STOP] Processing final chunk...")

# Start Background Processing Thread
threading.Thread(target=processing_loop, daemon=True).start()

# Start Microphone Stream
# blocksize=1600 gives ~0.1s chunks for responsive silence detection
stream = sd.InputStream(samplerate=16000, channels=1, callback=audio_callback, blocksize=1600)
stream.start()

print(f"\nReady! Focus a text field and press {HOTKEY_TOGGLE} to dictate.")
print("Note: Text will appear after a 2-3 second pause in your speech.")
keyboard.add_hotkey(HOTKEY_TOGGLE, toggle_recording)
keyboard.wait()