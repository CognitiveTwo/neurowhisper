# Platform detection and cross-platform compatibility
import sys

# Set Windows App User Model ID for proper taskbar icon (Windows only)
if sys.platform == 'win32':
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('neuroflash.neurowhisper.1.0')
    except:
        pass

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import customtkinter as ctk
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Platform abstraction layer for keyboard/sound/clipboard
try:
    from platform_utils import (
        keyboard_module as keyboard,
        play_system_sound,
        play_feedback_sound_async,
        paste_from_clipboard,
        type_text,
        get_paste_shortcut,
        IS_MAC,
        IS_WINDOWS,
        PLATFORM_NAME
    )
    PLATFORM_UTILS_AVAILABLE = True
except ImportError:
    # Fallback to direct imports if platform_utils not available
    import keyboard
    IS_MAC = False
    IS_WINDOWS = True
    PLATFORM_NAME = 'windows'
    PLATFORM_UTILS_AVAILABLE = False
    try:
        import winsound
    except ImportError:
        winsound = None

from faster_whisper import WhisperModel

# Backend abstraction layer
try:
    from backends import create_backend, get_available_backends, detect_best_backend
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import json
import time
import shutil
import datetime
import webbrowser

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

# --- CONFIGURATION ---
CONFIG_FILE = "whisper_config.json"
STATS_FILE = "whisper_stats.json"
TRANSCRIPTIONS_DIR = "transcriptions"
TYPING_WPM = 40  # Assumed manual typing speed for "Time Saved" calc

# Friendly names to Model IDs
MODEL_MAP = {
    "Large v3 (Best quality, 10GB+ VRAM GPU)": "large-v3",
    "Distil-Multi (Good GPU, 6GB+ VRAM)": "bofenghuang/whisper-large-v3-distil-multi4-v0.2",
    "Medium (GPU or fast CPU, 4GB+ RAM)": "medium",
    "Small (CPU-friendly, 2GB+ RAM)": "small",
    "Base (Minimal resources, quick)": "base"
}
MODEL_MAP_REVERSE = {v: k for k, v in MODEL_MAP.items()}

def detect_cuda_available():
    """Check if CUDA is available and usable for inference.
    Returns tuple: (is_available: bool, reason: str)
    """
    try:
        import ctranslate2
        # Check if CUDA is supported by ctranslate2
        supported_devices = ctranslate2.get_supported_compute_types("cuda")
        if supported_devices:
            return True, "CUDA detected and available"
        else:
            return False, "CUDA not supported by ctranslate2"
    except Exception as e:
        error_str = str(e).lower()
        if any(kw in error_str for kw in ['cuda', 'cudnn', 'cublas', 'dll', 'library']):
            return False, f"CUDA libraries not found: {e}"
        return False, f"CUDA detection error: {e}"

def get_optimal_device_config():
    """Detect and return optimal device and compute_type settings."""
    cuda_available, reason = detect_cuda_available()
    if cuda_available:
        return "cuda", "float16", reason
    else:
        return "cpu", "int8", reason

DEFAULT_CONFIG = {
    "model_key": "Small (CPU-friendly, 2GB+ RAM)",  # Default to Small for broader compatibility
    "backend": "faster-whisper",  # Use faster-whisper by default for reliability
    "device": "cpu",  # Start with CPU, user can change to auto/cuda
    "compute_type": "int8",  # int8 for CPU efficiency
    "silence_threshold": 0.015,
    "live_pause": 0.8,
    "hotkey_live": "f9",
    "hotkey_batch": "ctrl+alt+s",
    "input_device": None,
    "always_on_top": True,
    # Online mode settings
    "transcription_mode": "local",  # "local" or "online"
    "openai_api_key": "",
    "openai_transcription_model": "whisper-1",
    "openai_edit_model": "gpt-4o-mini",
    "openai_edit_prompt": "",  # Empty = use default
    "openai_language": "auto"
}

# --- THEME COLORS ---
COLOR_BG = "#0f1419"           # Darker background
COLOR_FG = "#ffffff"           # White text
COLOR_ACCENT_BG = "#1a2129"    # Elevated panels
COLOR_TEXT_BG = "#1a2129"      # Card backgrounds
COLOR_TEAL = "#2dd4bf"         # Primary accent (bars, highlights)
COLOR_TEAL_DIM = "#134e4a"     # Subtle teal for backgrounds
COLOR_BORDER = "#2a3441"       # Subtle borders
COLOR_LIVE = "#ef4444"         # Red for live mode
COLOR_BATCH = "#3b82f6"        # Blue for batch mode  
COLOR_IDLE = "#374151"         # Gray
COLOR_ONLINE = "#a855f7"       # Purple for online mode
COLOR_TRANSCRIBING = "#c9a962" # Muted gold for transcribing
COLOR_READY = "#7cb886"        # Sage green for ready to paste

# Configure CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class WhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("neurowhisper")
        self.root.geometry("900x1000")  # Larger window to show all content
        
        # Set window icon (both title bar and taskbar)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except:
                pass
        # Also try wm_iconphoto for taskbar (works better on some Windows versions)
        if os.path.exists(png_path):
            try:
                from PIL import Image, ImageTk
                icon_img = Image.open(png_path)
                icon_photo = ImageTk.PhotoImage(icon_img)
                self.root.wm_iconphoto(True, icon_photo)
                self._icon_photo = icon_photo  # Keep reference to prevent garbage collection
            except:
                pass
        
        # Apply Dark Theme (keep ttk style for some legacy widgets)
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        self.root.configure(bg=COLOR_BG)
        
        self.config = self.load_config()
        
        # State
        self.mode = None 
        self.stopping = False
        self.model = None
        self.audio_queue = queue.Queue()
        self.msg_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = True
        
        # Stats State - Load persistent stats
        self.stats = self.load_stats()
        self.total_words = self.stats.get('total_words', 0)
        self.total_audio_duration = self.stats.get('total_audio_duration', 0.0)
        
        # Today's stats - persist across restarts within the same day
        today_str = datetime.datetime.now().date().isoformat()
        today_data = self.stats.get('today_data', {})
        if today_data.get('date') == today_str:
            # Same day - restore today's stats
            self.today_words = today_data.get('words', 0)
            self.today_audio_duration = today_data.get('audio_duration', 0.0)
        else:
            # New day - reset today's stats
            self.today_words = 0
            self.today_audio_duration = 0.0
        self.session_date = today_str
        
        # Transcription history
        self.history = []  # List of (timestamp, text, latency_ms) tuples
        self.history_index = -1  # -1 = showing latest live transcription
        self.current_transcription = ""  # Currently displayed transcription
        self.current_latency = 0  # Latency in ms for current transcription
        self.last_transcription_latency = 0  # Last measured transcription latency
        
        # VU Meter Smoothing State
        self.current_display_volume = 0.0
        
        # Buffers
        self.batch_audio = [] 
        self.live_buffer = [] # Class var to fix "last bit" bug
        self.live_backup_buffer = []
        self.live_buffer_lock = threading.Lock()  # Prevent race conditions on buffer
        
        # Mini Window Reference
        self.mini_window = None
        self.mini_vu = None
        self.mini_canvas = None
        
        # Streaming Batch Transcription
        self.batch_segments = []           # Ordered list of {seq, audio, result, status}
        self.batch_segment_lock = threading.Lock()
        self.batch_segment_seq = 0         # Sequence counter for ordering
        self.batch_executor = None         # ThreadPoolExecutor for parallel transcription
        self.batch_pending_audio = []      # Current audio buffer not yet segmented
        self.batch_silence_count = 0       # Silence detection counter

        # Online Streaming Transcription (similar to batch but uses OpenAI API)
        self.online_segments = []          # Ordered list of {seq, audio, result, status}
        self.online_segment_lock = threading.Lock()
        self.online_segment_seq = 0        # Sequence counter for ordering
        self.online_executor = None        # ThreadPoolExecutor for parallel API calls
        self.online_pending_audio = []     # Current audio buffer for online mode
        self.online_silence_count = 0      # Silence detection counter for online
        self.online_with_edit = False      # Whether current online session uses GPT editing

        # Latency tracking for performance metrics
        self.session_latencies = []  # List of latency values (ms) for this session
        self.total_latencies = self.stats.get('total_latencies', [])  # All-time latencies

        self.setup_ui()
        self.load_recent_transcriptions()  # Load history from last 2 days
        self.update_transcription_display()  # Show initial display
        self.root.attributes('-topmost', self.config['always_on_top'])
        
        # Threads
        threading.Thread(target=self.processing_loop, daemon=True).start()
        threading.Thread(target=self.load_model, daemon=True).start()
        
        # Audio
        self.restart_audio_stream()

        # Hotkeys - mode-aware (work in both Local and Online mode)
        keyboard.add_hotkey(self.config['hotkey_live'], self._handle_live_hotkey)
        keyboard.add_hotkey(self.config['hotkey_batch'], self._handle_batch_hotkey)
        
        # Minimization Binding
        self.root.bind("<Unmap>", self.on_minimize)
        self.root.bind("<Map>", self.on_restore)
        
        # UI Update Loop
        self.root.after(50, self.update_gui_loop)
        
        # Delayed histogram update (after canvas is fully sized)
        self.root.after(200, self.update_histogram)

    def configure_styles(self):
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("TLabelframe", background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("TLabelframe.Label", background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("TButton", background=COLOR_ACCENT_BG, foreground=COLOR_FG, borderwidth=1)
        self.style.map("TButton", background=[('active', '#505050')])
        self.style.configure("TCheckbutton", background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Horizontal.TProgressbar", troughcolor=COLOR_ACCENT_BG, background="#00e676", bordercolor=COLOR_BG, lightcolor="#00e676", darkcolor="#00e676")

    def _create_tooltip(self, widget, text):
        """Create a hover tooltip for a widget"""
        tooltip = None
        
        def show_tooltip(event):
            nonlocal tooltip
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.configure(bg="#333")
            
            label = tk.Label(tooltip, text=text, justify="left",
                           bg="#333", fg="#fff", font=("Arial", 9),
                           relief="solid", borderwidth=1, padx=8, pady=4)
            label.pack()
        
        def hide_tooltip(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None
        
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def play_feedback_sound(self, start=True):
        # Use platform-aware sound function
        if PLATFORM_UTILS_AVAILABLE:
            play_feedback_sound_async(start=start)
        elif IS_WINDOWS:
            try:
                freq = 800 if start else 400
                dur = 150
                threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()
            except:
                pass


    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded = json.load(f)
                    if "model_size" in loaded:
                        old_val = loaded.pop("model_size")
                        if old_val in MODEL_MAP_REVERSE:
                            loaded["model_key"] = MODEL_MAP_REVERSE[old_val]
                        else:
                            loaded["model_key"] = DEFAULT_CONFIG["model_key"]
                    config = {**DEFAULT_CONFIG, **loaded}
            except:
                config = DEFAULT_CONFIG.copy()
        else:
            config = DEFAULT_CONFIG.copy()
        
        # Auto-detect device if set to 'auto' or if this is first run
        if config.get('device') == 'auto' or config.get('compute_type') == 'auto':
            device, compute_type, reason = get_optimal_device_config()
            config['device'] = device
            config['compute_type'] = compute_type
            # Log the auto-detection result (will show in console)
            print(f"[Auto-detect] {reason} -> Using {device.upper()} mode")
            # Save the detected config so it persists
            try:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f)
            except:
                pass
        
        return config

    def load_stats(self):
        """Load persistent statistics from file"""
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r') as f:
                    data = json.load(f)
                    # Ensure hourly_data exists
                    if 'hourly_data' not in data:
                        data['hourly_data'] = {}
                    return data
            except:
                pass
        return {'total_words': 0, 'total_audio_duration': 0.0, 'hourly_data': {}}

    def save_stats(self):
        """Save statistics to file for persistence"""
        self.stats['total_words'] = self.total_words
        self.stats['total_audio_duration'] = self.total_audio_duration
        # Save today's data for persistence across restarts
        self.stats['today_data'] = {
            'date': self.session_date,
            'words': self.today_words,
            'audio_duration': self.today_audio_duration
        }
        try:
            with open(STATS_FILE, 'w') as f:
                json.dump(self.stats, f)
        except:
            pass

    def record_hourly_words(self, word_count):
        """Record words for the current hour in stats"""
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%H")
        
        if 'hourly_data' not in self.stats:
            self.stats['hourly_data'] = {}
        if date_str not in self.stats['hourly_data']:
            self.stats['hourly_data'][date_str] = {}
        
        current = self.stats['hourly_data'][date_str].get(hour_str, 0)
        self.stats['hourly_data'][date_str][hour_str] = current + word_count

    def format_speaking_time(self, seconds):
        """Format speaking duration in d/h/m format"""
        minutes = seconds / 60.0
        if minutes < 60:
            return f"{minutes:.1f}m"
        
        total_minutes = int(minutes)
        hours = total_minutes // 60
        mins = total_minutes % 60
        
        if hours < 24:
            return f"{hours}h {mins}m"
        
        days = hours // 24
        remaining_hours = hours % 24
        return f"{days}d {remaining_hours}h {mins}m"

    def format_time_saved(self, minutes):
        """
        Format time saved intelligently:
        - Under 60 minutes: show minutes (e.g., "45m")
        - 60+ minutes: show hours + minutes (e.g., "2h 15m")
        - 24+ hours: show days + hours + minutes (e.g., "3d 5h 30m")
        """
        if minutes < 60:
            return f"{minutes:.1f}m"
        
        total_minutes = int(minutes)
        hours = total_minutes // 60
        mins = total_minutes % 60
        
        if hours < 24:
            return f"{hours}h {mins}m"
        
        days = hours // 24
        remaining_hours = hours % 24
        return f"{days}d {remaining_hours}h {mins}m"

    def get_safe_model_path(self, friendly_name):
        safe_name = friendly_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        local_models_root = os.path.join(os.getcwd(), "models")
        return os.path.join(local_models_root, safe_name)

    def is_model_downloaded(self, friendly_name):
        target_dir = self.get_safe_model_path(friendly_name)
        # Simple check: if directory exists and has content
        return os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0

    def save_config(self):
        self.config['model_key'] = self.model_var.get()
        self.config['live_pause'] = float(self.live_pause_var.get())
        self.config['always_on_top'] = self.top_var.get()
        
        device_name = self.device_var.get()
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['name'] == device_name and d['max_input_channels'] > 0:
                self.config['input_device'] = i
                break
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)
        
        self.root.attributes('-topmost', self.config['always_on_top'])
        messagebox.showinfo("Saved", "Settings saved.\nRestart app to apply Model changes.")
        self.restart_audio_stream()

    def setup_ui(self):
        # --- TOP HEADER BAR ---
        header_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG, corner_radius=0)
        header_frame.pack(fill="x", padx=15, pady=(15, 10))
        
        # Left: "neurowhisper" title (neuro white, whisper blue)
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(side="left")
        
        title_neuro = ctk.CTkLabel(title_frame, text="neuro", text_color=COLOR_FG, 
                               font=ctk.CTkFont(family="Arial", size=20, weight="bold"))
        title_neuro.pack(side="left")
        
        title_whisper = ctk.CTkLabel(title_frame, text="whisper", text_color=COLOR_BATCH, 
                                 font=ctk.CTkFont(family="Arial", size=20, weight="bold"))
        title_whisper.pack(side="left")
        
        # Mode Toggle (LOCAL / ONLINE)
        mode_toggle_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        mode_toggle_frame.pack(side="left", padx=(20, 10))
        
        mode_container = ctk.CTkFrame(mode_toggle_frame, fg_color=COLOR_ACCENT_BG,
                                      corner_radius=15, border_width=1, border_color=COLOR_BORDER)
        mode_container.pack()
        
        mode_inner = ctk.CTkFrame(mode_container, fg_color="transparent")
        mode_inner.pack(padx=3, pady=3)
        
        # Track current transcription mode
        self.transcription_mode = tk.StringVar(value=self.config.get('transcription_mode', 'local'))
        
        self.btn_local = ctk.CTkButton(mode_inner, text="⚡ LOCAL",
                                       fg_color=COLOR_TEAL if self.transcription_mode.get() == 'local' else COLOR_ACCENT_BG,
                                       hover_color=COLOR_TEAL,
                                       text_color=COLOR_BG if self.transcription_mode.get() == 'local' else "#888888",
                                       font=ctk.CTkFont(size=11, weight="bold"),
                                       corner_radius=12, width=80, height=28,
                                       command=lambda: self.switch_transcription_mode('local'))
        self.btn_local.pack(side="left", padx=1)
        
        self.btn_online = ctk.CTkButton(mode_inner, text="☁️ ONLINE",
                                        fg_color=COLOR_ONLINE if self.transcription_mode.get() == 'online' else COLOR_ACCENT_BG,
                                        hover_color=COLOR_ONLINE,
                                        text_color=COLOR_FG if self.transcription_mode.get() == 'online' else "#888888",
                                        font=ctk.CTkFont(size=11, weight="bold"),
                                        corner_radius=12, width=85, height=28,
                                        command=lambda: self.switch_transcription_mode('online'))
        self.btn_online.pack(side="left", padx=1)
        
        self._create_tooltip(self.btn_local, "LOCAL: Uses your computer's hardware\nfor transcription. No internet needed.")
        self._create_tooltip(self.btn_online, "ONLINE: Uses OpenAI's cloud API.\nRequires API key and internet connection.")
        
        # Center: Segmented toggle for recording modes
        toggle_outer = ctk.CTkFrame(header_frame, fg_color="transparent")
        toggle_outer.pack(side="left", expand=True)
        
        # Pill-shaped container for toggle
        self.toggle_container = ctk.CTkFrame(toggle_outer, fg_color=COLOR_ACCENT_BG, 
                                        corner_radius=20, border_width=1, border_color=COLOR_BORDER)
        self.toggle_container.pack(padx=5, pady=5)
        
        # Inner frame for buttons
        self.toggle_inner = ctk.CTkFrame(self.toggle_container, fg_color="transparent")
        self.toggle_inner.pack(padx=4, pady=4)
        
        # Create buttons for LOCAL mode (BATCH | LIVE)
        batch_hotkey = self.config['hotkey_batch'].upper()
        self.btn_batch = ctk.CTkButton(self.toggle_inner, text=f"BATCH    {batch_hotkey}", 
                                       fg_color=COLOR_ACCENT_BG, hover_color=COLOR_BATCH,
                                       text_color="#888888", font=ctk.CTkFont(size=13, weight="bold"),
                                       corner_radius=16, width=130, height=36,
                                       command=self.toggle_batch_mode)
        
        # Separator line
        self.separator = ctk.CTkFrame(self.toggle_inner, fg_color=COLOR_BORDER, width=1, height=28)
        
        # LIVE button
        live_hotkey = self.config['hotkey_live'].upper()
        self.btn_live = ctk.CTkButton(self.toggle_inner, text=f"LIVE    {live_hotkey}", 
                                      fg_color=COLOR_ACCENT_BG, hover_color=COLOR_LIVE,
                                      text_color="#888888", font=ctk.CTkFont(size=13, weight="bold"),
                                      corner_radius=16, width=110, height=36,
                                      command=self.toggle_live_mode)
        
        # Create buttons for ONLINE mode (TRANSCRIBE | TRANSCRIBE+EDIT) - use same hotkeys
        self.btn_transcribe = ctk.CTkButton(self.toggle_inner, text=f"🎤 TRANSCRIBE    {batch_hotkey}", 
                                            fg_color=COLOR_ACCENT_BG, hover_color=COLOR_ONLINE,
                                            text_color="#888888", font=ctk.CTkFont(size=13, weight="bold"),
                                            corner_radius=16, width=170, height=36,
                                            command=self.toggle_online_transcribe)
        
        self.btn_transcribe_edit = ctk.CTkButton(self.toggle_inner, text=f"✨ +EDIT    {live_hotkey}", 
                                                 fg_color=COLOR_ACCENT_BG, hover_color=COLOR_ONLINE,
                                                 text_color="#888888", font=ctk.CTkFont(size=13, weight="bold"),
                                                 corner_radius=16, width=130, height=36,
                                                 command=self.toggle_online_transcribe_edit)
        
        # Show appropriate buttons based on mode
        self._update_mode_buttons()
        
        # Create dummy badge references for compatibility (just use the buttons)
        self.batch_badge = self.btn_batch
        self.live_badge = self.btn_live
        self.lbl_batch = self.btn_batch
        self.lbl_live = self.btn_live
        self.batch_segment = self.btn_batch
        self.live_segment = self.btn_live
        
        self._create_tooltip(self.btn_batch, "BATCH MODE: Records until you press again.\nTranscribes everything at once and pastes result.\nBest for longer dictation.")
        self._create_tooltip(self.btn_live, "LIVE MODE: Types as you speak in real-time.\nWords appear directly where your cursor is.\nBest for quick notes and immediate input.")
        self._create_tooltip(self.btn_transcribe, "TRANSCRIBE: Records audio and sends to\nOpenAI Whisper API for transcription.")
        self._create_tooltip(self.btn_transcribe_edit, "TRANSCRIBE+EDIT: Transcribes with OpenAI,\nthen uses GPT to clean up and copy-edit the text.")
        
        # Right: Branding
        branding_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        branding_frame.pack(side="right")
        
        self.author_label = ctk.CTkLabel(branding_frame, text="Created by ", 
                                         text_color="#6b7280", font=ctk.CTkFont(size=11))
        self.author_label.pack(side="left")
        
        self.author_link = ctk.CTkLabel(branding_frame, text="DR.M", text_color=COLOR_TEAL, 
                                        font=ctk.CTkFont(size=11, weight="bold"), cursor="hand2")
        self.author_link.pack(side="left")
        self.author_link.bind("<Button-1>", lambda e: webbrowser.open("https://www.linkedin.com/in/drjonathanmall/"))

        # VU Meter (mic level bar) - using CTk progress bar
        vu_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG, corner_radius=0)
        vu_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        self.vu_meter = ctk.CTkProgressBar(vu_frame, progress_color=COLOR_TEAL, 
                                           fg_color=COLOR_ACCENT_BG, corner_radius=5, height=8)
        self.vu_meter.pack(fill="x")
        self.vu_meter.set(0)

        # --- STATISTICS DASHBOARD (Two-Row Layout) ---
        stats_container = ctk.CTkFrame(self.root, fg_color=COLOR_BG, corner_radius=0)
        stats_container.pack(fill="x", padx=15, pady=5)
        
        # --- ROW 1: TODAY'S STATS + TODAY'S HISTOGRAM ---
        today_row = ctk.CTkFrame(stats_container, fg_color=COLOR_ACCENT_BG, 
                                 corner_radius=12, border_width=1, border_color=COLOR_BORDER)
        today_row.pack(fill="x", pady=(0, 5))
        
        # Today stats on left
        today_card = ctk.CTkFrame(today_row, fg_color="transparent", corner_radius=0)
        today_card.pack(side="left", fill="y", padx=(10, 0), pady=10)
        
        # Today header
        tk.Label(today_card, text="Today", bg=COLOR_ACCENT_BG, fg=COLOR_FG, 
                font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        
        # Words stat
        tk.Label(today_card, text="Words", bg=COLOR_ACCENT_BG, fg="#9ca3af", font=("Arial", 9)).grid(row=1, column=0, sticky="w", padx=(0, 15))
        self.lbl_session_words = tk.Label(today_card, text="0", bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, 
                                          font=("Arial", 14, "bold"))
        self.lbl_session_words.grid(row=2, column=0, sticky="w")
        
        # Speaking time stat
        tk.Label(today_card, text="Speaking Time", bg=COLOR_ACCENT_BG, fg="#9ca3af", font=("Arial", 9)).grid(row=1, column=1, sticky="w")
        self.lbl_session_duration = tk.Label(today_card, text="0m", bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, 
                                             font=("Arial", 14, "bold"))
        self.lbl_session_duration.grid(row=2, column=1, sticky="w")
        
        # Saved stat
        tk.Label(today_card, text="Saved", bg=COLOR_ACCENT_BG, fg="#9ca3af", font=("Arial", 9)).grid(row=3, column=0, sticky="w", pady=(8, 0), padx=(0, 15))
        self.lbl_session_time = tk.Label(today_card, text="0m", bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, 
                                         font=("Arial", 14, "bold"))
        self.lbl_session_time.grid(row=4, column=0, sticky="w")
        
        # Speed stat
        tk.Label(today_card, text="Speed", bg=COLOR_ACCENT_BG, fg="#9ca3af", font=("Arial", 9)).grid(row=3, column=1, sticky="w", pady=(8, 0))
        self.lbl_session_speed = tk.Label(today_card, text="0 WPM", bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, 
                                          font=("Arial", 14, "bold"))
        self.lbl_session_speed.grid(row=4, column=1, sticky="w")

        # Today's histogram on right
        self.histogram_canvas_today = tk.Canvas(today_row, height=100, bg=COLOR_ACCENT_BG, 
                                                highlightthickness=0)
        self.histogram_canvas_today.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # Store bar info for today
        self.histogram_bars_today = []
        
        # --- ROW 2: ALL-TIME STATS + ALL-TIME HISTOGRAM ---
        alltime_row = ctk.CTkFrame(stats_container, fg_color=COLOR_ACCENT_BG, 
                                   corner_radius=12, border_width=1, border_color=COLOR_BORDER)
        alltime_row.pack(fill="x")
        
        # All-Time stats on left
        alltime_card = ctk.CTkFrame(alltime_row, fg_color="transparent", corner_radius=0)
        alltime_card.pack(side="left", fill="y", padx=(10, 0), pady=10)
        
        # All-Time header
        tk.Label(alltime_card, text="All-Time", bg=COLOR_ACCENT_BG, fg=COLOR_FG, 
                font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        
        # Words stat
        tk.Label(alltime_card, text="Words", bg=COLOR_ACCENT_BG, fg="#9ca3af", font=("Arial", 9)).grid(row=1, column=0, sticky="w", padx=(0, 15))
        self.lbl_alltime_words = tk.Label(alltime_card, text="0", bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, 
                                          font=("Arial", 14, "bold"))
        self.lbl_alltime_words.grid(row=2, column=0, sticky="w")
        
        # Speaking time stat
        tk.Label(alltime_card, text="Speaking Time", bg=COLOR_ACCENT_BG, fg="#9ca3af", font=("Arial", 9)).grid(row=1, column=1, sticky="w")
        self.lbl_alltime_duration = tk.Label(alltime_card, text="0m", bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, 
                                             font=("Arial", 14, "bold"))
        self.lbl_alltime_duration.grid(row=2, column=1, sticky="w")
        
        # Saved stat
        tk.Label(alltime_card, text="Saved", bg=COLOR_ACCENT_BG, fg="#9ca3af", font=("Arial", 9)).grid(row=3, column=0, sticky="w", pady=(8, 0), padx=(0, 15))
        self.lbl_alltime_time = tk.Label(alltime_card, text="0m", bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, 
                                         font=("Arial", 14, "bold"))
        self.lbl_alltime_time.grid(row=4, column=0, sticky="w")
        
        # Speed stat
        tk.Label(alltime_card, text="Speed", bg=COLOR_ACCENT_BG, fg="#9ca3af", font=("Arial", 9)).grid(row=3, column=1, sticky="w", pady=(8, 0))
        self.lbl_alltime_speed = tk.Label(alltime_card, text="0 WPM", bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, 
                                          font=("Arial", 14, "bold"))
        self.lbl_alltime_speed.grid(row=4, column=1, sticky="w")

        # All-Time histogram on right
        self.histogram_canvas_alltime = tk.Canvas(alltime_row, height=100, bg=COLOR_ACCENT_BG, 
                                                  highlightthickness=0)
        self.histogram_canvas_alltime.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # Store bar info for all-time
        self.histogram_bars_alltime = []


        # --- COLLAPSIBLE SETTINGS ---
        # Header with toggle button
        settings_header = ttk.Frame(self.root)
        settings_header.pack(fill="x", padx=10, pady=(5, 0))
        
        # Determine if this is first run (config collapsed not set yet)
        is_first_run = 'config_collapsed' not in self.config
        self.config_collapsed = tk.BooleanVar(value=not is_first_run)  # Expanded on first run only
        
        self.settings_toggle_btn = tk.Button(settings_header, text="▼ Configuration" if not self.config_collapsed.get() else "▶ Configuration",
                                             bg=COLOR_ACCENT_BG, fg=COLOR_FG, font=("Arial", 9, "bold"),
                                             relief="flat", cursor="hand2", command=self.toggle_settings)
        self.settings_toggle_btn.pack(side="left", anchor="w")
        
        # Settings content frame (collapsible)
        self.settings_content = ttk.Frame(self.root)
        
        r1 = ttk.Frame(self.settings_content)
        r1.pack(fill="x", padx=5, pady=5)
        ttk.Label(r1, text="Model:").pack(side="left")
        
        # --- DYNAMIC MODEL LIST ---
        self.model_map_display = {}
        display_values = []
        current_display = self.config['model_key']

        for friendly in MODEL_MAP.keys():
            is_down = self.is_model_downloaded(friendly)
            prefix = "✓ " if is_down else "↓ "
            display_name = f"{prefix}{friendly}"
            self.model_map_display[display_name] = friendly
            display_values.append(display_name)
            
            if friendly == self.config['model_key']:
                current_display = display_name

        self.model_var = tk.StringVar(value=current_display)
        self.combo = ttk.Combobox(r1, textvariable=self.model_var, values=display_values, width=35, state="readonly")
        self.combo.pack(side="left", padx=5)
        self.combo.bind("<<ComboboxSelected>>", self.on_model_select)

        ttk.Label(r1, text="Pause (s):").pack(side="left", padx=(10,0))
        # Question mark with tooltip for pause explanation
        pause_help = ttk.Label(r1, text="?", foreground="#6b7280", 
                               font=("Arial", 9, "bold"), cursor="question_arrow")
        pause_help.pack(side="left", padx=(0, 2))
        self._create_tooltip(pause_help, "Pause Threshold (seconds):\nHow long to wait after you stop speaking\nbefore processing the audio in LIVE mode.\n\nShorter = faster response but may cut you off\nLonger = waits for natural pauses")
        
        self.live_pause_var = tk.DoubleVar(value=self.config['live_pause'])
        pause_spin = ttk.Spinbox(r1, from_=0.5, to=3.0, increment=0.1, textvariable=self.live_pause_var, width=5)
        pause_spin.pack(side="left", padx=5)
        # Auto-save on pause change
        self.live_pause_var.trace_add("write", self.on_setting_change)

        self.top_var = tk.BooleanVar(value=self.config.get('always_on_top', True))
        cb = ttk.Checkbutton(r1, text="On Top", variable=self.top_var, command=self.on_top_change)
        cb.pack(side="left", padx=10)

        r2 = ttk.Frame(self.settings_content)
        r2.pack(fill="x", padx=5, pady=5)
        ttk.Label(r2, text="Mic:").pack(side="left")
        
        devices = sd.query_devices()
        input_devices = [d['name'] for d in devices if d['max_input_channels'] > 0]
        current_dev_id = self.config.get('input_device')
        default_dev = ""
        if current_dev_id is not None and current_dev_id < len(devices):
             default_dev = devices[current_dev_id]['name']
        elif input_devices:
             default_dev = input_devices[0]

        self.device_var = tk.StringVar(value=default_dev)
        mic_combo = ttk.Combobox(r2, textvariable=self.device_var, values=input_devices, state="readonly")
        mic_combo.pack(side="left", fill="x", expand=True, padx=5)
        mic_combo.bind("<<ComboboxSelected>>", self.on_mic_change)
        
        # --- BACKEND/ACCELERATION SELECTOR ---
        r_backend = ttk.Frame(self.settings_content)
        r_backend.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(r_backend, text="Acceleration:").pack(side="left")
        
        # Backend options: Auto detects best, or force specific
        backend_options = ["Auto", "OpenVINO (Intel GPU)", "CUDA (NVIDIA GPU)", "CPU only"]
        backend_map = {"Auto": "auto", "OpenVINO (Intel GPU)": "openvino", 
                       "CUDA (NVIDIA GPU)": "cuda", "CPU only": "faster-whisper"}
        self.backend_map_display = {v: k for k, v in backend_map.items()}
        
        current_backend = self.config.get('backend', 'auto')
        current_backend_display = self.backend_map_display.get(current_backend, "Auto")
        
        self.backend_var = tk.StringVar(value=current_backend_display)
        backend_combo = ttk.Combobox(r_backend, textvariable=self.backend_var, 
                                      values=backend_options, width=20, state="readonly")
        backend_combo.pack(side="left", padx=5)
        backend_combo.bind("<<ComboboxSelected>>", self.on_backend_change)
        
        self._create_tooltip(backend_combo, "Acceleration Mode:\\n• Auto: Detects best option\\n• OpenVINO: Intel GPU/NPU (your Intel Arc)\\n• CUDA: NVIDIA GPUs\\n• CPU only: Works everywhere")
        
        # Show current detected backend
        self.backend_status_label = ttk.Label(r_backend, text="", foreground=COLOR_TEAL)
        self.backend_status_label.pack(side="left", padx=10)
        
        # Update status to show current config backend
        current_be = self.config.get('backend', 'faster-whisper')
        current_dev = self.config.get('device', 'cpu')
        self.backend_status_label.config(text=f"[{current_be} / {current_dev}]")
        
        # --- HOTKEY CONFIGURATION ROW ---
        r3 = ttk.Frame(self.settings_content)
        r3.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(r3, text="Hotkeys:").pack(side="left")
        
        # Batch hotkey
        ttk.Label(r3, text="Batch:").pack(side="left", padx=(15, 5))
        self.batch_hotkey_var = tk.StringVar(value=self.config['hotkey_batch'])
        self.batch_hotkey_btn = tk.Button(r3, textvariable=self.batch_hotkey_var, 
                                          bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, 
                                          font=("Arial", 9, "bold"), width=15,
                                          cursor="hand2", relief="groove",
                                          command=lambda: self.start_hotkey_capture('batch'))
        self.batch_hotkey_btn.pack(side="left", padx=2)
        self._create_tooltip(self.batch_hotkey_btn, "Click and press your desired shortcut for BATCH mode")
        
        # Live hotkey
        ttk.Label(r3, text="Live:").pack(side="left", padx=(15, 5))
        self.live_hotkey_var = tk.StringVar(value=self.config['hotkey_live'])
        self.live_hotkey_btn = tk.Button(r3, textvariable=self.live_hotkey_var, 
                                         bg=COLOR_ACCENT_BG, fg=COLOR_TEAL, 
                                         font=("Arial", 9, "bold"), width=15,
                                         cursor="hand2", relief="groove",
                                         command=lambda: self.start_hotkey_capture('live'))
        self.live_hotkey_btn.pack(side="left", padx=2)
        self._create_tooltip(self.live_hotkey_btn, "Click and press your desired shortcut for LIVE mode")
        
        # State for hotkey capture
        self.capturing_hotkey = None  # 'batch' or 'live' or None
        
        # --- OPENAI SETTINGS (for Online mode) ---
        openai_frame = ctk.CTkFrame(self.settings_content, fg_color=COLOR_ACCENT_BG, corner_radius=8)
        openai_frame.pack(fill="x", padx=5, pady=(10, 5))
        
        openai_header = ctk.CTkLabel(openai_frame, text="☁️ OpenAI Settings (Online Mode)", text_color=COLOR_FG, 
                                     font=ctk.CTkFont(size=11, weight="bold"))
        openai_header.pack(anchor="w", padx=10, pady=(8, 5))
        
        # API Key row
        api_row = ttk.Frame(openai_frame)
        api_row.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(api_row, text="API Key:").pack(side="left")
        self.api_key_var = tk.StringVar(value=self.config.get('openai_api_key', ''))
        self.api_key_entry = ttk.Entry(api_row, textvariable=self.api_key_var, width=40, show="•")
        self.api_key_entry.pack(side="left", padx=5)
        
        # Show/Hide toggle
        self.api_key_visible = tk.BooleanVar(value=False)
        def toggle_api_visibility():
            if self.api_key_visible.get():
                self.api_key_entry.config(show="")
            else:
                self.api_key_entry.config(show="•")
        
        show_btn = tk.Button(api_row, text="👁", bg=COLOR_ACCENT_BG, fg=COLOR_FG, font=("Arial", 10),
                            cursor="hand2", relief="flat",
                            command=lambda: (self.api_key_visible.set(not self.api_key_visible.get()), toggle_api_visibility()))
        show_btn.pack(side="left", padx=2)
        self._create_tooltip(show_btn, "Show/hide API key")
        
        save_key_btn = tk.Button(api_row, text="Save", bg=COLOR_TEAL, fg=COLOR_BG, font=("Arial", 9, "bold"),
                                cursor="hand2", command=self._save_openai_settings)
        save_key_btn.pack(side="left", padx=5)
        
        test_btn = tk.Button(api_row, text="Test", bg=COLOR_ACCENT_BG, fg=COLOR_FG, font=("Arial", 9),
                            cursor="hand2", command=self._test_openai_connection)
        test_btn.pack(side="left", padx=2)
        
        # Model selection row
        model_row = ttk.Frame(openai_frame)
        model_row.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(model_row, text="Transcription:").pack(side="left")
        self.openai_trans_model_var = tk.StringVar(value=self.config.get('openai_transcription_model', 'whisper-1'))
        trans_model_combo = ttk.Combobox(model_row, textvariable=self.openai_trans_model_var,
                                         values=["whisper-1", "gpt-4o-transcribe"], width=18, state="readonly")
        trans_model_combo.pack(side="left", padx=5)
        trans_model_combo.bind("<<ComboboxSelected>>", lambda e: self._auto_save_openai_settings())
        
        ttk.Label(model_row, text="Edit Model:").pack(side="left", padx=(10, 0))
        self.openai_edit_model_var = tk.StringVar(value=self.config.get('openai_edit_model', 'gpt-4o-mini'))
        edit_model_combo = ttk.Combobox(model_row, textvariable=self.openai_edit_model_var,
                                        values=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], width=15, state="readonly")
        edit_model_combo.pack(side="left", padx=5)
        edit_model_combo.bind("<<ComboboxSelected>>", lambda e: self._auto_save_openai_settings())
        
        ttk.Label(model_row, text="Language:").pack(side="left", padx=(10, 0))
        self.openai_lang_var = tk.StringVar(value=self.config.get('openai_language', 'auto'))
        lang_combo = ttk.Combobox(model_row, textvariable=self.openai_lang_var,
                                  values=["auto", "en", "de", "es", "fr", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"],
                                  width=6, state="readonly")
        lang_combo.pack(side="left", padx=5)
        lang_combo.bind("<<ComboboxSelected>>", lambda e: self._auto_save_openai_settings())
        
        # Edit prompt row
        prompt_row = ttk.Frame(openai_frame)
        prompt_row.pack(fill="x", padx=10, pady=(5, 10))
        
        ttk.Label(prompt_row, text="Edit Prompt:").pack(side="left")
        prompt_help = ttk.Label(prompt_row, text="?", foreground="#6b7280", 
                               font=("Arial", 9, "bold"), cursor="question_arrow")
        prompt_help.pack(side="left", padx=(0, 5))
        self._create_tooltip(prompt_help, "Custom system prompt for GPT editing.\nLeave empty to use default prompt.\nThe transcribed text will be appended to this prompt.")
        
        self.openai_prompt_var = tk.StringVar(value=self.config.get('openai_edit_prompt', ''))
        prompt_entry = ttk.Entry(prompt_row, textvariable=self.openai_prompt_var, width=60)
        prompt_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        # --- DOWNLOAD PROGRESS ---
        self.progress_frame = ttk.Frame(self.settings_content)
        self.progress_frame.pack(fill="x", padx=5, pady=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", side="left", expand=True)
        self.progress_label = ttk.Label(self.progress_frame, text="0%", width=6, anchor="e", font=("Arial", 8))
        self.progress_label.pack(side="right", padx=5)
        self.progress_frame.pack_forget()  # Hide by default
        
        # --- INSTRUCTIONS / HELP SECTION ---
        help_frame = ctk.CTkFrame(self.settings_content, fg_color=COLOR_ACCENT_BG, corner_radius=8)
        help_frame.pack(fill="x", padx=5, pady=(10, 5))
        
        help_header = ctk.CTkLabel(help_frame, text="📖 Instructions", text_color=COLOR_FG, 
                                   font=ctk.CTkFont(size=11, weight="bold"))
        help_header.pack(anchor="w", padx=10, pady=(8, 5))
        
        # Build help text with current hotkeys
        batch_hk = self.config['hotkey_batch'].upper()
        live_hk = self.config['hotkey_live'].upper()
        help_text = f"""How to Use:
• Press {batch_hk} for BATCH mode - records until pressed again, then transcribes all at once
• Press {live_hk} for LIVE mode - types words as you speak in real-time
• Hotkeys are customizable above - click the button and press your desired shortcut
• Use the mini floating window (drag-able) for quick access

Troubleshooting:
• Model not loading? Try a smaller model (Medium or Small)
• Slow transcription? Use 'cuda' device if you have an NVIDIA GPU
• No audio? Check your microphone selection above
• First run downloads the model (~1-3GB) - this is normal"""
        
        help_label = ctk.CTkLabel(help_frame, text=help_text, text_color="#9ca3af",
                                  font=ctk.CTkFont(size=10), justify="left", anchor="w")
        help_label.pack(anchor="w", padx=10, pady=(0, 10))
        
        # Pack settings content based on initial state
        if not self.config_collapsed.get():
            self.settings_content.pack(fill="x", padx=10, pady=(0, 5))
        
        # Save the collapsed state for next session
        if is_first_run:
            self.config['config_collapsed'] = True
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f)

        # --- LAST TRANSCRIPTION PANEL ---
        transcription_frame = ttk.LabelFrame(self.root, text="Last Transcription")
        transcription_frame.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        
        # Header row with navigation and copy button
        trans_header = ttk.Frame(transcription_frame)
        trans_header.pack(fill="x", padx=5, pady=(5, 0))
        
        # Navigation buttons (left side)
        nav_frame = ttk.Frame(trans_header)
        nav_frame.pack(side="left")
        
        self.btn_prev = tk.Button(nav_frame, text="◀", bg=COLOR_ACCENT_BG, fg=COLOR_FG, 
                                  font=("Arial", 10), width=3, cursor="hand2",
                                  command=lambda: self.navigate_history(-1))
        self.btn_prev.pack(side="left", padx=2)
        
        self.history_label = tk.Label(nav_frame, text="(0/0)", bg=COLOR_BG, fg="#888", font=("Arial", 9))
        self.history_label.pack(side="left", padx=5)
        
        self.btn_next = tk.Button(nav_frame, text="▶", bg=COLOR_ACCENT_BG, fg=COLOR_FG,
                                  font=("Arial", 10), width=3, cursor="hand2",
                                  command=lambda: self.navigate_history(1))
        self.btn_next.pack(side="left", padx=2)
        
        # Word count label (center)
        self.word_count_label = tk.Label(trans_header, text="0 words", bg=COLOR_BG, fg="#00e676", font=("Arial", 10, "bold"))
        self.word_count_label.pack(side="left", padx=20)
        
        # Copy button (right side)
        self.btn_copy = tk.Button(trans_header, text="📋 Copy", bg=COLOR_ACCENT_BG, fg=COLOR_FG,
                                  font=("Arial", 9), cursor="hand2",
                                  command=self.copy_current_transcription)
        self.btn_copy.pack(side="right", padx=5)
        
        # Transcription text area (selectable) - taller now
        self.transcription_text = tk.Text(transcription_frame, height=6, font=("Consolas", 11),
                                          bg=COLOR_TEXT_BG, fg=COLOR_FG, insertbackground='white',
                                          wrap="word", padx=10, pady=10)
        self.transcription_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.transcription_text.insert("1.0", "No transcriptions yet...")
        self.transcription_text.config(state='disabled')

        # --- SYSTEM LOG AREA (smaller) ---
        log_frame = ttk.LabelFrame(self.root, text="System Log")
        log_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.text_area = scrolledtext.ScrolledText(log_frame, state='disabled', font=("Consolas", 9), 
                                                   bg=COLOR_TEXT_BG, fg=COLOR_FG, insertbackground='white',
                                                   height=4)
        self.text_area.pack(fill="x", padx=5, pady=5)

        # --- STATUS BAR ---
        self.status_var = tk.StringVar(value="Loading...")
        lbl_stat = tk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", bg=COLOR_ACCENT_BG, fg=COLOR_FG)
        lbl_stat.pack(fill="x", side="bottom")

    def toggle_settings(self):
        """Toggle the collapsible settings panel"""
        if self.config_collapsed.get():
            # Expand
            self.settings_content.pack(fill="x", padx=10, pady=(0, 5), after=self.settings_toggle_btn.master)
            self.settings_toggle_btn.config(text="▼ Configuration")
            self.config_collapsed.set(False)
        else:
            # Collapse
            self.settings_content.pack_forget()
            self.settings_toggle_btn.config(text="▶ Configuration")
            self.config_collapsed.set(True)

    def on_setting_change(self, *args):
        # Auto-save pause setting (debounced by trace)
        try:
            self.config['live_pause'] = float(self.live_pause_var.get())
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f)
        except:
            pass  # Ignore errors during typing

    def on_top_change(self):
        self.config['always_on_top'] = self.top_var.get()
        self.root.attributes('-topmost', self.config['always_on_top'])
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)

    def on_mic_change(self, event=None):
        device_name = self.device_var.get()
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['name'] == device_name and d['max_input_channels'] > 0:
                self.config['input_device'] = i
                break
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)
        self.restart_audio_stream()

    def on_backend_change(self, event=None):
        """Handle backend selection change."""
        display_val = self.backend_var.get()
        backend_map = {"Auto": "auto", "OpenVINO (Intel GPU)": "openvino", 
                       "CUDA (NVIDIA GPU)": "cuda", "CPU only": "faster-whisper"}
        new_backend = backend_map.get(display_val, "auto")
        
        if new_backend != self.config.get('backend', 'auto'):
            self.config['backend'] = new_backend
            # Reset device to auto when backend changes
            self.config['device'] = 'auto'
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f)
            
            messagebox.showinfo("Backend Changed", 
                f"Backend changed to: {display_val}\n\nRestart the app to apply the new acceleration mode.")

    def _update_mode_buttons(self):
        """Show/hide appropriate buttons based on transcription mode."""
        # Hide all buttons first
        for widget in [self.btn_batch, self.separator, self.btn_live, 
                       self.btn_transcribe, self.btn_transcribe_edit]:
            widget.pack_forget()
        
        if self.transcription_mode.get() == 'local':
            # Show BATCH and LIVE buttons
            self.btn_batch.pack(side="left", padx=2)
            self.separator.pack(side="left", padx=4)
            self.btn_live.pack(side="left", padx=2)
        else:
            # Show TRANSCRIBE and TRANSCRIBE+EDIT buttons
            self.btn_transcribe.pack(side="left", padx=2)
            self.btn_transcribe_edit.pack(side="left", padx=2)

    def switch_transcription_mode(self, new_mode):
        """Switch between local and online transcription modes."""
        if new_mode == self.transcription_mode.get():
            return
        
        # Check if online mode requires API key
        if new_mode == 'online':
            if not self.config.get('openai_api_key'):
                messagebox.showwarning("API Key Required", 
                    "Please configure your OpenAI API key in the settings.\n\n"
                    "Go to Configuration > OpenAI Settings to add your key.")
                return
        
        self.transcription_mode.set(new_mode)
        self.config['transcription_mode'] = new_mode
        
        # Save config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)
        
        # Update button appearances
        if new_mode == 'local':
            self.btn_local.configure(fg_color=COLOR_TEAL, text_color=COLOR_BG)
            self.btn_online.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888")
        else:
            self.btn_local.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888")
            self.btn_online.configure(fg_color=COLOR_ONLINE, text_color=COLOR_FG)
        
        # Show appropriate action buttons
        self._update_mode_buttons()
        
        self.msg_queue.put(f"Switched to {new_mode.upper()} mode")

    def _handle_batch_hotkey(self):
        """Mode-aware handler for batch/transcribe hotkey."""
        if self.transcription_mode.get() == 'online':
            self.toggle_online_transcribe()
        else:
            self.toggle_batch_mode()

    def _handle_live_hotkey(self):
        """Mode-aware handler for live/transcribe+edit hotkey."""
        if self.transcription_mode.get() == 'online':
            self.toggle_online_transcribe_edit()
        else:
            self.toggle_live_mode()


    def toggle_online_transcribe(self):
        """Toggle online transcription (record and send to OpenAI)."""
        if self.mode == 'online_transcribe':
            # Stop recording and transcribe
            self._stop_online_recording(with_edit=False)
        elif self.mode is None:
            # Start recording
            self._start_online_recording(with_edit=False)
        else:
            # Already in another mode - stop it (any key = stop)
            self._stop_any_recording()

    def toggle_online_transcribe_edit(self):
        """Toggle online transcription with GPT editing."""
        if self.mode == 'online_transcribe_edit':
            # Stop recording and transcribe+edit
            self._stop_online_recording(with_edit=True)
        elif self.mode is None:
            # Start recording
            self._start_online_recording(with_edit=True)
        else:
            # Already in another mode - stop it (any key = stop)
            self._stop_any_recording()

    def _start_online_recording(self, with_edit=False):
        """Start recording for online transcription with streaming segments."""
        mode_name = 'online_transcribe_edit' if with_edit else 'online_transcribe'
        self.mode = mode_name
        self.online_with_edit = with_edit
        
        # Reset streaming state
        self.online_segments = []
        self.online_segment_seq = 0
        self.online_pending_audio = []
        self.online_silence_count = 0
        
        # Create thread pool for parallel API calls (2 workers for API rate limits)
        from concurrent.futures import ThreadPoolExecutor
        self.online_executor = ThreadPoolExecutor(max_workers=2)
        
        # Update UI
        btn = self.btn_transcribe_edit if with_edit else self.btn_transcribe
        btn.configure(fg_color=COLOR_ONLINE, text_color=COLOR_FG)
        
        self.play_feedback_sound(start=True)
        self.status_var.set(f"🎤 Recording{' + GPT edit' if with_edit else ''}...")
        self.msg_queue.put(f"Recording started (online streaming mode)")

    def _queue_online_segment(self):
        """Queue current pending audio for background OpenAI transcription."""
        with self.online_segment_lock:
            if not self.online_pending_audio:
                return
            
            seq = self.online_segment_seq
            self.online_segment_seq += 1
            audio_data = self.online_pending_audio.copy()
            
            segment = {
                'seq': seq,
                'audio': audio_data,
                'result': None,
                'status': 'pending'
            }
            self.online_segments.append(segment)
            self.online_pending_audio = []
            self.online_silence_count = 0
        
        # Submit to thread pool
        if self.online_executor:
            self.online_executor.submit(self._transcribe_online_segment, seq)
        
        self.log_internal(f"☁️ Queued segment {seq + 1} for OpenAI")

    def _transcribe_online_segment(self, seq):
        """Transcribe a single segment using OpenAI API (runs in thread pool)."""
        segment = None
        with self.online_segment_lock:
            for s in self.online_segments:
                if s['seq'] == seq:
                    segment = s
                    s['status'] = 'transcribing'
                    break
        
        if not segment:
            return
        
        try:
            from backends.openai_backend import OpenAIBackend
            
            # Write segment to temp file
            audio_np = np.concatenate(segment['audio'], axis=0).flatten()
            temp_file = f"temp_online_seg_{seq}.wav"
            wav.write(temp_file, 16000, (audio_np * 32767).astype(np.int16))
            
            backend = OpenAIBackend()
            backend.configure(
                api_key=self.config.get('openai_api_key', ''),
                transcription_model=self.config.get('openai_transcription_model', 'whisper-1'),
                edit_model=self.config.get('openai_edit_model', 'gpt-4o-mini'),
                edit_prompt=self.config.get('openai_edit_prompt', '') or None,
                language=self.config.get('openai_language', 'auto')
            )
            
            start_time = time.time()
            
            if self.online_with_edit:
                _, edited_text = backend.transcribe_and_edit(temp_file)
                text = edited_text
            else:
                segments = backend.transcribe(temp_file)
                text = " ".join(seg.text for seg in segments).strip()
            
            latency = (time.time() - start_time) * 1000
            
            with self.online_segment_lock:
                segment['result'] = text
                segment['status'] = 'done'
                segment['latency'] = latency
            
            self.log_internal(f"☁️ Segment {seq + 1} done ({latency:.0f}ms)")
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            with self.online_segment_lock:
                segment['status'] = 'error'
                segment['result'] = ''
            self.log_internal(f"☁️ Segment {seq + 1} error: {e}")

    def _stop_online_recording(self, with_edit=False):
        """Stop recording, process final segment, and combine all results."""
        self.play_feedback_sound(start=False)
        
        # Queue any remaining audio
        if self.online_pending_audio:
            self._queue_online_segment()
        
        # Check if we have any segments
        if not self.online_segments:
            self.mode = None
            self._reset_online_buttons()
            return
        
        self.mode = None
        
        # Show TRANSCRIBING state
        btn = self.btn_transcribe_edit if with_edit else self.btn_transcribe
        btn.configure(fg_color=COLOR_TRANSCRIBING, text_color="white", text="⏳ Finishing...")
        self.status_var.set("⏳ Waiting for transcriptions...")
        self.update_mini_window_color(COLOR_TRANSCRIBING)
        
        # Wait for all segments to complete in background
        def wait_and_combine():
            try:
                # Wait for executor to finish
                if self.online_executor:
                    self.online_executor.shutdown(wait=True)
                    self.online_executor = None
                
                # Combine results in order
                with self.online_segment_lock:
                    results = []
                    total_latency = 0
                    total_audio_duration = 0
                    for seg in sorted(self.online_segments, key=lambda x: x['seq']):
                        if seg['result']:
                            results.append(seg['result'])
                        total_latency += seg.get('latency', 0)
                        # Calculate audio duration from segment
                        if seg['audio']:
                            total_audio_duration += len(seg['audio']) * 0.1
                
                result_text = " ".join(results).strip()
                
                if result_text:
                    # Copy to clipboard and paste
                    def do_paste():
                        self.root.clipboard_clear()
                        self.root.clipboard_append(result_text)
                        self.root.update()
                        keyboard.send('ctrl+v')
                    self.root.after(0, do_paste)
                    
                    # Update stats
                    word_count = len(result_text.split())
                    
                    self.today_words += word_count
                    self.today_audio_duration += total_audio_duration
                    self.total_words += word_count
                    self.total_audio_duration += total_audio_duration
                    self.record_hourly_words(word_count)
                    self.save_stats()
                    
                    # Save to history
                    self.save_transcription(result_text, total_latency)
                    
                    num_segments = len(self.online_segments)
                    self.msg_queue.put(f"☁️ Transcribed {word_count} words from {num_segments} segments ({total_latency:.0f}ms total)")
                    
                    # Show READY state
                    def show_ready():
                        if with_edit:
                            self.btn_transcribe_edit.configure(fg_color=COLOR_READY, text_color="white", text="✓ Done!")
                        else:
                            self.btn_transcribe.configure(fg_color=COLOR_READY, text_color="white", text="✓ Done!")
                        self.status_var.set("✅ Copied & Pasted!")
                        self.update_mini_window_color(COLOR_READY)
                        self.root.after(2000, self._reset_online_buttons)
                    self.root.after(0, show_ready)
                else:
                    self.msg_queue.put("No transcription result")
                    self.root.after(0, self._reset_online_buttons)
                    
            except Exception as e:
                self.msg_queue.put(f"Online transcription failed: {e}")
                self.root.after(0, self._reset_online_buttons)
        
        threading.Thread(target=wait_and_combine, daemon=True).start()

    def _reset_online_buttons(self):
        """Reset online mode button colors and text."""
        batch_hk = self.config.get('hotkey_batch', 'f8').upper()
        live_hk = self.config.get('hotkey_live', 'f9').upper()
        self.btn_transcribe.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888", text=f"🎤 TRANSCRIBE    {batch_hk}")
        self.btn_transcribe_edit.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888", text=f"✨ +EDIT    {live_hk}")
        self.update_mini_window_color(COLOR_IDLE)
        self.status_var.set("Ready")

    def _save_openai_settings(self):
        """Save OpenAI settings to config."""
        self.config['openai_api_key'] = self.api_key_var.get().strip()
        self.config['openai_transcription_model'] = self.openai_trans_model_var.get()
        self.config['openai_edit_model'] = self.openai_edit_model_var.get()
        self.config['openai_language'] = self.openai_lang_var.get()
        self.config['openai_edit_prompt'] = self.openai_prompt_var.get().strip()
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)
        
        self.msg_queue.put("OpenAI settings saved")
        messagebox.showinfo("Saved", "OpenAI settings saved successfully.")

    def _auto_save_openai_settings(self):
        """Silently save OpenAI settings when combo boxes change."""
        self.config['openai_transcription_model'] = self.openai_trans_model_var.get()
        self.config['openai_edit_model'] = self.openai_edit_model_var.get()
        self.config['openai_language'] = self.openai_lang_var.get()
        
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f)
            self.log_internal(f"Settings saved: {self.openai_trans_model_var.get()}")
        except:
            pass

    def _test_openai_connection(self):
        """Test the OpenAI API connection."""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("No API Key", "Please enter an API key first.")
            return
        
        self.status_var.set("Testing OpenAI connection...")
        
        def test_connection():
            try:
                from backends.openai_backend import OpenAIBackend
                
                backend = OpenAIBackend()
                backend.configure(api_key=api_key)
                success, message = backend.test_connection()
                
                def show_result():
                    self.status_var.set("Ready")
                    if success:
                        messagebox.showinfo("Connection Test", f"✅ {message}")
                    else:
                        messagebox.showerror("Connection Test", f"❌ {message}")
                
                self.root.after(0, show_result)
                
            except ImportError as e:
                self.root.after(0, lambda: messagebox.showerror("Error", 
                    f"OpenAI library not installed.\n\nRun: pip install openai"))
                self.root.after(0, lambda: self.status_var.set("Ready"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Test failed: {e}"))
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        threading.Thread(target=test_connection, daemon=True).start()

    def start_hotkey_capture(self, hotkey_type):
        """Start capturing a new hotkey for batch or live mode - supports multi-key combos"""
        self.capturing_hotkey = hotkey_type
        self._hotkey_parts = set()  # Track currently held keys
        self._hotkey_confirmed = False
        
        # Update button appearance to show capture mode
        if hotkey_type == 'batch':
            self.batch_hotkey_btn.config(bg=COLOR_TEAL, fg=COLOR_BG, text="Press keys... (Enter to confirm)")
        else:
            self.live_hotkey_btn.config(bg=COLOR_TEAL, fg=COLOR_BG, text="Press keys... (Enter to confirm)")
        
        def update_preview():
            """Update the button text to show current key combo"""
            if not self._hotkey_parts:
                preview = "Press keys..."
            else:
                # Order: ctrl, alt, shift, win, then other keys
                ordered = []
                for mod in ['ctrl', 'alt', 'shift', 'win']:
                    if mod in self._hotkey_parts:
                        ordered.append(mod)
                for key in sorted(self._hotkey_parts):
                    if key not in ['ctrl', 'alt', 'shift', 'win']:
                        ordered.append(key)
                preview = '+'.join(ordered)
            
            preview_text = f"{preview} (Enter=OK, Esc=Cancel)"
            if hotkey_type == 'batch':
                self.batch_hotkey_btn.config(text=preview_text)
            else:
                self.live_hotkey_btn.config(text=preview_text)
        
        def on_key_down(event):
            if self.capturing_hotkey is None or self._hotkey_confirmed:
                return
            
            key_name = event.name.lower()
            
            # Enter confirms the hotkey
            if key_name in ('enter', 'return'):
                if self._hotkey_parts:
                    self._hotkey_confirmed = True
                    # Build final hotkey string
                    ordered = []
                    for mod in ['ctrl', 'alt', 'shift', 'win']:
                        if mod in self._hotkey_parts:
                            ordered.append(mod)
                    for key in sorted(self._hotkey_parts):
                        if key not in ['ctrl', 'alt', 'shift', 'win']:
                            ordered.append(key)
                    new_hotkey = '+'.join(ordered)
                    
                    current_type = self.capturing_hotkey
                    self.capturing_hotkey = None
                    keyboard.unhook(on_key_down)
                    keyboard.unhook(on_key_up)
                    self.root.after(10, lambda: self.finish_hotkey_capture(current_type, new_hotkey))
                return
            
            # Escape cancels
            if key_name == 'escape':
                keyboard.unhook(on_key_down)
                keyboard.unhook(on_key_up)
                self.root.after(10, self.cancel_hotkey_capture)
                return
            
            # Normalize modifier names
            if key_name in ('left ctrl', 'right ctrl', 'control'):
                key_name = 'ctrl'
            elif key_name in ('left alt', 'right alt'):
                key_name = 'alt'
            elif key_name in ('left shift', 'right shift'):
                key_name = 'shift'
            elif key_name in ('left windows', 'right windows', 'windows'):
                key_name = 'win'
            
            self._hotkey_parts.add(key_name)
            self.root.after(1, update_preview)
        
        def on_key_up(event):
            if self.capturing_hotkey is None or self._hotkey_confirmed:
                return
            
            key_name = event.name.lower()
            
            # Normalize modifier names
            if key_name in ('left ctrl', 'right ctrl', 'control'):
                key_name = 'ctrl'
            elif key_name in ('left alt', 'right alt'):
                key_name = 'alt'
            elif key_name in ('left shift', 'right shift'):
                key_name = 'shift'
            elif key_name in ('left windows', 'right windows', 'windows'):
                key_name = 'win'
            
            # Don't remove keys - we want to capture all pressed keys
            # Only update if the key was in our set
            # self._hotkey_parts.discard(key_name)
            # self.root.after(1, update_preview)
        
        # Hook both key down and key up
        keyboard.hook(on_key_down, suppress=False)
        on_key_up_hook = keyboard.on_release(on_key_up, suppress=False)
        
        # Store hook reference for cleanup
        self._hotkey_hooks = (on_key_down, on_key_up_hook)

    def finish_hotkey_capture(self, hotkey_type, new_hotkey):
        """Finish capturing hotkey and save"""
        try:
            # Reset button appearance
            if hotkey_type == 'batch':
                self.batch_hotkey_btn.config(bg=COLOR_ACCENT_BG, fg=COLOR_TEAL)
                old_hotkey = self.config.get('hotkey_batch', 'ctrl+alt+s')
                self.batch_hotkey_var.set(new_hotkey)
                self.config['hotkey_batch'] = new_hotkey
                
                # Update keyboard hooks - use mode-aware handler
                try:
                    keyboard.remove_hotkey(old_hotkey)
                except:
                    pass
                keyboard.add_hotkey(new_hotkey, self._handle_batch_hotkey)
                
                # Update button labels in header (both local and online)
                self.btn_batch.configure(text=f"BATCH    {new_hotkey.upper()}")
                self.btn_transcribe.configure(text=f"🎤 TRANSCRIBE    {new_hotkey.upper()}")
            else:
                self.live_hotkey_btn.config(bg=COLOR_ACCENT_BG, fg=COLOR_TEAL)
                old_hotkey = self.config.get('hotkey_live', 'f9')
                self.live_hotkey_var.set(new_hotkey)
                self.config['hotkey_live'] = new_hotkey
                
                # Update keyboard hooks - use mode-aware handler
                try:
                    keyboard.remove_hotkey(old_hotkey)
                except:
                    pass
                keyboard.add_hotkey(new_hotkey, self._handle_live_hotkey)
                
                # Update button labels in header (both local and online)
                self.btn_live.configure(text=f"LIVE    {new_hotkey.upper()}")
                self.btn_transcribe_edit.configure(text=f"✨ +EDIT    {new_hotkey.upper()}")
            
            # Save config
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f)
            
            self.log(f"Hotkey updated: {hotkey_type.upper()} → {new_hotkey}")
        except Exception as ex:
            self.log(f"Error setting hotkey: {ex}")

    def cancel_hotkey_capture(self):
        """Cancel hotkey capture and reset UI"""
        hotkey_type = self.capturing_hotkey
        self.capturing_hotkey = None
        
        if hotkey_type == 'batch':
            self.batch_hotkey_btn.config(bg=COLOR_ACCENT_BG, fg=COLOR_TEAL)
            self.batch_hotkey_var.set(self.config['hotkey_batch'])
        elif hotkey_type == 'live':
            self.live_hotkey_btn.config(bg=COLOR_ACCENT_BG, fg=COLOR_TEAL)
            self.live_hotkey_var.set(self.config['hotkey_live'])

    def on_model_select(self, event=None):
        display_val = self.model_var.get()
        real_key = self.model_map_display.get(display_val, display_val)
        
        # Fallback
        if real_key not in MODEL_MAP:
             for k in MODEL_MAP.keys():
                 if k in display_val:
                     real_key = k
                     break
        
        # If already same as config, ignore (e.g. startup)
        if real_key == self.config['model_key']: return

        # Check download status - only prompt if not downloaded
        if not self.is_model_downloaded(real_key):
             confirm = messagebox.askyesno(
                 "Download Required", 
                 f"The model '{real_key}' is not downloaded yet.\n\nDo you want to download it now?\n(This may take a few minutes depending on your internet speed)\n\nClick Yes to Download & Switch.\nClick No to Cancel."
             )
             if not confirm:
                 # Revert to previous
                 self.refresh_model_list()
                 return
        
        # Auto-save for ALL model changes (downloaded or just confirmed download)
        self.save_config()

    def save_config(self):
        # STRIP DISPLAY PREFIX (✓ or ↓)
        display_val = self.model_var.get()
        real_key = self.model_map_display.get(display_val, display_val)
        
        # Fallback if map fail
        if real_key not in MODEL_MAP:
             for k in MODEL_MAP.keys():
                 if k in display_val:
                     real_key = k
                     real_key = k
                     break

        # Capture old key to check for changes
        old_model_key = self.config.get('model_key', '')

        self.config['model_key'] = real_key
        self.config['live_pause'] = float(self.live_pause_var.get())
        self.config['always_on_top'] = self.top_var.get()
        
        device_name = self.device_var.get()
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['name'] == device_name and d['max_input_channels'] > 0:
                self.config['input_device'] = i
                break
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)
        
        self.root.attributes('-topmost', self.config['always_on_top'])
        
        if real_key != old_model_key:
             # Simply reload model in a thread
             threading.Thread(target=self.load_model, daemon=True).start()
             if not self.is_model_downloaded(real_key):
                  messagebox.showinfo("Downloading", f"Downloading {real_key}...\nPlease wait and check the progress bar.")
             else:
                  messagebox.showinfo("Saved", f"Switched to {real_key}.\nLoading in background...")
        else:
             messagebox.showinfo("Saved", "Settings saved.")

        self.restart_audio_stream()

    def refresh_model_list(self):
        # Re-scan download status
        current_selection = self.model_var.get()
        pure_name = self.model_map_display.get(current_selection, current_selection)
        
        self.model_map_display = {}
        display_values = []
        
        new_selection_display = current_selection

        for friendly in MODEL_MAP.keys():
            is_down = self.is_model_downloaded(friendly)
            prefix = "✓ " if is_down else "↓ "
            display_name = f"{prefix}{friendly}"
            self.model_map_display[display_name] = friendly
            display_values.append(display_name)
            
            if friendly == pure_name:
                new_selection_display = display_name
                
        self.combo['values'] = display_values
        self.model_var.set(new_selection_display)
    def toggle_live_mode(self):
        """F9 Toggled"""
        if self.stopping: return

        if self.mode == "batch": 
            # Stop batch mode instead of blocking (any key = stop)
            self._stop_any_recording()
            return

        if self.mode == "live":
            # STOP
            self.play_feedback_sound(start=False)
            self.stopping = True
            self.btn_live.configure(text="Stopping...", fg_color="#500000", text_color="white") 
            self.status_var.set("Catching final words...")
            threading.Thread(target=self._delayed_stop_live).start()
        else:
            # START
            self.play_feedback_sound(start=True)
            self.mode = "live"
            with self.audio_queue.mutex: self.audio_queue.queue.clear()
            self.live_buffer = [] 
            self.live_backup_buffer = [] 
            # Highlight LIVE button, dim BATCH button
            self.btn_live.configure(fg_color=COLOR_LIVE, text_color="white", text="LIVE    F9")
            self.btn_batch.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888", text="BATCH    F8")
            self.status_var.set(f"🔴 LIVE TYPING ({self.config['hotkey_live']})")
            self.update_mini_window_color(COLOR_LIVE)

    def _delayed_stop_live(self):
        time.sleep(0.6) 
        
        # --- FIX: Atomically capture and clear the remaining buffer ---
        # This prevents the race condition where processing_loop could
        # process the same chunk we're about to process here.
        with self.live_buffer_lock:
            remaining_buffer = self.live_buffer.copy()
            self.live_buffer = []
        
        if remaining_buffer and self.model:
            self.log_internal("Flushing final buffer...")
            self.process_live_chunk(remaining_buffer)
        # -----------------------------------------

        self.mode = None
        self.stopping = False
        # Reset both buttons to idle state
        self.btn_live.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888", text="LIVE    F9")
        self.btn_batch.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888", text="BATCH    F8")
        self.status_var.set("Ready. (Processing Live Backup...)")
        self.update_mini_window_color(COLOR_IDLE)
        self.finalize_live_backup()

    def _reset_buttons_to_idle(self):
        """Reset both mode buttons to idle state"""
        batch_hk = self.config.get('hotkey_batch', 'f8').upper()
        live_hk = self.config.get('hotkey_live', 'f9').upper()
        self.btn_batch.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888", text=f"BATCH    {batch_hk}")
        self.btn_live.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888", text=f"LIVE    {live_hk}")
        self.update_mini_window_color(COLOR_IDLE)

    def _stop_any_recording(self):
        """Stop any active recording mode (unified stop behavior)."""
        if self.stopping:
            return
        if self.mode == "live":
            self.toggle_live_mode()  # Will trigger stop
        elif self.mode == "batch":
            self.toggle_batch_mode()  # Will trigger stop
        elif self.mode == 'online_transcribe':
            self._stop_online_recording(with_edit=False)
        elif self.mode == 'online_transcribe_edit':
            self._stop_online_recording(with_edit=True)

    def toggle_batch_mode(self):
        """F8 Toggled"""
        if self.stopping: return

        if self.mode == "live":
            # Stop live mode instead of blocking (any key = stop)
            self._stop_any_recording()
            return

        if self.mode == "batch":
            # STOP
            self.play_feedback_sound(start=False)
            self.stopping = True
            self.btn_batch.configure(text="Stopping...", fg_color="#002050", text_color="white") 
            self.status_var.set("Catching final words...")
            threading.Thread(target=self._delayed_stop_batch).start()
        else:
            # START
            self.play_feedback_sound(start=True)
            self.mode = "batch"
            self.batch_audio = []
            with self.audio_queue.mutex: self.audio_queue.queue.clear()
            
            # Initialize streaming batch state
            self.batch_segments = []
            self.batch_segment_seq = 0
            self.batch_pending_audio = []
            self.batch_silence_count = 0
            
            # Create thread pool - limit workers to avoid GPU memory issues
            max_workers = 2 if self.config.get('device') == 'cpu' else 1
            self.batch_executor = ThreadPoolExecutor(max_workers=max_workers)
            
            # Highlight BATCH button, dim LIVE button
            self.btn_batch.configure(fg_color=COLOR_BATCH, text_color="white", text="BATCH    F8")
            self.btn_live.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888", text="LIVE    F9")
            self.status_var.set(f"🔵 BATCH RECORDING ({self.config['hotkey_batch']})...")
            self.update_mini_window_color(COLOR_BATCH)

    def _delayed_stop_batch(self):
        time.sleep(0.6)
        self.mode = None
        self.stopping = False
        
        # Show TRANSCRIBING state (amber)
        batch_hk = self.config.get('hotkey_batch', 'f8').upper()
        self.btn_batch.configure(fg_color=COLOR_TRANSCRIBING, text_color="white", text=f"⏳ Transcribing...")
        self.btn_live.configure(fg_color=COLOR_ACCENT_BG, text_color="#888888", text=f"LIVE    {self.config.get('hotkey_live', 'f9').upper()}")
        self.status_var.set("⏳ Processing Batch... Please Wait.")
        self.update_mini_window_color(COLOR_TRANSCRIBING)
        self.finalize_batch()

    def finalize_batch(self):
        if not self.batch_audio and not self.batch_segments:
            self.log_internal("No audio recorded.")
            self._reset_buttons_to_idle()
            self.status_var.set("Ready.")
            return
        
        try:
            # Track duration using the full audio buffer
            audio_np = np.concatenate(self.batch_audio, axis=0).flatten()
            duration = len(audio_np) / 16000.0
            self.total_audio_duration += duration
            self.today_audio_duration += duration
            
            # 1. Queue any remaining pending audio
            if self.batch_pending_audio:
                self._queue_batch_segment()
            
            # 2. Wait for all segments to complete transcription
            if self.batch_executor:
                self.log_internal("⏳ Waiting for transcription threads...")
                self.status_var.set("⏳ Finalizing transcription...")
                self.batch_executor.shutdown(wait=True)
                self.batch_executor = None
            
            # 3. Stitch results in correct order
            with self.batch_segment_lock:
                ordered = sorted(self.batch_segments, key=lambda x: x['seq'])
                results = [s['result'] for s in ordered if s['result'] and not s['result'].startswith('[Error')]
                full_text = " ".join(results).strip()
            
            # 4. Output the combined text
            if full_text:
                self.log_internal(f"✅ Batch complete: {len(ordered)} segments stitched")
                self.result_queue.put(f"[BATCH] {full_text}")
                self.root.clipboard_clear()
                self.root.clipboard_append(full_text)
                self.root.update()
                
                # Show READY state (green) - text is in clipboard
                batch_hk = self.config.get('hotkey_batch', 'f8').upper()
                self.btn_batch.configure(fg_color=COLOR_READY, text_color="white", text=f"✓ Ready to Paste")
                self.status_var.set("✅ Ready! Text copied to clipboard. Pasting...")
                self.update_mini_window_color(COLOR_READY)
                self.root.update()
                
                # Auto-paste
                # Platform-aware paste (Cmd+V on Mac, Ctrl+V on Windows)
            paste_key = get_paste_shortcut() if PLATFORM_UTILS_AVAILABLE else 'ctrl+v'
            keyboard.send(paste_key)
                
                # Return to idle after a delay
                self.root.after(2000, self._reset_buttons_to_idle)
            else:
                self.log_internal("Batch result was empty.")
                self._reset_buttons_to_idle()
                
        except Exception as e:
            self.log_internal(f"Batch Error: {e}")
            self._reset_buttons_to_idle()
        
        self.status_var.set("Ready.")

    def finalize_live_backup(self):
        if not self.live_backup_buffer: return
        try:
            # We don't add duration here because live mode already adds it chunk by chunk
            audio_np = np.concatenate(self.live_backup_buffer, axis=0).flatten()
            wav.write("temp_live_backup.wav", 16000, audio_np)
            if not self.model: return
            segments, _ = self.model.transcribe(
                "temp_live_backup.wav", beam_size=5, vad_filter=True, condition_on_previous_text=True 
            )
            text = " ".join([s.text for s in segments]).strip()
            if text:
                self.result_queue.put(f"[BACKUP] {text}")
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.root.update()
                self.status_var.set("Ready. (Backup in Clipboard)")
            else:
                self.status_var.set("Ready.")
        except Exception as e:
            self.log_internal(f"Backup Error: {e}")

    # --- MINIMIZATION LOGIC ---
    def on_minimize(self, event):
        if self.root.state() == 'iconic' and self.mini_window is None:
            self.create_mini_window()

    def on_restore(self, event):
        if self.root.state() == 'normal' and self.mini_window is not None:
            self.destroy_mini_window()

    def create_mini_window(self):
        self.mini_window = tk.Toplevel(self.root)
        self.mini_window.overrideredirect(True) 
        self.mini_window.attributes('-topmost', True)
        self.mini_window.geometry("150x40+50+50") 
        
        bg_col = COLOR_IDLE
        if self.mode == "live": bg_col = COLOR_LIVE
        elif self.mode == "batch": bg_col = COLOR_BATCH
        
        self.mini_window.configure(bg=bg_col)
        
        self.mini_canvas = tk.Canvas(self.mini_window, bg=bg_col, highlightthickness=0, height=40)
        self.mini_canvas.pack(fill="both", expand=True)
        
        # UI Elements
        self.mini_canvas.create_text(5, 12, text="Whisper Running", fill="white", 
                                     font=("Arial", 9, "bold"), anchor="w", tags="status_text")
        
        self.mini_canvas.create_text(145, 12, text="0w | 0.0m", fill="#dddddd", 
                                     font=("Arial", 7), anchor="e", tags="stats_text")
        
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = 150
        h = 40
        self.mini_window.geometry(f'{w}x{h}+{sw-w-20}+{sh-h-60}')
        
        self.update_mini_window_color(bg_col)
        
        # --- DRAG FUNCTIONALITY ---
        self._mini_drag_data = {"x": 0, "y": 0}
        
        def start_drag(event):
            self._mini_drag_data["x"] = event.x
            self._mini_drag_data["y"] = event.y
        
        def do_drag(event):
            x = self.mini_window.winfo_x() + (event.x - self._mini_drag_data["x"])
            y = self.mini_window.winfo_y() + (event.y - self._mini_drag_data["y"])
            self.mini_window.geometry(f"+{x}+{y}")
        
        self.mini_canvas.bind("<Button-1>", start_drag)
        self.mini_canvas.bind("<B1-Motion>", do_drag)
        self.mini_window.bind("<Button-1>", start_drag)
        self.mini_window.bind("<B1-Motion>", do_drag)

    def destroy_mini_window(self):
        if self.mini_window:
            self.mini_window.destroy()
            self.mini_window = None
            self.mini_vu = None
            self.mini_canvas = None

    def update_mini_window_color(self, color):
        if self.mini_window and self.mini_canvas:
            self.mini_window.configure(bg=color)
            self.mini_canvas.configure(bg=color)

    # --- AUDIO LOOP ---
    def processing_loop(self):
        silent_chunks = 0
        current_live_duration = 0
        
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.1)
                
                # SMOOTHING
                raw_amp = np.sqrt(np.mean(data**2))
                if raw_amp > self.current_display_volume:
                    self.current_display_volume = raw_amp 
                else:
                    self.current_display_volume *= 0.85 
                
                if self.mode == "batch":
                    self.batch_audio.append(data)
                    self.batch_pending_audio.append(data)
                    
                    # VAD: detect pause for streaming segmentation
                    if raw_amp < self.config['silence_threshold']:
                        self.batch_silence_count += 1
                    else:
                        self.batch_silence_count = 0
                    
                    # Calculate pending audio duration (~100ms per chunk at 16kHz, 1600 samples)
                    pending_duration = len(self.batch_pending_audio) * 0.1
                    silence_duration = self.batch_silence_count * 0.1
                    
                    # Progressive pause threshold: encourages ~20s segments
                    # - Under 10s: require 2.0s pause (don't cut too early)
                    # - 10s-20s: linearly decrease from 2.0s to 1.5s (progressively easier to cut)
                    # - Over 20s: use 1.5s pause (eager to segment)
                    # This creates more evenly spaced ~20s segments
                    if pending_duration < 10.0:
                        pause_threshold_batch = 2.0
                    elif pending_duration < 20.0:
                        # Linear interpolation: 2.0 at 10s → 1.5 at 20s
                        progress = (pending_duration - 10.0) / 10.0  # 0 to 1
                        pause_threshold_batch = 2.0 - (0.5 * progress)  # 2.0 → 1.5
                    else:
                        pause_threshold_batch = 1.5
                    
                    # Segment when: (pause detected AND 10s+ audio) OR 60s max reached
                    is_natural_break = silence_duration > pause_threshold_batch and pending_duration > 10.0
                    is_max_duration = pending_duration > 60.0
                    
                    # Queue segment when pause detected or max duration reached
                    if (is_natural_break or is_max_duration) and not self.stopping:
                        self._queue_batch_segment()
                elif self.mode in ("online_transcribe", "online_transcribe_edit"):
                    # Online mode: streaming transcription with pause detection
                    self.online_pending_audio.append(data)
                    online_pending_duration = len(self.online_pending_audio) * 0.1  # ~100ms per chunk
                    
                    # Check for silence
                    if raw_amp < self.config['silence_threshold']:
                        self.online_silence_count += 1
                    else:
                        self.online_silence_count = 0
                    
                    online_silence_duration = self.online_silence_count * 0.1
                    
                    # Same progressive threshold as batch: ~20s target segments
                    if online_pending_duration < 10.0:
                        online_pause_threshold = 2.0
                    elif online_pending_duration < 20.0:
                        progress = (online_pending_duration - 10.0) / 10.0
                        online_pause_threshold = 2.0 - (0.5 * progress)
                    else:
                        online_pause_threshold = 1.5
                    
                    is_natural_break = online_silence_duration > online_pause_threshold and online_pending_duration > 10.0
                    is_max_duration = online_pending_duration > 60.0
                    
                    # Queue segment when pause or max duration
                    if (is_natural_break or is_max_duration) and not self.stopping:
                        self._queue_online_segment()
                elif self.mode == "live":
                    with self.live_buffer_lock:
                        self.live_buffer.append(data)
                    self.live_backup_buffer.append(data) 
                    chunk_dur = len(data) / 16000
                    current_live_duration += chunk_dur
            
                if raw_amp < self.config['silence_threshold']:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                
                pause_threshold = self.live_pause_var.get()
                is_silence = (silent_chunks * 0.1) > pause_threshold
                is_too_long = current_live_duration > 15.0

                if (is_silence or is_too_long) and current_live_duration > 0.5 and not self.stopping:
                    # Atomically capture and clear buffer to prevent race conditions
                    with self.live_buffer_lock:
                        buffer_to_process = self.live_buffer.copy()
                        self.live_buffer = []
                    if self.model and buffer_to_process:
                        self.process_live_chunk(buffer_to_process)
                    current_live_duration = 0
                    silent_chunks = 0
            
            except queue.Empty:
                self.current_display_volume *= 0.85
                pass
            except Exception:
                pass

    def process_live_chunk(self, audio_data):
        if not audio_data: return
        audio_np = np.concatenate(audio_data, axis=0).flatten()
        
        # --- TRACK DURATION ---
        duration = len(audio_np) / 16000.0
        self.total_audio_duration += duration
        self.today_audio_duration += duration
        # ----------------------
        
        wav.write("temp_live.wav", 16000, audio_np)
        try:
            # Track transcription latency
            start_time = time.perf_counter()
            
            segments, _ = self.model.transcribe("temp_live.wav", beam_size=1, vad_filter=True)
            text = "".join([s.text for s in segments]).strip()
            
            # Calculate and record latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.session_latencies.append(latency_ms)
            self.total_latencies.append(latency_ms)
            
            hallucinations = ["thank you.", "you", ".", "Thank you."]
            if text.lower().strip() in [h.lower() for h in hallucinations] or not text: return 
            
            # Log with latency for performance tracking
            self.log_internal(f"🎤 LIVE ⚡{latency_ms:.0f}ms")
            self.last_transcription_latency = latency_ms  # Store for display
            self.result_queue.put(f"[LIVE] {text}")
            keyboard.write(text + " ")
        except Exception as e:
            self.log_internal(f"Transcription error: {e}")

    def _queue_batch_segment(self):
        """Queue current pending audio for background transcription"""
        with self.batch_segment_lock:
            if not self.batch_pending_audio:
                return
            
            seq = self.batch_segment_seq
            self.batch_segment_seq += 1
            audio_data = self.batch_pending_audio.copy()
            
            segment = {
                'seq': seq,
                'audio': audio_data,
                'result': None,
                'status': 'pending'
            }
            self.batch_segments.append(segment)
            self.batch_pending_audio = []
            self.batch_silence_count = 0
        
        # Submit to thread pool (created on batch start)
        if self.batch_executor:
            self.batch_executor.submit(self._transcribe_segment, seq)
        
        self._update_batch_progress()
        self.log_internal(f"📦 Queued segment {seq + 1} for transcription")

    def _transcribe_segment(self, seq):
        """Transcribe a single segment (runs in thread pool)"""
        segment = None
        with self.batch_segment_lock:
            for s in self.batch_segments:
                if s['seq'] == seq:
                    segment = s
                    s['status'] = 'transcribing'
                    break
        
        if not segment or not self.model:
            return
        
        try:
            audio_np = np.concatenate(segment['audio'], axis=0).flatten()
            temp_file = f"temp_batch_seg_{seq}.wav"
            wav.write(temp_file, 16000, audio_np)
            
            # Track transcription latency
            start_time = time.perf_counter()
            
            segments, _ = self.model.transcribe(
                temp_file, beam_size=5, vad_filter=True
            )
            text = " ".join([s.text for s in segments]).strip()
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            with self.batch_segment_lock:
                segment['result'] = text
                segment['status'] = 'done'
                segment['latency_ms'] = latency_ms
            
            # Cleanup temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            # Log with latency info for performance comparison
            self.log_internal(f"✅ Segment {seq + 1} ⚡{latency_ms:.0f}ms")
            
        except Exception as e:
            with self.batch_segment_lock:
                segment['status'] = 'error'
                segment['result'] = f"[Error: {e}]"
            self.log_internal(f"❌ Segment {seq + 1} error: {e}")
        
        self._update_batch_progress()

    def _update_batch_progress(self):
        """Update UI with batch transcription progress"""
        with self.batch_segment_lock:
            total = len(self.batch_segments)
            done = sum(1 for s in self.batch_segments if s['status'] == 'done')
        
        if total > 0:
            progress_text = f"🔵 BATCH ({done}/{total} chunks)"
            self.status_var.set(progress_text)

    def _show_progress(self, label_text="Loading..."):
        """Show progress bar in indeterminate mode"""
        self.progress_bar.configure(mode="indeterminate")
        self.progress_frame.pack(fill="x", padx=5, pady=0)
        self.progress_bar.start(15)
        self.progress_label.configure(text=label_text)
    
    def _hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", value=0)
        self.progress_frame.pack_forget()

    def log_internal(self, msg):
        self.msg_queue.put(msg)
    
    def copy_text(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        messagebox.showinfo("Copied", "Text copied to clipboard!")

    def save_transcription(self, text, latency_ms=0):
        """Save transcription to monthly log file and add to history"""
        if not text.strip():
            return
            
        timestamp = datetime.datetime.now()
        
        # Add to in-memory history with latency
        self.history.append((timestamp, text, latency_ms))
        self.current_transcription = text
        self.current_latency = latency_ms
        self.history_index = len(self.history) - 1  # Point to the latest
        
        # Save to monthly file
        try:
            os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
            filename = timestamp.strftime("%Y-%m") + ".txt"
            filepath = os.path.join(TRANSCRIPTIONS_DIR, filename)
            
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp_str}] {text}\n")
        except Exception as e:
            self.log_internal(f"Error saving transcription: {e}")
        
        # Update display
        self.update_transcription_display()

    def load_recent_transcriptions(self):
        """Load transcriptions from last 2 calendar days + today"""
        self.history = []
        today = datetime.date.today()
        cutoff_date = today - datetime.timedelta(days=2)
        
        try:
            if not os.path.exists(TRANSCRIPTIONS_DIR):
                return
            
            # Get relevant month files (current and possibly previous month)
            months_to_check = set()
            for i in range(3):  # Today and past 2 days might span 2 months
                check_date = today - datetime.timedelta(days=i)
                months_to_check.add(check_date.strftime("%Y-%m"))
            
            for month_str in sorted(months_to_check):
                filepath = os.path.join(TRANSCRIPTIONS_DIR, f"{month_str}.txt")
                if not os.path.exists(filepath):
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse timestamp and text
                        # Format: [YYYY-MM-DD HH:MM:SS] text
                        if line.startswith('[') and '] ' in line:
                            try:
                                ts_end = line.index('] ')
                                ts_str = line[1:ts_end]
                                text = line[ts_end + 2:]
                                
                                timestamp = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                                
                                # Check if within date range
                                if timestamp.date() >= cutoff_date:
                                    self.history.append((timestamp, text))
                            except (ValueError, IndexError):
                                continue
            
            # Sort by timestamp
            self.history.sort(key=lambda x: x[0])
            
            if self.history:
                self.history_index = len(self.history) - 1
                self.current_transcription = self.history[-1][1]
                
        except Exception as e:
            self.log_internal(f"Error loading transcription history: {e}")

    def navigate_history(self, direction):
        """Navigate through transcription history. direction: -1 for prev, 1 for next"""
        if not self.history:
            return
        
        new_index = self.history_index + direction
        
        # Clamp to valid range
        if new_index < 0:
            new_index = 0
        elif new_index >= len(self.history):
            new_index = len(self.history) - 1
        
        if new_index != self.history_index:
            self.history_index = new_index
            entry = self.history[new_index]
            self.current_transcription = entry[1]
            self.current_latency = entry[2] if len(entry) > 2 else 0
            self.update_transcription_display()

    def copy_current_transcription(self):
        """Copy the currently displayed transcription to clipboard"""
        if self.current_transcription:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.current_transcription)
            self.root.update()
            self.log_internal("📋 Transcription copied to clipboard")

    def update_transcription_display(self):
        """Update the transcription text area and navigation label"""
        # Update navigation label
        total = len(self.history)
        current = self.history_index + 1 if total > 0 else 0
        self.history_label.config(text=f"({current}/{total})")
        
        # Update word count with latency
        word_count = len(self.current_transcription.split()) if self.current_transcription else 0
        latency_str = f" ⚡{self.current_latency:.0f}ms" if self.current_latency > 0 else ""
        self.word_count_label.config(text=f"{word_count} words{latency_str}")
        
        # Auto-expand height for longer text (4 lines default, 8 for long text)
        # Estimate: ~80 chars per line at default width
        text_length = len(self.current_transcription) if self.current_transcription else 0
        new_height = 8 if text_length > 320 else 4  # Expand if > ~4 lines of text
        self.transcription_text.config(height=new_height)
        
        # Update text display
        self.transcription_text.config(state='normal')
        self.transcription_text.delete("1.0", "end")
        
        if self.current_transcription:
            self.transcription_text.insert("1.0", self.current_transcription)
        else:
            self.transcription_text.insert("1.0", "No transcriptions yet...")
        
        self.transcription_text.config(state='disabled')

    def get_histogram_data(self, for_today=True):
        """Get hourly word counts for histogram display"""
        hourly_data = self.stats.get('hourly_data', {})
        
        if for_today:
            # Get today's data only
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            day_data = hourly_data.get(today_str, {})
            return {int(h): count for h, count in day_data.items()}
        else:
            # Aggregate all days - return raw totals
            aggregated = {}
            for date_str, day_data in hourly_data.items():
                for hour_str, count in day_data.items():
                    hour = int(hour_str)
                    aggregated[hour] = aggregated.get(hour, 0) + count
            return aggregated

    def get_histogram_stats(self):
        """Get average and std dev per hour for all-time data"""
        hourly_data = self.stats.get('hourly_data', {})
        
        # Collect values per hour across all days
        hour_values = {h: [] for h in range(24)}
        
        for date_str, day_data in hourly_data.items():
            for hour_str, count in day_data.items():
                hour = int(hour_str)
                hour_values[hour].append(count)
        
        # Calculate statistics for each hour
        stats = {}
        for hour in range(6, 24):  # 6am to midnight
            values = hour_values[hour]
            if values:
                avg = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((x - avg) ** 2 for x in values) / len(values)
                    std = variance ** 0.5
                else:
                    std = 0
                stats[hour] = {'avg': avg, 'std': std, 'count': len(values)}
            else:
                stats[hour] = {'avg': 0, 'std': 0, 'count': 0}
        
        return stats

    def update_histogram(self):
        """Redraw both histograms"""
        # Draw today's histogram (dynamic hour range)
        today_data = self.get_histogram_data(for_today=True)
        self.draw_histogram_on_canvas(self.histogram_canvas_today, today_data, 
                                       fixed_hour_range=None, bars_list_name='histogram_bars_today')
        
        # Draw all-time histogram with averages and error bars
        alltime_stats = self.get_histogram_stats()
        self.draw_histogram_with_errorbars(self.histogram_canvas_alltime, alltime_stats)

    def draw_histogram_on_canvas(self, canvas, data, fixed_hour_range=None, bars_list_name='histogram_bars'):
        """Draw histogram bars on a specific canvas with teal gradient 3D effect"""
        canvas.delete("all")
        
        # Store bar rectangles for hover detection
        bars_list = []
        setattr(self, bars_list_name, bars_list)
        
        canvas.update_idletasks()
        width = canvas.winfo_width() or 200
        height = canvas.winfo_height() or 100
        
        if not data and not fixed_hour_range:
            canvas.create_text(width / 2, height / 2, 
                              text="No activity", fill="#666", font=("Arial", 9))
            return
        
        # Determine hour range - ALWAYS use 6am to 10pm for consistency
        min_hour, max_hour = 6, 22  # Fixed range for uniform bar width
        
        hours = list(range(min_hour, max_hour + 1))
        num_bars = len(hours)
        
        # Padding - extra space at bottom for hour labels
        left_padding = 8
        right_padding = 8
        top_padding = 8
        bottom_padding = 18  # Extra space for hour labels
        
        bar_area_width = width - left_padding - right_padding
        bar_area_height = height - top_padding - bottom_padding
        bar_width = max(8, (bar_area_width / num_bars) - 3)
        bar_spacing = bar_area_width / num_bars
        
        # Get max value for scaling
        max_val = max(data.values()) if data and data.values() else 1
        
        # Draw bars with teal gradient effect
        for i, hour in enumerate(hours):
            count = data.get(hour, 0) if data else 0
            bar_height = (count / max_val) * bar_area_height if max_val > 0 else 0
            
            x = left_padding + i * bar_spacing + bar_spacing / 2 - bar_width / 2
            y_bottom = height - bottom_padding
            y_top = y_bottom - max(bar_height, 2)  # Minimum 2px height for visibility
            
            if count > 0:
                # Teal gradient: lighter top, darker bottom
                intensity = count / max_val if max_val > 0 else 0
                
                # Base teal colors
                r_light = int(45 + intensity * 30)
                g_light = int(212 - intensity * 40)
                b_light = int(191 - intensity * 30)
                
                r_dark = int(19 + intensity * 20)
                g_dark = int(78 - intensity * 20)
                b_dark = int(74 - intensity * 20)
                
                color_light = f"#{r_light:02x}{g_light:02x}{b_light:02x}"
                color_dark = f"#{r_dark:02x}{g_dark:02x}{b_dark:02x}"
                
                # Draw 3D effect - darker base
                canvas.create_rectangle(x + 2, y_top + 2, x + bar_width + 2, y_bottom + 2, 
                                        fill=color_dark, outline="")
                
                # Main bar with teal color
                bar_id = canvas.create_rectangle(x, y_top, x + bar_width, y_bottom, 
                                                fill=color_light, outline="")
                
                # Highlight on top edge
                canvas.create_line(x, y_top, x + bar_width, y_top, 
                                  fill=COLOR_TEAL, width=1)
                
                bars_list.append({
                    'id': bar_id, 'hour': hour, 'count': count, 'max_val': max_val,
                    'x1': x, 'y1': y_top, 'x2': x + bar_width, 'y2': y_bottom
                })
        
        # Draw hour labels at bottom (every 4 hours: 6, 10, 14, 18, 22)
        for i, hour in enumerate(hours):
            if hour % 4 == 2 or hour == 6 or hour == 22:  # Show 6, 10, 14, 18, 22
                x = left_padding + i * bar_spacing + bar_spacing / 2
                canvas.create_text(x, height - 2, text=str(hour), 
                                  fill="#888888", font=("Arial", 7), anchor="s")
        
        # Bind hover events for axis labels
        canvas.bind("<Motion>", lambda e, c=canvas, b=bars_list_name, mv=max_val, hrs=hours: self._on_hist_hover(e, c, b))
        canvas.bind("<Leave>", lambda e, c=canvas: c.delete("tooltip"))
    
    def _on_hist_hover(self, event, canvas, bars_list_name):
        """Show tooltip when hovering over a histogram bar"""
        canvas.delete("tooltip")
        
        bars_list = getattr(self, bars_list_name, [])
        
        for bar in bars_list:
            if bar['x1'] <= event.x <= bar['x2'] and bar['y1'] <= event.y <= bar['y2']:
                hour = bar['hour']
                count = bar['count']
                tooltip_text = f"{count}w @ {hour}:00"
                
                tx = (bar['x1'] + bar['x2']) / 2
                ty = max(15, bar['y1'] - 10)
                
                text_id = canvas.create_text(tx, ty, text=tooltip_text, fill="#fff", 
                                            font=("Arial", 8, "bold"), tags="tooltip")
                bbox = canvas.bbox(text_id)
                if bbox:
                    canvas.create_rectangle(bbox[0] - 3, bbox[1] - 2,
                                           bbox[2] + 3, bbox[3] + 2,
                                           fill="#333", outline="#555", tags="tooltip")
                    canvas.tag_raise(text_id)
                break

    def draw_histogram_with_errorbars(self, canvas, stats):
        """Draw histogram with teal gradient bars and std dev error bars"""
        canvas.delete("all")
        self.histogram_bars_alltime = []
        
        canvas.update_idletasks()
        width = canvas.winfo_width() or 200
        height = canvas.winfo_height() or 100
        
        # Use same hour range as today histogram for alignment (6am to 10pm)
        hours = list(range(6, 23))  # 6am to 10pm (matching today)
        num_bars = len(hours)
        
        # Padding - extra space at bottom for hour labels
        left_padding = 8
        right_padding = 8
        top_padding = 8
        bottom_padding = 18  # Extra space for hour labels
        
        bar_area_width = width - left_padding - right_padding
        bar_area_height = height - top_padding - bottom_padding
        bar_width = max(8, (bar_area_width / num_bars) - 3)
        bar_spacing = bar_area_width / num_bars
        
        # Get max value (avg + std) for scaling
        max_val = 1
        for hour in hours:
            if hour in stats:
                top = stats[hour]['avg'] + stats[hour]['std']
                if top > max_val:
                    max_val = top
        
        # Draw bars with teal gradient effect
        for i, hour in enumerate(hours):
            stat = stats.get(hour, {'avg': 0, 'std': 0, 'count': 0})
            avg = stat['avg']
            std = stat['std']
            
            x = left_padding + i * bar_spacing + bar_spacing / 2 - bar_width / 2
            y_bottom = height - bottom_padding
            
            # Calculate bar height based on average
            bar_height = (avg / max_val) * bar_area_height if max_val > 0 else 0
            y_top = y_bottom - max(bar_height, 2) if avg > 0 else y_bottom
            
            if avg > 0:
                # Teal gradient: lighter top, darker bottom
                intensity = avg / max_val if max_val > 0 else 0
                
                # Base teal colors
                r_light = int(45 + intensity * 30)
                g_light = int(212 - intensity * 40)
                b_light = int(191 - intensity * 30)
                
                r_dark = int(19 + intensity * 20)
                g_dark = int(78 - intensity * 20)
                b_dark = int(74 - intensity * 20)
                
                color_light = f"#{r_light:02x}{g_light:02x}{b_light:02x}"
                color_dark = f"#{r_dark:02x}{g_dark:02x}{b_dark:02x}"
                
                # Draw 3D effect - darker base
                canvas.create_rectangle(x + 2, y_top + 2, x + bar_width + 2, y_bottom + 2, 
                                        fill=color_dark, outline="")
                
                # Main bar with teal color
                bar_id = canvas.create_rectangle(x, y_top, x + bar_width, y_bottom, 
                                                fill=color_light, outline="")
                
                # Highlight on top edge
                canvas.create_line(x, y_top, x + bar_width, y_top, 
                                  fill=COLOR_TEAL, width=1)
                
                self.histogram_bars_alltime.append({
                    'id': bar_id, 'hour': hour, 
                    'avg': avg, 'std': std, 'count': stat['count'], 'max_val': max_val,
                    'x1': x, 'y1': y_top, 'x2': x + bar_width, 'y2': y_bottom
                })
        
        # Draw hour labels at bottom (every 4 hours: 6, 10, 14, 18, 22)
        for i, hour in enumerate(hours):
            if hour % 4 == 2 or hour == 6 or hour == 22:  # Show 6, 10, 14, 18, 22
                x = left_padding + i * bar_spacing + bar_spacing / 2
                canvas.create_text(x, height - 2, text=str(hour), 
                                  fill="#888888", font=("Arial", 7), anchor="s")
        
        # Bind hover for stats tooltip
        canvas.bind("<Motion>", self._on_alltime_hist_hover)
        canvas.bind("<Leave>", lambda e: canvas.delete("tooltip"))

    def _on_alltime_hist_hover(self, event):
        """Show tooltip with avg±std when hovering over all-time histogram bar"""
        canvas = self.histogram_canvas_alltime
        canvas.delete("tooltip")
        
        for bar in self.histogram_bars_alltime:
            if bar['x1'] <= event.x <= bar['x2'] and bar['y1'] <= event.y <= bar['y2']:
                hour = bar['hour']
                avg = bar['avg']
                std = bar['std']
                count = bar['count']
                tooltip_text = f"{avg:.0f}±{std:.0f} words @ {hour}:00\n({count} days)"
                
                tx = (bar['x1'] + bar['x2']) / 2
                ty = max(20, bar['y1'] - 15)
                
                text_id = canvas.create_text(tx, ty, text=tooltip_text, fill="#fff", 
                                            font=("Arial", 7), tags="tooltip", justify="center")
                bbox = canvas.bbox(text_id)
                if bbox:
                    canvas.create_rectangle(bbox[0] - 3, bbox[1] - 2,
                                           bbox[2] + 3, bbox[3] + 2,
                                           fill="#333", outline="#555", tags="tooltip")
                    canvas.tag_raise(text_id)
                break


    def update_gui_loop(self):
        # 1. Logs
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            self.text_area.config(state='normal')
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.text_area.insert('1.0', f"[{timestamp}] {msg}\n")
            self.text_area.config(state='disabled')

        # 2. Results
        while not self.result_queue.empty():
            text_entry = self.result_queue.get()
            clean_text = text_entry.replace("[LIVE] ", "").replace("[BATCH] ", "").replace("[BACKUP] ", "")
            
            # --- COUNT WORDS ---
            if clean_text:
                count = len(clean_text.split())
                self.total_words += count
                self.today_words += count
                self.record_hourly_words(count)  # Track hourly activity
                self.save_stats()  # Persist stats after each transcription
                self.save_transcription(clean_text, self.last_transcription_latency)  # Save with latency
                self.last_transcription_latency = 0  # Reset after use
                self.update_histogram()  # Refresh histogram display
            # -------------------

            # Log the result to system log (simpler format now)
            self.text_area.config(state='normal')
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            mode_tag = "LIVE" if "[LIVE]" in text_entry else ("BATCH" if "[BATCH]" in text_entry else "BACKUP")
            self.text_area.insert('1.0', f"[{timestamp}] ✅ {mode_tag}: {len(clean_text.split())} words transcribed\n")
            self.text_area.config(state='disabled')

        # 3. Update Main VU Meter (CTkProgressBar uses 0-1 range)
        display_val = min(1.0, self.current_display_volume * 20)  # Normalize to 0-1
        self.vu_meter.set(display_val)

        # --- UPDATE ALL STATISTICS ---
        # All-time stats
        alltime_mins_saved = self.total_words / TYPING_WPM
        alltime_spoken_mins = self.total_audio_duration / 60.0
        
        # Today's session stats
        today_mins_saved = self.today_words / TYPING_WPM
        today_spoken_mins = self.today_audio_duration / 60.0
        
        # Calculate Dictation Speed (WPM) - all time and session
        alltime_speed = 0
        if alltime_spoken_mins > 0:
            alltime_speed = int(self.total_words / alltime_spoken_mins)
        
        session_speed = 0
        if today_spoken_mins > 0:
            session_speed = int(self.today_words / today_spoken_mins)

        # Update Session Statistics (Today) - values only since we have icon labels
        self.lbl_session_words.config(text=f"{self.today_words:,}")
        self.lbl_session_time.config(text=f"{self.format_time_saved(today_mins_saved)}")
        self.lbl_session_duration.config(text=f"{self.format_speaking_time(self.today_audio_duration)}")
        self.lbl_session_speed.config(text=f"{session_speed} WPM")
        
        # Calculate average latency
        session_avg_latency = sum(self.session_latencies) / len(self.session_latencies) if self.session_latencies else 0
        alltime_avg_latency = sum(self.total_latencies[-100:]) / len(self.total_latencies[-100:]) if self.total_latencies else 0
        
        # Update latency display if labels exist
        if hasattr(self, 'lbl_session_latency'):
            self.lbl_session_latency.config(text=f"⚡{session_avg_latency:.0f}ms")
        if hasattr(self, 'lbl_alltime_latency'):
            self.lbl_alltime_latency.config(text=f"⚡{alltime_avg_latency:.0f}ms")
        
        # Update All-Time Statistics - values only
        self.lbl_alltime_words.config(text=f"{self.total_words:,}")
        self.lbl_alltime_time.config(text=f"{self.format_time_saved(alltime_mins_saved)}")
        self.lbl_alltime_duration.config(text=f"{self.format_speaking_time(self.total_audio_duration)}")
        self.lbl_alltime_speed.config(text=f"{alltime_speed} WPM")

        # Update Mini Window with TODAY's stats only
        if self.mini_window and self.mini_canvas:
            self.mini_canvas.delete("vu_bar")
            bar_width = min(150, display_val * 3)
            self.mini_canvas.create_rectangle(0, 35, bar_width, 40, fill="white", tags="vu_bar")
            
            status_txt = "Whisper Idle"
            if self.mode == "live": status_txt = "Live Mode"
            elif self.mode == "batch":
                # Show batch chunk progress in mini overlay
                with self.batch_segment_lock:
                    total = len(self.batch_segments)
                    done = sum(1 for s in self.batch_segments if s['status'] == 'done')
                status_txt = f"Batch ({done}/{total})" if total > 0 else "Batch Mode"
            
            # Show today's stats in mini overlay
            stats_txt = f"{self.today_words}w | -{self.format_time_saved(today_mins_saved)}"
            self.mini_canvas.itemconfigure("status_text", text=status_txt)
            self.mini_canvas.itemconfigure("stats_text", text=stats_txt)

        self.root.after(30, self.update_gui_loop)

    def load_model(self):
        try:
            friendly_name = self.config['model_key']
            model_id = MODEL_MAP.get(friendly_name, "medium")
            
            # --- 1. REFRESH DROPDOWN INDICATORS ---
            # This ensures we see the arrow/checkmark correctly on startup/reload
            self.root.after(0, self.refresh_model_list)
            
            self.log_internal(f"Init {friendly_name}...")
            
            safe_name = friendly_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            local_models_root = os.path.join(os.getcwd(), "models")
            target_dir = os.path.join(local_models_root, safe_name)
            
            # --- 2. DETERMINE BACKEND ---
            backend_name = self.config.get('backend', 'auto')
            device = self.config.get('device', 'auto')
            compute_type = self.config.get('compute_type', 'auto')
            
            # Use backend abstraction if available
            if BACKENDS_AVAILABLE and backend_name != 'legacy':
                # Auto-detect best backend if needed
                if backend_name == 'auto' or device == 'auto':
                    detected_backend, detected_device, reason = detect_best_backend()
                    if backend_name == 'auto':
                        backend_name = detected_backend
                    if device == 'auto':
                        device = detected_device
                    self.log_internal(f"[Auto-detect] {reason}")
                
                # Create backend instance
                self.backend = create_backend(backend_name)
                self.current_backend_name = backend_name
                self.current_device = device
                
                # Resolve compute type
                if compute_type == 'auto':
                    if hasattr(self.backend, 'get_optimal_compute_type'):
                        compute_type = self.backend.get_optimal_compute_type(device)
                    else:
                        compute_type = 'int8' if device == 'cpu' else 'float16'
                
                self.log_internal(f"Loading {backend_name} backend ({device}, {compute_type})...")
                
                # Progress bar for download
                if not self.is_model_downloaded(friendly_name) and snapshot_download:
                    self.log_internal("Downloading model (this may take a while)...")
                    self.root.after(0, lambda: self._show_progress("Downloading..."))
                
                # Load model via backend - with fallback on failure
                try:
                    self.backend.load_model(
                        model_key=friendly_name,
                        device=device,
                        compute_type=compute_type,
                        model_path=target_dir
                    )
                    
                    # For compatibility with existing transcription code
                    self.model = self.backend.model if hasattr(self.backend, 'model') else self.backend
                    
                    self.root.after(0, self._hide_progress)
                    self.root.after(0, self.refresh_model_list)
                    self.log_internal(f"Model Ready. [{backend_name.upper()} / {device.upper()}]")
                    self.status_var.set(f"Ready [{backend_name} / {device}]")
                    return
                except Exception as backend_error:
                    # Backend failed - fall back to faster-whisper
                    self.log_internal(f"⚠️ {backend_name} failed: {backend_error}")
                    self.log_internal("Falling back to faster-whisper (CPU)...")
                    self.root.after(0, self._hide_progress)
                    # Fall through to legacy code below
            
            # --- FALLBACK: Original WhisperModel code (legacy mode) ---
            self.backend = None
            self.current_backend_name = 'faster-whisper'
            self.current_device = self.config['device']
            
            # --- 2. PROGRESS BAR SETUP ---
            # Check if likely already downloaded first to avoid flashing progress bar
            if not self.is_model_downloaded(friendly_name) and snapshot_download:
                self.log_internal("Downloading model (this may take a while)...")
                
                # Show progress bar in indeterminate mode (animated)
                def show_progress():
                    self.progress_bar.configure(mode="indeterminate")
                    self.progress_frame.pack(fill="x", padx=5, pady=0)
                    self.progress_bar.start(15)  # Animation speed
                    self.progress_label.configure(text="Downloading...")
                
                def hide_progress():
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate", value=0)
                    self.progress_frame.pack_forget()
                
                self.root.after(0, show_progress)
                
                try:
                   repo_id = model_id
                   if "/" not in model_id: repo_id = f"systran/faster-whisper-{model_id}"
                   
                   snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
                finally:
                   # Hide progress bar
                   self.root.after(0, hide_progress)

            # ----------------------------------
            
            if not os.path.exists(target_dir) or not os.listdir(target_dir):
                self.migrate_from_cache(model_id, target_dir)

            os.makedirs(target_dir, exist_ok=True)
            
            # Check if model files exist directly in target_dir (from snapshot_download)
            # Look for model.bin which is the main model file
            model_bin_path = os.path.join(target_dir, "model.bin")
            if os.path.exists(model_bin_path):
                # Model was downloaded via snapshot_download directly into target_dir
                # Load from the local path directly
                self.log_internal(f"Loading model from local path...")
                self.model = WhisperModel(
                    target_dir,  # Use the directory path directly
                    device=self.config['device'], 
                    compute_type=self.config['compute_type']
                )
            else:
                # Fallback: model needs to be downloaded or is in cache format
                self.model = WhisperModel(
                    model_id, 
                    device=self.config['device'], 
                    compute_type=self.config['compute_type'],
                    download_root=target_dir 
                )
            
            # Refresh again to show checkmark now that it's downloaded
            self.root.after(0, self.refresh_model_list)
            
            self.log_internal("Model Ready.")
            self.status_var.set("Ready.")
        except Exception as e:
            error_str = str(e).lower()
            self.log_internal(f"Load Error: {e}")
            self.root.after(0, lambda: self.progress_frame.pack_forget())
            
            # Check for CUDA/cuDNN specific errors
            cuda_error_keywords = ['cudnn', 'cublas', 'cuda', 'nvrtc', 'could not load library', 'dll']
            is_cuda_error = any(kw in error_str for kw in cuda_error_keywords)
            
            if is_cuda_error and self.config.get('device') == 'cuda':
                # Schedule dialog on main thread
                self.root.after(0, lambda err=e: self._show_cuda_error_dialog(str(err)))
            else:
                import traceback
                traceback.print_exc()

    def migrate_from_cache(self, model_id, target_dir):
        try:
            repo_id = model_id
            if "/" not in model_id: repo_id = f"systran/faster-whisper-{model_id}"
            cache_name = f"models--{repo_id.replace('/', '--')}"
            default_cache_path = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", cache_name))
            
            if os.path.exists(default_cache_path):
                snapshots_dir = os.path.join(default_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = os.listdir(snapshots_dir)
                    if snapshots:
                        source_dir = os.path.join(snapshots_dir, snapshots[-1])
                        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        except Exception as e:
            self.log_internal(f"Migration skip: {e}")

    def _show_cuda_error_dialog(self, error_msg):
        """Show a helpful dialog when CUDA/cuDNN fails to load"""
        CUDA_DLL_URL = "https://github.com/Purfview/whisper-standalone-win/releases/tag/libs"
        
        dialog = tk.Toplevel(self.root)
        dialog.title("CUDA Libraries Required")
        dialog.geometry("500x320")
        dialog.configure(bg=COLOR_BG)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 250
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 160
        dialog.geometry(f"+{x}+{y}")
        
        # Title
        tk.Label(dialog, text="⚠️ CUDA Libraries Not Found", 
                bg=COLOR_BG, fg="#FFA726", font=("Arial", 14, "bold")).pack(pady=(15, 10))
        
        # Explanation
        explanation = (
            "GPU acceleration requires NVIDIA cuDNN and cuBLAS libraries.\n\n"
            "To enable GPU mode:\n"
            "1. Click 'Download CUDA DLLs' below\n"
            "2. Download the archive for your system\n"
            "3. Extract DLL files to the WhisperTyper folder\n"
            "4. Restart the application\n\n"
            "Or use CPU mode (slower but works without GPU)."
        )
        tk.Label(dialog, text=explanation, bg=COLOR_BG, fg=COLOR_FG, 
                font=("Arial", 10), justify="left", wraplength=450).pack(padx=20, pady=5)
        
        # Error details (collapsible)
        error_frame = ttk.LabelFrame(dialog, text="Error Details")
        error_frame.pack(fill="x", padx=20, pady=10)
        error_label = tk.Label(error_frame, text=error_msg[:200] + "..." if len(error_msg) > 200 else error_msg,
                              bg=COLOR_BG, fg="#888", font=("Consolas", 8), wraplength=440)
        error_label.pack(padx=10, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=15)
        
        def open_download():
            webbrowser.open(CUDA_DLL_URL)
        
        def use_cpu():
            dialog.destroy()
            self._switch_to_cpu_mode()
        
        tk.Button(btn_frame, text="📥 Download CUDA DLLs", command=open_download,
                 bg="#1976d2", fg="white", font=("Arial", 10, "bold"), 
                 padx=15, pady=8, cursor="hand2").pack(side="left", padx=10)
        
        tk.Button(btn_frame, text="🖥️ Use CPU Mode", command=use_cpu,
                 bg=COLOR_ACCENT_BG, fg="white", font=("Arial", 10), 
                 padx=15, pady=8, cursor="hand2").pack(side="left", padx=10)
        
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy,
                 bg=COLOR_IDLE, fg="white", font=("Arial", 10), 
                 padx=15, pady=8).pack(side="left", padx=10)

    def _switch_to_cpu_mode(self):
        """Switch to CPU mode and reload the model"""
        self.log_internal("Switching to CPU mode...")
        self.status_var.set("Switching to CPU mode...")
        
        # Update config
        self.config['device'] = 'cpu'
        self.config['compute_type'] = 'int8'  # Best for CPU
        
        # Save config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)
        
        # Reload model on CPU
        threading.Thread(target=self.load_model, daemon=True).start()

    def audio_callback(self, indata, frames, time, status):
        if self.mode is not None:
            self.audio_queue.put(indata.copy())

    def restart_audio_stream(self):
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        dev_id = self.config.get('input_device')
        try:
            self.stream = sd.InputStream(device=dev_id, samplerate=16000, channels=1, callback=self.audio_callback, blocksize=1600)
            self.stream.start()
        except:
            self.log_internal("Audio Device Fail.")

if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        app.running = False