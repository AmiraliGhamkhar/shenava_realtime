"""
Configuration for Shenava Real-time ASR with Overlay and Injector
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
from pathlib import Path


class OutputMode(Enum):
    """Output modes for transcription"""
    OVERLAY_ONLY = "overlay"
    INJECT_ONLY = "inject"
    BOTH = "both"
    CLIPBOARD = "clipboard"
    CONSOLE = "console"


class OverlayPosition(Enum):
    """Overlay window positions"""
    TOP = "top"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    CUSTOM = "custom"


class InjectorMode(Enum):
    """Text injection modes"""
    KEYBOARD_SIMULATION = "keyboard"  # Simulate typing
    CLIPBOARD_PASTE = "clipboard"     # Copy to clipboard and paste
    DIRECT_INPUT = "direct"           # Direct input (Windows only)


@dataclass
class AudioConfig:
    """Audio capture configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024  # Samples per chunk
    buffer_duration: float = 1.0  # Seconds of audio before processing
    overlap_duration: float = 0.2  # Overlap between buffers
    
    # VAD (Voice Activity Detection)
    # RMS thresholds tuned to a measured ambient floor of ~0.008 p99;
    # typical speech lands at 0.02-0.2 RMS
    vad_threshold: float = 0.02
    vad_min_speech: float = 0.1  # Minimum speech duration
    vad_min_silence: float = 0.3  # Minimum silence to trigger processing (lower = faster finalize)
    
    # Noise gate
    noise_gate_threshold: float = 0.01
    apply_noise_reduction: bool = True
    
    # Maximum speech segment before forced processing (mirrors ASRConfig.max_buffer_duration)
    max_buffer_duration: float = 15.0
    
    # Minimum seconds of speech before emitting a partial transcription
    partial_buffer_duration: float = 0.5


@dataclass
class ASRConfig:
    """ASR model configuration"""
    model_name: str = "Reza2kn/Shenava-Koochik-v1.0"
    # Local .nemo file; takes precedence over model_name when set
    model_path: Optional[str] = r"C:\Users\ali\Desktop\localASR-shenava\shenava-koochik\shenava-koochik-v1.0.nemo"
    device: str = "cuda"  # Will auto-detect
    context_size: List[int] = field(default_factory=lambda: [70, 13])
    
    # Streaming settings
    streaming_chunk_duration: float = 1.0  # Process every 1 second
    streaming_overlap: float = 0.1
    max_buffer_duration: float = 30.0  # Maximum audio buffer
    
    # Post-processing
    apply_itn: bool = True
    remove_repetitions: bool = True
    confidence_threshold: float = 0.5
    
    # CPU inference tuning (measured: 4 threads optimal on 8-core; CTC decode ~16x faster than RNNT
    # and is the decode path used by this model's published deployment exports)
    num_threads: int = 4
    decoder_type: str = "ctc"  # "ctc" or "rnnt"


@dataclass
class OverlayConfig:
    """Overlay window configuration"""
    enabled: bool = True
    position: OverlayPosition = OverlayPosition.BOTTOM
    custom_position: Tuple[int, int] = (100, 100)
    
    # Window settings
    width: int = 600
    height: int = 80
    opacity: float = 0.9
    always_on_top: bool = True
    click_through: bool = False
    borderless: bool = True
    
    # Text settings
    font_family: str = "Vazirmatn"
    font_size: int = 16
    text_color: str = "#FFFFFF"
    background_color: str = "#1E1E1E"
    border_color: str = "#4A9EFF"
    border_width: int = 2
    
    # Animation
    fade_duration: int = 200  # milliseconds
    max_lines: int = 2
    scroll_speed: int = 50  # pixels per second
    
    # Behavior
    show_confidence: bool = False
    show_partial: bool = True  # Show partial results
    auto_hide_delay: float = 3.0  # Hide after silence


@dataclass
class InjectorConfig:
    """Text injector configuration"""
    enabled: bool = True
    mode: InjectorMode = InjectorMode.KEYBOARD_SIMULATION
    
    # Injection settings
    delay_between_keys: float = 0.01  # Seconds between keystrokes
    delay_between_words: float = 0.05
    inject_on_complete: bool = True  # Only inject complete sentences
    inject_partial: bool = False
    
    # Target window (optional)
    target_window: Optional[str] = None  # Window title to inject into
    target_process: Optional[str] = None  # Process name
    
    # Keyboard settings
    layout: str = "persian"  # Keyboard layout
    use_unicode: bool = True
    
    # Special keys
    send_enter_after: bool = False
    send_space_after: bool = True


@dataclass
class HotkeyConfig:
    """Global hotkey configuration"""
    toggle_recording: str = "ctrl+alt+r"  # Start/stop recording
    toggle_overlay: str = "ctrl+alt+o"  # Show/hide overlay
    toggle_injector: str = "ctrl+alt+i"  # Enable/disable injection
    clear_transcript: str = "ctrl+alt+c"  # Clear current transcript
    emergency_stop: str = "ctrl+alt+q"  # Emergency stop
    
    # Mode switching
    cycle_output_mode: str = "ctrl+alt+m"
    cycle_injector_mode: str = "ctrl+alt+n"


@dataclass
class AppConfig:
    """Main application configuration"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    injector: InjectorConfig = field(default_factory=InjectorConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    
    # General settings
    output_mode: OutputMode = OutputMode.BOTH
    debug: bool = False
    log_file: str = "shenava_realtime.log"
    save_transcripts: bool = True
    transcripts_dir: str = "transcripts"
    
    # State
    is_recording: bool = False
    overlay_visible: bool = True
    injector_active: bool = True


class ConfigManager:
    """Manage configuration loading/saving"""
    
    @staticmethod
    def load(config_path: Optional[Path] = None) -> AppConfig:
        """Load configuration from file or create default"""
        if config_path and config_path.exists():
            import json
            with open(config_path, 'r') as f:
                data = json.load(f)
            return AppConfig(**data)
        return AppConfig()
    
    @staticmethod
    def save(config: AppConfig, config_path: Path):
        """Save configuration to file"""
        import json
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=str)