"""Small, typed configuration for the Shenava application."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class OutputMode(str, Enum):
    OVERLAY_ONLY = "overlay"
    INJECT_ONLY = "inject"
    BOTH = "both"
    CLIPBOARD = "clipboard"
    CONSOLE = "console"


class OverlayPosition(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    CUSTOM = "custom"


class InjectorMode(str, Enum):
    KEYBOARD_SIMULATION = "keyboard"
    CLIPBOARD_PASTE = "clipboard"
    DIRECT_INPUT = "direct"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    buffer_duration: float = 1.0
    overlap_duration: float = 0.2
    vad_threshold: float = 0.02
    vad_min_speech: float = 0.1
    vad_min_silence: float = 0.3
    noise_gate_threshold: float = 0.01
    apply_noise_reduction: bool = True
    max_buffer_duration: float = 15.0
    partial_buffer_duration: float = 0.5


@dataclass
class ASRConfig:
    model_name: str = "Reza2kn/Shenava-Koochik-v1.0"
    # Set SHENAVA_MODEL_PATH or this field to use an offline local model.
    model_path: Optional[str] = field(default_factory=lambda: os.getenv("SHENAVA_MODEL_PATH"))
    device: str = "cuda"
    context_size: list[int] = field(default_factory=lambda: [70, 13])
    streaming_chunk_duration: float = 1.0
    streaming_overlap: float = 0.1
    max_buffer_duration: float = 30.0
    apply_itn: bool = True
    remove_repetitions: bool = True
    confidence_threshold: float = 0.5
    num_threads: int = 4
    decoder_type: str = "ctc"


@dataclass
class OverlayConfig:
    enabled: bool = True
    position: OverlayPosition = OverlayPosition.BOTTOM
    custom_position: Tuple[int, int] = (100, 100)
    width: int = 600
    height: int = 80
    opacity: float = 0.9
    always_on_top: bool = True
    click_through: bool = False
    borderless: bool = True
    font_family: str = "Vazirmatn"
    font_size: int = 16
    text_color: str = "#FFFFFF"
    background_color: str = "#1E1E1E"
    border_color: str = "#4A9EFF"
    border_width: int = 2
    fade_duration: int = 200
    max_lines: int = 2
    scroll_speed: int = 50
    show_confidence: bool = False
    show_partial: bool = True
    auto_hide_delay: float = 3.0


@dataclass
class InjectorConfig:
    enabled: bool = True
    mode: InjectorMode = InjectorMode.KEYBOARD_SIMULATION
    delay_between_keys: float = 0.01
    delay_between_words: float = 0.05
    inject_on_complete: bool = True
    inject_partial: bool = False
    target_window: Optional[str] = None
    target_process: Optional[str] = None
    layout: str = "persian"
    use_unicode: bool = True
    send_enter_after: bool = False
    send_space_after: bool = True


@dataclass
class HotkeyConfig:
    toggle_recording: str = "ctrl+alt+r"
    toggle_overlay: str = "ctrl+alt+o"
    toggle_injector: str = "ctrl+alt+i"
    clear_transcript: str = "ctrl+alt+c"
    emergency_stop: str = "ctrl+alt+q"
    cycle_output_mode: str = "ctrl+alt+m"
    cycle_injector_mode: str = "ctrl+alt+n"


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    injector: InjectorConfig = field(default_factory=InjectorConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    output_mode: OutputMode = OutputMode.BOTH
    debug: bool = False
    log_file: str = "shenava_realtime.log"
    save_transcripts: bool = True
    transcripts_dir: str = "transcripts"
    is_recording: bool = False
    overlay_visible: bool = True
    injector_active: bool = True


class ConfigManager:
    """Load and save a complete configuration without losing nested settings."""

    @staticmethod
    def load(config_path: Optional[Path] = None) -> AppConfig:
        if not config_path or not config_path.exists():
            return AppConfig()
        with config_path.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Configuration root must be a JSON object")

        def section(cls, name: str):
            values = data.get(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"Configuration section '{name}' must be an object")
            if name == "position" or name == "mode":
                return values
            if name == "overlay":
                values["position"] = OverlayPosition(values.get("position", OverlayPosition.BOTTOM))
            if name == "injector":
                values["mode"] = InjectorMode(values.get("mode", InjectorMode.KEYBOARD_SIMULATION))
            return cls(**values)

        return AppConfig(
            audio=section(AudioConfig, "audio"),
            asr=section(ASRConfig, "asr"),
            overlay=section(OverlayConfig, "overlay"),
            injector=section(InjectorConfig, "injector"),
            hotkeys=section(HotkeyConfig, "hotkeys"),
            output_mode=OutputMode(data.get("output_mode", OutputMode.BOTH)),
            **{key: data[key] for key in ("debug", "log_file", "save_transcripts", "transcripts_dir", "is_recording", "overlay_visible", "injector_active") if key in data},
        )

    @staticmethod
    def save(config: AppConfig, config_path: Path):
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False, default=lambda value: value.value if isinstance(value, Enum) else str(value)), encoding="utf-8")
