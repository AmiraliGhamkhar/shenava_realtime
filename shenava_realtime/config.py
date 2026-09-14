"""Typed configuration for the Shenava real-time ASR app.

All runtime knobs live here as small dataclasses.  A handful of them can be
overridden with environment variables (optionally read from a local ``.env``
file) so the app can be retargeted without editing code:

``SHENAVA_MODEL_PATH``, ``SHENAVA_MODEL_NAME``, ``SHENAVA_DEVICE``,
``SHENAVA_DECODER``, ``SHENAVA_NUM_THREADS``, ``SHENAVA_OUTPUT_MODE``,
``SHENAVA_INJECTOR_MODE``, ``SHENAVA_AUDIO_DEVICE``, ``SHENAVA_LOG_LEVEL``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_FILE = REPO_ROOT / "shenava-koochik" / "shenava-koochik-v1.0.nemo"
DEFAULT_MODEL_NAME = "Reza2kn/Shenava-Koochik-v1.0"


class OutputMode(str, Enum):
    OVERLAY_ONLY = "overlay"
    INJECT_ONLY = "inject"
    BOTH = "both"
    CLIPBOARD = "clipboard"
    CONSOLE = "console"


class InjectorMode(str, Enum):
    AUTO = "auto"
    CLIPBOARD = "clipboard"
    KEYBOARD = "keyboard"


class OverlayPosition(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    CUSTOM = "custom"


class DigitStyle(str, Enum):
    ASCII = "ascii"
    PERSIAN = "persian"


# --------------------------------------------------------------------------- #
# .env handling (tiny loader; avoids a python-dotenv dependency)
# --------------------------------------------------------------------------- #
def load_dotenv(path: Optional[Path] = None) -> None:
    """Populate ``os.environ`` from a ``.env`` file without overriding exports."""
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        logger.warning("Could not read %s: %s", env_path, exc)


def _env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else None


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r (not a number)", name, raw)
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r (not an integer)", name, raw)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class AudioConfig:
    """16 kHz mono capture + RMS VAD tuning."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024  # 64 ms at 16 kHz
    device: Optional[Any] = None  # sounddevice input device index/name

    # Voice activity detection (hysteresis: onset > offset)
    vad_onset_rms: float = 0.015
    vad_offset_rms: float = 0.008
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 700
    vad_pre_speech_ms: int = 320
    vad_max_speech_s: float = 20.0

    # Bounded capture queue: oldest chunk is dropped when the consumer stalls
    queue_max_chunks: int = 32

    # Heartbeat: the mic callback pushes blocks continuously, so a gap this
    # long means the input device went silent/disconnected (not merely quiet).
    dropout_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if self.sample_rate != 16000 or self.channels != 1:
            raise ValueError("Shenava requires 16000 Hz mono audio")
        if not 1 <= self.chunk_size <= 16000 or not 1 <= self.queue_max_chunks <= 1024:
            raise ValueError("Invalid audio block/queue size")
        if not 0 < self.dropout_timeout_s <= 60:
            raise ValueError("dropout_timeout_s must be within (0, 60] seconds")
        from .vad import VADConfig
        VADConfig(sample_rate=self.sample_rate, onset_rms=self.vad_onset_rms,
                  offset_rms=self.vad_offset_rms, min_speech_ms=self.vad_min_speech_ms,
                  min_silence_ms=self.vad_min_silence_ms, pre_speech_ms=self.vad_pre_speech_ms,
                  max_speech_s=self.vad_max_speech_s)

    @property
    def chunk_duration(self) -> float:
        return self.chunk_size / float(self.sample_rate)


@dataclass
class ASRConfig:
    """Shenava CTC, native encoder lookahead and bounded endpoint fallback."""

    model_name: str = DEFAULT_MODEL_NAME
    model_path: Optional[str] = field(
        default_factory=lambda: _env("SHENAVA_MODEL_PATH") or str(DEFAULT_MODEL_FILE)
    )
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:1"
    decoder_type: str = "ctc"
    num_threads: int = 4
    confidence_threshold: float = 0.0  # deprecated; uncalibrated scores are not filtered
    allow_download: bool = False
    require_streaming: bool = False
    right_context: int = 13
    max_segment_s: float = 22.0
    commit_on_endpoint: bool = True

    # How often to hand new audio to the native stream (encoder chunk sizes
    # come from checkpoint metadata). Legacy window knobs are ignored.
    partial_interval_s: float = 0.5
    left_context_s: float = 2.0
    max_window_s: float = 10.0
    use_cache_aware_streaming: bool = True

    def __post_init__(self) -> None:
        if self.decoder_type != "ctc":
            raise ValueError("Only greedy CTC decoding is supported")
        if self.right_context not in (0, 1, 6, 13):
            raise ValueError("right_context must be 0, 1, 6 or 13")
        if not 0 < self.partial_interval_s <= 5 or not 0 < self.max_segment_s <= 120:
            raise ValueError("Invalid ASR interval or segment limit")
        if self.num_threads < 1 or self.holdback_words < 0:
            raise ValueError("Invalid thread count or holdback")

    # Transcript stabilization: words kept un-committed until they stop moving.
    holdback_words: int = 2


@dataclass
class PostProcessConfig:
    """Deterministic FST/rule post-processing switches."""

    enabled: bool = True
    normalize_unicode: bool = True
    remove_repetitions: bool = False
    convert_numbers: bool = True
    digits: DigitStyle = DigitStyle.ASCII
    medical_terms: bool = True
    units: bool = True
    punctuation: bool = True
    join_persian_affixes: bool = True


@dataclass
class OverlayConfig:
    enabled: bool = True
    position: OverlayPosition = OverlayPosition.BOTTOM
    custom_position: Tuple[int, int] = (100, 100)
    width: int = 620
    height: int = 90
    opacity: float = 0.92
    always_on_top: bool = True
    font_family: str = "Vazirmatn"
    font_size: int = 16
    text_color: str = "#FFFFFF"
    background_color: str = "#1E1E1E"
    border_color: str = "#4A9EFF"
    max_chars: int = 200
    show_confidence: bool = False
    show_partial: bool = True
    auto_hide_delay: float = 4.0


@dataclass
class InjectorConfig:
    enabled: bool = True
    mode: InjectorMode = InjectorMode.AUTO
    delay_between_keys: float = 0.0
    delay_after_paste_s: float = 0.05
    restore_clipboard: bool = True
    send_space_after: bool = False
    send_enter_after: bool = False
    skip_consecutive_duplicates: bool = False


@dataclass
class HotkeyConfig:
    toggle_recording: str = "ctrl+alt+r"
    toggle_overlay: str = "ctrl+alt+o"
    toggle_injector: str = "ctrl+alt+i"
    clear_transcript: str = "ctrl+alt+c"
    emergency_stop: str = "ctrl+alt+q"


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    injector: InjectorConfig = field(default_factory=InjectorConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    output_mode: OutputMode = OutputMode.BOTH
    debug: bool = False
    log_level: str = "INFO"
    save_transcripts: bool = False
    transcripts_dir: str = "transcripts"
    clinical_sqlite: Optional[str] = None
    clinical_jsonl: Optional[str] = None

    @classmethod
    def from_env(cls, config_path: Optional[Path] = None) -> "AppConfig":
        """Build a config from defaults, an optional JSON file, then env vars."""
        load_dotenv()
        config = ConfigManager.load(config_path) if config_path else cls()
        apply_env_overrides(config)
        return config


def apply_env_overrides(config: AppConfig) -> AppConfig:
    """Apply the documented ``SHENAVA_*`` environment overrides in place."""
    model_path = _env("SHENAVA_MODEL_PATH")
    if model_path:
        config.asr.model_path = model_path
    model_name = _env("SHENAVA_MODEL_NAME")
    if model_name:
        config.asr.model_name = model_name
    device = _env("SHENAVA_DEVICE")
    if device:
        config.asr.device = device
    decoder = _env("SHENAVA_DECODER")
    if decoder:
        config.asr.decoder_type = decoder
    config.asr.num_threads = _env_int("SHENAVA_NUM_THREADS", config.asr.num_threads)
    config.asr.partial_interval_s = _env_float(
        "SHENAVA_PARTIAL_INTERVAL_S", config.asr.partial_interval_s
    )
    config.asr.allow_download = _env_bool("SHENAVA_ALLOW_DOWNLOAD", config.asr.allow_download)
    config.asr.require_streaming = _env_bool("SHENAVA_REQUIRE_STREAMING", config.asr.require_streaming)
    config.asr.right_context = _env_int("SHENAVA_RIGHT_CONTEXT", config.asr.right_context)
    audio_device = _env("SHENAVA_AUDIO_DEVICE")
    if audio_device:
        config.audio.device = int(audio_device) if audio_device.isdigit() else audio_device
    config.asr.__post_init__()
    config.audio.__post_init__()

    output_mode = _env("SHENAVA_OUTPUT_MODE")
    if output_mode:
        try:
            config.output_mode = OutputMode(output_mode.lower())
        except ValueError:
            logger.warning(
                "Ignoring SHENAVA_OUTPUT_MODE=%r (expected one of %s)",
                output_mode,
                ", ".join(mode.value for mode in OutputMode),
            )

    injector_mode = _env("SHENAVA_INJECTOR_MODE")
    if injector_mode:
        try:
            config.injector.mode = InjectorMode(injector_mode.lower())
        except ValueError:
            logger.warning(
                "Ignoring SHENAVA_INJECTOR_MODE=%r (expected one of %s)",
                injector_mode,
                ", ".join(mode.value for mode in InjectorMode),
            )

    log_level = _env("SHENAVA_LOG_LEVEL")
    if log_level:
        config.log_level = log_level
    if _env_bool("SHENAVA_DEBUG", config.debug):
        config.debug = True
    return config


# --------------------------------------------------------------------------- #
# JSON persistence
# --------------------------------------------------------------------------- #
_ENUM_SECTIONS = {
    "overlay": ("position", OverlayPosition),
    "injector": ("mode", InjectorMode),
}


def _decode_enum(value: Any, enum_cls: type) -> Any:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


class ConfigManager:
    """Load and save a complete configuration without losing nested settings."""

    @staticmethod
    def load(config_path: Optional[Path] = None) -> AppConfig:
        if not config_path:
            return AppConfig()
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning("Config file %s not found; using defaults", config_path)
            return AppConfig()

        with config_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Configuration root must be a JSON object")

        mapping = {
            "audio": AudioConfig,
            "asr": ASRConfig,
            "postprocess": PostProcessConfig,
            "overlay": OverlayConfig,
            "injector": InjectorConfig,
            "hotkeys": HotkeyConfig,
        }
        kwargs: Dict[str, Any] = {}
        for name, cls in mapping.items():
            values = data.get(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"Configuration section '{name}' must be an object")
            known = {f.name for f in fields(cls)}
            unknown = set(values) - known
            if unknown:
                logger.warning("Ignoring unknown %s keys: %s", name, ", ".join(sorted(unknown)))
                values = {key: value for key, value in values.items() if key in known}
            if name in _ENUM_SECTIONS:
                key, enum_cls = _ENUM_SECTIONS[name]
                if key in values:
                    values[key] = _decode_enum(values[key], enum_cls)
            if name == "postprocess" and "digits" in values:
                values["digits"] = _decode_enum(values["digits"], DigitStyle)
            kwargs[name] = cls(**values)

        if "output_mode" in data:
            kwargs["output_mode"] = _decode_enum(data["output_mode"], OutputMode)
        for key in ("debug", "log_level", "save_transcripts", "transcripts_dir", "clinical_sqlite", "clinical_jsonl"):
            if key in data:
                kwargs[key] = data[key]
        return AppConfig(**kwargs)

    @staticmethod
    def save(config: AppConfig, config_path: Path) -> None:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        def default(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if is_dataclass(value):
                return asdict(value)
            return str(value)

        config_path.write_text(
            json.dumps(asdict(config), indent=2, ensure_ascii=False, default=default),
            encoding="utf-8",
        )
