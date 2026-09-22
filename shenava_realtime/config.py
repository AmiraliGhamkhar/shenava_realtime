"""Typed configuration for the Shenava real-time ASR app.

All runtime knobs live here as small dataclasses.  A handful of them can be
overridden with environment variables (optionally read from a local ``.env``
file) so the app can be retargeted without editing code:

``SHENAVA_ASR_BACKEND`` (unprefixed ``ASR_BACKEND`` also accepted),
``SHENAVA_MODEL_PATH``, ``SHENAVA_TOKENS_PATH``, ``SHENAVA_DEVICE``,
``SHENAVA_NUM_THREADS``, ``SHENAVA_SAMPLE_RATE``, ``SHENAVA_FEATURE_DIM``,
``SHENAVA_DECODING_METHOD``, ``SHENAVA_SECOND_PASS``,
``SHENAVA_REQUIRE_STREAMING``, ``SHENAVA_OUTPUT_MODE``,
``SHENAVA_INJECTOR_MODE``, ``SHENAVA_AUDIO_DEVICE``,
``SHENAVA_VAD_ADAPTIVE``, ``SHENAVA_VAD_MIN_SPEECH_MS``,
``SHENAVA_VAD_MIN_SILENCE_MS``, ``SHENAVA_VAD_PRE_SPEECH_MS``,
``SHENAVA_VAD_MAX_SPEECH_S``, ``SHENAVA_VAD_HANGOVER_MS``,
``SHENAVA_VAD_SPECTRAL_GATE``, ``SHENAVA_VAD_MAX_SPEECH_OVERLAP_S``,
``SHENAVA_SECOND_PASS_BEAM``, ``SHENAVA_LINGUISTIC_ENDPOINTING`` and
``SHENAVA_LOG_LEVEL``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
# Canonical on-disk location for the provisioned sherpa-onnx model (see
# models/shenava/README.md for the download command and pinned revision).
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "shenava"
DEFAULT_MODEL_FILE = DEFAULT_MODEL_DIR / "model.int8.onnx"
DEFAULT_TOKENS_FILE = DEFAULT_MODEL_DIR / "tokens.txt"
# CTC greedy search is the only decoding method the runtime path supports.
DECODING_METHODS = ("greedy_search",)
DEFAULT_DECODING_METHOD = "greedy_search"


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
def _read_env_pairs(text: str):
    """Yield ``(key, value)`` pairs from ``.env`` text (quoted values kept).

    Handles the cases the naive ``strip().strip(quote)`` parser got wrong:

    * values that contain ``=`` or ``#`` inside quotes;
    * escaped quotes and backslashes inside double-quoted values
      (single-quoted values stay verbatim, POSIX-style);
    * values spanning several lines inside an unterminated quote;
    * unquoted values lose an inline ``# comment`` (with a preceding space),
      while a ``#`` glued to the value is kept (hex colours, URLs).
    """
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        key, raw = key.strip(), raw.strip()
        if not key:
            continue
        if raw[:1] in ('"', "'"):
            quote = raw[0]
            chunk = raw[1:]
            while True:
                end = chunk.find(quote)
                if end >= 0:
                    value = chunk[:end]
                    break  # anything after the closing quote is ignored
                if index >= len(lines):
                    value = chunk  # unterminated quote: take what is there
                    break
                chunk += "\n" + lines[index]
                index += 1
            if quote == '"':
                value = value.replace('\\"', '"').replace("\\\\", "\\")
            yield key, value
        else:
            if " #" in raw:
                raw = raw.split(" #", 1)[0].rstrip()
            yield key, raw


def load_dotenv(path: Optional[Path] = None) -> None:
    """Populate ``os.environ`` from a ``.env`` file without overriding exports."""
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        for key, value in _read_env_pairs(env_path.read_text(encoding="utf-8")):
            if key not in os.environ:
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
    # 150 ms keeps very short medical terms ("سی تی", "IV") while still
    # filtering clicks and breaths; 600 ms + hangover endpoints Persian
    # dictation without the old 0.7 s dead pause between sentences.
    vad_min_speech_ms: int = 150
    vad_min_silence_ms: int = 600
    vad_pre_speech_ms: int = 400
    vad_max_speech_s: float = 20.0
    # Speech framing kept after the silence threshold is met, so word-final
    # stops are never clipped by the endpoint decision.
    vad_hangover_ms: int = 150
    # Opt-in: audio duplicated into the next segment at a forced cap cut
    # (decoder continuity). Default 0 preserves the clinical invariant that
    # every sample lands in exactly one segment.
    vad_max_speech_overlap_s: float = 0.0
    # Opt-in spectral flatness gate: noise-like frames (keyboard, equipment
    # beeps) cannot start a speech segment when enabled.
    vad_spectral_gate: bool = False
    vad_spectral_flatness_threshold: float = 0.8

    # Adaptive noise floor: onset/offset are derived from the measured floor
    # and clamped into [threshold, threshold * vad_adaptive_max_gain].
    vad_adaptive: bool = True
    vad_noise_init_ms: int = 500
    # 4 s halflife keeps equipment transients (monitor beeps, ventilator
    # cycles) from dragging the floor up within a single noise episode.
    vad_noise_halflife_ms: int = 4000
    vad_onset_snr: float = 4.0
    vad_offset_snr: float = 2.0
    vad_adaptive_max_gain: float = 8.0

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
                  max_speech_s=self.vad_max_speech_s, adaptive=self.vad_adaptive,
                  noise_init_ms=self.vad_noise_init_ms,
                  noise_halflife_ms=self.vad_noise_halflife_ms,
                  onset_snr=self.vad_onset_snr, offset_snr=self.vad_offset_snr,
                  adaptive_max_gain=self.vad_adaptive_max_gain,
                  hangover_ms=self.vad_hangover_ms,
                  max_speech_overlap_s=self.vad_max_speech_overlap_s,
                  spectral_gate=self.vad_spectral_gate,
                  spectral_flatness_threshold=self.vad_spectral_flatness_threshold)

    @property
    def chunk_duration(self) -> float:
        return self.chunk_size / float(self.sample_rate)


@dataclass
class ASRConfig:
    """Sherpa-ONNX streaming CTC (Shenava-Koochik-v1.0), CPU/INT8 only."""

    # Which backend implementation to construct. Sherpa-ONNX CTC is the only
    # supported value; kept explicit (rather than hard-coded) so tests and
    # tooling can name it and so an unsupported value fails clearly.
    asr_backend: str = "sherpa_onnx_ctc"
    model_path: Optional[str] = field(
        default_factory=lambda: _env("SHENAVA_MODEL_PATH") or str(DEFAULT_MODEL_FILE)
    )
    tokens_path: Optional[str] = field(
        default_factory=lambda: _env("SHENAVA_TOKENS_PATH") or str(DEFAULT_TOKENS_FILE)
    )
    device: str = "cpu"  # sherpa-onnx CPU provider only; no GPU in this backend
    num_threads: int = 4
    sample_rate: int = 16000
    feature_dim: int = 80
    decoding_method: str = DEFAULT_DECODING_METHOD
    # Uncalibrated confidence threshold. The decoder exposes a stability
    # proxy (see asr_backend.SherpaStream), not a calibrated probability, so
    # this never *filters* text; when > 0 an utterance below it is flagged
    # with the "low_confidence" review reason instead.
    confidence_threshold: float = 0.0
    # Production invariant: the desktop app must fail at startup unless the
    # loaded recognizer behaves as an online streaming recognizer.  Endpoint
    # fallback remains available only for explicit tests/tools by setting this
    # to False (or SHENAVA_REQUIRE_STREAMING=0).
    require_streaming: bool = True
    max_segment_s: float = 22.0
    commit_on_endpoint: bool = True

    # How often to hand new audio to the streaming recognizer. 0.25 s keeps
    # partials responsive for dictation (the first decode of an utterance
    # comes even sooner; see streaming.CacheAwareDecoder).
    partial_interval_s: float = 0.25
    # First decode of an utterance after this much pending audio (responsive
    # silence-to-speech transition); later decodes wait partial_interval_s.
    first_decode_s: float = 0.15
    use_cache_aware_streaming: bool = True

    # Utterance-end second pass (natural endpoints only, endpoint-commit mode):
    # "off" = streaming greedy only; "greedy" = one offline greedy re-decode
    # via the same recognizer. "context" (CTC beam + hotword biasing) is not
    # supported by the sherpa-onnx backend: it needs raw per-frame emissions
    # and the model's own tokenizer object, neither exposed by sherpa-onnx's
    # public Python API, and is rejected at startup with a clear message.
    second_pass: str = "greedy"
    second_pass_min_utterance_s: float = 0.5
    # Opt-in: build a second recognizer with sherpa's modified_beam_search
    # for the utterance-end second pass. Model-dependent: when the runtime
    # refuses it, the greedy second pass stays active (explicit fallback).
    second_pass_beam: bool = False
    # Opt-in dual endpointing: the model's own linguistic endpointer may close
    # an utterance while the VAD still reports speech. The RMS VAD remains the
    # primary endpointer; this only adds a second, content-aware signal.
    linguistic_endpointing: bool = False
    # Benchmark mode is offline-only (tools/): synthetic RTF/throughput runs.
    benchmark_mode: bool = False

    def __post_init__(self) -> None:
        if self.asr_backend != "sherpa_onnx_ctc":
            raise ValueError(
                f"asr_backend must be 'sherpa_onnx_ctc', got {self.asr_backend!r}"
            )
        if (self.device or "").strip().lower() != "cpu":
            raise ValueError(
                "The sherpa-onnx backend is CPU-only; device must be 'cpu' "
                f"(got {self.device!r}). GPU/CUDA is not supported by this "
                "migration."
            )
        if self.sample_rate != 16000:
            raise ValueError("Shenava requires a 16 kHz mono frontend")
        if not 1 <= int(self.feature_dim) <= 512:
            raise ValueError("feature_dim must be within [1, 512]")
        if self.decoding_method not in DECODING_METHODS:
            raise ValueError(
                f"decoding_method must be one of {', '.join(DECODING_METHODS)}"
            )
        if not 0 < self.partial_interval_s <= 5 or not 0 < self.max_segment_s <= 120:
            raise ValueError("Invalid ASR interval or segment limit")
        if not math.isfinite(self.first_decode_s) or not (
            0 <= self.first_decode_s <= self.partial_interval_s
        ):
            raise ValueError(
                "first_decode_s must be within [0, partial_interval_s]"
            )
        if self.num_threads < 1 or self.holdback_words < 0:
            raise ValueError("Invalid thread count or holdback")
        if self.second_pass not in ("off", "greedy", "context"):
            raise ValueError('second_pass must be "off", "greedy" or "context"')
        if self.second_pass == "context":
            raise ValueError(
                'second_pass="context" (CTC beam + hotword biasing over raw '
                "emissions) is not supported by the sherpa-onnx backend: "
                "sherpa-onnx's public Python API exposes only decoded text, "
                "not per-frame emissions or the model's tokenizer object. "
                'Use second_pass="greedy" or "off".'
            )
        if not 0 < self.second_pass_min_utterance_s <= 5:
            raise ValueError("second_pass_min_utterance_s must be within (0, 5] seconds")

    # Transcript stabilization: words kept un-committed until they stop moving.
    holdback_words: int = 2
    # Scale the holdback with hypothesis length (see stabilizer.StabilizerConfig):
    # short dictations are injected sooner, long monologues stay conservative.
    adaptive_holdback: bool = True


@dataclass
class PostProcessConfig:
    """Deterministic FST/rule post-processing switches."""

    enabled: bool = True
    normalize_unicode: bool = True
    # CTC repetition artifacts are common and the FST already protects
    # legitimate doubles ("سی سی"), so this is on by default.
    remove_repetitions: bool = True
    convert_numbers: bool = True
    digits: DigitStyle = DigitStyle.ASCII
    medical_terms: bool = True
    units: bool = True
    punctuation: bool = True
    # Opt-in: append sentence-final punctuation to completed dictation (CTC
    # models emit none). Intended for endpoint-commit mode; early-commit
    # injects per delta and would punctuate every fragment.
    restore_punctuation: bool = False
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


# Legacy NeMo-era environment variables that no longer apply. Set is a clear
# migration error rather than a silently ignored knob.
_REMOVED_ENV_VARS = {
    "SHENAVA_RIGHT_CONTEXT": "streaming right-context is fixed by the ONNX export",
    "SHENAVA_MODEL_NAME": "network model provisioning was removed; set SHENAVA_MODEL_PATH "
                          "and SHENAVA_TOKENS_PATH to a locally provisioned sherpa-onnx model "
                          "(see models/shenava/README.md)",
    "SHENAVA_ALLOW_DOWNLOAD": "automatic downloads were removed; provision the model with the "
                              "documented `hf download` command in models/shenava/README.md",
    "SHENAVA_DECODER": "the RNNT head was removed; the sherpa-onnx backend is CTC-only",
    "SHENAVA_BEAM_SIZE": "the CTC beam + hotword-bias second pass was removed "
                         "(sherpa-onnx exposes no raw emissions); use SHENAVA_SECOND_PASS=greedy",
    "SHENAVA_HOTWORD_ALIASES": "decoder-time hotword biasing was removed with the beam second pass",
    "SHENAVA_HOTWORD_PHONETIC": "decoder-time hotword biasing was removed with the beam second pass",
    "SHENAVA_CUDA_GRAPH_STREAMING": "CUDA is not supported by the sherpa-onnx CPU backend",
}


def _check_removed_env_vars() -> None:
    for name, message in _REMOVED_ENV_VARS.items():
        if _env(name) is not None:
            raise ValueError(f"{name} is no longer supported: {message}")


def apply_env_overrides(config: AppConfig) -> AppConfig:
    """Apply the documented ``SHENAVA_*`` environment overrides in place."""
    _check_removed_env_vars()

    asr_backend = _env("SHENAVA_ASR_BACKEND") or _env("ASR_BACKEND")
    if asr_backend:
        config.asr.asr_backend = asr_backend
    model_path = _env("SHENAVA_MODEL_PATH")
    if model_path:
        if model_path.endswith(".nemo"):
            raise ValueError(
                f"SHENAVA_MODEL_PATH={model_path!r} points at a legacy .nemo "
                "checkpoint. The runtime now loads a sherpa-onnx model: set "
                "SHENAVA_MODEL_PATH to model.int8.onnx and SHENAVA_TOKENS_PATH "
                "to tokens.txt (see models/shenava/README.md)."
            )
        config.asr.model_path = model_path
    tokens_path = _env("SHENAVA_TOKENS_PATH")
    if tokens_path:
        config.asr.tokens_path = tokens_path
    device = _env("SHENAVA_DEVICE")
    if device:
        config.asr.device = device
    config.asr.num_threads = _env_int("SHENAVA_NUM_THREADS", config.asr.num_threads)
    config.asr.sample_rate = _env_int("SHENAVA_SAMPLE_RATE", config.asr.sample_rate)
    config.asr.feature_dim = _env_int("SHENAVA_FEATURE_DIM", config.asr.feature_dim)
    decoding_method = _env("SHENAVA_DECODING_METHOD")
    if decoding_method:
        config.asr.decoding_method = decoding_method
    config.asr.partial_interval_s = _env_float(
        "SHENAVA_PARTIAL_INTERVAL_S", config.asr.partial_interval_s
    )
    config.asr.require_streaming = _env_bool("SHENAVA_REQUIRE_STREAMING", config.asr.require_streaming)
    second_pass = _env("SHENAVA_SECOND_PASS")
    if second_pass:
        config.asr.second_pass = second_pass.strip().lower()
    config.asr.benchmark_mode = _env_bool("SHENAVA_BENCHMARK_MODE", config.asr.benchmark_mode)
    config.audio.vad_adaptive = _env_bool("SHENAVA_VAD_ADAPTIVE", config.audio.vad_adaptive)
    config.audio.vad_min_speech_ms = _env_int("SHENAVA_VAD_MIN_SPEECH_MS", config.audio.vad_min_speech_ms)
    config.audio.vad_min_silence_ms = _env_int("SHENAVA_VAD_MIN_SILENCE_MS", config.audio.vad_min_silence_ms)
    config.audio.vad_pre_speech_ms = _env_int("SHENAVA_VAD_PRE_SPEECH_MS", config.audio.vad_pre_speech_ms)
    config.audio.vad_max_speech_s = _env_float("SHENAVA_VAD_MAX_SPEECH_S", config.audio.vad_max_speech_s)
    config.audio.vad_hangover_ms = _env_int("SHENAVA_VAD_HANGOVER_MS", config.audio.vad_hangover_ms)
    config.audio.vad_max_speech_overlap_s = _env_float(
        "SHENAVA_VAD_MAX_SPEECH_OVERLAP_S", config.audio.vad_max_speech_overlap_s
    )
    config.audio.vad_spectral_gate = _env_bool("SHENAVA_VAD_SPECTRAL_GATE", config.audio.vad_spectral_gate)
    config.asr.second_pass_beam = _env_bool("SHENAVA_SECOND_PASS_BEAM", config.asr.second_pass_beam)
    config.asr.linguistic_endpointing = _env_bool(
        "SHENAVA_LINGUISTIC_ENDPOINTING", config.asr.linguistic_endpointing
    )
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
