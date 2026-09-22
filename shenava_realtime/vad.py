"""Energy (RMS) based voice activity detection with speech/silence hysteresis.

A small explicit state machine — no dependencies beyond numpy, no model::

    SILENCE --rms >= onset_rms--> PENDING --speech >= min_speech_ms--> SPEECH
    SPEECH  --rms <  offset_rms for >= min_silence_ms----------------> SILENCE

``onset_rms`` is higher than ``offset_rms`` (hysteresis) so that quiet word
endings do not chop a sentence in half, while ``min_speech_ms``/``min_silence_ms``
filter clicks and short breaths.  A bounded pre-roll ring buffer keeps the first
phonemes of an utterance, which are otherwise lost while the detector is still
in SILENCE, and a short hangover keeps the last frames of speech after the
silence threshold is met so word-final stops are never clipped.

The class is deliberately free of I/O and threads: feed it audio frames and it
returns events, which makes the transitions directly unit-testable.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VADState(str, Enum):
    SILENCE = "silence"
    PENDING = "pending"
    SPEECH = "speech"


class EventType(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VADConfig:
    sample_rate: int = 16000
    onset_rms: float = 0.015
    offset_rms: float = 0.008
    min_speech_ms: int = 250
    min_silence_ms: int = 700
    pre_speech_ms: int = 320
    max_speech_s: float = 20.0
    # Frames of speech kept after min_silence is reached so the tail of the
    # final word is inside the segment (word-final stops). 0 preserves the
    # endpoint timing that earlier callers pinned; the production AudioConfig
    # default is 150 ms.
    hangover_ms: int = 0
    # Opt-in: samples duplicated at the head of the next segment at a forced
    # cap cut, so the second decode keeps acoustic context across the split.
    max_speech_overlap_s: float = 0.0
    # Opt-in spectral gate: noise-like frames (flat spectrum) cannot start a
    # speech segment. Speech has formant structure (peaked spectrum); with the
    # gate on, a PENDING transition additionally requires the frame not to be
    # flat.
    spectral_gate: bool = False
    spectral_flatness_threshold: float = 0.8

    # Adaptive noise floor (bounded).  ``onset_rms``/``offset_rms`` remain the
    # lower bound; the adaptive thresholds are noise_floor * snr, clamped into
    # [configured threshold, configured threshold * adaptive_max_gain].
    adaptive: bool = True
    noise_init_ms: int = 500        # quiet audio used to seed the floor
    noise_halflife_ms: int = 2000   # decay of the floor estimate in silence
    onset_snr: float = 4.0
    offset_snr: float = 2.0
    adaptive_max_gain: float = 8.0

    def __post_init__(self) -> None:
        if self.adaptive:
            adaptive_values = (self.onset_snr, self.offset_snr,
                               self.adaptive_max_gain, self.noise_init_ms,
                               self.noise_halflife_ms)
            if not all(math.isfinite(v) for v in adaptive_values):
                raise ValueError("Adaptive VAD settings must be finite")
            if self.onset_snr <= 0 or self.offset_snr <= 0:
                raise ValueError("VAD SNR factors must be positive")
            if self.offset_snr > self.onset_snr:
                raise ValueError("offset_snr must be <= onset_snr for hysteresis to work")
            if not 1.0 <= self.adaptive_max_gain <= 64.0:
                raise ValueError("adaptive_max_gain must be within [1, 64]")
            if not 0 < self.noise_init_ms <= 5000:
                raise ValueError("noise_init_ms must be within (0, 5000] ms")
            if not 0 < self.noise_halflife_ms <= 60000:
                raise ValueError("noise_halflife_ms must be within (0, 60000] ms")
        values = (self.onset_rms, self.offset_rms, self.max_speech_s,
                  self.min_speech_ms, self.min_silence_ms, self.pre_speech_ms)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("VAD settings must be finite")
        if min(self.min_speech_ms, self.min_silence_ms) <= 0 or self.pre_speech_ms < 0:
            raise ValueError("Invalid VAD durations")
        if self.pre_speech_ms > 2000 or self.max_speech_s > 120 or max(self.min_speech_ms, self.min_silence_ms) > 10000:
            raise ValueError("VAD durations exceed bounded realtime limits")
        if self.max_speech_s <= self.min_speech_ms / 1000:
            raise ValueError("Maximum segment must exceed onset duration")
        if self.onset_rms <= 0 or self.offset_rms <= 0:
            raise ValueError("VAD thresholds must be positive")
        if self.offset_rms > self.onset_rms:
            raise ValueError("offset_rms must be <= onset_rms for hysteresis to work")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not math.isfinite(self.max_speech_overlap_s) or not (
            0 <= self.max_speech_overlap_s <= self.max_speech_s
        ):
            raise ValueError(
                "max_speech_overlap_s must be finite and within [0, max_speech_s]"
            )
        if not math.isfinite(self.spectral_flatness_threshold) or not (
            0 < self.spectral_flatness_threshold < 1
        ):
            raise ValueError("spectral_flatness_threshold must be within (0, 1)")
        if self.hangover_ms < 0 or self.hangover_ms > self.min_silence_ms:
            raise ValueError("hangover_ms must be within [0, min_silence_ms]")

    def _ms_to_samples(self, milliseconds: float) -> int:
        return max(1, int(round(self.sample_rate * milliseconds / 1000.0)))

    @property
    def pre_speech_samples(self) -> int:
        return self._ms_to_samples(self.pre_speech_ms)

    @property
    def min_speech_samples(self) -> int:
        return self._ms_to_samples(self.min_speech_ms)

    @property
    def min_silence_samples(self) -> int:
        return self._ms_to_samples(self.min_silence_ms)

    @property
    def noise_init_samples(self) -> int:
        return self._ms_to_samples(self.noise_init_ms)

    @property
    def max_speech_samples(self) -> int:
        return max(self.min_speech_samples + 1, int(round(self.sample_rate * self.max_speech_s)))

    @property
    def hangover_samples(self) -> int:
        return self._ms_to_samples(self.hangover_ms)

    @property
    def max_speech_overlap_samples(self) -> int:
        return int(round(self.sample_rate * self.max_speech_overlap_s))


def spectral_flatness(frame: np.ndarray) -> float:
    """Spectral flatness (geometric/arithmetic mean of the power spectrum).

    Near 1.0 for flat, noise-like spectra (keyboard, equipment beeps); lower
    for spectra with formant structure, i.e. speech. Returns 1.0 for silent
    frames so the gate stays out of the way when there is no signal to judge.
    """
    if frame is None or frame.size == 0:
        return 1.0
    spectrum = np.abs(np.fft.rfft(frame))
    power = np.square(spectrum, dtype=np.float64) + 1e-12
    log_power = np.log(power)
    geometric = float(np.exp(log_power.mean()))
    arithmetic = float(power.mean())
    return float(_clamp(geometric / arithmetic, 0.0, 1.0))


@dataclass
class VADEvent:
    type: EventType
    audio: Optional[np.ndarray] = None
    duration_s: float = 0.0
    rms: float = 0.0
    # True only for a segment-cap cut (keep_speaking): the endpoint is an
    # artifact of the 20 s limit, not a natural phrase boundary.
    forced: bool = False


# Nominal frame used to size the noise window when the caller's block size is
# unknown; the window length is bounded in frames either way.
_NOMINAL_FRAME_MS = 64.0
# Hard bound on the retained window, so memory cannot grow with session length.
MAX_NOISE_WINDOW_FRAMES = 256


def _window_frames(config: "VADConfig") -> int:
    """Window length in frames: ``noise_halflife_ms`` worth, bounded."""
    frames = int(round(config.noise_halflife_ms / _NOMINAL_FRAME_MS))
    return max(2, min(MAX_NOISE_WINDOW_FRAMES, frames))


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def rms(frame: np.ndarray) -> float:
    """Root-mean-square level of a frame (0.0 for an empty frame)."""
    if frame is None or frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


class EnergyVAD:
    """Stateful RMS voice activity detector."""

    def __init__(self, config: Optional[VADConfig] = None) -> None:
        self.config = config or VADConfig()
        self.state: VADState = VADState.SILENCE
        self._preroll: Deque[np.ndarray] = deque()
        self._preroll_samples = 0
        self._segment: List[np.ndarray] = []
        self._segment_samples = 0
        self._silence_samples = 0
        self._speech_samples = 0
        self.last_rms: float = 0.0
        # Long-monologue cap bookkeeping (soft warning state, issue #19).
        self._soft_warning_emitted = False
        # Overlap audio carried from the previous forced cut into the next
        # segment head (issue #19; opt-in via config.max_speech_overlap_s).
        self._pending_overlap: Optional[np.ndarray] = None
        self._pending_overlap_samples = 0
        # Adaptive noise floor: bounded window of recent non-speech frame
        # levels; the floor is their minimum (see _update_noise_floor).
        self._noise_window: Deque[float] = deque()
        self._noise_window_frames = _window_frames(self.config)
        self._noise_floor: float = self.config.offset_rms / max(1e-9, self.config.offset_snr)
        self._noise_samples = 0
        self.onset_threshold: float = self.config.onset_rms
        self.offset_threshold: float = self.config.offset_rms

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.state = VADState.SILENCE
        self._preroll.clear()
        self._preroll_samples = 0
        self._segment.clear()
        self._segment_samples = 0
        self._silence_samples = 0
        self._speech_samples = 0
        self._soft_warning_emitted = False
        self._pending_overlap = None
        self._pending_overlap_samples = 0
        self.last_rms = 0.0
        self._noise_window.clear()
        self._noise_floor = self.config.offset_rms / max(1e-9, self.config.offset_snr)
        self._noise_samples = 0
        self.onset_threshold = self.config.onset_rms
        self.offset_threshold = self.config.offset_rms

    @property
    def in_speech(self) -> bool:
        return self.state is VADState.SPEECH

    @property
    def speech_duration_s(self) -> float:
        return self._segment_samples / float(self.config.sample_rate)

    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray) -> List[VADEvent]:
        """Feed one audio frame (float32 mono) and return any state events."""
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return []
        if frame.size > self.config.sample_rate:
            raise ValueError("Feed VAD blocks of at most one second")

        if not np.isfinite(frame).all():
            raise ValueError("Non-finite audio frame")
        frame = frame.copy()
        level = rms(frame)
        self.last_rms = level
        self._update_noise_floor(level, frame.size)
        is_speech = level >= (
            self.onset_threshold if self.state is VADState.SILENCE else self.offset_threshold
        )
        size = frame.size
        events: List[VADEvent] = []

        if self.state is VADState.SILENCE:
            if is_speech:
                # Opt-in spectral gate: noise-like frames cannot start a
                # speech segment (they never carry formant structure).
                if (self.config.spectral_gate
                        and spectral_flatness(frame) >= self.config.spectral_flatness_threshold):
                    self._push_preroll(frame)
                    return events
                self.state = VADState.PENDING
                self._speech_samples = size
                self._segment = [frame]
                self._segment_samples = size
                self._silence_samples = 0
                if self._speech_samples >= self.config.min_speech_samples:
                    events.append(self._start_speech())
            else:
                self._push_preroll(frame)
            return events

        if self.state is VADState.PENDING:
            self._segment.append(frame)
            self._segment_samples += size
            if is_speech:
                self._speech_samples += size
                self._silence_samples = 0
            else:
                self._silence_samples += size
            if self._speech_samples >= self.config.min_speech_samples:
                events.append(self._start_speech())
            elif self._silence_samples >= self.config.min_silence_samples:
                # Short burst that never became speech: drop it.
                self._drop_segment()
            elif self._segment_samples >= self.config.max_speech_samples:
                # The segment cap hit while still unconfirmed. Promote only
                # when the buffered audio is speech-dominant overall — a
                # speaker whose level oscillates around the offset threshold
                # (truncation bug #6) — otherwise this was a noise burst
                # (click, breath) and is dropped as before.
                if 2 * self._speech_samples >= self._segment_samples:
                    events.append(self._start_speech())
                else:
                    self._drop_segment()
            return events

        # SPEECH
        self._segment.append(frame)
        self._segment_samples += size
        if not is_speech:
            self._silence_samples += size
            # Natural endpoint: the silence must also cover the hangover
            # window. Every frame is appended above, so the extra hangover
            # audio is inside the segment (no duplication); if speech resumes
            # within it the endpoint is cancelled and a soft word-final
            # consonant that dipped below the offset threshold is kept.
            # hangover_ms = 0 preserves the endpoint timing exactly.
            if self._silence_samples >= self.config.min_silence_samples + self.config.hangover_samples:
                events.append(self._end_speech())
                return events
        else:
            self._silence_samples = 0
            self._soft_warning_emitted = False
        if self._segment_samples >= self.config.max_speech_samples:
            # Long monologue: cut a segment without losing the following audio.
            if not self._soft_warning_emitted:
                self._soft_warning_emitted = True
                logger.warning(
                    "VAD segment nearing the %.0fs cap (%.1fs spoken); "
                    "a forced cut will split this utterance",
                    self.config.max_speech_s,
                    self.speech_duration_s,
                )
            events.append(self._end_speech(keep_speaking=True))
            start_event = VADEvent(EventType.SPEECH_START, audio=np.empty(0, np.float32))
            # Opt-in continuity (config.max_speech_overlap_s): duplicate the
            # configured overlap into the next segment's head so the decoder
            # does not restart cold across the forced cut. The next segment
            # receives it through the START event's audio (as pre-roll).
            if self._pending_overlap is not None and self._pending_overlap.size:
                start_event.audio = self._pending_overlap
                self._pending_overlap = None
                self._pending_overlap_samples = 0
            events.append(start_event)
        return events

    def flush(self) -> List[VADEvent]:
        """Close an open segment (called when capture stops mid-utterance)."""
        if self.state is not VADState.SPEECH or not self._segment:
            self.reset()
            return []
        return [self._end_speech()]

    # ------------------------------------------------------------------ #
    def _update_noise_floor(self, level: float, size: int) -> None:
        """Track the local noise floor and derive bounded onset/offset levels.

        The estimate is *minimum statistics*: the floor is the smallest frame
        level seen in a bounded recent window of non-speech frames, not their
        average.  That distinction is what makes the detector safe for a quiet
        speaker — modulated speech dips between syllables, so its minimum stays
        near the true background, while steady room noise does not dip and
        correctly raises the floor.

        Only frames observed in SILENCE feed the window, so an open segment can
        never drag the floor up under the speaker.  Until ``noise_init_ms`` of
        background has been measured — or while the floor is below the
        configured static offset — the configured thresholds are used
        unchanged, so a quiet room, a quiet speaker and a speaker who starts
        talking immediately all behave exactly as before.  Derived thresholds are clamped into ``[configured, configured *
        adaptive_max_gain]``: bounded, and identical to the static detector in
        clean audio.
        """
        config = self.config
        if not config.adaptive:
            return
        if self.state is VADState.SILENCE:
            self._noise_window.append(max(level, 1e-6))
            self._noise_samples += size
            while len(self._noise_window) > self._noise_window_frames:
                self._noise_window.popleft()
            self._noise_floor = min(self._noise_window)
        # Adaptation is one-directional and only engages in a genuinely noisy
        # room: while the measured floor sits below the configured static
        # offset the static thresholds are used verbatim.  A quiet room (or an
        # unmeasured one) therefore behaves exactly like the previous
        # detector, and a quiet speaker can never be raised out of range by
        # adaptation — only real background noise moves the thresholds, and
        # only upward, within the configured clamp.
        if (self._noise_samples < config.noise_init_samples
                or self._noise_floor <= config.offset_rms):
            self.onset_threshold = config.onset_rms
            self.offset_threshold = config.offset_rms
            return
        self.onset_threshold = _clamp(
            self._noise_floor * config.onset_snr,
            config.onset_rms,
            config.onset_rms * config.adaptive_max_gain,
        )
        self.offset_threshold = _clamp(
            self._noise_floor * config.offset_snr,
            config.offset_rms,
            config.offset_rms * config.adaptive_max_gain,
        )
        if self.offset_threshold > self.onset_threshold:
            self.offset_threshold = self.onset_threshold

    @property
    def noise_floor(self) -> float:
        """Current noise-floor estimate (diagnostics/tests)."""
        return self._noise_floor

    # ------------------------------------------------------------------ #
    def _push_preroll(self, frame: np.ndarray) -> None:
        self._preroll.append(frame)
        self._preroll_samples += frame.size
        limit = max(0, int(self.config.pre_speech_ms * self.config.sample_rate / 1000))
        while self._preroll_samples > limit and len(self._preroll) > 1:
            self._preroll_samples -= self._preroll.popleft().size
        if self._preroll_samples > limit:
            excess = self._preroll_samples - limit
            self._preroll[0] = self._preroll[0][excess:].copy()
            self._preroll_samples = limit

    def _start_speech(self) -> VADEvent:
        audio = np.concatenate([*self._preroll, *self._segment]) if self._preroll else np.concatenate(self._segment)
        self._preroll.clear()
        self._preroll_samples = 0
        self.state = VADState.SPEECH
        self._silence_samples = 0
        return VADEvent(
            type=EventType.SPEECH_START,
            audio=audio,
            duration_s=audio.size / float(self.config.sample_rate),
            rms=self.last_rms,
        )

    def _end_speech(self, keep_speaking: bool = False) -> VADEvent:
        audio = np.concatenate(self._segment)
        duration = audio.size / float(self.config.sample_rate)
        if keep_speaking:
            # Opt-in continuity: stash the configured overlap so the next
            # START event can carry it into the following segment's head.
            overlap_samples = self.config.max_speech_overlap_samples
            if overlap_samples and audio.size:
                take = min(overlap_samples, audio.size)
                self._pending_overlap = audio[-take:].copy()
                self._pending_overlap_samples = take
            else:
                self._pending_overlap = None
                self._pending_overlap_samples = 0
        self._segment = []
        self._segment_samples = 0
        self._silence_samples = 0
        self._speech_samples = 0
        self.state = VADState.SPEECH if keep_speaking else VADState.SILENCE
        return VADEvent(
            type=EventType.SPEECH_END,
            audio=audio,
            duration_s=duration,
            rms=self.last_rms,
            forced=keep_speaking,
        )

    def _drop_segment(self) -> None:
        self._segment = []
        self._segment_samples = 0
        self._silence_samples = 0
        self._speech_samples = 0
        self.state = VADState.SILENCE

    # ------------------------------------------------------------------ #
    def frames_for(self, milliseconds: float) -> int:
        """Helper for callers/tests: samples covered by ``milliseconds``."""
        return self.config._ms_to_samples(milliseconds)
