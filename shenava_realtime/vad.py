"""Energy (RMS) based voice activity detection with speech/silence hysteresis.

A small explicit state machine — no dependencies beyond numpy, no model::

    SILENCE --rms >= onset_rms--> PENDING --speech >= min_speech_ms--> SPEECH
    SPEECH  --rms <  offset_rms for >= min_silence_ms----------------> SILENCE

``onset_rms`` is higher than ``offset_rms`` (hysteresis) so that quiet word
endings do not chop a sentence in half, while ``min_speech_ms``/``min_silence_ms``
filter clicks and short breaths.  A bounded pre-roll ring buffer keeps the first
phonemes of an utterance, which are otherwise lost while the detector is still
in SILENCE.

The class is deliberately free of I/O and threads: feed it audio frames and it
returns events, which makes the transitions directly unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


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

    def __post_init__(self) -> None:
        if self.onset_rms <= 0 or self.offset_rms <= 0:
            raise ValueError("VAD thresholds must be positive")
        if self.offset_rms > self.onset_rms:
            raise ValueError("offset_rms must be <= onset_rms for hysteresis to work")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

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
    def max_speech_samples(self) -> int:
        return max(self.min_speech_samples + 1, int(round(self.sample_rate * self.max_speech_s)))


@dataclass
class VADEvent:
    type: EventType
    audio: Optional[np.ndarray] = None
    duration_s: float = 0.0
    rms: float = 0.0


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
        self._preroll: List[np.ndarray] = []
        self._preroll_samples = 0
        self._segment: List[np.ndarray] = []
        self._segment_samples = 0
        self._silence_samples = 0
        self._speech_samples = 0
        self.last_rms: float = 0.0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.state = VADState.SILENCE
        self._preroll.clear()
        self._preroll_samples = 0
        self._segment.clear()
        self._segment_samples = 0
        self._silence_samples = 0
        self._speech_samples = 0
        self.last_rms = 0.0

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

        level = rms(frame)
        self.last_rms = level
        is_speech = level >= (
            self.config.onset_rms if self.state is VADState.SILENCE else self.config.offset_rms
        )
        size = frame.size
        events: List[VADEvent] = []

        if self.state is VADState.SILENCE:
            if is_speech:
                self.state = VADState.PENDING
                self._speech_samples = size
                self._segment = [frame]
                self._segment_samples = size
                self._silence_samples = 0
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
            return events

        # SPEECH
        self._segment.append(frame)
        self._segment_samples += size
        self._silence_samples = self._silence_samples + size if not is_speech else 0
        if self._silence_samples >= self.config.min_silence_samples:
            events.append(self._end_speech())
        elif self._segment_samples >= self.config.max_speech_samples:
            # Long monologue: cut a segment without losing the following audio.
            events.append(self._end_speech(keep_speaking=True))
        return events

    def flush(self) -> List[VADEvent]:
        """Close an open segment (called when capture stops mid-utterance)."""
        if self.state is VADState.SILENCE or not self._segment:
            self.reset()
            return []
        return [self._end_speech()]

    # ------------------------------------------------------------------ #
    def _push_preroll(self, frame: np.ndarray) -> None:
        self._preroll.append(frame)
        self._preroll_samples += frame.size
        limit = self.config.pre_speech_samples
        while self._preroll_samples > limit and len(self._preroll) > 1:
            self._preroll_samples -= self._preroll.pop(0).size

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
