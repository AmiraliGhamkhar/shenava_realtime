"""Incremental decoding strategies for the streaming ASR session.

Two implementations share one interface:

``WindowedDecoder``
    For checkpoints without a streaming config (the default for Shenava
    Koochik).  It keeps the utterance audio and decodes only a bounded window —
    ``left_context`` seconds of already-decoded audio plus everything new — so
    work per step stays constant instead of growing with the utterance.

``CacheAwareDecoder``
    For checkpoints that ship NeMo's streaming config: new audio is fed through
    ``CacheAwareStreamInfer``, which reuses the encoder/decoder caches.

Both return the hypothesis for the audio decoded so far, plus a ``reset`` flag
that tells the caller the window was cut (long monologue) and that the text
starts a fresh accumulation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .asr_backend import CacheAwareStream
from .config import ASRConfig

logger = logging.getLogger(__name__)

_EMPTY = np.zeros(0, dtype=np.float32)


@dataclass
class DecodeResult:
    text: str
    confidence: float = 0.0
    reset: bool = False
    decoded_seconds: float = 0.0


class StreamingDecoder(ABC):
    """Feeds audio in and returns the running hypothesis."""

    name = "decoder"

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def push(self, audio: np.ndarray, force: bool = False) -> Optional[DecodeResult]:
        ...

    def finalize(self) -> Optional[DecodeResult]:
        """Flush whatever audio has not been decoded yet."""
        return self.push(_EMPTY, force=True)


class WindowedDecoder(StreamingDecoder):
    """Constant-work incremental decoding over a sliding window."""

    name = "windowed"

    def __init__(
        self,
        transcribe: Callable[[np.ndarray], "tuple[str, float]"],
        sample_rate: int = 16000,
        left_context_s: float = 2.0,
        max_window_s: float = 10.0,
        min_new_audio_s: float = 0.25,
    ) -> None:
        self._transcribe = transcribe
        self._sample_rate = max(1, int(sample_rate))
        self._left_samples = int(round(left_context_s * self._sample_rate))
        self._max_window_samples = max(self._left_samples + 1, int(round(max_window_s * self._sample_rate)))
        self._min_new_samples = max(1, int(round(min_new_audio_s * self._sample_rate)))
        self._buffer = _EMPTY
        self._decoded = 0
        self._pending_reset = False

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._buffer = _EMPTY
        self._decoded = 0
        self._pending_reset = False

    @property
    def buffered_seconds(self) -> float:
        return self._buffer.size / float(self._sample_rate)

    # ------------------------------------------------------------------ #
    def push(self, audio: np.ndarray, force: bool = False) -> Optional[DecodeResult]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size:
            self._buffer = np.concatenate([self._buffer, samples]) if self._buffer.size else samples.copy()

        new_samples = self._buffer.size - self._decoded
        if new_samples <= 0 and not force:
            return None
        if not force and new_samples < self._min_new_samples:
            return None
        if self._buffer.size == 0:
            return None

        reset = self._pending_reset
        self._pending_reset = False

        start = max(0, self._decoded - self._left_samples)
        window = self._buffer[start:]
        text, confidence = self._transcribe(window)
        self._decoded = self._buffer.size

        # Keep the window bounded: drop the audio that is far enough behind the
        # decode position.  The next decode then starts a new accumulation.
        if self._buffer.size > self._max_window_samples:
            keep_from = max(0, self._decoded - self._left_samples)
            if keep_from > 0:
                self._buffer = np.ascontiguousarray(self._buffer[keep_from:])
                self._decoded -= keep_from
                self._pending_reset = True
                logger.debug(
                    "window trimmed to %.2fs; next partial starts a new accumulation",
                    self._buffer.size / float(self._sample_rate),
                )

        return DecodeResult(
            text=text,
            confidence=confidence,
            reset=reset,
            decoded_seconds=window.size / float(self._sample_rate),
        )


class CacheAwareDecoder(StreamingDecoder):
    """Thin wrapper that batches mic blocks into NeMo streaming chunks."""

    name = "cache-aware"

    def __init__(
        self,
        stream: CacheAwareStream,
        sample_rate: int = 16000,
        min_new_audio_s: float = 0.25,
    ) -> None:
        self._stream = stream
        self._sample_rate = max(1, int(sample_rate))
        self._min_new_samples = max(1, int(round(min_new_audio_s * self._sample_rate)))
        self._pending = _EMPTY

    def reset(self) -> None:
        self._pending = _EMPTY
        self._stream.reset()

    def push(self, audio: np.ndarray, force: bool = False) -> Optional[DecodeResult]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size:
            self._pending = np.concatenate([self._pending, samples]) if self._pending.size else samples.copy()
        if self._pending.size == 0:
            return None
        if not force and self._pending.size < self._min_new_samples:
            return None

        chunk = self._pending
        self._pending = _EMPTY
        text, confidence = self._stream.push(chunk)
        return DecodeResult(
            text=text,
            confidence=confidence,
            reset=False,
            decoded_seconds=chunk.size / float(self._sample_rate),
        )


def make_decoder(
    backend,
    asr_config: Optional[ASRConfig] = None,
    sample_rate: int = 16000,
) -> StreamingDecoder:
    """Pick the best decoder the backend supports (cache-aware, else windowed)."""
    config = asr_config or ASRConfig()
    factory = getattr(backend, "create_stream", None)
    if callable(factory):
        try:
            stream = factory()
        except Exception:
            logger.exception("cache-aware stream creation failed; using windowed decoding")
            stream = None
        if stream is not None:
            logger.info("streaming decoder: cache-aware")
            return CacheAwareDecoder(
                stream,
                sample_rate=sample_rate,
                min_new_audio_s=config.partial_interval_s,
            )

    logger.info(
        "streaming decoder: windowed (left context %.1fs, max window %.1fs)",
        config.left_context_s,
        config.max_window_s,
    )
    return WindowedDecoder(
        backend.transcribe,
        sample_rate=sample_rate,
        left_context_s=config.left_context_s,
        max_window_s=config.max_window_s,
        min_new_audio_s=config.partial_interval_s,
    )
