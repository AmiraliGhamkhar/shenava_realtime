"""Native cache-aware decoding with bounded endpoint-only fallback."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional
from .asr_backend import ASRBackend

import numpy as np

from .asr_backend import CacheAwareStream
from .config import ASRConfig
from .second_pass import BeamSecondPass, GreedySecondPass, SecondPassDecoder

logger = logging.getLogger(__name__)

_EMPTY = np.zeros(0, dtype=np.float32)

# CTC models decode in fixed frames (10 ms for the Shenava frontend). Non-forced
# flushes hand the recognizer only whole frames and keep the sub-frame
# remainder pending, so decode boundaries never fall mid-frame (issue #15).
FRAME_ALIGN_S = 0.01


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

    def utterance_audio(self) -> Optional[np.ndarray]:
        """Raw audio of the current segment, for an optional second pass.

        ``None`` when the decoder does not retain it (e.g. endpoint mode,
        where the single offline decode already *is* the full-context pass).
        """
        return None


class EndpointDecoder(StreamingDecoder):
    """Explicit fallback for offline/smaller checkpoints: one decode per segment."""
    name = "endpoint"

    def __init__(self, transcribe: Callable[[np.ndarray], tuple[str, float]],
                 sample_rate: int = 16000, max_segment_s: float = 22.0) -> None:
        self.transcribe = transcribe
        self.limit = int(sample_rate * max_segment_s)
        self.reset()

    def reset(self) -> None:
        self.blocks = []
        self.samples = 0

    def push(self, audio: np.ndarray, force: bool = False) -> Optional[DecodeResult]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError("Non-finite audio")
        if self.samples + samples.size > self.limit:
            # Visible abort, not silent loss: the caller must split the audio
            # (VAD cap or start a new segment) instead of dropping samples.
            raise ValueError(
                f"Endpoint segment limit exceeded: {self.samples} buffered + "
                f"{samples.size} incoming > {self.limit} samples; split the audio "
                "(lower vad_max_speech_s or max_segment_s) before decoding"
            )
        if samples.size:
            self.blocks.append(samples.copy())
            self.samples += samples.size
        if not force or not self.samples:
            return None
        audio = np.concatenate(self.blocks)
        self.reset()
        text, score = self.transcribe(audio)
        return DecodeResult(text, score)


class CacheAwareDecoder(StreamingDecoder):
    """Thin wrapper that batches mic blocks into the backend's streaming chunks."""

    name = "cache-aware"

    def __init__(
        self,
        stream: CacheAwareStream,
        sample_rate: int = 16000,
        min_new_audio_s: float = 0.25,
        max_segment_s: float = 22.0,
        first_decode_s: float = 0.0,
    ) -> None:
        self._stream = stream
        self._sample_rate = max(1, int(sample_rate))
        self._min_new_samples = max(1, int(round(min_new_audio_s * self._sample_rate)))
        # Adaptive batching: the first decode of an utterance happens as soon
        # as ``first_decode_s`` of audio is pending (silence-to-speech must
        # feel immediate); every later decode waits for ``min_new_audio_s``.
        # 0 disables the early decode (pinned batching behaviour in tests).
        self._first_decode_samples = max(0, int(round(first_decode_s * self._sample_rate)))
        self._decoded_any = False
        # Sub-frame remainder kept pending so decode boundaries never split a
        # recognizer frame (issue #15).
        self._frame_samples = max(1, int(round(FRAME_ALIGN_S * self._sample_rate)))
        # Bounded per-utterance audio buffer for the optional second pass
        # (at most the segment cap: ~1.4 MB at 22 s).
        self._segment_limit = max(1, int(max_segment_s * self._sample_rate))
        self._pending = _EMPTY
        self._utterance = _EMPTY
        self._utterance_samples = 0

    def reset(self) -> None:
        self._pending = _EMPTY
        self._utterance = _EMPTY
        self._utterance_samples = 0
        self._decoded_any = False
        self._stream.reset()

    def push(self, audio: np.ndarray, force: bool = False) -> Optional[DecodeResult]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all() or samples.size > self._sample_rate * 120:
            raise ValueError("Invalid streaming audio block")
        if samples.size:
            self._pending = np.concatenate([self._pending, samples]) if self._pending.size else samples.copy()
            if self._utterance_samples + samples.size <= self._segment_limit:
                self._utterance = (
                    np.concatenate([self._utterance, samples])
                    if self._utterance.size
                    else samples.copy()
                )
                self._utterance_samples += samples.size
        if self._pending.size == 0:
            return None
        if not force:
            # Adaptive batching (#13): a fresh utterance decodes as soon as
            # first_decode_s is pending; afterwards min_new_audio_s applies.
            threshold = self._min_new_samples
            if not self._decoded_any and self._first_decode_samples:
                threshold = min(threshold, self._first_decode_samples)
            if self._pending.size < threshold:
                return None
            # Frame alignment (#15): feed whole recognizer frames only and
            # keep the sub-frame remainder for the next push.
            fed = self._pending.size - (self._pending.size % self._frame_samples)
            if fed == 0:
                return None
            chunk = self._pending[:fed]
            self._pending = self._pending[fed:].copy()
        else:
            chunk = self._pending
            self._pending = _EMPTY
        self._decoded_any = True
        text, confidence = self._stream.push(chunk)
        return DecodeResult(
            text=text,
            confidence=confidence,
            reset=False,
            decoded_seconds=chunk.size / float(self._sample_rate),
        )

    def utterance_audio(self) -> Optional[np.ndarray]:
        return None if self._utterance_samples == 0 else self._utterance

    def finalize(self) -> Optional[DecodeResult]:
        result = self.push(_EMPTY, force=True)
        finalize = getattr(self._stream, "finalize", None)
        if callable(finalize):
            text, score = finalize()
            return DecodeResult(text, score)
        return result


def make_decoder(
    backend: ASRBackend,
    asr_config: Optional[ASRConfig] = None,
    sample_rate: int = 16000,
) -> StreamingDecoder:
    """Select native streaming or an explicit bounded endpoint-only fallback."""
    config = asr_config or ASRConfig()
    config.__post_init__()
    factory = getattr(backend, "create_stream", None)
    if config.use_cache_aware_streaming and callable(factory):
        # Initialization failures are not evidence of missing capabilities.
        # Fail visibly rather than hiding a broken native adapter.
        stream = factory()
        if stream is not None:
            logger.info("streaming decoder: cache-aware")
            return CacheAwareDecoder(
                stream, sample_rate, config.partial_interval_s, config.max_segment_s,
                first_decode_s=config.first_decode_s,
            )
    if config.require_streaming:
        raise RuntimeError("Checkpoint does not support configured cache-aware streaming")
    logger.warning("Native streaming unavailable/disabled; using endpoint-only CTC fallback")
    return EndpointDecoder(backend.transcribe, sample_rate, config.max_segment_s)


def make_second_pass(
    backend: ASRBackend,
    asr_config: Optional[ASRConfig] = None,
) -> Optional[SecondPassDecoder]:
    """Build the optional utterance-end second pass from configuration.

    ``off`` -> None.  ``greedy`` -> one offline re-decode via the backend's
    own ``transcribe()`` (same model, same recognizer, no bias).
    ``second_pass="context"`` (CTC beam + hotword biasing over raw emissions)
    is rejected at ``ASRConfig.__post_init__`` — sherpa-onnx's public Python
    API exposes only decoded text, not per-frame emissions or the model's
    tokenizer, so that mode cannot be implemented for this backend.
    """
    config = asr_config or ASRConfig()
    config.__post_init__()
    if config.second_pass == "off":
        return None
    transcribe = getattr(backend, "transcribe", None)
    if not callable(transcribe):
        logger.warning("second pass disabled: backend has no transcribe()")
        return None
    # Opt-in beam second pass (#10): sherpa supports modified_beam_search for
    # models exported with an LM/tokenizer object; when the backend can build
    # such a recognizer it is preferred, otherwise the greedy offline
    # re-decode stays active (explicit, logged fallback — never silent).
    if config.second_pass_beam:
        factory = getattr(backend, "create_beam_recognizer", None)
        if callable(factory):
            try:
                beam_transcribe = factory()
            except Exception as exc:
                logger.warning("beam second pass unavailable (%s); using greedy", exc)
            else:
                if beam_transcribe is not None:
                    return BeamSecondPass(beam_transcribe)
        else:
            logger.info(
                "second_pass_beam is enabled but the backend exposes no "
                "create_beam_recognizer(); using the greedy second pass"
            )
    return GreedySecondPass(transcribe)
