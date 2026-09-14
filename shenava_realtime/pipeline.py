"""Utterance pipeline: decoder -> stabilizer -> FST post-processing -> deltas.

This is the piece that guarantees the two properties the output stage relies on:

* **no duplicates** — text is emitted exactly once, as the difference between
  what was already emitted and the newly post-processed committed text;
* **stable only** — nothing is emitted until the stabilizer has committed it,
  i.e. until the model stopped rewriting it.

It has no threads and no audio hardware in it, so it is unit-testable with a
stub decoder.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from .postprocessor import PostProcessor
from .stabilizer import StabilizerConfig, TranscriptStabilizer
from .streaming import StreamingDecoder

logger = logging.getLogger(__name__)


def text_delta(previous: str, current: str) -> str:
    """Return the suffix of ``current`` that ``previous`` does not already have.

    The comparison backs off to a word boundary so a partially recognised word
    is never emitted twice.  When the model rewrites already emitted text the
    divergence is logged: injected text cannot be un-typed, so the best we can
    do is emit the new tail.
    """
    if current == previous:
        return ""
    if current.startswith(previous):
        return current[len(previous) :]

    index = 0
    limit = min(len(previous), len(current))
    while index < limit and previous[index] == current[index]:
        index += 1
    if index and current[index - 1] != " ":
        boundary = current.rfind(" ", 0, index)
        index = boundary + 1 if boundary >= 0 else 0
    if index < len(previous):
        logger.debug("transcript diverged after %d chars; emitting the new tail only", index)
    return current[index:]


class TranscriptionPipeline:
    """One utterance's worth of streaming state."""

    def __init__(
        self,
        decoder: StreamingDecoder,
        postprocessor: Optional[PostProcessor] = None,
        stabilizer: Optional[TranscriptStabilizer] = None,
        holdback_words: int = 2,
        commit_on_endpoint: bool = True,
    ) -> None:
        self.commit_on_endpoint = commit_on_endpoint
        self._latest = ""
        self.decoder = decoder
        self.postprocessor = postprocessor or PostProcessor()
        self.stabilizer = stabilizer or TranscriptStabilizer(StabilizerConfig(holdback_words=holdback_words))
        self._emitted = ""
        self._prefix = ""
        self._confidence = 0.0
        self._decodes = 0
        self._decode_seconds = 0.0
        self._active = False

    # ------------------------------------------------------------------ #
    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def committed_text(self) -> str:
        """Post-processed text that is safe to output (already emitted)."""
        return self._emitted

    @property
    def partial_text(self) -> str:
        """Live view: committed text plus the still-moving tail."""
        full = self.postprocessor.process(self._latest if self.commit_on_endpoint else self.stabilizer.full_text)
        if not full:
            return self._emitted
        return f"{self._prefix}{full}" if self._prefix else full

    @property
    def decodes(self) -> int:
        return self._decodes

    @property
    def decode_seconds(self) -> float:
        return self._decode_seconds

    # ------------------------------------------------------------------ #
    def start_utterance(self, preroll: Optional[np.ndarray] = None) -> List[str]:
        """Begin a new utterance, discarding any leftover state."""
        self.decoder.reset()
        self.stabilizer.reset()
        self._emitted = ""
        self._prefix = ""
        self._confidence = 0.0
        self._latest = ""
        self._active = True
        if preroll is not None and np.asarray(preroll).size:
            return self.push_audio(np.asarray(preroll, dtype=np.float32))
        return []

    def push_audio(self, chunk: np.ndarray) -> List[str]:
        """Feed a block of audio; returns any newly stable text deltas."""
        if not self._active:
            return []
        started = time.perf_counter()
        result = self.decoder.push(chunk)
        self._decode_seconds += time.perf_counter() - started
        if result is None:
            return []
        self._decodes += 1
        self._confidence = result.confidence or self._confidence
        return self._fold(result.text, reset=result.reset)

    def end_utterance(self) -> List[str]:
        """Flush the decoder and commit the tail of the utterance."""
        if not self._active:
            return []
        self._active = False
        deltas: List[str] = []
        started = time.perf_counter()
        try:
            result = self.decoder.finalize()
        except Exception:
            self.abort()
            raise
        finally:
            self._decode_seconds += time.perf_counter() - started
        if result is not None:
            self._decodes += 1
            self._confidence = result.confidence or self._confidence
            if result.reset:
                deltas.extend(self._commit_everything())
            self._latest = result.text
            if self.commit_on_endpoint:
                self.stabilizer.reset()
            self.stabilizer.update(result.text)
        deltas.extend(self._commit_everything())
        return [delta for delta in deltas if delta]

    def abort(self) -> None:
        """Drop the utterance without emitting anything (hotkey: clear)."""
        self.decoder.reset()
        self.stabilizer.reset()
        self._emitted = ""
        self._prefix = ""
        self._active = False

    # ------------------------------------------------------------------ #
    def _fold(self, hypothesis: str, reset: bool = False) -> List[str]:
        deltas: List[str] = []
        if reset:
            # The decoder cut the window (long monologue): commit what we have,
            # then start a fresh accumulation whose text is appended below.
            deltas.extend(self._commit_everything())
            self._prefix = _separated(self._emitted)
            self.stabilizer.reset()
        self._latest = hypothesis
        if hypothesis:
            self.stabilizer.update(hypothesis)
            delta = "" if self.commit_on_endpoint else self._emit()
            if delta:
                deltas.append(delta)
        return [delta for delta in deltas if delta]

    def _commit_everything(self) -> List[str]:
        if self.commit_on_endpoint:
            self.stabilizer.reset()
            self.stabilizer.update(self._latest)
        self.stabilizer.finalize()
        delta = self._emit()
        return [delta] if delta else []

    def _emit(self) -> str:
        """Post-process the committed text and return what has not been sent."""
        processed = f"{self._prefix}{self.postprocessor.process(self.stabilizer.committed_text)}"
        if self._emitted and not processed.startswith(self._emitted):
            logger.warning("Suppressing a rewrite of already emitted text")
            return ""
        delta = processed[len(self._emitted):]
        self._emitted = processed
        return delta


def _separated(text: str) -> str:
    """Ensure a segment boundary is followed by a space."""
    if text and not text.endswith(" "):
        return text + " "
    return text
