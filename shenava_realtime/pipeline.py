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

from . import fa_numbers
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
        # Forced VAD/ASR-cap boundary tracking (never set by a natural end).
        self._boundary_forced = False
        self._split_continues = False

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
        """Begin a new utterance, discarding any leftover state.

        ``_split_continues`` deliberately survives: a forced boundary already
        tagged the previous segment's tail as an unfinished number phrase, so
        this segment's leading number words must stay unparsed.
        """
        self.decoder.reset()
        self.stabilizer.reset()
        self._emitted = ""
        self._prefix = ""
        self._confidence = 0.0
        self._latest = ""
        self._boundary_forced = False
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

    def end_utterance(self, forced: bool = False) -> List[str]:
        """Flush the decoder and commit the tail of the utterance.

        ``forced`` marks a VAD/ASR segment-cap cut rather than a natural
        endpoint: the speaker was still talking, so a number phrase left open
        at this boundary must not be parsed as a complete value
        (see :func:`shenava_realtime.fa_numbers.open_number_tail`).
        """
        if not self._active:
            return []
        self._active = False
        self._boundary_forced = forced
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
                self._boundary_forced = True  # decoder cap: also a forced cut
                deltas.extend(self._commit_everything())
                self._boundary_forced = forced
            self._latest = result.text
            if self.commit_on_endpoint:
                self.stabilizer.reset()
            self.stabilizer.update(result.text)
        deltas.extend(self._commit_everything())
        self._boundary_forced = False
        return [delta for delta in deltas if delta]

    def abort(self) -> None:
        """Drop the utterance without emitting anything (hotkey: clear)."""
        self.decoder.reset()
        self.stabilizer.reset()
        self._emitted = ""
        self._prefix = ""
        self._boundary_forced = False
        self._split_continues = False
        self._active = False

    # ------------------------------------------------------------------ #
    def _fold(self, hypothesis: str, reset: bool = False) -> List[str]:
        deltas: List[str] = []
        if reset:
            # The decoder cut the window (long monologue): commit what we have,
            # then start a fresh accumulation whose text is appended below.
            # This is a forced cut, not a natural phrase boundary.
            self._boundary_forced = True
            deltas.extend(self._commit_everything())
            self._boundary_forced = False
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
        """Post-process the committed text and return what has not been sent.

        A forced segment boundary must never complete a number phrase: the
        open tail of the previous accumulation and the leading number words
        of the next one are kept as spoken words (and flagged in the log)
        instead of being parsed into independent — possibly wrong — values.
        """
        committed = self.stabilizer.committed_text
        leading = ""
        if self._split_continues:
            self._split_continues = False
            if committed:
                leading = fa_numbers.leading_number_span(committed)
        tail = (
            fa_numbers.open_number_tail(committed)
            if committed and self._boundary_forced
            else ""
        )
        if tail:
            self._split_continues = True
        if leading or tail:
            logger.warning(
                "number phrase %r spans a forced segment boundary; "
                "keeping the spoken words unparsed",
                " ".join(part for part in (leading, tail) if part),
            )
        head = committed
        if leading:
            head = head[len(leading):].lstrip()
        if tail:
            head = head[: len(head) - len(tail)].rstrip()
        processed_head = self.postprocessor.process(head) if head.strip() else ""
        body = " ".join(part for part in (leading, processed_head, tail) if part)
        processed = f"{self._prefix}{body}" if self._prefix else body
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
