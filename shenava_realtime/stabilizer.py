"""Transcript stabilization: commit only what stopped changing.

Streaming ASR rewrites the tail of its hypothesis on every decode, so emitting
each partial verbatim duplicates and un-types text.  The stabilizer keeps two
pieces of state::

    committed_text   words that will never change again (safe to inject)
    current_partial  the moving tail, shown live but never injected

A word is committed only when (a) it survived unchanged since the previous
hypothesis and (b) it is far enough from the end of the hypothesis
(``holdback_words``).  Commits are strictly prefixes of the hypothesis, so the
same word can never be committed twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class StabilizerConfig:
    holdback_words: int = 2
    max_partial_words: int = 240


def _words(text: str) -> List[str]:
    return [word for word in (text or "").split(" ") if word]


def common_prefix_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Number of leading tokens shared by two word sequences."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


class TranscriptStabilizer:
    """Turns a stream of growing hypotheses into one-time word commits."""

    def __init__(self, config: Optional[StabilizerConfig] = None) -> None:
        self.config = config or StabilizerConfig()
        self._committed: List[str] = []
        self._partial: List[str] = []

    # ------------------------------------------------------------------ #
    @property
    def committed_text(self) -> str:
        return " ".join(self._committed)

    @property
    def current_partial(self) -> str:
        return " ".join(self._partial)

    @property
    def full_text(self) -> str:
        return " ".join([*self._committed, *self._partial])

    @property
    def committed_words(self) -> int:
        return len(self._committed)

    # ------------------------------------------------------------------ #
    def update(self, hypothesis: str) -> str:
        """Fold in a new hypothesis and return the newly committed words.

        Returns an empty string when nothing became stable yet.
        """
        words = _words(hypothesis)
        if self.config.max_partial_words and len(words) > self.config.max_partial_words:
            words = words[-self.config.max_partial_words :]

        previous = [*self._committed, *self._partial]
        stable = common_prefix_length(previous, words)

        holdback = max(0, self.config.holdback_words)
        commit_upto = min(stable, len(words) - holdback) if holdback else stable
        commit_upto = max(commit_upto, len(self._committed))
        commit_upto = min(commit_upto, len(words))

        newly = words[len(self._committed) : commit_upto]
        if newly:
            self._committed.extend(newly)
        self._partial = words[commit_upto:]
        return " ".join(newly)

    def finalize(self, hypothesis: Optional[str] = None) -> str:
        """End of utterance: commit everything and return the final delta."""
        if hypothesis is None:
            newly = list(self._partial)
            self._committed.extend(newly)
            self._partial = []
            return " ".join(newly)

        words = _words(hypothesis)
        shared = common_prefix_length(self._committed, words)
        if shared < len(self._committed):
            # The last decode rewrote text that was already committed (and
            # injected).  Committed text cannot be retracted, so only genuinely
            # new trailing words are appended.
            logger.debug(
                "final hypothesis diverged from committed text at word %d/%d",
                shared,
                len(self._committed),
            )
        newly = words[len(self._committed) :]
        self._committed.extend(newly)
        self._partial = []
        return " ".join(newly)

    def reset(self) -> None:
        self._committed = []
        self._partial = []

    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        return {
            "committed_words": len(self._committed),
            "partial_words": len(self._partial),
            "committed_text": self.committed_text,
            "current_partial": self.current_partial,
        }
