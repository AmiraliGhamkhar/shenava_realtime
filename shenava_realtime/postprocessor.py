"""Deterministic post-processing of raw ASR output.

Pipeline (all stages are pure string rules, in this order)::

    1. Persian Unicode normalization (letters, digits, marks, spacing)
    2. FST phrase rewriting: medical terms, abbreviations, units, percent
    3. consecutive-repetition removal (a common CTC artifact)
    4. spoken numbers -> digits, %, °, blood pressure "120/80"
    5. Persian clitic joining (ZWNJ) + punctuation/whitespace tidy-up

The order matters:

* the FST runs *before* repetition removal, otherwise a legitimate doubled
  abbreviation such as ``سی سی`` (cc) would be collapsed to a single ``سی``;
* the FST runs *before* number conversion, otherwise ``سی ای بی جی`` (CABG) and
  ``سی سی`` (cc) would be read as the number 30;
* clitic joining runs *last*, so the ZWNJ it inserts can never hide a phrase
  from the FST (and re-processing already processed text stays stable).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import fa_numbers
from .config import PostProcessConfig
from .fst import TrieFST
from .lexicon import build_rewriter
from .text_normalize import ZWNJ, fix_punctuation, join_affixes, match_key, normalize

logger = logging.getLogger(__name__)

_RE_MULTI_SPACE = re.compile(r"[ \t\u00a0]{2,}")


class PostProcessor:
    """Applies the deterministic text pipeline to a hypothesis."""

    def __init__(
        self,
        config: Optional[PostProcessConfig] = None,
        extra_terms: Optional[Dict[str, str]] = None,
    ) -> None:
        self.config = config or PostProcessConfig()
        self.fst: TrieFST = build_rewriter(
            medical_terms=self.config.medical_terms,
            units=self.config.units,
            extra_terms=extra_terms,
        )
        # Latin tokens the FST emits; used to tell "یک" (1) from "یک" (a/an).
        self._unit_tokens: Set[str] = {
            match_key(token)
            for _, output in self.fst.rules
            for token in output.split()
            if token
        }
        self._unit_tokens.discard("")
        logger.debug("PostProcessor ready with %d rewrite rules", len(self.fst))

    # ------------------------------------------------------------------ #
    def process(self, text: str) -> str:
        """Run the full pipeline on a (possibly partial) hypothesis."""
        if not text:
            return ""
        if not self.config.enabled:
            return text.strip()

        text = self.normalize_unicode(text)
        text = self.apply_rules(text)
        if self.config.remove_repetitions:
            text = self.remove_repetitions(text)
        if self.config.convert_numbers:
            text = fa_numbers.convert_numbers(
                text,
                digits=self.config.digits.value,
                unit_tokens=self._unit_tokens,
            )
        return self.finalize(text)

    # ------------------------------------------------------------------ #
    def normalize_unicode(self, text: str) -> str:
        if not self.config.normalize_unicode:
            return text.strip()
        return normalize(
            text,
            to_ascii_digits=(self.config.digits.value == "ascii"),
            # Clitics are joined at the end, after the dictionary pass.
            join_persian_affixes=False,
            punctuation=self.config.punctuation,
        )

    def apply_rules(self, text: str) -> str:
        """FST phrase pass (medical terms / abbreviations / units)."""
        if not text or not len(self.fst):
            return text

        # A ZWNJ inside a token can hide two dictionary words ("سی‌ای"), so the
        # token list is expanded for matching and rebuilt for the leftovers.
        expanded: List[str] = []
        starts_token: List[bool] = []
        for token in text.split(" "):
            if not token:
                continue
            parts = token.split(ZWNJ)
            for position, part in enumerate(parts):
                expanded.append(part)
                starts_token.append(position == 0)

        out: List[str] = []
        for replacement, start, end in self.fst.rewrite_spans(expanded):
            if replacement is not None:
                if replacement:
                    out.extend(replacement.split())
                continue
            out.extend(self._rebuild(expanded, starts_token, start, end))
        return " ".join(part for part in out if part)

    @staticmethod
    def _rebuild(
        expanded: Sequence[str], starts_token: Sequence[bool], start: int, end: int
    ) -> List[str]:
        """Undo the ZWNJ expansion for tokens the FST left untouched."""
        words: List[str] = []
        for index in range(start, end):
            if starts_token[index] or not words:
                words.append(expanded[index])
            else:
                words[-1] = f"{words[-1]}{ZWNJ}{expanded[index]}"
        return words

    @staticmethod
    def remove_repetitions(text: str) -> str:
        """Collapse immediately repeated words (``بیمار بیمار`` -> ``بیمار``)."""
        words = text.split(" ")
        out: List[str] = []
        for word in words:
            if not word:
                continue
            if out and match_key(out[-1]) == match_key(word):
                continue
            out.append(word)
        return " ".join(out)

    def finalize(self, text: str) -> str:
        if not text:
            return ""
        if self.config.join_persian_affixes:
            text = join_affixes(text)
        if self.config.punctuation:
            text = fix_punctuation(text)
        return _RE_MULTI_SPACE.sub(" ", text).strip()

    # ------------------------------------------------------------------ #
    def unit_tokens(self) -> Sequence[str]:
        """Tokens the FST can emit (``mg``, ``%``, ``bpm``, ``°C``, ...)."""
        return sorted(self._unit_tokens)

    def rules(self) -> Sequence[Tuple[str, str]]:
        """All rewrite rules, for inspection and tests."""
        return self.fst.rules
