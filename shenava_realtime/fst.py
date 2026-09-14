"""A minimal deterministic finite-state transducer for phrase rewriting.

This is the lightweight FST used for medical terms, abbreviations and units.
It is a trie automaton with an output string at each accepting node:

* input alphabet  = normalized word tokens (see :func:`text_normalize.match_key`)
* transitions     = deterministic (one child per token, no epsilon moves)
* matching        = longest match, scanned strictly left to right

No fuzzy matching, no edit distance, no embeddings: a phrase is rewritten only
when its token sequence matches a rule exactly.  Rules are plain data
(``lexicon.py``), so the same table could be compiled by Pynini/OpenFst later
without touching this logic.
"""

from __future__ import annotations

import re

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .text_normalize import match_key


class _Node:
    __slots__ = ("children", "output")

    def __init__(self) -> None:
        self.children: Dict[str, "_Node"] = {}
        self.output: Optional[str] = None


class TrieFST:
    """Deterministic longest-match phrase transducer over word sequences."""

    def __init__(self) -> None:
        self._root = _Node()
        self._rules: List[Tuple[str, str]] = []
        self.max_phrase_len = 0

    def add(self, phrase: str, output: str) -> None:
        """Add ``phrase`` (space separated) -> ``output`` (space separated)."""
        tokens = phrase.split()
        if not tokens:
            raise ValueError("phrase must contain at least one word")
        node = self._root
        for token in tokens:
            key = match_key(token)
            if not key:
                raise ValueError(f"phrase {phrase!r} contains an empty token")
            node = node.children.setdefault(key, _Node())
        if node.output is not None and node.output != output:
            raise ValueError(
                f"conflicting rules for {phrase!r}: {node.output!r} vs {output!r}"
            )
        node.output = output
        self._rules.append((phrase, output))
        self.max_phrase_len = max(self.max_phrase_len, len(tokens))

    def add_many(self, mapping: Mapping[str, str]) -> None:
        for phrase, output in mapping.items():
            self.add(phrase, output)

    @property
    def rules(self) -> Sequence[Tuple[str, str]]:
        return tuple(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def rewrite_spans(self, tokens: Iterable[str]) -> List[Tuple[Optional[str], int, int]]:
        """Transduce and report which input tokens each output came from.

        Returns a list of ``(replacement, start, end)`` spans covering the whole
        input.  ``replacement`` is ``None`` for a span that matched no rule and
        must be kept verbatim, otherwise it is the rule output.
        """
        tokens = list(tokens)
        spans: List[Tuple[Optional[str], int, int]] = []
        index = 0
        count = len(tokens)
        unmatched_start = 0
        while index < count:
            node = self._root
            best_output: Optional[str] = None
            best_end = index
            cursor = index
            while cursor < count:
                if cursor > index and re.search(r"[.,!?;:،؛؟()]$", tokens[cursor - 1]):
                    break
                node = node.children.get(match_key(tokens[cursor]))
                if node is None:
                    break
                cursor += 1
                if node.output is not None:
                    best_output = node.output
                    best_end = cursor
            if best_output is None:
                index += 1
                continue
            if index > unmatched_start:
                spans.append((None, unmatched_start, index))
            leading = re.match(r"^[\(\"\']*", tokens[index]).group()
            trailing = re.search(r"[.,!?;:،؛؟)\"\']*$", tokens[best_end - 1]).group()
            spans.append((leading + best_output + trailing, index, best_end))
            index = best_end
            unmatched_start = index
        if unmatched_start < count:
            spans.append((None, unmatched_start, count))
        return spans

    def rewrite(self, tokens: Iterable[str]) -> List[str]:
        """Transduce a token sequence, replacing the longest matching phrases."""
        tokens = list(tokens)
        out: List[str] = []
        for replacement, start, end in self.rewrite_spans(tokens):
            if replacement is None:
                out.extend(tokens[start:end])
            elif replacement:
                out.extend(replacement.split())
        return out

    def apply(self, text: str) -> str:
        """Convenience wrapper: text in, text out (single-spaced)."""
        if not text:
            return text
        return " ".join(self.rewrite(text.split()))
