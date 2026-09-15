"""Dependency-free token Aho--Corasick matcher.

The abstraction deliberately does not expose implementation details, allowing a
future optional pyahocorasick backend.  It emits *all* matches (including nested
and overlapping matches); selection belongs to :mod:`span_resolver`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from .text_normalize import match_key, normalize

_TOKEN_RE = re.compile(r"[\w\u200c]+", re.UNICODE)
_HARD_BOUNDARY_RE = re.compile(r"[.!?;:،؛؟\n]")


@dataclass(frozen=True)
class Pattern:
    phrase: str
    output: str
    payload: Any = None
    priority: int = 0
    source: str = "exact"


@dataclass(frozen=True)
class Match:
    pattern: Pattern
    start: int
    end: int
    token_start: int
    token_end: int
    text: str


class _Node:
    __slots__ = ("next", "fail", "outputs")
    def __init__(self) -> None:
        self.next: dict[str, int] = {}
        self.fail = 0
        self.outputs: list[int] = []


class AhoCorasickMatcher:
    """Multi-pattern matcher with compatibility helpers from the old TrieFST."""
    def __init__(self) -> None:
        self._nodes = [_Node()]
        self._patterns: list[Pattern] = []
        self._compiled = True
        self._keys: dict[tuple[str, ...], str] = {}
        self.max_phrase_len = 0

    def add(self, phrase: str, output: str, *, payload: Any = None,
            priority: int = 0, source: str = "exact") -> None:
        normalized = normalize(phrase, join_persian_affixes=False, punctuation=False)
        keys = tuple(match_key(x) for x in normalized.replace("\u200c", " ").split())
        if not keys or any(not x for x in keys):
            raise ValueError("phrase must contain at least one word")
        previous = self._keys.get(keys)
        if previous is not None and previous != output:
            raise ValueError(f"conflicting rules for {phrase!r}: {previous!r} vs {output!r}")
        if previous == output:
            return
        self._keys[keys] = output
        node = 0
        for key in keys:
            child = self._nodes[node].next.get(key)
            if child is None:
                child = self._new_node()
                self._nodes[node].next[key] = child
            node = child
        index = len(self._patterns)
        self._patterns.append(Pattern(phrase, output, payload, priority, source))
        self._nodes[node].outputs.append(index)
        self.max_phrase_len = max(self.max_phrase_len, len(keys))
        self._compiled = False

    def _new_node(self) -> int:
        self._nodes.append(_Node())
        return len(self._nodes) - 1

    def add_many(self, mapping: Mapping[str, str]) -> None:
        for phrase, output in mapping.items():
            self.add(phrase, output)

    def _compile(self) -> None:
        if self._compiled:
            return
        queue = deque()
        for child in self._nodes[0].next.values():
            self._nodes[child].fail = 0
            queue.append(child)
        while queue:
            parent = queue.popleft()
            for key, child in self._nodes[parent].next.items():
                queue.append(child)
                fail = self._nodes[parent].fail
                while fail and key not in self._nodes[fail].next:
                    fail = self._nodes[fail].fail
                self._nodes[child].fail = self._nodes[fail].next.get(key, 0)
                self._nodes[child].outputs.extend(self._nodes[self._nodes[child].fail].outputs)
        self._compiled = True

    @property
    def rules(self) -> Sequence[tuple[str, str]]:
        return tuple((p.phrase, p.output) for p in self._patterns)

    def __len__(self) -> int:
        return len(self._patterns)

    def find(self, text: str) -> list[Match]:
        """Return all candidates with exact character and token offsets."""
        self._compile()
        tokens = list(_TOKEN_RE.finditer(text))
        state = 0
        found: list[Match] = []
        for i, token in enumerate(tokens):
            # Punctuation/newline resets state, prohibiting cross-sentence matches.
            if i and _HARD_BOUNDARY_RE.search(text[tokens[i - 1].end():token.start()]):
                state = 0
            key = match_key(token.group())
            while state and key not in self._nodes[state].next:
                state = self._nodes[state].fail
            state = self._nodes[state].next.get(key, 0)
            for pattern_index in self._nodes[state].outputs:
                pattern = self._patterns[pattern_index]
                length = len(normalize(pattern.phrase, join_persian_affixes=False,
                                       punctuation=False).replace("\u200c", " ").split())
                first = i - length + 1
                if first < 0:
                    continue
                start, end = tokens[first].start(), token.end()
                found.append(Match(pattern, start, end, first, i + 1, text[start:end]))
        return found

    # Compatibility: token input receives covering spans like TrieFST did.
    def rewrite_spans(self, tokens: Iterable[str]) -> list[tuple[Optional[str], int, int]]:
        items = list(tokens)
        text = " ".join(items)
        offsets = []
        cursor = 0
        for item in items:
            offsets.append((cursor, cursor + len(item)))
            cursor += len(item) + 1
        from .span_resolver import SpanResolver
        selected, _ = SpanResolver().resolve(self.find(text))
        out, cursor_token = [], 0
        for match in selected:
            start_token = next(i for i, pair in enumerate(offsets) if pair[1] > match.start)
            end_token = next(i + 1 for i, pair in enumerate(offsets) if pair[1] >= match.end)
            if cursor_token < start_token:
                out.append((None, cursor_token, start_token))
            leading = re.match(r"^[('\"]*", items[start_token]).group()
            trailing = re.search(r"[.,!?;:،؛؟)'\"]*$", items[end_token - 1]).group()
            out.append((leading + match.pattern.output + trailing, start_token, end_token))
            cursor_token = end_token
        if cursor_token < len(items):
            out.append((None, cursor_token, len(items)))
        return out

    def rewrite(self, tokens: Iterable[str]) -> list[str]:
        items = list(tokens); out = []
        for replacement, start, end in self.rewrite_spans(items):
            out.extend(items[start:end] if replacement is None else replacement.split())
        return out

    def apply(self, text: str) -> str:
        return " ".join(self.rewrite(text.split())) if text else text
