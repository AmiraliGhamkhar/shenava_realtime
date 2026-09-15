"""Deterministic conflict resolution for candidate spans."""
from __future__ import annotations
from typing import Iterable
from .aho_corasick import Match

_SOURCE_SCORE = {"exact": 5, "normalized": 4, "alias": 3, "phonetic_variant": 2,
                 "contextual": 1}
_RISK_SCORE = {"low": 3, "medium": 2, "high": 1}


class SpanResolver:
    """Resolve by leftmost, longest, priority, risk, then context policy.

    Unsafe candidates are retained in the review list, never silently selected.
    """
    def resolve(self, candidates: Iterable[Match], *, protected=()) -> tuple[list[Match], list[Match]]:
        allowed, review = [], []
        protected = tuple(protected)
        for candidate in candidates:
            rule = candidate.pattern.payload
            overlaps = any(candidate.start < p.end and p.start < candidate.end for p in protected)
            risk = getattr(rule, "risk", "low")
            requires_context = bool(getattr(rule, "requires_context", False))
            enabled = bool(getattr(rule, "enabled", True))
            # A future contextual resolver may mark evidence explicitly. It is disabled now.
            context_ok = bool(getattr(rule, "context_satisfied", False))
            if overlaps or not enabled or risk == "high" or (requires_context and not context_ok):
                review.append(candidate)
            else:
                allowed.append(candidate)
        allowed.sort(key=lambda m: (m.start, -(m.end - m.start), -m.pattern.priority,
                                    -_RISK_SCORE.get(getattr(m.pattern.payload, "risk", "low"), 0),
                                    -_SOURCE_SCORE.get(m.pattern.source, 0), m.pattern.phrase))
        selected, cursor = [], -1
        for match in allowed:
            if match.start < cursor:
                continue
            selected.append(match)
            cursor = match.end
        return selected, review
