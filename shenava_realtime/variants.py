"""Evidence-driven collection of observed ASR variants for a reference term.

Workflow (human-in-the-loop, no automatic promotion)::

    reference term -> observed ASR variants -> reviewed variant -> terminology rule

``collect_variants`` projects a term's span in the *reference* text onto the
*raw* ASR text of the same utterance using a deterministic token alignment,
and reports the raw substring that corresponds to the term.  The output is a
review summary: nothing is written into ``terminology.json`` automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ObservedVariant:
    form: str
    count: int
    source_rows: tuple[int, ...] = field(default_factory=tuple)


def _token_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        start = index
        while index < len(text) and not text[index].isspace():
            index += 1
        if start < index:
            spans.append((start, index))
    return spans


def align_tokens(raw: str, reference: str) -> list[tuple[int | None, int | None]]:
    """Token-level edit alignment.

    Returns a list of ``(raw_token_index | None, reference_token_index |
    None)`` pairs: an entry with a None on one side is an insertion/deletion.
    Deterministic; ties break in favour of substitutions, then deletions.
    """
    raw_tokens = _token_spans(raw)
    ref_tokens = _token_spans(reference)
    n, m = len(raw_tokens), len(ref_tokens)
    cost = [[1] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            cost[i + 1][j + 1] = 0 if _same_token(raw, reference, raw_tokens[i], ref_tokens[j]) else 1
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost[i][j], dp[i - 1][j] + 1, dp[i][j - 1] + 1
            )
    alignment: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if dp[i][j] == dp[i - 1][j - 1] + cost[i][j]:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((i - 1, None))
            i -= 1
        else:
            alignment.append((None, j - 1))
            j -= 1
    while i > 0:
        alignment.append((i - 1, None))
        i -= 1
    while j > 0:
        alignment.append((None, j - 1))
        j -= 1
    alignment.reverse()
    return alignment


def _same_token(raw: str, reference: str, a: tuple[int, int], b: tuple[int, int]) -> bool:
    from .text_normalize import match_key

    return match_key(raw[a[0]:a[1]]) == match_key(reference[b[0]:b[1]])


def project_span(raw: str, reference: str, start: int, end: int) -> str:
    """Map a character span of ``reference`` onto the aligned raw text.

    A single-token term (e.g. ``اکوکاردیوگرافی``) that the ASR emitted as
    several tokens (``اکو کاردیوگرافی``, ``سی ای بی جی``) cannot be
    expressed by token edit alignment alone, so when the term is larger than
    its one mapped raw token, adjacent raw tokens that were *deleted* by the
    alignment are absorbed.  When the mapped token already is the whole term
    (verbatim) nothing is absorbed, so a stray adjacent word does not turn a
    correct recognition into a fake variant.  This is a review heuristic: the
    result is always shown to a human, never applied.
    """
    from .text_normalize import match_key

    ref_tokens = _token_spans(reference)
    raw_tokens = _token_spans(raw)
    if not ref_tokens or not raw_tokens or end <= start:
        return ""
    # Tokens fully inside the span first; fall back to overlapping tokens so
    # a multi-word term that shares a token boundary still projects.
    ref_indices = [i for i, t in enumerate(ref_tokens) if t[0] >= start and t[1] <= end]
    if not ref_indices:
        ref_indices = [i for i, t in enumerate(ref_tokens) if t[0] < end and t[1] > start]
    if not ref_indices:
        return ""
    wanted = set(ref_indices)
    alignment = align_tokens(raw, reference)
    mapped = [
        raw_idx
        for raw_idx, ref_idx in alignment
        if ref_idx is not None and ref_idx in wanted
    ]
    if not mapped:
        return ""
    lo, hi = min(mapped), max(mapped)
    if len(wanted) == 1:
        term_key = match_key(reference[start:end])
        mapped_key = match_key(raw[raw_tokens[lo][0]:raw_tokens[lo][1]])
        if mapped_key and mapped_key != term_key:
            deleted = {r for r, _ in alignment if _ is None}
            while lo - 1 >= 0 and (lo - 1) in deleted:
                lo -= 1
            while hi + 1 < len(raw_tokens) and (hi + 1) in deleted:
                hi += 1
    return raw[raw_tokens[lo][0]:raw_tokens[hi][1]]


def find_term_spans(text: str, term: str) -> list[tuple[int, int]]:
    spans = []
    index = 0
    needle = term.strip()
    if not needle:
        return spans
    while True:
        index = text.find(needle, index)
        if index < 0:
            return spans
        spans.append((index, index + len(needle)))
        index += len(needle)


def collect_variants(
    rows: Iterable[dict],
    term: str,
    *,
    raw_field: str = "raw",
    reference_field: str = "reference",
) -> dict:
    """Aggregate observed raw variants of ``term`` across corpus rows.

    Each row needs ``raw`` (ASR output) and ``reference`` (expected output);
    rows where ``term`` is not present in the reference are ignored.
    """
    counts: dict[str, dict] = {}
    row_number = 0
    for row in rows:
        row_number += 1
        reference = row.get(reference_field, "") or ""
        raw = row.get(raw_field, "") or ""
        if not reference or not raw:
            continue
        for start, end in find_term_spans(reference, term):
            variant = project_span(raw, reference, start, end).strip()
            if not variant:
                continue
            if variant == term.strip():
                continue  # not a variant: the term was recognised verbatim
            entry = counts.setdefault(variant, {"count": 0, "rows": []})
            entry["count"] += 1
            entry["rows"].append(row_number)
    variants = [
        ObservedVariant(form, info["count"], tuple(info["rows"]))
        for form, info in sorted(counts.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    ]
    return {"term": term.strip(), "rows_scanned": row_number, "variants": variants}


def suggest_rule_fragment(term: str, summary: dict, *, rule_id: str, category: str = "medical_term") -> dict:
    """Render a *review* fragment for terminology.json — never written automatically."""
    return {
        "id": rule_id,
        "canonical": term,
        "review_candidates": [v.form for v in summary["variants"]],
        "category": category,
        "note": "add only after human review; do not promote automatically",
    }
