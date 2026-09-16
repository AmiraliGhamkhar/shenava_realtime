"""Terminology-driven decoder hotword lists (bounded, conservative).

The reviewed rules that already drive post-ASR normalization
(``data/terminology.json``) are the single source for decoder-time biasing —
there is no second medical dictionary.  Biases are small log-prob boosts
applied *only while the decode path continues a hotword prefix*.  A rule's
category picks the default boost; the optional ``TerminologyRule.bias`` field
is the minimal per-rule override a reviewer uses to justify an exception
(force ``0`` to suppress, or a bounded value to opt an otherwise unboosted
category such as anatomy in).  Units are never decoder hotwords.
"""
from __future__ import annotations

from typing import Iterable, Optional

from .second_pass import Hotword
from .terminology import MAX_HOTWORD_BIAS, TerminologyRule

# Log-prob boosts by category.  Deliberately small: a boost of 1.0 makes a
# token ~2.7x more likely, enough to tip a near-tie without converting
# unrelated words into medical terms.
BIAS_BY_CATEGORY = {
    "medication": 1.0,
    "procedure": 0.75,
    "imaging": 0.75,
    "abbreviation": 0.5,
    "disease": 0.5,
    "symptom": 0.5,
    "medical_term": 0.25,
}
MAX_BIAS = MAX_HOTWORD_BIAS
# Units are rendered outputs (%, °), not clinical vocabulary to recover.
# Anatomy and other generic categories have no default boost: they appear in
# the hotword list only through an explicit, reviewed ``rule.bias``.
_EXCLUDED_CATEGORIES = {"unit"}


def _bias_for(rule: TerminologyRule) -> Optional[float]:
    """The reviewable log-prob boost for one rule, or ``None`` = skip."""
    if rule.category in _EXCLUDED_CATEGORIES:
        return None
    if rule.bias is not None:
        return float(rule.bias) if rule.bias > 0 else None
    bias = BIAS_BY_CATEGORY.get(rule.category)
    return min(MAX_BIAS, bias) if bias is not None else None


def build_hotwords(
    rules: Iterable[TerminologyRule],
    *,
    specialty: Optional[str] = None,
    max_hotwords: int = 64,
) -> list[Hotword]:
    """Build the active hotword list from terminology rules.

    ``specialty`` (e.g. ``"cardiology"``) restricts the list to that specialty
    plus the general set; ``None`` uses only the general set — the whole
    vocabulary is never enabled indiscriminately.  Selection is deterministic:
    highest priority first, then rule id.
    """
    if not 1 <= int(max_hotwords) <= 512:
        raise ValueError("max_hotwords must be within [1, 512]")
    scored = [
        (rule, _bias_for(rule))
        for rule in rules
        if rule.enabled
        and (rule.specialty == "general" or (specialty and rule.specialty == specialty))
    ]
    selected_rules = [(rule, bias) for rule, bias in scored if bias is not None]
    selected_rules.sort(key=lambda item: (-item[0].priority, item[0].id))

    hotwords: list[Hotword] = []
    seen: set[tuple[str, float]] = set()
    for rule, bias in selected_rules:
        for form in rule.spoken_forms:
            key = (form, bias)
            if form and key not in seen:
                seen.add(key)
                hotwords.append(Hotword(form, bias))
                if len(hotwords) >= max_hotwords:
                    return hotwords
    return hotwords
