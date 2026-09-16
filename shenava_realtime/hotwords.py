"""Terminology-driven decoder hotword lists (bounded, conservative).

The reviewed rules that already drive post-ASR normalization
(``data/terminology.json``) are the single source for decoder-time biasing —
there is no second medical dictionary.  Biases are small log-prob boosts
applied *only while the decode path continues a hotword prefix*; units and
anatomy are excluded because they are too common to bias safely.
"""
from __future__ import annotations

from typing import Iterable, Optional

from .second_pass import Hotword
from .terminology import TerminologyRule

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
MAX_BIAS = 1.5
# Categories never boosted (too generic in ordinary Persian dictation).
_EXCLUDED_CATEGORIES = {"unit", "anatomy"}


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
    selected_rules = [
        rule
        for rule in rules
        if rule.enabled
        and rule.category not in _EXCLUDED_CATEGORIES
        and rule.category in BIAS_BY_CATEGORY
        and (rule.specialty == "general" or (specialty and rule.specialty == specialty))
    ]
    selected_rules.sort(key=lambda rule: (-rule.priority, rule.id))

    hotwords: list[Hotword] = []
    seen: set[tuple[str, float]] = set()
    for rule in selected_rules:
        bias = min(MAX_BIAS, BIAS_BY_CATEGORY[rule.category])
        for form in rule.spoken_forms:
            key = (form, bias)
            if form and key not in seen:
                seen.add(key)
                hotwords.append(Hotword(form, bias))
                if len(hotwords) >= max_hotwords:
                    return hotwords
    return hotwords
