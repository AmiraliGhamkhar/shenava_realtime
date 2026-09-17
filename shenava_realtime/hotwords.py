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
# Reviewed aliases/phonetic variants are weaker evidence than the reviewed
# spoken form and are biased proportionally lower when enabled.
VARIANT_BIAS_FACTOR = 0.5
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
    use_aliases: bool = False,
    use_phonetic_variants: bool = False,
) -> list[Hotword]:
    """Build the active hotword list from terminology rules.

    ``specialty`` (e.g. ``"cardiology"``) restricts the list to that specialty
    plus the general set; ``None`` uses only the general set — the whole
    vocabulary is never enabled indiscriminately.  Selection is deterministic:
    highest priority first, then rule id.

    Reviewed spoken forms are always eligible.  Reviewed ``aliases`` and
    ``phonetic_variants`` participate only when explicitly enabled, and are
    biased at half the rule's boost: they are the weaker evidence of the three
    and must not outrank the reviewed spoken form.  Nothing is generated: every
    phrase comes from ``terminology.json``.
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
    seen: set[str] = set()
    for rule, bias in selected_rules:
        for phrase, phrase_bias in _phrases(rule, bias, use_aliases, use_phonetic_variants):
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            hotwords.append(Hotword(phrase, min(MAX_BIAS, phrase_bias)))
            if len(hotwords) >= max_hotwords:
                return hotwords
    return hotwords


def _phrases(rule: TerminologyRule, bias: float, use_aliases: bool,
             use_phonetic_variants: bool) -> list[tuple[str, float]]:
    """Eligible (phrase, bias) pairs for one rule, reviewed sources only."""
    items = [(form, bias) for form in rule.spoken_forms]
    if use_aliases:
        items += [(alias, bias * VARIANT_BIAS_FACTOR) for alias in rule.aliases]
    if use_phonetic_variants:
        items += [(v, bias * VARIANT_BIAS_FACTOR) for v in rule.phonetic_variants]
    # Latin/punctuation-only aliases (e.g. "CABG", "%") are rendered outputs,
    # not spoken phrases: they are never decoder hotwords.
    return [(phrase, value) for phrase, value in items if _is_spoken(phrase)]


def _is_spoken(phrase: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in phrase)
