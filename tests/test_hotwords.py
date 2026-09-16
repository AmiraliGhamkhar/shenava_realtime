"""Decoder hotword lists: bounded, conservative, deterministic, single-source."""
from shenava_realtime.hotwords import BIAS_BY_CATEGORY, MAX_BIAS, build_hotwords
from shenava_realtime.terminology import TerminologyRule

RULES = [
    TerminologyRule("drug.1", "متفورمین", ("متفورمین",), category="medication",
                    specialty="general", priority=50),
    TerminologyRule("drug.2", "وازوپرسین", ("وازوپرسین",), category="medication",
                    specialty="general", priority=50),
    TerminologyRule("term.cardio", "استنوز", ("استنوز",), category="medical_term",
                    specialty="cardiology", priority=95),
    TerminologyRule("proc.cardio", "آنژیوگرافی", ("آنژیوگرافی",), category="procedure",
                    specialty="cardiology", priority=90),
    TerminologyRule("abbr.1", "اکو کاردیوگرافی", ("اکو کاردیوگرافی",),
                    category="imaging", specialty="general", priority=60),
    TerminologyRule("unit.mg", "mg", ("میلی گرم",), category="unit", priority=80),
    TerminologyRule("anat.knee", "زانو", ("زانو",), category="anatomy", priority=50),
    TerminologyRule("off.1", "خاموش", ("خاموش",), category="medical_term",
                    specialty="general", priority=99, enabled=False),
]


def test_units_anatomy_and_disabled_rules_are_never_boosted():
    phrases = {h.phrase for h in build_hotwords(RULES)}
    assert "میلی گرم" not in phrases
    assert "زانو" not in phrases
    assert "خاموش" not in phrases


def test_specialty_subset_keeps_general_terms():
    cardiology = {h.phrase for h in build_hotwords(RULES, specialty="cardiology")}
    assert cardiology == {"استنوز", "آنژیوگرافی", "متفورمین", "وازوپرسین", "اکو کاردیوگرافی"}


def test_none_specialty_uses_only_general_terms():
    phrases = {h.phrase for h in build_hotwords(RULES)}
    assert phrases == {"متفورمین", "وازوپرسین", "اکو کاردیوگرافی"}


def test_selection_is_deterministic_priority_then_id():
    ordered = [h.phrase for h in build_hotwords(RULES, specialty="cardiology")]
    assert ordered[0] == "استنوز"          # priority 95
    assert ordered.index("آنژیوگرافی") == 1  # priority 90
    assert "متفورمین" in ordered and "وازوپرسین" in ordered  # tied priority: id order


def test_bias_values_are_conservative_and_capped():
    for hotword in build_hotwords(RULES):
        assert 0.0 < hotword.bias <= MAX_BIAS
    by_phrase = {h.phrase: h.bias for h in build_hotwords(RULES)}
    assert by_phrase["متفورمین"] == BIAS_BY_CATEGORY["medication"]
    assert by_phrase["اکو کاردیوگرافی"] == BIAS_BY_CATEGORY["imaging"]


def test_list_is_bounded_and_deduplicated():
    big = [
        TerminologyRule(f"t.{i:03d}", f"واژه{i}", (f"واژه{i}",), category="medical_term",
                        specialty="general", priority=50)
        for i in range(10)
    ] + [TerminologyRule("dup", "مشترک", ("مشترک",), category="medical_term",
                         specialty="general", priority=90)]
    limited = build_hotwords(big, max_hotwords=4)
    assert len(limited) == 4
    phrases = [h.phrase for h in limited]
    assert len(phrases) == len(set(phrases))


def test_invalid_limits_rejected():
    import pytest
    with pytest.raises(ValueError):
        build_hotwords(RULES, max_hotwords=0)
    with pytest.raises(ValueError):
        build_hotwords(RULES, max_hotwords=513)
