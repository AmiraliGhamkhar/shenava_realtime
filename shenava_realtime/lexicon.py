"""Deterministic rewrite tables (medical terms, abbreviations, units).

Every entry is an exact phrase -> replacement mapping consumed by the
Aho-Corasick matcher (``aho_corasick.py``) and the measurement grammar.  Keys
are matched after Persian normalization, and multi-word keys win over shorter
ones because the resolver takes the longest match.

Spelling variants (with and without a ZWNJ, singular/plural) are listed
explicitly rather than guessed at runtime.  The reviewed source of truth for
pipeline terminology is ``data/terminology.json``; keep the unit table here
in sync with its ``unit`` rules.
"""

from __future__ import annotations

from typing import Dict

from .aho_corasick import AhoCorasickMatcher

# --------------------------------------------------------------------------- #
# Medical terminology / abbreviations (Persian spoken form -> standard acronym)
# --------------------------------------------------------------------------- #
MEDICAL_TERMS: Dict[str, str] = {
    # Cardiology / interventions
    "کابج": "CABG",
    "ک ا ب ج": "CABG",
    "سی ای بی جی": "CABG",
    "پی سی آی": "PCI",
    "پی تی سی آی": "PTCA",
    "ای کی جی": "EKG",
    "ای سی جی": "ECG",
    "نوار قلب": "ECG",
    "اکو کاردیوگرافی": "اکوکاردیوگرافی",
    "انفارکتوس میوکارد": "MI",
    "نارسایی احتقانی قلب": "CHF",
    "بیماری عروق کرونر": "CAD",
    "فشار خون": "BP",
    "ب پی": "BP",
    # Diagnostics / labs
    "سی تی اسکن": "CT",
    "سی تی": "CT",
    "ام آر آی": "MRI",
    "سی بی سی": "CBC",
    "قند خون ناشتا": "FBS",
    "هموگلوبین ای وان سی": "HbA1c",
    "ای ان آر": "INR",
    "سی آر پی": "CRP",
    "بی ان پی": "BNP",
    "اکسیژن خون": "SpO2",
    "اشباع اکسیژن": "SpO2",
    # Care units / procedures
    "آی سی یو": "ICU",
    "سی سی یو": "CCU",
    "ان آی سی یو": "NICU",
    "سی پی آر": "CPR",
    "دی وی تی": "DVT",
    "آمبولی ریه": "PE",
    "پی ای": "PE",
    "سکته مغزی": "CVA",
    "بیماری مزمن کلیه": "CKD",
    # Routes / lines
    "آی وی": "IV",
    "آی ام": "IM",
    "پی او": "PO",
    "اس سی": "SC",
    "آنژیوکت": "IV cannula",
    "سی وی سی": "CVC",
    "ان جی تی": "NGT",
    "ساتوراسیون": "SpO2",
}

# --------------------------------------------------------------------------- #
# Units of measurement
# --------------------------------------------------------------------------- #
UNITS: Dict[str, str] = {
    # Mass
    "میکرو گرم": "mcg",
    "میکروگرم": "mcg",
    "میلی گرم": "mg",
    "میلیگرم": "mg",
    "گرم": "g",
    "کیلو گرم": "kg",
    "کیلوگرم": "kg",
    # Volume
    "میلی لیتر": "mL",
    "میلیلیتر": "mL",
    "لیتر": "L",
    "سی سی": "cc",
    # Length
    "میلی متر": "mm",
    "میلیمتر": "mm",
    "سانتی متر": "cm",
    "سانتیمتر": "cm",
    "متر": "m",
    # Chemistry / concentrations
    "میلی اکی والان": "mEq/L",
    "میلی اکیوالان": "mEq/L",
    "واحد بین المللی": "IU",
    "واحد انسولین": "U insulin",
    # Temperature
    "درجه سانتیگراد": "°C",
    "درجه سانتی گراد": "°C",
    "سانتیگراد": "°C",
    "درجه": "°",
    # Rates
    "ضربان در دقیقه": "bpm",
    "ضربان قلب در دقیقه": "bpm",
    "بار در دقیقه": "bpm",
    "نفس در دقیقه": "rpm",
    "دور در دقیقه": "rpm",
    "قطره در دقیقه": "gtt/min",
    "میلی گرم در دقیقه": "mg/min",
    "میلی گرم در ساعت": "mg/h",
    "میلی گرم در روز": "mg/day",
    "میلی لیتر در ساعت": "mL/h",
    "میلی لیتر در دقیقه": "mL/min",
    "میلی گرم در دسی لیتر": "mg/dL",
    "میلی مول در لیتر": "mmol/L",
    "میلی گرم در کیلو گرم": "mg/kg",
    "میلیگرم در کیلوگرم": "mg/kg",
    "میلی متر جیوه": "mmHg",
    "میکروگرم در میلی لیتر": "mcg/mL",
    "μg": "mcg",
    "milligram": "mg",
    "milligrams": "mg",
    "میکرو گرم در دقیقه": "mcg/min",
    # Percent (kept last: the trie takes the longest match anyway)
    "درصد": "%",
    "در صد": "%",
}


def build_rewriter(
    *,
    medical_terms: bool = True,
    units: bool = True,
    extra_terms: Dict[str, str] | None = None,
) -> AhoCorasickMatcher:
    """Build the deterministic multi-pattern transducer (compatibility API)."""
    fst = AhoCorasickMatcher()
    if medical_terms:
        fst.add_many(MEDICAL_TERMS)
        fst.add_many({value: value for value in MEDICAL_TERMS.values() if value.isascii()})
    if units:
        fst.add_many(UNITS)
    if extra_terms:
        fst.add_many(extra_terms)
    return fst
