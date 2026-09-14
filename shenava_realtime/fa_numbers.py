"""Spoken Persian numbers -> digits (deterministic parser, no models).

Handles the forms that show up in dictation::

    سی و پنج            -> 35
    صد و بیست           -> 120
    دو هزار و سیصد      -> 2300
    یک و نیم            -> 1.5

plus the medical shorthands built on top of them::

    سی و پنج درصد        -> 35%
    صد و بیست روی هشتاد  -> 120/80
    سی و هشت درجه        -> 38°

The grammar is a small recursive-free accumulation over an explicit word table;
anything not in the table is passed through untouched.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .text_normalize import digits_to_ascii, digits_to_persian, match_key

DIGITS: Dict[str, int] = {
    "صفر": 0,
    "یک": 1,
    "دو": 2,
    "سه": 3,
    "چهار": 4,
    "پنج": 5,
    "شش": 6,
    "هفت": 7,
    "هشت": 8,
    "نه": 9,
}

TEENS: Dict[str, int] = {
    "ده": 10,
    "یازده": 11,
    "دوازده": 12,
    "سیزده": 13,
    "چهارده": 14,
    "پانزده": 15,
    "شانزده": 16,
    "هفده": 17,
    "هجده": 18,
    "نوزده": 19,
}

TENS: Dict[str, int] = {
    "بیست": 20,
    "سی": 30,
    "چهل": 40,
    "پنجاه": 50,
    "شصت": 60,
    "هفتاد": 70,
    "هشتاد": 80,
    "نود": 90,
}

HUNDREDS: Dict[str, int] = {
    "صد": 100,
    "یکصد": 100,
    "دویست": 200,
    "سیصد": 300,
    "چهارصد": 400,
    "پانصد": 500,
    "ششصد": 600,
    "هفتصد": 700,
    "هشتصد": 800,
    "نهصد": 900,
}

SCALES: Dict[str, int] = {
    "هزار": 1_000,
    "میلیون": 1_000_000,
    "میلیارد": 1_000_000_000,
}

CONNECTOR = "و"
HALF = "نیم"

_STARTERS: Set[str] = set(DIGITS) | set(TEENS) | set(TENS) | set(HUNDREDS) | set(SCALES)
_VALUE_WORDS: Dict[str, int] = {**DIGITS, **TEENS, **TENS, **HUNDREDS}

# "یک" doubles as the indefinite article ("a patient"); only read it as the
# number 1 when it is part of a longer phrase or when it quantifies a unit.
_AMBIGUOUS_ONES = {"یک", "1"}

_BP_RE = re.compile(r"(?<![\d/.])(\d{2,3})\s+(?:روی|بر\s+روی|به|خط|از)\s+(\d{2,3})(?![\d/.])")
_PERCENT_RE = re.compile(r"(\d)\s+(%|٪)")
_DEGREE_RE = re.compile(r"(\d)\s+(°C|°F|°)")
_STANDALONE_PERCENT_RE = re.compile(r"(\d)\s+درصد\b")


def parse_number_phrase(
    words: Sequence[str],
    start: int,
    unit_tokens: Optional[Set[str]] = None,
) -> Optional[Tuple[float, int]]:
    """Parse the number phrase beginning at ``words[start]``.

    Returns ``(value, index_after_phrase)`` or ``None`` when the token at
    ``start`` does not begin a number phrase.
    """
    if start >= len(words):
        return None
    first_key = match_key(words[start])
    if first_key not in _STARTERS:
        return None

    total = 0.0
    current = 0.0
    value_words = 0
    index = start
    count = len(words)

    while index < count:
        key = match_key(words[index])
        if key in _VALUE_WORDS:
            current += _VALUE_WORDS[key]
            value_words += 1
            index += 1
        elif key in SCALES:
            current = (current if current else 1) * SCALES[key]
            total += current
            current = 0.0
            value_words += 1
            index += 1
        elif key == CONNECTOR:
            next_key = match_key(words[index + 1]) if index + 1 < count else ""
            if next_key in _STARTERS or next_key == HALF:
                index += 1
            else:
                break
        elif key == HALF:
            current += 0.5
            value_words += 1
            index += 1
        else:
            break

    if value_words == 0:
        return None

    # Guard the article reading of "یک": keep the word unless it clearly
    # quantifies something numeric.
    if value_words == 1 and match_key(words[start]) in _AMBIGUOUS_ONES:
        next_key = match_key(words[index]) if index < count else ""
        if not unit_tokens or next_key not in unit_tokens:
            return None

    return total + current, index


def format_number(value: float, digits: str = "ascii") -> str:
    """Render a numeric value with the requested digit style."""
    text = str(int(value)) if float(value).is_integer() else f"{value:.10g}"
    return text if digits == "ascii" else digits_to_persian(text)


def convert_numbers(
    text: str,
    *,
    digits: str = "ascii",
    unit_tokens: Optional[Set[str]] = None,
) -> str:
    """Replace spoken numbers (and the medical patterns built on them)."""
    if not text:
        return text

    text = digits_to_ascii(text)
    words: List[str] = text.split(" ")
    out: List[str] = []
    index = 0
    count = len(words)

    while index < count:
        parsed = parse_number_phrase(words, index, unit_tokens)
        if parsed is None:
            out.append(words[index])
            index += 1
            continue
        value, index = parsed
        out.append(format_number(value, digits))

    result = " ".join(word for word in out if word)
    result = _STANDALONE_PERCENT_RE.sub(r"\1%", result)
    result = _PERCENT_RE.sub(r"\1%", result)
    result = _DEGREE_RE.sub(r"\1\2", result)
    return _BP_RE.sub(lambda match: _blood_pressure(match, digits), result)


def _blood_pressure(match: "re.Match[str]", digits: str = "ascii") -> str:
    systolic = int(match.group(1))
    diastolic = int(match.group(2))
    # Only rewrite when the pair is physiologically plausible; otherwise leave
    # the sentence alone instead of guessing.
    if 60 <= systolic <= 260 and 30 <= diastolic <= 160 and systolic > diastolic:
        return format_number(systolic, digits) + "/" + format_number(diastolic, digits)
    return match.group(0)
