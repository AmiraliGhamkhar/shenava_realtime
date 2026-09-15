"""Offset-preserving Persian number grammar (no guessing or fuzzy matching)."""
from __future__ import annotations
import re
from fractions import Fraction
from typing import Iterable

from .fa_numbers import parse_number_phrase
from .spans import NumberSpan
from .text_normalize import digits_to_ascii, match_key

_TOKEN_RE = re.compile(r"-?\d+(?:[./]\d+)?|[\w\u200c]+", re.UNICODE)
_HARD = re.compile(r"[.!?;،؛؟\n]")
_FRACTIONS = {"نیم": Fraction(1, 2), "نصف": Fraction(1, 2), "یک دوم": Fraction(1, 2),
              "یک سوم": Fraction(1, 3), "دو سوم": Fraction(2, 3),
              "یک چهارم": Fraction(1, 4), "سه چهارم": Fraction(3, 4)}


class PersianNumberGrammar:
    def parse(self, text: str, *, unit_tokens: set[str] | None = None) -> list[NumberSpan]:
        tokens = list(_TOKEN_RE.finditer(text))
        words = [m.group() for m in tokens]
        spans: list[NumberSpan] = []
        i = 0
        while i < len(tokens):
            raw = digits_to_ascii(words[i])
            if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
                value = float(raw) if "." in raw else int(raw)
                spans.append(NumberSpan(words[i], value, tokens[i].start(), tokens[i].end(),
                                        "decimal" if "." in raw else "cardinal"))
                i += 1; continue
            # Explicit lexical fractions, longest first.
            fraction_hit = None
            for phrase, value in sorted(_FRACTIONS.items(), key=lambda x: -len(x[0])):
                n = len(phrase.split())
                if " ".join(match_key(x) for x in words[i:i+n]) == phrase and self._safe(tokens, text, i, i+n):
                    fraction_hit = (n, value); break
            if fraction_hit:
                n, value = fraction_hit
                spans.append(NumberSpan(text[tokens[i].start():tokens[i+n-1].end()], float(value),
                                        tokens[i].start(), tokens[i+n-1].end(), "fraction"))
                i += n; continue
            # ``هر دو زانو`` is bilateral anatomy, not a quantity to render.
            if i and match_key(words[i - 1]) == "هر" and match_key(words[i]) == "دو":
                i += 1; continue
            parsed = parse_number_phrase(words, i, unit_tokens)
            if parsed is None:
                i += 1; continue
            value, end = parsed
            if not self._safe(tokens, text, i, end):
                i += 1; continue
            number_type = "cardinal"
            # Spoken decimal; fractional side is interpreted digit-by-digit only.
            if end < len(words) and match_key(words[end]) == "ممیز" and end + 1 < len(words):
                frac_end = end + 1
                frac_digits = []
                while frac_end < len(words):
                    p = parse_number_phrase(words, frac_end, unit_tokens)
                    if p is None or p[1] != frac_end + 1:
                        break
                    digit = int(p[0])
                    if not 0 <= digit <= 9:
                        break
                    frac_digits.append(str(digit)); frac_end += 1
                if frac_digits and self._safe(tokens, text, i, frac_end):
                    value = float(f"{int(value)}.{''.join(frac_digits)}")
                    end = frac_end; number_type = "decimal"
            start = tokens[i].start()
            if i and match_key(words[i-1]) == "منفی" and self._safe(tokens, text, i-1, end):
                start = tokens[i-1].start(); value = -value; i -= 1
            spans.append(NumberSpan(text[start:tokens[end-1].end()], int(value) if float(value).is_integer() else value,
                                    start, tokens[end-1].end(), number_type))
            i = end
        return self._merge_ranges(text, spans)

    @staticmethod
    def _safe(tokens, text, start, end) -> bool:
        return all(not _HARD.search(text[tokens[x].end():tokens[x+1].start()])
                   for x in range(start, end - 1))

    @staticmethod
    def _merge_ranges(text: str, spans: list[NumberSpan]) -> list[NumberSpan]:
        out: list[NumberSpan] = []; i = 0
        while i < len(spans):
            if i + 1 < len(spans):
                between = text[spans[i].end:spans[i+1].start]
                if re.fullmatch(r"\s*(?:تا|الی|[-–])\s*", between):
                    out.append(NumberSpan(text[spans[i].start:spans[i+1].end],
                        f"{spans[i].value}-{spans[i+1].value}", spans[i].start,
                        spans[i+1].end, "range")); i += 2; continue
            out.append(spans[i]); i += 1
        return out
