"""Persian/Arabic Unicode normalization (deterministic, rule based).

No models, no fuzzy matching: explicit code-point maps plus a fixed set of
regular expressions.  The pass is idempotent, so it is safe to run on partial
hypotheses as often as they arrive.
"""

from __future__ import annotations

import re
import unicodedata

ZWNJ = "\u200c"
ZWJ = "\u200d"
ZWSP = "\u200b"
BOM = "\ufeff"
TATWEEL = "\u0640"

# Arabic glyphs that Persian text should not contain.
ARABIC_TO_PERSIAN = {
    "\u064a": "\u06cc",  # ARABIC YEH -> PERSIAN YEH        (ي -> ی)
    "\u0649": "\u06cc",  # ALEF MAKSURA -> PERSIAN YEH      (ى -> ی)
    "\u0643": "\u06a9",  # ARABIC KAF -> PERSIAN KEHEH      (ك -> ک)
    "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE -> ALEF    (أ -> ا)
    "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW -> ALEF    (إ -> ا)
    "\u0671": "\u0627",  # ALEF WASLA -> ALEF               (ٱ -> ا)
    "\u0629": "\u0647",  # TEH MARBUTA -> HEH               (ة -> ه)
    "\u0624": "\u0648",  # WAW WITH HAMZA -> WAW            (ؤ -> و)
}

# Harakat / koranic marks / filler characters that carry no meaning here.
_STRIP_RANGES = ((0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06ED))
_STRIP_CHARS = {chr(code) for start, end in _STRIP_RANGES for code in range(start, end + 1)}
_STRIP_CHARS.update({ZWJ, ZWSP, BOM, TATWEEL, "\u200e", "\u200f", *map(chr, range(0x202a, 0x202f)), *map(chr, range(0x2066, 0x206a))})

_ARABIC_INDIC_TO_ASCII = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669", "0123456789")
_PERSIAN_TO_ASCII = str.maketrans("\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9", "0123456789")
_ASCII_TO_PERSIAN = str.maketrans("0123456789", "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9")

QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})

# Deterministic Persian orthography: clitics written with a ZWNJ.
_PREFIX_CLITICS = ("می", "نمی")
_SUFFIX_CLITICS = (
    "ها", "های", "هایی", "تر", "ترین", "ای",
    "ام", "ات", "اش", "مان", "تان", "شان",
)

_PUNCT = ".,!?;:\u060c\u061b\u061f%)"
_RE_SPACE_BEFORE_PUNCT = re.compile(rf"\s+([{re.escape(_PUNCT)}])")
_RE_SPACE_AFTER_PUNCT = re.compile(rf"([{re.escape(_PUNCT[:-2])}])(?=\S)")
_RE_REPEAT_ZWNJ = re.compile(r"(?:\u200c)+")
_RE_MULTI_SPACE = re.compile(r"[ \t\u00a0]{2,}")
_RE_WHITESPACE = re.compile(r"\s+")


def strip_marks(text: str) -> str:
    """Remove harakat, tatweel, ZWJ/ZWSP/BOM (ZWNJ is preserved)."""
    return "".join(ch for ch in text if ch not in _STRIP_CHARS)


def map_letters(text: str) -> str:
    return text.translate(str.maketrans(ARABIC_TO_PERSIAN))


def digits_to_ascii(text: str) -> str:
    return text.translate(_ARABIC_INDIC_TO_ASCII).translate(_PERSIAN_TO_ASCII)


def digits_to_persian(text: str) -> str:
    return digits_to_ascii(text).translate(_ASCII_TO_PERSIAN)


def fix_punctuation(text: str) -> str:
    """No space before punctuation, exactly one space after it."""
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    # Keep decimals, thousands separators and ratios/times intact.
    text = _RE_SPACE_AFTER_PUNCT.sub(
        lambda m: m.group(0) if m.start() > 0 and text[m.start()-1].isdigit()
        and m.end() < len(text) and text[m.end()].isdigit() else m.group(0) + " ", text
    )
    return text


def _is_persian_word(word: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in word.replace(ZWNJ, ""))


def join_affixes(text: str) -> str:
    """Insert the missing ZWNJ between clitics and the word they attach to."""
    words = text.split(" ")
    out: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word in _PREFIX_CLITICS and index + 1 < len(words) and _is_persian_word(words[index + 1]):
            out.append(f"{word}{ZWNJ}{words[index + 1]}")
            index += 2
            continue
        if out and word in _SUFFIX_CLITICS and _is_persian_word(out[-1]):
            out[-1] = f"{out[-1]}{ZWNJ}{word}"
            index += 1
            continue
        out.append(word)
        index += 1
    return " ".join(out)


def normalize(
    text: str,
    *,
    to_ascii_digits: bool = True,
    join_persian_affixes: bool = True,
    punctuation: bool = True,
) -> str:
    """Full normalization pass.

    NFKC first (folds Arabic presentation forms and compatibility glyphs), then
    the explicit Persian maps, then whitespace/punctuation tidy-up.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = strip_marks(text)
    text = map_letters(text)
    text = text.translate(QUOTES).replace("٫", ".")
    text = re.sub(r"(?<=\d)٬(?=\d{3}(?:\D|$))", "", text)
    text = digits_to_ascii(text) if to_ascii_digits else digits_to_persian(text)

    # Collapse any whitespace run (tabs/newlines/NBSP from ASR output) and drop
    # stray ZWNJ at word edges.
    text = _RE_WHITESPACE.sub(" ", text).strip()
    words = [word.strip(ZWNJ) for word in text.split(" ")]
    text = " ".join(word for word in words if word)

    if join_persian_affixes:
        text = join_affixes(text)
    if punctuation:
        text = fix_punctuation(text)

    text = _RE_REPEAT_ZWNJ.sub(ZWNJ, text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def match_key(token: str) -> str:
    """Canonical form of a token, used only for deterministic dictionary lookups."""
    token = token.replace(ZWNJ, "").replace(ZWJ, "").lower()
    token = strip_marks(token)
    token = map_letters(token)
    token = digits_to_ascii(token)
    return token.strip(_PUNCT + "()\"'")
