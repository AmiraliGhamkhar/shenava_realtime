"""Persian/Arabic Unicode normalization."""

from shenava_realtime.text_normalize import (
    ZWNJ,
    digits_to_ascii,
    digits_to_persian,
    join_affixes,
    match_key,
    normalize,
)


def test_arabic_letters_map_to_persian():
    assert normalize("علي كتاب") == "علی کتاب"
    assert normalize("مريض") == "مریض"


def test_hamza_and_teh_marbuta_are_folded():
    assert normalize("أحمد إبن") == "احمد ابن"
    assert normalize("الزهرة") == "الزهره"  # ARABIC TEH MARBUTA -> HEH
    assert normalize("مريضه") == "مریضه"


def test_diacritics_and_tatweel_are_removed():
    assert normalize("بَـیمار") == "بیمار"
    assert normalize("کِتابٌ") == "کتاب"


def test_zero_width_characters_are_tidy():
    # ZWJ/ZWSP/BOM disappear, ZWNJ inside a word survives, edge ZWNJ does not.
    assert normalize("بیمار\u200b\u200d") == "بیمار"
    assert normalize("\u200cبیمار\u200c") == "بیمار"
    assert normalize("کتاب\u200cخانه") == "کتاب\u200cخانه"


def test_digits_are_normalized_to_ascii_by_default():
    assert normalize("۱۲۳ و ٤٥٦") == "123 و 456"
    assert digits_to_ascii("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩") == "01234567890123456789"


def test_digits_can_be_kept_persian():
    assert normalize("120/80", to_ascii_digits=False) == "۱۲۰/۸۰"
    assert digits_to_persian("120") == "۱۲۰"


def test_whitespace_and_punctuation_spacing():
    assert normalize("بیمار   آمد .  دکتر") == "بیمار آمد. دکتر"
    assert normalize("فشار\tخون\n۱۲۰") == "فشار خون 120"
    assert normalize("  سطر\u00a0اول  \r\n  سطر دوم ") == "سطر اول سطر دوم"


def test_clitics_are_joined_with_zwnj():
    assert join_affixes("می روم") == f"می{ZWNJ}روم"
    assert join_affixes("کتاب ها") == f"کتاب{ZWNJ}ها"
    assert join_affixes("نمی توانم") == f"نمی{ZWNJ}توانم"


def test_clitics_are_not_joined_to_latin_words():
    assert join_affixes("CABG ها") == "CABG ها"


def test_normalize_is_idempotent():
    samples = [
        "بیمار تحت عمل CABG قرار گرفت",
        "می روم و کتاب ها را می بینم",
        "فشار خون 120/80 و ضربان 75",
        "علي ۵ میلی گرم متفورال خورد",
    ]
    for sample in samples:
        once = normalize(sample)
        assert normalize(once) == once


def test_full_pass_end_to_end():
    text = "بيمار   ۵ ميلي گرم متفورال  گرفت ."
    assert normalize(text) == "بیمار 5 میلی گرم متفورال گرفت."


def test_match_key_is_lookup_canonical():
    assert match_key("میلی‌گرم") == match_key("میلیگرم")
    assert match_key("CABG") == match_key("cabg")
    assert match_key("کابج،") == "کابج"
    assert match_key("۵") == "5"
