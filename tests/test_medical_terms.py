"""Medical terminology / abbreviation rewriting (deterministic FST)."""

import pytest

from shenava_realtime.lexicon import MEDICAL_TERMS, UNITS, build_rewriter
from shenava_realtime.postprocessor import PostProcessor


@pytest.fixture(scope="module")
def processor() -> PostProcessor:
    return PostProcessor()


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("کابج", "CABG"),
        ("سی ای بی جی", "CABG"),
        ("سي اي بي جي", "CABG"),  # Arabic yeh from a different ASR/tokenizer
        ("پی سی آی", "PCI"),
        ("ای کی جی", "EKG"),
        ("ای سی جی", "ECG"),
        ("نوار قلب", "ECG"),
        ("ام آر آی", "MRI"),
        ("سی تی اسکن", "CT"),
        ("آی سی یو", "ICU"),
        ("سی پی آر", "CPR"),
        ("سی بی سی", "CBC"),
        ("آی وی", "IV"),
    ],
)
def test_medical_abbreviations(processor: PostProcessor, spoken: str, expected: str):
    assert processor.process(spoken) == expected


def test_abbreviation_inside_a_sentence(processor: PostProcessor):
    assert (
        processor.process("بیمار کاندید کابج است")
        == "بیمار کاندید CABG است"
    )


def test_letter_spoken_phrase_is_not_read_as_a_number(processor: PostProcessor):
    # "سی" on its own is 30; inside the CABG phrase it must stay a letter.
    assert processor.process("سی ای بی جی") == "CABG"
    assert "30" not in processor.process("سی ای بی جی")


def test_unknown_words_are_untouched(processor: PostProcessor):
    text = "بیمار با شکایت درد سینه مراجعه کرد"
    assert processor.process(text) == text


def test_tables_have_no_conflicting_rules():
    # A conflict would raise while building the automaton.
    fst = build_rewriter()
    assert len(fst) == len(MEDICAL_TERMS) + len(UNITS)
    assert fst.max_phrase_len >= 3


def test_longest_match_wins():
    fst = build_rewriter()
    # "سی سی یو" (CCU) must beat "سی سی" (cc).
    assert fst.apply("بیمار در سی سی یو بستری است") == "بیمار در CCU بستری است"
    assert fst.apply("پانصد سی سی سرم") == "پانصد cc سرم"


def test_repetition_removal_does_not_eat_doubled_letters(processor: PostProcessor):
    # "سی سی" is a legitimate doubled token, not a CTC repetition.
    assert processor.process("سی سی نرمال سالین") == "cc نرمال سالین"


def test_extra_terms_can_be_injected():
    processor = PostProcessor(extra_terms={"آزیترومایسین": "azithromycin"})
    assert processor.process("آزیترومایسین") == "azithromycin"
