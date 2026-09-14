"""End-to-end post-processing on realistic dictation, plus the pipeline order."""

import pytest

from shenava_realtime.config import PostProcessConfig
from shenava_realtime.postprocessor import PostProcessor


@pytest.fixture(scope="module")
def processor() -> PostProcessor:
    return PostProcessor()


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        # The examples from the specification.
        ("کابج", "CABG"),
        ("سی ای بی جی", "CABG"),
        ("پی سی آی", "PCI"),
        ("پنج میلی گرم", "5 mg"),
        ("سی و پنج درصد", "35%"),
        ("صد و بیست روی هشتاد", "120/80"),
    ],
)
def test_specification_examples(processor: PostProcessor, spoken: str, expected: str):
    assert processor.process(spoken) == expected


def test_full_medical_note(processor: PostProcessor):
    spoken = (
        "بيمار مرد ۵۶ ساله با فشار خون صد و بيست روي هشتاد و ضربان هفتاد و پنج "
        "بار در دقيقه مراجعه كرد . دوز متفورال پنج ميلي گرم و اكسيژن خون نود و پنج درصد"
    )
    assert processor.process(spoken) == (
        "بیمار مرد 56 ساله با BP 120/80 و ضربان 75 bpm مراجعه کرد. "
        "دوز متفورال 5 mg و SpO2 95%"
    )


def test_normalization_runs_before_everything(processor: PostProcessor):
    # Arabic yeh/kaf and Arabic-Indic digits must not block the FST or numbers.
    assert processor.process("سي و پنج درصد") == "35%"
    assert processor.process("٥ ميلي گرم") == "5 mg"


def test_disabled_postprocessing_returns_trimmed_text():
    config = PostProcessConfig(enabled=False)
    processor = PostProcessor(config)
    assert processor.process("  سي و پنج درصد  ") == "سي و پنج درصد"


def test_individual_switches():
    config = PostProcessConfig(convert_numbers=False)
    assert PostProcessor(config).process("پنج میلی گرم") == "پنج mg"

    config = PostProcessConfig(medical_terms=False, units=True)
    assert PostProcessor(config).process("کابج") == "کابج"
    assert PostProcessor(config).process("پنج میلی گرم") == "5 mg"

    config = PostProcessConfig(units=False, medical_terms=True)
    assert PostProcessor(config).process("کابج") == "CABG"
    assert PostProcessor(config).process("پنج میلی گرم") == "5 میلی گرم"

    config = PostProcessConfig(remove_repetitions=False)
    assert PostProcessor(config).process("بیمار بیمار آمد") == "بیمار بیمار آمد"


def test_persian_digit_output_style():
    config = PostProcessConfig()
    config.digits = config.digits.__class__("persian")
    assert PostProcessor(config).process("صد و بیست روی هشتاد") == "۱۲۰/۸۰"


def test_empty_and_none_input(processor: PostProcessor):
    assert processor.process("") == ""
    assert processor.process("   ") == ""


def test_processing_is_idempotent_on_real_output(processor: PostProcessor):
    samples = [
        "بیمار تحت عمل CABG قرار گرفت",
        "BP 120/80 و ضربان 75 bpm",
        "دوز 5 mg دو بار در روز",
        "می‌رود و کتاب‌ها را می‌بیند",
    ]
    for sample in samples:
        once = processor.process(sample)
        assert processor.process(once) == once
