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


@pytest.mark.parametrize("spoken,expected", [
    ("نبض چهل در دقیقه", "نبض 40 bpm"),
    ("نفس بیست در دقیقه", "نفس 20 rpm"),
    ("ضربان قلب نود و دو در دقیقه", "ضربان قلب 92 bpm"),
    ("وازوپرسین بیست و پنج میلی گرم در کیلو گرم", "وازوپرسین 25 mg/kg"),
    ("وازوپرسین بیست و پنج mg در kg", "وازوپرسین 25 mg/kg"),
    ("فشار خون هشتاد روی صد و بیست", "BP 80 روی 120"),
    ("اشباع اکسیژن صد و دو درصد", "SpO2 102%"),
])
def test_rate_and_ratio_grammars_preserve_source_on_suspicion(processor, spoken, expected):
    # Doses/units get parsed; impossible values stay verbatim and are flagged.
    assert processor.process(spoken) == expected


def test_rate_keyword_never_rewrites_an_already_unitful_value(processor):
    # "نبض 70" keeps its own form; no unit is invented where absent.
    assert processor.process("نبض ۷۰") == "نبض 70"
    assert processor.last_result.review_required


def test_self_alias_is_not_flagged_as_unsafe(processor):
    # A rule whose output equals the matched text (Latin "EKG" for "EKG") is a
    # no-op rewrite, not a protected-span conflict: review must stay off.
    assert processor.process("بیمار EKG مثبت است") == "بیمار EKG مثبت است"
    assert not processor.last_result.review_required
    assert processor.last_result.review_reasons == []


def test_suspicious_values_are_reviewed_but_preserved(processor):
    out = processor.process("دمای بدن پنجاه درجه")
    assert out == "دمای بدن 50°"
    assert processor.last_result.review_required
    assert "suspicious_value:temperature[°]=50" in processor.last_result.review_reasons
    assert any(i.reason == "suspicious_value:temperature[°]=50"
               for i in processor.last_result.value_issues)


def test_unknown_unit_variant_preserved_not_invented(processor):
    # ASR heard a unit variant the reviewed lexicon does not contain: the
    # number is converted, the unfamiliar unit word is preserved verbatim.
    assert processor.process("متفورمین پانصد ملی گرم") == "متفورمین 500 ملی g"
    assert not any(r.startswith("suspicious_value") for r in processor.last_result.review_reasons)
