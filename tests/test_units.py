"""Unit rewriting and unit + number combinations."""

import pytest

from shenava_realtime.postprocessor import PostProcessor


@pytest.fixture(scope="module")
def processor() -> PostProcessor:
    return PostProcessor()


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("پنج میلی گرم", "5 mg"),
        ("پنج میلیگرم", "5 mg"),
        ("ده میلی لیتر", "10 mL"),
        ("دو گرم", "2 g"),
        ("هفتاد کیلوگرم", "70 kg"),
        ("پانصد سی سی", "500 cc"),
        ("دو سانتی متر", "2 cm"),
        ("یک میلی اکی والان", "1 mEq/L"),
        ("ده واحد بین المللی", "10 IU"),
        ("بیست قطره در دقیقه", "20 gtt/min"),
        ("هفتاد و پنج بار در دقیقه", "75 bpm"),
        ("بیست نفس در دقیقه", "20 rpm"),
        ("پنج میلی گرم در روز", "5 mg/day"),
        ("ده میلی لیتر در ساعت", "10 mL/h"),
        ("سی و هشت درجه سانتیگراد", "38°C"),
        ("سی و پنج درصد", "35%"),
    ],
)
def test_units(processor: PostProcessor, spoken: str, expected: str):
    assert processor.process(spoken) == expected


def test_unit_follows_the_number_without_extra_space(processor: PostProcessor):
    result = processor.process("متفورال پنج میلی گرم دو بار در روز")
    assert result.startswith("متفورال 5 mg")


def test_percent_attaches_to_the_number(processor: PostProcessor):
    assert processor.process("اشباع اکسیژن نود و پنج درصد") == "SpO2 95%"


def test_degree_attaches_to_the_number(processor: PostProcessor):
    assert processor.process("دمای بدن سی و هفت درجه") == "دمای بدن 37°"


def test_unknown_unit_like_words_are_left_alone(processor: PostProcessor):
    assert processor.process("یک لیوان آب") == "یک لیوان آب"


def test_unit_conversion_is_idempotent(processor: PostProcessor):
    once = processor.process("پنج میلی گرم")
    assert processor.process(once) == once
