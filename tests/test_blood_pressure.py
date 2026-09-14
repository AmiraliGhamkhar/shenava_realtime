"""Blood pressure and other "number connector number" medical patterns."""

import pytest

from shenava_realtime.fa_numbers import convert_numbers
from shenava_realtime.postprocessor import PostProcessor


@pytest.fixture(scope="module")
def processor() -> PostProcessor:
    return PostProcessor()


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("صد و بیست روی هشتاد", "120/80"),
        ("فشار خون صد و بیست روی هشتاد", "BP 120/80"),
        ("صد و سی به هشتاد و پنج", "130/85"),
        ("صد و ده خط هفتاد", "110/70"),
        ("صد و چهل بر روی نود", "140/90"),
    ],
)
def test_blood_pressure(processor: PostProcessor, spoken: str, expected: str):
    assert processor.process(spoken) == expected


def test_bp_in_a_full_sentence(processor: PostProcessor):
    result = processor.process("فشار خون بیمار صد و بیست روی هشتاد و ضربان هفتاد و پنج")
    assert result == "BP بیمار 120/80 و ضربان 75"


def test_implausible_pairs_are_left_alone():
    # 20/300 is not a blood pressure; do not invent a ratio.
    assert convert_numbers("بیست روی سیصد") == "20 روی 300"


def test_diastolic_above_systolic_is_left_alone():
    assert convert_numbers("هشتاد روی صد و بیست") == "80 روی 120"


def test_short_numbers_are_not_treated_as_bp():
    # Single digits cannot be a blood pressure reading.
    assert convert_numbers("پنج روی سه") == "5 روی 3"


def test_connector_outside_bp_ranges_is_kept():
    assert convert_numbers("پنج به ده رساند") == "5 به 10 رساند"


def test_bp_conversion_is_idempotent(processor: PostProcessor):
    once = processor.process("صد و بیست روی هشتاد")
    assert processor.process(once) == once
