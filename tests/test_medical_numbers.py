"""Persian medical number/measurement coverage and plausibility safety.

Two invariants are asserted everywhere below:

1. a value is never silently "fixed" — the source span survives in the output;
2. an implausible value produces a deterministic ``suspicious_value:*`` review
   reason from the single shared plausibility implementation
   (:mod:`shenava_realtime.value_validation`).
"""
import pytest

from shenava_realtime.fa_numbers import (
    OPEN_CONNECTORS,
    leading_number_span,
    open_number_tail,
)
from shenava_realtime.postprocessor import PostProcessor
from shenava_realtime.value_validation import plausibility_reason


@pytest.fixture()
def processor():
    return PostProcessor()


def run(processor, text):
    output = processor.process(text)
    reasons = processor.last_result.review_reasons if processor.last_result else []
    return output, list(reasons)


# --------------------------------------------------------------------------- #
# Cardinals, decimals, fractions, negatives, ranges
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spoken,expected", [
    ("پنج", "5"),
    ("هفده", "17"),
    ("نود و دو", "92"),
    ("صد و بیست", "120"),
    ("سیصد و پنجاه", "350"),
    ("هزار", "1000"),
])
def test_persian_cardinals(processor, spoken, expected):
    assert run(processor, spoken)[0] == expected


@pytest.mark.parametrize("spoken,expected", [
    ("سه ممیز چهار", "3.4"),
    ("دو و نیم", "2.5"),
    ("سی و هفت و نیم", "37.5"),
])
def test_decimals_and_halves(processor, spoken, expected):
    assert run(processor, spoken)[0] == expected


def test_fractions_are_converted_without_inventing_precision(processor):
    output, _ = run(processor, "یک سوم")
    assert output.startswith("0.33")


def test_negative_numbers_keep_their_sign(processor):
    assert run(processor, "منفی دو")[0] == "-2"


def test_numeric_ranges_are_preserved_as_ranges(processor):
    output, _ = run(processor, "بین پنج تا ده میلی گرم")
    assert "5-10 mg" in output


# --------------------------------------------------------------------------- #
# Clinical measurement forms
# --------------------------------------------------------------------------- #
def test_blood_pressure_pair(processor):
    assert run(processor, "فشار خون صد و بیست روی هشتاد")[0] == "BP 120/80"


def test_dose_with_unit(processor):
    output, reasons = run(processor, "متفورمین پانصد میلی گرم")
    assert "500 mg" in output
    assert "dose_value" in reasons


def test_mg_per_kg_dose(processor):
    output, _ = run(processor, "پنج میلی گرم بر کیلوگرم")
    assert "5 mg" in output and "kg" in output


def test_rate_per_minute(processor):
    assert run(processor, "ضربان قلب نود و دو در دقیقه")[0] == "ضربان قلب 92 bpm"


def test_oxygen_saturation(processor):
    assert run(processor, "اشباع اکسیژن نود و هشت درصد")[0] == "SpO2 98%"


def test_temperature(processor):
    assert run(processor, "دمای سی و هفت و نیم درجه")[0] == "دمای 37.5°"


def test_mixed_persian_and_latin_numeric_forms(processor):
    # Latin digits already present in the decoder output must survive intact.
    assert "120/80" in run(processor, "BP 120/80")[0]
    output, _ = run(processor, "دوز 5 mg و ده میلی گرم")
    assert "5 mg" in output and "10 mg" in output


# --------------------------------------------------------------------------- #
# Plausibility: flagged, never corrected
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spoken,fragment,label", [
    ("منفی دو درجه", "-2°", "temperature"),
    ("دمای شصت درجه", "60°", "temperature"),
])
def test_implausible_values_are_flagged_and_preserved(processor, spoken, fragment, label):
    output, reasons = run(processor, spoken)
    assert fragment in output                     # source value preserved
    assert any(r.startswith(f"suspicious_value:{label}") for r in reasons), reasons


def test_abnormal_but_possible_values_are_not_flagged(processor):
    output, reasons = run(processor, "اشباع اکسیژن هشتاد و چهار درصد")
    assert "84%" in output
    assert not any(r.startswith("suspicious_value") for r in reasons), reasons


def test_one_shared_plausibility_implementation_is_used():
    """The pipeline label and the record-level check must agree exactly."""
    assert plausibility_reason("%", 150.0) == "suspicious_value:percentage=150"
    assert plausibility_reason("%", 98.0) is None
    assert plausibility_reason("°", -2.0) == "suspicious_value:temperature[°]=-2"
    assert plausibility_reason("bpm", 500.0) == "suspicious_value:pulse=500"
    assert plausibility_reason("mg", 0.0) == "suspicious_value:dose=0"
    assert plausibility_reason("mg", 500.0) is None


def test_no_value_is_rewritten_to_a_plausible_one(processor):
    """Regression guard: a suspicious value keeps its digits, unchanged."""
    output, reasons = run(processor, "اشباع اکسیژن صد و پنجاه درصد")
    assert "150" in output and "100" not in output
    assert any("suspicious_value" in r for r in reasons)


# --------------------------------------------------------------------------- #
# Forced-boundary open-number-tail protection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,tail", [
    ("دوز سی و", "سی و"),                       # dangling conjunction
    ("دمای سی و هفت ممیز", "سی و هفت ممیز"),     # dangling decimal marker
    ("فشار خون صد و بیست روی", "صد و بیست روی"), # BP ratio awaiting diastolic
    ("بین پنج تا", "پنج تا"),                    # range awaiting its upper bound
])
def test_open_number_tails_are_detected(text, tail):
    assert open_number_tail(text) == tail


@pytest.mark.parametrize("text", [
    "دوز پنج میلی گرم",     # complete
    "بیمار آمد",            # no numbers at all
    "بیمار تا فردا",        # connector without a preceding value
    "سی",                   # a single value word is complete
])
def test_complete_or_non_numeric_phrases_have_no_open_tail(text):
    assert open_number_tail(text) == ""


def test_leading_number_span_marks_a_possible_continuation():
    assert leading_number_span("هشتاد و دو") == "هشتاد و دو"
    assert leading_number_span("پنج میلی گرم") == "پنج"
    assert leading_number_span("بیمار آمد") == ""


def test_a_trailing_connector_is_not_part_of_the_leading_run():
    # It belongs to the next continuation; keeping it would reopen a closed phrase.
    assert leading_number_span("ده میلی گرم روی") == "ده"


def test_open_connectors_are_an_explicit_reviewed_list():
    assert set(OPEN_CONNECTORS) == {"روی", "خط", "تا", "بر"}
