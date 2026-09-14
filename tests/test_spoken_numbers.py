"""Spoken Persian numbers -> digits."""

import pytest

from shenava_realtime.fa_numbers import convert_numbers, format_number, parse_number_phrase


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("پنج", "5"),
        ("ده", "10"),
        ("یازده", "11"),
        ("بیست", "20"),
        ("سی و پنج", "35"),
        ("صد و بیست", "120"),
        ("دویست و پنجاه", "250"),
        ("هزار", "1000"),
        ("دو هزار", "2000"),
        ("دو هزار و سیصد", "2300"),
        ("دو هزار و سیصد و چهل و پنج", "2345"),
        ("یک میلیون", "1000000"),
        ("یک و نیم", "1.5"),
        ("سه و نیم", "3.5"),
        ("نود و نه", "99"),
    ],
)
def test_spoken_numbers(spoken: str, expected: str):
    assert convert_numbers(spoken) == expected


def test_numbers_inside_sentences():
    assert convert_numbers("بیمار پنج روز بستری بود") == "بیمار 5 روز بستری بود"
    assert convert_numbers("دوز دارو دو و نیم میلی گرم است") == "دوز دارو 2.5 میلی گرم است"


def test_already_written_digits_pass_through():
    assert convert_numbers("فشار 120 است") == "فشار 120 است"


def test_lone_yek_is_treated_as_an_article():
    # "یک بیمار" means "a patient", not "1 patient".
    assert convert_numbers("یک بیمار مراجعه کرد") == "یک بیمار مراجعه کرد"
    # ...but it is a quantity in front of a unit.
    assert convert_numbers("یک mg", unit_tokens={"mg"}) == "1 mg"


def test_yek_inside_a_phrase_is_a_number():
    assert convert_numbers("یک هزار") == "1000"


def test_conjunction_is_not_swallowed():
    assert convert_numbers("درد و تب دارد") == "درد و تب دارد"


def test_parse_returns_span():
    words = "صد و بیست روی هشتاد".split()
    value, end = parse_number_phrase(words, 0)
    assert value == 120
    assert end == 3  # consumed "صد و بیست"
    assert parse_number_phrase(words, end) is None or parse_number_phrase(words, end)[0] == 80


def test_parse_refuses_non_numbers():
    assert parse_number_phrase("بیمار آمد".split(), 0) is None
    assert parse_number_phrase([], 0) is None


def test_format_number_styles():
    assert format_number(120.0) == "120"
    assert format_number(2.5) == "2.5"
    assert format_number(120.0, digits="persian") == "۱۲۰"
    assert format_number(2.5, digits="persian") == "۲.۵"


def test_conversion_is_idempotent():
    once = convert_numbers("سی و پنج درصد از بیماران")
    assert convert_numbers(once) == once
