"""Deterministic plausibility validation: preserve source text, never correct."""
from shenava_realtime.postprocessor import PostProcessor
from shenava_realtime.spans import MeasurementSpan, NumberSpan
from shenava_realtime.value_validation import (
    ValueIssue,
    bp_plausible,
    plausibility_reason,
    validate_bp_pairs,
    validate_measurements,
)


def m(text, value, unit, start, end, kind="measurement"):
    return MeasurementSpan(text, value, unit, start, end, kind)


def test_percentage_bounds():
    assert validate_measurements([m("95%", 95, "%", 0, 3, "percentage")]) == []
    issues = validate_measurements([m("102%", 102, "%", 0, 4, "percentage")])
    assert [i.reason for i in issues] == ["suspicious_value:percentage=102"]
    assert issues[0].text == "102%"  # source span preserved, not rewritten


def test_temperature_bounds_celsius_and_fahrenheit():
    assert validate_measurements([m("38.5°", 38.5, "°", 0, 5, "temperature")]) == []
    # Severe hypothermia is possible, not impossible: 24° stays unflagged,
    # 23° is below the documented survivable range and is flagged.
    assert validate_measurements([m("24°", 24, "°", 0, 3, "temperature")]) == []
    issues = validate_measurements([m("23°", 23, "°", 0, 3, "temperature")])
    assert [i.reason for i in issues] == ["suspicious_value:temperature[°]=23"]
    issues = validate_measurements([m("50°", 50, "°", 0, 3, "temperature")])
    assert [i.reason for i in issues] == ["suspicious_value:temperature[°]=50"]
    assert validate_measurements([m("98.6°F", 98.6, "°F", 0, 5, "temperature")]) == []
    issues = validate_measurements([m("60°F", 60, "°F", 0, 4, "temperature")])
    assert [i.reason for i in issues] == ["suspicious_value:temperature[°F]=60"]
    # A unit without a typed kind is not judged (no opinions on unknown kinds).
    assert validate_measurements([m("50°", 50, "°", 0, 3)]) == []


def test_pulse_bounds():
    assert validate_measurements([m("72 bpm", 72, "bpm", 0, 6)]) == []
    issues = validate_measurements([m("1000 bpm", 1000, "bpm", 0, 8)])
    assert [i.reason for i in issues] == ["suspicious_value:pulse=1000"]
    assert [i.kind for i in issues] == ["measurement"]


def test_nonpositive_dose_flagged():
    issues = validate_measurements([m("-5 mg", -5, "mg", 0, 5, "dose")])
    assert [i.reason for i in issues] == ["suspicious_value:dose=-5"]
    assert validate_measurements([m("0 mcg", 0, "mcg", 0, 4, "dose")]) != []


def test_unclassified_measurements_are_not_judged():
    # A plain measurement with an unusual unit is preserved without an opinion.
    assert validate_measurements([m("5 g", 5, "g", 0, 3, "weight")]) == []
    assert validate_measurements([m("3-5", "3-5", "mg", 0, 5)]) == []
    assert validate_measurements([]) == []
    assert validate_measurements(["not a span"]) == []


def test_bp_pair_connector_detection():
    text = "فشار 80 روی 120 ثبت شد"
    numbers = [NumberSpan("هشتاد", 80, 5, 8), NumberSpan("صد و بیست", 120, 12, 19)]
    issues = validate_bp_pairs(text, numbers)
    assert [i.reason for i in issues] == ["suspicious_value:bp_pair=80/120"]
    assert issues[0].kind == "bp_pair"


def test_bp_pair_plausible_values_pass():
    text = "فشار 120 بر روی 80 ثبت شد"
    numbers = [NumberSpan("صد و بیست", 120, 5, 12), NumberSpan("هشتاد", 80, 17, 21)]
    assert validate_bp_pairs(text, numbers) == []
    # "خط" is an accepted connector too.
    text = "خون 90 خط 60"
    numbers = [NumberSpan("نود", 90, 3, 5), NumberSpan("شصت", 60, 9, 11)]
    assert validate_bp_pairs(text, numbers) == []


def test_non_connector_pairs_ignored():
    text = "دوز 5 از 10"
    numbers = [NumberSpan("پنج", 5, 4, 5), NumberSpan("ده", 10, 7, 8)]
    assert validate_bp_pairs(text, numbers) == []


def test_end_to_end_suspicious_values_flagged_not_corrected():
    p = PostProcessor()
    out = p.process("اشباع اکسیژن صد و دو درصد و دمای بدن پنجاه درجه")
    assert "SpO2 102%" in out  # impossible value kept verbatim
    assert "50°" in out
    assert p.last_result.review_required
    reasons = p.last_result.review_reasons
    assert "suspicious_value:percentage=102" in reasons
    assert "suspicious_value:temperature[°]=50" in reasons
    issues = [i.reason for i in p.last_result.value_issues]
    assert "suspicious_value:percentage=102" in issues
    assert "suspicious_value:temperature[°]=50" in issues


def test_plausible_values_do_not_flag():
    p = PostProcessor()
    out = p.process("فشار خون صد و بیست روی هشتاد")
    assert out == "BP 120/80"
    assert p.last_result.value_issues == []
    assert "suspicious_value" not in " ".join(p.last_result.review_reasons)


def test_issue_shape():
    issue = ValueIssue("percentage", "999%", "suspicious_value:percentage=999", 0, 4)
    assert (issue.kind, issue.text, issue.start, issue.end) == ("percentage", "999%", 0, 4)
    assert isinstance(issue, ValueIssue)


# --------------------------------------------------------------------------- #
# One shared set of ranges: the unit-driven check used by persisted records
# must agree with the pipeline's typed-span check and with the BP grammar.
# --------------------------------------------------------------------------- #
def test_plausibility_reason_mirrors_the_pipeline_labels_and_ranges():
    # (unit, value, expected reason or None) — identical verdicts on both sides.
    cases = [
        ("%", 102.0, "suspicious_value:percentage=102"),
        ("%", 84.0, None),           # abnormal but possible: no opinion
        ("%", 0.0, None),
        ("%", -3.0, "suspicious_value:percentage=-3"),
        ("°", 23.0, "suspicious_value:temperature[°]=23"),
        ("°C", 24.5, None),          # 24–45: the documented survivable body range
        ("°C", 24.0, None),
        ("°F", 76.0, None),          # 75–113 for Fahrenheit
        ("°F", 114.0, "suspicious_value:temperature[°F]=114"),
        ("bpm", 1000.0, "suspicious_value:pulse=1000"),
        ("mg", 0.0, "suspicious_value:dose=0"),
        ("mcg", -5.0, "suspicious_value:dose=-5"),
        ("kg", -5.0, None),          # not a dose unit: no opinion (pipeline too)
        ("mmHg", 1.0, None),
        ("unknown", 999.0, None),
    ]
    for unit, value, expected in cases:
        assert plausibility_reason(unit, value) == expected, (unit, value)
    kinds = {"%": "percentage", "°": "temperature", "°C": "temperature",
             "°F": "temperature", "bpm": "rate", "mg": "dose", "mcg": "dose",
             "kg": "weight", "mmHg": "blood_pressure", "unknown": "measurement"}
    for unit, value, expected in cases:
        # The typed-span check (kind-gated) must reach the same verdicts as
        # the unit-driven check used by persisted clinical records.
        span = MeasurementSpan(f"{value:g}{unit}", value, unit, 0, 4, kinds[unit])
        reasons = [i.reason for i in validate_measurements([span])]
        assert reasons == ([expected] if expected else []), (unit, value, reasons)


def test_clinical_records_use_the_pipeline_ranges():
    from shenava_realtime.clinical import extract_record
    # 24.5 °C is plausible: neither the pipeline nor the record may flag it.
    record = extract_record("دمای بدن 24.5 °C")
    assert not any(r.startswith("suspicious_value") for r in record["review_reasons"])
    record = extract_record("دمای بدن 23 °C")
    assert "suspicious_value:temperature[°C]=23" in record["review_reasons"]
    # Malformed dose: same label as the pipeline, never a correction.
    record = extract_record("متفورمین 0 mg")
    assert "suspicious_value:dose=0" in record["review_reasons"]
    # Implausible persisted BP pair keeps the shared bp_pair reason.
    record = extract_record("BP 80/120 ثبت شد")
    assert "suspicious_value:bp_pair=80/120" in record["review_reasons"]
    assert record["text"] == "BP 80/120 ثبت شد"  # preserved, not rewritten
    # Merged with the caller's pipeline reasons without dropping either.
    record = extract_record("BP 80/120", review_reasons=["dose_value"])
    assert set(record["review_reasons"]) == {"dose_value", "suspicious_value:bp_pair=80/120"}


def test_bp_plausible_is_the_single_gate():
    assert bp_plausible(120, 80) and bp_plausible(260, 160)
    assert not bp_plausible(80, 120)   # reversed
    assert not bp_plausible(261, 120)
    assert not bp_plausible(120, 29)
