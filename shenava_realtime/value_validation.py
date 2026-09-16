"""Deterministic plausibility validation of structured clinical values.

Validation never rewrites: an obviously malformed value keeps its source text
and adds a structured review reason (``suspicious_value:<kind>=<value>``).
Ranges are explicit and conservative — they flag values that are *impossible*
or near-impossible, not merely unusual.  A genuinely abnormal but possible
reading (e.g. SpO2 84%) must stay untouched and still pass review normally.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .spans import MeasurementSpan, NumberSpan

# (low, high) inclusive plausibility ranges per measurement kind / unit.
# Body temperature bounds span the documented survivable range; values outside
# are recorded as impossible, not merely abnormal.  These are the single
# source of truth: the pipeline (`validate_measurements`), the persisted
# clinical records and the BP grammar gates all apply exactly these numbers,
# so a value can never be "suspicious" in one layer and clean in another.
PERCENTAGE_RANGE = (0.0, 100.0)
TEMPERATURE_RANGE = (24.0, 45.0)      # body temperature in ° / °C
TEMPERATURE_F_RANGE = (75.0, 113.0)   # °F only
PULSE_RANGE = (20.0, 300.0)           # bpm heart rate
BP_SYSTOLIC_RANGE = (60, 260)
BP_DIASTOLIC_RANGE = (30, 160)
_TEMPERATURE_UNITS = {"°", "°C"}
_PULSE_UNITS = {"bpm"}
# Units whose positive value is required evidence of a well-formed dose.
DOSE_UNITS = frozenset({"mg", "mcg", "IU", "mL", "U insulin"})
_BP_CONNECTOR = re.compile(r"^\s*(?:روی|بر\s+روی|خط)\s*$")


def bp_plausible(systolic, diastolic) -> bool:
    """Physiological gate shared by the BP grammars and the record checks."""
    return (BP_SYSTOLIC_RANGE[0] <= systolic <= BP_SYSTOLIC_RANGE[1]
            and BP_DIASTOLIC_RANGE[0] <= diastolic <= BP_DIASTOLIC_RANGE[1]
            and systolic > diastolic)


# Unit -> (review label, range). Mirrors the pipeline's kind-driven labels.
_UNIT_CHECKS = {
    "%": ("percentage", PERCENTAGE_RANGE),
    "°": ("temperature[°]", TEMPERATURE_RANGE),
    "°C": ("temperature[°C]", TEMPERATURE_RANGE),
    "°F": ("temperature[°F]", TEMPERATURE_F_RANGE),
    "bpm": ("pulse", PULSE_RANGE),
}


def plausibility_reason(unit: str, value: float) -> str | None:
    """Review reason for a unit/value pair, or ``None`` when plausible.

    Unit-driven (persisted records carry no span kinds); it applies the same
    ranges and the same ``suspicious_value:<kind>=<value>`` labels as the
    pipeline's kind-gated :func:`validate_measurements`.
    """
    if unit in DOSE_UNITS and value <= 0:
        return f"suspicious_value:dose={value:g}"
    check = _UNIT_CHECKS.get(unit)
    if check is None:
        return None
    label, (low, high) = check
    if low <= value <= high:
        return None
    return f"suspicious_value:{label}={value:g}"


@dataclass(frozen=True)
class ValueIssue:
    kind: str  # "percentage" | "temperature" | "pulse" | "dose" | "bp_pair"
    text: str  # the preserved source span
    reason: str  # "suspicious_value:..."
    start: int
    end: int


def _numeric(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def validate_measurements(measurements) -> list[ValueIssue]:
    """Check typed measurement spans against explicit plausibility ranges."""
    issues: list[ValueIssue] = []
    for m in measurements:
        if not isinstance(m, MeasurementSpan):
            continue
        value = _numeric(m.value)
        if value is None:
            continue  # "5-10" ranges and BP pairs are handled elsewhere
        kind = m.kind
        label: str | None = None
        if kind == "percentage":
            low, high = PERCENTAGE_RANGE
            label = "percentage"
        elif kind == "temperature":
            low, high = TEMPERATURE_F_RANGE if m.unit == "°F" else TEMPERATURE_RANGE
            label = f"temperature[{m.unit}]"
        elif m.unit in _PULSE_UNITS:
            low, high = PULSE_RANGE
            label = "pulse"
        elif kind == "dose":
            if value <= 0:
                issues.append(ValueIssue("dose", m.text, f"suspicious_value:dose={value:g}", m.start, m.end))
            continue
        else:
            continue
        if not low <= value <= high:
            issues.append(
                ValueIssue(kind, m.text, f"suspicious_value:{label}={value:g}", m.start, m.end)
            )
    return issues


def validate_bp_pairs(text: str, numbers: list[NumberSpan]) -> list[ValueIssue]:
    """Explicit BP-connector pairs that are *not* a plausible BP.

    The grammar already refuses to form such a ratio (the text is preserved);
    this only records why a human should look at it, e.g. ``80 روی 120``.
    """
    issues: list[ValueIssue] = []
    for left, right in zip(numbers, numbers[1:]):
        connector = text[left.end:right.start]
        if not _BP_CONNECTOR.fullmatch(connector):
            continue
        s, d = _numeric(left.value), _numeric(right.value)
        if s is None or d is None:
            continue
        if not bp_plausible(s, d):
            issues.append(
                ValueIssue(
                    "bp_pair",
                    text[left.start:right.end],
                    f"suspicious_value:bp_pair={int(s)}/{int(d)}",
                    left.start,
                    right.end,
                )
            )
    return issues
