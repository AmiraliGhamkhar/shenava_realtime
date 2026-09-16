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
# are recorded as impossible, not merely abnormal.
_RANGE = {
    "percentage": (0.0, 100.0),
    "temperature": (24.0, 45.0),   # body temperature in ° / °C
    "temperature_f": (75.0, 113.0),  # °F only
    "pulse": (20.0, 300.0),        # bpm heart rate
}
_TEMPERATURE_UNITS = {"°", "°C"}
_PULSE_UNITS = {"bpm"}
_BP_CONNECTOR = re.compile(r"^\s*(?:روی|بر\s+روی|خط)\s*$")
_BP_SYSTOLIC = (60, 260)
_BP_DIASTOLIC = (30, 160)


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
            low, high = _RANGE["percentage"]
            label = "percentage"
        elif kind == "temperature":
            low, high = _RANGE["temperature_f"] if m.unit == "°F" else _RANGE["temperature"]
            label = f"temperature[{m.unit}]"
        elif m.unit in _PULSE_UNITS:
            low, high = _RANGE["pulse"]
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
        plausible = (
            _BP_SYSTOLIC[0] <= s <= _BP_SYSTOLIC[1]
            and _BP_DIASTOLIC[0] <= d <= _BP_DIASTOLIC[1]
            and s > d
        )
        if not plausible:
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
