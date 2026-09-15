"""Shared immutable span models for deterministic medical normalization.

Offsets are half-open character offsets into ``normalized_text``.  Stages never
mutate spans; rendering is a final, right-to-left operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    text: str
    kind: str
    protected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def overlaps(self, other: "TextSpan") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class NumberSpan:
    spoken: str
    value: int | float
    start: int
    end: int
    type: str = "cardinal"
    protected: bool = True


@dataclass(frozen=True)
class MeasurementSpan:
    text: str
    value: int | float | str
    unit: str
    start: int
    end: int
    kind: str = "measurement"
    protected: bool = True
    systolic: Optional[int] = None
    diastolic: Optional[int] = None


@dataclass(frozen=True)
class ClinicalSpan:
    concept: str
    start: int
    end: int
    assertion: Literal["present", "negated", "possible", "historical", "unknown"] = "unknown"
    experiencer: Literal["patient", "other", "unknown"] = "patient"
    laterality: Literal["right", "left", "bilateral", "unspecified"] = "unspecified"


@dataclass(frozen=True)
class MedicationSpan:
    medication: str
    start: int
    end: int
    dose: int | float | None = None
    unit: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    protected: bool = True


@dataclass(frozen=True)
class Replacement:
    start: int
    end: int
    value: str
    kind: str
    priority: int = 0
