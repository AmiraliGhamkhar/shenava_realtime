"""Typed measurement, medication, negation and laterality grammars."""
from __future__ import annotations
import re
from typing import Iterable

from .aho_corasick import AhoCorasickMatcher
from .lexicon import UNITS
from .spans import ClinicalSpan, MeasurementSpan, MedicationSpan, NumberSpan, TextSpan
from .text_normalize import match_key


class MeasurementGrammar:
    def __init__(self, enabled: bool = True) -> None:
        self.units = AhoCorasickMatcher()
        if enabled:
            self.units.add_many(UNITS)

    def parse(self, text: str, numbers: list[NumberSpan]) -> list[MeasurementSpan]:
        results: list[MeasurementSpan] = []
        unit_matches = sorted(self.units.find(text), key=lambda m: (m.start, -(m.end-m.start)))
        # value + explicit unit; whitespace only means no inferred structure.
        for number in numbers:
            for unit in unit_matches:
                if unit.start >= number.end and not text[number.end:unit.start].strip():
                    kind = self._kind(unit.pattern.output)
                    value = number.value
                    results.append(MeasurementSpan(text[number.start:unit.end], value,
                        unit.pattern.output, number.start, unit.end, kind))
                    break
        # Specialized, explicit BP connector. Plausibility is a safety policy,
        # not inference: implausible values remain unchanged.
        for left, right in zip(numbers, numbers[1:]):
            connector = text[left.end:right.start]
            if re.fullmatch(r"\s*(?:روی|بر\s+روی|خط)\s*", connector):
                if isinstance(left.value, (int, float)) and isinstance(right.value, (int, float)):
                    s, d = int(left.value), int(right.value)
                    if 60 <= s <= 260 and 30 <= d <= 160 and s > d:
                        results.append(MeasurementSpan(text[left.start:right.end], f"{s}/{d}",
                            "mmHg", left.start, right.end, "blood_pressure", True, s, d))
        # Prefer BP/longer spans over component measurements.
        results.sort(key=lambda x: (x.start, -(x.end-x.start), x.kind != "blood_pressure"))
        selected: list[MeasurementSpan] = []
        for item in results:
            if any(item.start < x.end and x.start < item.end for x in selected):
                continue
            selected.append(item)
        return selected

    @staticmethod
    def _kind(unit: str) -> str:
        if unit == "%": return "percentage"
        if unit in {"°", "°C", "°F"}: return "temperature"
        if unit in {"kg", "g"}: return "weight"
        if unit in {"cm", "m", "mm"}: return "length"
        if unit in {"mg", "mcg", "IU", "mL"}: return "dose"
        if unit in {"mg/dL", "mmol/L", "mEq/L"}: return "laboratory"
        return "measurement"


_MEDICATIONS = ("متفورمین", "آسپرین", "انسولین")  # explicit dictionary evidence
_ROUTES = {"خوراکی": "oral", "داخل وریدی": "IV", "وریدی": "IV",
           "عضلانی": "IM", "زیر جلدی": "SC", "IV": "IV", "IM": "IM", "PO": "PO"}
_FREQUENCIES = {"روزی دو بار": "twice_daily", "دو بار در روز": "twice_daily",
                "روزی یک بار": "once_daily", "یک بار در روز": "once_daily",
                "هر هشت ساعت": "every_8_hours", "هر دوازده ساعت": "every_12_hours"}


class MedicationGrammar:
    def parse(self, text: str, measurements: list[MeasurementSpan],
              numbers: list[NumberSpan] | None = None) -> list[MedicationSpan]:
        results = []; numbers = numbers or []
        for medication in _MEDICATIONS:
            for hit in re.finditer(rf"(?<!\w){re.escape(medication)}(?!\w)", text):
                # Relationship is accepted only with an adjacent dose expression.
                dose = next((m for m in measurements if m.kind == "dose" and m.start >= hit.end()
                             and re.fullmatch(r"\s*", text[hit.end():m.start])), None)
                route = self._near(text, hit.start(), hit.end(), _ROUTES)
                frequency = self._near(text, hit.start(), dose.end if dose else hit.end(), _FREQUENCIES)
                duration = None
                for number in numbers:
                    prefix = text[max(hit.end(), number.start-12):number.start]
                    suffix = text[number.end:number.end+8]
                    unit_hit = re.match(r"\s*(روز|هفته|ماه)(?:\W|$)", suffix)
                    if "به مدت" in prefix and unit_hit:
                        duration = f"{number.value}_{unit_hit.group(1)}"; break
                end = max(hit.end(), dose.end if dose else hit.end())
                results.append(MedicationSpan(medication, hit.start(), end,
                    dose.value if dose else None, dose.unit if dose else None, route, frequency, duration))
        return results

    @staticmethod
    def _near(text: str, start: int, end: int, mapping: dict[str, str]) -> str | None:
        window = text[max(0, start-32):min(len(text), end+48)]
        for phrase, canonical in sorted(mapping.items(), key=lambda x: -len(x[0])):
            if phrase in window: return canonical
        return None


_ANATOMY = ("زانو", "دست", "پا", "چشم", "گوش", "کلیه", "ریه")
_LATERALITY = {"راست": "right", "چپ": "left", "دو طرفه": "bilateral"}


class ClinicalContextGrammar:
    def parse(self, text: str, concepts: Iterable[tuple[str, int, int]]) -> list[ClinicalSpan]:
        results = []
        for concept, start, end in concepts:
            before, after = text[max(0, start-32):start], text[end:min(len(text), end+40)]
            assertion = "present"
            if re.search(r"(?:بدون)\s*$", before) or re.match(r"\s*(?:ندارد|وجود ندارد|مشاهده نشد|منفی است|نیست)", after):
                assertion = "negated"
            elif re.search(r"(?:احتمال|مشکوک به)\s*$", before): assertion = "possible"
            elif re.search(r"سابقه\s*$", before) or re.match(r"\s+سابقه دارد", after): assertion = "historical"
            results.append(ClinicalSpan(concept, start, end, assertion))
        return results

    def anatomy(self, text: str) -> list[ClinicalSpan]:
        results = []
        for anatomy in _ANATOMY:
            for hit in re.finditer(rf"(?<!\w){anatomy}(?!\w)", text):
                before, after = text[max(0, hit.start()-16):hit.start()], text[hit.end():hit.end()+16]
                lat = "bilateral" if re.search(r"هر دو\s*$", before) else "unspecified"
                for word, value in _LATERALITY.items():
                    if re.match(rf"\s+{word}(?:\W|$)", after): lat = value
                results.append(ClinicalSpan(anatomy, hit.start(), hit.end(), "present", "patient", lat))
        return results
