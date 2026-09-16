"""Production-oriented deterministic medical normalization orchestration."""
from __future__ import annotations
from dataclasses import dataclass, field
import re

from .medical_grammar import ClinicalContextGrammar, MeasurementGrammar, MedicationGrammar
from .number_grammar import PersianNumberGrammar
from .span_resolver import SpanResolver
from .spans import ClinicalSpan, MedicationSpan, MeasurementSpan, NumberSpan, Replacement, TextSpan
from .terminology import TerminologyRule, build_matcher, default_rules
from .text_normalize import digits_to_persian
from .value_validation import ValueIssue, validate_bp_pairs, validate_measurements


@dataclass
class ProcessingResult:
    raw_text: str
    normalized_text: str
    canonical_text: str
    terminology: list = field(default_factory=list)
    numbers: list[NumberSpan] = field(default_factory=list)
    measurements: list[MeasurementSpan] = field(default_factory=list)
    medications: list[MedicationSpan] = field(default_factory=list)
    clinical: list[ClinicalSpan] = field(default_factory=list)
    protected_spans: list = field(default_factory=list)
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    value_issues: list[ValueIssue] = field(default_factory=list)


class ProtectedSpanDetector:
    _patterns = (
        ("date_time", re.compile(r"(?<!\w)\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,4})?(?!\w)")),
        ("time", re.compile(r"(?<!\w)\d{1,2}:\d{2}(?!\w)")),
        ("ratio", re.compile(r"(?<!\w)\d+(?:\.\d+)?/\d+(?:\.\d+)?(?!\w)")),
        ("abbreviation", re.compile(r"(?<!\w)(?:CABG|PCI|PTCA|EKG|ECG|CT|MRI|CBC|BP|SpO2|HbA1c|INR|CRP|BNP|IV|IM|PO|SC)(?!\w)")),
    )
    def detect(self, text: str) -> list[TextSpan]:
        spans = []
        for kind, pattern in self._patterns:
            spans.extend(TextSpan(m.start(), m.end(), m.group(), kind, True) for m in pattern.finditer(text))
        return sorted(spans, key=lambda x: (x.start, -(x.end-x.start)))


class MedicalNormalizationPipeline:
    def __init__(self, rules: list[TerminologyRule] | None = None, *, digits="ascii") -> None:
        self.rules = default_rules() if rules is None else rules
        self.matcher = build_matcher(self.rules)
        self.resolver = SpanResolver(); self.numbers = PersianNumberGrammar()
        units_enabled = any(rule.category == "unit" for rule in self.rules)
        self.measurements = MeasurementGrammar(units_enabled); self.medications = MedicationGrammar()
        self.context = ClinicalContextGrammar(); self.protector = ProtectedSpanDetector()
        self.digits = digits

    def process_normalized(self, text: str, *, convert_numbers=True) -> ProcessingResult:
        initial_protected = self.protector.detect(text)
        candidates = self.matcher.find(text)
        # A candidate whose output is exactly the matched text (a self-alias
        # such as the Latin "EKG" for "EKG") is a no-op rewrite, not a
        # conflict: it must not be flagged as unsafe just because the span is
        # protected.  Case normalisation (cabg -> CABG) is NOT a no-op.
        candidates = [c for c in candidates if c.pattern.output != c.text]
        terms, review = self.resolver.resolve(candidates, protected=initial_protected)
        unit_tokens = {r.canonical.lower() for r in self.rules if r.category == "unit"}
        unit_tokens.update(form.split()[0].replace("\u200c", "") for r in self.rules
                           if r.category == "unit" for form in r.spoken_forms if form)
        numbers = self.numbers.parse(text, unit_tokens=unit_tokens) if convert_numbers else []
        measurements = self.measurements.parse(text, numbers) if convert_numbers else []
        medications = self.medications.parse(text, measurements, numbers)

        # Clinical concepts include explicit dictionary entities and conservative anatomy.
        concept_items = [(m.pattern.output, m.start, m.end) for m in terms
                         if getattr(m.pattern.payload, "negation_sensitive", False)]
        for concept in ("تب", "درد", "تنگی نفس"):
            concept_items.extend((concept, m.start(), m.end()) for m in re.finditer(rf"(?<!\w){concept}(?!\w)", text))
        concept_items = list(dict.fromkeys(concept_items))
        clinical = self.context.parse(text, concept_items) + self.context.anatomy(text)

        replacements: list[Replacement] = [Replacement(m.start, m.end, m.pattern.output,
            "terminology", 300 + m.pattern.priority) for m in terms]
        occupied_measurements = {(m.start, m.end) for m in measurements}
        for measurement in measurements:
            value = self._format(measurement.value)
            rendered = value
            if measurement.kind == "blood_pressure": rendered = value
            elif measurement.unit in {"%", "°", "°C", "°F"}: rendered += measurement.unit
            else: rendered += " " + measurement.unit
            replacements.append(Replacement(measurement.start, measurement.end, rendered, "measurement", 500))
        for number in numbers:
            if any(number.start >= m.start and number.end <= m.end for m in measurements): continue
            replacements.append(Replacement(number.start, number.end, self._format(number.value), "number", 150))

        # One final conflict pass: typed semantic spans outrank terminology.
        replacements.sort(key=lambda x: (x.start, -x.priority, -(x.end-x.start)))
        selected: list[Replacement] = []
        for item in replacements:
            conflicts = [x for x in selected if item.start < x.end and x.start < item.end]
            if not conflicts:
                selected.append(item)
            elif item.priority > max(x.priority for x in conflicts):
                selected = [x for x in selected if x not in conflicts] + [item]
        selected.sort(key=lambda x: x.start)
        canonical = self.render(text, selected)
        semantic_terms = [TextSpan(m.start, m.end, m.text,
            "unit" if m.pattern.payload.category == "unit" else "negation_sensitive",
            True, {"rule_id": m.pattern.payload.id}) for m in terms
            if m.pattern.payload.category == "unit" or m.pattern.payload.negation_sensitive]
        protected = [*initial_protected, *semantic_terms, *numbers, *measurements, *medications]

        # Structured review signals. Flags only: they never alter the text.
        reasons: set[str] = set()
        for match in terms:
            payload = match.pattern.payload
            if payload.risk in ("medium", "high") or payload.priority >= 90:
                reasons.add("rare_medical_term")
            if payload.negation_sensitive:
                reasons.add("negation_sensitive")
            if payload.laterality_sensitive:
                reasons.add("laterality_sensitive")
        if medications:
            reasons.add("drug_name")
        if any(med.dose is not None for med in medications) or any(
            m.kind == "dose" for m in measurements
        ):
            reasons.add("dose_value")
        # Any spoken number that was converted to a digit is a review flag:
        # a human should confirm the value, not the formatting.
        if numbers:
            reasons.add("numeric_value")
        if any(c.assertion == "negated" for c in clinical):
            reasons.add("negation_sensitive")
        if any(c.laterality != "unspecified" for c in clinical):
            reasons.add("laterality_sensitive")
        issues = (
            validate_measurements(measurements) + validate_bp_pairs(text, numbers)
            if convert_numbers
            else []
        )
        reasons.update(issue.reason for issue in issues)
        reasons.update(f"unsafe_terminology:{m.pattern.payload.id}" for m in review)

        return ProcessingResult(text, text, canonical, terms, numbers, measurements,
            medications, clinical, protected, bool(reasons), sorted(reasons), issues)

    def _format(self, value) -> str:
        if isinstance(value, str): result = value
        elif isinstance(value, float) and not value.is_integer(): result = f"{value:.10g}"
        else: result = str(int(value))
        return result if self.digits == "ascii" else digits_to_persian(result)

    @staticmethod
    def render(text: str, replacements: list[Replacement]) -> str:
        for item in sorted(replacements, key=lambda x: x.start, reverse=True):
            text = text[:item.start] + item.value + text[item.end:]
        return text
