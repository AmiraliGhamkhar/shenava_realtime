"""Structured, data-driven terminology rules and matcher construction."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .aho_corasick import AhoCorasickMatcher


@dataclass(frozen=True)
class TerminologyRule:
    id: str
    canonical: str
    spoken_forms: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    phonetic_variants: tuple[str, ...] = ()
    category: str = "medical_term"
    specialty: str = "general"
    priority: int = 50
    risk: str = "low"
    requires_context: bool = False
    negation_sensitive: bool = False
    laterality_sensitive: bool = False
    number_sensitive: bool = False
    enabled: bool = True


_DEFAULT_DATA = Path(__file__).with_name("data") / "terminology.json"


def load_rules(path: str | Path = _DEFAULT_DATA) -> list[TerminologyRule]:
    """Load and strictly validate the dependency-free JSON terminology schema."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("rules"), list):
        raise ValueError("unsupported terminology schema")
    rules = []
    for raw in payload["rules"]:
        item = dict(raw)
        for field_name in ("spoken_forms", "aliases", "phonetic_variants"):
            item[field_name] = tuple(item.get(field_name, ()))
        rule = TerminologyRule(**item)
        if rule.risk not in {"low", "medium", "high"} or not rule.id or not rule.canonical:
            raise ValueError(f"invalid terminology rule: {rule.id!r}")
        rules.append(rule)
    return rules


def default_rules(*, medical_terms: bool = True, units: bool = True) -> list[TerminologyRule]:
    """Return enabled reviewed records, filtered by legacy feature switches."""
    return [rule for rule in load_rules() if rule.enabled and
            ((rule.category == "unit" and units) or (rule.category != "unit" and medical_terms))]


def build_matcher(rules: Iterable[TerminologyRule]) -> AhoCorasickMatcher:
    matcher = AhoCorasickMatcher()
    for rule in rules:
        variants = ((form, "exact") for form in rule.spoken_forms)
        variants = list(variants) + [(x, "alias") for x in rule.aliases] + [
            (x, "phonetic_variant") for x in rule.phonetic_variants]
        for phrase, source in variants:
            # Punctuation-only canonicals (%, °) are rendered outputs, not token patterns.
            if not any(ch.isalnum() or "\u0600" <= ch <= "\u06ff" for ch in phrase):
                continue
            matcher.add(phrase, rule.canonical, payload=rule,
                        priority=rule.priority, source=source)
    return matcher


class ContextResolver:
    """Disabled extension point for bounded contextual/LLM candidate resolution."""
    enabled = False
    def resolve(self, *_args, **_kwargs):  # pragma: no cover - extension contract
        raise RuntimeError("contextual resolution is disabled by default")
