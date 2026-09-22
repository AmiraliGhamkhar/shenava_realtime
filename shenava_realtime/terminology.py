"""Structured, data-driven terminology rules and matcher construction."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .aho_corasick import AhoCorasickMatcher

# Hard bound for the optional decoder-bias metadata below.  Kept small on
# purpose: a log-prob boost of 1.5 is already e^1.5 ~ 4.5x, enough to tip a
# near-tie without turning unrelated words into medical terms.
MAX_HOTWORD_BIAS = 1.5


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
    # Decoder-bias override (see hotwords.py): None = category default,
    # 0 = never boosted, >0 = an explicitly reviewed boost for a category
    # that is not boosted by default (e.g. a rare anatomy term).
    bias: Optional[float] = None


_DEFAULT_DATA = Path(__file__).with_name("data") / "terminology.json"


# Module-level parse cache (#17): PostProcessor (and therefore the engine)
# is instantiated once per process in production, but tests and tools build it
# repeatedly; re-reading and re-validating terminology.json each time is pure
# overhead. The file is immutable within a run.
_RULES_CACHE: Dict[str, Tuple[TerminologyRule, ...]] = {}


def load_rules(path: str | Path = _DEFAULT_DATA) -> list[TerminologyRule]:
    """Load and strictly validate the dependency-free JSON terminology schema."""
    resolved = Path(path)
    # Key on path + mtime + size so a file rewritten at the same path (tests
    # write terminology.json variants into tmp dirs) invalidates the cache.
    try:
        stat = resolved.stat()
        key = f"{resolved}|{stat.st_mtime_ns}|{stat.st_size}"
    except OSError:
        key = str(resolved)
    cached = _RULES_CACHE.get(key)
    if cached is not None:
        return list(cached)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
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
        if rule.bias is not None:
            if isinstance(rule.bias, bool) or not isinstance(rule.bias, (int, float)) \
                    or not 0.0 <= float(rule.bias) <= MAX_HOTWORD_BIAS:
                raise ValueError(
                    f"terminology rule {rule.id!r}: bias must be null or within "
                    f"[0, {MAX_HOTWORD_BIAS}]"
                )
        rules.append(rule)
    _RULES_CACHE[key] = tuple(rules)
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
