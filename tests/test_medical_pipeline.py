from shenava_realtime.aho_corasick import AhoCorasickMatcher
from shenava_realtime.medical_pipeline import MedicalNormalizationPipeline
from shenava_realtime.number_grammar import PersianNumberGrammar
from shenava_realtime.postprocessor import PostProcessor
from shenava_realtime.span_resolver import SpanResolver
from shenava_realtime.terminology import TerminologyRule, build_matcher


def test_aho_returns_nested_overlapping_offsets():
    matcher = AhoCorasickMatcher(); matcher.add("سی تی", "CT"); matcher.add("سی تی آنژیوگرافی", "CTA")
    matches = matcher.find("امروز سی تی آنژیوگرافی شد")
    assert [(m.text, m.start, m.end) for m in matches] == [
        ("سی تی", 6, 11), ("سی تی آنژیوگرافی", 6, 22)]


def test_resolver_is_leftmost_longest_then_priority():
    rules = [TerminologyRule("short", "CT", ("سی تی",), priority=100),
             TerminologyRule("long", "CTA", ("سی تی آنژیوگرافی",), priority=1)]
    selected, review = SpanResolver().resolve(build_matcher(rules).find("سی تی آنژیوگرافی"))
    assert not review and [x.pattern.output for x in selected] == ["CTA"]
    same = [TerminologyRule("low", "A", ("سي",), priority=1),
            TerminologyRule("high", "B", ("سی",), priority=9)]
    # normalized duplicate with conflicting output is rejected at construction.
    try: build_matcher(same)
    except ValueError: pass
    else: raise AssertionError("conflicting normalized patterns must fail")


def test_explicit_alias_and_phonetic_variant_candidates():
    rule = TerminologyRule("drug.explicit", "داروی استاندارد", ("نام اصلی",),
                           aliases=("نام مستعار",), phonetic_variants=("نام تلفظی",))
    matcher = build_matcher([rule])
    assert matcher.find("نام مستعار")[0].pattern.source == "alias"
    assert matcher.find("نام تلفظی")[0].pattern.source == "phonetic_variant"


def test_high_risk_and_contextual_rule_is_preserved_for_review():
    rule = TerminologyRule("unsafe", "X", ("عبارت مبهم",), risk="high")
    pipeline = MedicalNormalizationPipeline([rule])
    result = pipeline.process_normalized("عبارت مبهم")
    assert result.canonical_text == "عبارت مبهم"
    assert result.review_required and result.review_reasons == ["unsafe_terminology:unsafe"]


def test_number_semantic_spans_decimal_fraction_range_negative_and_boundaries():
    grammar = PersianNumberGrammar()
    assert [(x.value, x.type) for x in grammar.parse("دو ممیز پنج")] == [(2.5, "decimal")]
    assert [(x.value, x.type) for x in grammar.parse("سه چهارم")] == [(0.75, "fraction")]
    assert [(x.value, x.type) for x in grammar.parse("منفی پنج")] == [(-5, "cardinal")]
    assert [(x.value, x.type) for x in grammar.parse("پنج تا ده")] == [("5-10", "range")]
    assert [x.value for x in grammar.parse("بیست، سی")] == [20, 30]


def test_typed_medication_negation_laterality_and_protection():
    processor = PostProcessor()
    assert processor.process("متفورمین پانصد میلی گرم روزی دو بار") == "متفورمین 500 mg روزی 2 بار"
    result = processor.last_result
    drug = result.medications[0]
    assert (drug.medication, drug.dose, drug.unit, drug.frequency) == ("متفورمین", 500, "mg", "twice_daily")
    processor.process("متفورمین پانصد میلی گرم خوراکی به مدت پنج روز")
    drug = processor.last_result.medications[0]
    assert (drug.route, drug.duration) == ("oral", "5_روز")
    processor.process("بدون تب و درد زانو راست")
    result = processor.last_result
    assert next(x for x in result.clinical if x.concept == "تب").assertion == "negated"
    assert next(x for x in result.clinical if x.concept == "زانو").laterality == "right"
    assert processor.process("هر دو زانو") == "هر دو زانو"
    assert processor.last_result.clinical[0].laterality == "bilateral"
    processor.process("ساعت 12:30 و فشار 120/80")
    assert {x.kind for x in processor.last_result.protected_spans if hasattr(x, "kind")} >= {"time", "ratio"}


def test_unknown_and_mixed_language_are_idempotent():
    processor = PostProcessor(); raw = "unknown داروی ناشناخته و MRI"
    once = processor.process(raw)
    assert once == raw and processor.process(once) == once


def test_no_phrase_crosses_sentence_punctuation():
    matcher = AhoCorasickMatcher(); matcher.add("سی تی", "CT")
    assert matcher.find("سی، تی") == []
