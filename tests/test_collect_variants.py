"""Evidence-driven variant collection: reference -> observed -> reviewed -> rule."""
from shenava_realtime.variants import (
    align_tokens,
    collect_variants,
    find_term_spans,
    project_span,
    suggest_rule_fragment,
)


def test_align_tokens_with_inserted_and_missing_tokens():
    raw = "ک ا ب ج برای بیمار"
    reference = "CABG برای بیمار"
    pairs = align_tokens(raw, reference)
    raw_tokens = raw.split()
    ref_tokens = reference.split()
    # Every pair references existing token indices, and every reference
    # token is aligned to exactly one raw token (no duplicate slots).
    for r, f in pairs:
        if r is not None:
            assert 0 <= r < len(raw_tokens)
        if f is not None:
            assert 0 <= f < len(ref_tokens)
    aligned_refs = [f for _, f in pairs if f is not None]
    assert len(aligned_refs) == len(set(aligned_refs))
    # Two of the four spoken letters are deletions; "برای"/"بیمار" match 1:1.
    assert (4, 1) in pairs and (5, 2) in pairs
    assert sum(1 for r, _ in pairs if r is not None and _ is None) >= 2
    # Degenerate inputs are safe.
    assert align_tokens("", "x") == [(None, 0)]
    assert align_tokens("x", "") == [(0, None)]
    assert align_tokens("", "") == []


def test_project_span_stays_inside_token_boundaries():
    raw = "بیمار زیر کابج است"
    reference = "بیمار زیر CABG است"
    spans = find_term_spans(reference, "CABG")
    assert spans == [(10, 14)]
    assert project_span(raw, reference, *spans[0]) == "کابج"
    # Projection never crosses the term's token boundary into neighbours.
    assert project_span(raw, reference, 6, 9) == "زیر"


def test_project_span_handles_code_switching_words():
    raw = "بیمار EKG مثبت بود"
    reference = "بیمار EKG مثبت بود"
    spans = find_term_spans(reference, "EKG")
    assert project_span(raw, reference, *spans[0]) == "EKG"
    # A verbatim term is *not* reported as a variant.
    summary = collect_variants(
        [{"raw": raw, "reference": reference}], term="EKG")
    assert summary["variants"] == []


def test_project_span_with_missing_token_falls_back_to_overlapping_tokens():
    # The reference term is one token the raw text split into two words.
    raw = "بیمار اکو کاردیوگرافی طبیعی"
    reference = "بیمار اکوکاردیوگرافی طبیعی"
    spans = find_term_spans(reference, "اکوکاردیوگرافی")
    assert project_span(raw, reference, *spans[0]) == "اکو کاردیوگرافی"


def test_project_span_never_returns_out_of_bounds_text():
    # Empty / degenerate spans are rejected, not mis-projected.
    assert project_span("abc", "abc", 0, 0) == ""
    assert project_span("abc", "abc", 2, 1) == ""


def test_collect_variants_aggregates_and_orders_by_count():
    rows = [
        {"raw": "کابج انجام شد", "reference": "CABG انجام شد"},
        {"raw": "سی ای بی جی انجام شد", "reference": "CABG انجام شد"},
        {"raw": "کابج ثبت شد", "reference": "CABG ثبت شد"},
        {"raw": "CABG ثبت شد", "reference": "CABG ثبت شد"},   # verbatim: ignored
        {"raw": "متن بی ارتباط", "reference": "متن بی ارتباط"},  # no term: ignored
    ]
    summary = collect_variants(rows, "CABG")
    assert summary["rows_scanned"] == 5
    assert [v.form for v in summary["variants"]] == ["کابج", "سی ای بی جی"]
    assert [v.count for v in summary["variants"]] == [2, 1]
    assert summary["variants"][0].source_rows == (1, 3)


def test_suggest_rule_fragment_is_review_only():
    rows = [{"raw": "کابج انجام شد", "reference": "CABG انجام شد"}]
    summary = collect_variants(rows, "CABG")
    fragment = suggest_rule_fragment("CABG", summary, rule_id="term.review.cabg",
                                     category="procedure")
    assert fragment["canonical"] == "CABG"
    assert fragment["review_candidates"] == ["کابج"]
    assert fragment["note"]
    # The fragment is a plain dict: nothing here can write terminology.json.
    assert isinstance(fragment, dict)
