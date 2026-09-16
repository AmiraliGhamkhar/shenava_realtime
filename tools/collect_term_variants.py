#!/usr/bin/env python3
"""Collect recurring ASR variants of a known medical term from a corpus.

Evidence-driven workflow (a human reviews the output before anything is
trusted):

    reference term -> observed ASR variants -> reviewed variant -> terminology rule

Input is a JSONL corpus (same schema as the medical regression corpus:
``raw`` = ASR output, ``reference`` = expected output).  For every row where
the term appears in the reference, the term's span is projected onto the raw
ASR text and the corresponding raw substring is aggregated as an observed
variant.  Nothing is written into terminology.json automatically; use
``--suggest`` to print a review fragment to copy after human review.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shenava_realtime.variants import collect_variants, suggest_rule_fragment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, nargs="?",
                        default=Path(__file__).resolve().parents[1] / "tests/corpus/medical_regression.jsonl")
    parser.add_argument("--term", required=True, help="canonical term as it appears in references (e.g. CABG)")
    parser.add_argument("--rule-id", default=None, help="id to use in the suggested rule fragment")
    parser.add_argument("--category", default="medical_term")
    parser.add_argument("--suggest", action="store_true",
                        help="print a review fragment for terminology.json (never written)")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = collect_variants(rows, args.term)
    print(json.dumps({
        "term": summary["term"],
        "rows_scanned": summary["rows_scanned"],
        "variants": [
            {"form": v.form, "count": v.count, "source_rows": list(v.source_rows)}
            for v in summary["variants"]
        ],
    }, ensure_ascii=False, indent=2))
    if args.suggest and summary["variants"]:
        rule_id = args.rule_id or f"term.review.{args.term.lower().replace(' ', '_')}"
        print(json.dumps(suggest_rule_fragment(args.term, summary, rule_id=rule_id,
                                               category=args.category), ensure_ascii=False, indent=2))
        print("\nReview each candidate form before adding it as a spoken_form/alias in terminology.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
