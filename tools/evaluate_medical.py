#!/usr/bin/env python3
"""Reproducible text/corpus evaluation for ASR and deterministic post-processing.

JSONL rows contain ``raw``, ``reference`` and optional ``entities`` entries of
``{"text": ..., "type": term|medication|number|unit|negation|laterality}``.
This evaluates text output only; it does not make clinical-validity claims.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shenava_realtime.postprocessor import PostProcessor


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


def error_rate(hypothesis: str, reference: str, *, chars=False) -> float:
    left = list(hypothesis.replace(" ", "")) if chars else hypothesis.split()
    right = list(reference.replace(" ", "")) if chars else reference.split()
    return distance(left, right) / max(1, len(right))


def evaluate(path: Path) -> dict[str, float | int]:
    processor = PostProcessor(); rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
    raw_wer = raw_cer = post_wer = post_cer = 0.0
    category_total = {x: 0 for x in ("term", "medication", "number", "unit", "negation", "laterality")}
    category_errors = dict(category_total); tp = fp = fn = 0
    for row in rows:
        raw, reference = row["raw"], row["reference"]
        output = processor.process(raw); result = processor.last_result
        raw_wer += error_rate(raw, reference); raw_cer += error_rate(raw, reference, chars=True)
        post_wer += error_rate(output, reference); post_cer += error_rate(output, reference, chars=True)
        expected = {(e["type"], e["text"]) for e in row.get("entities", [])}
        predicted = set()
        if result:
            predicted |= {("term", m.pattern.output) for m in result.terminology
                          if m.pattern.payload.category in {"abbreviation", "imaging", "procedure"}}
            predicted |= {("medication", m.medication) for m in result.medications}
            predicted |= {("number", str(n.value)) for n in result.numbers}
            predicted |= {("unit", m.unit) for m in result.measurements}
            predicted |= {("negation", c.concept) for c in result.clinical if c.assertion == "negated"}
            predicted |= {("laterality", c.laterality) for c in result.clinical if c.laterality != "unspecified"}
        tp += len(expected & predicted); fp += len(predicted - expected); fn += len(expected - predicted)
        for category, value in expected:
            category_total[category] += 1
            if (category, value) not in predicted: category_errors[category] += 1
    count = max(1, len(rows)); precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    names = {"term": "Medical Term Error Rate", "medication": "Medication Error Rate",
             "number": "Number Error Rate", "unit": "Unit Error Rate",
             "negation": "Negation Error Rate", "laterality": "Laterality Error Rate"}
    metrics = {"Samples": len(rows), "Raw ASR WER": raw_wer/count, "Raw ASR CER": raw_cer/count,
               "Post-processing WER": post_wer/count, "Post-processing CER": post_cer/count}
    metrics.update({names[k]: category_errors[k] / max(1, category_total[k]) for k in names})
    metrics.update({"Entity Precision": precision, "Entity Recall": recall,
                    "Entity F1": 2*precision*recall/max(1e-12, precision+recall)})
    return metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("corpus", type=Path, nargs="?",
        default=Path(__file__).resolve().parents[1] / "tests/corpus/medical_regression.jsonl")
    args = parser.parse_args()
    metrics = evaluate(args.corpus)
    for key, value in metrics.items(): print(f"{key}: {value if isinstance(value, int) else f'{value:.4f}'}")

if __name__ == "__main__": main()
