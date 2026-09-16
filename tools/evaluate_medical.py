#!/usr/bin/env python3
"""Reproducible text/corpus evaluation for ASR and deterministic post-processing.

JSONL rows contain ``raw``, ``reference`` and optional ``entities`` entries of
``{"text": ..., "type": term|medication|number|unit|dose|negation|laterality}``.
A row with ``"forced": true`` is evaluated through the pipeline's forced
endpoint path (a VAD/ASR segment-cap cut, not a natural endpoint) and its
reference must show the preserved spoken words.

Reports raw-ASR and post-processing WER/CER, per-category error rates, a
weighted medical-essential error rate (high-risk clinical tokens count more
than ordinary prose) and entity precision/recall/F1. This is an engineering
regression score: it does not measure recognition accuracy on real audio and
must not be presented as clinical validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shenava_realtime.postprocessor import PostProcessor

# Weight of a reference character that belongs to a reviewed entity span.
ESSENTIAL_WEIGHT = 3.0
ORDINARY_WEIGHT = 1.0


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


def weighted_distance(hypothesis: str, reference: str, weights: list[float]) -> float:
    """Levenshtein distance where each operation costs the weight of the
    reference character it touches (insertions borrow the preceding weight)."""
    a, b = list(hypothesis), list(reference)
    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + (weights[0] if m else ORDINARY_WEIGHT)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + weights[j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else weights[j - 1])
            delete = dp[i - 1][j] + weights[j - 1]
            insert = dp[i][j - 1] + (weights[j - 1] if j > 0 else ORDINARY_WEIGHT)
            dp[i][j] = min(sub, delete, insert)
    return dp[n][m]


def essential_error_rate(hypothesis: str, reference: str, entity_texts) -> float:
    """Weighted character error rate over the full reference string (spaces
    included) so preserved separators are weighted too."""
    weights = [ORDINARY_WEIGHT] * len(reference)
    for text in entity_texts:
        if not text:
            continue
        index = 0
        while True:
            index = reference.find(text, index)
            if index < 0:
                break
            for j in range(index, index + len(text)):
                weights[j] = ESSENTIAL_WEIGHT
            index += len(text)
    total = max(1.0, sum(weights))
    return weighted_distance(hypothesis, reference, weights) / total


def forced_output(raw: str) -> str:
    """Run one segment through the pipeline's forced-endpoint path."""
    from shenava_realtime.pipeline import TranscriptionPipeline
    from shenava_realtime.streaming import DecodeResult, StreamingDecoder

    class OneShot(StreamingDecoder):
        name = "oneshot"

        def __init__(self):
            self.text = raw

        def reset(self):
            pass

        def push(self, audio: "np.ndarray", force: bool = False):
            return DecodeResult(text=self.text, confidence=0.0, reset=False)

        def finalize(self):
            return DecodeResult(text=self.text, confidence=0.0, reset=False)

    pipeline = TranscriptionPipeline(OneShot(), holdback_words=2)
    pipeline.start_utterance()
    return "".join(pipeline.end_utterance(forced=True))


def entity_texts(row: dict, output: str, result) -> set[str]:
    """The entity strings expected to survive in the output."""
    expected = {(e["type"], e["text"]) for e in row.get("entities", [])}
    return {text for _, text in expected}


def _fmt_num(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def evaluate(path: Path) -> dict[str, float | int]:
    processor = PostProcessor(); rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
    raw_wer = raw_cer = post_wer = post_cer = essential = 0.0
    forced_errors = 0; forced_total = 0
    category_total = {x: 0 for x in ("term", "medication", "number", "unit", "dose",
                                     "negation", "laterality")}
    category_errors = dict(category_total); tp = fp = fn = 0
    for row in rows:
        raw, reference = row["raw"], row["reference"]
        forced = bool(row.get("forced"))
        processed = processor.process(raw)
        result = processor.last_result
        if forced:
            output = forced_output(raw)
            forced_total += 1
            if output != reference:
                forced_errors += 1
        else:
            output = processed
        raw_wer += error_rate(raw, reference); raw_cer += error_rate(raw, reference, chars=True)
        post_wer += error_rate(output, reference); post_cer += error_rate(output, reference, chars=True)
        essential += essential_error_rate(output, reference, entity_texts(row, output, None))
        expected = {(e["type"], e["text"]) for e in row.get("entities", [])}
        predicted = set()
        if result:
            predicted |= {("term", m.pattern.output) for m in result.terminology
                          if m.pattern.payload.category in {"abbreviation", "imaging", "procedure"}}
            predicted |= {("medication", m.medication) for m in result.medications}
            predicted |= {("number", _fmt_num(n.value)) for n in result.numbers}
            predicted |= {("unit", m.unit) for m in result.measurements}
            predicted |= {("dose", f"{_fmt_num(m.dose)} {m.unit}") for m in result.medications
                          if m.dose is not None and m.unit}
            predicted |= {("dose", f"{_fmt_num(m.value)} {m.unit}") for m in result.measurements
                          if m.kind == "dose"}
            predicted |= {("negation", c.concept) for c in result.clinical if c.assertion == "negated"}
            predicted |= {("laterality", c.laterality) for c in result.clinical if c.laterality != "unspecified"}
        tp += len(expected & predicted); fp += len(predicted - expected); fn += len(expected - predicted)
        for category, _value in expected:
            if category in category_total:
                category_total[category] += 1
                if (category, _value) not in predicted:
                    category_errors[category] += 1
    count = max(1, len(rows)); precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    names = {"term": "Medical Term Error Rate", "medication": "Medication Error Rate",
             "number": "Number Error Rate", "unit": "Unit Error Rate",
             "dose": "Dose Error Rate",
             "negation": "Negation Error Rate", "laterality": "Laterality Error Rate"}
    metrics = {"Samples": len(rows), "Raw ASR WER": raw_wer / count, "Raw ASR CER": raw_cer / count,
               "Post-processing WER": post_wer / count, "Post-processing CER": post_cer / count,
               "Medical Essential Error": essential / count}
    metrics.update({names[k]: category_errors[k] / max(1, category_total[k]) for k in names})
    metrics["Forced-boundary Error Rate"] = forced_errors / max(1, forced_total)
    metrics.update({"Entity Precision": precision, "Entity Recall": recall,
                    "Entity F1": 2*precision*recall/max(1e-12, precision+recall)})
    return metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("corpus", type=Path, nargs="?",
        default=Path(__file__).resolve().parents[1] / "tests/corpus/medical_regression.jsonl")
    args = parser.parse_args()
    metrics = evaluate(args.corpus)
    for key, value in metrics.items(): print(f"{key}: {value if isinstance(value, int) else f'{value:.4f}'}")
    print("(engineering regression only — not clinical validation)")

if __name__ == "__main__": main()
