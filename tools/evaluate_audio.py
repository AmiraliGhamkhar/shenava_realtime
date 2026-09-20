#!/usr/bin/env python3
"""Reproducible **real-audio** evaluation for the Shenava realtime stack.

This is the tool that measures recognition accuracy.  ``evaluate_medical.py``
next to it only scores the deterministic text pipeline on a text corpus and is
*not* evidence of ASR accuracy.

Corpus format — JSONL or CSV metadata plus WAV files::

    {"audio": "clips/0001.wav", "reference": "فشار خون صد و بیست روی هشتاد"}
    {"audio": "clips/0002.wav", "reference": "...", "forced": false}

``audio`` is resolved relative to the metadata file.  WAVs must be 16 kHz mono
(16-bit PCM or 32-bit float); anything else is reported as an error row rather
than silently resampled.

Reported metrics
----------------
* WER, CER, insertion/deletion/substitution counts
* medical-term, drug-name, dose/number, BP and abbreviation error rates
* forced-boundary error rate
* end-to-end latency per utterance and real-time factor

Error attribution
-----------------
Every run separates the error sources instead of reporting one number:

``acoustic``       reference vs. raw decoder output (normalisation off)
``normalization``  raw decoder output vs. its deterministic normalisation
``terminology``    reviewed-term errors surviving normalisation
``numbers``        number/measurement errors surviving normalisation
``endpointing``    forced-boundary rows whose output does not match

``--system`` records only a label for the report; it never changes the maths.
The only supported system is the sherpa-onnx streaming CTC backend
(``sherpa-onnx-ctc``); the historical NeMo CTC/RNNT labels (``v1.0-ctc``,
``v1.5-ctc``, ``v1.5-rnnt``) are gone along with the NeMo backend.

Usage::

    python tools/evaluate_audio.py corpus.jsonl --report out.json
    python tools/evaluate_audio.py corpus.jsonl \\
        --model models/shenava/model.int8.onnx --tokens models/shenava/tokens.txt
    python tools/evaluate_audio.py --compare a.json b.json

The evaluator is deliberately separate from runtime code: it imports the
pipeline but is never imported by it.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from shenava_realtime.postprocessor import PostProcessor  # noqa: E402
from shenava_realtime.terminology import default_rules  # noqa: E402

SYSTEMS = ("sherpa-onnx-ctc",)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[./]\d+)?")
_BP_RE = re.compile(r"\d{2,3}\s*/\s*\d{2,3}")
_LATIN_ABBREV_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{1,7}\b")


# --------------------------------------------------------------------------- #
# Edit distance with operation counts
# --------------------------------------------------------------------------- #
@dataclass
class EditCounts:
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    reference_length: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def rate(self) -> float:
        return self.errors / max(1, self.reference_length)

    def __iadd__(self, other: "EditCounts") -> "EditCounts":
        self.substitutions += other.substitutions
        self.insertions += other.insertions
        self.deletions += other.deletions
        self.reference_length += other.reference_length
        return self


def align(hypothesis: list, reference: list) -> EditCounts:
    """Levenshtein alignment returning S/I/D counts (deterministic backtrace)."""
    n, m = len(hypothesis), len(reference)
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if hypothesis[i - 1] == reference[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j - 1] + cost, dp[i - 1, j] + 1, dp[i, j - 1] + 1)
    counts = EditCounts(reference_length=m)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if hypothesis[i - 1] == reference[j - 1] else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost:
                counts.substitutions += cost
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            counts.insertions += 1
            i -= 1
            continue
        counts.deletions += 1
        j -= 1
    return counts


def words(text: str) -> list[str]:
    return text.split()


def characters(text: str) -> list[str]:
    return list(text.replace(" ", ""))


# --------------------------------------------------------------------------- #
# Category extraction (set based: what a reviewer checks first)
# --------------------------------------------------------------------------- #
class Categories:
    """Reviewed terminology drives the term/drug/abbreviation categories."""

    def __init__(self) -> None:
        rules = default_rules()
        self.terms: set[str] = set()
        self.drugs: set[str] = set()
        self.abbreviations: set[str] = set()
        for rule in rules:
            forms = {rule.canonical, *rule.spoken_forms, *rule.aliases}
            forms = {f for f in forms if f}
            if rule.category == "medication":
                self.drugs |= forms
            if rule.category == "abbreviation":
                self.abbreviations |= forms
            if rule.category not in ("unit",):
                self.terms |= forms

    @staticmethod
    def _present(text: str, vocabulary: set[str]) -> set[str]:
        return {phrase for phrase in vocabulary if phrase and phrase in text}

    def medical_terms(self, text: str) -> set[str]:
        return self._present(text, self.terms)

    def drug_names(self, text: str) -> set[str]:
        return self._present(text, self.drugs)

    def abbreviations_in(self, text: str) -> set[str]:
        found = self._present(text, self.abbreviations)
        return found | set(_LATIN_ABBREV_RE.findall(text))

    @staticmethod
    def numbers(text: str) -> set[str]:
        return set(_NUMBER_RE.findall(text))

    @staticmethod
    def blood_pressures(text: str) -> set[str]:
        return {m.group(0).replace(" ", "") for m in _BP_RE.finditer(text)}


@dataclass
class SetErrors:
    """Missed + spurious items over a category, as an error rate."""

    missed: int = 0
    spurious: int = 0
    expected: int = 0

    def update(self, hypothesis: set[str], reference: set[str]) -> None:
        self.missed += len(reference - hypothesis)
        self.spurious += len(hypothesis - reference)
        self.expected += len(reference)

    @property
    def rate(self) -> float:
        return (self.missed + self.spurious) / max(1, self.expected)


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
@dataclass
class CorpusRow:
    audio: Path
    reference: str
    forced: bool = False
    row_id: str = ""


def load_corpus(path: Path) -> list[CorpusRow]:
    base = path.parent
    rows: list[CorpusRow] = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            records: Iterable[dict] = list(csv.DictReader(handle))
    else:
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    for index, record in enumerate(records):
        audio = record.get("audio") or record.get("wav") or record.get("path")
        if not audio:
            raise ValueError(f"row {index}: no 'audio' field (this is the real-audio evaluator)")
        reference = str(record.get("reference") or record.get("text") or "").strip()
        if not reference:
            raise ValueError(f"row {index}: no reference transcript")
        forced = str(record.get("forced", "")).strip().lower() in {"1", "true", "yes"}
        candidate = Path(audio)
        rows.append(CorpusRow(
            audio=candidate if candidate.is_absolute() else base / candidate,
            reference=reference,
            forced=forced,
            row_id=str(record.get("id") or index),
        ))
    return rows


def read_wav(path: Path) -> np.ndarray:
    """Read a 16 kHz mono WAV as float32; anything else is an explicit error."""
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16000 or handle.getnchannels() != 1:
            raise ValueError(
                f"{path.name}: expected 16 kHz mono, got "
                f"{handle.getframerate()} Hz / {handle.getnchannels()} ch "
                "(the evaluator never resamples silently)"
            )
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width == 2:
        return (np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0)
    if width == 4:
        return np.frombuffer(frames, dtype="<f4").astype(np.float32)
    raise ValueError(f"{path.name}: unsupported sample width {width * 8} bit")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    system: str
    model: str
    decoder: str
    samples: int = 0
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    attribution: dict[str, float] = field(default_factory=dict)


def build_backend(args: argparse.Namespace):
    """Load the configured sherpa-onnx model (lazy import)."""
    from shenava_realtime.asr_backend import SherpaOnnxASR
    from shenava_realtime.config import AppConfig

    config = AppConfig.from_env()
    if args.model:
        config.asr.model_path = args.model
    if args.tokens:
        config.asr.tokens_path = args.tokens
    if args.device:
        config.asr.device = args.device
    config.asr.benchmark_mode = True
    config.asr.__post_init__()
    backend = SherpaOnnxASR(config.asr)
    backend.load()
    return backend, config


def evaluate(args: argparse.Namespace) -> Report:
    rows = load_corpus(args.corpus)
    backend, config = build_backend(args)
    processor = PostProcessor(config.postprocess)
    raw_processor = PostProcessor(config.postprocess)
    raw_processor.config.enabled = False
    categories = Categories()

    report = Report(
        system=args.system,
        model=str(config.asr.model_path),
        decoder="ctc",
    )
    wer = EditCounts()
    cer = EditCounts()
    acoustic_wer = EditCounts()
    normalization_wer = EditCounts()
    term_errors = SetErrors()
    drug_errors = SetErrors()
    number_errors = SetErrors()
    bp_errors = SetErrors()
    abbreviation_errors = SetErrors()
    forced_total = forced_wrong = 0
    latencies: list[float] = []
    audio_seconds = 0.0

    for row in rows:
        try:
            audio = read_wav(row.audio)
        except (OSError, ValueError) as exc:
            report.errors.append(f"{row.row_id}: {exc}")
            continue
        started = time.perf_counter()
        raw_text, _ = backend.transcribe(audio)
        latency = time.perf_counter() - started
        output = processor.process(raw_text)

        latencies.append(latency)
        audio_seconds += audio.size / 16000.0
        report.samples += 1

        reference = row.reference
        wer += align(words(output), words(reference))
        cer += align(characters(output), characters(reference))
        # Acoustic: raw decoder text against the reference.
        acoustic_wer += align(words(raw_text), words(reference))
        # Normalization: how much the deterministic stage moved the text.
        normalization_wer += align(words(output), words(raw_text))

        term_errors.update(categories.medical_terms(output), categories.medical_terms(reference))
        drug_errors.update(categories.drug_names(output), categories.drug_names(reference))
        number_errors.update(categories.numbers(output), categories.numbers(reference))
        bp_errors.update(categories.blood_pressures(output), categories.blood_pressures(reference))
        abbreviation_errors.update(
            categories.abbreviations_in(output), categories.abbreviations_in(reference)
        )
        if row.forced:
            forced_total += 1
            if output.strip() != reference.strip():
                forced_wrong += 1

    total_latency = sum(latencies)
    report.metrics = {
        "WER": wer.rate,
        "CER": cer.rate,
        "Substitutions": wer.substitutions,
        "Insertions": wer.insertions,
        "Deletions": wer.deletions,
        "Medical Term Error Rate": term_errors.rate,
        "Drug Name Error Rate": drug_errors.rate,
        "Dose/Number Error Rate": number_errors.rate,
        "BP Error Rate": bp_errors.rate,
        "Abbreviation Error Rate": abbreviation_errors.rate,
        "Forced-boundary Error Rate": forced_wrong / max(1, forced_total),
        "Mean Latency (s)": total_latency / max(1, len(latencies)),
        "Max Latency (s)": max(latencies) if latencies else 0.0,
        "Real-time Factor": total_latency / max(1e-9, audio_seconds),
        "Audio Seconds": audio_seconds,
    }
    report.attribution = {
        "acoustic": acoustic_wer.rate,
        "normalization": normalization_wer.rate,
        "terminology": term_errors.rate,
        "numbers": number_errors.rate,
        "endpointing": forced_wrong / max(1, forced_total),
    }
    return report


def print_report(report: Report) -> None:
    print(f"system: {report.system}  model: {report.model}  head: {report.decoder}")
    print(f"samples: {report.samples}")
    if report.errors:
        print(f"unreadable rows: {len(report.errors)}")
        for message in report.errors[:10]:
            print(f"  ! {message}")
    print("-- metrics --")
    for key, value in report.metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("-- error attribution (separate sources, not additive) --")
    for key, value in report.attribution.items():
        print(f"{key}: {value:.4f}")
    print("Measured on the supplied corpus only; not clinical validation.")


def compare(paths: list[Path]) -> None:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    keys = sorted({key for report in reports for key in report["metrics"]})
    header = ["metric"] + [f"{r['system']}" for r in reports]
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for key in keys:
        values = []
        for report in reports:
            value = report["metrics"].get(key)
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        print(" | ".join([key, *values]))
    print("\nDifferences are measurements on the same corpus, not a claim of "
          "general accuracy improvement.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("corpus", type=Path, nargs="?",
                        help="JSONL/CSV metadata with audio + reference columns")
    parser.add_argument("--system", choices=SYSTEMS, default="sherpa-onnx-ctc",
                        help="label for this run (default sherpa-onnx-ctc; the only supported system)")
    parser.add_argument("--model", default=None, help="path to the sherpa-onnx model.int8.onnx")
    parser.add_argument("--tokens", default=None, help="path to the sherpa-onnx tokens.txt")
    parser.add_argument("--device", default=None, help="cpu (only supported value)")
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--compare", type=Path, nargs="+", default=None,
                        help="print a comparison table of previously written reports")
    args = parser.parse_args(argv)

    if args.compare:
        compare(args.compare)
        return 0
    if args.corpus is None:
        parser.error("a corpus is required unless --compare is used")
    report = evaluate(args)
    print_report(report)
    if args.report:
        args.report.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
