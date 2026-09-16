#!/usr/bin/env python3
"""Reproducible WAV evaluation for Shenava checkpoints.

JSONL rows contain ``audio`` (WAV path relative to the metadata file),
``reference``, and optional ``forced``/``id``.  This tool deliberately keeps
model execution separate from the realtime worker and reports raw decoder
scores; post-processing is not mixed into acoustic WER.
"""
from __future__ import annotations
import argparse, json, time, wave
from pathlib import Path
import numpy as np


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono PCM16 WAV")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def edit_counts(hyp: list[str], ref: list[str]) -> tuple[int, int, int]:
    # deterministic Levenshtein traceback: substitution, deletion, insertion
    dp = [[(0, 0, 0, 0)] * (len(ref) + 1) for _ in range(len(hyp) + 1)]
    for i in range(1, len(hyp)+1): dp[i][0] = (i, 0, i, 0)
    for j in range(1, len(ref)+1): dp[0][j] = (j, 0, 0, j)
    for i in range(1, len(hyp)+1):
        for j in range(1, len(ref)+1):
            same = hyp[i-1] == ref[j-1]
            choices = [(dp[i-1][j-1][0] + (not same), 0 if same else 1, 0, 0),
                       (dp[i-1][j][0] + 1, 0, 1, 0),
                       (dp[i][j-1][0] + 1, 0, 0, 1)]
            dp[i][j] = min(choices, key=lambda x: x)
    _, s, d, ins = dp[-1][-1]
    return int(s), int(d), int(ins)


def run(metadata: Path, model: str | None, decoder: str) -> dict:
    from shenava_realtime.asr_backend import NeMoASR
    from shenava_realtime.config import ASRConfig
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    config = ASRConfig(model_path=model, decoder_type=decoder)
    backend = NeMoASR(config)
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0,
              "reference_chars": 0, "hyp_words": 0, "audio_seconds": 0.0, "decode_seconds": 0.0,
              "utterances": len(rows), "forced": 0, "forced_errors": 0}
    started = time.perf_counter()
    for row in rows:
        audio = read_wav((metadata.parent / row["audio"]).resolve())
        begin = time.perf_counter(); text, _ = backend.transcribe(audio); elapsed = time.perf_counter() - begin
        reference = str(row["reference"])
        hyp_words, ref_words = text.split(), reference.split()
        s, d, ins = edit_counts(hyp_words, ref_words)
        totals["substitutions"] += s; totals["deletions"] += d; totals["insertions"] += ins
        totals["reference_words"] += len(ref_words); totals["hyp_words"] += len(hyp_words)
        totals["reference_chars"] += len(reference.replace(" ", ""))
        cs, cd, ci = edit_counts(list(text.replace(" ", "")), list(reference.replace(" ", "")))
        totals.setdefault("char_errors", 0); totals["char_errors"] += cs + cd + ci
        for category in ("medical_terms", "drug_names", "doses_numbers", "bp", "abbreviations"):
            expected = row.get(category, row.get(category[:-1] if category.endswith("s") else category, []))
            if isinstance(expected, str): expected = [expected]
            if expected:
                totals.setdefault(category + "_total", 0); totals[category + "_total"] += len(expected)
                totals.setdefault(category + "_errors", 0); totals[category + "_errors"] += sum(x not in text for x in expected)
        totals["audio_seconds"] += audio.size / 16000; totals["decode_seconds"] += elapsed
        if row.get("forced"):
            totals["forced"] += 1; totals["forced_errors"] += int(text.strip() != reference.strip())
    errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    totals.update({"decoder": decoder, "model": config.model_name, "wer": errors / max(1, totals["reference_words"]),
                   "cer": totals.get("char_errors", 0) / max(1, totals["reference_chars"]),
                   "forced_boundary_error_rate": totals["forced_errors"] / max(1, totals["forced"]),
                   "real_time_factor": totals["decode_seconds"] / max(1e-9, totals["audio_seconds"]),
                   "wall_seconds": time.perf_counter() - started})
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Shenava CTC/RNNT on labelled 16 kHz WAV metadata")
    parser.add_argument("metadata", type=Path); parser.add_argument("--model")
    parser.add_argument("--decoder", choices=("ctc", "rnnt"), default="ctc")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(args.metadata, args.model, args.decoder)
    print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
    print("Engineering measurement only; not clinical validation.")

if __name__ == "__main__":
    main()
