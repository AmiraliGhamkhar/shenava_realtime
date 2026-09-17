"""Real-audio evaluator: alignment maths, corpus loading, WAV validation.

The evaluator is a tool, not runtime code, but its arithmetic has to be right
or every benchmark number it prints is wrong.
"""
import json
import wave

import numpy as np
import pytest

from tools.evaluate_audio import (
    Categories,
    EditCounts,
    SetErrors,
    align,
    characters,
    compare,
    load_corpus,
    read_wav,
    words,
)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def test_identical_sequences_have_no_errors():
    counts = align(words("بیمار تب دارد"), words("بیمار تب دارد"))
    assert (counts.substitutions, counts.insertions, counts.deletions) == (0, 0, 0)
    assert counts.rate == 0.0


def test_substitution_insertion_and_deletion_are_counted_separately():
    assert align(words("a x c"), words("a b c")).substitutions == 1
    inserted = align(words("a b c d"), words("a b c"))
    assert (inserted.insertions, inserted.substitutions, inserted.deletions) == (1, 0, 0)
    deleted = align(words("a c"), words("a b c"))
    assert (deleted.deletions, deleted.substitutions, deleted.insertions) == (1, 0, 0)


def test_word_error_rate_is_errors_over_reference_length():
    counts = align(words("a x c d"), words("a b c"))
    assert counts.reference_length == 3
    assert counts.errors == 2  # one substitution + one insertion
    assert counts.rate == pytest.approx(2 / 3)


def test_empty_hypothesis_is_all_deletions():
    counts = align([], words("a b c"))
    assert counts.deletions == 3 and counts.rate == 1.0


def test_character_error_rate_ignores_spaces():
    assert characters("ab c") == ["a", "b", "c"]
    assert align(characters("abc"), characters("a bc")).errors == 0


def test_edit_counts_accumulate_across_rows():
    total = EditCounts()
    total += align(words("a x"), words("a b"))
    total += align(words("c"), words("c"))
    assert total.substitutions == 1 and total.reference_length == 3


# --------------------------------------------------------------------------- #
# Category error rates
# --------------------------------------------------------------------------- #
def test_set_errors_count_missed_and_spurious():
    errors = SetErrors()
    errors.update({"a", "z"}, {"a", "b"})
    assert (errors.missed, errors.spurious, errors.expected) == (1, 1, 2)
    assert errors.rate == 1.0


def test_set_errors_rate_is_zero_when_nothing_is_expected_or_produced():
    errors = SetErrors()
    errors.update(set(), set())
    assert errors.rate == 0.0


def test_categories_are_driven_by_the_reviewed_terminology():
    categories = Categories()
    assert "متفورمین" in categories.drug_names("بیمار متفورمین می‌گیرد")
    assert categories.drug_names("بیمار آمد") == set()
    assert "CABG" in categories.abbreviations_in("بیمار CABG شد")


def test_number_and_bp_extraction():
    assert Categories.numbers("دوز 2.5 mg و 10 mg") == {"2.5", "10"}
    assert Categories.blood_pressures("BP 120/80 و ضربان 75") == {"120/80"}
    assert Categories.blood_pressures("بدون فشار") == set()


# --------------------------------------------------------------------------- #
# Corpus + WAV handling
# --------------------------------------------------------------------------- #
def _write_wav(path, samples, rate=16000, channels=1, width=2):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes((np.asarray(samples) * 32767).astype("<i2").tobytes())


def test_jsonl_corpus_resolves_audio_relative_to_the_metadata(tmp_path):
    (tmp_path / "clips").mkdir()
    _write_wav(tmp_path / "clips" / "a.wav", np.zeros(160))
    meta = tmp_path / "corpus.jsonl"
    meta.write_text(
        json.dumps({"audio": "clips/a.wav", "reference": "سلام"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = load_corpus(meta)
    assert rows[0].audio == tmp_path / "clips" / "a.wav"
    assert rows[0].reference == "سلام" and rows[0].forced is False


def test_csv_corpus_and_forced_flag(tmp_path):
    meta = tmp_path / "corpus.csv"
    meta.write_text("audio,reference,forced\na.wav,متن,true\n", encoding="utf-8")
    rows = load_corpus(meta)
    assert rows[0].forced is True and rows[0].reference == "متن"


def test_corpus_rows_without_audio_or_reference_fail_loudly(tmp_path):
    meta = tmp_path / "corpus.jsonl"
    meta.write_text(json.dumps({"reference": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'audio' field"):
        load_corpus(meta)
    meta.write_text(json.dumps({"audio": "a.wav"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no reference"):
        load_corpus(meta)


def test_wav_reading_returns_float32_in_range(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path, np.linspace(-0.5, 0.5, 320))
    audio = read_wav(path)
    assert audio.dtype == np.float32 and audio.size == 320
    assert -1.0 <= float(audio.min()) and float(audio.max()) <= 1.0


def test_non_16k_or_stereo_audio_is_rejected_never_resampled(tmp_path):
    path = tmp_path / "bad.wav"
    _write_wav(path, np.zeros(320), rate=8000)
    with pytest.raises(ValueError, match="16 kHz mono"):
        read_wav(path)
    stereo = tmp_path / "stereo.wav"
    _write_wav(stereo, np.zeros(320), channels=2)
    with pytest.raises(ValueError, match="16 kHz mono"):
        read_wav(stereo)


def test_compare_prints_one_row_per_metric(tmp_path, capsys):
    for name, wer in (("v1.5-ctc", 0.10), ("v1.5-rnnt", 0.12)):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "system": name, "model": "m", "decoder": "ctc", "samples": 1,
            "errors": [], "metrics": {"WER": wer}, "attribution": {},
        }), encoding="utf-8")
    compare([tmp_path / "v1.5-ctc.json", tmp_path / "v1.5-rnnt.json"])
    output = capsys.readouterr().out
    assert "v1.5-ctc" in output and "v1.5-rnnt" in output
    assert "0.1000" in output and "0.1200" in output
    assert "not a claim of" in output
