"""Pure helpers of the ASR backend (no torch, no checkpoint)."""

from types import SimpleNamespace

import pytest

from shenava_realtime.asr_backend import NeMoASR, estimate_confidence, score_to_confidence
from shenava_realtime.config import ASRConfig


@pytest.mark.parametrize(
    ("score", "text", "expected"),
    [
        (None, "سلام", 0.0),
        (0.85, "سلام", 0.85),
        (-0.5, "سلام", pytest.approx(0.6065, abs=1e-3)),
        (-4.0, "یک دو سه چهار", pytest.approx(0.3679, abs=1e-3)),
        (float("nan"), "سلام", 0.0),
        (float("inf"), "سلام", 0.0),
    ],
)
def test_score_to_confidence(score, text, expected):
    assert score_to_confidence(score, text) == expected


def test_confidence_is_length_normalized():
    short = score_to_confidence(-2.0, "یک")
    long = score_to_confidence(-2.0, "یک دو سه چهار")
    assert long > short


def test_estimate_confidence_penalizes_repetitions():
    assert estimate_confidence("") == 0.0
    assert estimate_confidence("بیمار آمد") == 1.0
    assert estimate_confidence("بیمار بیمار آمد") < estimate_confidence("بیمار آمد")
    assert estimate_confidence("بیمار") < estimate_confidence("بیمار آمد")


def test_extract_handles_ne_mo_shapes():
    extract = NeMoASR._extract

    assert extract(None) == ("", 0.0)
    assert extract([]) == ("", 0.0)
    assert extract("متن ساده")[0] == "متن ساده"
    assert extract(["متن ساده"])[0] == "متن ساده"
    assert extract({"text": "متن", "score": 0.5}) == ("متن", 0.5)

    hypothesis = SimpleNamespace(text="  بیمار آمد  ", score=-0.2)
    text, confidence = extract([hypothesis])
    assert text == "بیمار آمد"
    assert 0.0 < confidence <= 1.0

    # A non-numeric score must not raise.
    odd = SimpleNamespace(text="بیمار", score="n/a")
    text, confidence = extract([odd])
    assert text == "بیمار"
    assert confidence > 0.0


def test_device_resolution_defaults_to_cpu_without_cuda(monkeypatch):
    backend = NeMoASR(ASRConfig(device="auto"))
    monkeypatch.setattr(
        backend,
        "_import_torch",
        lambda: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    assert backend.resolve_device() == "cpu"


def test_device_resolution_honours_an_explicit_cpu(monkeypatch):
    backend = NeMoASR(ASRConfig(device="cpu"))
    monkeypatch.setattr(
        backend,
        "_import_torch",
        lambda: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    assert backend.resolve_device() == "cpu"


def test_cuda_request_falls_back_when_unavailable(monkeypatch):
    backend = NeMoASR(ASRConfig(device="cuda"))
    monkeypatch.setattr(
        backend,
        "_import_torch",
        lambda: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    assert backend.resolve_device() == "cpu"


def test_transcribe_of_empty_audio_short_circuits():
    backend = NeMoASR(ASRConfig())
    backend._model = object()  # pretend it is loaded; the call must not reach it
    import numpy as np

    assert backend.transcribe(np.zeros(0, dtype=np.float32)) == ("", 0.0)


def test_relative_checkpoint_paths_resolve_against_the_repo(tmp_path, monkeypatch):
    import shenava_realtime.asr_backend as module

    checkpoint = tmp_path / "shenava.nemo"
    checkpoint.write_bytes(b"not really a model")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    backend = NeMoASR(ASRConfig(model_path="shenava.nemo"))
    assert backend.resolve_checkpoint_path() == checkpoint

    backend = NeMoASR(ASRConfig(model_path=str(checkpoint)))
    assert backend.resolve_checkpoint_path() == checkpoint

    backend = NeMoASR(ASRConfig(model_path="missing.nemo"))
    assert backend.resolve_checkpoint_path() is None

    backend = NeMoASR(ASRConfig(model_path=None))
    assert backend.resolve_checkpoint_path() is None
