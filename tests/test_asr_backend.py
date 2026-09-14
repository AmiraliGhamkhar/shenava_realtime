"""Pure helpers of the ASR backend (no torch, no checkpoint)."""

from types import SimpleNamespace

import pytest

from shenava_realtime.asr_backend import NeMoASR
from shenava_realtime.config import ASRConfig


def test_extract_handles_ne_mo_shapes():
    extract = NeMoASR._extract

    assert extract(None) == ("", 0.0)
    assert extract([]) == ("", 0.0)
    assert extract("متن ساده")[0] == "متن ساده"
    assert extract(["متن ساده"])[0] == "متن ساده"
    assert extract({"text": "متن", "score": 0.5}) == ("متن", 0.0)

    hypothesis = SimpleNamespace(text="  بیمار آمد  ", score=-0.2)
    text, confidence = extract([hypothesis])
    assert text == "بیمار آمد"
    assert confidence == 0.0

    # A non-numeric score must not raise.
    odd = SimpleNamespace(text="بیمار", score="n/a")
    text, confidence = extract([odd])
    assert text == "بیمار"
    assert confidence == 0.0


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


def test_missing_checkpoint_does_not_import_torch_or_download(monkeypatch):
    backend = NeMoASR(ASRConfig(model_path="/missing/checkpoint.nemo"))
    monkeypatch.setattr(backend, "_import_torch", lambda: pytest.fail("must fail before importing torch"))
    with pytest.raises(FileNotFoundError, match="Local Shenava"):
        backend.load()
