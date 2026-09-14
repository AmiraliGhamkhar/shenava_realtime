"""Pure helpers of the ASR backend (no torch, no checkpoint)."""

import types
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


# --------------------------------------------------------------------------- #
# Right-context vs. encoder metadata: reject, never degrade silently.
# --------------------------------------------------------------------------- #
def _model(contexts=None, native=True):
    encoder = types.SimpleNamespace(att_context_size_all=contexts)
    model = types.SimpleNamespace(encoder=encoder)
    if native:
        model.conformer_stream_step = lambda **kwargs: None
    return model


def test_supported_contexts_enable_streaming():
    contexts = [[70, 13], [70, 6], [70, 1], [70, 0]]
    for right in (0, 1, 6, 13):
        assert NeMoASR._check_streaming_context(_model(contexts), right) is True


def test_unsupported_right_context_fails_at_startup():
    """Regression: [70, 13] on a [70, 0]-only encoder degraded to endpoint-only."""
    model = _model([[70, 0]])
    with pytest.raises(RuntimeError, match="right_context=13"):
        NeMoASR._check_streaming_context(model, 13)


def test_offline_checkpoint_keeps_the_documented_endpoint_fallback():
    contexts = [[70, 13], [70, 6], [70, 1], [70, 0]]
    assert NeMoASR._check_streaming_context(_model(contexts, native=False), 13) is False
    # No streaming API even at a matching context: endpoint-only, no raise.
    assert NeMoASR._check_streaming_context(_model(contexts, native=False), 0) is False


def test_missing_context_metadata_never_enables_streaming():
    assert NeMoASR._check_streaming_context(_model(None), 13) is False
