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
# Capability probing: heads, streaming contexts, blanks — probed, not assumed.
# --------------------------------------------------------------------------- #
from shenava_realtime.asr_backend import CapabilityUnavailable, ModelCapabilities


class _Tokenizer:
    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def get_vocab(self):
        return {f"t{i}": i for i in range(128)}


def _model(contexts=None, native=True, *, ctc=True, rnnt=False,
           sample_rate=16000, tokenizer=True):
    encoder = types.SimpleNamespace(att_context_size_all=contexts)
    model = types.SimpleNamespace(
        encoder=encoder,
        cfg=SimpleNamespace(preprocessor=SimpleNamespace(sample_rate=sample_rate)),
    )
    if native:
        model.conformer_stream_step = lambda **kwargs: None
    if ctc:
        model.ctc_decoder = SimpleNamespace(blank_index=0)
    if rnnt:
        model.joint = SimpleNamespace()
        model.decoder = SimpleNamespace()
        model.decoding = SimpleNamespace(blank_id=127)
    if tokenizer:
        model.tokenizer = _Tokenizer()
    return model


def _probe(model, **config_kwargs):
    return NeMoASR(ASRConfig(**config_kwargs))._probe_capabilities(model)


CONTEXTS = [[70, 13], [70, 6], [70, 1], [70, 0]]


def test_supported_contexts_enable_streaming():
    capabilities = _probe(_model(CONTEXTS))
    assert capabilities.supports_cache_aware_streaming is True
    assert capabilities.streaming_contexts == ((70, 13), (70, 6), (70, 1), (70, 0))
    for right in (0, 1, 6, 13):
        NeMoASR(ASRConfig(right_context=right))._validate_startup(capabilities, "ctc")


def test_unsupported_right_context_fails_at_startup():
    """Regression: [70, 13] on a [70, 0]-only encoder degraded to endpoint-only."""
    capabilities = _probe(_model([[70, 0]]))
    backend = NeMoASR(ASRConfig(right_context=13))
    with pytest.raises(CapabilityUnavailable, match="right_context=13"):
        backend._validate_startup(capabilities, "ctc")


def test_offline_checkpoint_keeps_the_documented_endpoint_fallback():
    capabilities = _probe(_model(CONTEXTS, native=False))
    assert capabilities.supports_cache_aware_streaming is False
    # Endpoint-only is allowed unless streaming was explicitly required.
    NeMoASR(ASRConfig(right_context=13))._validate_startup(capabilities, "ctc")
    with pytest.raises(CapabilityUnavailable, match="require_streaming"):
        NeMoASR(ASRConfig(require_streaming=True))._validate_startup(capabilities, "ctc")


def test_missing_context_metadata_never_enables_streaming():
    assert _probe(_model(None)).supports_cache_aware_streaming is False


def test_capabilities_report_both_heads_of_a_hybrid_checkpoint():
    capabilities = _probe(_model(CONTEXTS, rnnt=True))
    assert capabilities.heads() == ("ctc", "rnnt")
    assert capabilities.rnnt_blank_index == 127
    assert capabilities.ctc_blank_index == 0
    assert capabilities.vocab_size == 128
    assert "heads=ctc,rnnt" in capabilities.describe()


def test_rnnt_request_fails_clearly_on_a_ctc_only_checkpoint():
    capabilities = _probe(_model(CONTEXTS, rnnt=False))
    assert capabilities.has_rnnt_head is False
    with pytest.raises(CapabilityUnavailable, match="no\\s+RNNT"):
        NeMoASR(ASRConfig(decoder_type="rnnt"))._validate_startup(capabilities, "rnnt")


def test_rnnt_requires_a_resolvable_blank_id():
    model = _model(CONTEXTS, rnnt=True)
    model.decoding = SimpleNamespace()
    model.joint = SimpleNamespace()
    model.decoder = SimpleNamespace()
    capabilities = _probe(model)
    with pytest.raises(CapabilityUnavailable, match="blank"):
        NeMoASR(ASRConfig(decoder_type="rnnt"))._validate_startup(capabilities, "rnnt")


def test_non_16k_frontend_fails_at_startup():
    capabilities = _probe(_model(CONTEXTS, sample_rate=8000))
    with pytest.raises(CapabilityUnavailable, match="16 kHz"):
        NeMoASR(ASRConfig())._validate_startup(capabilities, "ctc")


def test_head_selection_uses_the_models_own_api():
    calls = []
    model = _model(CONTEXTS, rnnt=True)
    model.change_decoding_strategy = lambda cfg, decoder_type=None: calls.append(decoder_type)
    NeMoASR(ASRConfig())._select_head(model, "rnnt")
    assert calls == ["rnnt"]


def test_head_selection_fails_when_the_checkpoint_cannot_switch_heads():
    model = _model(CONTEXTS)

    def only_ctc(cfg):  # no decoder_type keyword: single-head checkpoint
        return None

    model.change_decoding_strategy = only_ctc
    NeMoASR(ASRConfig())._select_head(model, "ctc")  # allowed
    with pytest.raises(CapabilityUnavailable, match="decoder_type"):
        NeMoASR(ASRConfig())._select_head(model, "rnnt")


# --------------------------------------------------------------------------- #
# CUDA-graph streaming: optional optimisation, eager is the reference path
# --------------------------------------------------------------------------- #
def test_cuda_graph_streaming_is_off_by_default_and_touches_nothing():
    model = _model(CONTEXTS, rnnt=True)
    model.decoding = SimpleNamespace(blank_id=1, decoding=SimpleNamespace())
    NeMoASR(ASRConfig())._apply_cuda_graph_option(model)
    assert not hasattr(model.decoding.decoding, "use_cuda_graph_decoder")


def test_cuda_graph_streaming_is_applied_when_requested_and_supported():
    model = _model(CONTEXTS, rnnt=True)
    model.decoding = SimpleNamespace(
        blank_id=1, decoding=SimpleNamespace(use_cuda_graph_decoder=False)
    )
    NeMoASR(ASRConfig(cuda_graph_streaming=True))._apply_cuda_graph_option(model)
    assert model.decoding.decoding.use_cuda_graph_decoder is True


def test_requesting_cuda_graph_streaming_without_support_fails_explicitly():
    model = _model(CONTEXTS, rnnt=True)
    model.decoding = SimpleNamespace(blank_id=1, decoding=SimpleNamespace())
    with pytest.raises(CapabilityUnavailable, match="use_cuda_graph_decoder"):
        NeMoASR(ASRConfig(cuda_graph_streaming=True))._apply_cuda_graph_option(model)
