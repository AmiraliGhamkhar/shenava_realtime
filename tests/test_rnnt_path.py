"""RNNT backend path: capability gating, head selection, decoder-aware second pass.

No NeMo/torch here — these tests drive the seams the real checkpoint plugs into
and assert the failure behaviour: a missing capability must raise, never fall
back to the other head.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from shenava_realtime.config import ASRConfig
from shenava_realtime.rnnt_stream import (
    RNNTCapabilityError,
    RNNTOfflineDecoder,
    RNNTSecondPass,
    _extract_text,
    require_rnnt_head,
)
from shenava_realtime.second_pass import SecondPassUtterance
from shenava_realtime.streaming import make_second_pass


# --------------------------------------------------------------------------- #
# Configuration: ctc | rnnt | auto, CTC stays the default
# --------------------------------------------------------------------------- #
def test_ctc_is_the_default_head():
    assert ASRConfig().decoder_type == "ctc"
    assert ASRConfig().resolved_decoder == "ctc"


def test_auto_resolves_explicitly_to_the_production_head():
    assert ASRConfig(decoder_type="auto").resolved_decoder == "ctc"


def test_rnnt_is_selectable_and_kept_distinct():
    assert ASRConfig(decoder_type="rnnt").resolved_decoder == "rnnt"


def test_unknown_decoder_names_are_rejected():
    with pytest.raises(ValueError, match="decoder_type must be one of"):
        ASRConfig(decoder_type="transducer")
    with pytest.raises(ValueError):
        ASRConfig(decoder_type="")


def test_decoder_names_are_normalised():
    assert ASRConfig(decoder_type=" RNNT ").resolved_decoder == "rnnt"


# --------------------------------------------------------------------------- #
# Capability gating: no silent fallback to CTC
# --------------------------------------------------------------------------- #
def _hybrid_model():
    return SimpleNamespace(
        joint=SimpleNamespace(), decoder=SimpleNamespace(),
        decoding=SimpleNamespace(blank_id=127),
    )


def test_require_rnnt_head_accepts_a_hybrid_checkpoint():
    require_rnnt_head(_hybrid_model())  # does not raise


def test_require_rnnt_head_rejects_a_ctc_only_checkpoint():
    model = SimpleNamespace(decoder=SimpleNamespace())  # no joint
    with pytest.raises(RNNTCapabilityError, match="prediction network/joint"):
        require_rnnt_head(model)


def test_require_rnnt_head_rejects_a_model_without_a_decoding_strategy():
    model = _hybrid_model()
    model.decoding = None
    with pytest.raises(RNNTCapabilityError, match="decoding strategy"):
        require_rnnt_head(model)


def test_offline_rnnt_decoder_refuses_a_checkpoint_without_the_head():
    with pytest.raises(RNNTCapabilityError):
        RNNTOfflineDecoder(SimpleNamespace(decoder=SimpleNamespace()), SimpleNamespace())


# --------------------------------------------------------------------------- #
# Offline RNNT decoding uses the model's own transcribe
# --------------------------------------------------------------------------- #
class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_torch():
    return SimpleNamespace(no_grad=lambda: _NoGrad())


def _recording_model(text="بیمار تب دارد"):
    model = _hybrid_model()
    model.calls = []

    def transcribe(audio, **kwargs):
        model.calls.append((audio[0].size, kwargs))
        return [SimpleNamespace(text=text)]

    model.transcribe = transcribe
    return model


def test_offline_decode_returns_text_and_no_invented_confidence():
    model = _recording_model()
    decoder = RNNTOfflineDecoder(model, _fake_torch())
    text, confidence = decoder.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "بیمار تب دارد"
    assert confidence == 0.0  # scores are uncalibrated; never presented as one
    assert decoder.decodes == 1
    assert model.calls[0][0] == 16000


def test_offline_decode_of_empty_audio_does_not_touch_the_model():
    model = _recording_model()
    decoder = RNNTOfflineDecoder(model, _fake_torch())
    assert decoder.transcribe(np.zeros(0, dtype=np.float32)) == ("", 0.0)
    assert model.calls == [] and decoder.decodes == 0


def test_statistics_are_tracked_separately_from_ctc():
    decoder = RNNTOfflineDecoder(_recording_model(), _fake_torch())
    decoder.transcribe(np.zeros(1600, dtype=np.float32))
    decoder.transcribe(np.zeros(1600, dtype=np.float32))
    assert decoder.decodes == 2


def test_transcribe_output_shapes_are_all_handled():
    assert _extract_text(None) == ""
    assert _extract_text([]) == ""
    assert _extract_text(["  متن  "]) == "متن"
    assert _extract_text([{"text": "متن"}]) == "متن"
    assert _extract_text([[SimpleNamespace(text="متن")]]) == "متن"


# --------------------------------------------------------------------------- #
# Second pass stays decoder-aware: RNNT is never re-decoded with CTC
# --------------------------------------------------------------------------- #
def test_rnnt_second_pass_re_decodes_with_rnnt():
    second_pass = RNNTSecondPass(RNNTOfflineDecoder(_recording_model("متن نهایی"), _fake_torch()))
    utterance = SecondPassUtterance(audio=np.zeros(16000, dtype=np.float32), duration_s=1.0)
    assert second_pass.decode_greedy(utterance) == "متن نهایی"
    assert second_pass.runs == 1


def test_rnnt_second_pass_has_no_hotword_beam_variant():
    """Reviewed CTC hotword bias is not transferable to the transducer."""
    second_pass = RNNTSecondPass(RNNTOfflineDecoder(_recording_model(), _fake_torch()))
    utterance = SecondPassUtterance(audio=np.zeros(16000, dtype=np.float32), duration_s=1.0)
    assert second_pass.decode_with_context(utterance, [object()]) is None


def test_rnnt_second_pass_returns_none_for_empty_audio():
    second_pass = RNNTSecondPass(RNNTOfflineDecoder(_recording_model(), _fake_torch()))
    utterance = SecondPassUtterance(audio=np.zeros(0, dtype=np.float32), duration_s=0.0)
    assert second_pass.decode_greedy(utterance) is None


def test_rnnt_selection_requires_a_backend_that_can_build_the_rnnt_pass():
    class CTCOnlyBackend:
        def transcribe(self, audio):
            return "", 0.0

    config = ASRConfig(decoder_type="rnnt", second_pass="greedy")
    with pytest.raises(RuntimeError, match="decoder=rnnt requires"):
        make_second_pass(CTCOnlyBackend(), config)


def test_rnnt_selection_uses_the_backends_own_factory():
    built = []

    class HybridBackend:
        def transcribe(self, audio):
            return "", 0.0

        def build_second_pass(self, config):
            built.append(config.resolved_decoder)
            return RNNTSecondPass(RNNTOfflineDecoder(_recording_model(), _fake_torch()))

    result = make_second_pass(HybridBackend(), ASRConfig(decoder_type="rnnt"))
    assert built == ["rnnt"]
    assert result.name == "rnnt-endpoint"


def test_ctc_selection_never_builds_an_rnnt_pass():
    class Backend:
        def transcribe(self, audio):
            return "متن", 0.0

    result = make_second_pass(Backend(), ASRConfig(decoder_type="ctc", second_pass="greedy"))
    assert result.name == "greedy"


def test_second_pass_off_is_honoured_for_both_heads():
    class Backend:
        def transcribe(self, audio):
            return "", 0.0

        def build_second_pass(self, config):  # pragma: no cover - must not run
            raise AssertionError("second_pass=off must not build a decoder")

    assert make_second_pass(Backend(), ASRConfig(second_pass="off")) is None
    assert make_second_pass(
        Backend(), ASRConfig(decoder_type="rnnt", second_pass="off")
    ) is None
