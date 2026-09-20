"""Second-pass decoding: greedy offline re-decode, pipeline integration.

The CTC-beam + hotword-biasing ("context") second pass required raw
per-frame emission logits and the model's own tokenizer object, both
NeMo-specific and not exposed by sherpa-onnx's public Python API. It was
removed along with ``CTCBeamDecoder``/``BeamSecondPass``; ``second_pass =
"context"`` is now rejected at ``ASRConfig.__post_init__`` (see
test_config_v15.py). This module keeps the greedy second pass and pipeline
integration tests, which are backend-agnostic.
"""
import numpy as np
import pytest

from shenava_realtime.config import ASRConfig
from shenava_realtime.pipeline import TranscriptionPipeline
from shenava_realtime.second_pass import GreedySecondPass, SecondPassUtterance
from shenava_realtime.streaming import DecodeResult, StreamingDecoder, make_second_pass
from tests.fakes import FakeBackend


# --------------------------------------------------------------------------- #
# Second-pass decoders
# --------------------------------------------------------------------------- #
class AudioCapturingDecoder(StreamingDecoder):
    """Scripted streaming decoder that keeps its utterance audio."""
    name = "audio-capture"

    def __init__(self, text):
        self.text = text
        self._audio = np.zeros(0, dtype=np.float32)
        self.resets = 0

    def reset(self):
        self.resets += 1
        self._audio = np.zeros(0, dtype=np.float32)

    def push(self, audio, force=False):
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size:
            self._audio = np.concatenate([self._audio, samples]) if self._audio.size else samples.copy()
        return DecodeResult(text=self.text, confidence=0.0, reset=False)

    def finalize(self):
        return DecodeResult(text=self.text, confidence=0.0, reset=False)

    def utterance_audio(self):
        return None if self._audio.size == 0 else self._audio


def pipeline_with_second_pass(stream_text, **kwargs):
    decoder = AudioCapturingDecoder(stream_text)
    pipeline = TranscriptionPipeline(
        decoder, holdback_words=0,
        second_pass=kwargs.pop("second_pass", None),
        hotwords=kwargs.pop("hotwords", []),
        **kwargs,
    )
    return pipeline, decoder


def test_second_pass_replaces_unemitted_final_text():
    calls = []
    pipeline, _decoder = pipeline_with_second_pass(
        "بیمار دوز دو و نیم میلی گرم",
        second_pass=GreedySecondPass(lambda a: (calls.append(a) or ("بیمار دوز 2.5 mg", 0.0))),
    )
    pipeline.start_utterance()
    pipeline.push_audio(np.zeros(1600, dtype=np.float32))  # 0.1 s < min
    deltas = pipeline.push_audio(np.zeros(16000, dtype=np.float32))  # now 1.1 s
    deltas.extend(pipeline.end_utterance())
    assert "".join(deltas) == "بیمار دوز 2.5 mg"
    assert calls, "the second pass was not used"
    assert "decoder_disagreement" in pipeline.last_review_reasons
    assert pipeline.second_pass_stats == {"runs": 1, "rewrites": 1, "fallbacks": 0}


def test_second_pass_skips_short_utterances():
    calls = []
    pipeline, _decoder = pipeline_with_second_pass(
        "بیمار", second_pass=GreedySecondPass(lambda a: (calls.append(a) or ("بیمار", 0.0))))
    pipeline.start_utterance()
    pipeline.push_audio(np.zeros(4000, dtype=np.float32))  # 0.25 s
    pipeline.end_utterance()
    assert calls == [] and pipeline.second_pass_stats["runs"] == 0


def test_second_pass_never_runs_on_forced_endpoints():
    calls = []
    pipeline, _decoder = pipeline_with_second_pass(
        "دوز سی و", second_pass=GreedySecondPass(lambda a: (calls.append(a) or ("x", 0.0))))
    pipeline.start_utterance()
    pipeline.push_audio(np.zeros(16000, dtype=np.float32))
    pipeline.end_utterance(forced=True)
    assert calls == []
    assert "forced_boundary" in pipeline.last_review_reasons


def test_second_pass_agreement_is_not_a_disagreement():
    pipeline, _decoder = pipeline_with_second_pass(
        "بیمار", second_pass=GreedySecondPass(lambda a: ("بیمار", 0.0)))
    pipeline.start_utterance()
    pipeline.push_audio(np.zeros(16000, dtype=np.float32))
    pipeline.end_utterance()
    assert pipeline.second_pass_stats == {"runs": 1, "rewrites": 0, "fallbacks": 0}
    assert "decoder_disagreement" not in pipeline.last_review_reasons


def test_second_pass_failure_is_explicit_and_keeps_greedy():
    def boom(audio):
        raise RuntimeError("emissions unavailable")

    pipeline, _decoder = pipeline_with_second_pass(
        "بیمار", second_pass=GreedySecondPass(boom))
    pipeline.start_utterance()
    pipeline.push_audio(np.zeros(16000, dtype=np.float32))
    deltas = pipeline.end_utterance()
    assert "".join(deltas) == "بیمار"  # streaming greedy kept
    assert pipeline.second_pass_stats["fallbacks"] == 1


def test_second_pass_inert_in_early_commit_mode():
    pipeline, _decoder = pipeline_with_second_pass(
        "بیمار", second_pass=GreedySecondPass(lambda a: ("x", 0.0)),
        commit_on_endpoint=False)
    assert pipeline.second_pass is None
    pipeline.start_utterance()
    pipeline.push_audio(np.zeros(16000, dtype=np.float32))
    pipeline.end_utterance()
    assert pipeline.second_pass_stats["runs"] == 0


# --------------------------------------------------------------------------- #
# Factory / configuration
# --------------------------------------------------------------------------- #
def test_make_second_pass_off_and_greedy():
    backend = FakeBackend()
    assert make_second_pass(backend, ASRConfig(second_pass="off")) is None
    decoder = make_second_pass(backend, ASRConfig(second_pass="greedy"))
    assert isinstance(decoder, GreedySecondPass)
    text = decoder.decode_greedy(SecondPassUtterance(np.zeros(160, np.float32), 0.01))
    assert text == "واژه"


def test_second_pass_context_is_rejected_at_config_time():
    """sherpa-onnx exposes no raw emissions/tokenizer; "context" cannot run."""
    with pytest.raises(ValueError, match="context"):
        ASRConfig(second_pass="context")


def test_make_second_pass_degrades_gracefully_without_transcribe():
    class BareBackend:
        def create_stream(self):
            return None

    assert make_second_pass(BareBackend(), ASRConfig(second_pass="greedy")) is None


def test_config_rejects_invalid_second_pass_values():
    with pytest.raises(ValueError, match="second_pass"):
        ASRConfig(second_pass="beam-only")
