"""Second-pass decoding: CTC beam, hotword biasing, pipeline integration."""
import numpy as np
import pytest

from shenava_realtime.config import ASRConfig
from shenava_realtime.hotwords import build_hotwords, BIAS_BY_CATEGORY
from shenava_realtime.pipeline import TranscriptionPipeline
from shenava_realtime.second_pass import (
    BeamSecondPass,
    CTCBeamDecoder,
    GreedySecondPass,
    Hotword,
    SecondPassUtterance,
    hotword_token_bias,
)
from shenava_realtime.streaming import DecodeResult, StreamingDecoder, make_second_pass
from shenava_realtime.terminology import TerminologyRule
from tests.fakes import FakeBackend


def make_emissions(frame_probs):
    """Log-probs [T, V] from per-frame distributions (deterministic)."""
    arr = np.array(frame_probs, dtype=np.float64)
    return arr - np.log(np.exp(arr).sum(axis=-1, keepdims=True))


# --------------------------------------------------------------------------- #
# CTC beam decoder (pure, deterministic)
# --------------------------------------------------------------------------- #
def test_beam_decoder_is_greedy_without_bias():
    # One frame favouring token 1, one favouring token 2 (blank is 0).
    emissions = make_emissions([[ -1.0, -0.1, -5.0], [-1.0, -5.0, -0.1]])
    tokens, score = CTCBeamDecoder(beam_size=2).decode(emissions)
    assert tokens == [1, 2]
    assert score < 0.0


def test_beam_decoder_uses_ctc_collapse_semantics():
    # Consecutive repeats without a blank collapse to a single emission...
    emissions = make_emissions([[-1.0, -0.1, -5.0], [-1.0, -0.1, -5.0]])
    assert CTCBeamDecoder(4).decode(emissions)[0] == [1]
    # ...but a blank between repeats separates them, keeping both.
    emissions = make_emissions([
        [-1.0, -0.1, -5.0],   # a
        [-0.1, -5.0, -5.0],   # blank
        [-1.0, -0.1, -5.0],   # a
    ])
    assert CTCBeamDecoder(4).decode(emissions)[0] == [1, 1]


def test_beam_decoder_is_deterministic():
    rng = np.random.default_rng(1)
    base = np.array([-0.4, -0.2, -0.7])
    emissions = make_emissions(base + 0.01 * rng.standard_normal((20, 3)))
    a = CTCBeamDecoder(beam_size=4).decode(emissions)
    b = CTCBeamDecoder(beam_size=4).decode(emissions)
    assert a == b
    # Ties between non-empty decodes break to the lexicographically smaller
    # sequence (blank strictly worse here).
    tie = make_emissions([[-2.0, -1.0, -1.0]])
    assert CTCBeamDecoder(2).decode(tie)[0] == [1]


def test_beam_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        CTCBeamDecoder(beam_size=0)
    with pytest.raises(ValueError):
        CTCBeamDecoder(beam_size=33)
    with pytest.raises(ValueError):
        CTCBeamDecoder().decode(np.zeros((3, 1)))
    with pytest.raises(ValueError):
        CTCBeamDecoder().decode(np.zeros((3, 3)), blank=5)


def test_hotword_bias_tips_a_near_tie():
    # Token 2 is 0.15 worse than token 1; a bias of 0.5 for token 2 flips it.
    emissions = make_emissions([[-1.05, -1.0, -1.15]])
    assert CTCBeamDecoder(2).decode(emissions)[0] == [1]
    bias = hotword_token_bias([((2,), 0.5)])
    assert CTCBeamDecoder(2).decode(emissions, token_bias=bias)[0] == [2]


def test_hotword_prefix_bias_only_applies_on_the_hotword_path():
    # Hotword "1 2" biased +1.0: a path that never starts with 1 gets nothing.
    emissions = make_emissions([
        [-1.0, -1.0, -1.05],  # frame 0
        [-1.0, -5.0, -0.2],   # frame 1
    ])
    bias = hotword_token_bias([((1, 2), 1.0)])
    tokens, _ = CTCBeamDecoder(2).decode(emissions, token_bias=bias)
    assert tokens == [1, 2]  # the biased prefix wins despite frame-0 tie


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


def test_beam_second_pass_with_synthetic_emissions():
    # Vocab: 0 blank, 1 "متن", 2 "بیمار".  Token 2 is 0.1 log-prob worse;
    # the hotword "بیمار" (+1.0) must tip the decode.
    def emissions_fn(audio):
        return make_emissions([[-1.05, -1.0, -1.15]])

    pass_obj = BeamSecondPass(
        emissions_fn,
        tokenize=lambda text: [2 if text == "بیمار" else 1 if text == "متن" else []],
        decode_tokens=lambda ids: " ".join({1: "متن", 2: "بیمار"}[i] for i in ids),
        blank_index=0,
        beam_size=2,
        greedy=GreedySecondPass(lambda a: ("متن بیمار", 0.0)),
    )
    utterance = SecondPassUtterance(np.zeros(1600, np.float32), 0.1)
    assert pass_obj.decode_with_context(utterance, [Hotword("بیمار", 1.0)]) == "بیمار"
    # Without hotwords the unbiased beam keeps the (slightly) better token.
    assert pass_obj.decode_with_context(utterance, []) == "متن"
    # A hotword whose tokens are not in the vocabulary cannot change anything.
    assert pass_obj.decode_with_context(utterance, [Hotword("ناشناخته", 1.0)]) == "متن"


def test_beam_second_pass_failure_semantics():
    utterance = SecondPassUtterance(np.zeros(1600, np.float32), 0.1)

    def broken_emissions(audio):
        raise RuntimeError("no CTC logits")

    pass_obj = BeamSecondPass(
        broken_emissions,
        tokenize=lambda text: [1],
        decode_tokens=lambda ids: "متن",
        blank_index=0,
        greedy=GreedySecondPass(lambda a: ("بیمار", 0.0)),
    )
    # Emission failure propagates so the pipeline records the fallback and
    # keeps the streaming text — it is never a silent greedy switch.
    with pytest.raises(RuntimeError):
        pass_obj.decode_with_context(utterance, [Hotword("بیمار", 1.0)])

    # A hotword the tokenizer cannot handle is skipped: the beam runs unbiased.
    pass_obj = BeamSecondPass(
        lambda a: make_emissions([[-1.05, -1.0, -1.15]]),
        tokenize=lambda text: [],
        decode_tokens=lambda ids: " ".join({1: "متن", 2: "بیمار"}[i] for i in ids),
        blank_index=0,
        greedy=GreedySecondPass(lambda a: ("بیمار", 0.0)),
    )
    assert pass_obj.decode_with_context(utterance, [Hotword("بیمار", 1.0)]) == "متن"

    # An all-blank beam degrades to the greedy offline pass.
    pass_obj = BeamSecondPass(
        lambda a: make_emissions([[-0.1, -5.0, -5.0]]),
        tokenize=lambda text: [1],
        decode_tokens=lambda ids: "متن",
        blank_index=0,
        greedy=GreedySecondPass(lambda a: ("بیمار", 0.0)),
    )
    assert pass_obj.decode_with_context(utterance, [Hotword("بیمار", 1.0)]) == "بیمار"


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


def test_make_second_pass_context_requires_a_capability():
    with pytest.raises(RuntimeError, match="context"):
        make_second_pass(FakeBackend(), ASRConfig(second_pass="context"))


def test_make_second_pass_context_refuses_a_capability_poor_backend():
    class BareBackend:
        def create_stream(self):
            return None

    # Greedy degrades gracefully (warning + None)...
    assert make_second_pass(BareBackend(), ASRConfig(second_pass="greedy")) is None
    # ...but a specifically requested context mode is a startup error.
    with pytest.raises(RuntimeError, match="context"):
        make_second_pass(BareBackend(), ASRConfig(second_pass="context"))


def test_config_rejects_invalid_second_pass_values():
    with pytest.raises(ValueError, match="second_pass"):
        ASRConfig(second_pass="beam-only")
    with pytest.raises(ValueError):
        ASRConfig(second_pass_beam_size=0)
    with pytest.raises(ValueError):
        ASRConfig(hotword_max=0)
    with pytest.raises(ValueError):
        ASRConfig(hotword_specialty="  ")


def test_hotword_list_is_bounded_and_specialty_filtered():
    rules = [
        TerminologyRule("drug.1", "متفورمین", ("متفورمین",), category="medication",
                        specialty="general", priority=50),
        TerminologyRule("term.cardio", "استنوز", ("استنوز",), category="medical_term",
                        specialty="cardiology", priority=95),
        TerminologyRule("term.pulmo", "تنگی نفس", ("تنگی نفس",), category="symptom",
                        specialty="pulmonology", priority=50),
        TerminologyRule("unit.mg", "mg", ("میلی گرم",), category="unit", priority=80),
        TerminologyRule("anatomy.knee", "زانو", ("زانو",), category="anatomy", priority=50),
    ]
    general = build_hotwords(rules)
    assert [h.phrase for h in general] == ["متفورمین"]  # units/anatomy never boosted
    assert general[0].bias == BIAS_BY_CATEGORY["medication"]

    cardiology = build_hotwords(rules, specialty="cardiology")
    assert {h.phrase for h in cardiology} == {"استنوز", "متفورمین"}
    # Highest priority first, then general.
    assert cardiology[0].phrase == "استنوز"

    all_bounded = build_hotwords(rules, max_hotwords=1)
    assert len(all_bounded) == 1
    with pytest.raises(ValueError):
        build_hotwords(rules, max_hotwords=0)
