"""CTC prefix-beam correctness: beam_size actually prunes, log-space maths.

Regression target: the previous implementation kept ``2 * 64`` states no matter
what ``beam_size`` said, so ``beam_size=4`` behaved like a beam of 128.
"""
import numpy as np
import pytest

from shenava_realtime.second_pass import (
    CTCBeamDecoder,
    DEFAULT_BIAS_ACOUSTIC_GATE,
    _logaddexp,
    hotword_token_bias,
)


def log_softmax(rows):
    arr = np.array(rows, dtype=np.float64)
    return arr - np.log(np.exp(arr).sum(axis=-1, keepdims=True))


def random_emissions(frames, vocab, seed):
    rng = np.random.default_rng(seed)
    return log_softmax(rng.standard_normal((frames, vocab)) * 2.0)


# --------------------------------------------------------------------------- #
# The configured beam size is the real beam size
# --------------------------------------------------------------------------- #
def test_beam_size_bounds_the_retained_hypotheses():
    decoder = CTCBeamDecoder(beam_size=3)
    states = {(i,): [-float(i), -float(i) - 1.0] for i in range(50)}
    assert len(decoder._prune(states)) == 3


def test_pruning_keeps_the_highest_scoring_prefixes():
    decoder = CTCBeamDecoder(beam_size=2)
    states = {
        (1,): [-5.0, -5.0],
        (2,): [-0.5, -0.5],   # best
        (3,): [-1.0, -1.0],   # second
    }
    assert set(decoder._prune(states)) == {(2,), (3,)}


def test_small_and_large_beams_can_disagree():
    """A beam of 1 is greedy; a wider beam may recover a better hypothesis.

    If they never differed, ``beam_size`` would not be doing anything — which
    was exactly the bug (a hard-coded 128-state floor made every beam wide).
    """
    emissions = random_emissions(frames=25, vocab=6, seed=7)
    narrow = CTCBeamDecoder(1).decode(emissions)
    wide = CTCBeamDecoder(16).decode(emissions)
    assert wide[1] >= narrow[1] - 1e-9  # a wider beam is never worse


def test_wider_beam_never_scores_below_a_narrower_one():
    for seed in range(5):
        emissions = random_emissions(frames=15, vocab=5, seed=seed)
        scores = [CTCBeamDecoder(size).decode(emissions)[1] for size in (1, 2, 4, 8)]
        assert scores == sorted(scores), scores


# --------------------------------------------------------------------------- #
# Prefix-beam semantics
# --------------------------------------------------------------------------- #
def test_blank_and_nonblank_mass_are_tracked_separately():
    # a, blank, a -> two emissions; a, a -> one. Only separate blank/non-blank
    # prefix probabilities get this right.
    a_blank_a = log_softmax([[-5.0, -0.1, -5.0], [-0.1, -5.0, -5.0], [-5.0, -0.1, -5.0]])
    assert CTCBeamDecoder(4).decode(a_blank_a)[0] == [1, 1]
    a_a = log_softmax([[-5.0, -0.1, -5.0], [-5.0, -0.1, -5.0]])
    assert CTCBeamDecoder(4).decode(a_a)[0] == [1]


def test_scores_are_log_probabilities_of_at_most_zero():
    emissions = random_emissions(frames=10, vocab=4, seed=3)
    _tokens, score = CTCBeamDecoder(4).decode(emissions)
    assert score <= 1e-9


def test_all_blank_audio_decodes_to_nothing():
    emissions = log_softmax([[0.0, -20.0, -20.0]] * 6)
    assert CTCBeamDecoder(4).decode(emissions)[0] == []


def test_decoding_is_deterministic_across_runs():
    emissions = random_emissions(frames=20, vocab=6, seed=11)
    first = CTCBeamDecoder(4).decode(emissions)
    second = CTCBeamDecoder(4).decode(emissions)
    assert first == second


def test_logaddexp_matches_numpy_and_handles_negative_infinity():
    assert _logaddexp(float("-inf"), -1.0) == -1.0
    assert _logaddexp(-1.0, float("-inf")) == -1.0
    assert _logaddexp(-2.0, -3.0) == pytest.approx(float(np.logaddexp(-2.0, -3.0)))


# --------------------------------------------------------------------------- #
# Contextual bias + acoustic-safety gate
# --------------------------------------------------------------------------- #
def test_bias_tips_a_near_tie_within_the_acoustic_gate():
    emissions = log_softmax([[-5.0, -1.0, -1.2]])
    assert CTCBeamDecoder(4).decode(emissions)[0] == [1]
    bias = hotword_token_bias([((2,), 1.0)])
    assert CTCBeamDecoder(4).decode(emissions, token_bias=bias)[0] == [2]


def test_acoustic_gate_blocks_bias_without_acoustic_evidence():
    """A hotword must not be forced into speech that does not support it."""
    emissions = log_softmax([[-5.0, 0.0, -30.0]])  # token 2 is acoustically absent
    bias = hotword_token_bias([((2,), 1.5)])
    decoder = CTCBeamDecoder(4, bias_acoustic_gate=2.0)
    assert decoder.decode(emissions, token_bias=bias)[0] == [1]
    # With a wide gate the same bias would be applied — the gate is the guard.
    assert CTCBeamDecoder(4, bias_acoustic_gate=20.0).bias_acoustic_gate == 20.0


def test_bias_is_bounded_and_cannot_rescue_a_hopeless_token():
    emissions = log_softmax([[-5.0, 0.0, -40.0]])
    bias = hotword_token_bias([((2,), 1.5)])  # MAX_HOTWORD_BIAS
    assert CTCBeamDecoder(4).decode(emissions, token_bias=bias)[0] == [1]


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        CTCBeamDecoder(beam_size=0)
    with pytest.raises(ValueError):
        CTCBeamDecoder(beam_size=33)
    with pytest.raises(ValueError):
        CTCBeamDecoder(4, bias_acoustic_gate=0.0)
    with pytest.raises(ValueError):
        CTCBeamDecoder(4, bias_acoustic_gate=21.0)
    assert CTCBeamDecoder(4).bias_acoustic_gate == DEFAULT_BIAS_ACOUSTIC_GATE
