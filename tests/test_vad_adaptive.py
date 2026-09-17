"""Adaptive noise-floor VAD: bounded thresholds across realistic conditions.

These are behavioural regression tests, not threshold tuning: the assertions
are about *segmentation outcomes and bounds* (did the phrase survive as one
segment, did the floor stay inside its configured clamp), never about a
particular RMS value fitted to a synthetic signal.  Real thresholds must still
be set from measured recordings.
"""
import numpy as np
import pytest

from shenava_realtime.vad import EnergyVAD, EventType, VADConfig
from tests.test_vad_scenarios import (
    FRAME,
    blocks_of,
    segments,
    silence_seconds,
    speech_seconds,
)

STATIC = VADConfig(adaptive=False)


def run(blocks, config=None, flush=True):
    vad = EnergyVAD(config or VADConfig())
    events = []
    for block in blocks:
        events.extend(vad.process(block))
    if flush:
        events.extend(vad.flush())
    return vad, events


def durations(events):
    return [e.duration_s for e in segments(events)]


# --------------------------------------------------------------------------- #
# Configuration bounds
# --------------------------------------------------------------------------- #
def test_adaptive_settings_are_validated():
    with pytest.raises(ValueError, match="SNR"):
        VADConfig(onset_snr=0.0)
    with pytest.raises(ValueError, match="hysteresis"):
        VADConfig(onset_snr=2.0, offset_snr=4.0)
    with pytest.raises(ValueError, match="adaptive_max_gain"):
        VADConfig(adaptive_max_gain=0.5)
    with pytest.raises(ValueError, match="noise_init_ms"):
        VADConfig(noise_init_ms=0)
    with pytest.raises(ValueError, match="noise_halflife_ms"):
        VADConfig(noise_halflife_ms=0)


def test_static_mode_ignores_adaptive_settings_entirely():
    vad = EnergyVAD(VADConfig(adaptive=False))
    for block in blocks_of(silence_seconds(2.0, noise=0.004, seed=1)):
        vad.process(block)
    assert vad.onset_threshold == vad.config.onset_rms
    assert vad.offset_threshold == vad.config.offset_rms


def test_thresholds_stay_inside_the_configured_clamp():
    config = VADConfig(adaptive_max_gain=4.0)
    vad, _ = run(blocks_of(silence_seconds(4.0, noise=0.05, seed=2)), config, flush=False)
    assert config.onset_rms <= vad.onset_threshold <= config.onset_rms * 4.0
    assert config.offset_rms <= vad.offset_threshold <= config.offset_rms * 4.0
    assert vad.offset_threshold <= vad.onset_threshold  # hysteresis preserved


def test_noise_floor_is_only_estimated_from_background():
    """Loud speech must never drag the floor up under the speaker."""
    vad, _ = run(blocks_of(
        silence_seconds(1.0, noise=0.001, seed=3),
        speech_seconds(3.0, 0.08, seed=4),
    ), flush=False)
    assert vad.noise_floor < 0.01


# --------------------------------------------------------------------------- #
# Speaker/environment scenarios
# --------------------------------------------------------------------------- #
def test_quiet_speaker_in_a_quiet_room_is_still_captured():
    blocks = blocks_of(
        silence_seconds(1.0, noise=0.0005, seed=5),
        speech_seconds(1.5, 0.02, seed=6),
        silence_seconds(1.2, noise=0.0005, seed=7),
    )
    assert len(segments(run(blocks)[1])) == 1


def test_normal_speaker_matches_the_static_detector():
    blocks = blocks_of(
        silence_seconds(1.0),
        speech_seconds(1.5, 0.05, seed=8),
        silence_seconds(1.2),
    )
    adaptive = durations(run(blocks)[1])
    static = durations(run(blocks, STATIC)[1])
    assert len(adaptive) == len(static) == 1
    assert adaptive[0] == pytest.approx(static[0], abs=0.15)


def test_noisy_environment_raises_the_thresholds_above_the_static_ones():
    """Steady room noise lifts the floor, so the detector demands more level."""
    vad, _events = run(blocks_of(silence_seconds(3.0, noise=0.012, seed=9)), flush=False)
    assert vad.noise_floor > vad.config.offset_rms
    assert vad.onset_threshold > vad.config.onset_rms
    assert vad.offset_threshold > vad.config.offset_rms


def test_noise_that_already_exceeds_the_static_onset_is_not_rescued():
    """Documented limit: noise louder than the static onset reads as speech.

    The floor is only measured in SILENCE, so a room already above the
    configured onset needs its static thresholds raised by measurement — the
    adaptive layer bounds and refines, it does not replace tuning.
    """
    vad, events = run(blocks_of(silence_seconds(3.0, noise=0.02, seed=9)), flush=False)
    assert vad.onset_threshold == vad.config.onset_rms
    assert len(segments(events)) <= 1  # bounded, not a stream of fragments


def test_speech_over_a_raised_floor_is_still_one_segment():
    blocks = blocks_of(
        silence_seconds(2.0, noise=0.012, seed=30),
        speech_seconds(1.5, 0.06, seed=31, noise=0.012),
        silence_seconds(2.0, noise=0.012, seed=32),
    )
    vad, events = run(blocks, flush=False)
    assert vad.onset_threshold > vad.config.onset_rms
    segs = segments(events)
    assert len(segs) == 1 and not segs[0].forced


def test_low_snr_speech_over_noise_is_one_segment():
    blocks = blocks_of(
        silence_seconds(1.0, noise=0.006, seed=10),
        speech_seconds(1.6, 0.05, seed=11, noise=0.006),
        silence_seconds(1.2, noise=0.006, seed=12),
    )
    segs = segments(run(blocks)[1])
    assert len(segs) == 1 and not segs[0].forced


def test_short_hesitation_does_not_split_the_utterance():
    blocks = blocks_of(
        silence_seconds(0.8),
        speech_seconds(1.0, 0.05, seed=13),
        silence_seconds(0.3),                  # < min_silence_ms (700)
        speech_seconds(1.0, 0.05, seed=14),
        silence_seconds(1.2),
    )
    assert len(segments(run(blocks)[1])) == 1


def test_long_pause_splits_and_never_joins_audio_across_it():
    blocks = blocks_of(
        silence_seconds(0.8),
        speech_seconds(1.2, 0.05, seed=15),
        silence_seconds(1.5),                  # >= min_silence_ms
        speech_seconds(1.2, 0.05, seed=16),
        silence_seconds(1.5),
    )
    segs = segments(run(blocks)[1])
    assert len(segs) == 2
    # Each segment is bounded by its own speech plus pre-roll and hangover.
    assert all(s.duration_s < 1.2 + 0.32 + 0.7 + 0.2 for s in segs)


def test_speech_beginning_before_the_floor_is_confirmed_is_not_lost():
    """No seeded background at all: the static thresholds must apply."""
    blocks = blocks_of(speech_seconds(1.5, 0.05, seed=17), silence_seconds(1.2))
    vad, events = run(blocks)
    assert len(segments(events)) == 1
    assert vad.onset_threshold == vad.config.onset_rms


def test_speech_ending_near_the_threshold_keeps_the_tail():
    """A decaying word ending in the hysteresis band must not be truncated."""
    tail = np.linspace(0.05, 0.009, int(0.6 * 16000)).astype(np.float32)
    blocks = blocks_of(
        silence_seconds(1.0),
        speech_seconds(1.0, 0.05, seed=18),
        tail,
        silence_seconds(1.2),
    )
    segs = segments(run(blocks)[1])
    assert len(segs) == 1
    assert segs[0].duration_s > 1.5  # the decaying tail is inside the segment


def test_continuous_dictation_stays_one_segment_until_the_cap():
    blocks = blocks_of(
        silence_seconds(0.8),
        speech_seconds(12.0, 0.05, seed=19),
        silence_seconds(1.2),
    )
    segs = segments(run(blocks)[1])
    assert len(segs) == 1 and not segs[0].forced


def test_the_hard_segment_cap_still_applies_under_adaptation():
    config = VADConfig(max_speech_s=3.0)
    blocks = blocks_of(silence_seconds(0.5), speech_seconds(8.0, 0.05, seed=20))
    segs = segments(run(blocks, config)[1])
    assert len(segs) >= 2
    assert segs[0].forced is True                    # cap cut, not a phrase end
    assert all(s.duration_s <= 3.05 for s in segs)   # bounded


def test_pre_roll_and_minimum_durations_are_retained():
    config = VADConfig(pre_speech_ms=320, min_speech_ms=250, min_silence_ms=700)
    blocks = blocks_of(
        silence_seconds(1.0, noise=0.001, seed=21),
        speech_seconds(1.0, 0.05, seed=22),
        silence_seconds(1.2, noise=0.001, seed=23),
    )
    segs = segments(run(blocks, config)[1])
    assert len(segs) == 1
    # speech + pre-roll + the silence hangover, and nothing unbounded.
    assert 1.0 <= segs[0].duration_s <= 1.0 + 0.32 + 0.7 + 0.15


def test_a_click_shorter_than_min_speech_is_dropped():
    blocks = blocks_of(
        silence_seconds(1.0, noise=0.001, seed=24),
        speech_seconds(0.08, 0.09, seed=25),   # < min_speech_ms
        silence_seconds(1.5, noise=0.001, seed=26),
    )
    assert segments(run(blocks)[1]) == []


def test_reset_restores_the_initial_thresholds():
    vad, _ = run(blocks_of(silence_seconds(3.0, noise=0.012, seed=27)), flush=False)
    assert vad.onset_threshold > vad.config.onset_rms
    vad.reset()
    assert vad.onset_threshold == vad.config.onset_rms
    assert vad.offset_threshold == vad.config.offset_rms
    assert vad.state.value == "silence"


def test_events_still_carry_audio_and_forced_flags():
    _vad, events = run(blocks_of(
        silence_seconds(0.8), speech_seconds(1.0, 0.05, seed=28), silence_seconds(1.2)
    ))
    starts = [e for e in events if e.type == EventType.SPEECH_START]
    ends = segments(events)
    assert starts and ends
    assert isinstance(ends[0].audio, np.ndarray) and ends[0].audio.size > FRAME
    assert ends[0].forced is False
