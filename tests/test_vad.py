"""VAD state transitions (hysteresis, min speech/silence, pre-roll, flush)."""

import numpy as np
import pytest

from shenava_realtime.vad import EnergyVAD, EventType, VADConfig, VADState, rms

SR = 16000
LOUD = 0.05  # above onset_rms
QUIET = 0.001  # below offset_rms
MIDDLE = 0.010  # below onset_rms, above offset_rms (hysteresis band)


def block(seconds: float, amplitude: float, block_ms: int = 64) -> list:
    samples = int(round(seconds * SR))
    size = int(round(block_ms / 1000.0 * SR))
    return [np.full(size, amplitude, dtype=np.float32) for _ in range(samples // size)]


def run(vad: EnergyVAD, chunks) -> list:
    events = []
    for chunk in chunks:
        events.extend(vad.process(chunk))
    return events


@pytest.fixture()
def vad() -> EnergyVAD:
    return EnergyVAD(
        VADConfig(
            sample_rate=SR,
            onset_rms=0.015,
            offset_rms=0.008,
            min_speech_ms=250,
            min_silence_ms=700,
            pre_speech_ms=320,
            max_speech_s=20.0,
        )
    )


def test_rms_helper():
    assert rms(np.full(100, 0.1, dtype=np.float32)) == pytest.approx(0.1, abs=1e-6)
    assert rms(np.zeros(10, dtype=np.float32)) == 0.0
    assert rms(np.zeros(0, dtype=np.float32)) == 0.0


def test_silence_produces_no_events(vad: EnergyVAD):
    assert run(vad, block(2.0, QUIET)) == []
    assert vad.state is VADState.SILENCE


def test_short_click_never_becomes_speech(vad: EnergyVAD):
    events = run(vad, block(0.1, LOUD))  # 100 ms < min_speech_ms
    events += run(vad, block(1.0, QUIET))
    assert events == []
    assert vad.state is VADState.SILENCE


def test_sustained_speech_starts_and_ends_a_segment(vad: EnergyVAD):
    run(vad, block(0.5, QUIET))
    events = run(vad, block(1.0, LOUD))
    starts = [event for event in events if event.type is EventType.SPEECH_START]
    assert len(starts) == 1
    assert vad.state is VADState.SPEECH

    end_events = run(vad, block(1.0, QUIET))
    ends = [event for event in end_events if event.type is EventType.SPEECH_END]
    assert len(ends) == 1
    assert ends[0].audio is not None
    assert ends[0].duration_s > 1.0
    assert vad.state is VADState.SILENCE


def test_pre_roll_keeps_the_start_of_the_utterance(vad: EnergyVAD):
    run(vad, block(1.0, QUIET))  # fills the pre-roll ring buffer
    events = run(vad, block(0.5, LOUD))
    start = next(event for event in events if event.type is EventType.SPEECH_START)
    # pre-roll (<=320 ms) + the speech that triggered the detector
    assert 0.5 < start.duration_s <= 0.5 + 0.32 + 0.01


def test_hysteresis_keeps_the_segment_alive_in_the_middle_band(vad: EnergyVAD):
    run(vad, block(0.5, LOUD))
    assert vad.state is VADState.SPEECH
    # Level drops between offset and onset: speech must not end.
    events = run(vad, block(1.0, MIDDLE))
    assert events == []
    assert vad.state is VADState.SPEECH


def test_brief_pause_does_not_end_the_segment(vad: EnergyVAD):
    run(vad, block(0.5, LOUD))
    events = run(vad, block(0.3, QUIET))  # shorter than min_silence_ms
    assert events == []
    assert vad.state is VADState.SPEECH
    events = run(vad, block(0.5, LOUD))
    assert vad.state is VADState.SPEECH


def test_silence_longer_than_min_silence_ends_the_segment(vad: EnergyVAD):
    run(vad, block(0.5, LOUD))
    events = run(vad, block(0.8, QUIET))
    assert [event.type for event in events] == [EventType.SPEECH_END]


def test_flush_closes_an_open_segment(vad: EnergyVAD):
    run(vad, block(0.6, LOUD))
    events = vad.flush()
    assert [event.type for event in events] == [EventType.SPEECH_END]
    assert events[0].duration_s >= 0.5
    assert vad.state is VADState.SILENCE


def test_flush_in_silence_is_a_no_op(vad: EnergyVAD):
    run(vad, block(0.5, QUIET))
    assert vad.flush() == []


def test_long_monologue_is_cut_without_losing_state(vad: EnergyVAD):
    vad.config.max_speech_s = 1.0
    events = run(vad, block(1.5, LOUD))
    types = [event.type for event in events]
    assert types.count(EventType.SPEECH_START) == 2
    assert types.count(EventType.SPEECH_END) >= 1
    assert vad.state is VADState.SPEECH  # still speaking after the forced cut


def test_reset_clears_state(vad: EnergyVAD):
    run(vad, block(0.5, LOUD))
    vad.reset()
    assert vad.state is VADState.SILENCE
    assert vad.speech_duration_s == 0.0


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        VADConfig(onset_rms=0.01, offset_rms=0.02)  # offset above onset
    with pytest.raises(ValueError):
        VADConfig(onset_rms=0.0)


def test_empty_frames_are_ignored(vad: EnergyVAD):
    assert vad.process(np.zeros(0, dtype=np.float32)) == []
    assert vad.state is VADState.SILENCE
