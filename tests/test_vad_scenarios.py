"""VAD boundary regression coverage over realistic dictation scenarios.

All scenarios are deterministic synthetic envelopes (bounded overlap audio,
mono float32, 16 kHz).  They pin the hysteresis behaviour: short phrases,
continuous dictation, short/long pauses, low volume, noise floors and the
segment-cap cut.  No denoising happens here — the VAD only gates segments.
"""
import numpy as np
import pytest

from shenava_realtime.vad import EnergyVAD, EventType, VADConfig

SAMPLE_RATE = 16000
FRAME_MS = 64
FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 1024


def speech_seconds(seconds: float, amplitude: float, seed: int = 0,
                   noise: float = 0.0, syllable_hz: float = 4.0) -> np.ndarray:
    """Syllable-modulated speech envelope with optional stationary noise."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    env = 0.65 + 0.35 * np.sin(2 * np.pi * syllable_hz * t)
    env = np.clip(env, 0.25, 1.0)
    x = amplitude * env * (0.75 + 0.25 * rng.random(n))
    if noise:
        x = x + rng.normal(0.0, noise, n)
    return x.astype(np.float32)


def silence_seconds(seconds: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    if noise == 0.0:
        return np.zeros(int(round(seconds * SAMPLE_RATE)), dtype=np.float32)
    return (np.random.default_rng(seed).normal(0.0, noise, int(round(seconds * SAMPLE_RATE)))
            .astype(np.float32))


def run_vad(blocks, config=None, flush=True):
    vad = EnergyVAD(config or VADConfig())
    events = []
    for block in blocks:
        events.extend(vad.process(block))
    if flush:
        events.extend(vad.flush())
    return events


def blocks_of(*pieces) -> list[np.ndarray]:
    out = []
    for piece in pieces:
        for start in range(0, piece.size, FRAME):
            out.append(piece[start:start + FRAME])
    return out


def segments(events):
    return [e for e in events if e.type == EventType.SPEECH_END]


def test_short_phrases_produce_one_segment_each():
    events = run_vad(blocks_of(
        speech_seconds(0.8, 0.05, seed=1),
        silence_seconds(1.5),
        speech_seconds(0.9, 0.05, seed=2),
        silence_seconds(1.5),
    ))
    segs = segments(events)
    assert len(segs) == 2
    assert not any(s.forced for s in segs)
    assert all(s.audio.size > 0 for s in segs)
    # Segment audio includes bounded pre-roll plus the phrase.
    assert 0.8 < segs[0].duration_s < 0.8 + 0.32 + 0.70
    assert 0.9 < segs[1].duration_s < 0.9 + 0.32 + 0.70


def test_continuous_dictation_is_a_single_segment():
    # A 4 s utterance whose syllable dips stay above the offset threshold.
    events = run_vad(blocks_of(speech_seconds(4.0, 0.03, seed=3),
                               silence_seconds(1.5)))
    segs = segments(events)
    assert len(segs) == 1
    assert segs[0].duration_s >= 3.8
    assert not segs[0].forced


def test_short_pause_does_not_split():
    events = run_vad(blocks_of(
        speech_seconds(1.5, 0.04, seed=4),
        silence_seconds(0.4),          # < min_silence_ms (700)
        speech_seconds(1.5, 0.04, seed=5),
        silence_seconds(1.5),
    ))
    assert len(segments(events)) == 1


def test_long_pause_splits_into_two_segments():
    events = run_vad(blocks_of(
        speech_seconds(1.5, 0.04, seed=6),
        silence_seconds(1.2),          # >= 700 ms
        speech_seconds(1.5, 0.04, seed=7),
        silence_seconds(1.5),
    ))
    segs = segments(events)
    assert len(segs) == 2
    # Audio is never joined across the silence: each segment is bounded.
    assert all(1.4 < s.duration_s < 1.5 + 0.32 + 0.70 + 0.2 for s in segs)


def test_low_volume_is_dropped_by_default_configuration():
    # Below the onset threshold the default RMS VAD treats this as silence:
    # that is documented, conservative behaviour (no forced speech).
    events = run_vad(blocks_of(speech_seconds(1.5, 0.012, seed=8),
                               silence_seconds(1.0)))
    assert segments(events) == []


def test_low_volume_is_captured_with_explicit_settings():
    tuned = VADConfig(onset_rms=0.010, offset_rms=0.006)
    events = run_vad(blocks_of(speech_seconds(1.5, 0.012, seed=8),
                               silence_seconds(1.0)), config=tuned)
    segs = segments(events)
    assert len(segs) == 1
    assert segs[0].duration_s >= 1.2


def test_stationary_noise_floor_does_not_create_segments():
    # Noise below both thresholds: no false speech.
    events = run_vad(blocks_of(silence_seconds(3.0, noise=0.003, seed=9)))
    assert segments(events) == []


def test_noise_below_offset_keeps_phrase_boundaries():
    # The phrase is surrounded by a noisy floor below the offset threshold,
    # so the segment still ends at the natural boundary (plus bounded tail).
    events = run_vad(blocks_of(
        speech_seconds(1.5, 0.04, seed=10, noise=0.002),
        silence_seconds(1.5, noise=0.003, seed=11),
    ))
    segs = segments(events)
    assert len(segs) == 1
    assert 1.4 < segs[0].duration_s < 1.5 + 0.32 + 0.70 + 0.2


def test_segment_cap_cut_is_marked_forced_and_keeps_going():
    # 7 s of unbroken speech with a 2 s cap: three artifact cuts (forced)
    # and the natural end.  Every speech sample lands in exactly one
    # segment, in order, with the bounded trailing silence on the last one.
    utterance = speech_seconds(7.0, 0.04, seed=12)
    tail = silence_seconds(1.5)
    short_max = VADConfig(max_speech_s=2.0)
    events = run_vad(blocks_of(utterance, tail), config=short_max)
    segs = segments(events)
    assert len(segs) == 4
    assert [s.forced for s in segs] == [True, True, True, False]
    joined = np.concatenate([s.audio for s in segs])
    expected = np.concatenate([utterance, tail])[: joined.size]
    tail_samples = -(-int(0.70 * SAMPLE_RATE) // 1024) * 1024  # 700 ms in 64 ms blocks
    assert joined.size == utterance.size + tail_samples
    np.testing.assert_array_equal(joined, expected)


def test_sub_min_silence_blip_stays_in_the_open_segment():
    # Hysteresis: a 0.15 s blip separated by sub-700 ms pauses cannot split
    # the open segment — but it also cannot start one on its own, so the
    # phrase before it remains a single bounded segment.
    events = run_vad(blocks_of(
        speech_seconds(1.0, 0.04, seed=13),
        silence_seconds(0.3),
        speech_seconds(0.15, 0.05, seed=14),
        silence_seconds(1.5),
    ))
    segs = segments(events)
    assert len(segs) == 1
    assert 1.0 < segs[0].duration_s < 1.0 + 0.3 + 0.15 + 0.32 + 0.70


def test_queue_discontinuity_never_joins_audio_across_the_gap():
    # A capture discontinuity (device dropout / dropped queue region) resets
    # the VAD, so audio before and after the gap can never merge into one
    # segment, and the engine is told to abort the open utterance.
    from shenava_realtime.audio_capture import AudioCapture
    from shenava_realtime.config import AudioConfig

    capture = AudioCapture(AudioConfig())
    discontinuities = []
    capture.on_discontinuity = lambda: discontinuities.append(1)
    for block in blocks_of(speech_seconds(1.2, 0.04, seed=15))[:10]:
        capture._consume(block)
    assert capture.vad.state.value == "speech"

    capture._discontinuity.set()
    capture._consume(np.zeros(1024, dtype=np.float32))
    assert len(discontinuities) == 1
    assert capture.vad.state.value == "silence"

    starts, ends = [], []
    capture.on_speech_start = starts.append
    capture.on_speech_end = lambda duration, forced: ends.append((duration, forced))
    capture._consume(np.zeros(1024, dtype=np.float32))  # pre-roll after the gap
    for block in blocks_of(speech_seconds(1.0, 0.04, seed=16)):
        capture._consume(block)
    for event in capture.vad.flush():
        capture._dispatch(event)
    assert len(starts) == 1 and len(ends) == 1
    # on_speech_end receives (duration_s, forced): the new segment is only
    # the post-gap phrase, so nothing is silently joined across the dropout.
    duration_s, forced = ends[0]
    assert not forced
    assert duration_s <= 1.0 + 0.32 + 0.70 + 0.01


def test_incomplete_utterance_is_closed_by_flush():
    vad = EnergyVAD(VADConfig())
    for block in blocks_of(speech_seconds(1.0, 0.04, seed=16))[:12]:
        vad.process(block)
    events = vad.flush()
    ends = [e for e in events if e.type == EventType.SPEECH_END]
    assert len(ends) == 1 and ends[0].audio.size > 0 and not ends[0].forced
