"""Realtime robustness counters: overload has to be measurable, not inferred.

Every counter is asserted at the point the event happens, and the invariant
that matters clinically is re-checked here: audio is never joined across a
discontinuity, and decoding never continues as if missing audio existed.
"""
import time

import numpy as np
import pytest

from shenava_realtime.config import AppConfig
from shenava_realtime.realtime_engine import RealtimeASR
from tests.fakes import FakeAudioCapture, FakeBackend, RecordingTranscriber

BLOCK = 1024


def blocks(seconds, amplitude=0.05):
    count = int(seconds * 16000) // BLOCK
    return [np.full(BLOCK, amplitude, dtype=np.float32) for _ in range(count)]


def wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def make_engine(**asr_overrides):
    config = AppConfig()
    config.audio.queue_max_chunks = 128
    config.save_transcripts = False
    for key, value in asr_overrides.items():
        setattr(config.asr, key, value)
    config.asr.__post_init__()
    capture = FakeAudioCapture()
    backend = FakeBackend(RecordingTranscriber(lambda seconds: "بیمار تب دارد"))
    return RealtimeASR(config, backend=backend, audio_capture=capture), capture


# --------------------------------------------------------------------------- #
# Counters exist and start at zero
# --------------------------------------------------------------------------- #
REQUIRED_COUNTERS = (
    "queue_overflows",
    "dropped_items",
    "dropped_audio_seconds",
    "discontinuities",
    "decoder_errors",
    "forced_splits",
    "vad_dropouts",
)


def test_all_robustness_counters_are_reported_and_start_at_zero():
    engine, _capture = make_engine()
    stats = engine.get_statistics()
    for name in REQUIRED_COUNTERS:
        assert name in stats, name
        assert stats[name] == 0, name


def test_statistics_report_the_selected_head_and_second_pass():
    engine, _capture = make_engine(second_pass="off")
    stats = engine.get_statistics()
    assert stats["decoder_head"] == "ctc"       # CTC remains the default
    assert stats["second_pass"] == "off"
    assert "second_pass_stats" in stats


# --------------------------------------------------------------------------- #
# Queue overflow: counted, and the utterance is abandoned, not stitched
# --------------------------------------------------------------------------- #
def test_queue_overflow_is_counted_and_drops_are_measured_in_seconds():
    engine, _capture = make_engine()
    engine._queue.maxsize = 2
    # Fill past the bound without a worker draining it.
    engine._enqueue(("start", np.zeros(0, dtype=np.float32)))
    engine._enqueue(("audio", np.zeros(16000, dtype=np.float32)))
    engine._enqueue(("audio", np.zeros(16000, dtype=np.float32)))

    stats = engine.get_statistics()
    assert stats["queue_overflows"] >= 1
    assert stats["dropped_items"] >= 1
    assert stats["dropped_audio_seconds"] >= 1.0   # one second of audio was lost
    assert stats["discontinuities"] >= 1


def test_overflow_queues_a_reset_so_decoding_never_spans_the_gap():
    engine, _capture = make_engine()
    engine._queue.maxsize = 2
    engine._enqueue(("audio", np.zeros(1600, dtype=np.float32)))
    engine._enqueue(("audio", np.zeros(1600, dtype=np.float32)))
    engine._enqueue(("audio", np.zeros(1600, dtype=np.float32)))
    remaining = list(engine._queue.queue)
    assert remaining and remaining[0][0] == "clear"


def test_a_capture_discontinuity_is_counted_and_clears_the_pipeline():
    engine, capture = make_engine()
    engine._on_discontinuity()
    stats = engine.get_statistics()
    assert stats["discontinuities"] == 1
    assert stats["vad_dropouts"] == 1
    assert list(engine._queue.queue)[0][0] == "clear"


# --------------------------------------------------------------------------- #
# Forced splits and decoder errors
# --------------------------------------------------------------------------- #
def test_forced_segment_cuts_are_counted():
    engine, capture = make_engine()
    engine.start()
    try:
        capture.emit_utterance(blocks(1.0), forced=True)
        assert wait_for(lambda: engine.get_statistics()["forced_splits"] == 1)
    finally:
        engine.stop()
    assert engine.get_statistics()["forced_splits"] == 1


def test_natural_endpoints_are_not_counted_as_forced_splits():
    engine, capture = make_engine()
    engine.start()
    try:
        capture.emit_utterance(blocks(1.0), forced=False)
        assert wait_for(lambda: engine.get_statistics()["utterances"] == 1)
    finally:
        engine.stop()
    assert engine.get_statistics()["forced_splits"] == 0


def test_a_decoder_failure_is_counted_and_aborts_only_that_utterance():
    def explode(seconds):
        raise RuntimeError("decoder blew up")

    config = AppConfig()
    config.audio.queue_max_chunks = 128
    config.save_transcripts = False
    capture = FakeAudioCapture()
    engine = RealtimeASR(
        config, backend=FakeBackend(RecordingTranscriber(explode)), audio_capture=capture
    )
    engine.start()
    try:
        capture.emit_utterance(blocks(1.0))
        assert wait_for(lambda: engine.get_statistics()["decoder_errors"] >= 1)
    finally:
        engine.stop()
    stats = engine.get_statistics()
    assert stats["decoder_errors"] >= 1
    assert engine.transcript == ""       # nothing fabricated from the failure


# --------------------------------------------------------------------------- #
# Normal operation leaves the counters clean
# --------------------------------------------------------------------------- #
def test_a_clean_session_reports_no_drops_or_errors():
    engine, capture = make_engine()
    engine.start()
    try:
        capture.emit_utterance(blocks(1.0))
        assert wait_for(lambda: engine.get_statistics()["utterances"] == 1)
    finally:
        engine.stop()
    stats = engine.get_statistics()
    assert stats["queue_overflows"] == 0
    assert stats["dropped_items"] == 0
    assert stats["dropped_audio_seconds"] == 0.0
    assert stats["discontinuities"] == 0
    assert stats["decoder_errors"] == 0
