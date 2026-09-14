"""End-to-end engine test with a stub model and a stub microphone."""

import time

import numpy as np
import pytest

from shenava_realtime.config import AppConfig
from shenava_realtime.realtime_engine import RealtimeASR
from tests.fakes import FakeAudioCapture, FakeBackend, RecordingTranscriber

SR = 16000
PHRASE = "بیمار تحت عمل کابج"


def blocks(seconds: float, block_ms: int = 64) -> list:
    size = int(round(block_ms / 1000.0 * SR))
    return [np.zeros(size, dtype=np.float32) for _ in range(int(seconds * 1000 / block_ms))]


def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture()
def harness():
    config = AppConfig()
    config.save_transcripts = False
    config.asr.partial_interval_s = 0.4
    config.asr.left_context_s = 1.0
    config.asr.max_window_s = 6.0
    capture = FakeAudioCapture()
    backend = FakeBackend(RecordingTranscriber(lambda seconds: PHRASE if seconds >= 0.3 else ""))
    engine = RealtimeASR(config, backend=backend, audio_capture=capture)

    deltas: list[str] = []
    partials: list[str] = []
    utterances: list[str] = []
    engine.on_text_delta = lambda text, confidence: deltas.append(text)
    engine.on_partial = lambda text, confidence: partials.append(text)
    engine.on_utterance_end = lambda text, confidence: utterances.append(text)
    return engine, capture, deltas, partials, utterances


def test_utterance_is_transcribed_and_emitted_once(harness):
    engine, capture, deltas, partials, utterances = harness
    engine.start()
    try:
        capture.emit_utterance(blocks(3.0))
        assert wait_for(lambda: len(utterances) == 1), "the utterance was not finalised"
    finally:
        engine.stop()

    assert "".join(deltas) == "بیمار تحت عمل CABG"
    assert utterances == ["بیمار تحت عمل CABG"]
    assert partials, "partial updates should reach the overlay"
    words = "".join(deltas).split()
    assert len(words) == len(set(words)), "a word was emitted twice"


def test_two_utterances_are_kept_apart(harness):
    engine, capture, deltas, partials, utterances = harness
    engine.start()
    try:
        capture.emit_utterance(blocks(2.0))
        assert wait_for(lambda: len(utterances) == 1)
        capture.emit_utterance(blocks(2.0))
        assert wait_for(lambda: len(utterances) == 2)
    finally:
        engine.stop()

    assert len(utterances) == 2
    assert "".join(deltas) == "بیمار تحت عمل CABG بیمار تحت عمل CABG"
    assert engine.transcript == "بیمار تحت عمل CABG بیمار تحت عمل CABG"


def test_audio_outside_speech_is_ignored(harness):
    engine, capture, deltas, partials, utterances = harness
    engine.start()
    try:
        for block in blocks(1.0):
            capture.on_audio(block, False)  # not speech
        time.sleep(0.2)
    finally:
        engine.stop()
    assert deltas == []
    assert utterances == []


def test_clear_transcript_resets_state(harness):
    engine, capture, deltas, partials, utterances = harness
    engine.start()
    try:
        capture.emit_utterance(blocks(2.0))
        assert wait_for(lambda: len(utterances) == 1)
        engine.clear_transcript()
        assert wait_for(lambda: capture.cleared == 1)
        assert engine.transcript == ""
    finally:
        engine.stop()


def test_stop_is_clean_and_idempotent(harness):
    engine, capture, *_ = harness
    engine.start()
    engine.stop()
    engine.stop()  # second call must be a no-op
    assert capture.stopped == 1
    assert engine.is_running is False


def test_statistics_report_the_decoder_and_counts(harness):
    engine, capture, deltas, partials, utterances = harness
    engine.start()
    try:
        capture.emit_utterance(blocks(2.0))
        assert wait_for(lambda: len(utterances) == 1)
    finally:
        engine.stop()

    stats = engine.get_statistics()
    assert stats["decoder"] == "windowed"
    assert stats["utterances"] == 1
    assert stats["decodes"] >= 1
    assert stats["average_confidence"] > 0.0
    assert stats["total_audio_duration"] > 0.0


def test_a_backend_crash_does_not_kill_the_worker():
    config = AppConfig()
    config.save_transcripts = False
    config.asr.partial_interval_s = 0.4
    capture = FakeAudioCapture()
    calls = {"count": 0}

    def flaky(audio):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("decoder exploded")
        return PHRASE, 0.9

    engine = RealtimeASR(config, backend=FakeBackend(flaky), audio_capture=capture)
    utterances = []
    engine.on_utterance_end = lambda text, confidence: utterances.append(text)

    engine.start()
    try:
        capture.emit_utterance(blocks(3.0))
        assert wait_for(lambda: len(utterances) == 1), "the worker died after the first failure"
    finally:
        engine.stop()
    assert engine.is_running is False
    assert calls["count"] >= 2
    assert utterances == ["بیمار تحت عمل CABG"]
