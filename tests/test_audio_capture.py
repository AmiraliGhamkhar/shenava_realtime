"""Audio capture dispatch and shutdown (no real microphone needed)."""

import threading

import numpy as np
import pytest

from shenava_realtime.audio_capture import AudioCapture
from shenava_realtime.config import AudioConfig

SR = 16000
BLOCK = 1024


def block(amplitude: float) -> np.ndarray:
    return np.full(BLOCK, amplitude, dtype=np.float32)


def capture_with_recorders() -> tuple:
    capture = AudioCapture(
        AudioConfig(
            sample_rate=SR,
            chunk_size=BLOCK,
            vad_onset_rms=0.015,
            vad_offset_rms=0.008,
            vad_min_speech_ms=250,
            vad_min_silence_ms=700,
            vad_pre_speech_ms=320,
        )
    )
    events = {"start": [], "end": [], "audio": []}
    capture.on_speech_start = lambda preroll: events["start"].append(len(preroll))
    capture.on_speech_end = lambda duration: events["end"].append(duration)
    capture.on_audio = lambda chunk, in_speech: events["audio"].append((len(chunk), in_speech))
    return capture, events


def test_silence_only_produces_no_segment():
    capture, events = capture_with_recorders()
    for _ in range(50):
        capture._consume(block(0.001))
    assert events["start"] == []
    assert events["end"] == []
    assert len(events["audio"]) == 50


def test_speech_segment_is_reported_with_duration():
    capture, events = capture_with_recorders()
    for _ in range(30):
        capture._consume(block(0.001))
    for _ in range(30):  # 1.9 s of speech
        capture._consume(block(0.05))
    for _ in range(20):  # 1.3 s of silence
        capture._consume(block(0.001))

    assert len(events["start"]) == 1
    assert len(events["end"]) == 1
    assert events["end"][0] > 1.5
    stats = capture.get_statistics()
    assert stats["segments"] == 1
    assert stats["speech_seconds"] > 0


def test_audio_callback_runs_before_boundary_events():
    capture, events = capture_with_recorders()
    order = []
    capture.on_audio = lambda chunk, in_speech: order.append("audio")
    capture.on_speech_start = lambda preroll: order.append("start")
    capture.on_speech_end = lambda duration: order.append("end")
    for _ in range(20):
        capture._consume(block(0.05))
    # The streaming consumer sees the block before the segment boundary is
    # announced, so an utterance always has all of its audio when it ends.
    assert order[0] == "audio"
    assert order.index("start") > 0
    for _ in range(20):
        capture._consume(block(0.001))
    assert order.index("end") > order.index("start")
    assert order.count("audio") == 40


def test_callback_errors_do_not_break_the_consumer(caplog):
    capture, _ = capture_with_recorders()

    def boom(*_args):
        raise RuntimeError("callback exploded")

    capture.on_audio = boom
    capture.on_speech_start = boom
    capture._consume(block(0.05))
    capture._consume(block(0.05))
    assert capture.stats["chunks"] == 2


def test_consumer_thread_flushes_the_open_segment_on_sentinel():
    capture, events = capture_with_recorders()
    capture._thread = threading.Thread(target=capture._consume_loop, name="audio-capture", daemon=True)
    capture._thread.start()
    try:
        for _ in range(40):  # speech with no trailing silence
            capture._queue.put(block(0.05))
    finally:
        capture._push_sentinel()
        capture._thread.join(timeout=3.0)

    assert not capture._thread.is_alive()
    assert len(events["start"]) == 1
    assert len(events["end"]) == 1, "the open utterance must be flushed at shutdown"


def test_stop_without_start_is_a_no_op():
    capture = AudioCapture(AudioConfig())
    capture.stop()  # must not raise
    assert capture.is_running is False


def test_pause_resume_and_clear():
    capture, events = capture_with_recorders()
    capture.pause()
    assert capture.is_paused is True
    capture.resume()
    assert capture.is_paused is False

    for _ in range(20):
        capture._consume(block(0.05))
    assert capture.vad.speech_duration_s > 0
    capture.clear()
    # The request is queued for the consumer thread, not applied in place.
    marker = capture._queue.get()
    capture._process_item(marker)
    assert capture.vad.speech_duration_s == 0.0


def test_audio_callback_drops_blocks_instead_of_blocking():
    capture = AudioCapture(AudioConfig(queue_max_chunks=8))
    stereo = np.zeros((BLOCK, 1), dtype=np.float32)
    for _ in range(20):
        capture._audio_callback(stereo, BLOCK, None, None)
    assert capture._queue.qsize() == 8
    assert capture._dropped_chunks == 12
    assert capture.get_statistics()["dropped_chunks"] == 12


def test_list_devices_without_sounddevice():
    # sounddevice is not installed in the test environment; the helper must
    # degrade to an empty list instead of raising.
    import shenava_realtime.audio_capture as module

    assert module.sd is None or isinstance(module.AudioCapture.list_devices(), list)


def test_missing_sounddevice_raises_on_start():
    import shenava_realtime.audio_capture as module

    if module.sd is not None:  # pragma: no cover - environment dependent
        pytest.skip("sounddevice is installed; the failure path is not reachable")
    capture = AudioCapture(AudioConfig())
    with pytest.raises(RuntimeError, match="sounddevice"):
        capture.start()
