"""Audio capture dispatch and shutdown (no real microphone needed)."""

import threading
import time

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
    capture.on_speech_end = lambda duration, forced=False: events["end"].append(duration)
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
    capture.on_speech_end = lambda duration, forced=False: order.append("end")
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


def test_audio_capture_wires_adaptive_vad_disabled():
    capture = AudioCapture(AudioConfig(vad_adaptive=False))
    assert capture.vad.config.adaptive is False


def test_vad_adaptive_env_override_reaches_energy_vad(monkeypatch):
    from shenava_realtime.config import AppConfig, apply_env_overrides

    monkeypatch.setenv("SHENAVA_VAD_ADAPTIVE", "0")
    config = apply_env_overrides(AppConfig())
    capture = AudioCapture(config.audio)
    assert capture.vad.config.adaptive is False


def test_no_adaptive_vad_cli_reaches_energy_vad():
    from main import build_config, parse_args

    config = build_config(parse_args(["--no-adaptive-vad"]))
    capture = AudioCapture(config.audio)
    assert capture.vad.config.adaptive is False


def test_audio_capture_wires_adaptive_vad_tuning_fields():
    config = AudioConfig(
        vad_adaptive=True,
        vad_noise_init_ms=300,
        vad_noise_halflife_ms=1500,
        vad_onset_snr=3.0,
        vad_offset_snr=1.5,
        vad_adaptive_max_gain=4.0,
    )
    capture = AudioCapture(config)
    assert capture.vad.config.noise_init_ms == 300
    assert capture.vad.config.noise_halflife_ms == 1500
    assert capture.vad.config.onset_snr == 3.0
    assert capture.vad.config.offset_snr == 1.5
    assert capture.vad.config.adaptive_max_gain == 4.0


def test_microphone_level_diagnostics_do_not_alter_silent_audio(caplog):
    capture = AudioCapture(AudioConfig(vad_adaptive=False, chunk_size=BLOCK))
    observed = []
    capture.on_audio = lambda chunk, in_speech: observed.append(chunk.copy())
    sample = block(0.0)
    with caplog.at_level("WARNING"):
        for _ in range(34):  # > 2 seconds at 1024/16k
            capture._consume(sample)
    assert observed and all(np.array_equal(chunk, sample) for chunk in observed)
    assert "microphone frames are arriving but input level is very low" in caplog.text
    assert capture.stats["segments"] == 0


def test_first_useful_microphone_level_is_logged_once(caplog):
    capture = AudioCapture(AudioConfig(vad_adaptive=False, chunk_size=BLOCK))
    with caplog.at_level("INFO"):
        capture._consume(block(0.02))
        capture._consume(block(0.03))
    assert caplog.text.count("microphone input detected") == 1


def test_missing_sounddevice_raises_on_start():
    import shenava_realtime.audio_capture as module

    if module.sd is not None:  # pragma: no cover - environment dependent
        pytest.skip("sounddevice is installed; the failure path is not reachable")
    capture = AudioCapture(AudioConfig())
    with pytest.raises(RuntimeError, match="sounddevice"):
        capture.start()


# --------------------------------------------------------------------------- #
# Forced segment cuts carry an explicit flag (never look like natural ends).
# --------------------------------------------------------------------------- #
def test_forced_cut_reports_forced_true_natural_end_false():
    capture = AudioCapture(
        AudioConfig(
            sample_rate=SR,
            chunk_size=BLOCK,
            vad_min_speech_ms=64,
            vad_max_speech_s=1.0,  # above the 700 ms endpoint silence
            vad_pre_speech_ms=0,
        )
    )
    ends = []
    capture.on_speech_end = lambda duration, forced=False: ends.append(forced)
    for _ in range(20):  # 1.28 s of speech > 1.0 s cap -> forced cut
        capture._consume(block(0.05))
    assert ends == [True]
    for _ in range(16):  # 0.2 s residue + 0.7 s+ silence -> natural endpoint
        capture._consume(block(0.001))
    assert ends == [True, False]


# --------------------------------------------------------------------------- #
# Device dropout heartbeat: silence from the device is a visible error.
# --------------------------------------------------------------------------- #
def _live_capture(**overrides) -> AudioCapture:
    config = AudioConfig(sample_rate=SR, chunk_size=BLOCK, dropout_timeout_s=0.2, **overrides)
    capture = AudioCapture(config)
    capture._stream = type("FakeStream", (), {"active": True})()  # stream "open"
    with capture._lock:
        capture._last_block_time = time.monotonic()
    return capture


def test_recent_blocks_are_never_a_dropout():
    capture = _live_capture()
    capture._check_heartbeat()
    assert capture.stats.get("dropouts", 0) == 0
    assert capture.last_error is None


def test_silent_device_is_reported_exactly_once_and_aborts_open_utterance():
    capture = _live_capture()
    restarts = []
    capture.on_discontinuity = lambda: restarts.append(True)
    for _ in range(10):
        capture._consume(block(0.05))  # open a speech segment
    assert capture.is_speaking
    with capture._lock:
        capture._last_block_time -= 1.0  # simulate a long silent gap
    capture._check_heartbeat()
    assert capture.stats["dropouts"] == 1
    assert capture.last_error and "silent or disconnected" in capture.last_error
    assert capture.get_statistics()["last_error"] == capture.last_error
    assert not capture.is_speaking  # open utterance aborted, not stalled
    assert restarts == [True]
    capture._check_heartbeat()  # one report per dropout episode
    assert capture.stats["dropouts"] == 1


def test_heartbeat_ignores_paused_stopping_and_closed_stream():
    capture = _live_capture()
    with capture._lock:
        capture._last_block_time -= 60.0
    capture._paused = True
    capture._check_heartbeat()
    assert capture.stats.get("dropouts", 0) == 0
    capture._paused = False
    capture._stream = None
    capture._check_heartbeat()
    assert capture.stats.get("dropouts", 0) == 0
    capture._stream = type("FakeStream", (), {"active": True})()
    capture._stopping = True
    capture._check_heartbeat()
    assert capture.stats.get("dropouts", 0) == 0


def test_inactive_stream_is_reported_immediately():
    capture = _live_capture()
    capture._stream = type("FakeStream", (), {"active": False})()
    capture._check_heartbeat()
    assert capture.stats["dropouts"] == 1
    assert "inactive" in capture.last_error


def test_consumer_loop_reports_dropout_and_recovers():
    capture = _live_capture()
    capture.config.dropout_timeout_s = 0.15
    capture._thread = threading.Thread(target=capture._consume_loop, name="audio-capture", daemon=True)
    capture._thread.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and capture.stats.get("dropouts", 0) < 1:
        time.sleep(0.02)
    assert capture.stats.get("dropouts", 0) == 1, "silent device was not surfaced"
    capture._queue.put(block(0.05))  # audio returns: recovery is logged...
    capture._queue.put(None)  # ...then stop the consumer cleanly
    capture._thread.join(timeout=3.0)
    assert not capture._thread.is_alive()
    assert capture._dropout_reported is False
