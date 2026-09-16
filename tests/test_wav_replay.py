"""Real-WAV replay through the full engine (VAD + capture + pipeline).

The audio is a genuine PCM16 16 kHz mono WAV file decoded with the same
``wav_blocks`` helper the CLI replay uses; the ASR model is a labelled fake
backend, so this verifies the capture/endpoint/orchestration path — not model
accuracy.  The native streaming doubles in ``test_native_stream`` cover the
encoder-cache path separately.
"""
import numpy as np
import pytest
import wave

from shenava_realtime.config import AppConfig, OutputMode
from shenava_realtime.realtime_engine import RealtimeASR
from tests.fakes import FakeBackend, RecordingTranscriber

sys_path_shim = pytest.importorskip("tools.verify_pipeline")

PHRASE = "بیمار دوز دو و نیم میلی گرم"


def write_wav(path, seconds: float, amplitude: float = 0.15,
              tail_silence: float = 1.2, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = int(seconds * 16000)
    t = np.arange(n) / 16000.0
    env = np.clip(0.65 + 0.35 * np.sin(2 * np.pi * 4 * t), 0.25, 1.0)
    samples = amplitude * env * (0.75 + 0.25 * rng.random(n))
    tail = np.zeros(int(tail_silence * 16000))
    pcm = np.clip(np.concatenate([samples, tail]) * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(pcm.tobytes())


def run_replay(tmp_path, seconds: float = 1.0, text: str = PHRASE, tail: float = 1.2,
               amplitude: float = 0.15, second_pass: str = "off"):
    wav = tmp_path / "replay.wav"
    write_wav(wav, seconds, amplitude=amplitude, tail_silence=tail)
    blocks = list(sys_path_shim.wav_blocks(wav))
    assert blocks and all(b.size == 1024 for b in blocks[:-1])

    config = AppConfig()
    config.output_mode = OutputMode.CONSOLE
    config.save_transcripts = False
    config.asr.require_streaming = False
    config.asr.second_pass = second_pass
    backend = FakeBackend(RecordingTranscriber(lambda s: text))
    capture = sys_path_shim.ReplayCapture(config.audio, iter(blocks))
    engine = RealtimeASR(config, backend=backend, audio_capture=capture)
    records, deltas = [], []
    engine.on_utterance_end = lambda t, _: records.append(t)
    engine.on_text_delta = lambda t, _: deltas.append(t)
    try:
        engine.start()
        while not capture.done.wait(0.2):
            if not engine.is_running:
                raise RuntimeError("ASR worker exited during replay")
    finally:
        engine.stop()
    if capture.error:
        raise capture.error
    return engine, records, deltas, backend


def test_replay_wav_produces_one_canonical_record(tmp_path):
    engine, records, deltas, backend = run_replay(tmp_path)
    # The final utterance text is post-processed (numbers canonicalised).
    assert records == ["بیمار دوز 2.5 mg"]
    # Deltas reconstruct exactly the final text: no duplication, no loss,
    # and the delta stream is stable (each final emitted once at utterance end).
    assert "".join(deltas) == records[0]
    stats = engine.get_statistics()
    assert stats["decodes"] >= 1
    assert not stats.get("errors") and not stats.get("overruns")
    assert stats["second_pass"] == "off"
    assert stats["second_pass_stats"] == {"runs": 0, "rewrites": 0, "fallbacks": 0}
    # One offline decode per replayed segment.
    assert len(backend.transcriber.calls) == 1


def test_replay_greedy_second_pass_runs_offline(tmp_path):
    # Endpoint decoder keeps no streaming audio, so the offline greedy second
    # pass *is* the utterance decode: the record comes from it, not a second
    # decoding of already-streamed audio.
    engine, records, deltas, backend = run_replay(tmp_path, second_pass="greedy")
    assert records == ["بیمار دوز 2.5 mg"]
    assert "".join(deltas) == records[0]
    stats = engine.get_statistics()
    assert stats["second_pass"] == "greedy"


def test_silent_wav_yields_no_utterances(tmp_path):
    engine, records, deltas, _ = run_replay(tmp_path, seconds=1.0, amplitude=0.0005)
    assert records == [] and deltas == []
    stats = engine.get_statistics()
    assert not stats.get("errors") and not stats.get("overruns")


def test_invalid_wav_format_rejected(tmp_path):
    # Two channels must fail clearly rather than replay with shifted audio.
    wav = tmp_path / "stereo.wav"
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(np.zeros(4096, dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="PCM16, 16 kHz, mono"):
        list(sys_path_shim.wav_blocks(wav))
