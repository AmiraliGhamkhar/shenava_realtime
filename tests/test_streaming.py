"""Streaming decoders: bounded work per step, no full-buffer re-decoding."""

import numpy as np

from shenava_realtime.streaming import CacheAwareDecoder, WindowedDecoder, make_decoder
from tests.fakes import FakeBackend, RecordingTranscriber

SR = 16000


def audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_windowed_decoder_waits_for_enough_new_audio():
    transcriber = RecordingTranscriber()
    decoder = WindowedDecoder(transcriber, sample_rate=SR, min_new_audio_s=0.5)
    assert decoder.push(audio(0.2)) is None
    assert transcriber.calls == []
    assert decoder.push(audio(0.4)) is not None
    assert len(transcriber.calls) == 1


def test_windowed_decoder_never_decodes_the_whole_growing_buffer():
    transcriber = RecordingTranscriber()
    decoder = WindowedDecoder(
        transcriber,
        sample_rate=SR,
        left_context_s=1.0,
        max_window_s=3.0,
        min_new_audio_s=0.5,
    )
    for _ in range(40):  # 20 seconds of audio in half-second blocks
        decoder.push(audio(0.5))

    assert len(transcriber.calls) == 40
    # The naive implementation would decode 0.5s, 1.0s, ... 20s. The window is
    # bounded by max_window_s (+ one block of slack).
    assert max(transcriber.calls) <= 3.0 + 0.5 + 1e-6
    assert max(transcriber.calls) < 20.0


def test_windowed_decoder_signals_a_reset_after_trimming():
    transcriber = RecordingTranscriber()
    decoder = WindowedDecoder(
        transcriber,
        sample_rate=SR,
        left_context_s=1.0,
        max_window_s=2.0,
        min_new_audio_s=0.5,
    )
    results = [decoder.push(audio(0.5)) for _ in range(12)]
    results = [result for result in results if result is not None]
    assert any(result.reset for result in results), "long utterances must be cut into windows"
    assert results[0].reset is False


def test_windowed_finalize_flushes_pending_audio():
    transcriber = RecordingTranscriber()
    decoder = WindowedDecoder(transcriber, sample_rate=SR, min_new_audio_s=1.0)
    decoder.push(audio(0.3))
    assert transcriber.calls == []
    result = decoder.finalize()
    assert result is not None
    assert transcriber.calls == [0.3]


def test_windowed_reset_clears_the_buffer():
    transcriber = RecordingTranscriber()
    decoder = WindowedDecoder(transcriber, sample_rate=SR, min_new_audio_s=0.5)
    decoder.push(audio(1.0))
    decoder.reset()
    decoder.push(audio(0.5))
    assert transcriber.calls[-1] == 0.5


class FakeStream:
    """Stands in for NeMo's cache-aware stream."""

    def __init__(self) -> None:
        self.chunks = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def push(self, chunk: np.ndarray):
        self.chunks.append(len(chunk))
        return " ".join(["واژه"] * len(self.chunks)), 0.9


def test_cache_aware_decoder_passes_every_sample_exactly_once():
    stream = FakeStream()
    decoder = CacheAwareDecoder(stream, sample_rate=SR, min_new_audio_s=0.5)
    for _ in range(6):
        decoder.push(audio(0.25))
    decoder.finalize()
    assert sum(stream.chunks) == int(1.5 * SR)


def test_cache_aware_decoder_batches_small_blocks():
    stream = FakeStream()
    decoder = CacheAwareDecoder(stream, sample_rate=SR, min_new_audio_s=0.5)
    assert decoder.push(audio(0.1)) is None
    assert decoder.push(audio(0.1)) is None
    assert decoder.push(audio(0.4)) is not None
    assert len(stream.chunks) == 1


def test_make_decoder_prefers_cache_aware_streams():
    class CacheBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.stream = FakeStream()

        def create_stream(self):
            return self.stream

    decoder = make_decoder(CacheBackend())
    assert isinstance(decoder, CacheAwareDecoder)


def test_make_decoder_falls_back_to_windowed():
    decoder = make_decoder(FakeBackend())
    assert isinstance(decoder, WindowedDecoder)
    assert decoder.name == "windowed"


def test_make_decoder_survives_a_broken_stream_factory():
    class BrokenBackend(FakeBackend):
        def create_stream(self):
            raise RuntimeError("no streaming support")

    assert isinstance(make_decoder(BrokenBackend()), WindowedDecoder)
