"""Streaming decoders: bounded work per step, no full-buffer re-decoding."""

import numpy as np
import pytest
from shenava_realtime.config import ASRConfig

from shenava_realtime.streaming import CacheAwareDecoder, EndpointDecoder, make_decoder
from tests.fakes import FakeBackend, RecordingTranscriber

SR = 16000


def audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_endpoint_decodes_exactly_once_and_is_bounded():
    transcriber = RecordingTranscriber()
    decoder = EndpointDecoder(transcriber, max_segment_s=2)
    for _ in range(4):
        assert decoder.push(audio(.5)) is None
    assert transcriber.calls == []
    with pytest.raises(ValueError):
        decoder.push(audio(.1))
    assert decoder.finalize() is not None
    assert transcriber.calls == [2.0]
    assert decoder.finalize() is None
    assert decoder.samples == 0


def test_endpoint_reset_discards_pending_audio():
    transcriber = RecordingTranscriber()
    decoder = EndpointDecoder(transcriber)
    decoder.push(audio(1))
    decoder.reset()
    assert decoder.finalize() is None
    assert not transcriber.calls


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


def test_make_decoder_falls_back_to_endpoint():
    decoder = make_decoder(FakeBackend(), ASRConfig(require_streaming=False))
    assert isinstance(decoder, EndpointDecoder)
    assert decoder.name == "endpoint"


def test_make_decoder_survives_a_broken_stream_factory():
    class BrokenBackend(FakeBackend):
        def create_stream(self):
            raise RuntimeError("no streaming support")

    with pytest.raises(RuntimeError):
        make_decoder(BrokenBackend())


def test_streaming_disable_and_strict_mode():
    with pytest.raises(RuntimeError, match="does not support"):
        make_decoder(FakeBackend(), ASRConfig(require_streaming=True))
    class Backend(FakeBackend):
        def create_stream(self):
            pytest.fail("disabled stream must not be created")
    assert isinstance(
        make_decoder(Backend(), ASRConfig(use_cache_aware_streaming=False, require_streaming=False)),
        EndpointDecoder,
    )


def test_native_finalize_is_called_even_without_pending_audio():
    class Stream(FakeStream):
        def finalize(self):
            return "پایان", 0.0
    decoder = CacheAwareDecoder(Stream())
    decoder.push(audio(1))
    assert decoder.finalize().text == "پایان"
