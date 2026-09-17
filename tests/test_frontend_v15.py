"""Frontend parity for the v1.5 paths: RNNT streaming reuses the CTC accounting.

The RNNT stream is deliberately the CTC stream plus a capability check, so the
property that matters is that it *stays* that way: identical 16 kHz frame
accounting, no duplicated frames, no dropped final frames, and the same final
flush.  These tests drive both streams over the same audio and compare.
"""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from shenava_realtime.native_stream import NeMoCacheAwareStream
from shenava_realtime.rnnt_stream import RNNTCacheAwareStream, RNNTCapabilityError
from tests.test_frontend import (
    CACHE,
    CHUNK_SIZE,
    HOP,
    N_FFT,
    TOLERANCE,
    CenteredFrontend,
    Model,
    signal_1_5_s,
)
from tests.test_native_stream import Torch


class HybridModel(Model):
    """A v1.5-shaped double: the CTC encoder plus an RNNT prediction net/joint."""

    def __init__(self):
        super().__init__()
        self.joint = SimpleNamespace()
        self.decoder = SimpleNamespace()
        self.decoding = SimpleNamespace(blank_id=1023)


@pytest.fixture()
def nemo_stubs(monkeypatch):
    utils = ModuleType("nemo.collections.asr.parts.utils.streaming_utils")
    utils.CacheAwareStreamingAudioBuffer = lambda model, **kw: SimpleNamespace(
        preprocessor=CenteredFrontend(), model_normalize_type="None")
    features = ModuleType("nemo.collections.asr.parts.preprocessing.features")
    features.normalize_batch = lambda chunk, lengths, normalize_type: (chunk, None, None)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, features.__name__, features)


CHUNKS = ((0, 37), (37, 1000), (1037, 3), (1040, 5199), (6239, 8191), (14430, 9570))


def drive(stream, audio):
    for offset, size in CHUNKS:
        stream.push(audio[offset:offset + size])
    return stream.finalize()


# --------------------------------------------------------------------------- #
# Capability gating
# --------------------------------------------------------------------------- #
def test_rnnt_stream_requires_the_rnnt_head(nemo_stubs):
    with pytest.raises(RNNTCapabilityError, match="prediction network/joint"):
        RNNTCacheAwareStream(Model(), Torch(), max_segment_s=10)


def test_rnnt_stream_accepts_a_hybrid_checkpoint(nemo_stubs):
    stream = RNNTCacheAwareStream(HybridModel(), Torch(), max_segment_s=10)
    assert stream.head == "rnnt"


def test_rnnt_stream_still_enforces_the_16k_frontend(nemo_stubs):
    class WrongRate(HybridModel):
        cfg = SimpleNamespace(preprocessor=SimpleNamespace(
            window_stride=HOP / 16000, sample_rate=22050))

    with pytest.raises(RuntimeError, match="16 kHz"):
        RNNTCacheAwareStream(WrongRate(), Torch(), max_segment_s=10)


# --------------------------------------------------------------------------- #
# Frame accounting parity
# --------------------------------------------------------------------------- #
def test_rnnt_and_ctc_streams_produce_identical_encoder_windows(nemo_stubs):
    audio = signal_1_5_s()
    ctc_model, rnnt_model = Model(), HybridModel()
    drive(NeMoCacheAwareStream(ctc_model, Torch(), max_segment_s=10), audio)
    drive(RNNTCacheAwareStream(rnnt_model, Torch(), max_segment_s=10), audio)

    assert len(rnnt_model.windows) == len(ctc_model.windows)
    for rnnt_window, ctc_window in zip(rnnt_model.windows, ctc_model.windows):
        np.testing.assert_allclose(rnnt_window, ctc_window, rtol=0.0, atol=TOLERANCE)


def test_rnnt_stream_consumes_every_sample_exactly_once(nemo_stubs):
    audio = signal_1_5_s()
    stream = RNNTCacheAwareStream(HybridModel(), Torch(), max_segment_s=10)
    drive(stream, audio)
    assert stream.total == audio.size                       # nothing dropped
    assert stream.next_frame == audio.size // HOP + 1       # no duplicate frames


def test_rnnt_stream_keeps_the_raw_buffer_bounded_and_hop_aligned(nemo_stubs):
    audio = signal_1_5_s()
    stream = RNNTCacheAwareStream(HybridModel(), Torch(), max_segment_s=10)
    for offset, size in CHUNKS:
        stream.push(audio[offset:offset + size])
        assert stream.raw.size <= stream.margin + N_FFT
        assert stream.raw_start % HOP == 0


def test_encoder_windows_stay_within_the_configured_chunk_and_cache(nemo_stubs):
    audio = signal_1_5_s()
    model = HybridModel()
    drive(RNNTCacheAwareStream(model, Torch(), max_segment_s=10), audio)
    assert model.windows
    assert all(window.size <= CHUNK_SIZE + CACHE for window in model.windows)


# --------------------------------------------------------------------------- #
# Final flush behaviour
# --------------------------------------------------------------------------- #
def test_final_flush_emits_text_and_is_idempotent(nemo_stubs):
    audio = signal_1_5_s()
    stream = RNNTCacheAwareStream(HybridModel(), Torch(), max_segment_s=10)
    text, score = drive(stream, audio)
    assert text and score == 0.0
    assert stream.finalize() == (text, score)   # finalize twice, same result


def test_pushing_after_finalize_is_refused(nemo_stubs):
    stream = RNNTCacheAwareStream(HybridModel(), Torch(), max_segment_s=10)
    stream.push(signal_1_5_s()[:8000])
    stream.finalize()
    with pytest.raises(RuntimeError, match="already finalized"):
        stream.push(np.zeros(1600, dtype=np.float32))


def test_reset_allows_reuse_and_clears_the_frame_cursor(nemo_stubs):
    stream = RNNTCacheAwareStream(HybridModel(), Torch(), max_segment_s=10)
    stream.push(signal_1_5_s()[:8000])
    stream.finalize()
    stream.reset()
    assert stream.total == 0 and stream.next_frame == 0 and stream.closed is False
    stream.push(signal_1_5_s()[:8000])
    assert stream.finalize()[0]


def test_the_segment_cap_is_enforced_on_the_rnnt_path(nemo_stubs):
    stream = RNNTCacheAwareStream(HybridModel(), Torch(), max_segment_s=0.5)
    with pytest.raises(ValueError, match="segment limit"):
        stream.push(np.zeros(16000, dtype=np.float32))
