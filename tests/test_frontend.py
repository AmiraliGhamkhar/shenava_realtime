"""Frontend correctness: streaming features must align with the offline path.

The double below models NeMo's *centered* STFT faithfully (each frame is the
mean of a zero-padded window centred on the hop grid — a rectangular-window
STFT proxy).  The same short deterministic "WAV" is processed through the
streaming frontend in irregular chunks and through one offline preprocessor
call.  Every encoder window the stream re-encodes must contain exactly the
offline frames at that absolute position — bounded overlap only, no shift
across chunk boundaries, no lost or duplicated frames.  Agreement is checked
within a bounded tolerance (exact here for the double), never by demanding
bitwise equality of the two paths.
"""
from contextlib import nullcontext
from types import SimpleNamespace, ModuleType
import sys

import numpy as np
import pytest

from shenava_realtime.native_stream import NeMoCacheAwareStream
from tests.test_native_stream import Tensor, Torch

N_FFT = 16
HOP = 4
CHUNK_SIZE = 500
CACHE = 2
SHIFT = 4
TOLERANCE = 1e-12  # bounded agreement tolerance


class CenteredFrontend:
    """Faithful centered-window feature (rectangular window STFT proxy)."""
    featurizer = SimpleNamespace(n_fft=N_FFT, frame_splicing=1, exact_pad=False)

    def __call__(self, input_signal, length):
        x = np.asarray(input_signal.array[0], dtype=np.float64)
        t = x.size // HOP + 1
        frames = np.empty((1, 1, t), dtype=np.float64)
        half = N_FFT // 2
        for i in range(t):
            win_start = i * HOP - half  # signal index of window[0] (may be < 0)
            window = np.zeros(N_FFT)
            src = max(0, win_start)
            dst = src - win_start
            count = min(N_FFT - dst, x.size - src)
            if count > 0:
                window[dst:dst + count] = x[src:src + count]
            frames[0, 0, i] = window.mean()
        return Tensor(frames), Tensor([t])


class Model:
    device = "cpu"
    cfg = SimpleNamespace(preprocessor=SimpleNamespace(
        window_stride=HOP / 16000, sample_rate=16000))

    def __init__(self):
        self.windows = []
        self.encoder = SimpleNamespace(
            streaming_cfg=SimpleNamespace(chunk_size=[CHUNK_SIZE, CHUNK_SIZE],
                                          shift_size=[SHIFT, SHIFT],
                                          pre_encode_cache_size=[0, CACHE],
                                          drop_extra_pre_encoded=1),
            get_initial_cache_state=lambda batch_size: (0, 0, 0))

    def conformer_stream_step(self, **kwargs):
        self.windows.append(kwargs["processed_signal"].array[0, 0].copy())
        index = (kwargs["cache_last_channel"] or 0) + 1
        return index, ["متن"], index, index, index, index


@pytest.fixture()
def stream(monkeypatch):
    utils = ModuleType("nemo.collections.asr.parts.utils.streaming_utils")
    utils.CacheAwareStreamingAudioBuffer = lambda model, **kw: SimpleNamespace(
        preprocessor=CenteredFrontend(), model_normalize_type="None")
    features = ModuleType("nemo.collections.asr.parts.preprocessing.features")
    features.normalize_batch = lambda chunk, lengths, normalize_type: (chunk, None, None)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, features.__name__, features)
    return NeMoCacheAwareStream(Model(), Torch(), max_segment_s=10)


def signal_1_5_s() -> np.ndarray:
    # Deterministic "short WAV": chirp + bounded noise, mono float32.
    rng = np.random.default_rng(7)
    t = np.arange(24000) / 16000.0
    x = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(24000)
    return x.astype(np.float32)


def test_streaming_windows_match_offline_frames_within_bounded_tolerance(stream):
    audio = signal_1_5_s()
    preprocessor, torch = stream.preprocessor, stream.torch
    with torch.inference_mode():
        reference, _ = preprocessor(
            torch.from_numpy(audio).unsqueeze(0),
            torch.tensor([audio.size], dtype=torch.long))
    offline = reference.array[0, 0]

    for offset, size in ((0, 37), (37, 1000), (1037, 3), (1040, 5199),
                         (6239, 8191), (14430, 9570)):
        stream.push(audio[offset:offset + size])
        # Retained raw audio: bounded, hop-aligned left overlap plus the
        # unstable right STFT edge (never a whole extra window).
        margin = stream.margin
        assert stream.raw.size <= margin + N_FFT
        assert stream.raw_start % HOP == 0

    text, score = stream.finalize()
    assert text.startswith("متن") and score == 0.0

    # Each encoder window holds exactly the offline frames at its absolute
    # position: the first window starts at frame 0, the second at the
    # pre-encode cache, and every further window advances by the shift.
    windows = stream.model.windows
    assert windows
    assert all(w.size <= CHUNK_SIZE + CACHE for w in windows)  # bounded overlap
    position = 0
    for window in windows[:-1]:
        assert position + window.size <= offline.size
        np.testing.assert_allclose(
            window, offline[position:position + window.size],
            rtol=0.0, atol=TOLERANCE)
        position += CACHE if position == 0 else SHIFT
    last = windows[-1]
    tail = offline.size - position
    np.testing.assert_allclose(last[:tail], offline[position:], rtol=0.0, atol=TOLERANCE)
    np.testing.assert_array_equal(last[tail:], 0.0)  # right padding only
    # No frame lost or duplicated: the final cursor reaches exactly the
    # centered frame count of the full signal.
    assert stream.next_frame == offline.size
    assert stream.total == audio.size


def test_frame_alignment_never_shifts_across_chunks(stream):
    audio = signal_1_5_s()
    for offset, size in ((0, 640), (640, 640), (1280, 16000), (17280, 6720)):
        stream.push(audio[offset:offset + size])
        assert stream.raw.size <= stream.margin + N_FFT
        assert stream.raw_start % HOP == 0
    stream.finalize()
    assert stream.next_frame == audio.size // HOP + 1  # no lost or duplicate frames


def test_unsupported_sample_rate_fails_at_initialization(monkeypatch):
    utils = ModuleType("nemo.collections.asr.parts.utils.streaming_utils")
    utils.CacheAwareStreamingAudioBuffer = lambda model, **kw: SimpleNamespace(
        preprocessor=CenteredFrontend(), model_normalize_type="None")
    features = ModuleType("nemo.collections.asr.parts.preprocessing.features")
    features.normalize_batch = lambda chunk, lengths, normalize_type: (chunk, None, None)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, features.__name__, features)

    class WrongRateModel(Model):
        cfg = SimpleNamespace(preprocessor=SimpleNamespace(
            window_stride=HOP / 16000, sample_rate=44100))

    with pytest.raises(RuntimeError, match="16 kHz"):
        NeMoCacheAwareStream(WrongRateModel(), Torch(), max_segment_s=10)
