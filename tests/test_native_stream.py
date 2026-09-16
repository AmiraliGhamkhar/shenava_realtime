"""Contract tests for the native adapter, using a tiny tensor/model double.

These verify state/chunk scheduling, not NeMo/checkpoint compatibility or WER.
Run tools/verify_pipeline.py with a local model and WAV for hardware validation.
"""
from contextlib import nullcontext
from types import SimpleNamespace, ModuleType
import sys
import numpy as np
import pytest
from shenava_realtime.native_stream import NeMoCacheAwareStream


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)
    def __getitem__(self, index):
        value = self.array[index]
        return Tensor(value) if isinstance(value, np.ndarray) else value
    def size(self, dim):
        return self.array.shape[dim]
    def clone(self):
        return Tensor(self.array.copy())
    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.array, dim))
    def to(self, device):
        return self


class Torch:
    long = np.int64
    inference_mode = staticmethod(nullcontext)
    from_numpy = staticmethod(Tensor)
    tensor = staticmethod(lambda value, **kw: Tensor(value))
    cat = staticmethod(lambda values, dim: Tensor(np.concatenate([v.array for v in values], axis=dim)))
    nn = SimpleNamespace(functional=SimpleNamespace(
        pad=lambda tensor, pad: Tensor(np.pad(tensor.array, [(0, 0), (0, 0), pad]))))


class Frontend:
    featurizer = SimpleNamespace(n_fft=16)
    def __call__(self, input_signal, length):
        # Deterministic feature frames on the global hop-aligned raw signal.
        x = input_signal.array[0, ::4]
        if input_signal.array.shape[-1] % 4 == 0:
            x = np.append(x, 0)
        return Tensor(x.reshape(1, 1, -1)), Tensor([len(x)])


class Model:
    device = 'cpu'
    cfg = SimpleNamespace(preprocessor=SimpleNamespace(window_stride=4/16000))
    def __init__(self):
        self.calls = []
        self.encoder = SimpleNamespace(
            streaming_cfg=SimpleNamespace(chunk_size=[5, 5], shift_size=[4, 4],
                                          pre_encode_cache_size=[0, 2], drop_extra_pre_encoded=1),
            get_initial_cache_state=lambda batch_size: (0, 0, 0))
    def conformer_stream_step(self, **kwargs):
        assert kwargs['cache_last_channel'] == (kwargs['previous_pred_out'] or 0)
        assert kwargs['processed_signal'].size(-1) <= 7
        index = kwargs['cache_last_channel'] + 1
        self.calls.append(kwargs)
        return index, [f'متن {index}'], index, index, index, index


@pytest.fixture
def stream(monkeypatch):
    utils = ModuleType('nemo.collections.asr.parts.utils.streaming_utils')
    utils.CacheAwareStreamingAudioBuffer = lambda model, **kw: SimpleNamespace(
        preprocessor=Frontend(), model_normalize_type='None')
    features = ModuleType('nemo.collections.asr.parts.preprocessing.features')
    features.normalize_batch = lambda chunk, lengths, normalize_type: (chunk, None, None)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, features.__name__, features)
    return NeMoCacheAwareStream(Model(), Torch(), max_segment_s=1)


def test_native_cache_handoff_final_tail_and_reset(stream):
    samples = np.arange(1000, dtype=np.float32)
    for i in range(0, len(samples), 37):
        stream.push(samples[i:i+37])
        assert len(stream.raw) < 70
        if stream.features is not None:
            assert stream.features.size(-1) < 12
    assert stream.next_frame < 250  # right STFT edge not yet stable
    text, score = stream.finalize()
    assert text.startswith('متن') and score == 0
    assert stream.next_frame == 251
    assert stream.model.calls[0]['drop_extra_pre_encoded'] == 0
    assert all(c['drop_extra_pre_encoded'] == 1 for c in stream.model.calls[1:])
    assert sum(bool(c['keep_all_outputs']) for c in stream.model.calls) == 1
    assert stream.model.calls[-1]['keep_all_outputs']
    count = len(stream.model.calls)
    assert stream.finalize() == (text, 0)
    assert len(stream.model.calls) == count
    with pytest.raises(RuntimeError):
        stream.push(samples)
    stream.reset()
    assert stream.total == 0 and stream.channel == 0
    assert stream.predictions is None and stream.text == ''


def test_native_does_not_decode_utterance_windows_and_bounds_audio(stream):
    stream.push(np.zeros(16000, np.float32))
    assert max(c['processed_signal'].size(-1) for c in stream.model.calls) <= 7
    with pytest.raises(ValueError, match='limit'):
        stream.push(np.zeros(1, np.float32))
    stream.reset()
    with pytest.raises(ValueError, match='Non-finite'):
        stream.push(np.array([np.nan], np.float32))


def test_final_flush_has_no_duplicate_or_lost_frames(stream):
    # The centered frontend of a 1 s signal has 4001 frames.  After an
    # irregular push sequence plus the final flush, the frame cursor must be
    # exactly 4001: every frame consumed once, in order, by the cache-aware
    # encoder (overlapping chunks are re-encoded, never re-consumed).
    n = 16000
    audio = np.sin(np.linspace(0.0, 10.0, n)).astype(np.float32)
    position = 0
    for size in (511, 37, 5000, 6000, 4452):
        position += size
        stream.push(audio[position - size:position])
    assert stream.total == n
    text, score = stream.finalize()
    assert text.startswith('متن') and score == 0
    assert stream.next_frame == n // 4 + 1
    # Exactly one call decodes the tail with all outputs; the encoder cache
    # was handed across calls (previous output becomes the next cache state).
    assert stream.model.calls[-1]['keep_all_outputs']
    assert sum(bool(c['keep_all_outputs']) for c in stream.model.calls) == 1
    # Encoder cache handoff: every call's cache state is the previous call's
    # prediction output (the double increments it by one per step).
    for previous, call in zip(stream.model.calls, stream.model.calls[1:]):
        assert call['cache_last_channel'] == previous['cache_last_channel'] + 1
