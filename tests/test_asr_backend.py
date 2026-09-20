"""Sherpa-ONNX ASR backend: explicit errors, single recognizer, stream lifecycle.

Uses a mock ``sherpa_onnx`` module (``tests/fake_sherpa_onnx.py``) so these
tests need neither the 132 MB model nor ONNX Runtime. A real-model smoke test
lives in ``test_asr_backend_smoke.py`` and is skipped unless the files exist.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from shenava_realtime.asr_backend import ModelLoadError, SherpaOnnxASR, StreamingUnavailable
from shenava_realtime.config import ASRConfig
from tests.fake_sherpa_onnx import FakeOnlineRecognizer


@pytest.fixture(autouse=True)
def fake_sherpa_onnx_module(monkeypatch):
    """Inject a fake ``sherpa_onnx`` module before the backend imports it."""
    FakeOnlineRecognizer.last_kwargs = {}
    FakeOnlineRecognizer.last_instance = None
    FakeOnlineRecognizer.fail_init = False
    FakeOnlineRecognizer.streams_created = 0
    module = types.ModuleType("sherpa_onnx")
    module.OnlineRecognizer = FakeOnlineRecognizer
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    yield
    monkeypatch.delitem(sys.modules, "sherpa_onnx", raising=False)


@pytest.fixture()
def model_files(tmp_path):
    model = tmp_path / "model.int8.onnx"
    tokens = tmp_path / "tokens.txt"
    model.write_bytes(b"not a real onnx file")
    tokens.write_text("<blk> 0\na 1\n", encoding="utf-8")
    return model, tokens


def make_backend(model_files, **overrides) -> SherpaOnnxASR:
    model, tokens = model_files
    config = ASRConfig(model_path=str(model), tokens_path=str(tokens), **overrides)
    return SherpaOnnxASR(config)


# --------------------------------------------------------------------------- #
# Explicit startup errors, no fallback
# --------------------------------------------------------------------------- #
def test_missing_model_file_is_a_clear_error(tmp_path):
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blk> 0\n", encoding="utf-8")
    backend = SherpaOnnxASR(ASRConfig(model_path=str(tmp_path / "missing.onnx"), tokens_path=str(tokens)))
    with pytest.raises(ModelLoadError, match="not found"):
        backend.load()


def test_missing_tokens_file_is_a_clear_error(tmp_path):
    model = tmp_path / "model.int8.onnx"
    model.write_bytes(b"x")
    backend = SherpaOnnxASR(ASRConfig(model_path=str(model), tokens_path=str(tmp_path / "missing.txt")))
    with pytest.raises(ModelLoadError, match="tokens"):
        backend.load()


def test_wrong_model_extension_is_rejected(tmp_path):
    model = tmp_path / "model.bin"
    tokens = tmp_path / "tokens.txt"
    model.write_bytes(b"x")
    tokens.write_text("<blk> 0\n", encoding="utf-8")
    backend = SherpaOnnxASR(ASRConfig(model_path=str(model), tokens_path=str(tokens)))
    with pytest.raises(ModelLoadError, match=r"\.onnx"):
        backend.load()


def test_legacy_nemo_path_is_rejected_with_a_migration_message(tmp_path):
    model = tmp_path / "shenava-koochik-v1.5.nemo"
    tokens = tmp_path / "tokens.txt"
    model.write_bytes(b"x")
    tokens.write_text("<blk> 0\n", encoding="utf-8")
    backend = SherpaOnnxASR(ASRConfig(model_path=str(model), tokens_path=str(tokens)))
    with pytest.raises(ModelLoadError, match="legacy NeMo checkpoint"):
        backend.load()


def test_wrong_tokens_extension_is_rejected(model_files, tmp_path):
    model, _ = model_files
    bogus_tokens = tmp_path / "tokens.json"
    bogus_tokens.write_text("{}", encoding="utf-8")
    backend = SherpaOnnxASR(ASRConfig(model_path=str(model), tokens_path=str(bogus_tokens)))
    with pytest.raises(ModelLoadError, match=r"\.txt"):
        backend.load()


def test_sherpa_onnx_not_importable_is_explicit(model_files, monkeypatch):
    monkeypatch.delitem(sys.modules, "sherpa_onnx", raising=False)
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sherpa_onnx":
            raise ImportError("no module named sherpa_onnx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    backend = make_backend(model_files)
    with pytest.raises(ModelLoadError, match="sherpa-onnx is required"):
        backend.load()


def test_wrong_sample_rate_is_rejected_before_loading(model_files):
    backend = make_backend(model_files)
    backend.config.sample_rate = 8000
    with pytest.raises(Exception):  # ASRConfig.__post_init__ raises ValueError
        backend.load()


def test_recognizer_init_failure_is_wrapped(model_files):
    FakeOnlineRecognizer.fail_init = True
    backend = make_backend(model_files)
    with pytest.raises(ModelLoadError, match="failed to initialise"):
        backend.load()


def test_missing_model_path_configured_is_explicit():
    backend = SherpaOnnxASR(ASRConfig(model_path=None, tokens_path=None))
    with pytest.raises(ModelLoadError, match="not configured"):
        backend.load()


# --------------------------------------------------------------------------- #
# Correct construction: CPU provider, threads, sample rate, feature dim,
# greedy decoding, sherpa's own endpointer disabled (single endpointer: VAD).
# --------------------------------------------------------------------------- #
def test_load_passes_cpu_provider_and_config_through(model_files):
    backend = make_backend(model_files, num_threads=2, feature_dim=80)
    backend.load()
    kwargs = FakeOnlineRecognizer.last_kwargs
    assert kwargs["provider"] == "cpu"
    assert kwargs["num_threads"] == 2
    assert kwargs["sample_rate"] == 16000
    assert kwargs["feature_dim"] == 80
    assert kwargs["decoding_method"] == "greedy_search"
    # Exactly one endpointer: the existing RMS VAD, never sherpa's own.
    assert kwargs["enable_endpoint_detection"] is False


def test_load_is_idempotent_single_recognizer_instance(model_files):
    backend = make_backend(model_files)
    backend.load()
    first = backend.recognizer
    backend.load()
    assert backend.recognizer is first


def test_capabilities_report_streaming_cpu_int8(model_files):
    backend = make_backend(model_files)
    caps = backend.capabilities()
    assert caps.streaming is True
    assert caps.provider == "cpu"
    assert "streaming=yes" in caps.describe()


def test_streaming_probe_executes_the_online_lifecycle(model_files):
    backend = make_backend(model_files)
    backend.load()
    events = FakeOnlineRecognizer.last_instance.events
    assert events[:2] == ["create_stream", "is_ready"]
    assert "decode_stream" in events
    assert events.count("get_result") >= 2
    assert events.index("decode_stream") < events.index("get_result")


def test_streaming_probe_rejects_missing_streaming_methods(model_files, monkeypatch):
    class MissingDecode(FakeOnlineRecognizer):
        decode_stream = None

    sys.modules["sherpa_onnx"].OnlineRecognizer = MissingDecode
    backend = make_backend(model_files)
    with pytest.raises(StreamingUnavailable, match="decode_stream"):
        backend.load()


def test_streaming_probe_does_more_than_check_method_existence(model_files):
    class BrokenDecode(FakeOnlineRecognizer):
        def decode_stream(self, stream):
            self.events.append("decode_stream")
            raise RuntimeError("decode lifecycle was exercised")

    sys.modules["sherpa_onnx"].OnlineRecognizer = BrokenDecode
    backend = make_backend(model_files)
    with pytest.raises(StreamingUnavailable, match="decode lifecycle was exercised"):
        backend.load()


def test_require_streaming_gate_has_no_fallback(model_files, monkeypatch):
    backend = make_backend(model_files, require_streaming=True)
    monkeypatch.setattr(backend, "_probe_streaming", lambda recognizer: False)
    with pytest.raises(StreamingUnavailable):
        backend.load()
    # No recognizer is exposed after a failed GATE check.
    assert backend._recognizer is None


# --------------------------------------------------------------------------- #
# Stream lifecycle: one long-lived recognizer, one stream per segment
# --------------------------------------------------------------------------- #
def test_transcribe_of_empty_audio_short_circuits(model_files):
    backend = make_backend(model_files)
    assert backend.transcribe(np.zeros(0, dtype=np.float32)) == ("", 0.0)


def test_create_stream_reuses_the_single_recognizer(model_files):
    backend = make_backend(model_files)
    backend.load()
    before = FakeOnlineRecognizer.streams_created
    stream_a = backend.create_stream()
    stream_b = backend.create_stream()
    assert stream_a._recognizer is stream_b._recognizer
    # Exactly one recognizer is ever constructed (see test_load_is_idempotent);
    # only streams are created per call.
    assert FakeOnlineRecognizer.streams_created == before + 2


def test_incremental_chunk_feeding_accumulates_without_whole_buffer_copies(model_files):
    backend = make_backend(model_files)
    stream = backend.create_stream()
    # 4 chunks of 400ms (1600 samples each at 16kHz) -> 4 decode steps.
    chunk = np.zeros(1600, dtype=np.float32)
    for _ in range(4):
        stream.accept(chunk)
        stream.decode_ready()
    assert stream.partial() == "tok0 tok1 tok2 tok3"


def test_partial_is_cumulative_and_grows_monotonically(model_files):
    backend = make_backend(model_files)
    stream = backend.create_stream()
    chunk = np.zeros(1600, dtype=np.float32)
    seen = []
    for _ in range(3):
        stream.push(chunk)
        seen.append(stream.partial())
    assert seen == ["tok0", "tok0 tok1", "tok0 tok1 tok2"]


def test_finalize_pads_the_tail_and_calls_input_finished(model_files):
    backend = make_backend(model_files)
    stream = backend.create_stream()
    stream.accept(np.zeros(1600, dtype=np.float32))
    stream.decode_ready()
    text, confidence = stream.finalize()
    assert confidence == 0.0
    assert stream._stream.finished is True
    # Tail padding (0.5s = 8000 samples) plus the 1600 already fed = 9600.
    assert len(stream._stream.samples) == 1600 + 8000


def test_finalized_stream_cannot_be_reused(model_files):
    backend = make_backend(model_files)
    stream = backend.create_stream()
    stream.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        stream.accept(np.zeros(100, dtype=np.float32))


def test_new_stream_per_segment_has_no_state_bleed(model_files):
    """Each VAD segment gets a fresh stream; a loud first segment must not
    influence the token count of a completely separate second segment."""
    backend = make_backend(model_files)
    stream1 = backend.create_stream()
    for _ in range(5):
        stream1.push(np.zeros(1600, dtype=np.float32))
    text1, _ = stream1.finalize()
    assert text1  # first segment produced tokens

    stream2 = backend.create_stream()
    assert stream2.partial() == ""  # brand-new stream starts empty
    stream2.push(np.zeros(1600, dtype=np.float32))
    assert stream2.partial() == "tok0"  # not "tok6" or similar carry-over


def test_bounded_memory_over_a_long_synthetic_stream(model_files):
    """~10 minutes of audio fed in small chunks: the wrapper must not
    accumulate the whole recording (only the fake stream, which we control,
    grows; the wrapper itself holds no growing per-chunk buffers)."""
    backend = make_backend(model_files)
    stream = backend.create_stream()
    chunk = np.zeros(1600, dtype=np.float32)  # 100ms
    steps = int(10 * 60 / 0.1)  # 10 minutes of 100ms chunks
    for _ in range(steps):
        stream.accept(chunk)
        stream.decode_ready()
    assert stream._stream.tokens == steps
    stats = backend.stats()
    assert stats["decode_count"] == steps
    assert stats["audio_seconds"] == pytest.approx(600.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Stats: never transcript/audio content
# --------------------------------------------------------------------------- #
def test_stats_never_contain_transcript_or_audio_keys(model_files):
    backend = make_backend(model_files)
    stream = backend.create_stream()
    stream.push(np.zeros(1600, dtype=np.float32))
    stream.finalize()
    stats = backend.stats()
    assert set(stats) == {
        "audio_seconds", "processing_seconds", "rtf", "decode_count",
        "dropped_chunks", "overruns", "decoder_errors", "segments",
    }
    assert stats["segments"] == 1
    assert stats["decode_count"] >= 1
    assert stats["rtf"] >= 0.0


def test_decoder_errors_are_counted(model_files, monkeypatch):
    backend = make_backend(model_files)
    stream = backend.create_stream()
    stream.accept(np.zeros(1600, dtype=np.float32))

    def boom(_stream):
        raise RuntimeError("synthetic decode failure")

    monkeypatch.setattr(stream._recognizer, "decode_stream", boom)
    with pytest.raises(RuntimeError):
        stream.decode_ready()
    assert backend.stats()["decoder_errors"] == 1


# --------------------------------------------------------------------------- #
# Streaming capability probe: full-lifecycle regression tests.
#
# The probe feeds PROBE_MAX_AUDIO_S (8.0s) of deterministic sine-wave audio
# in PROBE_CHUNK_S (0.1s) chunks, draining every decode step the recognizer
# reports ready for as it goes, then calls input_finished() and drains again.
# A decode step is only required *somewhere* in that whole transaction -- not
# necessarily before input_finished() -- so a model that needs more context
# than one second, or that only surfaces buffered features at end-of-stream,
# must still be accepted as a valid streaming model.
# --------------------------------------------------------------------------- #
def _make_stream_recognizer(is_ready, decode_stream, get_result, events=None):
    """Build a minimal fake OnlineRecognizer/OnlineStream pair for probe tests.

    ``is_ready``, ``decode_stream`` and ``get_result`` are callables invoked
    with the stream instance; ``events`` (if given) collects every method
    name called on either the recognizer or the stream, in order.
    """
    log = events if events is not None else []

    class _Stream:
        def __init__(self) -> None:
            self.total_samples = 0
            self.decoded = 0
            self.finished = False

        def accept_waveform(self, sample_rate, waveform) -> None:
            log.append("accept_waveform")
            self.total_samples += len(waveform)

        def input_finished(self) -> None:
            log.append("input_finished")
            self.finished = True

    class _Recognizer(FakeOnlineRecognizer):
        def create_stream(self):
            log.append("create_stream")
            return _Stream()

        def is_ready(self, stream):
            log.append("is_ready")
            return is_ready(stream)

        def decode_stream(self, stream):
            log.append("decode_stream")
            decode_stream(stream)

        def get_result(self, stream):
            log.append("get_result")
            return get_result(stream)

    return _Recognizer


def test_streaming_probe_accepts_delayed_readiness(model_files, monkeypatch):
    """Case B: not ready for several chunks, then becomes ready before EOF."""
    # PROBE_CHUNK_S=0.1s / PROBE_MAX_AUDIO_S=8.0s @ 16kHz -> ready only once
    # 3s (well past the old one-second assumption) has accumulated.
    READY_AFTER_SAMPLES = 3 * 16000

    def is_ready(stream):
        return stream.total_samples >= READY_AFTER_SAMPLES and stream.decoded < 1

    def decode_stream(stream):
        stream.decoded += 1

    def get_result(stream):
        return "delayed" if stream.decoded else ""

    events: list = []
    recognizer_cls = _make_stream_recognizer(is_ready, decode_stream, get_result, events)
    sys.modules["sherpa_onnx"].OnlineRecognizer = recognizer_cls
    backend = make_backend(model_files)
    backend.load()  # must not raise: readiness after >1s of audio is valid
    assert backend.capabilities().streaming is True
    assert "decode_stream" in events
    # The decode happened before input_finished() (case B).
    assert events.index("decode_stream") < events.index("input_finished")


def test_streaming_probe_accepts_decode_only_after_eof(model_files):
    """Case C: never ready before EOF; only ready once input_finished() runs."""

    def is_ready(stream):
        return stream.finished and stream.decoded < 1

    def decode_stream(stream):
        stream.decoded += 1

    def get_result(stream):
        return "eof-decoded" if stream.decoded else ""

    events: list = []
    recognizer_cls = _make_stream_recognizer(is_ready, decode_stream, get_result, events)
    sys.modules["sherpa_onnx"].OnlineRecognizer = recognizer_cls
    backend = make_backend(model_files)
    backend.load()  # must not raise: this is a valid streaming lifecycle
    assert backend.capabilities().streaming is True
    assert "decode_stream" in events
    # No decode before input_finished(); the only decode is after EOF.
    finish_index = events.index("input_finished")
    assert "decode_stream" not in events[:finish_index]
    assert "decode_stream" in events[finish_index:]


def test_streaming_probe_rejects_a_permanently_non_ready_stream(model_files):
    """Invalid case: never ready, even after input_finished(); must fail."""

    def is_ready(stream):
        return False

    def decode_stream(stream):  # pragma: no cover - never called
        pass

    def get_result(stream):
        return ""

    recognizer_cls = _make_stream_recognizer(is_ready, decode_stream, get_result)
    sys.modules["sherpa_onnx"].OnlineRecognizer = recognizer_cls
    backend = make_backend(model_files)
    with pytest.raises(StreamingUnavailable, match="no decode step"):
        backend.load()


def test_streaming_probe_normal_lifecycle_order(model_files):
    """A healthy model: decode both before and after input_finished(), with
    get_result() available at both points, in the documented order."""

    def is_ready(stream):
        return stream.total_samples >= 1600 and stream.decoded < stream.total_samples // 1600

    def decode_stream(stream):
        stream.decoded += 1

    def get_result(stream):
        return " ".join(f"tok{i}" for i in range(stream.decoded))

    events: list = []
    recognizer_cls = _make_stream_recognizer(is_ready, decode_stream, get_result, events)
    sys.modules["sherpa_onnx"].OnlineRecognizer = recognizer_cls
    backend = make_backend(model_files)
    backend.load()
    assert backend.capabilities().streaming is True

    assert events[0] == "create_stream"
    assert "accept_waveform" in events
    assert "decode_stream" in events
    assert "input_finished" in events

    finish_index = events.index("input_finished")
    # get_result() runs at least once before input_finished()...
    assert "get_result" in events[:finish_index]
    # ...decode_stream()/is_ready() continue afterwards to drain the tail...
    assert any(name in ("is_ready", "decode_stream") for name in events[finish_index + 1:])
    # ...and the transaction ends on a final get_result().
    assert events[-1] == "get_result"


def test_streaming_probe_feeds_multiple_small_chunks(model_files):
    """The probe must never send one big one-second (or larger) block."""
    chunk_sizes: list = []

    def is_ready(stream):
        return stream.total_samples >= 1600 and stream.decoded < stream.total_samples // 1600

    def decode_stream(stream):
        stream.decoded += 1

    def get_result(stream):
        return " ".join(f"tok{i}" for i in range(stream.decoded))

    class _Stream:
        def __init__(self) -> None:
            self.total_samples = 0
            self.decoded = 0
            self.finished = False

        def accept_waveform(self, sample_rate, waveform) -> None:
            chunk_sizes.append(len(waveform))
            self.total_samples += len(waveform)

        def input_finished(self) -> None:
            self.finished = True

    class _Recognizer(FakeOnlineRecognizer):
        def create_stream(self):
            return _Stream()

        def is_ready(self, stream):
            return is_ready(stream)

        def decode_stream(self, stream):
            decode_stream(stream)

        def get_result(self, stream):
            return get_result(stream)

    sys.modules["sherpa_onnx"].OnlineRecognizer = _Recognizer
    backend = make_backend(model_files)
    backend.load()

    # PROBE_CHUNK_S=0.1s @ 16 kHz -> 1600 samples/chunk; PROBE_MAX_AUDIO_S=8.0s
    # -> 80 chunks, never one 16000-sample (1s) or larger block.
    assert len(chunk_sizes) > 1
    assert all(size <= 1600 for size in chunk_sizes)
    assert sum(chunk_sizes) == int(8.0 * 16000)
