"""A mock ``sherpa_onnx`` module: no 132 MB model, no ONNX Runtime.

Mimics only the surface :mod:`shenava_realtime.asr_backend` uses:
``OnlineRecognizer.from_nemo_ctc(...)``, ``create_stream()``,
``accept_waveform()``, ``is_ready()``, ``decode_stream()``, ``get_result()``,
``input_finished()``.  Decoding is deterministic: each stream accumulates a
token per 1600-sample (100 ms) chunk it has accepted and not yet decoded, so
tests can assert exactly how much audio was fed and how many decode steps ran.
"""
from __future__ import annotations

from typing import Any, List


class FakeOnlineStream:
    def __init__(self, chunk_samples: int = 1600) -> None:
        self.chunk_samples = chunk_samples
        self.samples: List[float] = []
        self.pending = 0  # samples accepted but not yet consumed by decode_stream
        self.finished = False
        self.tokens = 0

    def accept_waveform(self, sample_rate, waveform) -> None:
        if int(sample_rate) != 16000:
            raise RuntimeError(f"unexpected sample rate {sample_rate}")
        seq = list(waveform)
        self.samples.extend(seq)
        self.pending += len(seq)

    def input_finished(self) -> None:
        self.finished = True


class FakeOnlineRecognizer:
    """Records every construction call so tests can assert on kwargs."""

    last_kwargs: dict = {}
    last_instance: "FakeOnlineRecognizer | None" = None
    fail_init: bool = False
    streams_created: int = 0

    def __init__(self, **kwargs: Any) -> None:
        if FakeOnlineRecognizer.fail_init:
            raise RuntimeError("synthetic init failure")
        self.kwargs = kwargs
        self.events: List[str] = []
        FakeOnlineRecognizer.last_kwargs = kwargs
        FakeOnlineRecognizer.last_instance = self

    @classmethod
    def from_nemo_ctc(cls, **kwargs: Any) -> "FakeOnlineRecognizer":
        return cls(**kwargs)

    def create_stream(self) -> FakeOnlineStream:
        self.events.append("create_stream")
        FakeOnlineRecognizer.streams_created += 1
        return FakeOnlineStream()

    def is_ready(self, stream: FakeOnlineStream) -> bool:
        self.events.append("is_ready")
        return stream.pending >= stream.chunk_samples

    def decode_stream(self, stream: FakeOnlineStream) -> None:
        self.events.append("decode_stream")
        if stream.pending < stream.chunk_samples:
            return
        stream.pending -= stream.chunk_samples
        stream.tokens += 1

    def get_result(self, stream: FakeOnlineStream) -> str:
        self.events.append("get_result")
        return " ".join(f"tok{i}" for i in range(stream.tokens))

    def reset(self, stream: FakeOnlineStream) -> None:
        self.events.append("reset")
        stream.tokens = 0
        stream.pending = 0
        stream.samples = []
