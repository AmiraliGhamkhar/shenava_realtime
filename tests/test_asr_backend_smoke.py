"""Optional real-model smoke test: skipped unless model files exist.

Provision the model per ``models/shenava/README.md`` and re-run:

    pytest tests/test_asr_backend_smoke.py -q

This never runs as part of a bare ``pytest -q`` in an environment without the
model files (132 MB `model.int8.onnx` + `tokens.txt`), matching acceptance
criterion #1 (tests pass without the model).
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from shenava_realtime.config import ASRConfig, DEFAULT_MODEL_FILE, DEFAULT_TOKENS_FILE

MODEL_PATH = Path(DEFAULT_MODEL_FILE)
TOKENS_PATH = Path(DEFAULT_TOKENS_FILE)

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.is_file() and TOKENS_PATH.is_file()),
    reason="real sherpa-onnx model not provisioned (see models/shenava/README.md)",
)


def test_real_model_loads_as_a_streaming_recognizer():
    from shenava_realtime.asr_backend import SherpaOnnxASR

    backend = SherpaOnnxASR(ASRConfig())
    backend.load()
    caps = backend.capabilities()
    assert caps.streaming is True
    assert caps.provider == "cpu"


def test_real_model_decodes_silence_without_error():
    from shenava_realtime.asr_backend import SherpaOnnxASR

    backend = SherpaOnnxASR(ASRConfig())
    text, confidence = backend.transcribe(np.zeros(16000, dtype=np.float32))
    assert isinstance(text, str)
    assert confidence == 0.0


def test_real_model_stream_lifecycle_matches_the_mock_contract():
    from shenava_realtime.asr_backend import SherpaOnnxASR

    backend = SherpaOnnxASR(ASRConfig())
    stream = backend.create_stream()
    stream.accept(np.zeros(1600, dtype=np.float32))
    stream.decode_ready()
    _ = stream.partial()
    text, confidence = stream.finalize()
    assert isinstance(text, str)
    assert confidence == 0.0
    with pytest.raises(RuntimeError):
        stream.accept(np.zeros(100, dtype=np.float32))


def test_real_model_completes_the_full_streaming_lifecycle_with_chunked_audio():
    """Exercise the exact lifecycle the startup probe (and production VAD
    segments) rely on: create_stream -> feed small chunks -> drain every
    decode step the recognizer is ready for -> input_finished() -> drain the
    remainder -> get_result().

    This checks the streaming *infrastructure* against the real model, not
    ASR accuracy: the probe audio is a deterministic low-amplitude tone, and
    no assertion is made about the transcribed text being non-empty or
    correct.
    """
    from shenava_realtime.asr_backend import SherpaOnnxASR

    backend = SherpaOnnxASR(ASRConfig())
    backend.load()
    recognizer = backend.recognizer

    sample_rate = int(backend.config.sample_rate)
    chunk_s = 0.1
    chunk_samples = int(chunk_s * sample_rate)
    total_seconds = 8.0
    total_samples = int(total_seconds * sample_rate)
    t = np.arange(total_samples, dtype=np.float32) / sample_rate
    audio = (0.02 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    stream = recognizer.create_stream()
    decode_steps_before_eof = 0
    for offset in range(0, total_samples, chunk_samples):
        chunk = audio[offset:offset + chunk_samples]
        stream.accept_waveform(sample_rate, chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
            decode_steps_before_eof += 1

    mid_result = recognizer.get_result(stream)
    assert isinstance(mid_result, str)

    stream.input_finished()

    decode_steps_after_eof = 0
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
        decode_steps_after_eof += 1

    final_result = recognizer.get_result(stream)
    assert isinstance(final_result, str)

    # The full lifecycle (chunks fed + input_finished + drain) must produce
    # at least one decode step; synthetic probe audio is not required to
    # produce non-empty transcription text.
    assert decode_steps_before_eof + decode_steps_after_eof >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
