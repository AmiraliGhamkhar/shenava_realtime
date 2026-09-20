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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
