"""Benchmark the Shenava sherpa-onnx decoder (development tool).

Reads the model path from the usual configuration (``SHENAVA_MODEL_PATH`` /
``SHENAVA_TOKENS_PATH`` / ``--model`` / ``--tokens``) instead of a hard-coded
absolute path, and measures both the offline decode latency and the native
streaming cost.

    python tools/profile_asr.py --seconds 1 2 5 10 --threads 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from shenava_realtime.config import AppConfig  # noqa: E402
from shenava_realtime.utils import setup_logging  # noqa: E402


def synthetic_speech(seconds: float, sample_rate: int = 16000, seed: int = 42) -> np.ndarray:
    """Bursty filtered noise: closer to speech than white noise for RTF numbers."""
    rng = np.random.default_rng(seed)
    count = int(seconds * sample_rate)
    white = rng.standard_normal(count).astype(np.float32)
    kernel = np.ones(16, dtype=np.float32) / 16
    filtered = np.convolve(white, kernel, mode="same")
    envelope = (np.sin(np.linspace(0, 30 * np.pi, count)) > 0).astype(np.float32)
    return (filtered * envelope * 0.08).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="path to model.int8.onnx")
    parser.add_argument("--tokens", default=None, help="path to tokens.txt")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seconds", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--streaming", action="store_true", help="also measure native streaming / endpoint fallback")
    args = parser.parse_args()

    setup_logging("WARNING")
    config = AppConfig.from_env()
    if args.model:
        config.asr.model_path = args.model
    if args.tokens:
        config.asr.tokens_path = args.tokens
    if args.threads:
        config.asr.num_threads = args.threads

    from shenava_realtime.asr_backend import SherpaOnnxASR

    backend = SherpaOnnxASR(config.asr)
    backend.load()
    print(f"device: {backend.device} | load: {backend.load_seconds:.2f}s | "
          f"{backend.capabilities().describe()}")

    print(f"\n{'audio':>7} | {'best':>7} | {'RTF':>6} | text")
    for seconds in args.seconds:
        audio = synthetic_speech(seconds)
        times = []
        text = ""
        for _ in range(max(1, args.repeats)):
            started = time.perf_counter()
            text, _confidence = backend.transcribe(audio)
            times.append(time.perf_counter() - started)
        best = min(times)
        print(f"{seconds:>6.1f}s | {best:>6.2f}s | {best / seconds:>5.2f}x | {text[:48]}")

    if args.streaming:
        from shenava_realtime.streaming import make_decoder

        decoder = make_decoder(backend, config.asr, config.audio.sample_rate)
        audio = synthetic_speech(10.0)
        block = int(0.5 * config.audio.sample_rate)
        started = time.perf_counter()
        decodes = 0
        for index in range(0, len(audio), block):
            if decoder.push(audio[index : index + block]) is not None:
                decodes += 1
        if decoder.finalize() is not None:
            decodes += 1
        elapsed = time.perf_counter() - started
        print(
            f"\nstreaming ({decoder.name}): {decodes} decodes in {elapsed:.2f}s "
            f"for 10.0s of audio (RTF {elapsed / 10.0:.2f}x)"
        )

    print("\nStats (backend.stats(), never transcript/audio content):", backend.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
