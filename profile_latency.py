"""One-off: benchmark NeMo transcribe() latency on CPU vs segment length and thread count."""
import os
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

print(f"cpu cores: {os.cpu_count()}, torch threads default: {torch.get_num_threads()}")

import nemo.collections.asr as nemo_asr

t0 = time.time()
model = nemo_asr.models.ASRModel.restore_from(
    r"C:\Users\ali\Desktop\localASR-shenava\shenava-koochik\shenava-koochik-v1.0.nemo"
)
model.eval()
print(f"model load: {time.time() - t0:.1f}s | type: {type(model).__name__}")

rng = np.random.default_rng(42)

# Speech-like audio: filtered noise bursts (closer RTF to real speech than pure noise)
def make_audio(seconds: float) -> np.ndarray:
    n = int(16000 * seconds)
    white = rng.standard_normal(n).astype(np.float32)
    # crude low-pass via cumulative smoothing -> speech-like spectral tilt
    kernel = np.ones(16, dtype=np.float32) / 16
    filtered = np.convolve(white, kernel, mode="same")
    env = (np.sin(np.linspace(0, 30 * np.pi, n)) > 0).astype(np.float32)  # on/off bursts
    return (filtered * env * 0.08).astype(np.float32)

def bench(seconds: float, threads: int, reps: int = 3) -> float:
    torch.set_num_threads(threads)
    audio = make_audio(seconds)
    times = []
    for _ in range(reps):
        t = time.time()
        with torch.no_grad():
            model.transcribe([audio], batch_size=1, verbose=False, return_hypotheses=True)
        times.append(time.time() - t)
    return min(times)

print("\nlatency (best of 3) by segment duration and thread count:")
print(f"{'dur':>6} | {'1 thr':>8} | {'2 thr':>8} | {'4 thr':>8} | {'all':>8} | RTF(all)")
for dur in (1.0, 2.0, 5.0, 10.0, 20.0):
    r1 = bench(dur, 1)
    r2 = bench(dur, 2)
    r4 = bench(dur, 4)
    rall = bench(dur, os.cpu_count())
    rtf = rall / dur
    print(f"{dur:>5.1f}s | {r1:>7.2f}s | {r2:>7.2f}s | {r4:>7.2f}s | {rall:>7.2f}s | {rtf:.2f}x")
