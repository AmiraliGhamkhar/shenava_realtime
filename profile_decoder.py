"""One-off: compare RNNT vs CTC decoding latency for the hybrid model on CPU."""
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.restore_from(
    r"C:\Users\ali\Desktop\shenava_realtime\shenava-koochik\shenava-koochik-v1.0.nemo"
)
model.eval()
torch.set_num_threads(4)

rng = np.random.default_rng(42)

def make_audio(seconds: float) -> np.ndarray:
    n = int(16000 * seconds)
    white = rng.standard_normal(n).astype(np.float32)
    kernel = np.ones(16, dtype=np.float32) / 16
    filtered = np.convolve(white, kernel, mode="same")
    env = (np.sin(np.linspace(0, 30 * np.pi, n)) > 0).astype(np.float32)
    return (filtered * env * 0.08).astype(np.float32)

audio = make_audio(10.0)

def bench(label: str, reps: int = 3) -> None:
    times = []
    for _ in range(reps):
        t = time.time()
        with torch.no_grad():
            model.transcribe([audio], batch_size=1, verbose=False, return_hypotheses=True)
        times.append(time.time() - t)
    print(f"{label}: best={min(times):.2f}s  all={[f'{t:.2f}' for t in times]}")

print("decoding strategy currently:", type(model.decoding).__name__)
bench("RNNT 10s")

model.change_decoding_strategy(decoder_type="ctc")
print("switched to:", type(model.decoding).__name__)
bench("CTC  10s")
