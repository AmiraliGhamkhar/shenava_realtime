"""One-off: measure mic ambient noise floor + level distribution for VAD tuning."""
import numpy as np
import sounddevice as sd

RATE = 16000
CHUNK = 1024  # matches AudioConfig.chunk_size
SECONDS = 10

print(f"Recording {SECONDS}s of ambient mic audio at {RATE} Hz...")
frames = []


def callback(indata, frames_count, time_info, status):
    frames.append(indata[:, 0].copy())


with sd.InputStream(channels=1, samplerate=RATE, blocksize=CHUNK, callback=callback):
    import time
    time.sleep(SECONDS)

audio = np.concatenate(frames).astype(np.float32)
n = len(audio)
energy = np.sqrt(np.mean(audio.reshape(-1, CHUNK)[: (n // CHUNK)] ** 2, axis=1))

print(f"chunks measured: {len(energy)}")
print(f"RMS energy: min={energy.min():.5f} p50={np.percentile(energy, 50):.5f} "
      f"p90={np.percentile(energy, 90):.5f} p99={np.percentile(energy, 99):.5f} max={energy.max():.5f}")
zcr = np.array([
    np.sum(np.diff(np.sign(audio[i * CHUNK:(i + 1) * CHUNK])) != 0) / CHUNK
    for i in range(n // CHUNK)
])
print(f"ZCR: p50={np.percentile(zcr, 50):.3f} p90={np.percentile(zcr, 90):.3f}")
print("Note: VAD fires speech if energy > threshold OR (0.05 < ZCR < 0.5 AND energy > threshold*0.5)")
