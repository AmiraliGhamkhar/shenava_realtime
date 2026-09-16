"""NeMo 2.4 cache-aware CTC adapter, batch size one.

Uses the public conformer_stream_step API and NeMo's preprocessor. Only
bounded STFT overlap is recomputed, never an encoder sliding window. Feature
normalization is online (per encoder chunk), not future/utterance dependent.
"""
from __future__ import annotations

from typing import Any
from numbers import Integral
import numpy as np


class NeMoCacheAwareStream:
    def __init__(self, model: Any, torch: Any, max_segment_s: float = 22) -> None:
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
        from nemo.collections.asr.parts.preprocessing.features import normalize_batch

        self.model, self.torch = model, torch
        helper = CacheAwareStreamingAudioBuffer(model, online_normalization=True)
        self.preprocessor = helper.preprocessor
        self.normalize_type = helper.model_normalize_type
        self.normalize_batch = normalize_batch
        self.cfg = model.encoder.streaming_cfg
        # Frontend audit: the app captures 16 kHz mono, so a checkpoint whose
        # frontend is configured for anything else must fail at startup.
        sample_rate = getattr(model.cfg.preprocessor, "sample_rate", 16000)
        if int(sample_rate) != 16000:
            raise RuntimeError(
                f"Shenava requires a 16 kHz mono frontend, got sample_rate={sample_rate}"
            )
        self.hop = int(round(model.cfg.preprocessor.window_stride * 16000))
        self.n_fft = int(self.preprocessor.featurizer.n_fft)
        if (getattr(self.preprocessor.featurizer, "exact_pad", False)
                or getattr(self.preprocessor.featurizer, "frame_splicing", 1) != 1):
            raise RuntimeError("Native frontend requires centered STFT without frame splicing")
        self.margin = ((self.n_fft + self.hop - 1) // self.hop + 1) * self.hop
        self.limit = int(max_segment_s * 16000)
        self.reset()

    def reset(self) -> None:
        self.channel, self.time, self.cache_len = self.model.encoder.get_initial_cache_state(batch_size=1)
        self.predictions = self.hypotheses = None
        self.raw = np.empty(0, np.float32)
        self.raw_start = self.next_frame = self.total = 0
        self.features = None
        self.position = self.step = 0
        self.text = ""
        self.closed = False

    def _size(self, value: Any) -> int:
        if isinstance(value, Integral):
            return int(value)
        return int(value[0 if self.step == 0 else 1])

    def push(self, audio: np.ndarray) -> tuple[str, float]:
        if self.closed:
            raise RuntimeError("Stream already finalized; reset before reuse")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError("Non-finite audio")
        if self.total + samples.size > self.limit:
            raise ValueError("ASR segment limit exceeded; reset at an endpoint")
        # Bound temporary frontend allocations even with file-sized pushes.
        for offset in range(0, samples.size, 1600):
            block = samples[offset:offset + 1600]
            self.raw = np.concatenate((self.raw, block))
            self.total += block.size
            self._features(final=False)
            self._decode(final=False)
        return self.text, 0.0

    def _features(self, final: bool) -> None:
        if not self.raw.size:
            return
        if self.raw.size <= self.n_fft // 2:
            if final:
                raise ValueError("Audio too short for the configured STFT")
            return
        # NeMo's centered STFT needs right context; never accept a frame whose
        # right edge is still moving. Retain aligned left overlap for preemphasis.
        stable = self.total // self.hop + 1 if final else max(0, (self.total - self.n_fft // 2) // self.hop + 1)
        if stable <= self.next_frame:
            return
        torch = self.torch
        with torch.inference_mode():
            signal = torch.from_numpy(self.raw).unsqueeze(0).to(self.model.device)
            length = torch.tensor([self.raw.size], device=self.model.device, dtype=torch.long)
            mel, lengths = self.preprocessor(input_signal=signal, length=length)
            begin = self.next_frame - self.raw_start // self.hop
            end = min(stable - self.raw_start // self.hop, int(lengths[0]))
            new = mel[:, :, begin:end]
            if new.size(-1):
                self.features = new.clone() if self.features is None else torch.cat((self.features, new), dim=-1)
                self.next_frame += new.size(-1)
        drop = max(0, self.next_frame * self.hop - self.margin - self.raw_start)
        self.raw = self.raw[drop:].copy()
        self.raw_start += drop

    def _decode(self, final: bool) -> None:
        torch = self.torch
        while self.features is not None:
            available = self.features.size(-1) - self.position
            chunk_size = self._size(self.cfg.chunk_size)
            shift = self._size(self.cfg.shift_size)
            if available <= 0 or (not final and available <= chunk_size):
                break  # retain one chunk so EOF can use keep_all_outputs
            count = min(chunk_size, available)
            cache_size = self._size(self.cfg.pre_encode_cache_size)
            start = max(0, self.position - cache_size)
            chunk = self.features[:, :, start:self.position + count]
            missing = cache_size - (self.position - start)
            if missing:
                chunk = torch.nn.functional.pad(chunk, (missing, 0))
            lengths = torch.tensor([chunk.size(-1)], dtype=torch.long, device=self.model.device)
            last = final and available <= chunk_size
            if last and count < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - count))
            with torch.inference_mode():
                chunk, _, _ = self.normalize_batch(chunk, lengths, normalize_type=self.normalize_type)
                result = self.model.conformer_stream_step(
                    processed_signal=chunk, processed_signal_length=lengths,
                    cache_last_channel=self.channel, cache_last_time=self.time,
                    cache_last_channel_len=self.cache_len,
                    previous_hypotheses=self.hypotheses, previous_pred_out=self.predictions,
                    drop_extra_pre_encoded=0 if self.step == 0 else self.cfg.drop_extra_pre_encoded,
                    keep_all_outputs=last, return_transcription=True,
                )
            self.predictions, texts, self.channel, self.time, self.cache_len, self.hypotheses = result
            if texts:
                self.text = (texts[0] if isinstance(texts[0], str) else texts[0].text).strip()
            self.step += 1
            self.position += count if last else shift
            keep = self._size(self.cfg.pre_encode_cache_size)
            drop = max(0, self.position - keep)
            self.features = self.features[:, :, drop:].clone()
            self.position -= drop
            if last:
                break

    def finalize(self) -> tuple[str, float]:
        if not self.closed:
            self._features(final=True)
            self._decode(final=True)
            self.closed = True
        return self.text, 0.0
