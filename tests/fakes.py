"""Test doubles: no model, no GPU, no audio hardware."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from shenava_realtime.pipeline import TranscriptionPipeline
from shenava_realtime.streaming import DecodeResult, StreamingDecoder


def frames(seconds: float, amplitude: float = 0.0, sample_rate: int = 16000, block_ms: int = 64) -> List[np.ndarray]:
    """Split ``seconds`` of constant-amplitude audio into capture-sized blocks."""
    total = int(round(seconds * sample_rate))
    block = int(round(block_ms / 1000.0 * sample_rate))
    return [np.full(block, amplitude, dtype=np.float32) for _ in range(total // block)]


class ScriptedDecoder(StreamingDecoder):
    """Returns a canned hypothesis per ``push`` (the last one repeats)."""

    name = "scripted"

    def __init__(self, hypotheses: Sequence[str], confidence: float = 0.9) -> None:
        self._hypotheses = list(hypotheses)
        self._index = 0
        self._confidence = confidence
        self.pushes = 0
        self.resets = 0
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self._index = 0

    def push(self, audio: np.ndarray, force: bool = False) -> Optional[DecodeResult]:
        if not self._hypotheses:
            return None
        self.pushes += 1
        if force and self._index >= len(self._hypotheses):
            text = self._hypotheses[-1]
        else:
            text = self._hypotheses[min(self._index, len(self._hypotheses) - 1)]
            self._index += 1
        reset = False
        if text.startswith("RESET:"):
            reset = True
            text = text[len("RESET:") :]
        return DecodeResult(text=text, confidence=self._confidence, reset=reset)


class RecordingTranscriber:
    """Stands in for the ASR model; records the length of every decode call."""

    def __init__(self, transcript_fn: Optional[Callable[[float], str]] = None) -> None:
        self.calls: List[float] = []
        self._transcript_fn = transcript_fn or (lambda seconds: " ".join(["واژه"] * max(1, int(seconds))))

    def __call__(self, audio: np.ndarray) -> Tuple[str, float]:
        seconds = len(audio) / 16000.0
        self.calls.append(seconds)
        return self._transcript_fn(seconds), 0.8


class FakeAudioCapture:
    """Replaces AudioCapture so the engine can run without a microphone."""

    def __init__(self) -> None:
        self.on_speech_start = None
        self.on_speech_end = None
        self.on_audio = None
        self.started = 0
        self.stopped = 0
        self.cleared = 0
        self._paused = False
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start(self) -> None:
        self.started += 1
        self._running = True

    def stop(self) -> None:
        self.stopped += 1
        self._running = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def clear(self) -> None:
        self.cleared += 1

    def get_statistics(self) -> dict:
        return {"chunks": 0, "segments": 0, "vad_state": "silence"}

    # --- helpers used by tests -------------------------------------------
    def emit_utterance(self, blocks: Sequence[np.ndarray], preroll_seconds: float = 0.0) -> None:
        if self.on_speech_start is not None:
            self.on_speech_start(np.zeros(int(preroll_seconds * 16000), dtype=np.float32))
        for block in blocks:
            if self.on_audio is not None:
                self.on_audio(block, True)
        if self.on_speech_end is not None:
            self.on_speech_end(len(blocks) * len(blocks[0]) / 16000.0 if len(blocks) else 0.0)


class FakeBackend:
    """Minimal ASR backend: decodes by looking at the audio length."""

    def __init__(self, transcriber: Optional[RecordingTranscriber] = None) -> None:
        self.transcriber = transcriber or RecordingTranscriber()
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio: np.ndarray) -> Tuple[str, float]:
        return self.transcriber(audio)

    def create_stream(self):
        return None


def make_pipeline(hypotheses: Sequence[str], **kwargs) -> Tuple[TranscriptionPipeline, ScriptedDecoder]:
    decoder = ScriptedDecoder(hypotheses)
    return TranscriptionPipeline(decoder=decoder, **kwargs), decoder
