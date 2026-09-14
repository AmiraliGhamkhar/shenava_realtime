"""Real-time ASR engine: microphone -> VAD -> streaming CTC -> stable text.

Threads:

* ``audio-capture`` (owned by :class:`AudioCapture`) runs the VAD and pushes
  work items onto ``self._queue``;
* ``asr-worker`` (owned here) is the only thread that touches the model, the
  stabilizer and the post-processor.

Callbacks are invoked on the worker thread and are expected to be cheap; the app
forwards them to the overlay queue and the injector queue.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .audio_capture import AudioCapture
from .config import AppConfig
from .pipeline import TranscriptionPipeline
from .postprocessor import PostProcessor
from .streaming import make_decoder
from .utils import TextBuffer

logger = logging.getLogger(__name__)

_START = "start"
_AUDIO = "audio"
_END = "end"
_CLEAR = "clear"


class RealtimeASR:
    """Streaming transcription engine with stable, one-shot text emission."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        backend: Any = None,
        audio_capture: Optional[AudioCapture] = None,
        pipeline: Optional[TranscriptionPipeline] = None,
    ) -> None:
        self.config = config or AppConfig()
        self.asr_config = self.config.asr

        if backend is None:
            from .asr_backend import NeMoASR  # lazy: keeps torch/NeMo out of tests

            backend = NeMoASR(self.asr_config)
        self.backend = backend
        load = getattr(backend, "load", None)
        if callable(load):
            load()

        self.audio_capture = audio_capture or AudioCapture(self.config.audio)
        self.audio_capture.on_speech_start = self._on_speech_start
        self.audio_capture.on_speech_end = self._on_speech_end
        self.audio_capture.on_audio = self._on_audio

        postprocessor = PostProcessor(self.config.postprocess)
        decoder = make_decoder(self.backend, self.asr_config, self.config.audio.sample_rate)
        self.pipeline = pipeline or TranscriptionPipeline(
            decoder=decoder,
            postprocessor=postprocessor,
            holdback_words=self.asr_config.holdback_words,
        )
        self.postprocessor = self.pipeline.postprocessor

        # Output callbacks
        self.on_partial: Optional[Callable[[str, float], None]] = None
        self.on_text_delta: Optional[Callable[[str, float], None]] = None
        self.on_utterance_end: Optional[Callable[[str, float], None]] = None

        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=512)
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._transcript = TextBuffer()
        self._partial_text = ""
        self._needs_separator = False

        self.stats: Dict[str, float] = {
            "utterances": 0,
            "decodes": 0,
            "decode_seconds": 0.0,
            "audio_seconds": 0.0,
            "average_confidence": 0.0,
        }
        self._confidence_sum = 0.0

    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def transcript(self) -> str:
        return self._transcript.get_text()

    @property
    def partial_transcript(self) -> str:
        return self._partial_text

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        with self._lock:
            if self.is_running:
                logger.debug("ASR engine already running")
                return
            self._worker = threading.Thread(target=self._work_loop, name="asr-worker", daemon=True)
            self._worker.start()
        try:
            self.audio_capture.start()
        except Exception:
            self._stop_worker()
            raise
        logger.info("real-time transcription started (output mode: %s)", self.config.output_mode.value)

    def stop(self) -> None:
        with self._lock:
            worker = self._worker
            if worker is None and not self.audio_capture.is_running:
                return
            self._worker = None
        try:
            self.audio_capture.stop()
        except Exception:
            logger.exception("error while stopping audio capture")
        if worker is not None:
            self._enqueue(None)
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning("ASR worker did not exit within 5s")
        if self.config.save_transcripts:
            self.save_transcript()
        logger.info(
            "real-time transcription stopped (%d utterances, %d decodes)",
            int(self.stats["utterances"]),
            int(self.stats["decodes"]),
        )

    def clear_transcript(self) -> None:
        self._needs_separator = False
        self._enqueue((_CLEAR,))
        self._transcript.clear()
        self._partial_text = ""
        self.audio_capture.clear()
        logger.info("transcript cleared")

    # ------------------------------------------------------------------ #
    # Audio callbacks (run on the audio-capture thread: enqueue only)
    # ------------------------------------------------------------------ #
    def _on_speech_start(self, preroll: np.ndarray) -> None:
        self._enqueue((_START, preroll))

    def _on_audio(self, chunk: np.ndarray, in_speech: bool) -> None:
        if not in_speech:
            return
        self._enqueue((_AUDIO, chunk))

    def _on_speech_end(self, duration: float) -> None:
        self._enqueue((_END, duration))

    def _enqueue(self, item: Optional[tuple]) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning("ASR queue is full; dropping the oldest work item")
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except queue.Empty:  # pragma: no cover - defensive
                pass

    # ------------------------------------------------------------------ #
    def _work_loop(self) -> None:
        logger.debug("ASR worker started")
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self._handle(item)
            except Exception:
                logger.exception("error while handling %s", item[0] if item else item)
        logger.debug("ASR worker stopped")

    def _handle(self, item: tuple) -> None:
        kind = item[0]
        if kind == _START:
            self.pipeline.start_utterance(item[1])
            self._partial_text = ""
        elif kind == _AUDIO:
            self._consume_deltas(self.pipeline.push_audio(item[1]))
        elif kind == _END:
            duration = float(item[1])
            self.stats["audio_seconds"] += duration
            self.stats["utterances"] += 1
            deltas = self.pipeline.end_utterance()
            self._consume_deltas(deltas)
            self.stats["decode_seconds"] = self.pipeline.decode_seconds
            confidence = self.pipeline.confidence
            if confidence > 0.0:
                self._confidence_sum += confidence
                self.stats["average_confidence"] = self._confidence_sum / max(1.0, self.stats["utterances"])
            text = self.pipeline.committed_text
            self._partial_text = ""
            if text:
                self._needs_separator = True
                self._transcript.add_text(text)
                self._notify(self.on_utterance_end, text, self.pipeline.confidence)
                logger.info("[%.1fs] %s", duration, text)
        elif kind == _CLEAR:
            self.pipeline.abort()
            self._partial_text = ""
            self._needs_separator = False

    def _consume_deltas(self, deltas: List[str]) -> None:
        for delta in deltas:
            if not delta:
                continue
            if self._needs_separator and not delta.startswith(" "):
                # Utterances are separate decodes; keep a space between them so
                # injected text does not run together ("CABG" + "بیمار").
                delta = " " + delta
            self._needs_separator = False
            self._notify(self.on_text_delta, delta, self.pipeline.confidence)
        self.stats["decodes"] = self.pipeline.decodes
        partial = self.pipeline.partial_text
        if partial and partial != self._partial_text:
            self._partial_text = partial
            self._notify(self.on_partial, partial, self.pipeline.confidence)

    @staticmethod
    def _notify(callback: Optional[Callable[[str, float], None]], text: str, confidence: float) -> None:
        if callback is None:
            return
        try:
            callback(text, confidence)
        except Exception:
            logger.exception("callback %s failed", getattr(callback, "__name__", callback))

    # ------------------------------------------------------------------ #
    def _stop_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            self._enqueue(None)
            worker.join(timeout=5.0)

    def save_transcript(self) -> Optional[Path]:
        """Persist the session transcript; returns the path when written."""
        text = self._transcript.get_text().strip()
        if not text:
            return None
        directory = Path(self.config.transcripts_dir)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"transcript_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            path.write_text(text + "\n", encoding="utf-8")
        except OSError:
            logger.exception("could not save the transcript")
            return None
        logger.info("transcript saved to %s", path)
        return path

    def get_statistics(self) -> Dict[str, Any]:
        audio_stats = self.audio_capture.get_statistics()
        return {
            **self.stats,
            "decoder": getattr(self.pipeline.decoder, "name", "unknown"),
            "total_audio_duration": self.stats["audio_seconds"],
            "total_processing_time": self.stats["decode_seconds"],
            "num_chunks_processed": int(self.stats["decodes"]),
            "average_confidence": self.stats["average_confidence"],
            "audio": audio_stats,
        }
