"""Microphone capture: 16 kHz mono, bounded queue, RMS VAD segmentation.

Threading model (exactly two threads, no shared mutable audio state):

* the PortAudio callback only copies the block and pushes it on a bounded
  queue — it never blocks on inference and never touches the VAD;
* a single consumer thread pops frames, runs the VAD and fires callbacks.

Shutdown is explicit: ``stop()`` closes the stream, pushes a sentinel, joins
the consumer thread, and the consumer flushes any open utterance before it
exits, so the last sentence is not lost and no thread outlives the object.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional at import time so tests run without it
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None

from .config import AudioConfig
from .vad import EnergyVAD, EventType, VADEvent, VADConfig, VADState

logger = logging.getLogger(__name__)

# Control marker pushed through the audio queue so that state changes happen on
# the consumer thread (the only thread allowed to touch the VAD).
_CLEAR = object()


class AudioCapture:
    """Captures microphone audio and segments it into utterances."""

    def __init__(self, config: Optional[AudioConfig] = None, vad: Optional[EnergyVAD] = None) -> None:
        self.config = config or AudioConfig()
        self.vad = vad or EnergyVAD(
            VADConfig(
                sample_rate=self.config.sample_rate,
                onset_rms=self.config.vad_onset_rms,
                offset_rms=self.config.vad_offset_rms,
                min_speech_ms=self.config.vad_min_speech_ms,
                min_silence_ms=self.config.vad_min_silence_ms,
                pre_speech_ms=self.config.vad_pre_speech_ms,
                max_speech_s=self.config.vad_max_speech_s,
            )
        )

        # Callbacks (set by the engine; invoked on the consumer thread only).
        self.on_speech_start: Optional[Callable[[np.ndarray], None]] = None
        self.on_speech_end: Optional[Callable[[float], None]] = None
        self.on_audio: Optional[Callable[[np.ndarray, bool], None]] = None

        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(
            maxsize=max(8, self.config.queue_max_chunks)
        )
        self._thread: Optional[threading.Thread] = None
        self._stream: Any = None
        self._paused = False
        self._lock = threading.RLock()
        self._dropped_chunks = 0
        self._warned_drop = False

        self.stats: Dict[str, float] = {
            "chunks": 0,
            "dropped_chunks": 0,
            "speech_chunks": 0,
            "audio_seconds": 0.0,
            "speech_seconds": 0.0,
            "segments": 0,
        }

    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_speaking(self) -> bool:
        return self.vad.state is VADState.SPEECH

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Open the input stream and start the consumer thread."""
        if sd is None:
            raise RuntimeError("sounddevice is required for audio capture (pip install sounddevice)")
        with self._lock:
            if self.is_running:
                logger.debug("audio capture already running")
                return

            logger.info(
                "starting audio capture: %d Hz, %d ch, %d-sample blocks",
                self.config.sample_rate,
                self.config.channels,
                self.config.chunk_size,
            )
            self.vad.reset()
            self._paused = False
            self._thread = threading.Thread(target=self._consume_loop, name="audio-capture", daemon=True)
            self._thread.start()

            try:
                self._stream = sd.InputStream(
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    blocksize=self.config.chunk_size,
                    dtype="float32",
                    device=self.config.device,
                    callback=self._audio_callback,
                )
                self._stream.start()
            except Exception:
                logger.exception("could not open the microphone")
                self._shutdown_thread()
                raise
            logger.info("audio capture running")

    def stop(self) -> None:
        """Close the stream, join the consumer and flush the open segment."""
        with self._lock:
            thread = self._thread
            stream = self._stream
            if thread is None and stream is None:
                return
            logger.info("stopping audio capture")
            self._thread = None
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("error while closing the audio stream")

        if thread is not None:
            self._push_sentinel()
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.warning("audio consumer thread did not exit within 3s")
        logger.info("audio capture stopped")

    def pause(self) -> None:
        self._paused = True
        logger.info("audio capture paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("audio capture resumed")

    def clear(self) -> None:
        """Reset the detector (hotkey: clear).

        The request travels through the audio queue so the VAD is only ever
        mutated by the consumer thread.
        """
        try:
            self._queue.put_nowait(_CLEAR)
        except queue.Full:
            logger.debug("audio queue full; the clear request was dropped")

    # ------------------------------------------------------------------ #
    def _audio_callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - I/O
        if status:
            logger.warning("audio stream status: %s", status)
        if self._paused:
            return
        try:
            block = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
            self._queue.put_nowait(block)
        except queue.Full:
            self._dropped_chunks += 1
            if not self._warned_drop:
                self._warned_drop = True
                logger.warning("audio queue full (%d); dropping blocks until it drains", self._queue.maxsize)

    def _push_sentinel(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            with self._queue.mutex:
                self._queue.queue.clear()
            try:
                self._queue.put_nowait(None)
            except queue.Full:  # pragma: no cover - defensive
                logger.error("could not signal the audio consumer to stop")

    def _shutdown_thread(self) -> None:
        thread, self._thread = self._thread, None
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("error while closing the audio stream")
        if thread is not None:
            self._push_sentinel()
            thread.join(timeout=3.0)

    # ------------------------------------------------------------------ #
    def _consume_loop(self) -> None:
        try:
            while True:
                block = self._queue.get()
                if block is None:
                    break
                self._process_item(block)
        except Exception:
            logger.exception("audio consumer crashed")
        finally:
            try:
                for event in self.vad.flush():
                    self._dispatch(event)
            except Exception:
                logger.exception("error while flushing the VAD segment")
            self.stats["dropped_chunks"] = self._dropped_chunks

    def _process_item(self, block) -> None:
        if block is _CLEAR:
            self.vad.reset()
            logger.debug("VAD reset")
            return
        self._consume(block)

    def _consume(self, block: np.ndarray) -> None:
        duration = block.size / float(self.config.sample_rate)

        # Hand the block to the streaming consumer *before* the boundary
        # events, so an utterance always has all of its audio when it ends.
        in_speech = self.vad.state is not VADState.SILENCE
        if self.on_audio is not None:
            try:
                self.on_audio(block, in_speech)
            except Exception:
                logger.exception("on_audio callback failed")

        try:
            events = self.vad.process(block)
        except Exception:
            logger.exception("VAD failed on a %d-sample block", block.size)
            return

        self.stats["chunks"] += 1
        self.stats["audio_seconds"] += duration
        if self.vad.state is not VADState.SILENCE:
            self.stats["speech_chunks"] += 1
            self.stats["speech_seconds"] += duration

        for event in events:
            self._dispatch(event)

    def _dispatch(self, event: VADEvent) -> None:
        if event.type is EventType.SPEECH_START:
            self.stats["segments"] += 1
            logger.info("speech started (%.2fs pre-roll)", event.duration_s)
            if self.on_speech_start is not None:
                try:
                    self.on_speech_start(event.audio if event.audio is not None else np.zeros(0, dtype=np.float32))
                except Exception:
                    logger.exception("on_speech_start callback failed")
        else:
            logger.info("speech ended (%.2fs)", event.duration_s)
            if self.on_speech_end is not None:
                try:
                    self.on_speech_end(event.duration_s)
                except Exception:
                    logger.exception("on_speech_end callback failed")

    # ------------------------------------------------------------------ #
    def get_statistics(self) -> Dict[str, float]:
        stats = dict(self.stats)
        stats["dropped_chunks"] = self._dropped_chunks
        stats["vad_state"] = self.vad.state.value
        return stats

    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """List input devices (empty list when sounddevice is unavailable)."""
        if sd is None:
            logger.warning("sounddevice is not installed; cannot list audio devices")
            return []
        devices = []
        for index, device in enumerate(sd.query_devices()):
            if device.get("max_input_channels", 0) > 0:
                devices.append(
                    {
                        "index": index,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "default_samplerate": device["default_samplerate"],
                    }
                )
        return devices
