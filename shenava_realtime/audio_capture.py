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
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional at import time so tests run without it
    import sounddevice as sd
except (ImportError, OSError):  # pragma: no cover
    sd = None

from .audio_diagnostics import analyze_segment, format_summary, to_dict
from .config import AudioConfig
from .vad import EnergyVAD, EventType, VADEvent, VADConfig, VADState

logger = logging.getLogger(__name__)

# Control marker pushed through the audio queue so that state changes happen on
# the consumer thread (the only thread allowed to touch the VAD).
_CLEAR = object()
_PAUSE = object()

# Microphone-level diagnostics are observation only.  They make it obvious when
# PortAudio is delivering frames but the selected input is effectively silent;
# they never change samples or VAD thresholds.
_MIC_LEVEL_WINDOW_S = 2.0
_MIC_LOW_LEVEL_FRACTION = 0.25


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
                adaptive=self.config.vad_adaptive,
                noise_init_ms=self.config.vad_noise_init_ms,
                noise_halflife_ms=self.config.vad_noise_halflife_ms,
                onset_snr=self.config.vad_onset_snr,
                offset_snr=self.config.vad_offset_snr,
                adaptive_max_gain=self.config.vad_adaptive_max_gain,
            )
        )

        # Callbacks (set by the engine; invoked on the consumer thread only).
        self.on_speech_start: Optional[Callable[[np.ndarray], None]] = None
        self.on_speech_end: Optional[Callable[[float, bool], None]] = None
        self.on_discontinuity: Optional[Callable[[], None]] = None
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
        self._discontinuity = threading.Event()
        # Device-dropout heartbeat state (consumer thread only, except the
        # resume() baseline reset which takes the lock).
        self._stopping = False
        self._dropout_reported = False
        self._last_block_time = 0.0
        self.last_error: Optional[str] = None

        # Rolling microphone levels (consumer thread only).  Used only for
        # startup diagnostics: frames-arriving-but-silent is a different
        # failure mode from a stalled/disconnected input device.
        self._level_window: Deque[tuple[float, float, int]] = deque()
        self._level_window_samples = 0
        self._mic_level_reported = False
        self._low_level_reported = False

        self.stats: Dict[str, float] = {
            "chunks": 0,
            "dropped_chunks": 0,
            "speech_chunks": 0,
            "audio_seconds": 0.0,
            "speech_seconds": 0.0,
            "segments": 0,
        }
        self._last_diagnostics = None

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
            self._discontinuity.clear()
            self._paused = False
            self._stopping = False
            self._dropout_reported = False
            self._reset_level_diagnostics()
            self.last_error = None
            with self._lock:
                self._last_block_time = time.monotonic()
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
            self._stream = None
            self._stopping = True

        if stream is not None:
            try:
                try:
                    stream.stop()
                finally:
                    stream.close()
            except Exception:
                logger.exception("error while closing the audio stream")

        if thread is not None:
            self._push_sentinel()
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.warning("audio consumer thread did not exit within 3s; restart blocked")
            else:
                self._thread = None
        logger.info("audio capture stopped")

    def pause(self) -> None:
        self._paused = True
        try:
            self._queue.put(_PAUSE, timeout=1)
        except queue.Full:
            self._discontinuity.set()
        logger.info("audio capture paused")

    def resume(self) -> None:
        self._paused = False
        # The consumer received no blocks while paused; restart the dropout
        # heartbeat baseline so resume does not read as a stall.
        with self._lock:
            self._last_block_time = time.monotonic()
        logger.info("audio capture resumed")

    def clear(self) -> None:
        """Reset the detector (hotkey: clear).

        The request travels through the audio queue so the VAD is only ever
        mutated by the consumer thread.
        """
        try:
            self._queue.put_nowait(_CLEAR)
        except queue.Full:
            self._discontinuity.set()
            logger.warning("audio queue full; scheduling detector reset")

    # ------------------------------------------------------------------ #
    def _audio_callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - I/O
        if status:
            if getattr(status, "input_overflow", False):
                self._discontinuity.set()
            logger.warning("audio stream status: %s", status)
        if self._paused:
            return
        try:
            block = np.array(indata[:, 0], dtype=np.float32, copy=True)
            self._queue.put_nowait(block)
        except queue.Full:
            self._discontinuity.set()
            self._dropped_chunks += 1
            if not self._warned_drop:
                self._warned_drop = True
                logger.warning("audio queue full (%d); dropping blocks until it drains", self._queue.maxsize)

    def _push_sentinel(self) -> None:
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            self._discontinuity.set()
            with self._queue.mutex:
                self._queue.queue.clear()
            try:
                self._queue.put_nowait(None)
            except queue.Full:  # pragma: no cover - defensive
                logger.error("could not signal the audio consumer to stop")

    def _shutdown_thread(self) -> None:
        self._stopping = True
        thread = self._thread
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                try:
                    stream.stop()
                finally:
                    stream.close()
            except Exception:
                logger.exception("error while closing the audio stream")
        if thread is not None:
            self._push_sentinel()
            thread.join(timeout=3.0)
            if not thread.is_alive():
                self._thread = None

    # ------------------------------------------------------------------ #
    def _consume_loop(self) -> None:
        # The microphone callback pushes blocks continuously, so a quiet
        # queue means the device stopped delivering audio, not that the
        # speaker paused. Poll with a timeout to detect that explicitly.
        poll = max(0.05, min(0.5, self.config.dropout_timeout_s / 4.0))
        try:
            while True:
                try:
                    block = self._queue.get(timeout=poll)
                except queue.Empty:
                    self._check_heartbeat()
                    continue
                if block is None:
                    break
                self._note_block_arrival()
                self._process_item(block)
        except Exception:
            logger.exception("audio consumer crashed")
            if self.on_discontinuity:
                self.on_discontinuity()
            self.vad.reset()
        finally:
            try:
                if self._discontinuity.is_set():
                    self.vad.reset()
                    if self.on_discontinuity:
                        self.on_discontinuity()
                for event in self.vad.flush():
                    self._dispatch(event)
            except Exception:
                logger.exception("error while flushing the VAD segment")
            self.stats["dropped_chunks"] = self._dropped_chunks

    def _note_block_arrival(self) -> None:
        with self._lock:
            self._last_block_time = time.monotonic()
        if self._dropout_reported:
            self._dropout_reported = False
            logger.warning("microphone blocks resumed after a capture dropout")

    def _reset_level_diagnostics(self) -> None:
        self._level_window.clear()
        self._level_window_samples = 0
        self._mic_level_reported = False
        self._low_level_reported = False

    def _observe_microphone_level(self, block: np.ndarray) -> None:
        """Log microphone level once; never alter audio or VAD thresholds."""
        samples = np.asarray(block, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        peak = float(np.max(np.abs(samples)))
        self.stats["microphone_rms"] = rms
        self.stats["microphone_peak"] = peak

        self._level_window.append((rms, peak, int(samples.size)))
        self._level_window_samples += int(samples.size)
        max_samples = int(round(_MIC_LEVEL_WINDOW_S * self.config.sample_rate))
        window_reached = self._level_window_samples >= max_samples
        while self._level_window_samples > max_samples and len(self._level_window) > 1:
            _old_rms, _old_peak, old_samples = self._level_window.popleft()
            self._level_window_samples -= old_samples

        onset = float(getattr(self.vad, "onset_threshold", self.config.vad_onset_rms))
        offset = float(getattr(self.vad, "offset_threshold", self.config.vad_offset_rms))
        low_level_rms = max(1e-4, min(onset, offset) * _MIC_LOW_LEVEL_FRACTION)
        if not self._mic_level_reported and rms >= low_level_rms:
            self._mic_level_reported = True
            logger.info("microphone input detected: rms=%.4f peak=%.4f", rms, peak)

        if self._low_level_reported or self.stats.get("segments", 0) > 0:
            return
        if self.vad.state is not VADState.SILENCE:
            return
        if not window_reached:
            return
        total_samples = sum(size for _rms, _peak, size in self._level_window)
        if total_samples <= 0:
            return
        # Weighted RMS over the small window; peaks are diagnostic only.
        mean_square = (
            sum((rms_value ** 2) * size for rms_value, _peak, size in self._level_window)
            / total_samples
        )
        window_rms = float(np.sqrt(mean_square))
        window_peak = max(
            (peak_value for _rms, peak_value, _size in self._level_window),
            default=0.0,
        )
        if window_rms < low_level_rms:
            self._low_level_reported = True
            logger.warning(
                "microphone frames are arriving but input level is very low: "
                "rms=%.4f peak=%.4f onset_rms=%.4f offset_rms=%.4f; "
                "check input device and microphone gain",
                window_rms,
                window_peak,
                onset,
                offset,
            )

    def _check_heartbeat(self) -> None:
        """Surface a silent/disconnected input device as a visible error.

        Distinct from queue overflow: nothing is arriving at all. An open
        utterance can then never complete, so it is aborted through the same
        explicit discontinuity path as any other audio gap.
        """
        if self._stopping or self._paused or self._stream is None:
            return
        with self._lock:
            quiet_s = time.monotonic() - self._last_block_time
        active = getattr(self._stream, "active", True)
        if active and quiet_s < self.config.dropout_timeout_s:
            return
        if self._dropout_reported:
            return
        self._dropout_reported = True
        self.stats["dropouts"] = self.stats.get("dropouts", 0) + 1
        self.last_error = (
            f"no microphone input for {quiet_s:.1f}s"
            + ("" if active else "; the stream reports inactive")
            + " (device silent or disconnected)"
        )
        logger.error("audio capture stalled: %s", self.last_error)
        if self.vad.in_speech:
            self.vad.reset()
            if self.on_discontinuity is not None:
                try:
                    self.on_discontinuity()
                except Exception:
                    logger.exception("on_discontinuity callback failed")

    def _process_item(self, block) -> None:
        if block is _PAUSE:
            for event in self.vad.flush():
                self._dispatch(event)
            return
        if block is _CLEAR:
            if self.on_discontinuity:
                self.on_discontinuity()
            self.vad.reset()
            logger.debug("VAD reset")
            return
        self._consume(block)

    def _consume(self, block: np.ndarray) -> None:
        if self._discontinuity.is_set():
            self._discontinuity.clear()
            if self.on_discontinuity:
                self.on_discontinuity()
            self.vad.reset()
            with self._queue.mutex:
                controls = [item for item in self._queue.queue if not isinstance(item, np.ndarray)]
                self._queue.queue.clear()
                self._queue.queue.extend(controls)
            return
        duration = block.size / float(self.config.sample_rate)
        self._observe_microphone_level(block)

        # Hand the block to the streaming consumer *before* the boundary
        # events, so an utterance always has all of its audio when it ends.
        in_speech = self.vad.state is VADState.SPEECH
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
            logger.info("SPEECH_START: speech started (%.2fs pre-roll)", event.duration_s)
            if self.on_speech_start is not None:
                try:
                    self.on_speech_start(event.audio if event.audio is not None else np.zeros(0, dtype=np.float32))
                except Exception:
                    logger.exception("on_speech_start callback failed")
        else:
            logger.info(
                "SPEECH_END: speech ended (%.2fs%s)",
                event.duration_s,
                ", forced segment cut" if event.forced else "",
            )
            if event.audio is not None and event.audio.size:
                # Diagnostics are metrics only: levels are logged and counted,
                # never used to alter the transcript.
                diagnostics = analyze_segment(
                    event.audio, low_rms=max(1e-4, self.config.vad_onset_rms * 0.75)
                )
                logger.info("segment audio: %s", format_summary(diagnostics))
                self._last_diagnostics = diagnostics
                if diagnostics.level == "clipped":
                    self.stats["clipped_segments"] = self.stats.get("clipped_segments", 0) + 1
                elif diagnostics.level == "low":
                    self.stats["low_level_segments"] = self.stats.get("low_level_segments", 0) + 1
            if self.on_speech_end is not None:
                try:
                    self.on_speech_end(event.duration_s, event.forced)
                except Exception:
                    logger.exception("on_speech_end callback failed")

    # ------------------------------------------------------------------ #
    def get_statistics(self) -> Dict[str, float]:
        stats = dict(self.stats)
        stats["dropped_chunks"] = self._dropped_chunks
        stats["vad_state"] = self.vad.state.value
        stats["last_error"] = self.last_error or ""
        if self._last_diagnostics is not None:
            stats["last_segment"] = to_dict(self._last_diagnostics)
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
