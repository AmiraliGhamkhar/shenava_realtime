"""Sherpa-ONNX streaming CTC ASR backend.

Everything heavy (``sherpa_onnx``, ONNX Runtime, the model files) is imported
lazily inside :meth:`SherpaOnnxASR.load`, so importing this module — and
running the test-suite — never touches the model.  ``torch``/``nemo``/CUDA are
never imported anywhere in this module.

The backend exposes a small, type-hinted surface:

``load()``
    build the single long-lived ``sherpa_onnx.OnlineRecognizer`` for the
    process (CPU, INT8, greedy CTC, sherpa's own endpointer disabled — the
    existing RMS VAD is the only endpointer);

``create_stream()``
    a fresh ``sherpa_onnx.OnlineStream`` (one per VAD segment; never reused
    across segments; never a substitute for a second recognizer);

``accept(stream, samples)`` / ``decode_ready(stream)`` / ``partial(stream)``
    feed audio and decode whatever the recognizer is ready to decode;

``finalize(stream)``
    flush the stream at a VAD endpoint (tail padding, ``input_finished``,
    drain, final text) and return the text; the caller then discards the
    stream and creates a new one for the next segment;

``stats()``
    runtime counters (audio seconds, processing seconds, RTF, decode count,
    decoder errors) — never transcript or audio content;

``transcribe(audio)``
    a convenience one-shot decode built from the same recognizer, used by the
    offline WAV tools and the utterance-end "greedy" second pass.

Target model: sherpa-onnx streaming NeMo-CTC export of
``Shenava-Koochik-v1.0`` (see ``models/shenava/README.md`` for the pinned HF
revision and checksums).  Loading is explicit and local-only: there is no
network fallback anywhere in this module.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple

import numpy as np

from .config import ASRConfig

logger = logging.getLogger(__name__)

# Tail padding appended at a VAD endpoint so the last word is not dropped by
# the streaming encoder's lookahead. Tuned against the model-contract spike
# (see Phase 0 report); re-validate against the real model once available.
FINALIZE_TAIL_PADDING_S = 0.5


class ASRBackend(Protocol):
    def transcribe(self, audio: np.ndarray) -> Tuple[str, float]:  # pragma: no cover
        ...

    def create_stream(self) -> Optional["CacheAwareStream"]:  # pragma: no cover
        ...


class CacheAwareStream(Protocol):
    def reset(self) -> None:  # pragma: no cover
        ...

    def push(self, audio: np.ndarray) -> Tuple[str, float]:  # pragma: no cover
        ...


class ModelLoadError(RuntimeError):
    """The configured model/tokens could not be loaded (explicit, no fallback)."""


class StreamingUnavailable(RuntimeError):
    """The loaded model does not support streaming; the GATE forbids a fallback."""


@dataclass(frozen=True)
class ModelCapabilities:
    """What the loaded recognizer actually exposes (probed, never assumed)."""

    provider: str
    num_threads: int
    sample_rate: int
    feature_dim: int
    decoding_method: str
    streaming: bool

    def describe(self) -> str:
        return (
            f"sherpa-onnx CTC: provider={self.provider} threads={self.num_threads} "
            f"sample_rate={self.sample_rate} feature_dim={self.feature_dim} "
            f"decoding={self.decoding_method} streaming={'yes' if self.streaming else 'no'}"
        )


@dataclass
class _Stats:
    audio_seconds: float = 0.0
    processing_seconds: float = 0.0
    decode_count: int = 0
    dropped_chunks: int = 0
    overruns: int = 0
    decoder_errors: int = 0
    segments: int = 0


class SherpaStream:
    """Thin wrapper around one ``sherpa_onnx.OnlineStream``.

    One instance per VAD segment. Never reused across segments (a fresh
    ``OnlineStream`` starts with a zero-filled decoder cache, which is the
    validated state-clean lifecycle for this backend — see Phase 0 report).
    """

    def __init__(self, recognizer: Any, raw_stream: Any, sample_rate: int, stats: _Stats) -> None:
        self._recognizer = recognizer
        self._stream = raw_stream
        self._sample_rate = sample_rate
        self._stats = stats
        self._closed = False
        self._last_text = ""

    # ------------------------------------------------------------------ #
    def accept(self, samples: np.ndarray) -> None:
        """Feed one contiguous float32 mono block in [-1, 1] at 16 kHz."""
        if self._closed:
            raise RuntimeError("Stream already finalized; create a new stream instead")
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        if not np.isfinite(samples).all():
            raise ValueError("Non-finite audio")
        self._stream.accept_waveform(self._sample_rate, samples)
        self._stats.audio_seconds += samples.size / float(self._sample_rate)

    def decode_ready(self) -> None:
        """Run every decode step the recognizer is currently ready for."""
        started = time.perf_counter()
        try:
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
                self._stats.decode_count += 1
                logger.debug("decoder step completed (decode_count=%d)", self._stats.decode_count)
        except Exception:
            self._stats.decoder_errors += 1
            raise
        finally:
            self._stats.processing_seconds += time.perf_counter() - started

    def partial(self) -> str:
        """The current (cumulative) hypothesis without finalizing."""
        text = self._recognizer.get_result(self._stream)
        self._last_text = text if isinstance(text, str) else str(text)
        return self._last_text

    # ------------------------------------------------------------------ #
    def push(self, audio: np.ndarray) -> Tuple[str, float]:
        """``CacheAwareStream`` compatibility: accept + decode + text."""
        self.accept(audio)
        self.decode_ready()
        return self.partial(), 0.0

    def finalize(self) -> Tuple[str, float]:
        """VAD-endpoint flush: tail padding, ``input_finished``, drain, final text.

        The stream must be discarded after this call; the caller creates a
        new stream (via :meth:`SherpaOnnxASR.create_stream`) for the next
        segment. This wrapper never reuses ``self`` after finalizing.
        """
        if not self._closed:
            pad = np.zeros(int(FINALIZE_TAIL_PADDING_S * self._sample_rate), dtype=np.float32)
            self.accept(pad)
            self._stream.input_finished()
            self.decode_ready()
            self.partial()
            self._closed = True
            self._stats.segments += 1
        return self._last_text, 0.0

    def reset(self) -> None:
        """``CacheAwareStream``/``StreamingDecoder`` contract compatibility.

        Per the locked design, streams are not reused across VAD segments —
        the engine creates a fresh :class:`SherpaStream` for every segment via
        ``backend.create_stream()``. This method exists only so callers that
        expect the ``reset()`` contract (e.g. ``streaming.CacheAwareDecoder``)
        keep working; it is a no-op here because there is no in-place state to
        clear (the object is discarded, not reused).
        """
        return None


class SherpaOnnxASR:
    """Loads the sherpa-onnx streaming CTC model and owns the one recognizer."""

    def __init__(self, config: Optional[ASRConfig] = None) -> None:
        self.config = config or ASRConfig()
        self._recognizer: Any = None
        self._capabilities: Optional[ModelCapabilities] = None
        self.device: str = "cpu"
        self.load_seconds: float = 0.0
        self._stats = _Stats()
        self._last_streaming_probe_error = ""

    # ------------------------------------------------------------------ #
    @property
    def recognizer(self) -> Any:
        if self._recognizer is None:
            self.load()
        return self._recognizer

    def _resolve_path(self, raw: Optional[str], label: str) -> Path:
        if not raw:
            raise ModelLoadError(
                f"{label} is not configured. Set SHENAVA_MODEL_PATH / "
                "SHENAVA_TOKENS_PATH (see models/shenava/README.md for how to "
                "provision the sherpa-onnx model)."
            )
        path = Path(raw).expanduser()
        if not path.is_absolute():
            from .config import REPO_ROOT
            path = REPO_ROOT / path
        return path

    def load(self) -> None:
        """Load the local sherpa-onnx model; explicit errors, no fallback."""
        if self._recognizer is not None:
            return
        self.config.__post_init__()

        model_path = self._resolve_path(self.config.model_path, "SHENAVA_MODEL_PATH")
        if not model_path.is_file():
            raise ModelLoadError(
                f"Local sherpa-onnx model not found: {model_path}. Provision it "
                "with the documented `hf download` command in "
                "models/shenava/README.md (no automatic downloads)."
            )
        if model_path.suffix == ".nemo":
            raise ModelLoadError(
                f"{model_path} is a legacy NeMo checkpoint; the runtime now "
                "loads a sherpa-onnx ONNX model (model.int8.onnx + tokens.txt)."
            )
        if model_path.suffix != ".onnx":
            raise ModelLoadError(
                f"SHENAVA_MODEL_PATH must point at a .onnx file, got {model_path}"
            )

        tokens_path = self._resolve_path(self.config.tokens_path, "SHENAVA_TOKENS_PATH")
        if not tokens_path.is_file():
            raise ModelLoadError(
                f"Local tokens file not found: {tokens_path}. Provision it with "
                "the documented `hf download` command in models/shenava/README.md."
            )
        if tokens_path.suffix != ".txt":
            raise ModelLoadError(
                f"SHENAVA_TOKENS_PATH must point at a .txt file, got {tokens_path}"
            )

        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelLoadError(
                "sherpa-onnx is required for the ASR backend. "
                "Install it: pip install -r requirements.txt"
            ) from exc

        if self.config.sample_rate != 16000:
            raise ModelLoadError("Shenava requires a 16 kHz mono frontend")

        started = time.time()
        try:
            recognizer = sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
                tokens=str(tokens_path),
                model=str(model_path),
                num_threads=int(self.config.num_threads),
                sample_rate=int(self.config.sample_rate),
                feature_dim=int(self.config.feature_dim),
                decoding_method=self.config.decoding_method,
                provider="cpu",
                # The RMS VAD already owns utterance boundaries; sherpa's own
                # endpointer must stay off so there is exactly one endpointer.
                enable_endpoint_detection=False,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"sherpa-onnx failed to initialise the recognizer from "
                f"{model_path} / {tokens_path}: {exc}"
            ) from exc
        if recognizer is None:
            raise ModelLoadError(
                "sherpa_onnx.OnlineRecognizer.from_nemo_ctc returned no recognizer"
            )

        # GATE: the model must load as a streaming (online) recognizer. There
        # is no fallback to OfflineRecognizer here, ever.
        streaming = self._probe_streaming(recognizer)
        if self.config.require_streaming and not streaming:
            reason = self._last_streaming_probe_error or "startup streaming probe failed"
            raise StreamingUnavailable(
                "SHENAVA_REQUIRE_STREAMING is set but the loaded model does not "
                "complete the online streaming lifecycle "
                "(create_stream -> accept_waveform -> is_ready/decode_stream -> "
                "get_result -> input_finished -> final get_result). Refusing to "
                f"fall back to an offline decoder. Probe failure: {reason}"
            )

        self.device = "cpu"
        self._recognizer = recognizer
        self._capabilities = ModelCapabilities(
            provider="cpu",
            num_threads=int(self.config.num_threads),
            sample_rate=int(self.config.sample_rate),
            feature_dim=int(self.config.feature_dim),
            decoding_method=self.config.decoding_method,
            streaming=streaming,
        )
        self.load_seconds = time.time() - started
        logger.info(
            "ASR backend ready in %.2fs — %s | model=%s tokens=%s",
            self.load_seconds, self._capabilities.describe(), model_path, tokens_path,
        )

    def _probe_streaming(self, recognizer: Any) -> bool:
        """Run a tiny real online transaction against the loaded recognizer.

        This is the production startup gate.  Method names alone are not
        enough: the Shenava model must create an ``OnlineStream``, accept valid
        16 kHz float32 audio, become ready, execute at least one
        ``decode_stream`` step, produce a result before and after
        ``input_finished()``, and drain cleanly.  The synthetic input is only
        one second of silence, so the probe is cheap and contains no user audio.
        """
        self._last_streaming_probe_error = ""
        stream: Any = None
        input_finished_called = False
        decode_steps = 0
        max_decode_steps = 256  # protects startup from a buggy always-ready fake/model

        def fail(message: str) -> bool:
            self._last_streaming_probe_error = message
            logger.warning("streaming capability probe failed: %s", message)
            return False

        try:
            create_stream = getattr(recognizer, "create_stream", None)
            is_ready = getattr(recognizer, "is_ready", None)
            decode_stream = getattr(recognizer, "decode_stream", None)
            get_result = getattr(recognizer, "get_result", None)
            if not callable(create_stream):
                return fail("recognizer has no callable create_stream()")
            if not callable(is_ready):
                return fail("recognizer has no callable is_ready(stream)")
            if not callable(decode_stream):
                return fail("recognizer has no callable decode_stream(stream)")
            if not callable(get_result):
                return fail("recognizer has no callable get_result(stream)")

            stream = create_stream()
            if stream is None:
                return fail("create_stream() returned None")
            accept_waveform = getattr(stream, "accept_waveform", None)
            input_finished = getattr(stream, "input_finished", None)
            if not callable(accept_waveform):
                return fail("stream has no callable accept_waveform(sample_rate, samples)")
            if not callable(input_finished):
                return fail("stream has no callable input_finished()")

            samples = np.zeros(int(self.config.sample_rate), dtype=np.float32)  # 1s silence
            accept_waveform(int(self.config.sample_rate), samples)

            while bool(is_ready(stream)):
                decode_stream(stream)
                decode_steps += 1
                if decode_steps > max_decode_steps:
                    return fail("decode_stream did not drain readiness during startup probe")
            if get_result(stream) is None:
                return fail("get_result() returned None before input_finished()")

            input_finished()
            input_finished_called = True

            while bool(is_ready(stream)):
                decode_stream(stream)
                decode_steps += 1
                if decode_steps > max_decode_steps:
                    return fail("decode_stream did not drain after input_finished()")
            if get_result(stream) is None:
                return fail("get_result() returned None after input_finished()")
            if decode_steps == 0:
                return fail("one second of valid 16 kHz audio produced no decode_stream() step")

            logger.debug("streaming startup probe completed with %d decode step(s)", decode_steps)
            return True
        except Exception as exc:
            self._last_streaming_probe_error = str(exc) or exc.__class__.__name__
            logger.warning("streaming capability probe failed: %s", self._last_streaming_probe_error)
            return False
        finally:
            if stream is not None and not input_finished_called:
                finish = getattr(stream, "input_finished", None)
                if callable(finish):
                    try:
                        finish()
                    except Exception:  # pragma: no cover - defensive cleanup only
                        logger.debug("probe stream cleanup failed", exc_info=True)

    # ------------------------------------------------------------------ #
    def capabilities(self) -> ModelCapabilities:
        if self._capabilities is None:
            self.load()
        assert self._capabilities is not None
        return self._capabilities

    def create_stream(self) -> SherpaStream:
        """A fresh stream for one VAD segment (never reused across segments)."""
        recognizer = self.recognizer
        raw_stream = recognizer.create_stream()
        return SherpaStream(recognizer, raw_stream, int(self.config.sample_rate), self._stats)

    # ------------------------------------------------------------------ #
    def transcribe(self, audio: np.ndarray) -> Tuple[str, float]:
        """Decode one full buffer via a throwaway stream (offline convenience)."""
        samples = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        if samples.size == 0:
            return "", 0.0
        stream = self.create_stream()
        stream.accept(samples)
        text, _ = stream.finalize()
        return text, 0.0

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Runtime counters only — never transcript or audio content."""
        audio = self._stats.audio_seconds
        processing = self._stats.processing_seconds
        return {
            "audio_seconds": audio,
            "processing_seconds": processing,
            "rtf": (processing / audio) if audio > 0 else 0.0,
            "decode_count": self._stats.decode_count,
            "dropped_chunks": self._stats.dropped_chunks,
            "overruns": self._stats.overruns,
            "decoder_errors": self._stats.decoder_errors,
            "segments": self._stats.segments,
        }
