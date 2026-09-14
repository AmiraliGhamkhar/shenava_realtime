"""NeMo/Shenava ASR backend.

Everything heavy (torch, NeMo, the checkpoint) is imported lazily inside
:func:`NeMoASR.load`, so importing this module — and running the test-suite —
never touches the model or a GPU.

The backend exposes a deliberately small surface:

``transcribe(audio) -> (text, confidence)``
    decode one 16 kHz mono float32 buffer;

``create_stream() -> CacheAwareStream | None``
    a cache-aware streaming session, or ``None`` when the checkpoint does not
    support it (the caller then falls back to windowed decoding).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple

import numpy as np

from .config import REPO_ROOT, ASRConfig

logger = logging.getLogger(__name__)

# Import paths that have hosted NeMo's cache-aware streaming helper.
_STREAM_INFER_MODULES = (
    "nemo.collections.asr.parts.streaming.cache_aware_streaming",
    "nemo.collections.asr.parts.utils.streaming_utils",
)


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


def score_to_confidence(score: Optional[float], text: str) -> float:
    """Map a NeMo hypothesis score onto a 0..1 confidence.

    NeMo reports summed log-probabilities (a negative number).  Normalising by
    the token count keeps long utterances from collapsing towards zero.
    """
    if score is None:
        return 0.0
    value = float(score)
    if 0.0 <= value <= 1.0:
        return value
    if math.isnan(value) or math.isinf(value):
        return 0.0
    tokens = max(1, len(text.split()))
    return float(min(1.0, max(0.0, math.exp(value / tokens))))


def estimate_confidence(text: str) -> float:
    """Cheap heuristic used when the model gives no usable score."""
    if not text:
        return 0.0
    words = text.split()
    score = 1.0
    score -= 0.1 * sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    if len(words) < 2:
        score -= 0.2
    return float(min(1.0, max(0.0, score)))


class NeMoASR:
    """Loads the Shenava checkpoint and decodes with its CTC head."""

    def __init__(self, config: Optional[ASRConfig] = None) -> None:
        self.config = config or ASRConfig()
        self._model: Any = None
        self._torch: Any = None
        self._stream_infer_cls: Any = None
        self.device: str = "cpu"
        self.load_seconds: float = 0.0

    # ------------------------------------------------------------------ #
    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    def resolve_device(self) -> str:
        torch = self._import_torch()
        requested = (self.config.device or "auto").strip().lower()
        if requested in ("", "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        return requested

    def _import_torch(self) -> Any:
        if self._torch is None:
            try:
                import torch  # noqa: PLC0415 - deliberately lazy
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "PyTorch is required for the ASR backend. "
                    "Install the ASR extras: pip install -r requirements-asr.txt"
                ) from exc
            self._torch = torch
        return self._torch

    def load(self) -> None:
        """Load the checkpoint (local ``.nemo`` first, then the HF hub)."""
        if self._model is not None:
            return
        torch = self._import_torch()
        started = time.time()

        try:
            import nemo.collections.asr as nemo_asr  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "nemo_toolkit[asr] is required for the ASR backend. "
                "Install the ASR extras: pip install -r requirements-asr.txt"
            ) from exc

        checkpoint = self.resolve_checkpoint_path()
        if checkpoint is not None:
            logger.info("loading local checkpoint: %s", checkpoint)
            model = nemo_asr.models.ASRModel.restore_from(str(checkpoint))
        else:
            if self.config.model_path:
                logger.warning(
                    "checkpoint %s not found; downloading %s",
                    self.config.model_path,
                    self.config.model_name,
                )
            model = nemo_asr.models.ASRModel.from_pretrained(self.config.model_name)

        self.device = self.resolve_device()
        model = model.to(self.device)
        model.eval()
        if self.device.startswith("cpu"):
            torch.set_num_threads(max(1, int(self.config.num_threads)))

        decoder_type = (self.config.decoder_type or "").strip().lower()
        if decoder_type and hasattr(model, "change_decoding_strategy"):
            try:
                model.change_decoding_strategy(decoder_type=decoder_type)
                logger.info("decoding strategy: %s", decoder_type)
            except Exception:
                logger.exception("could not switch to the %r decoder; keeping the default", decoder_type)

        self._model = model
        self._stream_infer_cls = self._find_stream_infer_class()
        self.load_seconds = time.time() - started
        logger.info("ASR model ready on %s in %.1fs (%s)", self.device, self.load_seconds, type(model).__name__)

    def resolve_checkpoint_path(self) -> Optional[Path]:
        """Find the configured checkpoint, tolerating relative paths.

        ``.env`` typically holds a path relative to the repository, so a run
        started from another working directory must not silently fall back to a
        network download.
        """
        raw = self.config.model_path
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        for path in (candidate, REPO_ROOT / candidate) if not candidate.is_absolute() else (candidate,):
            if path.is_file():
                return path
        return None

    def _find_stream_infer_class(self) -> Any:
        if not self.config.use_cache_aware_streaming:
            return None
        if getattr(self._model, "streaming_cfg", None) is None:
            logger.info("checkpoint has no streaming config; using windowed incremental decoding")
            return None
        import importlib

        for module_name in _STREAM_INFER_MODULES:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            cls = getattr(module, "CacheAwareStreamInfer", None)
            if cls is not None:
                logger.info("cache-aware streaming available via %s", module_name)
                return cls
        logger.info("CacheAwareStreamInfer not found in this NeMo build; using windowed decoding")
        return None

    # ------------------------------------------------------------------ #
    def transcribe(self, audio: np.ndarray) -> Tuple[str, float]:
        """Decode one buffer and return ``(text, confidence)``."""
        if self._model is None:
            self.load()
        samples = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        if samples.size == 0:
            return "", 0.0

        torch = self._import_torch()
        with torch.no_grad():
            outputs = self._model.transcribe(
                [samples],
                batch_size=1,
                verbose=False,
                return_hypotheses=True,
            )
        return self._extract(outputs)

    def create_stream(self) -> Optional["NeMoCacheAwareStream"]:
        if self._model is None:
            self.load()
        if self._stream_infer_cls is None:
            return None
        try:
            return NeMoCacheAwareStream(self._model, self._stream_infer_cls)
        except Exception:
            logger.exception("cache-aware streaming could not be initialised; using windowed decoding")
            return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract(outputs: Any) -> Tuple[str, float]:
        """Pull text/score out of the many shapes NeMo can return."""
        if outputs is None:
            return "", 0.0
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        if not outputs:
            return "", 0.0

        first = outputs[0]
        text = ""
        score: Optional[float] = None
        if isinstance(first, str):
            text = first
        elif isinstance(first, dict):
            text = str(first.get("text", "") or "")
            score = first.get("score")
        else:
            text = str(getattr(first, "text", "") or "")
            raw_score = getattr(first, "score", None)
            try:
                score = None if raw_score is None else float(raw_score)
            except (TypeError, ValueError):
                score = None

        text = text.strip()
        confidence = score_to_confidence(score, text)
        if confidence <= 0.0 and text:
            confidence = estimate_confidence(text)
        return text, confidence


class NeMoCacheAwareStream:
    """Adapter around NeMo's ``CacheAwareStreamInfer`` (true cache-aware mode).

    Only reachable for checkpoints that ship a streaming config; construction is
    smoke-tested by :meth:`NeMoASR.create_stream`, and any failure downgrades the
    session to windowed decoding instead of breaking the app.
    """

    def __init__(self, model: Any, stream_infer_cls: Any) -> None:
        self._infer = stream_infer_cls(model)
        if not hasattr(self._infer, "transcribe_chunk"):
            raise RuntimeError("CacheAwareStreamInfer has no transcribe_chunk()")
        self._state = self._initial_state()
        # Smoke test: a short silent chunk must survive the API we expect.
        silence = np.zeros(int(0.32 * 16000), dtype=np.float32)
        text, state = self._infer.transcribe_chunk(self._state, silence)
        self._state = state
        self._text = (getattr(text, "text", "") or "").strip() if not isinstance(text, str) else text.strip()

    def _initial_state(self) -> Any:
        factory = getattr(self._infer, "_get_initial_state", None)
        if factory is None:
            raise RuntimeError("CacheAwareStreamInfer has no _get_initial_state()")
        return factory(audio_signal=None) if _accepts_keyword(factory, "audio_signal") else factory()

    def reset(self) -> None:
        reset = getattr(self._infer, "reset_states", None)
        if callable(reset):
            reset(self._state)
            self._state = self._initial_state()
        else:
            self._state = self._initial_state()
        self._text = ""

    def push(self, audio: np.ndarray) -> Tuple[str, float]:
        samples = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        if samples.size == 0:
            return self._text, 0.0
        hypothesis, self._state = self._infer.transcribe_chunk(self._state, samples)
        if isinstance(hypothesis, str):
            self._text = hypothesis.strip()
            return self._text, estimate_confidence(self._text)
        text = str(getattr(hypothesis, "text", "") or "").strip()
        raw_score = getattr(hypothesis, "score", None)
        try:
            score = None if raw_score is None else float(raw_score)
        except (TypeError, ValueError):
            score = None
        confidence = score_to_confidence(score, text) or estimate_confidence(text)
        self._text = text
        return self._text, confidence


def _accepts_keyword(function: Any, keyword: str) -> bool:
    import inspect

    try:
        return keyword in inspect.signature(function).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
        return False
