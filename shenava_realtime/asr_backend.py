"""NeMo/Shenava ASR backend.

Everything heavy (torch, NeMo, the checkpoint) is imported lazily inside
:func:`NeMoASR.load`, so importing this module — and running the test-suite —
never touches the model or a GPU.

The backend exposes a deliberately small surface:

``transcribe(audio) -> (text, confidence)``
    decode one 16 kHz mono float32 buffer;

``create_stream() -> CacheAwareStream | None``
    a cache-aware streaming session, or ``None`` when the checkpoint does not
    support it (the caller then uses endpoint-only decoding).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple

import numpy as np

from .config import REPO_ROOT, ASRConfig

logger = logging.getLogger(__name__)

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


class NeMoASR:
    """Loads the Shenava checkpoint and decodes with its CTC head."""

    def __init__(self, config: Optional[ASRConfig] = None) -> None:
        self.config = config or ASRConfig()
        self._model: Any = None
        self._torch: Any = None
        self._streaming_supported = False
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
        """Load a trusted local checkpoint; network provisioning requires opt-in."""
        if self._model is not None:
            return
        self.config.__post_init__()
        checkpoint = self.resolve_checkpoint_path()
        if checkpoint is None and not self.config.allow_download:
            raise FileNotFoundError(
                f"Local Shenava checkpoint not found: {self.config.model_path}. "
                "Provision a trusted .nemo file or explicitly enable SHENAVA_ALLOW_DOWNLOAD."
            )
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
            model = nemo_asr.models.ASRModel.restore_from(str(checkpoint), map_location="cpu")
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

        from omegaconf import OmegaConf
        decoding = OmegaConf.create({"strategy": "greedy", "preserve_alignments": False})
        if hasattr(model, "ctc_decoder"):
            model.change_decoding_strategy(decoding, decoder_type="ctc")
        else:
            model.change_decoding_strategy(decoding)
        self._model = model
        self._streaming_supported = self._check_streaming_context(
            model, self.config.right_context
        )
        if self._streaming_supported:
            encoder = getattr(model, "encoder", None)
            encoder.set_default_att_context_size([70, self.config.right_context])
            encoder.setup_streaming_params()
        self.load_seconds = time.time() - started
        logger.info("ASR model ready on %s in %.1fs (%s)", self.device, self.load_seconds, type(model).__name__)

    @staticmethod
    def _check_streaming_context(model: Any, right_context: int) -> bool:
        """Decide native-streaming support from the actual encoder metadata.

        Returns ``True`` only when the encoder can stream at
        ``[70, right_context]``.  A streaming-capable encoder that lacks the
        requested context is a configuration error and raises here — at
        startup, before anything is decoded — instead of silently degrading
        to endpoint-only decoding.  Checkpoints without streaming support or
        without context metadata keep the documented endpoint-only fallback.
        """
        encoder = getattr(model, "encoder", None)
        contexts = getattr(encoder, "att_context_size_all", None)
        if not contexts:
            current_context = getattr(encoder, "att_context_size", None)
            if current_context is not None and len(current_context) == 2:
                contexts = [current_context]
        supported = [list(c) for c in contexts] if contexts else []
        if not supported:
            # No context metadata: only the offline endpoint path is knowable.
            return False
        requested = [70, right_context]
        if requested not in supported:
            if callable(getattr(model, "conformer_stream_step", None)):
                raise RuntimeError(
                    f"Encoder supports streaming contexts {supported}; "
                    f"right_context={right_context} is not one of them. "
                    "Choose a supported right context "
                    "(--right-context / SHENAVA_RIGHT_CONTEXT) or use a "
                    "checkpoint that matches the requested context."
                )
            # Offline checkpoint: endpoint-only fallback (documented).
            return False
        return callable(getattr(model, "conformer_stream_step", None))

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

    def create_stream(self) -> Optional[CacheAwareStream]:
        if self._model is None:
            self.load()
        if not self.config.use_cache_aware_streaming or not self._streaming_supported:
            return None
        from .native_stream import NeMoCacheAwareStream
        return NeMoCacheAwareStream(self._model, self._import_torch(), self.config.max_segment_s)

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
        if isinstance(first, str):
            text = first
        elif isinstance(first, dict):
            text = str(first.get("text", "") or "")
        elif isinstance(first, (list, tuple)):
            return NeMoASR._extract(first)
        else:
            text = str(getattr(first, "text", "") or "")

        # NeMo hypothesis scores are NOT calibrated confidence. Keep the
        # legacy tuple/callback slot at zero (unknown), never invent a value.
        return text.strip(), 0.0
