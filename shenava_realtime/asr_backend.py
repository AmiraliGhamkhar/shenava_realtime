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

    def capabilities(self) -> dict[str, bool]:  # pragma: no cover
        ...
        ...

    def create_stream(self) -> Optional["CacheAwareStream"]:  # pragma: no cover
        ...


class CacheAwareStream(Protocol):
    def reset(self) -> None:  # pragma: no cover
        ...

    def push(self, audio: np.ndarray) -> Tuple[str, float]:  # pragma: no cover
        ...


class SecondPassUnavailable(RuntimeError):
    """The context second pass cannot be built; fail clearly, do not degrade."""


def _select_logits(output: Any, vocab_size: Optional[int]) -> Any:
    """Find the CTC logits tensor among the shapes ``model.forward`` may return."""
    torch = None
    candidates = [output]
    if isinstance(output, (tuple, list)):
        candidates = list(output)
    for item in candidates:
        try:
            ndim = item.dim()
        except AttributeError:
            continue
        if ndim == 3 and item.shape[0] == 1:
            if vocab_size is None or int(item.shape[2]) == vocab_size:
                return item[0]
        if ndim == 2 and (vocab_size is None or int(item.shape[1]) == vocab_size):
            return item
    return None


class NeMoRNNTStream:
    """Small adapter for a checkpoint's *native* RNNT stream API.

    No prediction/joint logic lives here: NeMo owns the state and decoding.
    Checkpoint versions with a different API are rejected during capability
    probing instead of being treated as CTC streams.
    """
    def __init__(self, model: Any) -> None:
        self.model = model
        self.state: Any = None

    def reset(self) -> None:
        reset = getattr(self.model, "reset_rnnt_stream", None)
        if callable(reset):
            reset()
        self.state = None

    def push(self, audio: np.ndarray) -> Tuple[str, float]:
        result = self.model.rnnt_stream_step(np.asarray(audio, dtype=np.float32), state=self.state)
        if isinstance(result, tuple) and len(result) == 2:
            output, self.state = result
        else:
            output = result
        return NeMoASR._extract(output)

    def finalize(self) -> Tuple[str, float]:
        flush = getattr(self.model, "rnnt_stream_finalize", None)
        if not callable(flush):
            return "", 0.0
        return NeMoASR._extract(flush(state=self.state))


class NeMoASR:
    """Loads Shenava v1.5 and selects CTC (default) or native RNNT."""

    def __init__(self, config: Optional[ASRConfig] = None) -> None:
        self.config = config or ASRConfig()
        self._model: Any = None
        self._torch: Any = None
        self._streaming_supported = False
        self._capabilities: dict[str, bool] = {}
        self.selected_decoder = "ctc"
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
        selected = "ctc" if self.config.decoder_type == "auto" else self.config.decoder_type
        has_ctc = hasattr(model, "ctc_decoder") or hasattr(model, "ctc_decoding")
        has_rnnt = any(getattr(model, name, None) is not None for name in
                       ("rnnt_decoder", "rnnt_decoding", "prediction", "joint"))
        if selected == "rnnt":
            if not has_rnnt:
                raise RuntimeError("decoder=rnnt requested, but the checkpoint exposes no RNNT prediction/joint decoder")
            try:
                model.change_decoding_strategy(decoding, decoder_type="rnnt")
            except (TypeError, AttributeError) as exc:
                raise RuntimeError("decoder=rnnt is not supported by this NeMo/checkpoint API") from exc
        else:
            if not has_ctc:
                raise RuntimeError("decoder=ctc requested, but the checkpoint exposes no CTC decoder")
            try:
                model.change_decoding_strategy(decoding, decoder_type="ctc")
            except TypeError:
                model.change_decoding_strategy(decoding)
        self.selected_decoder = selected
        self._model = model
        self._streaming_supported = (self._check_streaming_context(model, self.config.right_context)
                                     if selected == "ctc" else self._check_rnnt_streaming(model))
        self._capabilities = {"ctc": bool(has_ctc), "rnnt": bool(has_rnnt),
                              "streaming": bool(self._streaming_supported),
                              "offline": True, "second_pass_context": selected == "ctc"}
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

    def capabilities(self) -> dict[str, bool]:
        if self._model is None:
            self.load()
        return dict(self._capabilities)

    @staticmethod
    def _check_rnnt_streaming(model: Any) -> bool:
        return callable(getattr(model, "rnnt_stream_step", None))

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
        if getattr(self, "selected_decoder", "ctc") == "rnnt":
            return NeMoRNNTStream(self._model)
        from .native_stream import NeMoCacheAwareStream
        return NeMoCacheAwareStream(self._model, self._import_torch(), self.config.max_segment_s)

    # ------------------------------------------------------------------ #
    # Context second pass (CTC beam + hotword biasing).  Every probe below is
    # explicit: a mismatched checkpoint API raises SecondPassUnavailable at
    # startup instead of silently running a different decoder.
    # ------------------------------------------------------------------ #
    def build_second_pass(self, config: "ASRConfig"):
        if getattr(self, "selected_decoder", self.config.decoder_type) == "rnnt":
            raise SecondPassUnavailable("RNNT context biasing requires a checkpoint-native RNNT decoder API; no CTC cross-decoding is attempted")
        from .second_pass import BeamSecondPass, GreedySecondPass

        model = self.model  # ensures the checkpoint is loaded
        torch = self._import_torch()
        tokenize, decode_tokens, vocab_size = self._probe_tokenizer(model)
        emissions_fn = self._probe_emissions(model, torch, vocab_size)
        blank = self._probe_blank_index(model, vocab_size)
        logger.info(
            "second pass ready: beam=%d vocab=%d blank=%d",
            config.second_pass_beam_size, vocab_size, blank,
        )
        return BeamSecondPass(
            emissions_fn,
            tokenize,
            decode_tokens,
            blank,
            config.second_pass_beam_size,
            GreedySecondPass(self.transcribe),
        )

    def _probe_tokenizer(self, model: Any) -> tuple:
        spec = getattr(model, "tokenizer", None) or getattr(model, "_tokenizer", None)
        if spec is None:
            raise SecondPassUnavailable("checkpoint exposes no tokenizer; hotword biasing needs the model's own BPE vocabulary")
        inner = getattr(spec, "tokenizer", spec)

        def tokenize(text: str) -> list[int]:
            if not hasattr(inner, "encode"):
                raise SecondPassUnavailable("tokenizer has no encode()")
            try:
                ids = inner.encode(text)
            except TypeError:
                ids = inner.encode(text, add_special_tokens=False)
            if isinstance(ids, str):
                ids = ids.split()
            return [int(x) for x in ids]

        vocab = None
        getter = getattr(inner, "get_vocab", None)
        if callable(getter):
            try:
                vocab = getter()
            except Exception:
                vocab = None
        if vocab is None:
            attr = getattr(inner, "vocab", None)
            if isinstance(attr, (list, dict)):
                vocab = attr
        if vocab is None:
            raise SecondPassUnavailable("tokenizer exposes no vocabulary for token->text decoding")
        if isinstance(vocab, dict):
            keys = list(vocab)[:8]
            if keys and all(isinstance(k, int) for k in keys):
                vocab = [vocab[k] for k in sorted(vocab)]
            else:
                vocab = [str(value) for value in vocab.values()]
        if not vocab:
            raise SecondPassUnavailable("tokenizer vocabulary is empty")

        if hasattr(inner, "decode"):
            def decode_tokens(ids) -> str:
                return inner.decode(list(ids))
        else:
            def decode_tokens(ids) -> str:
                parts = [vocab[i] if 0 <= i < len(vocab) else "" for i in ids]
                text = "".join(parts).replace("\u2581", " ").strip()
                return " ".join(text.split())
        return tokenize, decode_tokens, len(vocab)

    def _probe_emissions(self, model: Any, torch: Any, vocab_size: int):
        preprocessor = getattr(model, "preprocessor", None)
        if preprocessor is None or not hasattr(model, "forward"):
            raise SecondPassUnavailable("checkpoint has no preprocessor/forward for CTC emissions")

        def emissions_fn(audio: np.ndarray) -> np.ndarray:
            samples = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
            with torch.inference_mode():
                signal = torch.from_numpy(samples).unsqueeze(0).to(model.device)
                length = torch.tensor([samples.size], device=model.device, dtype=torch.long)
                processed, processed_length = preprocessor(input_signal=signal, length=length)
                output = model.forward(processed, processed_length)
            logits = _select_logits(output, vocab_size)
            if logits is None:
                raise SecondPassUnavailable(
                    "model.forward did not return CTC logits with vocabulary "
                    f"{vocab_size}; the context second pass is not available for this checkpoint"
                )
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            return log_probs.detach().cpu().numpy().astype(np.float32)

        return emissions_fn

    def _probe_blank_index(self, model: Any, vocab_size: int) -> int:
        for module_name in ("ctc_decoder", "_ctc_decoder", "ctc_decoding", "_ctc_decoding"):
            decoder = getattr(model, module_name, None)
            if decoder is None:
                continue
            for attr in ("ctc_blank_index", "blank_index", "_blank_index"):
                value = getattr(decoder, attr, None)
                if isinstance(value, int) and 0 <= value < vocab_size:
                    return value
        # NeMo convention: the CTC blank is token 0.
        logger.debug("no explicit ctc_blank_index found; using NeMo convention (0)")
        return 0

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
