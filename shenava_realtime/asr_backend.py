"""NeMo/Shenava ASR backend.

Everything heavy (torch, NeMo, the checkpoint) is imported lazily inside
:func:`NeMoASR.load`, so importing this module — and running the test-suite —
never touches the model or a GPU.

The backend exposes a deliberately small surface:

``transcribe(audio) -> (text, confidence)``
    decode one 16 kHz mono float32 buffer;

``create_stream() -> CacheAwareStream | None``
    a cache-aware streaming session, or ``None`` when the checkpoint does not
    support it (the caller then uses endpoint-only decoding);

``capabilities() -> ModelCapabilities``
    what the *loaded* checkpoint actually exposes (heads, streaming contexts,
    tokenizer/blank), probed rather than assumed.

The target checkpoint is the hybrid ``Reza2kn/Shenava-Koochik-v1.5``
(FastConformer, CTC + RNNT heads).  CTC is the production path; RNNT is
selectable for benchmarking.  A requested capability that the checkpoint does
not expose is a startup error, never a silent switch to the other head.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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


class SecondPassUnavailable(RuntimeError):
    """The context second pass cannot be built; fail clearly, do not degrade."""


class CapabilityUnavailable(RuntimeError):
    """A requested decoder/streaming capability is absent from the checkpoint."""


@dataclass(frozen=True)
class ModelCapabilities:
    """What the loaded checkpoint actually exposes (probed, never assumed)."""

    model_class: str
    has_ctc_head: bool
    has_rnnt_head: bool
    streaming_contexts: tuple[tuple[int, int], ...]
    supports_cache_aware_streaming: bool
    sample_rate: int
    vocab_size: Optional[int] = None
    ctc_blank_index: Optional[int] = None
    rnnt_blank_index: Optional[int] = None

    def heads(self) -> tuple[str, ...]:
        return tuple(
            name for name, present in (("ctc", self.has_ctc_head), ("rnnt", self.has_rnnt_head))
            if present
        )

    def describe(self) -> str:
        return (
            f"{self.model_class}: heads={','.join(self.heads()) or 'none'} "
            f"sample_rate={self.sample_rate} "
            f"streaming={'yes' if self.supports_cache_aware_streaming else 'no'} "
            f"contexts={[list(c) for c in self.streaming_contexts]} "
            f"vocab={self.vocab_size} ctc_blank={self.ctc_blank_index} "
            f"rnnt_blank={self.rnnt_blank_index}"
        )


def _decoding_config() -> Any:
    """Greedy decoding config: OmegaConf when NeMo is installed, dict otherwise."""
    settings = {"strategy": "greedy", "preserve_alignments": False}
    try:
        from omegaconf import OmegaConf  # noqa: PLC0415 - optional heavy dep
    except ImportError:
        return settings
    return OmegaConf.create(settings)


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


class NeMoASR:
    """Loads the Shenava checkpoint and decodes with its CTC head."""

    def __init__(self, config: Optional[ASRConfig] = None) -> None:
        self.config = config or ASRConfig()
        self._model: Any = None
        self._torch: Any = None
        self._streaming_supported = False
        self._capabilities: Optional[ModelCapabilities] = None
        self.decoder_type: str = self.config.resolved_decoder
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

        self.decoder_type = self.config.resolved_decoder
        capabilities = self._probe_capabilities(model)
        self._validate_startup(capabilities, self.decoder_type)
        self._select_head(model, self.decoder_type)
        self._model = model
        self._capabilities = capabilities
        self._streaming_supported = capabilities.supports_cache_aware_streaming
        if self._streaming_supported:
            encoder = model.encoder
            encoder.set_default_att_context_size([70, self.config.right_context])
            encoder.setup_streaming_params()
            self._apply_cuda_graph_option(model)
        self.load_seconds = time.time() - started
        logger.info("model capabilities — %s", capabilities.describe())
        logger.info(
            "ASR model ready on %s in %.1fs (%s, head=%s)",
            self.device, self.load_seconds, type(model).__name__, self.decoder_type,
        )

    # ------------------------------------------------------------------ #
    # Capability probing and startup validation
    # ------------------------------------------------------------------ #
    def capabilities(self) -> ModelCapabilities:
        """Probed capabilities of the loaded checkpoint (loads it if needed)."""
        if self._capabilities is None:
            self.load()
        assert self._capabilities is not None
        return self._capabilities

    def _probe_capabilities(self, model: Any) -> ModelCapabilities:
        """Inspect the actual loaded model; never infer from the model name."""
        encoder = getattr(model, "encoder", None)
        contexts = self._streaming_contexts(encoder)
        streaming = bool(contexts) and callable(getattr(model, "conformer_stream_step", None))
        sample_rate = int(getattr(getattr(model, "cfg", None), "preprocessor", None)
                          and getattr(model.cfg.preprocessor, "sample_rate", 0) or 0)
        vocab_size = None
        try:
            _tok, _dec, vocab_size = self._probe_tokenizer(model)
        except SecondPassUnavailable:
            vocab_size = None
        # A hybrid checkpoint exposes both heads; a pure model exposes one.
        has_ctc = any(
            callable(getattr(model, name, None)) or getattr(model, name, None) is not None
            for name in ("ctc_decoder", "ctc_decoding")
        ) or hasattr(model, "decoder") and not hasattr(model, "joint")
        has_rnnt = getattr(model, "joint", None) is not None and getattr(
            model, "decoder", None) is not None and hasattr(model, "decoding")
        return ModelCapabilities(
            model_class=type(model).__name__,
            has_ctc_head=bool(has_ctc),
            has_rnnt_head=bool(has_rnnt),
            streaming_contexts=contexts,
            supports_cache_aware_streaming=streaming,
            sample_rate=sample_rate,
            vocab_size=vocab_size,
            ctc_blank_index=self._probe_blank_index(model, vocab_size) if vocab_size else None,
            rnnt_blank_index=self._probe_rnnt_blank(model),
        )

    @staticmethod
    def _streaming_contexts(encoder: Any) -> tuple[tuple[int, int], ...]:
        contexts = getattr(encoder, "att_context_size_all", None)
        if not contexts:
            current = getattr(encoder, "att_context_size", None)
            contexts = [current] if current is not None and len(current) == 2 else []
        return tuple(tuple(int(v) for v in c) for c in contexts if len(c) == 2)

    def _validate_startup(self, capabilities: ModelCapabilities, decoder_type: str) -> None:
        """Fail at startup — with a useful message — on any missing capability."""
        if capabilities.sample_rate and capabilities.sample_rate != 16000:
            raise CapabilityUnavailable(
                "Shenava requires a 16 kHz mono frontend; this checkpoint's "
                f"preprocessor is configured for {capabilities.sample_rate} Hz"
            )
        if decoder_type == "ctc" and not capabilities.has_ctc_head:
            raise CapabilityUnavailable(
                f"decoder=ctc requested but {capabilities.model_class} exposes no CTC head "
                f"(available: {', '.join(capabilities.heads()) or 'none'})"
            )
        if decoder_type == "rnnt":
            if not capabilities.has_rnnt_head:
                raise CapabilityUnavailable(
                    f"decoder=rnnt requested but {capabilities.model_class} exposes no "
                    "RNNT prediction network/joint "
                    f"(available: {', '.join(capabilities.heads()) or 'none'}). "
                    "Use --decoder ctc or a hybrid checkpoint such as "
                    "Reza2kn/Shenava-Koochik-v1.5."
                )
            if capabilities.vocab_size is None:
                raise CapabilityUnavailable(
                    "decoder=rnnt requires the model's own tokenizer/vocabulary; "
                    "this checkpoint exposes none"
                )
            if capabilities.rnnt_blank_index is None:
                raise CapabilityUnavailable(
                    "decoder=rnnt requires a resolvable RNNT blank id; the loaded "
                    "checkpoint's joint/decoding exposes none"
                )
        if self.config.use_cache_aware_streaming and capabilities.streaming_contexts:
            requested = (70, int(self.config.right_context))
            if requested not in capabilities.streaming_contexts:
                raise CapabilityUnavailable(
                    "Encoder supports streaming contexts "
                    f"{[list(c) for c in capabilities.streaming_contexts]}; "
                    f"right_context={self.config.right_context} is not one of them. "
                    "Choose a supported right context (--right-context / "
                    "SHENAVA_RIGHT_CONTEXT)."
                )
        if self.config.require_streaming and not capabilities.supports_cache_aware_streaming:
            raise CapabilityUnavailable(
                "require_streaming is set but this checkpoint exposes no "
                "cache-aware streaming API (conformer_stream_step)"
            )

    def _select_head(self, model: Any, decoder_type: str) -> None:
        """Switch the model to the requested head via its own NeMo API."""
        decoding = _decoding_config()
        change = getattr(model, "change_decoding_strategy", None)
        if not callable(change):
            raise CapabilityUnavailable(
                f"{type(model).__name__} has no change_decoding_strategy(); "
                "explicit head selection is not possible for this checkpoint"
            )
        try:
            # Hybrid checkpoints (v1.5) take decoder_type; single-head models
            # reject it — that difference is probed, not guessed.
            change(decoding, decoder_type=decoder_type)
        except TypeError:
            if decoder_type != "ctc":
                raise CapabilityUnavailable(
                    f"{type(model).__name__}.change_decoding_strategy() does not accept "
                    f"decoder_type; head '{decoder_type}' cannot be selected"
                ) from None
            change(decoding)

    def _apply_cuda_graph_option(self, model: Any) -> None:
        """Optional CUDA-graph streaming; eager execution is the reference path.

        Off by default and never enabled implicitly — no profiling in this
        repository demonstrates a benefit.  When requested and unsupported by
        the installed NeMo/decoding objects, this raises rather than pretending
        the optimisation was applied.
        """
        if not self.config.cuda_graph_streaming:
            return
        decoding = getattr(model, "decoding", None)
        target = getattr(decoding, "decoding", None) if decoding is not None else None
        if target is None or not hasattr(target, "use_cuda_graph_decoder"):
            raise CapabilityUnavailable(
                "cuda_graph_streaming was requested but the loaded decoding "
                "strategy exposes no use_cuda_graph_decoder flag; unset "
                "SHENAVA_CUDA_GRAPH_STREAMING to use the eager reference path"
            )
        target.use_cuda_graph_decoder = True
        logger.info("CUDA-graph streaming enabled (optimisation, not the reference path)")

    def _probe_rnnt_blank(self, model: Any) -> Optional[int]:
        for holder_name in ("decoding", "joint", "decoder"):
            holder = getattr(model, holder_name, None)
            if holder is None:
                continue
            for attr in ("blank_id", "blank_idx", "_blank_index", "blank_index"):
                value = getattr(holder, attr, None)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return value
        return None

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
        """Cache-aware stream for the selected head, or ``None`` (endpoint-only).

        The RNNT stream reuses the tested CTC frontend accounting; only the
        active decoding head differs, and the required capability is checked
        before anything is decoded.
        """
        if self._model is None:
            self.load()
        if not self.config.use_cache_aware_streaming or not self._streaming_supported:
            return None
        torch = self._import_torch()
        if self.decoder_type == "rnnt":
            from .rnnt_stream import RNNTCacheAwareStream
            return RNNTCacheAwareStream(self._model, torch, self.config.max_segment_s)
        from .native_stream import NeMoCacheAwareStream
        return NeMoCacheAwareStream(self._model, torch, self.config.max_segment_s)

    def build_rnnt_decoder(self):
        """Offline RNNT decoder (benchmark + RNNT endpoint second pass)."""
        from .rnnt_stream import RNNTOfflineDecoder
        return RNNTOfflineDecoder(self.model, self._import_torch())

    # ------------------------------------------------------------------ #
    # Context second pass (CTC beam + hotword biasing).  Every probe below is
    # explicit: a mismatched checkpoint API raises SecondPassUnavailable at
    # startup instead of silently running a different decoder.
    # ------------------------------------------------------------------ #
    def build_second_pass(self, config: "ASRConfig"):
        """Decoder-aware endpoint second pass.

        CTC -> CTC beam + reviewed hotword bias.  RNNT -> RNNT re-decode.  The
        two heads are never cross-run here; that only happens in the offline
        benchmark tool.
        """
        from .second_pass import BeamSecondPass, GreedySecondPass

        model = self.model  # ensures the checkpoint is loaded
        if self.decoder_type == "rnnt":
            from .rnnt_stream import RNNTSecondPass
            logger.info("second pass ready: rnnt endpoint re-decode")
            return RNNTSecondPass(self.build_rnnt_decoder())
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
            config.hotword_acoustic_gate,
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

    def _probe_blank_index(self, model: Any, vocab_size: Optional[int]) -> int:
        for module_name in ("ctc_decoder", "_ctc_decoder", "ctc_decoding", "_ctc_decoding"):
            decoder = getattr(model, module_name, None)
            if decoder is None:
                continue
            for attr in ("ctc_blank_index", "blank_index", "_blank_index"):
                value = getattr(decoder, attr, None)
                if isinstance(value, int) and 0 <= value < (vocab_size or value + 1):
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
