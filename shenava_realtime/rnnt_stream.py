"""RNNT (transducer) decoding for the hybrid Shenava v1.5 checkpoint.

This module adds **no** transducer algorithm of its own.  Decoding is always
performed by the model's own prediction network, joint and greedy RNNT
decoding strategy — the same objects NeMo builds when
``change_decoding_strategy(decoder_type="rnnt")`` is called.  What lives here
is only the realtime plumbing:

* :class:`RNNTCacheAwareStream` drives ``conformer_stream_step`` with the RNNT
  head selected, reusing the frontend/cache accounting of the tested CTC
  adapter (:mod:`native_stream`) so frame handling stays identical;
* :class:`RNNTOfflineDecoder` is the utterance decode used for benchmarking
  and for the RNNT endpoint second pass.

CTC remains the production path.  Nothing here runs unless the RNNT head was
explicitly selected, and no CTC/RNNT cross-running happens outside the offline
benchmark tool.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

from .native_stream import NeMoCacheAwareStream

logger = logging.getLogger(__name__)


class RNNTCapabilityError(RuntimeError):
    """The loaded checkpoint cannot stream or decode with its RNNT head."""


class RNNTCacheAwareStream(NeMoCacheAwareStream):
    """Cache-aware streaming with the RNNT head active.

    ``conformer_stream_step`` dispatches to whichever head the model's current
    decoding strategy selects, so the only difference from the CTC stream is
    the required capability check: a hybrid checkpoint must actually expose a
    prediction network and joint, otherwise this fails immediately instead of
    quietly producing CTC text.
    """

    def __init__(self, model: Any, torch: Any, max_segment_s: float = 22.0) -> None:
        require_rnnt_head(model)
        super().__init__(model, torch, max_segment_s)

    @property
    def head(self) -> str:
        return "rnnt"


def require_rnnt_head(model: Any) -> None:
    """Fail clearly when the model has no usable RNNT prediction net/joint."""
    if getattr(model, "joint", None) is None or getattr(model, "decoder", None) is None:
        raise RNNTCapabilityError(
            f"{type(model).__name__} exposes no RNNT prediction network/joint; "
            "select --decoder ctc or load a hybrid checkpoint "
            "(Reza2kn/Shenava-Koochik-v1.5)"
        )
    decoding = getattr(model, "decoding", None)
    if decoding is None:
        raise RNNTCapabilityError(
            f"{type(model).__name__} exposes no RNNT decoding strategy object"
        )


class RNNTOfflineDecoder:
    """One offline utterance decode with the model's own RNNT decoding.

    Used by the RNNT endpoint second pass and by the benchmark tool.  It calls
    ``model.transcribe`` while the RNNT head is selected, so the prediction
    network, joint, tokenizer and blank are all the checkpoint's own.
    """

    name = "rnnt-offline"

    def __init__(self, model: Any, torch: Any) -> None:
        require_rnnt_head(model)
        self._model = model
        self._torch = torch
        self.decodes = 0

    def transcribe(self, audio: np.ndarray) -> Tuple[str, float]:
        samples = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        if samples.size == 0:
            return "", 0.0
        self.decodes += 1
        with self._torch.no_grad():
            outputs = self._model.transcribe(
                [samples], batch_size=1, verbose=False, return_hypotheses=True
            )
        return _extract_text(outputs), 0.0


def _extract_text(outputs: Any) -> str:
    """Pull the hypothesis text out of the shapes NeMo's transcribe returns."""
    if outputs is None:
        return ""
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not outputs:
        return ""
    first = outputs[0]
    if isinstance(first, (list, tuple)):
        return _extract_text(first)
    if isinstance(first, str):
        return first.strip()
    if isinstance(first, dict):
        return str(first.get("text", "") or "").strip()
    return str(getattr(first, "text", "") or "").strip()


class RNNTSecondPass:
    """Endpoint RNNT second pass: streaming RNNT -> offline RNNT re-decode.

    Deliberately decoder-aware: an RNNT stream is only ever re-decoded with
    RNNT.  There is no contextual/beam variant here — the reviewed hotword
    bias applies to CTC emissions and is not transferred to the transducer.
    """

    name = "rnnt-endpoint"

    def __init__(self, decoder: RNNTOfflineDecoder) -> None:
        self._decoder = decoder
        self.runs = 0

    def decode_greedy(self, utterance) -> Optional[str]:
        samples = np.asarray(utterance.audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None
        self.runs += 1
        text, _ = self._decoder.transcribe(samples)
        text = (text or "").strip()
        return text or None

    def decode_with_context(self, utterance, hotwords) -> Optional[str]:
        # No transducer-side contextual biasing: returning None makes the
        # caller use the plain RNNT re-decode, explicitly and predictably.
        return None
