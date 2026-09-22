"""Utterance-end second-pass decoding (optional, configuration-gated).

Live recognition stays on the streaming greedy CTC recognizer.  At a
*natural* utterance end the engine may re-decode the same utterance audio
once, offline, via the same model:

    greedy streaming CTC -> utterance-end greedy CTC (full-context re-decode)

Rules that keep this safe:

* the second pass only replaces the final hypothesis of an utterance that has
  not been emitted yet (endpoint-commit mode); it never rewrites text that was
  already injected;
* it never runs on a forced boundary (the audio is incomplete by definition);
* the implementation is deterministic and returns ``None`` when it cannot
  produce a usable hypothesis — the caller keeps the streaming greedy result
  and records the fallback explicitly, instead of switching modes silently.

A CTC-beam + terminology hotword-biasing second pass ("context" mode) existed
for the NeMo backend, which exposed raw per-frame emission logits and its own
BPE tokenizer.  sherpa-onnx's public Python API exposes only decoded text, not
per-frame emissions or a tokenizer object, so that mode is not implementable
here and is rejected at ``ASRConfig.__post_init__`` with a clear startup
error; there is no beam decoder or hotword-bias code path in this module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hotword:
    """One spoken phrase with a conservative log-prob boost (see hotwords.py).

    Retained as a data type because ``hotwords.build_hotwords`` still returns
    a list of these (used for observability/logging); no decoder in this
    backend consumes them for biasing.
    """
    phrase: str
    bias: float


@dataclass(frozen=True)
class SecondPassUtterance:
    """Opaque carrier for one utterance's raw 16 kHz mono float32 audio."""
    audio: np.ndarray
    duration_s: float


class SecondPassDecoder(ABC):
    """Small decoder interface: the pipeline knows nothing else about it."""

    name: str = "second-pass"

    @abstractmethod
    def decode_greedy(self, utterance: SecondPassUtterance) -> Optional[str]:
        """Re-decode the utterance greedily; None = no usable hypothesis."""

    def decode_with_context(
        self, utterance: SecondPassUtterance, hotwords: Sequence[Hotword]
    ) -> Optional[str]:
        """Decode with hotword/context biasing; None = caller uses greedy."""
        return None


class GreedySecondPass(SecondPassDecoder):
    """Offline greedy CTC via the existing backend (no new dependencies).

    The offline pass uses the model's own full-utterance context, so it is
    the safe default second pass: same model, same vocabulary, no bias.
    """

    name = "greedy"

    def __init__(self, transcribe: Callable[[np.ndarray], tuple[str, float]]) -> None:
        self._transcribe = transcribe

    def decode_greedy(self, utterance: SecondPassUtterance) -> Optional[str]:
        samples = np.asarray(utterance.audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None
        text, _ = self._transcribe(samples)  # score is uncalibrated; ignore
        text = (text or "").strip()
        return text or None


class BeamSecondPass(SecondPassDecoder):
    """Offline modified-beam-search re-decode (opt-in, model-dependent).

    Built only when ``ASRConfig.second_pass_beam`` is set *and* the backend
    exposes ``create_beam_recognizer()`` (a recognizer constructed with
    sherpa's ``modified_beam_search`` decoding). The wrapper is defensive on
    purpose: if the runtime/model rejects the method at decode time, the
    pass returns ``None`` and the pipeline keeps the greedy result with an
    explicit fallback counter — the same contract as every other optional
    stage.
    """

    name = "beam"

    def __init__(self, transcribe: Callable[[np.ndarray], tuple[str, float]]) -> None:
        self._transcribe = transcribe

    def decode_greedy(self, utterance: SecondPassUtterance) -> Optional[str]:
        samples = np.asarray(utterance.audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None
        try:
            text, _ = self._transcribe(samples)
        except Exception:
            logger.exception("beam second pass failed; caller keeps greedy result")
            return None
        text = (text or "").strip()
        return text or None
