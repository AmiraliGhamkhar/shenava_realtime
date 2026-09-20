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
from typing import Callable, Optional, Sequence

import numpy as np


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
