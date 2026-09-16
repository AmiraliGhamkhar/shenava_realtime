"""Utterance-end second-pass decoding (optional, configuration-gated).

Live recognition stays on the cache-aware greedy stream.  At a *natural*
utterance end the engine may re-decode the same utterance audio once:

    greedy streaming CTC -> utterance-end CTC (greedy, or beam + hotwords)

Rules that keep this safe:

* the second pass only replaces the final hypothesis of an utterance that has
  not been emitted yet (endpoint-commit mode); it never rewrites text that was
  already injected;
* it never runs on a forced boundary (the audio is incomplete by definition);
* implementations are deterministic and return ``None`` when they cannot
  produce a usable hypothesis — the caller keeps the streaming greedy result
  and records the fallback explicitly, instead of switching modes silently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Hotword:
    """One spoken phrase with a conservative log-prob boost (see hotwords.py)."""
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

    The offline pass uses the model's own preprocessor and full context, so it
    is the safe default second pass: same model, same BPE vocabulary, no bias.
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


class CTCBeamDecoder:
    """Deterministic CTC beam search with optional token-level log-prob bias.

    ``emissions`` must be log-probabilities of shape [T, V] (apply
    log-softmax upstream).  A token bias is added while the path so far is a
    prefix of a hotword — the only contextual biasing this module supports;
    there is no language model and no fuzzy matching.  Ties break on the
    lexicographically smaller token sequence, so output is deterministic.
    """

    def __init__(self, beam_size: int = 4) -> None:
        if not 1 <= int(beam_size) <= 32:
            raise ValueError("beam_size must be within [1, 32]")
        self.beam_size = int(beam_size)

    def decode(
        self,
        emissions: np.ndarray,
        blank: int = 0,
        token_bias: Optional[Callable[[tuple, int], float]] = None,
    ) -> tuple[list[int], float]:
        emissions = np.ascontiguousarray(emissions, dtype=np.float64)
        if emissions.ndim != 2 or emissions.shape[1] < 2:
            raise ValueError("emissions must have shape [T, V] with V >= 2")
        if not 0 <= int(blank) < emissions.shape[1]:
            raise ValueError("blank index out of range")

        # State: (sequence, last_token) -> log-prob.  Two entries per sequence:
        # one where the previous frame was blank, one where it was not.
        states: dict[tuple[tuple[int, ...], int], float] = {((), -1): 0.0}
        for row in emissions:
            next_states: dict[tuple[tuple[int, ...], int], float] = {}
            for (seq, last), logp in states.items():
                # Emit blank: sequence unchanged.
                key = (seq, blank)
                next_states[key] = max(next_states.get(key, -np.inf), logp + row[blank])
                # Emit each non-blank token.
                for v in range(row.shape[0]):
                    if v == blank:
                        continue
                    new_seq = seq + (v,) if last != v else seq
                    # The bias keys on the prefix *before* this token.
                    boost = token_bias(seq, v) if token_bias is not None else 0.0
                    key = (new_seq, v)
                    next_states[key] = max(
                        next_states.get(key, -np.inf), logp + row[v] + boost
                    )
            states = self._prune(next_states)
            if not states:
                break
        best_logp = max(logp for _, logp in states.items())
        sequences = sorted({
            seq for (seq, _last), logp in states.items() if logp == best_logp
        })
        return list(sequences[0]), float(best_logp)

    @staticmethod
    def _prune(states: dict[tuple[tuple[int, ...], int], float]) -> dict:
        if len(states) <= 2 * 64:
            return states
        # Keep the strongest state per sequence, then keep the best sequences.
        by_seq: dict[tuple[int, ...], tuple[int, float]] = {}
        for (seq, last), logp in states.items():
            if seq not in by_seq or logp > by_seq[seq][1]:
                by_seq[seq] = (last, logp)
        ranked = sorted(by_seq.items(), key=lambda kv: (-kv[1][1], kv[0]))[: 2 * 64]
        return {(seq, last): logp for seq, (last, logp) in ranked}


def hotword_token_bias(
    hotword_sequences: Sequence[tuple[tuple[int, ...], float]],
) -> Callable[[tuple, int], float]:
    """Build the prefix bias function: boost a token while it continues a
    hotword's token sequence.  ``hotword_sequences`` is [(token ids, bias)]."""
    prefix_bias: dict[tuple[tuple[int, ...], int], float] = {}
    for sequence, bias in hotword_sequences:
        if bias == 0.0 or not sequence:
            continue
        for k in range(len(sequence)):
            prefix_bias[(sequence[:k], sequence[k])] = bias
    return lambda seq, token: prefix_bias.get((seq, token), 0.0)


class BeamSecondPass(SecondPassDecoder):
    """CTC beam + terminology hotword biasing over model emissions.

    ``emissions_fn(audio) -> [T, V] log-probs``; ``tokenize(text) -> ids`` and
    ``decode_tokens(ids) -> text`` come from the model's own tokenizer (no new
    vocabulary).  Hotwords that cannot be tokenized are skipped; the beam then
    runs unbiased.  Emission failures propagate to the caller (the pipeline
    records the fallback and keeps the streaming text).  Only a beam that
    produced nothing (all blanks) degrades to the greedy offline pass.
    """

    name = "beam-hotword"

    def __init__(
        self,
        emissions_fn: Callable[[np.ndarray], np.ndarray],
        tokenize: Callable[[str], list[int]],
        decode_tokens: Callable[[list[int]], str],
        blank_index: int,
        beam_size: int = 4,
        greedy: Optional[SecondPassDecoder] = None,
    ) -> None:
        self._emissions = emissions_fn
        self._tokenize = tokenize
        self._decode_tokens = decode_tokens
        self._blank = int(blank_index)
        self.beam = CTCBeamDecoder(beam_size)
        self._greedy = greedy or GreedySecondPass(lambda a: ("", 0.0))

    def decode_greedy(self, utterance: SecondPassUtterance) -> Optional[str]:
        return self._greedy.decode_greedy(utterance)

    def decode_with_context(
        self, utterance: SecondPassUtterance, hotwords: Sequence[Hotword]
    ) -> Optional[str]:
        # Hotwords that cannot be tokenized are skipped (never guessed at);
        # the beam still runs, unbiased when nothing survives.  Emission
        # failures propagate so the pipeline records an explicit fallback.
        sequences = self._tokenized_hotwords(hotwords)
        emissions = self._emissions(np.asarray(utterance.audio, dtype=np.float32))
        bias = hotword_token_bias(sequences) if sequences else None
        tokens, _ = self.beam.decode(emissions, blank=self._blank, token_bias=bias)
        if not tokens:
            # The beam saw only blanks: the greedy offline pass is the result.
            return self.decode_greedy(utterance)
        text = self._decode_tokens(list(tokens))
        return text or None

    def _tokenized_hotwords(self, hotwords):
        sequences = []
        for hotword in hotwords:
            try:
                ids = [int(t) for t in self._tokenize(hotword.phrase)]
            except Exception:
                continue  # un-tokenizable phrase: skip, never guess
            if ids:
                sequences.append((tuple(ids), float(hotword.bias)))
        if not any(seq != () for seq, _ in sequences):
            return None
        return sequences
