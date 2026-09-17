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

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

NEG_INF = float("-inf")
# Acoustic-safety gate for contextual bias (nats below the frame's best token).
DEFAULT_BIAS_ACOUSTIC_GATE = 5.0
# Non-blank tokens this far below the frame best contribute no usable mass.
FRAME_TOKEN_CUTOFF = 12.0


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


def _logaddexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a)+exp(b)) without a numpy scalar round-trip."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


class CTCBeamDecoder:
    """Deterministic CTC prefix-beam search with optional token log-prob bias.

    ``emissions`` must be log-probabilities of shape [T, V] (apply log-softmax
    upstream).  This is the standard prefix beam: each surviving prefix carries
    two log probabilities — the mass ending in blank and the mass ending in a
    non-blank token — which are merged when ranking.  Everything is computed in
    log space; ``beam_size`` is the *actual* number of prefixes retained per
    frame (there is no hidden pruning floor).

    ``token_bias(prefix, token) -> float`` adds a bounded log-prob boost while
    the path continues a hotword prefix.  ``bias_acoustic_gate`` is the
    acoustic-safety gate: a bias is only applied when the biased token is
    within that many nats of the frame's best token, so contextual biasing can
    tip a near-tie but cannot force a medical term into unrelated speech.

    Ties break on the lexicographically smaller token sequence, so the output
    is deterministic for identical input.
    """

    def __init__(self, beam_size: int = 4, bias_acoustic_gate: float = DEFAULT_BIAS_ACOUSTIC_GATE) -> None:
        if not 1 <= int(beam_size) <= 32:
            raise ValueError("beam_size must be within [1, 32]")
        if not 0.0 < float(bias_acoustic_gate) <= 20.0:
            raise ValueError("bias_acoustic_gate must be within (0, 20] nats")
        self.beam_size = int(beam_size)
        self.bias_acoustic_gate = float(bias_acoustic_gate)

    def decode(
        self,
        emissions: np.ndarray,
        blank: int = 0,
        token_bias: Optional[Callable[[tuple, int], float]] = None,
    ) -> tuple[list[int], float]:
        emissions = np.ascontiguousarray(emissions, dtype=np.float64)
        if emissions.ndim != 2 or emissions.shape[1] < 2:
            raise ValueError("emissions must have shape [T, V] with V >= 2")
        blank = int(blank)
        if not 0 <= blank < emissions.shape[1]:
            raise ValueError("blank index out of range")

        # prefix -> [log p(ending in blank), log p(ending in non-blank)]
        beams: dict[tuple[int, ...], list[float]] = {(): [0.0, NEG_INF]}
        vocab = emissions.shape[1]
        for row in emissions:
            frame_best = float(row.max())
            # Only tokens with real acoustic mass can extend a prefix; this
            # bounds the work per frame without changing the retained beam.
            candidates = self._frame_tokens(row, blank, frame_best, vocab)
            next_beams: dict[tuple[int, ...], list[float]] = {}
            for prefix, (p_blank, p_nonblank) in beams.items():
                total = _logaddexp(p_blank, p_nonblank)
                last = prefix[-1] if prefix else -1
                # 1) blank: the prefix is unchanged, mass moves to p_blank.
                entry = next_beams.setdefault(prefix, [NEG_INF, NEG_INF])
                entry[0] = _logaddexp(entry[0], total + float(row[blank]))
                for token in candidates:
                    logp = float(row[token])
                    boost = 0.0
                    if token_bias is not None and logp >= frame_best - self.bias_acoustic_gate:
                        boost = float(token_bias(prefix, token))
                    logp += boost
                    if token == last:
                        # 2a) repeat without a blank: collapses into the prefix.
                        entry = next_beams.setdefault(prefix, [NEG_INF, NEG_INF])
                        entry[1] = _logaddexp(entry[1], p_nonblank + logp)
                        # 2b) repeat after a blank: a genuine second emission.
                        extended = next_beams.setdefault(prefix + (token,), [NEG_INF, NEG_INF])
                        extended[1] = _logaddexp(extended[1], p_blank + logp)
                    else:
                        extended = next_beams.setdefault(prefix + (token,), [NEG_INF, NEG_INF])
                        extended[1] = _logaddexp(extended[1], total + logp)
            beams = self._prune(next_beams)
            if not beams:
                break
        if not beams:
            return [], float(NEG_INF)
        best = min(
            beams.items(),
            key=lambda item: (-_logaddexp(item[1][0], item[1][1]), item[0]),
        )
        return list(best[0]), float(_logaddexp(best[1][0], best[1][1]))

    def _frame_tokens(self, row, blank: int, frame_best: float, vocab: int) -> list[int]:
        """Non-blank tokens worth expanding this frame (deterministic order)."""
        keep = np.flatnonzero(row >= frame_best - FRAME_TOKEN_CUTOFF)
        tokens = [int(t) for t in keep if int(t) != blank]
        if not tokens:
            tokens = [t for t in range(vocab) if t != blank]
        return tokens

    def _prune(self, beams: dict[tuple[int, ...], list[float]]) -> dict:
        """Keep exactly ``beam_size`` prefixes, ranked by total log-prob."""
        if len(beams) <= self.beam_size:
            return beams
        ranked = sorted(
            beams.items(),
            key=lambda item: (-_logaddexp(item[1][0], item[1][1]), item[0]),
        )[: self.beam_size]
        return dict(ranked)


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
        bias_acoustic_gate: float = DEFAULT_BIAS_ACOUSTIC_GATE,
    ) -> None:
        self._emissions = emissions_fn
        self._tokenize = tokenize
        self._decode_tokens = decode_tokens
        self._blank = int(blank_index)
        self.beam = CTCBeamDecoder(beam_size, bias_acoustic_gate)
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
