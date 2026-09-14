"""Streaming pipeline: deltas are emitted once and only when stable."""

import numpy as np

from shenava_realtime.pipeline import TranscriptionPipeline, text_delta
from tests.fakes import make_pipeline


def feed(pipeline: TranscriptionPipeline, chunks: int = 4) -> list:
    deltas = []
    for _ in range(chunks):
        deltas.extend(pipeline.push_audio(np.zeros(8000, dtype=np.float32)))
    return deltas


def test_nothing_is_emitted_before_text_stabilizes():
    pipeline, decoder = make_pipeline(["بیمار"], holdback_words=2)
    pipeline.start_utterance()
    assert feed(pipeline, 1) == []
    assert pipeline.committed_text == ""
    assert pipeline.partial_text == "بیمار"


def test_deltas_reconstruct_the_transcript_exactly_once():
    pipeline, _ = make_pipeline(
        [
            "بیمار",
            "بیمار تحت",
            "بیمار تحت عمل",
            "بیمار تحت عمل کابج",
            "بیمار تحت عمل کابج است",
        ],
        holdback_words=2,
    )
    pipeline.start_utterance()
    emitted = feed(pipeline, 5)
    emitted.extend(pipeline.end_utterance())

    joined = "".join(emitted)
    assert joined == "بیمار تحت عمل CABG است"
    assert pipeline.committed_text == joined
    # No word is emitted twice.
    words = joined.split()
    assert len(words) == len(set(words))


def test_a_rewritten_hypothesis_never_duplicates_output():
    pipeline, _ = make_pipeline(
        [
            "بیمار تحت عمل",
            "بیمار زیر عمل",
            "بیمار زیر عمل کابج",
            "بیمار زیر عمل کابج است",
        ],
        holdback_words=1,
    )
    pipeline.start_utterance()
    emitted = feed(pipeline, 4)
    emitted.extend(pipeline.end_utterance())
    joined = "".join(emitted)
    assert joined == "بیمار زیر عمل CABG است"
    assert joined.count("بیمار") == 1
    assert joined.count("عمل") == 1


def test_post_processing_is_applied_to_emitted_text():
    pipeline, _ = make_pipeline(["سی و پنج درصد اکسیژن"], holdback_words=0)
    pipeline.start_utterance()
    feed(pipeline, 1)
    emitted = pipeline.end_utterance()
    assert "".join(emitted) == "35% اکسیژن"


def test_window_reset_commits_before_restarting():
    pipeline, _ = make_pipeline(
        ["بیمار تحت عمل", "RESET:کابج انجام شد"],
        holdback_words=1,
    )
    pipeline.start_utterance()
    emitted = feed(pipeline, 2)
    emitted.extend(pipeline.end_utterance())
    # The reset commits the first window, then appends the next one (with the
    # FST applied: کابج -> CABG) separated by a space.
    assert "".join(emitted) == "بیمار تحت عمل CABG انجام شد"


def test_abort_discards_the_utterance():
    pipeline, _ = make_pipeline(["بیمار تحت عمل"], holdback_words=0)
    pipeline.start_utterance()
    feed(pipeline, 1)
    pipeline.abort()
    assert pipeline.committed_text == ""
    assert pipeline.is_active is False


def test_audio_outside_an_utterance_is_ignored():
    pipeline, decoder = make_pipeline(["بیمار"], holdback_words=0)
    assert pipeline.push_audio(np.zeros(8000, dtype=np.float32)) == []
    assert decoder.pushes == 0


def test_start_utterance_seeds_the_pre_roll():
    pipeline, decoder = make_pipeline(["بیمار"], holdback_words=0)
    pipeline.start_utterance(np.zeros(16000, dtype=np.float32))
    assert decoder.pushes == 1


def test_end_utterance_is_idempotent():
    pipeline, _ = make_pipeline(["بیمار تحت عمل"], holdback_words=0)
    pipeline.start_utterance()
    feed(pipeline, 1)
    first = pipeline.end_utterance()
    assert "".join(first) == "بیمار تحت عمل"
    assert pipeline.end_utterance() == []


def test_confidence_is_tracked():
    pipeline, _ = make_pipeline(["بیمار"], holdback_words=0)
    pipeline.start_utterance()
    feed(pipeline, 1)
    assert pipeline.confidence > 0.0


def test_text_delta_helper():
    assert text_delta("", "بیمار") == "بیمار"
    assert text_delta("بیمار", "بیمار تحت") == " تحت"
    assert text_delta("بیمار تحت", "بیمار تحت") == ""
    # A divergence falls back to the last word boundary, never mid-word.
    assert text_delta("بیمار تحت عمل", "بیمار زیر عمل") == "زیر عمل"
    assert text_delta("بیم", "بیمار") == "ار"


def test_decoder_reset_between_utterances():
    pipeline, decoder = make_pipeline(["بیمار"], holdback_words=0)
    pipeline.start_utterance()
    pipeline.start_utterance()
    assert decoder.reset_calls == 2


# --------------------------------------------------------------------------- #
# Forced VAD/ASR-cap boundaries must not complete number phrases.
# --------------------------------------------------------------------------- #
def _fresh(pipeline, decoder, hypothesis):
    decoder.reset()
    decoder._hypotheses[:] = [hypothesis]
    pipeline.start_utterance()


def test_forced_boundary_keeps_open_number_tail_unparsed():
    """'سی و' | forced cut | 'پنج' must not become '30' and '5'."""
    pipeline, decoder = make_pipeline(["بیمار دوز سی و"], holdback_words=2)
    pipeline.start_utterance()
    first = pipeline.end_utterance(forced=True)
    assert "".join(first) == "بیمار دوز سی و"  # spoken words kept, no digits

    _fresh(pipeline, decoder, "پنج میلی گرم")
    second = pipeline.end_utterance()  # natural end: leading run still protected
    assert "".join(second) == "پنج mg"
    assert "5" not in "".join(second)

    # The continuation flag is consumed: a later standalone phrase parses.
    _fresh(pipeline, decoder, "پنج")
    third = pipeline.end_utterance()
    assert "".join(third) == "5"


def test_natural_endpoint_keeps_per_utterance_parsing():
    """A natural end is not tagged: documented per-utterance behavior stands."""
    pipeline, decoder = make_pipeline(["سی و"], holdback_words=2)
    pipeline.start_utterance()
    assert "".join(pipeline.end_utterance()) == "30 و"

    _fresh(pipeline, decoder, "پنج")
    assert "".join(pipeline.end_utterance()) == "5"
    assert pipeline._split_continues is False


def test_asr_window_reset_protects_the_number_across_the_cap():
    """Decoder-cap reset inside one utterance gets the same protection."""
    pipeline, decoder = make_pipeline(["دوز سی و", "RESET:پنج میلی گرم"], holdback_words=2)
    pipeline.start_utterance()
    emitted = feed(pipeline, 1)  # "دوز سی و"
    emitted.extend(feed(pipeline, 1))  # RESET -> commit + fresh accumulation
    emitted.extend(pipeline.end_utterance())
    joined = "".join(emitted)
    assert "30" not in joined and joined.count("پنج") == 1
    assert "سی و" in joined and "پنج mg" in joined


def test_abort_clears_pending_forced_continuation():
    pipeline, decoder = make_pipeline(["دوز سی و"], holdback_words=2)
    pipeline.start_utterance()
    pipeline.end_utterance(forced=True)
    assert pipeline._split_continues is True
    pipeline.abort()
    assert pipeline._split_continues is False

    _fresh(pipeline, decoder, "پنج")
    assert "".join(pipeline.end_utterance()) == "5"
