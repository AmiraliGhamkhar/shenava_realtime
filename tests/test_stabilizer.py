"""Transcript stabilization: committed_text / current_partial semantics."""

from shenava_realtime.stabilizer import (
    StabilizerConfig,
    TranscriptStabilizer,
    common_prefix_length,
)


def make(holdback: int = 2) -> TranscriptStabilizer:
    return TranscriptStabilizer(StabilizerConfig(holdback_words=holdback))


def test_nothing_is_committed_until_it_is_stable():
    st = make()
    assert st.update("بیمار") == ""
    assert st.committed_text == ""
    assert st.current_partial == "بیمار"


def test_words_are_committed_once_they_stop_moving():
    st = make()
    st.update("بیمار")
    st.update("بیمار تحت")
    assert st.update("بیمار تحت عمل") == "بیمار"
    assert st.committed_text == "بیمار"
    assert st.current_partial == "تحت عمل"


def test_holdback_is_respected():
    st = make(holdback=1)
    st.update("یک دو سه")
    assert st.committed_text == ""  # first hypothesis: nothing to compare with
    assert st.update("یک دو سه چهار") == "یک دو سه"
    assert st.current_partial == "چهار"


def test_committed_text_never_shrinks_or_duplicates():
    st = make()
    seen: list[str] = []
    for hypothesis in [
        "بیمار",
        "بیمار تحت",
        "بیمار تحت عمل",
        "بیمار تحت عمل کابج",
        "بیمار تحت عمل کابج است",
    ]:
        delta = st.update(hypothesis)
        if delta:
            seen.append(delta)
    final = st.finalize()
    if final:
        seen.append(final)

    joined = " ".join(seen)
    assert joined == "بیمار تحت عمل کابج است"
    assert st.committed_text == "بیمار تحت عمل کابج است"
    assert st.current_partial == ""
    # Every word appears exactly once in what was emitted.
    assert joined.split() == list(dict.fromkeys(joined.split()))


def test_a_rewritten_tail_is_not_committed():
    st = make()
    st.update("بیمار تحت عمل")
    st.update("بیمار زیر عمل")  # model changed its mind about word 2
    # Only the word that survived both hypotheses (and sits outside the
    # holdback window) may be committed.
    assert st.committed_text == "بیمار"
    assert st.current_partial == "زیر عمل"
    st.update("بیمار زیر عمل کابج")
    assert st.committed_text == "بیمار زیر"


def test_committed_words_survive_a_shorter_hypothesis():
    st = make(holdback=1)
    st.update("یک دو سه")
    st.update("یک دو سه چهار")
    assert st.committed_text == "یک دو سه"
    st.update("یک دو")  # the model dropped words it had already emitted
    assert st.committed_text == "یک دو سه"
    assert st.current_partial == ""


def test_finalize_commits_the_pending_tail():
    st = make()
    st.update("بیمار تحت عمل")
    assert st.finalize() == "بیمار تحت عمل"
    assert st.current_partial == ""


def test_finalize_with_a_hypothesis_adds_only_new_words():
    st = make(holdback=1)
    st.update("بیمار تحت")
    st.update("بیمار تحت عمل")
    committed_before = st.committed_text
    delta = st.finalize("بیمار تحت عمل کابج")
    assert delta == "عمل کابج"
    assert st.committed_text == f"{committed_before} عمل کابج"


def test_finalize_keeps_committed_text_when_the_model_diverges():
    st = make(holdback=0)
    st.update("بیمار تحت عمل")
    st.update("بیمار تحت عمل کابج")  # the first three words are now stable
    assert st.committed_text == "بیمار تحت عمل"
    delta = st.finalize("بیمار زیر عمل")
    assert delta == ""
    assert st.committed_text == "بیمار تحت عمل"


def test_reset_clears_everything():
    st = make()
    st.update("بیمار تحت عمل")
    st.reset()
    assert st.committed_text == ""
    assert st.current_partial == ""
    assert st.full_text == ""


def test_common_prefix_length():
    assert common_prefix_length(["a", "b"], ["a", "b", "c"]) == 2
    assert common_prefix_length(["a", "x"], ["a", "b"]) == 1
    assert common_prefix_length([], ["a"]) == 0
