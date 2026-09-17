"""v1.5 configuration surface: model target, head selection, new bounded knobs."""
import pytest

from shenava_realtime.config import (
    DECODER_TYPES,
    DEFAULT_DECODER_TYPE,
    DEFAULT_MODEL_NAME,
    AppConfig,
    ASRConfig,
    AudioConfig,
    apply_env_overrides,
)
from shenava_realtime.hotwords import VARIANT_BIAS_FACTOR, build_hotwords
from shenava_realtime.terminology import MAX_HOTWORD_BIAS, TerminologyRule, default_rules


# --------------------------------------------------------------------------- #
# Model target
# --------------------------------------------------------------------------- #
def test_the_default_model_target_is_v1_5():
    assert DEFAULT_MODEL_NAME == "Reza2kn/Shenava-Koochik-v1.5"
    assert ASRConfig().model_name == DEFAULT_MODEL_NAME


def test_the_default_checkpoint_path_names_v1_5(monkeypatch):
    monkeypatch.delenv("SHENAVA_MODEL_PATH", raising=False)
    assert "v1.5" in ASRConfig().model_path


def test_downloads_still_require_explicit_permission():
    assert ASRConfig().allow_download is False


# --------------------------------------------------------------------------- #
# Decoder selection
# --------------------------------------------------------------------------- #
def test_the_documented_decoder_names():
    assert DECODER_TYPES == ("ctc", "rnnt", "auto")
    assert DEFAULT_DECODER_TYPE == "ctc"


@pytest.mark.parametrize("name,resolved", [
    ("ctc", "ctc"), ("rnnt", "rnnt"), ("auto", "ctc"),
])
def test_each_decoder_name_resolves_predictably(name, resolved):
    assert ASRConfig(decoder_type=name).resolved_decoder == resolved


def test_the_decoder_env_override_is_honoured(monkeypatch):
    monkeypatch.setenv("SHENAVA_DECODER", "rnnt")
    config = apply_env_overrides(AppConfig())
    assert config.asr.resolved_decoder == "rnnt"


# --------------------------------------------------------------------------- #
# New bounded knobs
# --------------------------------------------------------------------------- #
def test_beam_size_is_bounded():
    assert ASRConfig().second_pass_beam_size == 4
    with pytest.raises(ValueError):
        ASRConfig(second_pass_beam_size=0)
    with pytest.raises(ValueError):
        ASRConfig(second_pass_beam_size=33)


def test_the_acoustic_gate_is_bounded():
    assert ASRConfig().hotword_acoustic_gate == 5.0
    with pytest.raises(ValueError, match="acoustic_gate"):
        ASRConfig(hotword_acoustic_gate=0.0)
    with pytest.raises(ValueError, match="acoustic_gate"):
        ASRConfig(hotword_acoustic_gate=25.0)


def test_alias_and_phonetic_bias_are_opt_in():
    config = ASRConfig()
    assert config.hotword_use_aliases is False
    assert config.hotword_use_phonetic_variants is False


def test_benchmark_and_cuda_graph_modes_default_off():
    assert ASRConfig().benchmark_mode is False
    assert ASRConfig().cuda_graph_streaming is False


def test_adaptive_vad_is_on_by_default_and_can_be_disabled():
    assert AudioConfig().vad_adaptive is True
    AudioConfig(vad_adaptive=False).__post_init__()  # stays valid


def test_env_overrides_for_the_new_knobs(monkeypatch):
    monkeypatch.setenv("SHENAVA_BEAM_SIZE", "8")
    monkeypatch.setenv("SHENAVA_HOTWORD_ALIASES", "1")
    monkeypatch.setenv("SHENAVA_HOTWORD_PHONETIC", "true")
    monkeypatch.setenv("SHENAVA_VAD_ADAPTIVE", "0")
    monkeypatch.setenv("SHENAVA_BENCHMARK_MODE", "yes")
    config = apply_env_overrides(AppConfig())
    assert config.asr.second_pass_beam_size == 8
    assert config.asr.hotword_use_aliases is True
    assert config.asr.hotword_use_phonetic_variants is True
    assert config.asr.benchmark_mode is True
    assert config.audio.vad_adaptive is False


# --------------------------------------------------------------------------- #
# Hotword layer
# --------------------------------------------------------------------------- #
def rule(**kwargs):
    base = dict(id="r", canonical="X", spoken_forms=("واژه",), category="medication")
    base.update(kwargs)
    return TerminologyRule(**base)


def test_aliases_and_variants_are_excluded_unless_enabled():
    rules = [rule(aliases=("نام دیگر",), phonetic_variants=("واژه دیگر",))]
    phrases = {h.phrase for h in build_hotwords(rules)}
    assert phrases == {"واژه"}


def test_enabled_aliases_and_variants_participate_at_a_lower_bias():
    rules = [rule(aliases=("نام دیگر",), phonetic_variants=("واژه دیگر",))]
    hotwords = build_hotwords(rules, use_aliases=True, use_phonetic_variants=True)
    by_phrase = {h.phrase: h.bias for h in hotwords}
    assert set(by_phrase) == {"واژه", "نام دیگر", "واژه دیگر"}
    assert by_phrase["نام دیگر"] == by_phrase["واژه"] * VARIANT_BIAS_FACTOR
    assert by_phrase["واژه دیگر"] < by_phrase["واژه"]


def test_latin_aliases_are_never_decoder_hotwords():
    """Latin canonicals are rendered outputs, not spoken phrases."""
    rules = [rule(aliases=("CABG", "%"))]
    phrases = {h.phrase for h in build_hotwords(rules, use_aliases=True)}
    assert phrases == {"واژه"}


def test_units_are_never_boosted():
    rules = [rule(id="u", category="unit", spoken_forms=("میلی گرم",))]
    assert build_hotwords(rules) == []


def test_generic_anatomy_needs_an_explicit_reviewed_bias():
    assert build_hotwords([rule(id="a", category="anatomy")]) == []
    boosted = build_hotwords([rule(id="a", category="anatomy", bias=0.5)])
    assert [h.bias for h in boosted] == [0.5]


def test_every_bias_stays_within_the_reviewed_bound():
    hotwords = build_hotwords(default_rules(), max_hotwords=512,
                              use_aliases=True, use_phonetic_variants=True)
    assert hotwords
    assert all(0.0 < h.bias <= MAX_HOTWORD_BIAS for h in hotwords)


def test_the_active_hotword_count_is_bounded():
    hotwords = build_hotwords(default_rules(), max_hotwords=10,
                              use_aliases=True, use_phonetic_variants=True)
    assert len(hotwords) == 10


def test_hotword_selection_is_deterministic():
    first = build_hotwords(default_rules(), max_hotwords=32)
    second = build_hotwords(default_rules(), max_hotwords=32)
    assert first == second


def test_the_expanded_terminology_is_reachable_as_hotwords():
    general = {h.phrase for h in build_hotwords(default_rules(), max_hotwords=512)}
    assert "وارفارین" in general      # newly reviewed high-risk medication
    # Specialty terms stay out of the general set until that specialty is asked for.
    assert "سونوگرافی" not in general
    radiology = {h.phrase for h in build_hotwords(
        default_rules(), specialty="radiology", max_hotwords=512)}
    assert "سونوگرافی" in radiology
