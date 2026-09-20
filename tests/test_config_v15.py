"""Sherpa-ONNX configuration surface: model target, backend knobs, bounds."""
import pytest

from shenava_realtime.config import (
    DECODING_METHODS,
    DEFAULT_DECODING_METHOD,
    DEFAULT_MODEL_FILE,
    DEFAULT_TOKENS_FILE,
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
def test_the_default_model_target_is_the_sherpa_onnx_ctc_export(monkeypatch):
    # Isolated from whatever the process-wide .env may already have loaded
    # into os.environ (load_dotenv() never overrides an existing exported
    # variable, so once another test loads the repo .env this leaks for the
    # rest of the pytest session).
    monkeypatch.delenv("SHENAVA_MODEL_PATH", raising=False)
    monkeypatch.delenv("SHENAVA_TOKENS_PATH", raising=False)
    assert ASRConfig().model_path == str(DEFAULT_MODEL_FILE)
    assert ASRConfig().tokens_path == str(DEFAULT_TOKENS_FILE)


def test_the_default_checkpoint_path_names_the_canonical_models_directory(monkeypatch):
    monkeypatch.delenv("SHENAVA_MODEL_PATH", raising=False)
    assert "models/shenava" in ASRConfig().model_path.replace("\\", "/")


def test_downloads_are_never_automatic():
    """There is no allow_download knob: provisioning is manual-only, always."""
    assert not hasattr(ASRConfig(), "allow_download")


# --------------------------------------------------------------------------- #
# Backend selection: CTC-only, CPU-only
# --------------------------------------------------------------------------- #
def test_the_only_supported_backend_and_decoding_method():
    assert ASRConfig().asr_backend == "sherpa_onnx_ctc"
    assert DECODING_METHODS == ("greedy_search",)
    assert DEFAULT_DECODING_METHOD == "greedy_search"
    with pytest.raises(ValueError, match="asr_backend"):
        ASRConfig(asr_backend="nemo")


def test_device_is_cpu_only():
    assert ASRConfig().device == "cpu"
    with pytest.raises(ValueError, match="CPU-only"):
        ASRConfig(device="cuda")


def test_the_decoder_type_field_no_longer_exists():
    """The RNNT head was removed; there is no decoder_type/resolved_decoder."""
    config = ASRConfig()
    assert not hasattr(config, "decoder_type")
    assert not hasattr(config, "resolved_decoder")


def test_the_asr_backend_env_override_is_honoured(monkeypatch):
    monkeypatch.setenv("SHENAVA_ASR_BACKEND", "sherpa_onnx_ctc")
    config = apply_env_overrides(AppConfig())
    assert config.asr.asr_backend == "sherpa_onnx_ctc"


def test_the_unprefixed_asr_backend_alias_is_honoured(monkeypatch):
    monkeypatch.delenv("SHENAVA_ASR_BACKEND", raising=False)
    monkeypatch.setenv("ASR_BACKEND", "sherpa_onnx_ctc")
    config = apply_env_overrides(AppConfig())
    assert config.asr.asr_backend == "sherpa_onnx_ctc"


# --------------------------------------------------------------------------- #
# New bounded knobs
# --------------------------------------------------------------------------- #
def test_feature_dim_is_validated_against_the_model():
    assert ASRConfig().feature_dim == 80
    with pytest.raises(ValueError):
        ASRConfig(feature_dim=0)
    with pytest.raises(ValueError):
        ASRConfig(feature_dim=1000)


def test_num_threads_defaults_and_is_bounded():
    assert ASRConfig().num_threads == 4
    with pytest.raises(ValueError):
        ASRConfig(num_threads=0)


def test_sample_rate_is_locked_to_16k():
    assert ASRConfig().sample_rate == 16000
    with pytest.raises(ValueError, match="16 kHz"):
        ASRConfig(sample_rate=8000)


def test_require_streaming_is_the_production_default(monkeypatch):
    """No valid streaming recognizer means production startup failure.

    Tests and replay tools that intentionally exercise the endpoint fallback
    must opt out explicitly with ``require_streaming=False`` or
    ``SHENAVA_REQUIRE_STREAMING=0``.
    """
    assert ASRConfig().require_streaming is True
    monkeypatch.setenv("SHENAVA_REQUIRE_STREAMING", "0")
    config = apply_env_overrides(AppConfig())
    assert config.asr.require_streaming is False


def test_benchmark_mode_defaults_off():
    assert ASRConfig().benchmark_mode is False


def test_adaptive_vad_is_on_by_default_and_can_be_disabled():
    assert AudioConfig().vad_adaptive is True
    AudioConfig(vad_adaptive=False).__post_init__()  # stays valid


def test_env_overrides_for_the_new_knobs(monkeypatch):
    monkeypatch.setenv("SHENAVA_NUM_THREADS", "2")
    monkeypatch.setenv("SHENAVA_FEATURE_DIM", "80")
    monkeypatch.setenv("SHENAVA_DECODING_METHOD", "greedy_search")
    monkeypatch.setenv("SHENAVA_VAD_ADAPTIVE", "0")
    monkeypatch.setenv("SHENAVA_BENCHMARK_MODE", "yes")
    config = apply_env_overrides(AppConfig())
    assert config.asr.num_threads == 2
    assert config.asr.feature_dim == 80
    assert config.asr.decoding_method == "greedy_search"
    assert config.asr.benchmark_mode is True
    assert config.audio.vad_adaptive is False


# --------------------------------------------------------------------------- #
# Legacy NeMo-era env vars: explicit migration errors, not silently ignored
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,value", [
    ("SHENAVA_RIGHT_CONTEXT", "13"),
    ("SHENAVA_MODEL_NAME", "Reza2kn/Shenava-Koochik-v1.5"),
    ("SHENAVA_ALLOW_DOWNLOAD", "1"),
    ("SHENAVA_DECODER", "rnnt"),
    ("SHENAVA_BEAM_SIZE", "4"),
    ("SHENAVA_HOTWORD_ALIASES", "1"),
    ("SHENAVA_HOTWORD_PHONETIC", "1"),
    ("SHENAVA_CUDA_GRAPH_STREAMING", "1"),
])
def test_removed_env_vars_raise_a_clear_migration_error(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        apply_env_overrides(AppConfig())


def test_legacy_nemo_model_path_names_the_new_variables(monkeypatch):
    monkeypatch.setenv("SHENAVA_MODEL_PATH", "./Shenava-Koochik-v1.5/shenava-koochik-v1.5.nemo")
    with pytest.raises(ValueError, match="SHENAVA_TOKENS_PATH"):
        apply_env_overrides(AppConfig())


# --------------------------------------------------------------------------- #
# Hotword layer (unaffected by the ASR migration: build_hotwords is pure and
# backend-agnostic; only decoder-time biasing via it was removed)
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
