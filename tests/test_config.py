"""Configuration: defaults, env overrides and JSON round-trip."""

import json

import pytest

from shenava_realtime.config import (
    AppConfig,
    AudioConfig,
    ConfigManager,
    DigitStyle,
    InjectorMode,
    OutputMode,
    OverlayPosition,
    apply_env_overrides,
    load_dotenv,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "SHENAVA_ASR_BACKEND",
        "ASR_BACKEND",
        "SHENAVA_MODEL_PATH",
        "SHENAVA_TOKENS_PATH",
        "SHENAVA_DEVICE",
        "SHENAVA_NUM_THREADS",
        "SHENAVA_SAMPLE_RATE",
        "SHENAVA_FEATURE_DIM",
        "SHENAVA_DECODING_METHOD",
        "SHENAVA_OUTPUT_MODE",
        "SHENAVA_INJECTOR_MODE",
        "SHENAVA_AUDIO_DEVICE",
        "SHENAVA_LOG_LEVEL",
        "SHENAVA_PARTIAL_INTERVAL_S",
        "SHENAVA_REQUIRE_STREAMING",
        "SHENAVA_DEBUG",
        "SHENAVA_SECOND_PASS",
        "SHENAVA_BENCHMARK_MODE",
        "SHENAVA_VAD_ADAPTIVE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_16k_mono_ctc():
    config = AppConfig()
    assert config.audio.sample_rate == 16000
    assert config.audio.channels == 1
    assert config.audio.chunk_size == 1024
    assert config.asr.asr_backend == "sherpa_onnx_ctc"
    assert config.asr.device == "cpu"
    assert config.audio.vad_onset_rms > config.audio.vad_offset_rms


def test_env_overrides(monkeypatch, tmp_path):
    model = tmp_path / "model.int8.onnx"
    monkeypatch.setenv("SHENAVA_MODEL_PATH", str(model))
    monkeypatch.setenv("SHENAVA_DEVICE", "cpu")
    monkeypatch.setenv("SHENAVA_NUM_THREADS", "8")
    monkeypatch.setenv("SHENAVA_OUTPUT_MODE", "inject")
    monkeypatch.setenv("SHENAVA_INJECTOR_MODE", "keyboard")
    monkeypatch.setenv("SHENAVA_LOG_LEVEL", "DEBUG")

    config = apply_env_overrides(AppConfig())
    assert config.asr.model_path == str(model)
    assert config.asr.device == "cpu"
    assert config.asr.num_threads == 8
    assert config.output_mode is OutputMode.INJECT_ONLY
    assert config.injector.mode is InjectorMode.KEYBOARD
    assert config.log_level == "DEBUG"


def test_invalid_env_values_are_ignored(monkeypatch):
    monkeypatch.setenv("SHENAVA_NUM_THREADS", "many")
    monkeypatch.setenv("SHENAVA_OUTPUT_MODE", "telepathy")
    config = apply_env_overrides(AppConfig())
    assert config.asr.num_threads == 4
    assert config.output_mode is OutputMode.BOTH


def test_dotenv_is_loaded_without_overriding_exports(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SHENAVA_DEVICE=cpu\n# comment\nSHENAVA_NUM_THREADS='2'\n", encoding="utf-8")
    monkeypatch.setenv("SHENAVA_NUM_THREADS", "4")
    load_dotenv(env_file)
    import os

    assert os.environ["SHENAVA_DEVICE"] == "cpu"
    assert os.environ["SHENAVA_NUM_THREADS"] == "4"  # exported value wins


def test_json_round_trip(tmp_path):
    config = AppConfig()
    config.output_mode = OutputMode.CLIPBOARD
    config.overlay.position = OverlayPosition.TOP_RIGHT
    config.injector.mode = InjectorMode.CLIPBOARD
    config.postprocess.digits = DigitStyle.PERSIAN
    config.asr.holdback_words = 3

    path = tmp_path / "config.json"
    ConfigManager.save(config, path)
    loaded = ConfigManager.load(path)

    assert loaded.output_mode is OutputMode.CLIPBOARD
    assert loaded.overlay.position is OverlayPosition.TOP_RIGHT
    assert loaded.injector.mode is InjectorMode.CLIPBOARD
    assert loaded.postprocess.digits is DigitStyle.PERSIAN
    assert loaded.asr.holdback_words == 3
    assert loaded.audio.sample_rate == config.audio.sample_rate


def test_saved_json_is_readable(tmp_path):
    path = tmp_path / "config.json"
    ConfigManager.save(AppConfig(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["output_mode"] == "both"
    assert data["asr"]["asr_backend"] == "sherpa_onnx_ctc"
    assert data["audio"]["sample_rate"] == 16000


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"asr": {"device": "cpu", "bogus": 1}}), encoding="utf-8")
    config = ConfigManager.load(path)
    assert config.asr.device == "cpu"


def test_malformed_section_raises(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"asr": "not-an-object"}), encoding="utf-8")
    with pytest.raises(ValueError):
        ConfigManager.load(path)


def test_missing_file_falls_back_to_defaults(tmp_path):
    config = ConfigManager.load(tmp_path / "nope.json")
    assert config.asr.asr_backend == "sherpa_onnx_ctc"


def test_chunk_duration_helper():
    assert AppConfig().audio.chunk_duration == pytest.approx(0.064)


def test_dropout_timeout_is_bounded():
    assert AppConfig().audio.dropout_timeout_s == 2.0
    with pytest.raises(ValueError):
        AudioConfig(dropout_timeout_s=0)
    with pytest.raises(ValueError):
        AudioConfig(dropout_timeout_s=61)
    with pytest.raises(ValueError):
        AudioConfig(dropout_timeout_s=float("nan"))


def test_second_pass_defaults_and_env_overrides(monkeypatch):
    config = AppConfig()
    assert config.asr.second_pass == "greedy"
    assert config.asr.second_pass_min_utterance_s == 0.5

    monkeypatch.setenv("SHENAVA_SECOND_PASS", "OFF")
    config = apply_env_overrides(AppConfig())
    assert config.asr.second_pass == "off"


def test_second_pass_invalid_values_rejected(monkeypatch):
    from shenava_realtime.config import ASRConfig

    monkeypatch.setenv("SHENAVA_SECOND_PASS", "beam-only")
    with pytest.raises(ValueError, match="second_pass"):
        apply_env_overrides(AppConfig())
    with pytest.raises(ValueError):
        ASRConfig(second_pass="")
    with pytest.raises(ValueError, match="context"):
        ASRConfig(second_pass="context")
