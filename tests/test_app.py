"""Application lifecycle with injected hardware doubles, real engine and rules."""
import numpy as np
import pytest
from main import ShenavaApp, main
from shenava_realtime.config import AppConfig, OutputMode
from tests.fakes import FakeAudioCapture, FakeBackend


@pytest.fixture(autouse=True)
def keep_pytest_logging(monkeypatch):
    monkeypatch.setattr("main.setup_logging", lambda *args, **kwargs: None)

def test_application_starts_drains_and_stops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = AppConfig()
    config.overlay.enabled = config.injector.enabled = False
    config.output_mode = OutputMode.CONSOLE
    capture = FakeAudioCapture()
    original_start = capture.start
    def start():
        original_start()
        capture.emit_utterance([np.full(1024, .05, np.float32)] * 8)
    capture.start = start
    app = ShenavaApp(config, backend=FakeBackend(lambda _: ('پنج میلی گرم', 0.0)), audio_capture=capture)
    app.request_shutdown()
    app.start()
    assert app.asr.transcript == '5 mg'
    assert not app.asr.is_running
    assert not capture.is_running


def test_missing_model_startup_is_readable(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(['--model', str(tmp_path/'missing.nemo'), '--no-overlay', '--no-inject']) == 1
    assert 'Local Shenava checkpoint not found' in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Early-commit legacy mode must be refused with medical output paths.
# --------------------------------------------------------------------------- #
def _config(**changes):
    config = AppConfig()
    config.overlay.enabled = False
    config.injector.enabled = False
    config.asr.commit_on_endpoint = False
    for key, value in changes.items():
        setattr(config, key, value)
    return config


def test_early_commit_with_injection_is_refused_at_startup():
    config = _config()
    config.injector.enabled = True  # default desktop configuration
    with pytest.raises(ValueError, match="commit_on_endpoint"):
        ShenavaApp(config, backend=FakeBackend(), audio_capture=FakeAudioCapture())


def test_early_commit_with_clinical_persistence_is_refused():
    config = _config(clinical_jsonl="clinical.jsonl")
    with pytest.raises(ValueError, match="commit_on_endpoint"):
        ShenavaApp(config, backend=FakeBackend(), audio_capture=FakeAudioCapture())


def test_early_commit_without_medical_output_still_loads():
    config = _config()
    app = ShenavaApp(config, backend=FakeBackend(), audio_capture=FakeAudioCapture())
    assert app.asr.pipeline.commit_on_endpoint is False
