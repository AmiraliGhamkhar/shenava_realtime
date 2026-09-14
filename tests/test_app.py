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
