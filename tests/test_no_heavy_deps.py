"""The library must import — and the tests must run — without the model stack."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

MODULES = [
    "shenava_realtime",
    "shenava_realtime.config",
    "shenava_realtime.text_normalize",
    "shenava_realtime.fa_numbers",
    "shenava_realtime.fst",
    "shenava_realtime.lexicon",
    "shenava_realtime.postprocessor",
    "shenava_realtime.stabilizer",
    "shenava_realtime.vad",
    "shenava_realtime.streaming",
    "shenava_realtime.second_pass",
    "shenava_realtime.hotwords",
    "shenava_realtime.pipeline",
    "shenava_realtime.value_validation",
    "shenava_realtime.variants",
    "shenava_realtime.audio_diagnostics",
    "shenava_realtime.realtime_engine",
    "shenava_realtime.audio_capture",
    "shenava_realtime.asr_backend",
    "shenava_realtime.injector.text_injector",
    "shenava_realtime.hotkeys.hotkey_manager",
    "shenava_realtime.utils",
]

FORBIDDEN = ("torch", "nemo", "torchaudio", "sounddevice", "pynput", "pyautogui", "pyperclip")


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_the_model_stack(module: str):
    code = (
        f"import sys; import {module};"
        f"bad=[m for m in {FORBIDDEN!r} if m in sys.modules];"
        "print('LEAK:' + ','.join(bad)) if bad else print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK", f"{module} pulled in heavy deps: {result.stdout}"


def test_overlay_module_imports_without_tkinter():
    code = (
        "import sys;"
        "sys.modules['tkinter'] = None;"
        "import shenava_realtime.overlay.overlay_window;"
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
