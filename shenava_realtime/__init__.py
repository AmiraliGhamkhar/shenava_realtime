"""Shenava real-time Persian speech recognition.

Heavy dependencies (torch, NeMo, sounddevice, pynput, tkinter) are imported
lazily so configuration, the text pipeline and the test-suite work in a minimal
environment.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .config import (  # noqa: E402
    ASRConfig,
    AppConfig,
    AudioConfig,
    ConfigManager,
    DigitStyle,
    HotkeyConfig,
    InjectorConfig,
    InjectorMode,
    OutputMode,
    OverlayConfig,
    OverlayPosition,
    PostProcessConfig,
)

__version__ = "2.0.0"
__description__ = "Real-time Persian ASR with deterministic FST post-processing"

_LAZY_EXPORTS = {
    "RealtimeASR": (".realtime_engine", "RealtimeASR"),
    "AudioCapture": (".audio_capture", "AudioCapture"),
    "EnergyVAD": (".vad", "EnergyVAD"),
    "VADConfig": (".vad", "VADConfig"),
    "TranscriptStabilizer": (".stabilizer", "TranscriptStabilizer"),
    "TranscriptionPipeline": (".pipeline", "TranscriptionPipeline"),
    "PostProcessor": (".postprocessor", "PostProcessor"),
    "TrieFST": (".fst", "TrieFST"),
    "TextInjector": (".injector.text_injector", "TextInjector"),
    "ClipboardManager": (".injector.clipboard_manager", "ClipboardManager"),
    "KeyboardSimulator": (".injector.keyboard_simulator", "KeyboardSimulator"),
    "HotkeyManager": (".hotkeys.hotkey_manager", "HotkeyManager"),
    "OverlayManager": (".overlay.overlay_window", "OverlayManager"),
    "TextBuffer": (".utils", "TextBuffer"),
    "PerformanceMonitor": (".utils", "PerformanceMonitor"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY_EXPORTS])


__all__ = [
    "AppConfig",
    "ConfigManager",
    "AudioConfig",
    "ASRConfig",
    "PostProcessConfig",
    "OverlayConfig",
    "InjectorConfig",
    "HotkeyConfig",
    "OutputMode",
    "OverlayPosition",
    "InjectorMode",
    "DigitStyle",
    *_LAZY_EXPORTS,
]
