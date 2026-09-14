"""Shenava real-time Persian speech recognition.

Heavy optional dependencies are imported lazily so configuration and tooling can
be used without installing the microphone/model stack first.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .config import (
    AppConfig, ConfigManager, AudioConfig, ASRConfig, OverlayConfig,
    InjectorConfig, HotkeyConfig, OutputMode, OverlayPosition, InjectorMode,
)

__version__ = "1.0.0"
__description__ = "Real-time Persian ASR with overlay and text injection"

_LAZY_EXPORTS = {
    "RealtimeASR": (".realtime_engine", "RealtimeASR"),
    "AudioCapture": (".audio_capture", "AudioCapture"),
    "OverlayWindow": (".overlay.overlay_window", "OverlayWindow"),
    "OverlayManager": (".overlay.overlay_window", "OverlayManager"),
    "TextInjector": (".injector.text_injector", "TextInjector"),
    "ClipboardManager": (".injector.clipboard_manager", "ClipboardManager"),
    "HotkeyManager": (".hotkeys.hotkey_manager", "HotkeyManager"),
    "TextBuffer": (".utils", "TextBuffer"),
    "ConfidenceCalculator": (".utils", "ConfidenceCalculator"),
    "AudioLevelMeter": (".utils", "AudioLevelMeter"),
    "PerformanceMonitor": (".utils", "PerformanceMonitor"),
    "WindowManager": (".utils", "WindowManager"),
    "StateManager": (".utils", "StateManager"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        from importlib import import_module
        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AppConfig", "ConfigManager", "AudioConfig", "ASRConfig", "OverlayConfig",
    "InjectorConfig", "HotkeyConfig", "OutputMode", "OverlayPosition", "InjectorMode",
    *_LAZY_EXPORTS,
]
