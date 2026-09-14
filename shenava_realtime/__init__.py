"""
Shenava Real-time ASR with Overlay and Injector
"""

import os

# Prefer the standard HTTP download path on HuggingFace; the Xet transfer
# backend can drop connections on some networks.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .config import (
    AppConfig, 
    ConfigManager, 
    AudioConfig, 
    ASRConfig, 
    OverlayConfig, 
    InjectorConfig, 
    HotkeyConfig,
    OutputMode,
    OverlayPosition,
    InjectorMode
)
from .realtime_engine import RealtimeASR
from .audio_capture import AudioCapture
from .overlay.overlay_window import OverlayWindow, OverlayManager
from .injector.text_injector import TextInjector, ClipboardManager
from .hotkeys.hotkey_manager import HotkeyManager
from .utils import (
    TextBuffer,
    ConfidenceCalculator,
    AudioLevelMeter,
    PerformanceMonitor,
    WindowManager,
    StateManager
)

__version__ = "1.0.0"
__description__ = "Real-time Persian ASR with overlay and text injection"

__all__ = [
    "AppConfig",
    "ConfigManager",
    "RealtimeASR",
    "AudioCapture",
    "OverlayWindow",
    "OverlayManager",
    "TextInjector",
    "ClipboardManager",
    "HotkeyManager",
    "OutputMode",
    "OverlayPosition",
    "InjectorMode"
]