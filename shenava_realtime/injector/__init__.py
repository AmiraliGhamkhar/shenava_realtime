"""Text injection backends (clipboard / keyboard)."""

from .clipboard_manager import ClipboardManager
from .keyboard_simulator import KeyboardSimulator
from .text_injector import InjectionResult, TextInjector

__all__ = ["ClipboardManager", "KeyboardSimulator", "TextInjector", "InjectionResult"]
