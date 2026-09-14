"""Keyboard simulation via pynput (Unicode aware on Windows/macOS/X11)."""

from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_MODIFIERS = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "cmd": "cmd",
    "win": "cmd",
    "enter": "enter",
    "return": "enter",
    "tab": "tab",
    "space": "space",
    "esc": "esc",
}


class KeyboardSimulator:
    """Types text and presses key combos through pynput."""

    def __init__(self) -> None:
        self._controller = None
        self._key_enum = None
        try:
            from pynput.keyboard import Controller, Key  # type: ignore

            self._controller = Controller()
            self._key_enum = Key
        except ImportError:
            logger.warning("pynput is not installed; keyboard output is unavailable")
        except Exception:
            logger.exception("could not initialise the keyboard controller")

    @property
    def available(self) -> bool:
        return self._controller is not None

    # ------------------------------------------------------------------ #
    def type_text(self, text: str, interval: float = 0.0) -> bool:
        if not text or not self.available:
            return False
        try:
            if interval > 0:
                for char in text:
                    self._controller.type(char)
                    time.sleep(interval)
            else:
                self._controller.type(text)
            return True
        except Exception:
            logger.exception("typing failed for %d characters", len(text))
            return False

    def press(self, key: str) -> bool:
        if not self.available:
            return False
        try:
            target = self._resolve(key)
            self._controller.press(target)
            self._controller.release(target)
            return True
        except Exception:
            logger.exception("could not press %r", key)
            return False

    def press_combo(self, keys: Iterable[str]) -> bool:
        """Press a chord such as ``["ctrl", "v"]`` (keys released in reverse)."""
        if not self.available:
            return False
        resolved = [self._resolve(key) for key in keys]
        pressed: list = []
        try:
            for target in resolved:
                self._controller.press(target)
                pressed.append(target)
            for target in reversed(pressed):
                self._controller.release(target)
            return True
        except Exception:
            logger.exception("could not press the combo %s", list(keys))
            for target in reversed(pressed):
                try:
                    self._controller.release(target)
                except Exception:
                    logger.debug("could not release %s", target)
            return False

    # ------------------------------------------------------------------ #
    def _resolve(self, key: str):
        name = str(key).strip().lower()
        mapped = _MODIFIERS.get(name, name)
        special: Optional[object] = getattr(self._key_enum, mapped, None) if self._key_enum else None
        return special if special is not None else (key if len(str(key)) == 1 else mapped)
