"""Injects stable transcript deltas into the focused application.

Two backends:

``clipboard`` (default on Windows)
    write the delta to the clipboard as CF_UNICODETEXT and send ``Ctrl+V``.
    This is the only reliable way to insert Persian text: simulated keystrokes
    go through the active keyboard layout and mangle non-Latin characters.

``keyboard``
    type through pynput, which sends Unicode directly.  Used on platforms
    without clipboard control and as a fallback.

The injector only ever receives *deltas* (see ``pipeline.text_delta``), and it
additionally refuses to send the same delta twice in a row, so a duplicated
callback can never duplicate output.  The guard is consecutive-only on purpose:
a time window would swallow legitimate short deltas such as a repeated " و".
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

from ..config import InjectorConfig, InjectorMode
from .clipboard_manager import ClipboardManager
from .keyboard_simulator import KeyboardSimulator

logger = logging.getLogger(__name__)


class InjectionResult(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class InjectionTask:
    text: str
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 2


class TextInjector:
    """Queue-based text injector with a duplicate guard."""

    def __init__(
        self,
        config: Optional[InjectorConfig] = None,
        clipboard: Optional[ClipboardManager] = None,
        keyboard: Optional[KeyboardSimulator] = None,
    ) -> None:
        self.config = config or InjectorConfig()
        self.clipboard = clipboard if clipboard is not None else ClipboardManager()
        self.keyboard = keyboard if keyboard is not None else KeyboardSimulator()
        self.mode: InjectorMode = self._resolve_mode(self.config.mode)

        self.is_active: bool = self.config.enabled
        self._queue: "queue.Queue[Optional[InjectionTask]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._last_text = ""

        self.on_complete: Optional[Callable[[str], None]] = None
        self.on_failed: Optional[Callable[[str], None]] = None
        self.stats: Dict[str, int] = {
            "injections": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "characters": 0,
        }
        logger.info("text injector ready (mode: %s, active: %s)", self.mode.value, self.is_active)

    # ------------------------------------------------------------------ #
    def _resolve_mode(self, mode: InjectorMode) -> InjectorMode:
        if mode is not InjectorMode.AUTO:
            return mode
        # Clipboard paste is the Unicode-safe path on Windows.
        return InjectorMode.CLIPBOARD if sys.platform == "win32" else InjectorMode.KEYBOARD

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            if self.mode is InjectorMode.CLIPBOARD and not self.clipboard.available:
                logger.warning("clipboard unavailable; falling back to keyboard injection")
                self.mode = InjectorMode.KEYBOARD
            if self.mode is InjectorMode.KEYBOARD and not self.keyboard.available:
                logger.error("no injection backend available (install pynput or enable a clipboard)")
            self._thread = threading.Thread(target=self._loop, name="text-injector", daemon=True)
            self._thread.start()
            logger.info("text injector started")

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None:
            self._queue.put(None)
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.warning("text injector thread did not exit within 3s")
            logger.info("text injector stopped")

    def toggle(self) -> None:
        self.is_active = not self.is_active
        logger.info("text injection %s", "enabled" if self.is_active else "disabled")

    def set_mode(self, mode: InjectorMode) -> None:
        self.config.mode = mode
        self.mode = self._resolve_mode(mode)
        logger.info("injection mode: %s", self.mode.value)

    # ------------------------------------------------------------------ #
    def inject(self, text: str) -> bool:
        """Queue a delta for injection. Returns False when it was ignored.

        The delta is queued verbatim (including a leading space) so that the
        injected text keeps the exact spacing of the transcript.
        """
        if not self.is_active:
            return False
        if not text or not text.strip():
            return False
        self._queue.put(InjectionTask(text=text))
        return True

    def inject_now(self, text: str) -> InjectionResult:
        """Inject synchronously (used by tests and by ``--print``-style flows)."""
        return self._perform(text)

    def flush(self, timeout: float = 2.0) -> None:
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.01)

    # ------------------------------------------------------------------ #
    def _loop(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                break
            result = self._perform(task.text, task=task)
            if result is InjectionResult.FAILED and task.attempts < task.max_attempts:
                task.attempts += 1
                time.sleep(0.25)
                self._queue.put(task)

    def _perform(self, text: str, task: Optional[InjectionTask] = None) -> InjectionResult:
        text = text or ""
        if not text.strip():
            return InjectionResult.SKIPPED
        if not self.is_active:
            return InjectionResult.SKIPPED

        if self.config.skip_consecutive_duplicates and text == self._last_text:
            self.stats["skipped"] += 1
            logger.debug("skipping duplicate injection: %r", text[:40])
            return InjectionResult.SKIPPED

        self.stats["injections"] += 1
        ok = self._inject_clipboard(text) if self.mode is InjectorMode.CLIPBOARD else self._inject_keyboard(text)

        if ok:
            self._last_text = text
            self.stats["successful"] += 1
            self.stats["characters"] += len(text)
            if self.on_complete is not None:
                self._notify(self.on_complete, text)
            return InjectionResult.SUCCESS

        self.stats["failed"] += 1
        logger.warning("injection failed (%d chars)", len(text))
        if self.on_failed is not None:
            self._notify(self.on_failed, text)
        return InjectionResult.FAILED

    # ------------------------------------------------------------------ #
    def _inject_clipboard(self, text: str) -> bool:
        if not self.clipboard.available:
            logger.warning("clipboard unavailable; using the keyboard backend instead")
            return self._inject_keyboard(text)
        if not self.keyboard.available:
            logger.error("cannot paste: no keyboard backend to send Ctrl+V")
            return False

        previous = self.clipboard.get() if self.config.restore_clipboard else None
        if not self.clipboard.set(text):
            return False
        time.sleep(max(0.0, self.config.delay_after_paste_s))
        ok = self.keyboard.press_combo(["ctrl", "v"])
        if ok:
            self._finish_extras()
        if self.config.restore_clipboard:
            self.clipboard.restore(previous)
        return ok

    def _inject_keyboard(self, text: str) -> bool:
        if not self.keyboard.available:
            return False
        ok = self.keyboard.type_text(text, interval=self.config.delay_between_keys)
        if ok:
            self._finish_extras()
        return ok

    def _finish_extras(self) -> None:
        if self.config.send_space_after:
            self.keyboard.type_text(" ")
        if self.config.send_enter_after:
            self.keyboard.press("enter")

    @staticmethod
    def _notify(callback, text: str) -> None:
        try:
            callback(text)
        except Exception:
            logger.exception("injector callback failed")

    # ------------------------------------------------------------------ #
    def get_statistics(self) -> Dict[str, object]:
        stats = dict(self.stats)
        stats["mode"] = self.mode.value
        stats["active"] = int(self.is_active)
        return stats
