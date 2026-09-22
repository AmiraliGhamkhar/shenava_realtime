"""Global hotkeys (pynput).

The listener thread only records which modifiers are down and dispatches
matching callbacks on short-lived worker threads, so a slow callback (for
example a model reload) can never block key event delivery.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from ..config import HotkeyConfig

logger = logging.getLogger(__name__)

_MODIFIER_ORDER = ("ctrl", "alt", "shift", "cmd")
_MODIFIER_KEYS = {
    "ctrl", "ctrl_l", "ctrl_r",
    "alt", "alt_l", "alt_r", "alt_gr",
    "shift", "shift_l", "shift_r",
    "cmd", "cmd_l", "cmd_r",
}
# With Ctrl held, pynput reports control characters; map them back to letters.
_CONTROL_CHAR_OFFSET = 0x40

# Bounded worker pool for hotkey callbacks (#33): rapid key presses no longer
# spawn a fresh thread per press. The pool is small on purpose — hotkey
# actions are UI-level (toggle overlay, clear transcript), and running two
# instances of the same callback is prevented separately by
# ``_running_bindings``.
_POOL_SIZE = 2


@dataclass
class HotkeyBinding:
    key_combo: str
    callback: Callable[[], None]
    description: str
    is_active: bool = True


class HotkeyManager:
    """Binds ``ctrl+alt+r``-style combos to callbacks."""

    def __init__(self, config: Optional[HotkeyConfig] = None) -> None:
        self.config = config or HotkeyConfig()
        self._running_bindings: set[str] = set()
        self._pool = ThreadPoolExecutor(
            max_workers=_POOL_SIZE, thread_name_prefix="hotkey"
        )
        self.bindings: Dict[str, HotkeyBinding] = {}
        self._pressed: Set[str] = set()
        self._pressed_lock = threading.Lock()
        self._listener = None
        self._keyboard_module = None
        self.stats = {"presses": 0, "triggered": 0, "failed": 0}

        try:
            from pynput import keyboard  # type: ignore

            self._keyboard_module = keyboard
        except ImportError:
            logger.warning("pynput is not installed; global hotkeys are disabled")
        except Exception:
            logger.exception("could not import the pynput keyboard module")

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return self._keyboard_module is not None

    @property
    def is_running(self) -> bool:
        return self._listener is not None

    def start(self) -> None:
        if not self.available:
            return
        # stop() shuts down the callback executor. Recreate it on restart;
        # otherwise the second application run silently drops hotkey callbacks.
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=_POOL_SIZE, thread_name_prefix="hotkey"
            )
        with self._pressed_lock:
            self._running_bindings.clear()
        if self.is_running:
            logger.debug("hotkey listener already running")
            return
        with self._pressed_lock:
            self._pressed.clear()
        try:
            self._listener = self._keyboard_module.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.daemon = True
            self._listener.start()
            logger.info("hotkey listener started (%d bindings)", len(self.bindings))
        except Exception:
            logger.exception("could not start the hotkey listener")
            self._listener = None

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is None:
            return
        try:
            listener.stop()
            listener.join(timeout=2.0)
        except Exception:
            logger.exception("error while stopping the hotkey listener")
        with self._pressed_lock:
            self._pressed.clear()
        self._shutdown_pool()
        logger.info("hotkey listener stopped")

    def _shutdown_pool(self) -> None:
        """Drain pending callbacks and stop the bounded worker pool."""
        pool = self._pool
        self._pool = None
        if pool is None:
            return
        pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ #
    def bind(self, key_combo: str, callback: Callable[[], None], description: str = "") -> None:
        combo = normalize_combo(key_combo)
        if not combo:
            logger.warning("ignoring empty hotkey binding")
            return
        existing = self.bindings.get(combo)
        if existing is not None:
            logger.warning("rebinding %s (was: %s)", combo, existing.description)
        self.bindings[combo] = HotkeyBinding(
            key_combo=combo,
            callback=callback,
            description=description or getattr(callback, "__name__", "callback"),
        )
        logger.debug("bound %s -> %s", combo, description or getattr(callback, "__name__", "callback"))

    def unbind(self, key_combo: str) -> None:
        combo = normalize_combo(key_combo)
        if self.bindings.pop(combo, None) is not None:
            logger.info("unbound %s", combo)

    def get_bindings(self) -> Dict[str, str]:
        return {combo: binding.description for combo, binding in self.bindings.items()}

    def get_statistics(self) -> Dict[str, int]:
        return dict(self.stats)

    def setup_default_hotkeys(
        self,
        toggle_recording: Callable[[], None],
        toggle_overlay: Callable[[], None],
        toggle_injector: Callable[[], None],
        clear_transcript: Callable[[], None],
        emergency_stop: Callable[[], None],
    ) -> None:
        self.bind(self.config.toggle_recording, toggle_recording, "Toggle recording")
        self.bind(self.config.toggle_overlay, toggle_overlay, "Show/hide overlay")
        self.bind(self.config.toggle_injector, toggle_injector, "Toggle text injection")
        self.bind(self.config.clear_transcript, clear_transcript, "Clear transcript")
        self.bind(self.config.emergency_stop, emergency_stop, "Emergency stop")

    # ------------------------------------------------------------------ #
    def _on_press(self, key) -> None:
        try:
            name = self._key_name(key)
            if name is None:
                return
            with self._pressed_lock:
                if name in self._pressed:
                    return  # ignore OS key-repeat
                self._pressed.add(name)
                if name in _MODIFIER_KEYS:
                    return
                combo = self._current_combo(name)
                binding = self.bindings.get(combo)
            if binding is None or not binding.is_active:
                return
            with self._pressed_lock:
                if combo in self._running_bindings:
                    return
                self._running_bindings.add(combo)
            self.stats["presses"] += 1
            logger.debug("hotkey pressed: %s", combo)
            pool = self._pool
            if pool is None:  # stopped while the listener event was in flight
                return
            try:
                pool.submit(self._run, binding)
            except RuntimeError:  # interpreter shutdown
                logger.warning("hotkey pool unavailable; callback for %s dropped", combo)
        except Exception:
            logger.exception("error in the hotkey press handler")

    def _on_release(self, key) -> None:
        try:
            name = self._key_name(key)
            if name is None:
                return
            with self._pressed_lock:
                self._pressed.discard(name)
                # Release the generic modifier too (pynput reports ctrl_r etc.)
                base = name.split("_")[0]
                self._pressed.discard(base)
        except Exception:
            logger.exception("error in the hotkey release handler")

    def _current_combo(self, key_name: str) -> str:
        modifiers = [name for name in _MODIFIER_ORDER if self._has_modifier(name)]
        return "+".join([*modifiers, key_name])

    def _has_modifier(self, modifier: str) -> bool:
        return any(name == modifier or name.startswith(f"{modifier}_") for name in self._pressed)

    def _key_name(self, key) -> Optional[str]:
        char = getattr(key, "char", None)
        if char:
            if len(char) == 1 and "\x01" <= char <= "\x1a":
                char = chr(ord(char) + _CONTROL_CHAR_OFFSET)
            return char.lower()
        name = getattr(key, "name", None)
        if name:
            return str(name).lower()
        return None

    def _run(self, binding: HotkeyBinding) -> None:
        try:
            binding.callback()
            self.stats["triggered"] += 1
        except Exception:
            self.stats["failed"] += 1
            logger.exception("hotkey callback %s failed", binding.description)
        finally:
            with self._pressed_lock:
                self._running_bindings.discard(binding.key_combo)


def normalize_combo(key_combo: str) -> str:
    """Canonicalise ``"CTRL + Alt+R"`` to ``"ctrl+alt+r"``."""
    parts = [part.strip().lower() for part in (key_combo or "").split("+") if part.strip()]
    modifiers = sorted(
        (part for part in parts if part.split("_")[0] in _MODIFIER_ORDER),
        key=lambda part: _MODIFIER_ORDER.index(part.split("_")[0]),
    )
    keys: List[str] = [part for part in parts if part not in modifiers]
    return "+".join([*modifiers, *keys])
