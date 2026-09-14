"""Floating overlay that shows the live transcript.

Tkinter is single threaded, so every mutation coming from the ASR/injector
threads is pushed onto ``ui_queue`` and drained by a periodic ``after`` callback
on the Tk thread.  The window is created *and* destroyed on that same thread,
which is what makes start/stop reliable (calling ``destroy()`` from another
thread is undefined behaviour in Tk).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from ..config import OverlayConfig, OverlayPosition

logger = logging.getLogger(__name__)

FALLBACK_FONTS = ("Vazirmatn", "Segoe UI", "Tahoma", "Arial", "Helvetica")
_STATUS_COLOR_ACTIVE = "#4CAF50"
_STATUS_COLOR_IDLE = "#888888"
_STATUS_COLOR_RECORDING = "#4A9EFF"


class OverlayWindow:
    """The Tk window itself. Only ever touched from the Tk thread."""

    def __init__(self, config: OverlayConfig, ui_queue: "queue.Queue[Tuple[str, tuple]]") -> None:
        import tkinter as tk
        from tkinter import font as tkfont

        self._tk = tk
        self.config = config
        self.ui_queue = ui_queue
        self.is_visible = False
        self.last_text_time = 0.0
        self._auto_hide_id: Optional[str] = None
        self._drag_offset: Tuple[int, int] = (0, 0)

        self.root = tk.Tk()
        self.root.title("Shenava ASR")
        self.root.configure(bg=config.background_color)
        if config.always_on_top:
            self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", float(config.opacity))
        except Exception:
            logger.warning("window transparency is not supported here")
        self.root.overrideredirect(True)
        self.root.geometry(self._geometry_for(config.position))

        self._font_family = self._pick_font(tkfont)
        self._text_var = tk.StringVar(value="")
        self.label = tk.Label(
            self.root,
            textvariable=self._text_var,
            font=(self._font_family, config.font_size),
            fg=config.text_color,
            bg=config.background_color,
            wraplength=max(80, config.width - 24),
            justify="center",
            anchor="center",
        )
        self.label.pack(fill="both", expand=True, padx=12, pady=6)

        self.status = tk.Label(
            self.root,
            text="●",
            font=(self._font_family, 9),
            fg=_STATUS_COLOR_RECORDING,
            bg=config.background_color,
        )
        self.status.place(relx=1.0, rely=0.0, anchor="ne", x=-6, y=2)

        self.root.bind("<Button-1>", self._on_press)
        self.root.bind("<B1-Motion>", self._on_drag)
        self.root.bind("<Escape>", lambda _event: self.hide())

        self.show()
        self.root.after(50, self._pump)

    # ------------------------------------------------------------------ #
    def _pick_font(self, tkfont) -> str:
        try:
            available = set(tkfont.families())
        except Exception:
            logger.debug("could not enumerate fonts")
            return FALLBACK_FONTS[-1]
        for family in (self.config.font_family, *FALLBACK_FONTS):
            if family in available:
                return family
        return "TkDefaultFont"

    def _geometry_for(self, position: OverlayPosition) -> str:
        width, height = self.config.width, self.config.height
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        margin = 24
        bottom_y = screen_h - height - 90
        positions: Dict[OverlayPosition, Tuple[int, int]] = {
            OverlayPosition.TOP: ((screen_w - width) // 2, 50),
            OverlayPosition.BOTTOM: ((screen_w - width) // 2, bottom_y),
            OverlayPosition.TOP_LEFT: (margin, 50),
            OverlayPosition.TOP_RIGHT: (screen_w - width - margin, 50),
            OverlayPosition.BOTTOM_LEFT: (margin, bottom_y),
            OverlayPosition.BOTTOM_RIGHT: (screen_w - width - margin, bottom_y),
            OverlayPosition.CENTER: ((screen_w - width) // 2, (screen_h - height) // 2),
            OverlayPosition.CUSTOM: tuple(self.config.custom_position),  # type: ignore[assignment]
        }
        x, y = positions.get(position, positions[OverlayPosition.BOTTOM])
        return f"{width}x{height}+{max(0, int(x))}+{max(0, int(y))}"

    # ------------------------------------------------------------------ #
    def show(self) -> None:
        if self.is_visible:
            return
        self.is_visible = True
        self.root.deiconify()
        self.root.lift()
        self._schedule_auto_hide()

    def hide(self) -> None:
        if not self.is_visible:
            return
        self.is_visible = False
        if self._auto_hide_id is not None:
            try:
                self.root.after_cancel(self._auto_hide_id)
            except Exception:
                logger.debug("could not cancel the auto-hide timer")
            self._auto_hide_id = None
        self.root.withdraw()

    def toggle(self) -> None:
        self.hide() if self.is_visible else self.show()

    def update_text(self, text: str, confidence: float = 0.0) -> None:
        limit = max(40, self.config.max_chars)
        if len(text) > limit:
            text = "…" + text[-limit:]
        self._text_var.set(text)
        self.last_text_time = time.time()
        self.status.config(fg=_STATUS_COLOR_ACTIVE if text else _STATUS_COLOR_IDLE)
        # Legacy confidence argument is unknown, not a calibrated probability.
        self._schedule_auto_hide()
        if not self.is_visible:
            self.show()

    def update_partial(self, text: str) -> None:
        if not self.config.show_partial or not text:
            return
        self.update_text(text, 0.0)

    def set_recording_status(self, is_recording: bool) -> None:
        self.status.config(
            text="●" if is_recording else "○",
            fg=_STATUS_COLOR_RECORDING if is_recording else _STATUS_COLOR_IDLE,
        )

    def set_position(self, x: int, y: int) -> None:
        self.root.geometry(f"+{int(x)}+{int(y)}")

    # ------------------------------------------------------------------ #
    def _on_press(self, event) -> None:
        self._drag_offset = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def _on_drag(self, event) -> None:
        self.root.geometry(f"+{event.x_root - self._drag_offset[0]}+{event.y_root - self._drag_offset[1]}")

    def _schedule_auto_hide(self) -> None:
        if self.config.auto_hide_delay <= 0:
            return
        if self._auto_hide_id is not None:
            try:
                self.root.after_cancel(self._auto_hide_id)
            except Exception:
                logger.debug("could not cancel the auto-hide timer")
        self._auto_hide_id = self.root.after(int(self.config.auto_hide_delay * 1000), self.hide)

    def _pump(self) -> None:
        """Drain cross-thread updates and reschedule (runs on the Tk thread)."""
        try:
            while True:
                method, args = self.ui_queue.get_nowait()
                target: Optional[Callable[..., Any]] = getattr(self, method, None)
                if callable(target):
                    try:
                        target(*args)
                    except Exception:
                        logger.exception("overlay update %s failed", method)
                else:
                    logger.warning("unknown overlay command %r", method)
        except queue.Empty:
            pass
        self.root.after(50, self._pump)

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.root.mainloop()

    def shutdown(self) -> None:
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            logger.exception("error while destroying the overlay window")
        finally:
            self.is_visible = False


class OverlayManager:
    """Owns the overlay thread and offers thread-safe update methods."""

    def __init__(self, config: Optional[OverlayConfig] = None) -> None:
        self.config = config or OverlayConfig()
        self.ui_queue: "queue.Queue[Tuple[str, tuple]]" = queue.Queue(maxsize=64)
        self._thread: Optional[threading.Thread] = None
        self._window: Optional[OverlayWindow] = None
        self._ready = threading.Event()
        self._failed = False
        self._last_recording: Optional[bool] = None

    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, wait_s: float = 2.0) -> bool:
        if self.is_running:
            return True
        self._ready.clear()
        self._failed = False
        self._thread = threading.Thread(target=self._run, name="overlay", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=wait_s):
            logger.warning("overlay window did not come up within %.1fs", wait_s)
            return False
        if self._failed:
            return False
        logger.info("overlay window started")
        return True

    def _run(self) -> None:
        try:
            window = OverlayWindow(self.config, self.ui_queue)
        except Exception:
            self._failed = True
            self._ready.set()
            logger.exception("could not create the overlay window (is a display available?)")
            return
        self._window = window
        self._ready.set()
        try:
            window.run()
        except Exception:
            logger.exception("overlay main loop crashed")
        finally:
            self._window = None

    def stop(self) -> None:
        thread = self._thread
        window = self._window
        if window is not None:
            # Destroy on the Tk thread: quit() wakes the mainloop, which returns
            # and lets _run() finish on its own thread.
            self._post("shutdown")
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.warning("overlay thread did not exit within 3s; restart blocked")
            else:
                self._thread = None
                self._window = None
            logger.info("overlay window stopped")

    # ------------------------------------------------------------------ #
    def _post(self, method: str, *args: Any) -> None:
        try:
            self.ui_queue.put_nowait((method, args))
        except queue.Full:
            try:
                self.ui_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.ui_queue.put_nowait((method, args))
            except queue.Full:
                pass

    def update_text(self, text: str, confidence: float = 0.0) -> None:
        self._post("update_text", text, confidence)

    def update_partial(self, text: str) -> None:
        self._post("update_partial", text)

    def show(self) -> None:
        self._post("show")

    def hide(self) -> None:
        self._post("hide")

    def toggle(self) -> None:
        self._post("toggle")

    def set_recording_status(self, is_recording: bool) -> None:
        if is_recording == self._last_recording:
            return
        self._last_recording = is_recording
        self._post("set_recording_status", is_recording)
