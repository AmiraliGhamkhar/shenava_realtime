"""Clipboard access with proper Unicode (CF_UNICODETEXT) support.

On Windows the clipboard is driven directly through ``ctypes`` — no pywin32 and
no pyperclip needed — which is what makes Persian text paste correctly.  On
other platforms pyperclip is used when it happens to be installed.

``preserved()`` snapshots the current text and restores it after a paste, so
dictating does not destroy whatever the user had copied.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


class _Win32Clipboard:
    """Minimal CF_UNICODETEXT reader/writer using ctypes."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32

        # Explicit prototypes: handles are pointers and must not be truncated.
        self._kernel32.GlobalAlloc.restype = ctypes.c_void_p
        self._kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        self._kernel32.GlobalLock.restype = ctypes.c_void_p
        self._kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        self._kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        self._kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
        self._user32.SetClipboardData.restype = ctypes.c_void_p
        self._user32.SetClipboardData.argtypes = [wintypes.UINT, ctypes.c_void_p]
        self._user32.GetClipboardData.restype = ctypes.c_void_p
        self._user32.GetClipboardData.argtypes = [wintypes.UINT]
        self._user32.OpenClipboard.argtypes = [wintypes.HWND]

    @contextmanager
    def _open(self) -> Iterator[None]:
        if not self._user32.OpenClipboard(None):
            raise OSError(f"OpenClipboard failed (error {self._kernel32.GetLastError()})")
        try:
            yield
        finally:
            self._user32.CloseClipboard()

    def get(self) -> str:
        with self._open():
            handle = self._user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return ""
            pointer = self._kernel32.GlobalLock(handle)
            if not pointer:
                return ""
            try:
                return self._ctypes.c_wchar_p(pointer).value or ""
            finally:
                self._kernel32.GlobalUnlock(handle)

    def set(self, text: str) -> None:
        payload = text.encode("utf-16-le") + b"\x00\x00"
        with self._open():
            self._user32.EmptyClipboard()
            handle = self._kernel32.GlobalAlloc(GMEM_MOVEABLE, len(payload))
            if not handle:
                raise OSError("GlobalAlloc failed")
            pointer = self._kernel32.GlobalLock(handle)
            if not pointer:
                self._kernel32.GlobalFree(handle)
                raise OSError("GlobalLock failed")
            try:
                self._ctypes.memmove(pointer, payload, len(payload))
            finally:
                self._kernel32.GlobalUnlock(handle)
            if not self._user32.SetClipboardData(CF_UNICODETEXT, handle):
                # Ownership was not transferred, so the block is still ours.
                self._kernel32.GlobalFree(handle)
                raise OSError("SetClipboardData failed")


class ClipboardManager:
    """Cross-platform clipboard facade used by the text injector."""

    def __init__(self) -> None:
        self._backend = None
        self._kind = "none"
        if sys.platform == "win32":
            try:
                self._backend = _Win32Clipboard()
                self._kind = "win32"
            except Exception:
                logger.exception("Windows clipboard unavailable")
        if self._backend is None:
            try:
                import pyperclip  # type: ignore

                self._backend = pyperclip
                self._kind = "pyperclip"
            except ImportError:
                logger.warning("no clipboard backend available (install pyperclip on non-Windows)")

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def backend(self) -> str:
        return self._kind

    def get(self) -> Optional[str]:
        if not self.available:
            return None
        try:
            if self._kind == "win32":
                return self._backend.get()
            return self._backend.paste()
        except Exception:
            logger.exception("could not read the clipboard")
            return None

    def set(self, text: str) -> bool:
        if not self.available:
            return False
        try:
            if self._kind == "win32":
                self._backend.set(text)
            else:
                self._backend.copy(text)
            return True
        except Exception:
            logger.exception("could not write to the clipboard")
            return False

    def restore(self, previous: Optional[str]) -> bool:
        """Put previously saved text back (used after a clipboard paste)."""
        if previous is None:
            return False
        time.sleep(0.05)  # give the paste a chance to read the clipboard first
        return self.set(previous)
