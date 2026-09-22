"""Small shared helpers: logging setup, transcript buffer, performance stats."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root logging once (console, plus an optional file)."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(stream)

    if log_file:
        try:
            file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
            root.addHandler(file_handler)
        except OSError:
            logging.getLogger(__name__).warning("could not open log file %s", log_file)

    # Quiet down very chatty third-party loggers.
    for noisy in ("numba", "lightning.pytorch", "nemo", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class TextBuffer:
    """Thread-safe accumulator for finished utterances."""

    def __init__(self, max_chars: int = 20000) -> None:
        self.max_chars = max_chars
        self._text = ""
        self._segments: Deque[str] = deque(maxlen=200)
        self._lock = threading.Lock()

    def add_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._text = f"{self._text} {text}".strip()
            if len(self._text) > self.max_chars:
                self._text = self._text[-self.max_chars :]
            self._segments.append(text)

    def get_text(self) -> str:
        with self._lock:
            return self._text

    def get_last_n(self, count: int) -> List[str]:
        with self._lock:
            return list(self._segments)[-count:]

    def clear(self) -> None:
        with self._lock:
            self._text = ""
            self._segments.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._text)


class PerformanceMonitor:
    """Rolling window of timing samples (no external dependencies)."""

    def __init__(self, window: int = 100) -> None:
        self._samples: Dict[str, Deque[float]] = {
            "decode": deque(maxlen=window),
            "rtf": deque(maxlen=window),
        }
        self._lock = threading.Lock()

    def update_decode_time(self, seconds: float) -> None:
        with self._lock:
            self._samples["decode"].append(float(seconds))

    def update_rtf(self, rtf: float) -> None:
        with self._lock:
            self._samples["rtf"].append(float(rtf))

    def get_statistics(self) -> Dict[str, float]:
        # Sorting a ≤100-sample window is trivial; the point of this change is
        # that the lock is held only to *copy* the deques, so the sort (and any
        # future heavier percentile work) never blocks the producer thread.
        with self._lock:
            snapshot = {name: list(values) for name, values in self._samples.items()}
        stats: Dict[str, float] = {}
        for name, values in snapshot.items():
            if not values:
                continue
            ordered = sorted(values)
            stats[f"{name}_avg"] = sum(values) / len(values)
            stats[f"{name}_min"] = ordered[0]
            stats[f"{name}_max"] = ordered[-1]
            stats[f"{name}_p90"] = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
        return stats
