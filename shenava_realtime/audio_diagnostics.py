"""Lightweight audio segment diagnostics (metrics and logging only).

No denoising, no spectral subtraction, no rewriting: these numbers describe
the recorded segment so a quiet or clipping microphone is visible in the log
and in ``AudioCapture.get_statistics()``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

# A segment whose peak sits at full scale on more than this fraction of its
# samples is treating the recorder as the loudest thing in the room.
CLIP_THRESHOLD = 0.995
CLIP_RATIO_LIMIT = 0.005
SILENT_RMS = 1e-4


@dataclass(frozen=True)
class SegmentDiagnostics:
    rms: float
    peak: float
    clipping_ratio: float
    level: str  # "silent" | "low" | "ok" | "clipped"


def analyze_segment(
    audio: np.ndarray, *, full_scale: float = 1.0, low_rms: float = 0.01
) -> SegmentDiagnostics:
    """Deterministic level metrics for one float32 mono segment."""
    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        return SegmentDiagnostics(0.0, 0.0, 0.0, "silent")
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    clipping = float(np.mean(np.abs(samples) >= full_scale * CLIP_THRESHOLD))
    if rms < SILENT_RMS:
        level = "silent"
    elif peak >= full_scale * CLIP_THRESHOLD and clipping > CLIP_RATIO_LIMIT:
        level = "clipped"
    elif rms < low_rms:
        level = "low"
    else:
        level = "ok"
    return SegmentDiagnostics(round(rms, 6), round(peak, 6), round(clipping, 6), level)


def format_summary(diagnostics: SegmentDiagnostics) -> str:
    return (
        f"rms={diagnostics.rms:.3f} peak={diagnostics.peak:.3f} "
        f"clip={diagnostics.clipping_ratio * 100:.2f}% level={diagnostics.level}"
    )


def to_dict(diagnostics: SegmentDiagnostics) -> dict:
    return asdict(diagnostics)


def crest_factor(diagnostics: SegmentDiagnostics) -> float:
    """Peak/RMS ratio (0.0 for silent segments); a steady tone is ~1.0."""
    if diagnostics.rms < SILENT_RMS:
        return 0.0
    ratio = diagnostics.peak / diagnostics.rms
    return ratio if math.isfinite(ratio) else 0.0
