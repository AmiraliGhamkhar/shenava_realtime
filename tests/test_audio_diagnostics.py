"""Lightweight segment diagnostics: deterministic levels, metrics only."""
import numpy as np
import pytest

from shenava_realtime.audio_diagnostics import (
    SegmentDiagnostics,
    analyze_segment,
    crest_factor,
    format_summary,
    to_dict,
)


def test_empty_and_silent_segments():
    empty = analyze_segment(np.zeros(0, dtype=np.float32))
    assert empty.level == "silent" and empty.rms == 0.0 and empty.peak == 0.0
    silent = analyze_segment(np.zeros(1600, dtype=np.float32))
    assert silent.level == "silent"
    assert to_dict(empty)["level"] == "silent"
    assert crest_factor(empty) == 0.0


def test_normal_level_is_ok():
    rng = np.random.default_rng(0)
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000.0) + 0.01 * rng.standard_normal(16000)
    d = analyze_segment(tone.astype(np.float32))
    assert d.level == "ok"
    assert 0.0 < d.rms < 0.5 and d.peak >= d.rms
    assert d.clipping_ratio == 0.0
    assert "level=ok" in format_summary(d)


def test_low_volume_is_flagged():
    tone = 0.005 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000.0)
    d = analyze_segment(tone.astype(np.float32))
    assert d.level == "low"


def test_clipping_requires_both_peak_and_ratio():
    # Peak at full scale but a clean transient (ratio below the limit): ok.
    clip_transient = np.zeros(16000, dtype=np.float32)
    clip_transient[8000:8064] = 0.999
    assert analyze_segment(clip_transient).level == "ok"

    # Sustained full-scale: clipped.
    clipped = np.full(16000, 0.997, dtype=np.float32)
    d = analyze_segment(clipped)
    assert d.level == "clipped" and d.clipping_ratio > 0.005
    assert "level=clipped" in format_summary(d)


def test_diagnostics_are_deterministic():
    rng = np.random.default_rng(3)
    audio = (0.2 * rng.standard_normal(24000)).astype(np.float32)
    assert analyze_segment(audio) == analyze_segment(audio.copy())


def test_negative_and_overscaled_audio_handled():
    audio = np.linspace(-1.2, 1.2, 8000).astype(np.float32)
    d = analyze_segment(audio)
    assert d.peak >= 1.0
    assert d.level == "clipped"
    assert isinstance(d, SegmentDiagnostics)
