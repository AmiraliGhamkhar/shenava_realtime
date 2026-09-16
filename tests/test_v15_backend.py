"""Small capability/config regressions for the v1.5 migration."""
import types
import numpy as np
import pytest

from shenava_realtime.asr_backend import NeMoASR
from shenava_realtime.config import ASRConfig, DEFAULT_MODEL_NAME
from shenava_realtime.second_pass import CTCBeamDecoder
from shenava_realtime.vad import EnergyVAD, VADConfig


def test_v15_ctc_is_the_default_and_rnnt_is_selectable():
    assert DEFAULT_MODEL_NAME == "Reza2kn/Shenava-Koochik-v1.5"
    assert ASRConfig().decoder_type == "ctc"
    assert ASRConfig(decoder_type="rnnt").decoder_type == "rnnt"
    assert ASRConfig(decoder_type="auto").decoder_type == "auto"


def test_rnnt_streaming_is_not_claimed_for_a_ctc_only_model():
    model = types.SimpleNamespace(conformer_stream_step=lambda: None)
    assert NeMoASR._check_rnnt_streaming(model) is False
    model.rnnt_stream_step = lambda *args, **kwargs: None
    assert NeMoASR._check_rnnt_streaming(model) is True


def test_beam_size_is_the_actual_pruning_bound():
    decoder = CTCBeamDecoder(beam_size=2)
    states = {(tuple([i]), i): float(-i) for i in range(10)}
    pruned = decoder._prune(states)
    assert len({seq for seq, _last in pruned}) == 2


def test_adaptive_vad_has_bounded_quiet_floor_and_hysteresis():
    vad = EnergyVAD(VADConfig(adaptive=True, min_speech_ms=50, min_silence_ms=100))
    for _ in range(20):
        vad.process(np.full(160, 0.002, dtype=np.float32))
    assert 0 < vad.noise_floor <= vad.config.onset_rms
    assert vad.config.offset_rms < vad.config.onset_rms


def test_invalid_decoder_fails_at_configuration_boundary():
    with pytest.raises(ValueError, match="decoder_type"):
        ASRConfig(decoder_type="beam")
