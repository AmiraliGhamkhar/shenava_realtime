"""Regression tests for medical safety boundaries, not recognition accuracy."""
import json
import sqlite3
import numpy as np
import pytest

from shenava_realtime.audio_capture import AudioCapture
from shenava_realtime.clinical import ClinicalWorker, DictionaryNER, extract_record
from shenava_realtime.config import AudioConfig, PostProcessConfig
from shenava_realtime.fst import TrieFST
from shenava_realtime.postprocessor import PostProcessor
from shenava_realtime.vad import EnergyVAD, EventType, VADConfig
from tests.fakes import make_pipeline


@pytest.mark.parametrize('raw,expected', [
    ('دو و نیم میلی گرم', '2.5 mg'),
    ('یک ممیز پنج میلی گرم', '1.5 mg'),
    ('سه ممیز صفر پنج میلی گرم', '3.05 mg'),
    ('منفی پنج درجه سانتیگراد', '-5°C'),
    ('نه درد دارد نه تب', 'نه درد دارد نه تب'),
    ('سی و پنج درصد،', '35%،'),
    ('کابج.', 'CABG.'),
    ('(سی ای بی جی)،', '(CABG)،'),
    ('5.25 mg', '5.25 mg'),
    ('۵٫۲۵ میلی گرم', '5.25 mg'),
    ('پنج میلی گرم در دسی لیتر', '5 mg/dL'),
    ('cabg و HbA1c', 'CABG و HbA1c'),
    ('دو سه', '2 3'),
    ('از صد به هشتاد', 'از 100 به 80'),
])
def test_medical_text_safety(raw, expected):
    processor = PostProcessor()
    assert processor.process(raw) == expected
    assert processor.process(expected) == expected


def test_fst_longest_and_punctuation_boundary():
    fst = TrieFST()
    fst.add_many({'سی': '30', 'سی تی': 'CT'})
    assert fst.apply('سی تی،') == 'CT،'
    assert fst.apply('سی، تی') == '30، تی'
    with pytest.raises(ValueError):
        fst.add('سی', 'different')


def test_endpoint_prevents_numeric_and_dictionary_prefix_injection():
    pipeline, _ = make_pipeline(['دوز سی', 'دوز سی و', 'دوز سی و پنج',
                                'دوز سی و پنج میلی', 'دوز سی و پنج میلی گرم'])
    pipeline.start_utterance()
    for _ in range(5):
        assert pipeline.push_audio(np.zeros(8000, np.float32)) == []
    assert pipeline.end_utterance() == ['دوز 35 mg']


def test_final_hypothesis_can_rewrite_even_old_prefix_before_endpoint():
    pipeline, _ = make_pipeline(['درد دارد امروز', 'درد دارد امروز', 'درد ندارد امروز'], holdback_words=0)
    pipeline.start_utterance()
    for _ in range(3):
        assert not pipeline.push_audio(np.zeros(8000, np.float32))
    assert pipeline.end_utterance() == ['درد ندارد امروز']


def test_vad_pending_is_bounded_and_flush_does_not_emit_orphan_end():
    vad = EnergyVAD(VADConfig(min_speech_ms=250, max_speech_s=.3))
    vad.process(np.full(1000, .1, np.float32))
    assert vad.flush() == []
    for _ in range(100):
        vad.process(np.zeros(1000, np.float32))
        assert vad._segment_samples <= 5800


def test_max_segment_restart_preserves_all_audio_once():
    capture = AudioCapture(AudioConfig(vad_min_speech_ms=64, vad_max_speech_s=.3, vad_pre_speech_ms=0))
    received, starts, ends = [], [], []
    active = [False]
    def start(audio):
        starts.append(True)
        active[0] = True
        received.extend(audio)
    def audio(chunk, speech):
        if speech and active[0]:
            received.extend(chunk)
    def end(duration):
        ends.append(duration)
        active[0] = False
    capture.on_speech_start, capture.on_audio, capture.on_speech_end = start, audio, end
    samples = np.linspace(.05, .1, 16000, dtype=np.float32)
    for offset in range(0, len(samples), 1024):
        capture._consume(samples[offset:offset+1024])
    for event in capture.vad.flush():
        capture._dispatch(event)
    assert len(starts) == len(ends) >= 3
    np.testing.assert_array_equal(received, samples)


def test_audio_callback_owns_input_memory():
    capture = AudioCapture()
    block = np.full((1024, 1), .1, np.float32)
    capture._audio_callback(block, 1024, None, None)
    block[:] = 0
    assert capture._queue.get().mean() == pytest.approx(.1)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_audio_and_config_rejected(bad):
    with pytest.raises(ValueError):
        VADConfig(onset_rms=bad)
    with pytest.raises(ValueError):
        EnergyVAD().process(np.array([bad], np.float32))


def test_ner_rules_offsets_and_no_guessed_dose_links():
    text = 'بیمار تب ندارد. BP 120/80 و متفورمین 2.5 mg'
    record = extract_record(text)
    assert record['review_required'] is True
    for entity in record['entities']:
        assert text[entity['start']:entity['end']] == entity['text']
    assert next(e for e in record['entities'] if e['text'] == 'تب')['assertion'] == 'negated'
    assert {'value': '2.5', 'unit': 'mg'} .items() <= record['measurements'][0].items()
    assert 'dose' not in record
    assert DictionaryNER({'درد قفسه سینه': 'symptom'}).extract('درد، قفسه سینه') == []


def test_clinical_worker_stores_json_sqlite_and_drains(tmp_path):
    db, output = tmp_path/'clinical.sqlite', tmp_path/'clinical.jsonl'
    worker = ClinicalWorker(str(db), str(output))
    worker.start()
    assert worker.submit('تب ندارد. 5 mg')
    worker.stop()
    assert not worker.thread.is_alive()
    assert worker.error is None
    with sqlite3.connect(db) as conn:
        payload, = conn.execute('SELECT record_json FROM utterances').fetchone()
    assert json.loads(payload) == json.loads(output.read_text())


def test_clinical_startup_failure_is_explicit(tmp_path):
    worker = ClinicalWorker(str(tmp_path/'missing'/'file.sqlite'))
    with pytest.raises(RuntimeError):
        worker.start()
    worker.stop()


def test_asr_overrun_aborts_instead_of_continuing_cached_state():
    from shenava_realtime.config import AppConfig
    from shenava_realtime.realtime_engine import RealtimeASR
    from tests.fakes import FakeBackend, FakeAudioCapture
    config = AppConfig()
    config.audio.queue_max_chunks = 2
    engine = RealtimeASR(config, backend=FakeBackend(), audio_capture=FakeAudioCapture())
    engine.pipeline.start_utterance()
    engine._enqueue(('audio', np.zeros(1000, np.float32)))
    engine._enqueue(('audio', np.zeros(1000, np.float32)))
    engine._enqueue(('end', 1.0))
    assert engine.stats['overruns'] == 1
    while not engine._queue.empty():
        engine._handle(engine._queue.get_nowait())
    assert not engine.pipeline.is_active
    assert engine.transcript == ''
    engine._handle(('audio', np.zeros(1000, np.float32)))
    assert not engine.pipeline.is_active
    engine._handle(('start', np.zeros(0, np.float32)))
    assert engine.pipeline.is_active


def test_repeated_legitimate_utterances_are_not_deduplicated_by_default():
    from shenava_realtime.config import InjectorConfig, InjectorMode
    from shenava_realtime.injector.text_injector import TextInjector, InjectionResult
    from tests.test_text_injector import FakeKeyboard, FakeClipboard
    injector = TextInjector(InjectorConfig(mode=InjectorMode.KEYBOARD), FakeClipboard(), FakeKeyboard())
    assert injector.inject_now('5 mg') is InjectionResult.SUCCESS
    assert injector.inject_now('5 mg') is InjectionResult.SUCCESS


def test_capture_overrun_discards_gap_and_preserves_shutdown_marker():
    capture = AudioCapture()
    aborted = []
    capture.on_discontinuity = lambda: aborted.append(True)
    capture._queue.put(np.zeros(100, np.float32))
    capture._queue.put(None)
    capture._discontinuity.set()
    capture._consume(np.full(1024, .1, np.float32))
    assert aborted == [True]
    assert capture._queue.get_nowait() is None
    assert not capture.vad.in_speech
