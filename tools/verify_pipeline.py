"""Headless streaming replay: --self-test (synthetic backend), or --model + --wav.

Real WAV replay uses the real NeMo backend and VAD, at microphone speed. No
microphone, desktop, downloads, or test-suite imports are needed. Synthetic mode
verifies orchestration only and must not be reported as model accuracy testing.
"""
from __future__ import annotations
import argparse
import json
import sys
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from shenava_realtime.audio_capture import AudioCapture
from shenava_realtime.clinical import extract_record
from shenava_realtime.config import AppConfig, OutputMode
from shenava_realtime.realtime_engine import RealtimeASR
from shenava_realtime.utils import setup_logging


class ReplayCapture(AudioCapture):
    def __init__(self, config, blocks):
        super().__init__(config)
        self.blocks = blocks
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.error = None

    def start(self):
        self._thread = threading.Thread(target=self._replay, name='wav-replay', daemon=True)
        self._thread.start()

    def _replay(self):
        try:
            for block in self.blocks:
                if self.cancelled.is_set():
                    break
                self._consume(block)
                self.cancelled.wait(len(block) / self.config.sample_rate)
            for event in self.vad.flush():
                self._dispatch(event)
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()

    def stop(self):
        self.cancelled.set()
        if self._thread:
            self._thread.join(3)


def wav_blocks(path):
    with wave.open(str(path), 'rb') as source:
        if (source.getnchannels(), source.getframerate(), source.getsampwidth()) != (1, 16000, 2):
            raise ValueError('WAV must be PCM16, 16 kHz, mono; resample explicitly before replay')
        while data := source.readframes(1024):
            yield np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768


class SyntheticBackend:
    """A clearly labelled test double, not speech recognition."""
    def create_stream(self):
        return SyntheticStream()


class SyntheticStream:
    def reset(self):
        self.samples = 0
    def push(self, audio):
        self.samples += len(audio)
        return 'بیمار دوز دو و نیم میلی گرم', 0.0
    def finalize(self):
        return 'بیمار دوز دو و نیم میلی گرم', 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--wav', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--right-context', type=int, choices=[0, 1, 6, 13], default=13)
    parser.add_argument('--allow-endpoint', action='store_true')
    args = parser.parse_args(argv)
    if not args.self_test and not (args.model and args.wav):
        parser.error('Supply --self-test or both --model and --wav')
    config = AppConfig()
    config.output_mode = OutputMode.CONSOLE
    config.save_transcripts = False
    config.asr.model_path = args.model
    config.asr.device = args.device
    config.asr.right_context = args.right_context
    config.asr.require_streaming = not args.allow_endpoint
    setup_logging('INFO')
    backend = None
    if args.self_test:
        print('SYNTHETIC SELF-TEST: no real ASR model or microphone is exercised.', file=sys.stderr)
        backend = SyntheticBackend()
        samples = np.concatenate((np.full(16000, .05, np.float32), np.zeros(16000, np.float32)))
        blocks = (samples[i:i+1024] for i in range(0, len(samples), 1024))
    else:
        blocks = wav_blocks(args.wav)
    capture = ReplayCapture(config.audio, blocks)
    engine = RealtimeASR(config, backend=backend, audio_capture=capture)
    records, deltas = [], []
    engine.on_utterance_end = lambda text, _: records.append(extract_record(text))
    engine.on_text_delta = lambda text, _: deltas.append(text)
    try:
        engine.start()
        while not capture.done.wait(.2):
            if not engine.is_running:
                raise RuntimeError('ASR worker exited during replay')
    finally:
        engine.stop()
    if capture.error:
        raise capture.error
    if engine.is_running or engine.stats.get('errors') or engine.stats.get('overruns'):
        raise RuntimeError('Replay failed; inspect errors/overruns and shutdown status')
    if not records:
        raise RuntimeError('No speech decoded: check input audio, VAD thresholds and checkpoint')
    assert ''.join(deltas) == ' '.join(record['text'] for record in records)
    if args.self_test:
        assert [r['text'] for r in records] == ['بیمار دوز 2.5 mg']
    print(json.dumps({'mode': 'synthetic' if args.self_test else 'real-model',
                      'statistics': engine.get_statistics(), 'records': records}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
