#!/usr/bin/env python3
"""Shenava Real-time ASR — application entry point.

Pipeline::

    microphone -> audio capture -> adaptive RMS VAD -> bounded queue
              -> Shenava-Koochik-v1.0 streaming CTC decoder (sherpa-onnx, CPU)
              -> transcript stabilization -> endpoint second pass
              -> FST post-processing -> stable deltas -> clipboard / keyboard

Run ``python main.py --help`` for the command line switches; every option is
also available as a ``SHENAVA_*`` environment variable or in ``config.json``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from importlib import import_module
from pathlib import Path
from typing import Optional

# Make the package importable when the file is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):  # Persian text must survive Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from shenava_realtime.audio_capture import AudioCapture
from shenava_realtime.config import AppConfig, ConfigManager, OutputMode
from shenava_realtime.hotkeys.hotkey_manager import HotkeyManager
from shenava_realtime.injector.clipboard_manager import ClipboardManager
from shenava_realtime.injector.text_injector import TextInjector
from shenava_realtime.overlay.overlay_window import OverlayManager
from shenava_realtime.realtime_engine import RealtimeASR
from shenava_realtime.utils import PerformanceMonitor, setup_logging

logger = logging.getLogger("shenava.app")


def _sherpa_onnx_version() -> str:
    try:
        module = import_module("sherpa_onnx")
    except Exception:
        return "not available"
    return str(getattr(module, "__version__", "unknown"))


class ShenavaApp:
    """Wires the components together and owns the application lifecycle."""

    def __init__(self, config: Optional[AppConfig] = None, *, backend=None, audio_capture=None) -> None:
        self.config = config or AppConfig.from_env()
        if not self.config.asr.commit_on_endpoint and (
            self.config.injector.enabled
            or self.config.clinical_sqlite
            or self.config.clinical_jsonl
        ):
            # Early commit cannot retract an already injected prefix; refuse
            # the unsafe combination at startup instead of trusting a warning.
            raise ValueError(
                "commit_on_endpoint=false (early commit) cannot retract an "
                "injected prefix and must not be combined with text injection "
                "or clinical persistence. Enable endpoint commit (the default) "
                "or disable the injector and --clinical-sqlite/--clinical-jsonl."
            )
        setup_logging(self.config.log_level, log_file=Path("app.log"))

        self.clinical = None
        if self.config.clinical_sqlite or self.config.clinical_jsonl:
            from shenava_realtime.clinical import ClinicalWorker
            self.clinical = ClinicalWorker(self.config.clinical_sqlite, self.config.clinical_jsonl)
        self.performance = PerformanceMonitor()
        self.clipboard = ClipboardManager()
        self._shutdown = threading.Event()
        self._stopping = threading.Lock()

        logger.info("initialising Shenava Real-time ASR (output mode: %s)", self.config.output_mode.value)
        self.asr = RealtimeASR(self.config, backend=backend, audio_capture=audio_capture)
        self.overlay = OverlayManager(self.config.overlay) if self.config.overlay.enabled else None
        self.injector = TextInjector(self.config.injector) if self.config.injector.enabled else None
        self.hotkeys = HotkeyManager(self.config.hotkeys)

        self.asr.on_partial = self._on_partial
        self.asr.on_text_delta = self._on_text_delta
        self.asr.on_utterance_end = self._on_utterance_end
        if self.injector is not None:
            self.injector.on_failed = lambda text: logger.warning("injection failed (%d characters)", len(text))

        self._setup_hotkeys()

    # ------------------------------------------------------------------ #
    def _setup_hotkeys(self) -> None:
        self.hotkeys.setup_default_hotkeys(
            toggle_recording=self.toggle_recording,
            toggle_overlay=self.toggle_overlay,
            toggle_injector=self.toggle_injector,
            clear_transcript=self.clear_transcript,
            emergency_stop=self.request_shutdown,
        )
        for combo, description in self.hotkeys.get_bindings().items():
            logger.info("hotkey %-14s %s", combo, description)
        if not self.hotkeys.available:
            logger.warning("hotkeys are unavailable (pynput missing); use Ctrl+C to stop")

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        try:
            if self.clinical:
                self.clinical.start()
            self.hotkeys.start()
            if self.overlay is not None and not self.overlay.start():
                logger.error("overlay unavailable; continuing without it")
                self.overlay = None
            if self.injector is not None:
                self.injector.start()
            self.asr.start()
            if self.overlay is not None:
                self.overlay.set_recording_status(True)
        except Exception:
            logger.exception("startup failed")
            self.stop()
            raise

        self._log_startup_diagnostics()
        logger.info("listening — speak now (Ctrl+Alt+R toggles, Ctrl+Alt+Q quits)")
        try:
            while not self._shutdown.is_set():
                self._shutdown.wait(timeout=0.5)
                self._track_performance()
        except KeyboardInterrupt:
            logger.info("interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        if not self._stopping.acquire(blocking=False):
            return
        try:
            logger.info("stopping…")
            self.asr.stop()
            if self.clinical:
                self.clinical.stop()
            if self.injector is not None:
                self.injector.stop()
            if self.overlay is not None:
                self.overlay.stop()
            self.hotkeys.stop()
            self._print_statistics()
            logger.info("stopped")
        finally:
            self._stopping.release()

    def request_shutdown(self) -> None:
        logger.info("shutdown requested")
        self._shutdown.set()

    def _log_startup_diagnostics(self) -> None:
        """One-block summary of the active runtime path (no transcript content)."""
        asr = self.config.asr
        audio = self.config.audio
        stats = self.asr.get_statistics()
        streaming = "unknown"
        capabilities = getattr(self.asr.backend, "capabilities", None)
        if callable(capabilities):
            try:
                caps = capabilities()
                streaming = "yes" if caps.streaming else "no"
                logger.info("model capabilities — %s", caps.describe())
            except Exception as exc:  # capability reporting must not block startup
                logger.warning("could not report model capabilities: %s", exc)
        logger.info(
            "startup diagnostics: model path=%s tokens path=%s sherpa-onnx=%s "
            "streaming=%s input_device=%s sample_rate=%d vad_onset=%.4f "
            "vad_offset=%.4f adaptive_vad=%s decoder=%s second_pass=%s",
            asr.model_path,
            asr.tokens_path,
            _sherpa_onnx_version(),
            streaming,
            audio.device if audio.device is not None else "default",
            audio.sample_rate,
            audio.vad_onset_rms,
            audio.vad_offset_rms,
            "enabled" if audio.vad_adaptive else "disabled",
            stats["decoder"],
            stats["second_pass"],
        )
        logger.info(
            "backend=%s provider=%s threads=%d feature_dim=%d decoding=%s "
            "require_streaming=%s benchmark=%s output_mode=%s",
            asr.asr_backend,
            asr.device,
            asr.num_threads,
            asr.feature_dim,
            asr.decoding_method,
            asr.require_streaming,
            asr.benchmark_mode,
            self.config.output_mode.value,
        )

    # ------------------------------------------------------------------ #
    def _track_performance(self) -> None:
        stats = self.asr.get_statistics()
        audio = float(stats.get("total_audio_duration") or 0.0)
        processing = float(stats.get("total_processing_time") or 0.0)
        if audio > 0:
            self.performance.update_rtf(processing / audio)

    def _print_statistics(self) -> None:
        stats = self.asr.get_statistics()
        logger.info(
            "session: %d utterances, %d decodes, %.1fs audio (head=%s)",
            int(stats["utterances"]),
            int(stats["decodes"]),
            float(stats["total_audio_duration"]),
            stats.get("decoder_head", "ctc"),
        )
        logger.info(
            "robustness: %d queue overflows, %d dropped items (%.1fs audio), "
            "%d discontinuities, %d forced splits, %d decoder errors, "
            "second pass %s",
            int(stats.get("queue_overflows", 0)),
            int(stats.get("dropped_items", 0)),
            float(stats.get("dropped_audio_seconds", 0.0)),
            int(stats.get("discontinuities", 0)),
            int(stats.get("forced_splits", 0)),
            int(stats.get("decoder_errors", 0)),
            stats.get("second_pass_stats", {}),
        )
        if self.injector is not None:
            injector_stats = self.injector.get_statistics()
            logger.info(
                "injection: %s successful, %s failed, %s skipped, %s characters",
                injector_stats["successful"],
                injector_stats["failed"],
                injector_stats["skipped"],
                injector_stats["characters"],
            )
        for name, value in self.performance.get_statistics().items():
            logger.info("perf %s: %.3f", name, value)

    # ------------------------------------------------------------------ #
    # ASR callbacks (worker thread)
    # ------------------------------------------------------------------ #
    def _on_partial(self, text: str, confidence: float) -> None:
        if self.overlay is not None:
            self.overlay.update_partial(text)

    def _on_text_delta(self, delta: str, confidence: float) -> None:
        """Emit one stable delta — never the whole partial transcript."""
        mode = self.config.output_mode
        if mode in (OutputMode.INJECT_ONLY, OutputMode.BOTH) and self.injector is not None:
            self.injector.inject(delta)
        if mode is OutputMode.CONSOLE:
            print(delta, end=" " if delta.endswith(" ") else "", flush=True)
        if self.overlay is not None:
            self.overlay.update_text(self.asr.pipeline.committed_text, confidence)

    def _on_utterance_end(self, text: str, confidence: float) -> None:
        if self.clinical and not self.clinical.submit(
            text, review_reasons=self.asr.last_review_reasons
        ):
            logger.error("Completed utterance was not accepted by clinical output")
        if self.overlay is not None:
            self.overlay.update_text(text, confidence)
        if self.config.output_mode is OutputMode.CLIPBOARD:
            if self.clipboard.set(text):
                logger.debug("utterance copied to the clipboard")
            else:
                logger.warning("could not copy the utterance to the clipboard")
        if self.config.output_mode is OutputMode.CONSOLE:
            print(flush=True)

    # ------------------------------------------------------------------ #
    # Hotkey actions
    # ------------------------------------------------------------------ #
    def toggle_recording(self) -> None:
        if self.asr.audio_capture.is_paused:
            self.asr.audio_capture.resume()
            if self.overlay is not None:
                self.overlay.set_recording_status(True)
            logger.info("recording resumed")
        else:
            self.asr.audio_capture.pause()
            if self.overlay is not None:
                self.overlay.set_recording_status(False)
            logger.info("recording paused")

    def toggle_overlay(self) -> None:
        if self.overlay is None:
            logger.info("overlay is disabled")
            return
        self.overlay.toggle()

    def toggle_injector(self) -> None:
        if self.injector is None:
            logger.info("text injection is disabled")
            return
        self.injector.toggle()

    def clear_transcript(self) -> None:
        self.asr.clear_transcript()
        if self.overlay is not None:
            self.overlay.update_text("", 0.0)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shenava real-time Persian ASR")
    parser.add_argument("--self-test", action="store_true", help="headless synthetic streaming/startup check; not real recognition")
    parser.add_argument("--clinical-sqlite", help="optional SQLite output file")
    parser.add_argument("--clinical-jsonl", help="optional JSONL output file")
    parser.add_argument("--second-pass", choices=["off", "greedy"],
                        help="utterance-end second-pass decoder (natural endpoints only)")
    parser.add_argument("--require-streaming", action="store_true",
                        help="fail at startup if the model does not behave as an online/"
                             "streaming recognizer (default for the desktop app; retained for "
                             "explicit scripts/configs that opted out)")
    parser.add_argument("--config", type=Path, default=None, help="path to a JSON config file")
    parser.add_argument("--model", default=None, help="path to the sherpa-onnx model.int8.onnx")
    parser.add_argument("--tokens", default=None, help="path to the sherpa-onnx tokens.txt")
    parser.add_argument("--device", default=None, help="cpu only (sherpa-onnx backend has no GPU support)")
    parser.add_argument("--no-adaptive-vad", action="store_true",
                        help="use the fixed RMS thresholds instead of the adaptive noise floor")
    parser.add_argument("--threads", type=int, default=None, help="sherpa-onnx CPU thread count")
    parser.add_argument("--output-mode", choices=[mode.value for mode in OutputMode], default=None)
    parser.add_argument("--audio-device", default=None, help="input device index or name")
    parser.add_argument("--no-overlay", action="store_true", help="disable the overlay window")
    parser.add_argument("--no-inject", action="store_true", help="disable text injection")
    parser.add_argument("--save-config", type=Path, default=None, help="write the effective config and exit")
    parser.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.from_env(args.config)
    if args.model:
        config.asr.model_path = args.model
    if args.tokens:
        config.asr.tokens_path = args.tokens
    if args.device:
        config.asr.device = args.device
    if args.threads:
        config.asr.num_threads = args.threads
    if args.output_mode:
        config.output_mode = OutputMode(args.output_mode)
    if args.audio_device:
        config.audio.device = args.audio_device
    if args.no_overlay:
        config.overlay.enabled = False
    if args.no_inject:
        config.injector.enabled = False
    if args.log_level:
        config.log_level = args.log_level
    if args.clinical_sqlite:
        config.clinical_sqlite = args.clinical_sqlite
    if args.clinical_jsonl:
        config.clinical_jsonl = args.clinical_jsonl
    if args.second_pass:
        config.asr.second_pass = args.second_pass
    if args.no_adaptive_vad:
        config.audio.vad_adaptive = False
    if args.require_streaming:
        config.asr.require_streaming = True
    config.asr.__post_init__()
    config.audio.__post_init__()
    return config


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        from tools.verify_pipeline import main as verify
        return verify(["--self-test"])

    if args.list_devices:
        for device in AudioCapture.list_devices():
            print(f"[{device['index']}] {device['name']} ({device['channels']} ch)")
        return 0

    try:
        config = build_config(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.save_config:
        ConfigManager.save(config, args.save_config)
        print(f"config written to {args.save_config}")
        return 0

    try:
        app = ShenavaApp(config)
    except (RuntimeError, OSError, ValueError) as exc:
        # Missing model stack / unusable microphone: report without a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for signal_name in ("SIGINT", "SIGTERM"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is not None:
            try:
                signal.signal(signal_number, lambda *_: app.request_shutdown())
            except (ValueError, OSError):
                logger.debug("%s handler could not be installed", signal_name)
    try:
        app.start()
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
