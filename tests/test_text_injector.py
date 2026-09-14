"""Text injection: clipboard-first on Windows, delta-only, no duplicates."""

import pytest

from shenava_realtime.config import InjectorConfig, InjectorMode
from shenava_realtime.injector.text_injector import InjectionResult, TextInjector


class FakeClipboard:
    def __init__(self):
        self.text = ""
        self.history = []
        self.available = True
        self.backend = "fake"

    def get(self):
        return self.text

    def set(self, text):
        self.text = text
        self.history.append(text)
        return True

    def restore(self, previous):
        if previous is None:
            return False
        self.text = previous
        self.restored.append(previous)
        return True

    restored: list = []


class FakeKeyboard:
    def __init__(self):
        self.typed = []
        self.combos = []
        self.available = True

    def type_text(self, text, interval=0.0):
        self.typed.append(text)
        return True

    def press(self, key):
        self.combos.append((key,))
        return True

    def press_combo(self, keys):
        self.combos.append(tuple(keys))
        return True


@pytest.fixture()
def clipboard_config():
    config = InjectorConfig()
    config.mode = InjectorMode.CLIPBOARD
    config.delay_after_paste_s = 0.0
    config.skip_consecutive_duplicates = True
    return config


def make_injector(config, clipboard=None, keyboard=None):
    return TextInjector(
        config,
        clipboard=clipboard or FakeClipboard(),
        keyboard=keyboard or FakeKeyboard(),
    )


def test_clipboard_injection_uses_ctrl_v(clipboard_config):
    clipboard, keyboard = FakeClipboard(), FakeKeyboard()
    injector = make_injector(clipboard_config, clipboard, keyboard)
    assert injector.inject_now("بیمار تحت عمل CABG") is InjectionResult.SUCCESS
    assert clipboard.history[0] == "بیمار تحت عمل CABG"
    assert keyboard.combos == [("ctrl", "v")]
    assert keyboard.typed == []


def test_clipboard_is_restored_after_pasting(clipboard_config):
    clipboard = FakeClipboard()
    clipboard.text = "previous"
    injector = make_injector(clipboard_config, clipboard, FakeKeyboard())
    injector.inject_now("متن جدید")
    assert clipboard.text == "previous"


def test_clipboard_restore_can_be_disabled(clipboard_config):
    clipboard_config.restore_clipboard = False
    clipboard = FakeClipboard()
    clipboard.text = "previous"
    injector = make_injector(clipboard_config, clipboard, FakeKeyboard())
    injector.inject_now("متن جدید")
    assert clipboard.text == "متن جدید"


def test_keyboard_mode_types_unicode():
    config = InjectorConfig()
    config.mode = InjectorMode.KEYBOARD
    keyboard = FakeKeyboard()
    injector = make_injector(config, FakeClipboard(), keyboard)
    assert injector.inject_now("سلام دنیا") is InjectionResult.SUCCESS
    assert keyboard.typed == ["سلام دنیا"]


def test_auto_mode_prefers_the_clipboard_on_windows(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    injector = make_injector(InjectorConfig(), FakeClipboard(), FakeKeyboard())
    assert injector.mode is InjectorMode.CLIPBOARD


def test_auto_mode_uses_the_keyboard_elsewhere(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    injector = make_injector(InjectorConfig(), FakeClipboard(), FakeKeyboard())
    assert injector.mode is InjectorMode.KEYBOARD


def test_duplicate_text_is_skipped(clipboard_config):
    clipboard, keyboard = FakeClipboard(), FakeKeyboard()
    injector = make_injector(clipboard_config, clipboard, keyboard)
    assert injector.inject_now("mg") is InjectionResult.SUCCESS
    assert injector.inject_now("mg") is InjectionResult.SKIPPED
    # Only one paste happened; the second clipboard write is the restore of the
    # (empty) clipboard the user had before.
    assert keyboard.combos == [("ctrl", "v")]
    assert injector.get_statistics()["successful"] == 1
    assert injector.get_statistics()["skipped"] == 1


def test_guard_can_be_disabled(clipboard_config):
    clipboard_config.skip_consecutive_duplicates = False
    injector = make_injector(clipboard_config, FakeClipboard(), FakeKeyboard())
    assert injector.inject_now("mg") is InjectionResult.SUCCESS
    assert injector.inject_now("mg") is InjectionResult.SUCCESS


def test_non_consecutive_repeat_is_not_swallowed(clipboard_config):
    injector = make_injector(clipboard_config, FakeClipboard(), FakeKeyboard())
    assert injector.inject_now(" و") is InjectionResult.SUCCESS
    assert injector.inject_now(" بیمار") is InjectionResult.SUCCESS
    assert injector.inject_now(" و") is InjectionResult.SUCCESS


def test_empty_and_inactive_injections_are_ignored(clipboard_config):
    injector = make_injector(clipboard_config, FakeClipboard(), FakeKeyboard())
    assert injector.inject("   ") is False
    injector.is_active = False
    assert injector.inject("بیمار") is False
    assert injector.get_statistics()["injections"] == 0


def test_queue_worker_injects_and_stops_cleanly(clipboard_config):
    clipboard, keyboard = FakeClipboard(), FakeKeyboard()
    injector = make_injector(clipboard_config, clipboard, keyboard)
    injector.start()
    try:
        assert injector.inject("بیمار")
        assert injector.inject(" تحت عمل")
        injector.flush(timeout=2.0)
    finally:
        injector.stop()
        injector.stop()  # idempotent
    # Deltas keep their separating space, so the typed text stays readable.
    assert clipboard.history == ["بیمار", " تحت عمل"]
    assert injector.is_running is False
    assert injector.get_statistics()["successful"] == 2


def test_toggle_disables_output(clipboard_config):
    injector = make_injector(clipboard_config, FakeClipboard(), FakeKeyboard())
    injector.toggle()
    assert injector.is_active is False
    assert injector.inject("بیمار") is False
    injector.toggle()
    assert injector.is_active is True


def test_failure_is_reported(clipboard_config):
    class BrokenClipboard(FakeClipboard):
        def set(self, text):
            return False

    failures = []
    injector = make_injector(clipboard_config, BrokenClipboard(), FakeKeyboard())
    injector.on_failed = failures.append
    assert injector.inject_now("بیمار") is InjectionResult.FAILED
    assert failures == ["بیمار"]
    assert injector.get_statistics()["failed"] == 1


def test_falls_back_to_keyboard_without_a_clipboard():
    config = InjectorConfig()
    config.mode = InjectorMode.CLIPBOARD
    clipboard = FakeClipboard()
    clipboard.available = False
    keyboard = FakeKeyboard()
    injector = make_injector(config, clipboard, keyboard)
    injector.start()
    try:
        assert injector.inject_now("بیمار") is InjectionResult.SUCCESS
    finally:
        injector.stop()
    assert keyboard.typed == ["بیمار"]
