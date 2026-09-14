"""Pytest configuration.

The suite must run without torch/NeMo/sounddevice/pynput and without a GPU, so
nothing here imports the ASR backend; the few places that need one use the stubs
in ``tests/fakes.py``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
