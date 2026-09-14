"""Start the application detached and tee its output to ``app.log``.

Useful on Windows, where a console window per run is annoying::

    python run_app.py                 # start in the background
    python run_app.py -- --no-overlay # pass arguments through to main.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "app.log"


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]

    with LOG.open("w", encoding="utf-8") as log:
        kwargs: dict = {"cwd": ROOT, "stdout": log, "stderr": subprocess.STDOUT}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen([sys.executable, "-u", str(ROOT / "main.py"), *argv], **kwargs)

    print(f"started PID {process.pid}, logging to {LOG}")
    print(f"stop with: Ctrl+Alt+Q inside the app, or taskkill /F /PID {process.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
