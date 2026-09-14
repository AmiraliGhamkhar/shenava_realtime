"""Start the application in the background and write output to ``app.log``."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "app.log"
python = Path(sys.executable)

with LOG.open("w", encoding="utf-8") as log:
    kwargs = {"cwd": ROOT, "stdout": log, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen([str(python), "-u", str(ROOT / "main.py")], **kwargs)

print(f"Started PID {process.pid}, logging to {LOG}")
