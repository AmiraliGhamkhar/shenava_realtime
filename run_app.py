"""Launch the Shenava app as a detached background process."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = ROOT / "venv" / "Scripts" / "python.exe"
LOG = ROOT / "app.log"

proc = subprocess.Popen(
    [str(PY), "-u", str(ROOT / "main.py")],
    cwd=str(ROOT),
    stdout=open(LOG, "w", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)
print(f"Started PID {proc.pid}, logging to {LOG}")
