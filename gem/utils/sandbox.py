from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict


def run_python(code: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    """Run python code in a subprocess (best-effort sandbox)."""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as e:
        return {"exit_code": -1, "stdout": e.stdout or "", "stderr": (e.stderr or "") + "\nTIMEOUT"}

