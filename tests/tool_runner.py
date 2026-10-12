"""Run a vla_tools command the way the skills document it: `python -m vla_tools.<module> ...`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_tool(module: str, *args: str | Path, expect_code: int = 0) -> str:
    """Run the module with the current interpreter; return its stdout + stderr.

    Fails with the full output when the exit code differs from `expect_code`.
    """
    command = [sys.executable, "-m", f"vla_tools.{module}", *(str(a) for a in args)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    print(f"$ {' '.join(command[1:])}\n{result.stdout}")
    if result.returncode != expect_code:
        raise AssertionError(f"{module} exited {result.returncode}, expected {expect_code}:\n{output}")
    return output
