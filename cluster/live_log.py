"""Mirror launcher stdout and stderr to an append-only run log."""

from __future__ import annotations

import atexit
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


class _Tee:
    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, value: str) -> int:
        self.log.write(value)
        self.log.flush()
        self.terminal.write(value)
        self.terminal.flush()
        return len(value)

    def flush(self) -> None:
        self.log.flush()
        self.terminal.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()

    @property
    def encoding(self) -> str | None:
        return self.terminal.encoding


def activate_live_log(path: Path) -> None:
    """Append all subsequent Python output to path while keeping it live."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log = path.open("a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, log)
    sys.stderr = _Tee(original_stderr, log)

    def restore() -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log.close()

    atexit.register(restore)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Run log: {path}", flush=True)
