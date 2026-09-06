from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class JsonlFileStream:
    """Append complete JSON log lines without retaining an open file handle."""

    path: Path
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    errors: int = field(default=0, init=False)

    def write(self, value: str) -> int:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    return handle.write(value)
            except OSError:
                self.errors += 1
                return 0

    def flush(self) -> None:
        return


@dataclass
class TeeTextStream:
    streams: tuple[Any, ...]

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def console_and_file_stream(path: Path) -> TeeTextStream:
    return TeeTextStream((sys.stdout, JsonlFileStream(path)))
