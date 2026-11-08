"""File primitives shared by the client, the worker and the daemon: locked, fsync'd JSONL appends that survive a torn
last line, tolerant JSONL reads, and atomic JSON replacement (temp file + rename). Same crash-safety style as
`tune/records.py` and `tune/winner_cache.py`."""
from __future__ import annotations

import fcntl
import json
import os
import uuid
from pathlib import Path


def append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append rows as one write under an exclusive lock; terminate a torn last line first."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    with open(path, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                data = b"\n" + data
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path, start: int = 0) -> tuple[list[dict], int]:
    """(rows, end offset) of the complete lines from byte `start` on. A last line without its newline (being written, or
    torn) is not returned and not counted, so a reader that resumes from the offset sees it once it is complete;
    unparseable complete lines are skipped."""
    try:
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read()
    except FileNotFoundError:
        return [], start
    end = data.rfind(b"\n") + 1
    rows = []
    for line in data[:end].splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows, start + end


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, obj) -> None:
    """Replace `path` atomically: readers see the old or the new document, never a partial one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:12]}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
