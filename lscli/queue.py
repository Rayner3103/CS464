"""Search requests (`queue.jsonl`, spec 3.4). Clients and the CLI append; the daemon reads from its saved offset,
de-duplicates and, once every line is consumed, truncates the file under the same lock appenders take (so no request
can be lost between the read and the truncate)."""
from __future__ import annotations

import fcntl
import os
import time

from lscli.files import append_jsonl, read_jsonl

REASONS = ("manual", "miss", "stale", "superseded")   # priority order: a shape's request keeps its most urgent reason


def request(paths, M: int, N: int, K: int, dtype: str, env_id: str, reason: str) -> dict:
    if reason not in REASONS:
        raise ValueError(f"reason must be one of {REASONS}")
    row = {"t": time.time(), "op": "search", "M": int(M), "N": int(N), "K": int(K), "dtype": dtype, "env_id": env_id,
           "reason": reason}
    append_jsonl(paths.queue, [row])
    return row


def _offset(paths) -> int:
    try:
        return int(paths.queue_offset.read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def pending(paths) -> tuple[list[dict], int]:
    """(requests not yet consumed, de-duplicated on shape + dtype + env with the most urgent reason, oldest first; the
    end offset to pass to `consume`)."""
    start = _offset(paths)
    try:
        if start > os.path.getsize(paths.queue):
            start = 0                                    # truncated since the offset was saved
    except OSError:
        return [], 0
    rows, end = read_jsonl(paths.queue, start)
    out: dict[tuple, dict] = {}
    for r in rows:
        try:
            k = (int(r["M"]), int(r["N"]), int(r["K"]), r["dtype"], r["env_id"])
        except (KeyError, ValueError, TypeError):
            continue
        prev = out.get(k)
        if prev is None or REASONS.index(r.get("reason", "miss")) < REASONS.index(prev.get("reason", "miss")):
            out[k] = {**r, "t": prev["t"] if prev else r.get("t")}
    return sorted(out.values(), key=lambda r: r.get("t") or 0), end


def consume(paths, end: int) -> None:
    """Mark everything before `end` consumed; truncate the file when nothing was appended after it."""
    paths.root.mkdir(parents=True, exist_ok=True)
    with open(paths.queue, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        if f.tell() <= end:
            f.truncate(0)
            end = 0
        tmp = paths.queue_offset.with_name("queue.offset.tmp")
        tmp.write_text(str(end))
        os.replace(tmp, paths.queue_offset)
