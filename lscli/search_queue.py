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


def _rank(r: dict) -> int:
    """Priority of a request's reason; an unknown one (a newer lscli wrote it) ranks last instead of failing."""
    reason = r.get("reason")
    return REASONS.index(reason) if reason in REASONS else len(REASONS)


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
            if r.get("op", "search") != "search":
                continue
            k = (int(r["M"]), int(r["N"]), int(r["K"]), str(r["dtype"]), str(r["env_id"]))
        except (KeyError, ValueError, TypeError):
            continue
        prev = out.get(k)
        if prev is None or _rank(r) < _rank(prev):
            out[k] = {**r, "t": prev["t"] if prev else r.get("t")}
    return sorted(out.values(), key=lambda r: r.get("t") or 0), end


def consume(paths, end: int) -> None:
    """Mark everything before `end` consumed; truncate the file when nothing was appended after it. The offset is saved
    BEFORE the truncation: a crash in between re-delivers requests (de-duplicated, then dropped as satisfied) instead
    of skipping new ones."""
    paths.root.mkdir(parents=True, exist_ok=True)

    def save(n: int) -> None:
        tmp = paths.queue_offset.with_name("queue.offset.tmp")
        tmp.write_text(str(n))
        os.replace(tmp, paths.queue_offset)
    with open(paths.queue, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        if f.tell() <= end:
            save(0)
            f.truncate(0)
        else:
            save(end)


def retain(paths, keep) -> tuple[int, int]:
    """Rewrite the queue keeping only the requests `keep(request)` still wants (de-duplicated, most urgent reason,
    oldest time), in place under the appenders' lock, so one request that waits for a slow re-search never holds up
    the others and the file never grows without bound. Returns (lines before, requests kept)."""
    import json
    paths.root.mkdir(parents=True, exist_ok=True)
    with open(paths.queue, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        data = f.read()
        end = data.rfind(b"\n") + 1                     # an unterminated tail is being written: keep it as it is
        rows = []
        for line in data[:end].splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict):
                rows.append(r)
        save_offset = paths.queue_offset.with_name("queue.offset.tmp")
        save_offset.write_text("0")
        os.replace(save_offset, paths.queue_offset)
        by_key: dict[tuple, dict] = {}
        for r in rows:
            try:
                if r.get("op", "search") != "search":
                    continue
                k = (int(r["M"]), int(r["N"]), int(r["K"]), str(r["dtype"]), str(r["env_id"]))
            except (KeyError, ValueError, TypeError):
                continue
            prev = by_key.get(k)
            if prev is None or _rank(r) < _rank(prev):
                by_key[k] = {**r, "t": min(prev["t"], r.get("t", prev["t"])) if prev else r.get("t")}
        kept = [r for r in sorted(by_key.values(), key=lambda r: r.get("t") or 0) if keep(r)]
        f.seek(0)
        f.truncate(0)
        f.write(b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in kept) + data[end:])
        f.flush()
        os.fsync(f.fileno())
    return len(rows), len(kept)
