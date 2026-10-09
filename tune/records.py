"""Append-only JSONL dataset (plan 5.3). One row per shape x backend/config x dtype x layout x cache mode.

Raw samples go to a side file keyed by `record_id`. Writes are flushed and fsynced per row, so a
crash loses at most the row in flight. `completed_keys` lets runs resume.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

SCHEMA_VERSION = 1

# Order matches plan 5.3; extra fields are allowed after these.
FIELDS = [
    "schema_version", "record_id", "hw_id", "driver", "cuda", "torch", "triton",
    "M", "N", "K", "dtype", "layout", "alignment_flags",
    "kernel_family", "config_key", "config",
    "valid", "failure_reason", "max_abs_err",
    "time_median_us", "time_p10_us", "time_p90_us", "time_min_us", "time_max_us", "time_iqr_us",
    "rse", "n_samples", "converged", "cache_mode", "batch", "use_graph",
    "tflops", "clock_mhz", "temp_c", "power_w", "throttle_reasons",
    "timestamp", "session_id", "suite", "split",
]


def alignment_flags(M: int, N: int, K: int) -> dict:
    return {f"{d}_div{v}": x % v == 0 for d, x in (("M", M), ("N", N), ("K", K)) for v in (2, 4, 8, 16, 64)}


def measurement_key(row: dict) -> tuple:
    return (row["M"], row["N"], row["K"], row["dtype"], row["layout"], row["kernel_family"], row["config_key"],
            row["cache_mode"], row.get("batch", 1), row.get("use_graph", False))


class RecordWriter:
    def __init__(self, path: Path, raw_path: Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_path = Path(raw_path) if raw_path else self.path.with_name(self.path.stem + ".raw.jsonl")
        self._f = open(self.path, "a", buffering=1)
        self._raw = open(self.raw_path, "a", buffering=1)

    def write(self, row: dict, samples_us: list[float] | None = None) -> str:
        row = dict(row)
        row.setdefault("schema_version", SCHEMA_VERSION)
        row.setdefault("record_id", uuid.uuid4().hex)
        ordered = {k: row.get(k) for k in FIELDS}
        ordered.update({k: v for k, v in row.items() if k not in ordered})
        self._f.write(json.dumps(ordered, default=str) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        if samples_us is not None:
            self._raw.write(json.dumps({"record_id": row["record_id"], "samples_us": samples_us}) + "\n")
            self._raw.flush()
        return row["record_id"]

    def close(self):
        self._f.close()
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_records(path: Path) -> list[dict]:
    rows = []
    if not Path(path).exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # torn final line after a crash
                continue
    return rows


def completed_keys(path: Path) -> set[tuple]:
    return {measurement_key(r) for r in read_records(path)}


def read_raw(raw_path: Path) -> dict[str, list[float]]:
    out = {}
    if Path(raw_path).exists():
        with open(raw_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    out[d["record_id"]] = d["samples_us"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return out
