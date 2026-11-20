"""Crash recovery of `tune.runner.run` (deep check HAR-1). CPU only: environment capture, candidates, inputs, validation
and timing are fakes, so a candidate can "poison the CUDA context" without touching the GPU.

The bug this guards: `validate()` catches every exception, including a sticky CUDA fault, so the runner went on in a dead
context, every later candidate of the shape failed with the culprit's error and the first timing call killed the process
while `inflight.json` named an innocent candidate; all of them stayed invalid for good."""
import json
from types import SimpleNamespace

import pytest
import torch

import tune.runner as runner
from tune.correctness import CheckResult

POISON = "AcceleratorError: CUDA error: an illegal memory access was encountered"
SHAPE = [{"M": 8, "N": 8, "K": 8, "suite": "t"}]
MODES = ("cold", "hot")


class _Exit(Exception):
    pass


_RUNNER_OF: dict[int, str] = {}  # id(timed fn) -> candidate name, so the fake benchmark knows whose kernel it times


class _Cand:
    def __init__(self, key):
        self.family, self.config_key, self.config = "fake", key, {}
        self.backend = SimpleNamespace(name=key, make_runner=self._runner)

    def _runner(self, a, b, out=None):
        f = lambda: a @ b  # noqa: E731
        _RUNNER_OF[id(f)] = self.config_key
        return f


def _fake(monkeypatch, *, poison=(), crash_timing=()):
    """Install the fakes; returns the list of candidate names `validate` was called with."""
    calls = []
    monkeypatch.setattr(runner, "write_env", lambda *a, **k: {
        "hw_id": "hw", "driver": "d", "versions": {"torch_cuda": "c", "torch": "t", "triton": "tr"}})
    monkeypatch.setattr(runner, "provenance", lambda env: {})
    monkeypatch.setattr(runner, "candidates_for", lambda spec, M, N, K, *a: [_Cand(k) for k in ("good_A", "BAD", "good_C")])
    monkeypatch.setattr(runner, "placed_inputs", lambda M, N, K, dt, seed=0, placement=0, avoid=frozenset():
                        (torch.ones(M, K), torch.ones(K, N), None))
    monkeypatch.setattr(runner, "reference", lambda a, b: a @ b)

    def validate(backend, *a, **k):
        calls.append(backend.name)
        if backend.name in poison:
            return CheckResult(False, POISON, float("nan"), float("nan"))
        return CheckResult(True, None, 0.0, 0.0)

    def benchmark(fn, cfg):
        if _RUNNER_OF[id(fn)] in crash_timing:
            raise RuntimeError(POISON)        # a fault that surfaces while timing kills the process
        return SimpleNamespace(median_us=1.0)

    def _exit(code):
        raise _Exit(code)
    monkeypatch.setattr(runner, "validate", validate)
    monkeypatch.setattr(runner, "benchmark", benchmark)
    monkeypatch.setattr(runner, "pool", lambda parts: SimpleNamespace(
        median_us=1.0, p10_us=1.0, p90_us=1.0, min_us=1.0, max_us=1.0, iqr_us=0.0, rse=0.0, n_samples=1,
        converged=True, telemetry_after={}, telemetry_before={}, samples_us=[1.0]))
    monkeypatch.setattr(runner.os, "_exit", _exit)
    return calls


def _run(tmp_path):
    return runner.run(SHAPE, ["fake"], tmp_path, cache_modes=MODES, log=lambda *_: None)


def _rows(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "results.jsonl")]


def test_poisoned_validation_blames_only_the_culprit_and_resume_completes(tmp_path, monkeypatch):
    calls = _fake(monkeypatch, poison={"BAD"})
    with pytest.raises(_Exit) as e:
        _run(tmp_path)
    assert e.value.args[0] == runner.EXIT_POISONED
    assert calls == ["good_A", "BAD"]                     # stopped at the culprit: good_C never ran in a dead context
    assert not (tmp_path / "inflight.json").exists()
    rows = _rows(tmp_path)
    assert {(r["config_key"], r["cache_mode"]) for r in rows} == {("BAD", m) for m in MODES}
    assert all(not r["valid"] and r["failure_reason"].startswith("context_poisoned:") for r in rows)

    calls = _fake(monkeypatch, poison={"BAD"})            # the re-run (a fresh process in real life) resumes
    _run(tmp_path)
    assert "BAD" not in calls                              # already recorded, not run again
    rows = _rows(tmp_path)
    by = {(r["config_key"], r["cache_mode"]): r for r in rows}
    assert len(rows) == 6 and len(by) == 6
    assert all(by[(k, m)]["valid"] for k in ("good_A", "good_C") for m in MODES)
    assert not any(r["failure_reason"] == "crashed_previous_run" for r in rows)


def test_crash_while_timing_blames_every_cache_mode_of_that_candidate(tmp_path, monkeypatch):
    _fake(monkeypatch, crash_timing={"BAD"})
    with pytest.raises(RuntimeError):
        _run(tmp_path)
    marker = json.loads((tmp_path / "inflight.json").read_text())
    assert {(r["config_key"], r["cache_mode"]) for r in marker["rows"]} == {("BAD", m) for m in MODES}

    calls = _fake(monkeypatch)
    _run(tmp_path)
    assert "BAD" not in calls                              # both of its modes were recorded: no second crash
    by = {(r["config_key"], r["cache_mode"]): r for r in _rows(tmp_path)}
    assert all(by[("BAD", m)]["failure_reason"] == "crashed_previous_run" for m in MODES)
    assert all(by[(k, m)]["valid"] for k in ("good_A", "good_C") for m in MODES)


def test_resume_reads_the_old_single_row_inflight_format(tmp_path, monkeypatch):
    _fake(monkeypatch)
    row = {"M": 8, "N": 8, "K": 8, "dtype": "fp16", "layout": "NN", "kernel_family": "fake", "config_key": "BAD",
           "cache_mode": "cold", "batch": 1, "use_graph": False}
    from tune.records import measurement_key
    (tmp_path / "inflight.json").write_text(json.dumps({"key": list(measurement_key(row)), "row": row}))
    _run(tmp_path)
    by = {(r["config_key"], r["cache_mode"]): r for r in _rows(tmp_path)}
    assert by[("BAD", "cold")]["failure_reason"] == "crashed_previous_run"
    assert by[("BAD", "hot")]["valid"]
