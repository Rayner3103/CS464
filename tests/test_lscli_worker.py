"""lscli measure worker (lscli/worker.py) over its real protocol: a subprocess under this interpreter, on the GPU."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import requires_cuda

REPO = Path(__file__).resolve().parent.parent
TRITON = ["bm64_bn64_bk32_w4_s3_g1_sk1", "bm32_bn32_bk32_w4_s5_g1_sk1"]


def _start(*argv):
    return subprocess.Popen([sys.executable, "-m", "lscli.worker", *argv], cwd=REPO, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)


def _request(cache, **kw):
    return {"M": 256, "N": 256, "K": 256, "dtype": "fp16", "k": 2, "lt_n": 4, "confirm": 0,
            "proposals": [{"family": "triton", "key": k, "setup_s": 0.0} for k in TRITON]
            + [{"family": "cublaslt_rank", "key": "r0_a99999_t1_s1_sk1_r0_sw0_c0", "setup_s": 0.0}],   # rejected here
            "cache": str(cache), "env_extra": {"clock_lock": "unlocked", "env_id": "testenv"}, "order": "diverse",
            "ranker": "test", **kw}


def _lines(out: str):
    rows = [json.loads(x) for x in out.strip().splitlines()]                # every stdout line is protocol JSON
    return rows


@requires_cuda
def test_measure_writes_a_winner_with_torch_time_and_scope(tmp_path):
    p = _start("measure")
    out, err = p.communicate(json.dumps(_request(tmp_path / "w.jsonl")) + "\n", timeout=600)
    rows = _lines(out)
    assert p.returncode == 0, err[-2000:]
    res = rows[-1]
    assert res["op"] == "result" and res["written"] and res["torch_us"] > 0
    fams = {c.get("family") for c in res["candidates"]}
    assert fams <= {"triton", "cublaslt"} and "triton" in fams                 # the bogus ranker key was dropped
    assert all(not k.startswith("r0_") for k in res["pool"])
    line = json.loads((tmp_path / "w.jsonl").read_text())
    assert line["torch_us"] == res["torch_us"] and line["env"]["env_id"] == "testenv"
    assert line["env"]["clock_lock"] == "unlocked" and line["choice"]["key"] == res["choice"]["key"]
    v = subprocess.run([sys.executable, "-m", "lscli.worker", "verify", "--cache", str(tmp_path / "w.jsonl"),
                        "--env-extra", json.dumps({"clock_lock": "unlocked", "env_id": "testenv"})],
                       cwd=REPO, capture_output=True, text=True, timeout=600)
    doc = _lines(v.stdout)[-1]
    assert doc["failed"] == 0 and len(doc["entries"]) == 1 and doc["entries"][0]["stale"] == []


@requires_cuda
def test_router_round_trip_can_skip_the_triton_half(tmp_path):
    p = _start("measure")
    p.stdin.write(json.dumps(_request(tmp_path / "w.jsonl", router=True)) + "\n")
    p.stdin.flush()
    ask = json.loads(p.stdout.readline())
    assert ask["op"] == "route" and ask["t_lt_us"] > 0
    p.stdin.write(json.dumps({"skip": True, "p": 0.001, "threshold": 0.02}) + "\n")
    p.stdin.flush()
    res = json.loads(p.stdout.readline())
    p.wait(60)
    assert res["op"] == "result" and res["router"]["skipped"] and res["router"]["threshold"] == 0.02
    assert all(c["family"] == "cublaslt" for c in res["candidates"])


@requires_cuda
def test_a_reply_sent_together_with_the_request_is_not_lost(tmp_path):
    """The daemon may write its answer before the worker asks (pipelined): the worker must still read it."""
    p = _start("measure")
    req = json.dumps(_request(tmp_path / "w.jsonl", router=True, reply_timeout_s=5)) + "\n"
    p.stdin.write(req + json.dumps({"skip": True, "p": 0.0, "threshold": 0.02}) + "\n")
    p.stdin.flush()
    assert json.loads(p.stdout.readline())["op"] == "route"
    res = json.loads(p.stdout.readline())
    p.wait(60)
    assert res["router"]["skipped"]
    assert {c["family"] for c in res["candidates"]} == {"cublaslt"}
    line = json.loads((tmp_path / "w.jsonl").read_text())
    assert line["lt_rank"] == 0                         # the rejected ranker proposal was never timed: width 0


@requires_cuda
def test_router_without_an_answer_does_not_skip(tmp_path):
    p = _start("measure")
    p.stdin.write(json.dumps(_request(tmp_path / "w.jsonl", router=True, reply_timeout_s=0.5)) + "\n")
    p.stdin.flush()
    assert json.loads(p.stdout.readline())["op"] == "route"
    res = json.loads(p.stdout.readline())                                   # no answer sent
    p.wait(60)
    assert res["op"] == "result" and not res["router"]["skipped"]
    assert any(c["family"] == "triton" for c in res["candidates"])


@requires_cuda
def test_errors_are_reported_on_the_channel():
    p = _start("measure")
    out, _ = p.communicate(json.dumps({"M": 8, "N": 8, "K": 8, "dtype": "fp64", "k": 1}) + "\n", timeout=120)
    assert p.returncode == 1 and _lines(out)[-1]["op"] == "error"


@requires_cuda
def test_cache_ls_rates_tiers_against_the_registered_env(tmp_path, monkeypatch, capsys):
    from lscli import cli, envs
    from lscli.paths import Paths
    monkeypatch.setenv("LSCLI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LSCLI_CONFIG", str(tmp_path / "c.toml"))
    paths = Paths().ensure()
    eid, _ = envs.register(paths, sys.executable, {"torch": "x"})
    monkeypatch.chdir(REPO)
    p = _start("measure")
    req = _request(paths.winners)
    req["env_extra"]["env_id"] = eid
    p.communicate(json.dumps(req) + "\n", timeout=600)
    assert cli.main(["--json", "cache", "ls"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["tier"] == "current" and row["env_id"] == eid
    assert cli.main(["cache", "compact"]) == 0 and cli.main(["cache", "verify"]) == 0
