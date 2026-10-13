import json

from suites.generate import build
from tune.records import RecordWriter, alignment_flags, completed_keys, measurement_key, read_raw, read_records


def _row(M, cfg="a"):
    return {"M": M, "N": 8, "K": 8, "dtype": "fp16", "layout": "NN", "kernel_family": "f", "config_key": cfg,
            "cache_mode": "cold", "batch": 1, "use_graph": False}


def test_append_resume_and_torn_line(tmp_path):
    p = tmp_path / "r.jsonl"
    with RecordWriter(p) as w:
        rid = w.write(_row(1), samples_us=[1.0, 2.0])
        w.write(_row(2))
    with open(p, "a") as f:
        f.write('{"M": 3, "trunc')  # simulated crash mid-write
    rows = read_records(p)
    assert [r["M"] for r in rows] == [1, 2]
    assert measurement_key(_row(1)) in completed_keys(p)
    assert measurement_key(_row(1, cfg="b")) not in completed_keys(p)
    assert read_raw(p.with_name("r.raw.jsonl"))[rid] == [1.0, 2.0]


def test_alignment_flags():
    f = alignment_flags(64, 1000, 7)
    assert f["M_div64"] and f["N_div8"] and not f["N_div16"] and not f["K_div2"]


def test_suites_deterministic_and_split():
    a, b = build(68), build(68)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    for name in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        assert a[name], name
        for s in a[name]:
            assert s["split"] in ("tune", "heldout_region", "heldout_random")
            if 9000 <= s["K"] <= 12000:
                assert s["split"] == "heldout_region"
    assert len(a["noise"]) == 30
    # Region holdout keeps a real LLM dim (11008) unseen during tuning.
    assert any(s["K"] == 11008 and s["split"] == "heldout_region" for s in a["S2"])


def test_s6_depends_on_sm_count():
    assert build(68)["S6"] != build(108)["S6"]


def test_session_env_and_provenance(tmp_path, monkeypatch):
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from tune.env import provenance, write_env
    monkeypatch.setenv("TUNE_CLOCK_LOCK", "none")
    env = write_env(tmp_path, session_id="sess-x", kind="test")
    assert (tmp_path / "env" / "sess-x.json").exists() and (tmp_path / "env.json").exists()
    assert json.loads((tmp_path / "env.json").read_text())["session_id"] == "sess-x"
    lock = env["clock_lock"]
    assert lock["declared"] == "none" and lock["state"] == "unlocked" and lock["probe"] in ("likely_locked", "likely_unlocked")
    p = provenance(env)
    assert p["clock_lock_state"] == "unlocked" and p["env_file"] == "env/sess-x.json" and len(p["harness_commit"]) == 12


def test_noise_rows_carry_provenance(tmp_path, monkeypatch):
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from tune.noise import run_noise
    from tune.records import read_records
    monkeypatch.setenv("TUNE_CLOCK_LOCK", "none")
    path = run_noise([{"M": 64, "N": 64, "K": 64}], tmp_path, backends=("torch",), repeats=3, rounds=1,
                     cache_modes=("hot",), session_id="s-prov", log=lambda *_: None)
    row = read_records(path)[0]
    assert row["clock_lock_state"] == "unlocked" and row["env_file"] == "env/s-prov.json"
    assert row["harness_commit"]
