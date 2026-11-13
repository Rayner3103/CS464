"""Training this GPU's models in idle time (spec 4.5): `pipeline/train_best.sh`, one stage per job.

A training is a run directory `train/<hw_id>/<n>/` driven by `train_best.sh --only <stage>` (first stage with
`--run-dir --full|--quick`, later ones with `--resume`), always with `--skip-install --skip-validation` and the clock
policy. train_best stamps a finished stage (`stamps/<stage>.done`); a stage that was killed has no stamp and resumes per
shape the next time (sweeps write one file per finished shape; benchmarks one fsync'd row per measurement). A
preempted benchmark leaves `inflight.json`, which train_best's runner would read as a crash of that candidate: the
daemon removes it, because the kill was ours.

Retraining after an environment change runs in a NEW directory seeded from the previous run minus the outputs and
stamps of the stages that re-run: reusing the old directory would keep their old rows, because the stages resume per
shape and `--only` skips stamped stages. The outputs of the stages that do not re-run are hard links (big, and never
written again); everything else is copied, so the new run cannot change the old one through a shared inode.

When `package` finishes, the run's models become this GPU's model set (`models/<hw_id>/`, swapped in atomically), with
the training environment recorded for the next comparison.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from lscli.files import read_json, write_json

STAGES = ("preflight", "env", "check", "roofline", "tune_sweep", "heldout_sweep", "selectors", "train_v1", "tail_sweep",
          "train_v2", "lt_screen", "lt_fit", "tune_stage2_cold", "tune_stage2_graph", "router_fit", "bundle", "package",
          "summary")
# preflight is NOT CPU-only: train_best's check_prereqs refuses a GPU with compute processes on it.
CPU_STAGES = frozenset({"train_v2", "lt_fit", "router_fit", "package", "summary"})
RETRAIN = {
    "cublaslt": ("env", "lt_screen", "lt_fit", "tune_stage2_cold", "tune_stage2_graph", "router_fit", "package", "summary"),
    "triton": ("env", "check", "tune_sweep", "heldout_sweep", "selectors", "train_v1", "tail_sweep", "train_v2",
               "tune_stage2_cold", "tune_stage2_graph", "router_fit", "bundle", "package", "summary"),
}
# What each stage writes, relative to the run directory (DATA = data/<hw_id>). Deleted before a stage re-runs in a
# seeded directory, so nothing of the previous environment is resumed. Caches that depend on the environment
# (the cuBLASLt deep-search cache, the selectors' and topk's CV caches) belong to the stage that fills them.
OUTPUTS = {
    "tune_sweep": ("DATA/sweep_screen",),
    "heldout_sweep": ("DATA/sweep_screen_heldout",),
    "selectors": ("DATA/selector_picks.json", "DATA/selector_knnc_fit.json"),
    "train_v1": ("DATA/lgbm_rank.txt", "DATA/lgbm_rank.json", "DATA/lgbm_rank_cv_scores.npz", "DATA/learned_search.json",
                 "DATA/learned_picks.json", "DATA/learned_topk_picks.json", "DATA/online_lgbm_rank*.jsonl",
                 "DATA/online_triton_cache"),
    "tail_sweep": ("DATA/sweep_screen_tail",),
    "train_v2": ("DATA/lgbm_rank_v2.txt", "DATA/lgbm_rank_v2.json", "DATA/lgbm_rank_v2_cv_scores.npz"),
    "lt_screen": ("DATA/lt_screen",),
    "lt_fit": ("DATA/lt_rank.txt", "DATA/lt_rank.json"),
    "tune_stage2_cold": ("DATA/results_lock1755_tune477.jsonl", "DATA/results_lock1755_tune477.raw.jsonl", "DATA/lt_search",
                         "DATA/inflight.json"),
    "tune_stage2_graph": ("DATA/results_lock1755_tune477_graph.jsonl", "DATA/results_lock1755_tune477_graph.raw.jsonl"),
    "router_fit": ("DATA/router.txt", "DATA/router.json", "DATA/router.skipped"),
    "env": ("DATA/env.json", "DATA/env"),
    "check": ("DATA/check-*.json",),
    "bundle": ("bundle", "models/bundle.tar.xz", "DATA/bundle_build_cache"),
    "package": ("models/*.txt", "models/*.json", "models/MD5SUMS", "models/INSTALL.md"),
    "summary": ("SUMMARY.md",),
}
MAX_STAGE_FAILURES = 2
LINK_MIN_BYTES = 1 << 20     # seeding hard-links only big files of frozen outputs; small ones (subset files a later
                             # stage may rewrite in place, e.g. sweep_screen/subset_lt50us_477.json) are copied


def stages_for(change: set[str], with_bundle: bool) -> list[str]:
    """Stages to run for a set of changes ({"first"} = the whole training), in train_best order."""
    want = set(STAGES) if "first" in change else set().union(*(RETRAIN[c] for c in change if c in RETRAIN))
    return [s for s in STAGES if s in want and (with_bundle or s != "bundle")]


def changes(trained_env: dict | None, env: dict) -> set[str]:
    """What changed since the active models were trained: {"first"} when nothing was trained yet."""
    if not trained_env:
        return {"first"}
    out = set()
    if trained_env.get("cublaslt") != env.get("cublaslt"):
        out.add("cublaslt")
    if trained_env.get("triton") != env.get("triton") or trained_env.get("kernel_src") != env.get("kernel_src"):
        out.add("triton")
    return out


def _rm(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def output_paths(run: Path, stage: str) -> list[Path]:
    hw = (run / "hw_id").read_text().strip() if (run / "hw_id").exists() else None
    out = []
    for pat in OUTPUTS.get(stage, ()):
        if pat.startswith("DATA/"):
            if hw is None:
                continue
            base, pat = run / "data" / hw, pat[5:]
        else:
            base = run
        out += list(base.glob(pat)) if any(c in pat for c in "*?[") else [base / pat]
    return out


def seed(prev: Path | None, new: Path, stages: list[str]) -> None:
    """Create `new` from `prev` without the outputs and stamps of `stages`. The outputs of the stages that do NOT re-run
    are hard-linked (nothing writes them again); every other file (reports, logs, top-level files a stage may rewrite in
    place) is copied, so the new run can never modify the previous one through a shared inode."""
    new.mkdir(parents=True, exist_ok=True)
    if prev is None or not prev.exists():
        return
    rerun = set(stages)
    frozen = [p.resolve() for st in STAGES if st not in rerun for p in output_paths(prev, st) if p.exists()]
    dropped = [p.resolve() for st in rerun for p in output_paths(prev, st) if p.exists()]
    for root, dirs, files in os.walk(prev):
        rel = Path(root).relative_to(prev)
        rr = Path(root).resolve()
        if rel.parts[:1] == ("logs",) or any(rr == d or d in rr.parents for d in dropped):
            dirs[:] = []                                 # a re-run stage's output directory: not even created
            continue
        (new / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            src, dst = Path(root) / f, new / rel / f
            r = src.resolve()
            if any(r == d or d in r.parents for d in dropped):
                continue
            if rel.parts[:1] == ("stamps",) and f[:-5] in rerun:
                continue
            if src.is_symlink():
                os.symlink(os.readlink(src), dst)
            elif any(r == d or d in r.parents for d in frozen) and src.stat().st_size >= LINK_MIN_BYTES:
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)


class Training:
    """The training state of one GPU (`train/<hw_id>/state.json`)."""

    def __init__(self, paths, hw_id: str):
        self.paths, self.hw_id = paths, hw_id
        self.root = paths.train / hw_id
        self.state_path = self.root / "state.json"
        self.state = read_json(self.state_path, None) or {}

    def save(self) -> None:
        write_json(self.state_path, self.state)

    @property
    def run_dir(self) -> Path | None:
        return self.root / str(self.state["run"]) if self.state.get("run") else None

    def done(self) -> set[str]:
        d = self.run_dir
        return {p.name[:-5] for p in (d / "stamps").glob("*.done")} if d and (d / "stamps").exists() else set()

    def active(self) -> bool:
        return bool(self.state.get("stages")) and self.state.get("status") == "running"

    def next_stage(self) -> str | None:
        if not self.active():
            return None
        done = self.done()
        return next((s for s in self.state["stages"] if s not in done), None)

    def begin(self, change: set[str], stages: list[str], env: dict, reason: str, request_t: float | None = None) -> None:
        """Start a training run for `stages` (run in train_best order) in a new run directory, seeded from the previous
        one unless this is a full training. The number is one past every existing directory: a lost state file must
        never make it reuse (and delete) a finished run."""
        prev = self.run_dir
        nums = [int(d.name) for d in self.root.glob("*") if d.is_dir() and d.name.isdigit()]
        n = max(nums + [int(self.state.get("run") or 0)]) + 1
        new = self.root / str(n)
        seed(prev if "first" not in change else None, new, stages)
        self.state = {"run": n, "stages": [s for s in STAGES if s in set(stages)], "change": sorted(change),
                      "reason": reason, "env": env, "status": "running", "started": time.time(), "failures": {},
                      "previous": self.state.get("run"),
                      "request_t": request_t if request_t is not None else self.state.get("request_t")}
        self.save()

    def record_failure(self, stage: str) -> int:
        f = self.state.setdefault("failures", {})
        f[stage] = f.get(stage, 0) + 1
        if f[stage] >= MAX_STAGE_FAILURES:
            self.state["status"] = "failed"
        self.save()
        return f[stage]

    def record_success(self, stage: str) -> None:
        if self.state.get("failures", {}).pop(stage, None) is not None:
            self.save()

    def finish(self) -> None:
        self.state["status"] = "done"
        self.state["finished"] = time.time()
        self.save()

    def summary(self) -> dict:
        done = self.done()
        stages = self.state.get("stages") or []
        return {"run": self.state.get("run"), "status": self.state.get("status", "none"), "reason": self.state.get("reason"),
                "stage": self.next_stage(), "done": sum(1 for s in stages if s in done), "total": len(stages),
                "failures": self.state.get("failures", {}), "dir": str(self.run_dir) if self.run_dir else None}


def activate(paths, hw_id: str, run_dir: Path, trained_env: dict) -> Path:
    """Copy the packaged models of `run_dir` into models/<hw_id>/ (swapped atomically) with their MODELS.json."""
    src = run_dir / "models"
    final = paths.models / hw_id
    tmp = paths.models / f".{hw_id}.new-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    found = {}
    for kind, stem in (("triton", "lgbm_rank_v2"), ("lt_rank", "lt_rank"), ("router", "router")):
        if (src / f"{stem}.txt").exists() and (src / f"{stem}.json").exists():
            for ext in (".txt", ".json"):
                shutil.copy2(src / f"{stem}{ext}", tmp / f"{stem}{ext}")
            found[kind] = {"stem": stem}
    if "triton" not in found and (src / "lgbm_rank.txt").exists():       # v2 missing: fall back to v1
        for ext in (".txt", ".json"):
            shutil.copy2(src / f"lgbm_rank{ext}", tmp / f"lgbm_rank{ext}")
        found["triton"] = {"stem": "lgbm_rank"}
    if "triton" not in found:
        shutil.rmtree(tmp)
        raise FileNotFoundError(f"{src}: no Triton ranker to activate")
    from lscli.models import model_id
    for kind, m in found.items():
        m["id"] = model_id(tmp / m["stem"])
    stages = sorted(p.name[:-5] for p in (run_dir / "stamps").glob("*.done")) if (run_dir / "stamps").exists() else []
    write_json(tmp / "MODELS.json", {"hw_id": hw_id, "source": "native", "models": found, "trained_env": trained_env,
                                     "run": str(run_dir), "stages": stages, "activated": time.time()})
    old = None
    if final.exists():
        old = paths.models / f".{hw_id}.old-{int(time.time())}"
        os.replace(final, old)
    os.replace(tmp, final)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    return final


def trained_env(paths, hw_id: str) -> dict | None:
    doc = read_json(paths.models / hw_id / "MODELS.json", None)
    return (doc or {}).get("trained_env")
