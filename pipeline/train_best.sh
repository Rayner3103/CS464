#!/usr/bin/env bash
# Full training of the shipped selector on THIS GPU (e.g. an RTX 3090): every learned component of `tune.online.select_kernel`, fitted
# from this GPU's own measurements, then checked on the held-out shapes. Nothing is borrowed from the 3080.
#
#   pipeline/train_best.sh [--full | --quick | --smoke] [options]
#
# What "the best model" is (AGENTS.md section 6, reports/week1.md 2026-11-04/05): rank v2 over the Triton family (analytic features, near-top
# weights, the tail suite as extra tune data) + the learned cuBLASLt ranker (top-5 on top of cuBLASLt's heuristic top-16) + the cost-only
# family router, searched with the diversified top-3 online search. Training needs, in dependency order:
#
#   Triton screens   tune_sweep (477 tune shapes) · heldout_sweep (86, oracle + the picks of v1) · tail_sweep (150 extra tune shapes)
#   rank v1          selectors -> train_v1  (`tune topk`: grouped-CV search, saves lgbm_rank.*; v2 reuses its CV-chosen hyperparameters)
#   rank v2          train_v2  (`tune ranker-v2`, CPU)
#   cuBLASLt ranker  lt_screen (every cuBLASLt config of the tune shapes, ~70 min on a 3080) -> lt_fit (CPU)
#   router           tune_stage2_cold / _graph (full protocol: Triton screen top-8 vs B3+ on the tune shapes) -> router_fit (CPU)
#   bundle           bundle (OPTIONAL, --with-bundle): prebuilt Triton kernels of the new pools. A 3090 is sm_86 like the 3080, but the
#                    pools of a retrained ranker differ, so the committed archive no longer covers every pick
#   validation       validate_run -> validate_score -> validate_stage2_cold / _graph -> validate_compare: v1 and v2 + LT16 + ranker top-5 on the
#                    held-out shapes, scored against this GPU's own oracle and against B2 / B3 / B3+ (analysis.zero_shot + ranker_compare)
#   packaging        package (models/ + INSTALL.md) · summary
#
# Held-out shapes are only read by the selectors / validation stages and by `tune topk`'s held-out report; no fit sees them.
#
# Everything goes to pipeline/runs/train_best-<mode>-<timestamp>/ (data/<hw_id>/, reports/, logs/, models/); the repo's own data/ and
# reports/ are never touched. --resume <run dir> skips finished stages. One GPU job at a time (shared pipeline/.lock).
#
# Time (RTX 3080 numbers from reports/week1.md, extrapolated, not measured for this script): tune sweep ~17 h, held-out ~1.3 h, tail ~5 h,
# topk CV ~2 h, cuBLASLt screen ~1.2 h, tune stage 2 (B3+ at 20 s x 477 shapes + triton:sweep:8 + cublaslt:16) ~6-8 h, validation ~1.5 h.
# About 35 h on a 3080; a 3090 has more SMs and a faster memory system, but compile time on the host dominates the sweeps.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGES=(preflight install env check roofline
        tune_sweep heldout_sweep selectors train_v1 tail_sweep train_v2
        lt_screen lt_fit
        tune_stage2_cold tune_stage2_graph router_fit
        bundle
        validate_run validate_score validate_stage2_cold validate_stage2_graph validate_compare
        package summary)
GPU_STAGES=(env check roofline tune_sweep heldout_sweep selectors train_v1 tail_sweep lt_screen tune_stage2_cold tune_stage2_graph
            bundle validate_run validate_stage2_cold validate_stage2_graph)

MODE=full
LOCK=1755
ALLOW_UNLOCKED=0
RUN_DIR=""
VENV="$REPO/pipeline/.venv"
PYTHON_BIN="${PYTHON:-python3}"
SKIP_INSTALL=0
WORKERS=0
FROM=""
ONLY=""
CHECK_ONLY=0
FORCE=0
WITH_BUNDLE=0
SKIP_VALIDATION=0

usage() {
  cat <<EOF
Usage: pipeline/train_best.sh [mode] [options]

Modes (default --full):
  --full            the whole study on this GPU: 477 tune + 150 tail shapes, 86 held-out shapes, all 652 cuBLASLt screen shapes
  --quick           a reduced run for slow cards (RTX 2060: its Triton sweep took 17.7 h for 27 held-out shapes, so --full would take weeks): ~56 tune,
                    ~27 held-out and 30 tail shapes, 160 cuBLASLt screen shapes, B3+ at 10 s per shape. The fits are smaller, not meaningless
  --smoke           plumbing check with a handful of shapes per stage (the fits are meaningless; still hours of GPU time)

Options:
  --clock-lock MHZ|none   SM clock you locked beforehand with 'sudo nvidia-smi -lgc MHZ,MHZ' (default $LOCK; the 3080 study used 1755).
                          Choose a clock this card holds under load without power-capping (~80-85% of its maximum SM clock; the 3090
                          zero-shot run was unlocked). The script never calls sudo. 'none' needs --allow-unlocked
  --allow-unlocked        continue even if the GPU clock is not locked (results not comparable to the 3080 study)
  --with-bundle           also build, verify and pack the prebuilt Triton kernel bundle of the new rankers (stage bundle; ~10 min on 10 workers
                          + verify; written under the run directory, never into kernels/)
  --skip-validation       stop after training and packaging the models (skips the validate_* stages)
  --run-dir DIR           output directory (default pipeline/runs/train_best-<mode>-<timestamp>)
  --resume DIR            continue a previous run directory; finished stages are skipped (mode is read from it)
  --venv DIR              virtualenv location (default pipeline/.venv)
  --python BIN            interpreter used to create the venv (default python3, or \$PYTHON)
  --skip-install          use the current 'python' on PATH as is (checks the pinned versions, installs nothing)
  --workers N             parallel compile workers of the Triton sweeps (default 0 = auto: up to 10, capped by free RAM / VRAM)
  --from STAGE            start at STAGE (earlier stages are assumed done)
  --only STAGE[,STAGE]    run only these stages
  --force                 re-run stages even if marked done
  --check-only            run the prerequisite checks and exit
  --list-stages           print the stage names and exit
  -h, --help              this text
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
say() { printf '\n==> %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE=smoke ;;
    --quick) MODE=quick ;;
    --full) MODE=full ;;
    --clock-lock) LOCK="${2:?}"; shift ;;
    --allow-unlocked) ALLOW_UNLOCKED=1 ;;
    --with-bundle) WITH_BUNDLE=1 ;;
    --skip-validation) SKIP_VALIDATION=1 ;;
    --run-dir) RUN_DIR="${2:?}"; shift ;;
    --resume) RUN_DIR="${2:?}"; MODE=""; shift ;;
    --venv) VENV="${2:?}"; shift ;;
    --python) PYTHON_BIN="${2:?}"; shift ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --workers) WORKERS="${2:?}"; shift ;;
    --from) FROM="${2:?}"; shift ;;
    --only) ONLY="${2:?}"; shift ;;
    --force) FORCE=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    --list-stages) printf '%s\n' "${STAGES[@]}"; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

# ------------------------------------------------------------------ run directory and mode
if [[ -n "$RUN_DIR" && -z "$MODE" ]]; then
  [[ -f "$RUN_DIR/mode" ]] || die "$RUN_DIR is not a train_best run directory (no 'mode' file)"
  MODE="$(<"$RUN_DIR/mode")"
fi
[[ -n "$RUN_DIR" ]] || RUN_DIR="$REPO/pipeline/runs/train_best-$MODE-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"/{reports,logs,stamps,models}
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
REPORTS="$RUN_DIR/reports"
echo "$MODE" > "$RUN_DIR/mode"
[[ "$LOCK" == "none" && "$ALLOW_UNLOCKED" == 0 ]] && die "--clock-lock none needs --allow-unlocked"

# TUNE_PS / HELD_PS / TAIL_PS = shapes per suite in the sweep subsets (1000 = all); LT_SHAPES = cuBLASLt screen shapes (0 = all 652, spread
# over the size-sorted set otherwise); DEEP_S = seconds per shape of the cuBLASLt enumerate-and-screen baseline (B3+).
case "$MODE" in
  smoke) TUNE_PS=3;    HELD_PS=3;    TAIL_PS=3;    LT_SHAPES=12; DEEP_S=2;  CHECK_SUITE=S1 ;;
  quick) TUNE_PS=8;    HELD_PS=12;   TAIL_PS=30;   LT_SHAPES=160; DEEP_S=10; CHECK_SUITE=edge ;;
  full)  TUNE_PS=1000; HELD_PS=1000; TAIL_PS=1000; LT_SHAPES=0;  DEEP_S=20; CHECK_SUITE=edge ;;
  *) die "unknown mode '$MODE'" ;;
esac

if [[ "$SKIP_INSTALL" == 1 ]]; then PY="$(command -v python)" || die "no 'python' on PATH"; else PY="$VENV/bin/python"; fi

# One pipeline at a time (shared with pipeline/run.sh and zero_shot.sh): overlapping GPU jobs contaminate measurements.
exec 9>"$REPO/pipeline/.lock"
flock -n 9 || die "another pipeline run holds pipeline/.lock; never run two GPU jobs at once"

trap 'echo; echo "interrupted; stopping child processes" >&2; trap - TERM; kill 0' INT TERM

tune() { "$PY" -u -m tune --clock-lock "$LOCK" "$@"; }
zs() { "$PY" -u -m analysis.zero_shot --clock-lock "$LOCK" "$@"; }

# The data directory is named by the hardware id (analysis.zero_shot names its report after it). Resolved once and remembered for --resume.
DATA=""; HW=""
set_paths() {
  [[ -f "$RUN_DIR/hw_id" ]] || "$PY" -c 'from tune.env import hw_id; print(hw_id())' | tail -1 > "$RUN_DIR/hw_id"
  HW="$(<"$RUN_DIR/hw_id")"
  [[ -n "$HW" ]] || die "could not determine the hardware id (is the GPU visible to torch?)"
  DATA="$RUN_DIR/data/$HW"
  mkdir -p "$DATA"
  # Keep every cache the run writes inside the run directory: the cuBLASLt deep-search cache would otherwise land in the repo's data/<hw>/.
  export TUNE_LT_SEARCH_DIR="$DATA/lt_search"
  # The learned components are read from this run's own files, never from data/<hw>/ or the 3080 snapshot.
  export TUNE_LT_RANK="$DATA/lt_rank" TUNE_ROUTER="$DATA/router"
  [[ "$LOCK" == "none" ]] || export TUNE_CLOCK_LOCK="$LOCK"
}

# ------------------------------------------------------------------ checks
gpu_idle_check() {
  local apps
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "${apps//[[:space:]]/}" ]]; then
    echo "The GPU is not idle; compute processes found:" >&2
    echo "$apps" >&2
    die "stop them (or kill orphaned sweep workers by PID) and re-run with --resume $RUN_DIR"
  fi
}

check_prereqs() {
  say "Prerequisite checks"
  [[ "$(uname -s)" == Linux ]] || die "Linux only"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver (>= 580 for the CUDA 13.0 wheels)"
  local info drv
  info="$(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader | head -1)"
  echo "GPU: $info"
  drv="$(cut -d, -f2 <<<"$info" | tr -d ' ' | cut -d. -f1)"
  [[ "$drv" =~ ^[0-9]+$ && "$drv" -ge 580 ]] || die "driver ${drv:-?} is too old for the cu130 wheels (need >= 580)"
  [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -gt 1 ]] &&
    echo "note: several GPUs found; the harness uses device 0 (set CUDA_VISIBLE_DEVICES to choose)"
  command -v gcc >/dev/null || command -v cc >/dev/null || die "no C compiler (Triton builds its launcher with gcc): apt install build-essential"
  command -v git >/dev/null || die "git not found"
  # 9a2e9ab keeps picks the screen cut as `slow` in the statistics; earlier checkouts silently drop the bad picks.
  git merge-base --is-ancestor 9a2e9ab HEAD 2>/dev/null || die "this checkout does not contain commit 9a2e9ab (git pull)"
  local pyexe="$PYTHON_BIN"
  [[ "$SKIP_INSTALL" == 1 ]] && pyexe="$PY"
  command -v "$pyexe" >/dev/null || die "interpreter '$pyexe' not found (use --python)"
  "$pyexe" - <<'EOF' || die "Python >= 3.10 with development headers is required (apt install python3-dev python3-venv)"
import sys, sysconfig, os
assert sys.version_info >= (3, 10), sys.version
inc = sysconfig.get_paths()["include"]
assert os.path.exists(os.path.join(inc, "Python.h")), f"Python.h missing in {inc}"
EOF
  [[ "$SKIP_INSTALL" == 1 ]] || "$pyexe" -c "import venv, ensurepip" 2>/dev/null || die "python venv/ensurepip missing: apt install python3-venv"
  local free_gb
  free_gb="$(df -BG --output=avail "$RUN_DIR" | tail -1 | tr -dc 0-9)"
  [[ "$free_gb" -ge 20 ]] || die "only ${free_gb} GB free under $RUN_DIR; need >= 20 GB (screens, stage-2 raw samples; the optional bundle adds ~2 GB)"
  gpu_idle_check
  echo "Prerequisites OK. Clock lock: declared '$LOCK' (verified after the env stage)."
}

# ------------------------------------------------------------------ setup stages
stage_preflight() { check_prereqs; }

stage_install() {
  if [[ "$SKIP_INSTALL" == 1 ]]; then
    echo "--skip-install: checking pinned versions in $PY"
    "$PY" - <<'EOF'
import importlib.metadata as md, re, sys
bad = []
for line in open("requirements.txt"):
    m = re.match(r"([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
    if not m:
        continue
    name, want = m.groups()
    try:
        have = md.version(name).split("+")[0]
    except md.PackageNotFoundError:
        bad.append(f"{name}: missing (want {want})"); continue
    if have != want:
        bad.append(f"{name}: {have} (want {want})")
print("\n".join(bad) or "all pinned versions match")
sys.exit(1 if bad else 0)
EOF
  else
    if [[ ! -x "$VENV/bin/python" ]]; then "$PYTHON_BIN" -m venv "$VENV"; fi
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -r requirements.txt \
      --extra-index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
  fi
  "$PY" - <<'EOF'
import torch, triton, lightgbm
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("torch", torch.__version__, "triton", triton.__version__, "lightgbm", lightgbm.__version__,
      "| GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
EOF
  { echo "commit: $(git rev-parse HEAD) dirty: $(git status --porcelain --untracked-files=no | wc -l) files"
    "$PY" -m pip freeze; } > "$RUN_DIR/environment.txt"
}

# env.json is read by the analysis code (the GPU numbers behind the rankers' fill features); also where the declared lock is cross-checked.
stage_env() {
  tune env --out "$DATA"
  "$PY" - "$DATA/env.json" "$LOCK" "$ALLOW_UNLOCKED" <<'EOF'
import json, sys
env = json.load(open(sys.argv[1])); lock, allow = sys.argv[2], sys.argv[3] == "1"
cl = env["clock_lock"]
print("clock_lock:", {k: cl.get(k) for k in ("declared", "probe", "state", "lock_mhz", "declared_matches_probe")})
ok = cl.get("state") == "locked" and cl.get("declared_matches_probe") is True
if lock == "none":
    ok = cl.get("state") == "unlocked"
if not ok and not allow:
    sys.exit(f"The GPU clock does not look locked at {lock} MHz. Lock it first:\n"
             f"    sudo nvidia-smi -lgc {lock},{lock}        (undo later with: sudo nvidia-smi -rgc)\n"
             f"then re-run with --resume. Or pass --allow-unlocked to measure unlocked.")
if not ok:
    print("WARNING: unlocked / unverified clocks; compare regret and ratios within this GPU, not times with the 3080 study.")
print("GPU:", env["gpu"]["name"], "| SMs:", env["gpu"]["sm_count"], "| compute capability:", env["gpu"]["compute_capability"])
EOF
}

stage_check() { tune check --suite "$CHECK_SUITE" --backends triton:handpicked,triton_tutorial,torch --out "$DATA"; }
stage_roofline() { tune roofline --out "$DATA"; }

# ------------------------------------------------------------------ Triton screens and rank v1
stage_tune_sweep() { tune sweep --split tune --per-suite "$TUNE_PS" --workers "$WORKERS" --out "$DATA"; }

# The oracle and the held-out shapes of every later score. Never used for fitting.
stage_heldout_sweep() { tune sweep --split heldout --per-suite "$HELD_PS" --workers "$WORKERS" --out "$DATA"; }

# Trivial selectors: reference baselines, and selector_picks.json / subset_lt50us.json, which `tune topk` reads.
stage_selectors() { tune selectors --data "$DATA" --report "$REPORTS"; }

# Rank v1: grouped-CV search, refit on every tune shape, diversification rule fit on out-of-fold scores -> data/<hw>/lgbm_rank.{txt,json}.
# Its json holds the hyperparameters v2 reuses.
stage_train_v1() { tune topk --data "$DATA" --report "$REPORTS"; }

# Tail suite: 150 extra TUNE shapes aimed at the regimes where the ranker's tail sits (suites/v1/tail.json, reports/week1.md 2026-10-31).
stage_tail_sweep() { tune sweep --suite tail --per-suite "$TAIL_PS" --screen-dir sweep_screen_tail --workers "$WORKERS" --out "$DATA"; }

# Rank v2 (CPU): analytic roofline features + near-top weights + the tail screen as extra tune data -> data/<hw>/lgbm_rank_v2.{txt,json}.
stage_train_v2() { tune ranker-v2 --data "$DATA"; }

# ------------------------------------------------------------------ cuBLASLt ranker
# Every valid cuBLASLt config of every TUNE shape (never a held-out one), subprocess per shape, resumable.
stage_lt_screen() {
  local lim=()
  [[ "$LT_SHAPES" -gt 0 ]] && lim=(--limit "$LT_SHAPES" --spread)
  "$PY" -u -m baselines.cublaslt_screen --out "$DATA/lt_screen" "${lim[@]}"
}

# CPU: grouped-CV table, then data/<hw>/lt_rank.{txt,json} (hw_id, this GPU's SM count and the cuBLASLt version in the json).
stage_lt_fit() {
  "$PY" -u -m analysis.lt_rank --screen-dir "$DATA/lt_screen" --save "$DATA/lt_rank"
  "$PY" - "$DATA/lt_rank.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"cuBLASLt ranker fit on {m['n_tune_shapes']} tune shapes of {m['hw_id']} ({m['sms']} SMs), cuBLASLt {m['cublaslt_version']}")
EOF
}

# ------------------------------------------------------------------ router
# Training data of the router: the full protocol on every tune shape, one session: torch, the screen's top-8 Triton configs, cuBLASLt top-16
# and the searched cuBLASLt (B3+). The file names are the ones analysis/router.py reads (they say lock1755 whatever lock was used here; the
# lock is recorded in every row).
tune_stage2_env() { export TUNE_SWEEP_DIR="$DATA/sweep_screen"; }
TUNE_BACKENDS_TAG="torch,triton:sweep:8,cublaslt:16"

stage_tune_stage2_cold() {
  tune_stage2_env
  tune benchmark --shapes-file "$DATA/sweep_screen/subset.json" --backends "$TUNE_BACKENDS_TAG,cublaslt_deep:$DEEP_S" \
    --cache-modes cold --out-name results_lock1755_tune477.jsonl --out "$DATA"
}

# The tune shapes whose screen oracle is < 50 us are re-timed graph-amortized (--graph-batch 50), as for the held-out shapes.
stage_tune_stage2_graph() {
  tune_stage2_env
  "$PY" - "$DATA" <<'EOF'
import json, sys
from pathlib import Path
from analysis.zero_shot import SMALL_US, load_screen
d = Path(sys.argv[1]) / "sweep_screen"
screen = load_screen(d)
base = json.loads((d / "subset.json").read_text())
small = [s for s in base["shapes"] if screen.get((s["M"], s["N"], s["K"])) and min(screen[(s["M"], s["N"], s["K"])].values()) < SMALL_US]
(d / "subset_lt50us_477.json").write_text(json.dumps({**base, "note": f"tune shapes whose screen oracle is < {SMALL_US:.0f} us", "n": len(small), "shapes": small}))
print(f"{len(small)} of {len(base['shapes'])} tune shapes have a screen oracle under {SMALL_US:.0f} us")
EOF
  tune benchmark --shapes-file "$DATA/sweep_screen/subset_lt50us_477.json" --backends "$TUNE_BACKENDS_TAG,cublaslt_deep:$DEEP_S" \
    --cache-modes hot --graph-batch 50 --out-name results_lock1755_tune477_graph.jsonl --out "$DATA"
}

# CPU: LightGBM classifier P(Triton's best beats the cuBLASLt value by > the tie margin); the skip threshold comes from a rule fixed in
# analysis/router.py (out-of-fold mean loss <= 0.1%) -> data/<hw>/router.{txt,json}.
stage_router_fit() {
  # The router predicts "Triton's best beats cuBLASLt by more than the tie margin". On a GPU where it (almost) never does (the 2060: Triton
  # compiles to FMAs, 0.44x of B3+) there is nothing to learn: the fit would crash on a single class, so no router is trained, and
  # select_kernel then runs both halves, as it does on any GPU without a router model.
  local npos
  npos="$("$PY" - "$DATA" <<'EOF' | tail -1
import sys
from pathlib import Path
from analysis.router import tune_table
shapes, X, lt, tri, eps = tune_table(Path(sys.argv[1]))
print(int((tri * eps < lt).sum()))
EOF
)"
  echo "Triton beats B3+ by more than the tie margin on $npos tune shapes"
  if [[ "${npos:-0}" -lt 8 ]]; then
    echo "fewer than 8 positives: no router is trained for this GPU (the search will run both halves)"
    echo "Triton wins on $npos tune shapes" > "$DATA/router.skipped"
    return 0
  fi
  "$PY" -u -m analysis.router fit --data "$DATA" --save "$DATA/router"
  "$PY" - "$DATA/router.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"router fit on {m['n_tune_shapes']} tune shapes of {m['hw_id']}: out-of-fold AUC {m['oof_auc']:.3f}, skip threshold {m['threshold']}")
EOF
}

# ------------------------------------------------------------------ optional bundle
stage_bundle() {
  [[ "$WITH_BUNDLE" == 1 ]] || { echo "skipped (add --with-bundle)"; return; }
  tune bundle --data "$DATA" build --bundle-dir "$RUN_DIR/bundle" --workers "$WORKERS"
  tune bundle --data "$DATA" verify --bundle-dir "$RUN_DIR/bundle"
  tune bundle --data "$DATA" pack --bundle-dir "$RUN_DIR/bundle" --archive "$RUN_DIR/models/bundle.tar.xz"
}

# ------------------------------------------------------------------ validation (held-out)
validated() { [[ "$SKIP_VALIDATION" == 0 ]]; }

# Rank v1 and v2 (each `as_is`: they were fit on this GPU, so there is no source GPU to transfer from), diversified top-k + the short online
# search, cuBLASLt heuristic top-16 + the learned cuBLASLt ranker's top-5 in the same pool. Fresh Triton cache: compile cost is a true first use.
stage_validate_run() {
  validated || { echo "skipped (--skip-validation)"; return; }
  zs run --out "$DATA" --ranker v1="$DATA/lgbm_rank" v2="$DATA/lgbm_rank_v2" --variants as_is --k 5 --lt-n 16 --lt-rank 5 --fresh-cache
}

# CPU. The screen oracle is Triton-only, so this scores the Triton-only choice; k = 3 is the shipped default tier, k = 5 the frequent tier.
stage_validate_score() {
  validated || { echo "skipped (--skip-validation)"; return; }
  zs score --data "$DATA" --k 3 --report "$REPORTS/k3"
  zs score --data "$DATA" --k 5 --report "$REPORTS/k5"
}

# Stage 2: every pick re-timed with the full protocol in one session next to torch, the screen's best tiles, B2/B3, the ranker's own cuBLASLt
# picks and B3+. `picks` writes the Triton picks file and the list of held-out shapes under 50 us (CPU).
VAL_BACKENDS_TAIL="cublaslt:16,cublaslt_rank:5:skip16"   # skip16: the same list the online search timed (else ranker picks have no stage-2 time)
stage_validate_stage2_cold() {
  validated || { echo "skipped (--skip-validation)"; return; }
  zs picks --data "$DATA"
  export TUNE_SWEEP_DIR="$DATA/sweep_screen_heldout" TUNE_PICKS="$DATA/zero_shot_stage2_picks.json"
  tune benchmark --shapes-file "$DATA/sweep_screen_heldout/subset.json" \
    --backends "torch,triton:sweep:8,triton:picks,$VAL_BACKENDS_TAIL,cublaslt_deep:$DEEP_S" \
    --cache-modes cold --out-name results_zero_shot.jsonl --out "$DATA"
}

stage_validate_stage2_graph() {
  validated || { echo "skipped (--skip-validation)"; return; }
  export TUNE_SWEEP_DIR="$DATA/sweep_screen_heldout" TUNE_PICKS="$DATA/zero_shot_stage2_picks.json"
  local n
  n="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['n'])" "$DATA/zero_shot_lt50us.json")"
  if [[ "$n" -gt 0 ]]; then
    tune benchmark --shapes-file "$DATA/zero_shot_lt50us.json" \
      --backends "torch,triton:sweep:8,triton:picks,$VAL_BACKENDS_TAIL,cublaslt_deep:$DEEP_S" \
      --cache-modes hot --graph-batch 50 --out-name results_zero_shot_graph.jsonl --out "$DATA"
  else
    echo "no held-out shape under 50 us on this GPU: no graph pass"
    : > "$DATA/results_zero_shot_graph.jsonl"
  fi
}

# CPU. --report keeps ranker_compare from writing the repo's reports/ranker_compare.md (the 3080's). For these runs "+ LT16" means
# "+ LT16 + the ranker's top-5" (the rows' `choice` is the whole pool).
stage_validate_compare() {
  validated || { echo "skipped (--skip-validation)"; return; }
  "$PY" -m analysis.ranker_compare --data "$DATA" --report "$REPORTS" --results results_zero_shot.jsonl \
    --graph-results results_zero_shot_graph.jsonl --online v1=zero_shot_v1_as_is.jsonl v2=zero_shot_v2_as_is.jsonl
}

# ------------------------------------------------------------------ packaging
# The four model families, ready to copy into a checkout. models/ never overwrites anything by itself.
stage_package() {
  local f missing=0
  for f in lgbm_rank lgbm_rank_v2 lt_rank router; do
    if [[ -f "$DATA/$f.txt" && -f "$DATA/$f.json" ]]; then cp "$DATA/$f.txt" "$DATA/$f.json" "$RUN_DIR/models/"
    elif [[ "$f" == router && -f "$DATA/router.skipped" ]]; then echo "router: none trained ($(<"$DATA/router.skipped"))"
    else echo "MISSING: $DATA/$f.{txt,json} (its stage did not finish)"; missing=1; fi
  done
  cp "$DATA/env.json" "$RUN_DIR/models/env.json"
  (cd "$RUN_DIR/models" && md5sum ./*.txt ./*.json > MD5SUMS)
  cat > "$RUN_DIR/models/INSTALL.md" <<EOF
# Models trained on $HW

Run: $(basename "$RUN_DIR") (mode $MODE), clock lock $LOCK, commit $(git rev-parse --short HEAD).

Install into a checkout of this repo on the same machine (data/ is gitignored; check MD5SUMS after copying):

    mkdir -p data/$HW && for f in lgbm_rank lgbm_rank_v2 lt_rank router; do [ -f models/\$f.txt ] && cp models/\$f.txt models/\$f.json data/$HW/; done

\`tune.online.select_kernel\` then finds them (\`data/<hw>/lt_rank\`, \`data/<hw>/router\`; the Triton ranker is passed by the caller:
\`analysis.learned.Ranker.load('data/$HW/lgbm_rank_v2')\`). Every component refuses a model whose hw_id differs from the GPU it runs on.
A bundle built with --with-bundle is models/bundle.tar.xz: copy it to kernels/bundle.tar.xz to replace the committed archive (\`python -m tune bundle status\` says whether the committed one still matches).
EOF
  [[ "$missing" == 0 ]] || die "some models are missing; see above"
  cat "$RUN_DIR/models/MD5SUMS"
}

stage_summary() {
  "$PY" - "$RUN_DIR" "$MODE" "$HW" "$SKIP_VALIDATION" <<'EOF'
import json, sys
from pathlib import Path
from analysis.zero_shot import fmt
run, mode, hw, skip = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1"
data, rep = run / "data" / hw, run / "reports"
env = json.loads((data / "env.json").read_text())
cl = env["clock_lock"]
out = [f"# Full training of the shipped selector, `{run.name}` ({mode} mode)\n",
       f"GPU {env['gpu']['name']} ({hw}, {env['gpu']['sm_count']} SMs), driver {env['driver']}, clock lock: {cl.get('state')} "
       f"{cl.get('lock_mhz') or ''}. Commit {env['harness_git']['commit'][:7]}{' (dirty)' if env['harness_git']['dirty'] else ''}.\n",
       "## Models (models/, INSTALL.md says where they go)\n", "| model | fit on | notes |", "|---|---|---|"]
for stem, label in (("lgbm_rank", "rank v1 (Triton)"), ("lgbm_rank_v2", "rank v2 (Triton, shipped)"), ("lt_rank", "cuBLASLt ranker"), ("router", "family router")):
    p = data / f"{stem}.json"
    if not p.exists():
        out.append(f"| {label} | none | " + ("not trained: Triton (almost) never beats B3+ on this GPU, the search runs both halves" if stem == "router" and (data / "router.skipped").exists() else "stage did not finish") + " |")
        continue
    m = json.loads(p.read_text())
    note = {"lgbm_rank": lambda: f"pool {len(m['pool'])} configs, hp {m['hp']}",
            "lgbm_rank_v2": lambda: f"pool {len(m['pool'])} configs, extra screens {m['meta'].get('extra_screens')}",
            "lt_rank": lambda: f"{m['sms']} SMs, cuBLASLt {m['cublaslt_version']}",
            "router": lambda: f"out-of-fold AUC {m['oof_auc']:.3f}, skip threshold {m['threshold']}"}[stem]()
    n = m.get("n_tune_shapes") or (m.get("meta") or {}).get("n_tune_shapes")
    out.append(f"| {label} | {n} tune shapes | {note} |")
out.append("")
if not skip:
    for k in (3, 5):
        p = rep / f"k{k}" / f"zero_shot_{hw}_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        out.append(f"## Held-out, Triton-only online search, k = {k}: regret vs this GPU's own oracle, median / p95 / p99 (mean)\n")
        out.append(f"{s['n_screen_shapes']} held-out shapes in the screen.\n")
        out.append("| model | shapes | all | >= 50 us | < 50 us |\n|---|---|---|---|---|")
        for v, d in s["variants"].items():
            sc = d["scores"][f"online-{k}"]
            out.append(f"| {v} | {d['n_shapes']} | " + " | ".join(fmt(sc[g]) for g in ("all", ">= 50 us", "< 50 us")) + " |")
        out.append("")
    out.append("The joint choice (Triton top-k + cuBLASLt top-16 + ranker top-5) against B2 / B3 / B3+: `reports/ranker_compare.md`. Read the "
               "`regret > 50%` and `unscored` columns of the zero-shot reports first (docs/zero_shot_runbook.md, section 5).\n")
out.append("Not run here: the noise study (`python -m tune --clock-lock X noise --rounds 3 --repeats 50 --n-allocs 3`, needed before quoting "
           "a tie margin for this GPU) and the router's held-out score (`python -m analysis.router score`, which reads a stage-2 session of "
           "a run with this GPU's models). Then add a dated update to `reports/week1.md`.\n")
for p in sorted(rep.rglob("*.md")):
    out.append(f"- `reports/{p.relative_to(rep)}`")
(run / "SUMMARY.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
EOF
}

# ------------------------------------------------------------------ driver
is_gpu_stage() { local s; for s in "${GPU_STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }

selected() {  # is stage $1 selected by --from / --only?
  local s="$1"
  if [[ -n "$ONLY" ]]; then [[ ",$ONLY," == *",$s,"* ]]; return; fi
  if [[ -n "$FROM" ]]; then
    local seen=0 t
    for t in "${STAGES[@]}"; do [[ "$t" == "$FROM" ]] && seen=1; [[ "$t" == "$s" ]] && return $((1 - seen)); done
  fi
  return 0
}

if [[ -n "$FROM" ]]; then printf '%s\n' "${STAGES[@]}" | grep -qx "$FROM" || die "unknown stage '$FROM' (see --list-stages)"; fi
if [[ "$CHECK_ONLY" == 1 ]]; then check_prereqs; exit 0; fi

say "Full training, mode: $MODE | run dir: $RUN_DIR | clock lock: $LOCK | bundle: $WITH_BUNDLE | validation: $((1 - SKIP_VALIDATION))"
T_START=$SECONDS
for s in "${STAGES[@]}"; do
  selected "$s" || continue
  stamp="$RUN_DIR/stamps/$s.done"
  if [[ -f "$stamp" && "$FORCE" == 0 ]]; then echo "[skip] $s (done $(<"$stamp"))"; continue; fi
  # The install stage creates the interpreter the other stages use.
  if [[ "$s" != install && "$s" != preflight && ! -x "$PY" ]]; then die "$PY not found: run the install stage first"; fi
  if [[ "$s" != install && "$s" != preflight ]]; then set_paths; fi
  if is_gpu_stage "$s"; then gpu_idle_check; fi
  say "stage: $s"
  log="$RUN_DIR/logs/$s.log"
  t0=$SECONDS
  set +e
  "stage_$s" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "FAILED: stage '$s' (exit $rc). Log: $log" >&2
    echo "Fix the problem, then continue with: pipeline/train_best.sh --resume $RUN_DIR" >&2
    exit "$rc"
  fi
  date -Is > "$stamp"
  printf '[done] %s in %dm%02ds\n' "$s" $(((SECONDS - t0) / 60)) $(((SECONDS - t0) % 60))
done
printf '\nTraining finished in %dh%02dm. Summary: %s/SUMMARY.md, models: %s/models/\n' $(((SECONDS - T_START) / 3600)) \
  $((((SECONDS - T_START) % 3600) / 60)) "$RUN_DIR" "$RUN_DIR"
