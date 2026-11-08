#!/usr/bin/env bash
# One-command zero-shot transfer: install -> checks -> oracle (this GPU's own held-out screen) -> run the saved RTX 3080
# rankers (lgbm_rank v1 and v2, no refit) -> score -> stage 2 next to B2/B3/B3+ -> comparison. Long form, how to read the
# tables and the pitfalls: docs/zero_shot_runbook.md. Prerequisites and style are those of pipeline/run.sh.
#
#   pipeline/zero_shot.sh [--smoke | --quick | --full] [options]
#
# Everything a run produces goes to pipeline/runs/zero_shot-<mode>-<timestamp>/ (data/<hw_id>/, reports/, logs/); the repo's
# own data/ and reports/ are never touched (the 3080 model files are only read). --resume <run dir> skips finished stages.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGES=(preflight install tests env lt_screen lt_fit sweep run score picks stage2_cold stage2_graph compare summary)
GPU_STAGES=(env lt_screen sweep run stage2_cold stage2_graph)

MODE=quick
LOCK=1755
ALLOW_UNLOCKED=0
RUN_DIR=""
VENV="$REPO/pipeline/.venv"
PYTHON_BIN="${PYTHON:-python3}"
SKIP_INSTALL=0
SKIP_TESTS=0
WORKERS=0
FROM=""
ONLY=""
CHECK_ONLY=0
FORCE=0
MODELS=""
ORACLE=""
K=5
LT_N=16
LT_MODE=off
LT_RANK=5
LT_MODELS=

usage() {
  cat <<EOF
Usage: pipeline/zero_shot.sh [mode] [options]

Modes (default --quick):
  --smoke           plumbing check on 3 shapes per suite (results are not meaningful)
  --quick           the 27-shape held-out subset (the same 27 shapes as the 2060 / 3090 runs)
  --full            all 86 held-out shapes

Options:
  --models DIR            where lgbm_rank.{txt,json} and lgbm_rank_v2.{txt,json} (the 3080's) are. Default: data/src3080 if it
                          exists, else data/rtx3080_sm86_68sms_10gb. They are only read, never copied or overwritten
  --oracle DIR            a finished sweep_screen_heldout/ of THIS GPU to copy in instead of sweeping again (hours saved);
                          the sweep stage is then skipped. The screen's shape subset decides what is scored
  --k N                   Triton candidates per shape in the online search, stored for every k' <= N (default $K)
  --lt-n N                also short-time cuBLASLt's heuristic top N in the same pool, joint selection (default $LT_N; 0 = off)
  --lt-mode off|zero_shot|retrain
                          the learned cuBLASLt ranker (baselines/cublaslt_rank.py) in the joint search; default off (heuristic top-N only).
                            zero_shot  the 3080's lt_rank, no refit, as_is / native SM count like the Triton rankers (--lt-models DIR). Means
                                       something only on the same architecture (RTX 3090, sm_86); on a Turing card expect it to fail
                            retrain    screen every cuBLASLt config of the tune shapes on THIS GPU (stage lt_screen: ~70 min on the 3080,
                                       smoke 12 / quick 160 shapes, full 652), fit data/<hw>/lt_rank on it (stage lt_fit, CPU) and use that.
                                       The right choice for the RTX 2060 (sm_75); the 3090 can do either, and both is the comparison
  --lt-rank M             cuBLASLt ranker picks per search (default $LT_RANK; needs --lt-n > 0 and --lt-mode)
  --lt-models DIR         where the 3080's lt_rank.{txt,json} are for --lt-mode zero_shot (default: --models DIR)
  --clock-lock MHZ|none   SM clock you locked beforehand with 'sudo nvidia-smi -lgc MHZ,MHZ' (default $LOCK). The script never
                          calls sudo. 'none' = run unlocked (needs --allow-unlocked); then compare regret and ratios within
                          this GPU only, not times with the 3080's 1755 MHz runs
  --allow-unlocked        continue even if the GPU clock is not locked
  --run-dir DIR           output directory (default pipeline/runs/zero_shot-<mode>-<timestamp>)
  --resume DIR            continue a previous run directory; finished stages are skipped (mode is read from it)
  --venv DIR              virtualenv location (default pipeline/.venv)
  --python BIN            interpreter used to create the venv (default python3, or \$PYTHON)
  --skip-install          use the current 'python' on PATH as is (checks the pinned versions, installs nothing)
  --skip-tests            skip the CPU-only pytest stage
  --workers N             parallel compile workers for the oracle sweep (default 0 = auto: up to 10, capped by free RAM / VRAM)
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
    --models) MODELS="${2:?}"; shift ;;
    --oracle) ORACLE="${2:?}"; shift ;;
    --k) K="${2:?}"; shift ;;
    --lt-n) LT_N="${2:?}"; shift ;;
    --lt-mode) LT_MODE="${2:?}"; shift ;;
    --lt-rank) LT_RANK="${2:?}"; shift ;;
    --lt-models) LT_MODELS="${2:?}"; shift ;;
    --clock-lock) LOCK="${2:?}"; shift ;;
    --allow-unlocked) ALLOW_UNLOCKED=1 ;;
    --run-dir) RUN_DIR="${2:?}"; shift ;;
    --resume) RUN_DIR="${2:?}"; MODE=""; shift ;;
    --venv) VENV="${2:?}"; shift ;;
    --python) PYTHON_BIN="${2:?}"; shift ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
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
[[ "$K" =~ ^[0-9]+$ && "$K" -ge 3 ]] || die "--k must be an integer >= 3 (the scores are at k = 3 and k = $K)"
[[ "$LT_N" =~ ^[0-9]+$ ]] || die "--lt-n must be an integer"
[[ "$LT_MODE" =~ ^(off|zero_shot|retrain)$ ]] || die "--lt-mode must be off, zero_shot or retrain"
[[ "$LT_RANK" =~ ^[0-9]+$ ]] || die "--lt-rank must be an integer"
[[ "$LT_MODE" == off || ( "$LT_N" -gt 0 && "$LT_RANK" -gt 0 ) ]] || die "--lt-mode $LT_MODE needs --lt-n > 0 and --lt-rank > 0"
[[ "$LT_MODE" == off ]] && LT_RANK=0

# ------------------------------------------------------------------ run directory and mode
if [[ -n "$RUN_DIR" && -z "$MODE" ]]; then
  [[ -f "$RUN_DIR/mode" ]] || die "$RUN_DIR is not a zero-shot run directory (no 'mode' file)"
  MODE="$(<"$RUN_DIR/mode")"
fi
[[ -n "$RUN_DIR" ]] || RUN_DIR="$REPO/pipeline/runs/zero_shot-$MODE-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"/{reports,logs,stamps}
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
REPORTS="$RUN_DIR/reports"
echo "$MODE" > "$RUN_DIR/mode"
[[ "$LOCK" == "none" && "$ALLOW_UNLOCKED" == 0 ]] && die "--clock-lock none needs --allow-unlocked"

# LT_SHAPES = tune shapes of the cuBLASLt screen for --lt-mode retrain (0 = all 652; spread over the size-sorted set).
# PER_SUITE = held-out shapes per suite in the oracle subset (1000 = all 86; 12 = the 27-shape subset). DEEP_S = seconds per
# shape of the cuBLASLt enumerate-and-screen baseline (B3+) in stage 2.
case "$MODE" in
  smoke) PER_SUITE=3;    DEEP_S=2;  LT_SHAPES=12 ;;
  quick) PER_SUITE=12;   DEEP_S=20; LT_SHAPES=160 ;;
  full)  PER_SUITE=1000; DEEP_S=20; LT_SHAPES=0 ;;
  *) die "unknown mode '$MODE'" ;;
esac

if [[ "$SKIP_INSTALL" == 1 ]]; then PY="$(command -v python)" || die "no 'python' on PATH"; else PY="$VENV/bin/python"; fi

# One pipeline at a time (shared with pipeline/run.sh): overlapping GPU jobs contaminate measurements.
exec 9>"$REPO/pipeline/.lock"
flock -n 9 || die "another pipeline run holds pipeline/.lock; never run two GPU jobs at once"

trap 'echo; echo "interrupted; stopping child processes" >&2; trap - TERM; kill 0' INT TERM

tune() { "$PY" -u -m tune --clock-lock "$LOCK" "$@"; }
zs() { "$PY" -u -m analysis.zero_shot --clock-lock "$LOCK" "$@"; }

# The data directory is named by the hardware id: analysis.zero_shot names its report after it (zero_shot_<hw_id>.md) and
# analysis.cross_gpu takes it as a side. Resolved once (needs the interpreter) and remembered for --resume.
DATA=""; HW=""
set_paths() {
  [[ -f "$RUN_DIR/hw_id" ]] || "$PY" -c 'from tune.env import hw_id; print(hw_id())' | tail -1 > "$RUN_DIR/hw_id"
  HW="$(<"$RUN_DIR/hw_id")"
  [[ -n "$HW" ]] || die "could not determine the hardware id (is the GPU visible to torch?)"
  DATA="$RUN_DIR/data/$HW"
  mkdir -p "$DATA"
}

# ------------------------------------------------------------------ models
resolve_models() {
  if [[ -z "$MODELS" ]]; then
    local c
    for c in "$REPO/data/src3080" "$REPO/data/rtx3080_sm86_68sms_10gb"; do
      [[ -f "$c/lgbm_rank.txt" ]] && { MODELS="$c"; break; }
    done
    [[ -n "$MODELS" ]] || die "no 3080 model files found; copy lgbm_rank.{txt,json} and lgbm_rank_v2.{txt,json} into data/src3080/ or pass --models DIR"
  fi
  MODELS="$(cd "$MODELS" 2>/dev/null && pwd)" || die "--models directory does not exist"
  local f
  for f in lgbm_rank.txt lgbm_rank.json lgbm_rank_v2.txt lgbm_rank_v2.json; do
    [[ -f "$MODELS/$f" ]] || die "$MODELS/$f is missing (the four 3080 model files are needed; data/ is gitignored, copy them by hand)"
  done
}

resolve_lt_models() {
  [[ "$LT_MODE" == zero_shot ]] || return 0
  [[ -n "$LT_MODELS" ]] || LT_MODELS="$MODELS"
  LT_MODELS="$(cd "$LT_MODELS" 2>/dev/null && pwd)" || die "--lt-models directory does not exist"
  local f
  for f in lt_rank.txt lt_rank.json; do
    [[ -f "$LT_MODELS/$f" ]] || die "$LT_MODELS/$f is missing (--lt-mode zero_shot needs the 3080's cuBLASLt ranker; it is in data/<hw>/ or reports/snapshot/lock1755/)"
  done
}

# The cuBLASLt ranker's model for the run / stage-2 processes: retrain = the one fitted in this run, zero_shot = passed explicitly.
lt_env() {
  unset TUNE_LT_RANK TUNE_LT_RANK_FOREIGN TUNE_LT_RANK_SMS
  [[ "$LT_MODE" == retrain ]] && export TUNE_LT_RANK="$DATA/lt_rank"
  return 0
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
  [[ "$free_gb" -ge 20 ]] || die "only ${free_gb} GB free under $RUN_DIR; need >= 20 GB"
  resolve_models
  echo "3080 models: $MODELS"
  md5sum "$MODELS/lgbm_rank.txt" "$MODELS/lgbm_rank_v2.txt"
  if [[ "$LT_MODE" == zero_shot ]]; then resolve_lt_models; echo "3080 cuBLASLt ranker: $LT_MODELS"; md5sum "$LT_MODELS/lt_rank.txt"; fi
  [[ "$LT_MODE" != retrain ]] || echo "cuBLASLt ranker: will be retrained on this GPU ($LT_SHAPES tune shapes screened; 0 = all)"
  if [[ -n "$ORACLE" ]]; then
    [[ -f "$ORACLE/subset.json" ]] || die "--oracle $ORACLE has no subset.json (point it at a sweep_screen_heldout/ directory)"
  fi
  gpu_idle_check
  echo "Prerequisites OK. Clock lock: declared '$LOCK' (verified after the env stage)."
}

# ------------------------------------------------------------------ stages
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
    # torch is a +cu130 wheel; the extra index supplies it, PyPI everything else.
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

# CPU only: the zero-shot scoring and the cross-GPU comparison code.
stage_tests() {
  [[ "$SKIP_TESTS" == 1 ]] && { echo "skipped (--skip-tests)"; return; }
  "$PY" -m pytest -q tests/test_zero_shot.py tests/test_cross_gpu.py
}

# env.json is where the declared clock lock is cross-checked, and what analysis.cross_gpu reads for this GPU.
stage_env() {
  tune env --out "$DATA"
  resolve_models
  "$PY" - "$MODELS" <<'EOF'
import json, sys
for stem in ("lgbm_rank", "lgbm_rank_v2"):
    print(stem, "was fit on:", json.load(open(f"{sys.argv[1]}/{stem}.json"))["gpu"]["name"])
EOF
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
EOF
}

# --lt-mode retrain: screen every valid cuBLASLt config of the TUNE shapes on this GPU (never a held-out shape), subprocess per shape,
# resumable. Training data of this GPU's cuBLASLt ranker.
stage_lt_screen() {
  [[ "$LT_MODE" == retrain ]] || { echo "skipped (--lt-mode $LT_MODE)"; return; }
  [[ "$LOCK" == "none" ]] || export TUNE_CLOCK_LOCK="$LOCK"
  local lim=()
  [[ "$LT_SHAPES" -gt 0 ]] && lim=(--limit "$LT_SHAPES" --spread)
  "$PY" -u -m baselines.cublaslt_screen --out "$DATA/lt_screen" "${lim[@]}"
}

# CPU only: grouped-CV report and the saved model data/<hw>/lt_rank.{txt,json} (hw_id, SM count and cuBLASLt version in the json).
stage_lt_fit() {
  [[ "$LT_MODE" == retrain ]] || { echo "skipped (--lt-mode $LT_MODE)"; return; }
  "$PY" -u -m analysis.lt_rank --screen-dir "$DATA/lt_screen" --save "$DATA/lt_rank"
  "$PY" - "$DATA/lt_rank.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"cuBLASLt ranker fit on {m['n_tune_shapes']} tune shapes of {m['hw_id']} ({m['sms']} SMs), cuBLASLt {m['cublaslt_version']}")
EOF
}

# The oracle: this GPU's own screen of the held-out shapes (every pruned Triton config compiled, validated, timed).
# Never fit anything on these shapes. Resumes per shape.
stage_sweep() {
  if [[ -n "$ORACLE" ]]; then
    echo "--oracle: copying $ORACLE -> $DATA/sweep_screen_heldout (no sweep)"
    [[ -e "$DATA/sweep_screen_heldout" ]] || cp -r "$ORACLE" "$DATA/sweep_screen_heldout"
  else
    tune sweep --split heldout --per-suite "$PER_SUITE" --workers "$WORKERS" --out "$DATA"
  fi
  [[ -f "$DATA/sweep_screen_heldout/subset.json" ]] || die "no $DATA/sweep_screen_heldout/subset.json: the sweep did not produce a subset"
  # Scoring needs a complete oracle; a shape that did not fit in memory is recorded as skipped and retried on the next run.
  "$PY" - "$DATA/sweep_screen_heldout" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
want = json.loads((d / "subset.json").read_text())["n"]
done = sum(1 for p in d.glob("*_fp16.json") if json.loads(p.read_text()).get("complete"))
print(f"oracle: {done} of {want} held-out shapes complete")
if done < want:
    print("WARNING: incomplete oracle; shapes without a complete screen are not scored (re-run with --resume --from sweep to continue)")
EOF
}

# Rank, compile (empty Triton cache), validate and short-time on this GPU: v1/v2 x as_is/native, Triton top-K plus the
# cuBLASLt heuristic top-LT_N. Output: zero_shot_{v1,v2}_{as_is,native}.jsonl. Resumes per (shape, k).
stage_run() {
  resolve_models
  local lt=()
  lt_env
  case "$LT_MODE" in
    retrain)   lt=(--lt-rank "$LT_RANK") ;;
    zero_shot) resolve_lt_models; lt=(--lt-rank "$LT_RANK" --lt-rank-model "$LT_MODELS/lt_rank") ;;
  esac
  zs run --out "$DATA" --ranker v1="$MODELS/lgbm_rank" v2="$MODELS/lgbm_rank_v2" --k "$K" --lt-n "$LT_N" --fresh-cache "${lt[@]}"
}

# CPU only. One fixed report name per data directory, so k = 3 and k = 5 go to separate folders.
stage_score() {
  zs score --data "$DATA" --k 3 --report "$REPORTS/k3"
  [[ "$K" -ge 5 ]] && zs score --data "$DATA" --k 5 --report "$REPORTS/k5"
  return 0
}

# CPU only: writes zero_shot_stage2_picks.json (every pick as a backend candidate) and zero_shot_lt50us.json.
stage_picks() { zs picks --data "$DATA"; }

# Stage 2: every pick re-timed with the full protocol in one session with B1, the screen's best tiles, B2/B3 and B3+.
stage2_env() { export TUNE_SWEEP_DIR="$DATA/sweep_screen_heldout" TUNE_PICKS="$DATA/zero_shot_stage2_picks.json"; lt_env; }
STAGE2_BACKENDS="torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:$DEEP_S"
[[ "$LT_RANK" -gt 0 ]] && STAGE2_BACKENDS="torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_rank:$LT_RANK:skip16,cublaslt_deep:$DEEP_S"

# One benchmark pass per cuBLASLt-ranker variant. `cublaslt_rank:M` expands to whatever the ranker in this process proposes, so a
# zero-shot model needs one pass as_is and one native (the SM count differs, so may the picks); `benchmark` resumes per (shape,
# candidate key), so the second pass only times the picks the first did not propose. retrain / off: a single pass.
# usage: stage2_pass <benchmark args...>
stage2_pass() {
  local variants=(default) v
  [[ "$LT_MODE" == zero_shot ]] && { resolve_lt_models; variants=(as_is native); }
  for v in "${variants[@]}"; do
    if [[ "$v" != default ]]; then
      export TUNE_LT_RANK="$LT_MODELS/lt_rank" TUNE_LT_RANK_FOREIGN=1
      if [[ "$v" == native ]]; then export TUNE_LT_RANK_SMS=native; else unset TUNE_LT_RANK_SMS; fi
      echo "--- cuBLASLt ranker variant: $v"
    fi
    tune benchmark "$@"
  done
}

stage_stage2_cold() {
  stage2_env
  stage2_pass --shapes-file "$DATA/sweep_screen_heldout/subset.json" --backends "$STAGE2_BACKENDS" \
    --cache-modes cold --out-name results_zero_shot.jsonl --out "$DATA"
}

stage_stage2_graph() {
  stage2_env
  local n
  n="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['n'])" "$DATA/zero_shot_lt50us.json")"
  if [[ "$n" -gt 0 ]]; then
    stage2_pass --shapes-file "$DATA/zero_shot_lt50us.json" --backends "$STAGE2_BACKENDS" \
      --cache-modes hot --graph-batch 50 --out-name results_zero_shot_graph.jsonl --out "$DATA"
  else
    echo "no held-out shape under 50 us on this GPU: no graph pass"
    : > "$DATA/results_zero_shot_graph.jsonl"
  fi
}

# CPU only. --report keeps ranker_compare from writing the repo's reports/ranker_compare.md (the 3080's).
stage_compare() {
  local online=() m v
  for m in v1 v2; do for v in as_is native; do online+=("${m}_${v}=zero_shot_${m}_${v}.jsonl"); done; done
  "$PY" -m analysis.ranker_compare --data "$DATA" --report "$REPORTS" --results results_zero_shot.jsonl \
    --graph-results results_zero_shot_graph.jsonl --online "${online[@]}"
}

stage_summary() {
  "$PY" - "$RUN_DIR" "$MODE" "$HW" "$K" "$LT_N" <<'EOF'
import json, sys
from pathlib import Path
from analysis.zero_shot import fmt
run, mode, hw, k_max, lt_n = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
rep = run / "reports"
env = json.loads((run / "data" / hw / "env.json").read_text())
cl = env["clock_lock"]
out = [f"# Zero-shot run `{run.name}` ({mode} mode)\n",
       f"GPU {env['gpu']['name']} ({env['hw_id']}), driver {env['driver']}, clock lock: {cl.get('state')} "
       f"{cl.get('lock_mhz') or ''}. Commit {env['harness_git']['commit'][:7]}"
       f"{' (dirty)' if env['harness_git']['dirty'] else ''}. Triton top-k plus cuBLASLt top-{lt_n}.\n"]
for k in (3, 5):
    p = rep / f"k{k}" / f"zero_shot_{hw}_summary.json"
    if k > k_max or not p.exists():
        continue
    s = json.loads(p.read_text())
    out.append(f"## Online search, k = {k} (Triton-only choice), regret vs this GPU's own oracle: median / p95 / p99 (mean)\n")
    out.append(f"{s['n_screen_shapes']} held-out shapes in the screen.\n")
    out.append("| variant | shapes | all | >= 50 us | < 50 us |\n|---|---|---|---|---|")
    for v, d in s["variants"].items():
        sc = d["scores"][f"online-{k}"]
        out.append(f"| {v} | {d['n_shapes']} | " + " | ".join(fmt(sc[g]) for g in ("all", ">= 50 us", "< 50 us")) + " |")
    out.append("")
out.append("Read `regret > 50%` and `unscored` in the full reports first (docs/zero_shot_runbook.md, section 5).\n")
for p in sorted(rep.rglob("*.md")):
    out.append(f"- `reports/{p.relative_to(rep)}`")
out.append(f"\nRows and raw samples: `data/{hw}/`. For a comparison with another GPU: `python -m analysis.cross_gpu` "
           f"(docstring has the command). Then add a dated update to `reports/week1.md`.")
(run / "SUMMARY.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
EOF
}

# ------------------------------------------------------------------ driver
selected() {  # is stage $1 selected by --from / --only?
  local s="$1"
  if [[ -n "$ONLY" ]]; then [[ ",$ONLY," == *",$s,"* ]]; return; fi
  if [[ -n "$FROM" ]]; then
    local seen=0 t
    for t in "${STAGES[@]}"; do [[ "$t" == "$FROM" ]] && seen=1; [[ "$t" == "$s" ]] && return $((1 - seen)); done
  fi
  return 0
}
is_gpu_stage() { local s; for s in "${GPU_STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }

if [[ -n "$FROM" ]]; then printf '%s\n' "${STAGES[@]}" | grep -qx "$FROM" || die "unknown stage '$FROM' (see --list-stages)"; fi
if [[ "$CHECK_ONLY" == 1 ]]; then check_prereqs; exit 0; fi

say "Zero-shot mode: $MODE (per-suite $PER_SUITE, k $K, cuBLASLt top-$LT_N, cuBLASLt ranker: $LT_MODE${LT_RANK:+ top-$LT_RANK}) | run dir: $RUN_DIR | clock lock: $LOCK"
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
    echo "Fix the problem, then continue with: pipeline/zero_shot.sh --resume $RUN_DIR" >&2
    exit "$rc"
  fi
  date -Is > "$stamp"
  printf '[done] %s in %dm%02ds\n' "$s" $(((SECONDS - t0) / 60)) $(((SECONDS - t0) % 60))
done
printf '\nZero-shot run finished in %dh%02dm. Summary: %s/SUMMARY.md\n' $(((SECONDS - T_START) / 3600)) \
  $((((SECONDS - T_START) % 3600) / 60)) "$RUN_DIR"
