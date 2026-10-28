#!/usr/bin/env bash
# One-command pipeline: install -> measurement checks -> tune (Triton config sweep) -> train (learned selector)
# -> validate (held-out shapes, online search). See pipeline/README.md for prerequisites and details.
#
#   pipeline/run.sh [--smoke | --quick | --full] [options]
#
# Everything a run produces goes to pipeline/runs/<tag>/ (data/, reports/, logs/); existing data/ and reports/ in
# the repo are never touched. Re-running with --resume <run dir> skips finished stages.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGES=(preflight install tests env check roofline noise drift tune_sweep portfolio heldout_sweep portfolio_heldout
        selectors train validate_cold validate_graph online_plain online_diverse score summary)
GPU_STAGES=(tests env check roofline noise drift tune_sweep heldout_sweep selectors train validate_cold validate_graph
            online_plain online_diverse)

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

usage() {
  cat <<EOF
Usage: pipeline/run.sh [mode] [options]

Modes (default --quick):
  --smoke           plumbing check on a handful of shapes (results are not meaningful)
  --quick           reduced run: ~50 tune shapes, ~27 held-out shapes
  --full            the whole study: 477 tune + 86 held-out shapes, full noise study (a day or more)

Options:
  --clock-lock MHZ|none   SM clock you locked beforehand with 'sudo nvidia-smi -lgc MHZ,MHZ' (default $LOCK).
                          The script never calls sudo. 'none' = run unlocked (needs --allow-unlocked).
  --allow-unlocked        continue even if the GPU clock is not locked (results not comparable to the study)
  --run-dir DIR           output directory (default pipeline/runs/<mode>-<timestamp>)
  --resume DIR            continue a previous run directory; finished stages are skipped (mode is read from it)
  --venv DIR              virtualenv location (default pipeline/.venv)
  --python BIN            interpreter used to create the venv (default python3, or \$PYTHON)
  --skip-install          use the current 'python' on PATH as is (checks the pinned versions, installs nothing)
  --skip-tests            skip the pytest stage
  --workers N             parallel compile workers for the Triton sweep (default 0 = auto: up to 10, capped by free RAM / VRAM)
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

# ------------------------------------------------------------------ run directory and mode
if [[ -n "$RUN_DIR" && -z "$MODE" ]]; then
  [[ -f "$RUN_DIR/mode" ]] || die "$RUN_DIR is not a pipeline run directory (no 'mode' file)"
  MODE="$(<"$RUN_DIR/mode")"
fi
[[ -n "$RUN_DIR" ]] || RUN_DIR="$REPO/pipeline/runs/$MODE-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"/{data,reports,logs,stamps}
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
DATA="$RUN_DIR/data"
REPORTS="$RUN_DIR/reports"
echo "$MODE" > "$RUN_DIR/mode"
[[ "$LOCK" == "none" && "$ALLOW_UNLOCKED" == 0 ]] && die "--clock-lock none needs --allow-unlocked"

# Per-mode sizes. TUNE_PS / HELD_PS = shapes per suite in the sweep subsets (1000 = all). DEEP_S = seconds per shape
# of the cuBLASLt enumerate-and-screen baseline (B3+). NOISE_* = noise-floor study size.
case "$MODE" in
  smoke) TUNE_PS=3;    HELD_PS=3;    DEEP_S=2;  NOISE_ROUNDS=0; NOISE_REPEATS=0;  DRIFT_MIN=0;  CHECK_SUITE=S1 ;;
  quick) TUNE_PS=8;    HELD_PS=12;   DEEP_S=10; NOISE_ROUNDS=1; NOISE_REPEATS=20; DRIFT_MIN=0;  CHECK_SUITE=edge ;;
  full)  TUNE_PS=1000; HELD_PS=1000; DEEP_S=20; NOISE_ROUNDS=3; NOISE_REPEATS=50; DRIFT_MIN=10; CHECK_SUITE=edge ;;
  *) die "unknown mode '$MODE'" ;;
esac

if [[ "$SKIP_INSTALL" == 1 ]]; then PY="$(command -v python)" || die "no 'python' on PATH"; else PY="$VENV/bin/python"; fi

# One pipeline at a time: overlapping GPU jobs contaminate measurements.
exec 9>"$REPO/pipeline/.lock"
flock -n 9 || die "another pipeline run holds pipeline/.lock; never run two GPU jobs at once"

trap 'echo; echo "interrupted; stopping child processes" >&2; trap - TERM; kill 0' INT TERM

tune() { "$PY" -u -m tune --clock-lock "$LOCK" "$@"; }

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
  gpu_idle_check
  echo "Prerequisites OK. Clock lock: declared '$LOCK' (verified after the env stage)."
}

# ------------------------------------------------------------------ stages
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
import torch, triton, lightgbm, sklearn
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("torch", torch.__version__, "triton", triton.__version__, "lightgbm", lightgbm.__version__,
      "| GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
EOF
  { echo "commit: $(git rev-parse HEAD) dirty: $(git status --porcelain --untracked-files=no | wc -l) files"
    "$PY" -m pip freeze; } > "$RUN_DIR/environment.txt"
}

stage_preflight() { check_prereqs; }

stage_tests() {
  [[ "$SKIP_TESTS" == 1 ]] && { echo "skipped (--skip-tests)"; return; }
  "$PY" -m pytest -q
}

# env.json is read by the analysis code; also the place where the declared clock lock is cross-checked.
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
    print("WARNING: unlocked / unverified clocks; results are not comparable to the study.")
EOF
}

stage_check() { tune check --suite "$CHECK_SUITE" --backends triton:handpicked,triton_tutorial,torch --out "$DATA"; }
stage_roofline() { tune roofline --out "$DATA"; }

stage_noise() {
  [[ "$NOISE_ROUNDS" -gt 0 ]] || { echo "skipped in $MODE mode"; return; }
  tune noise --rounds "$NOISE_ROUNDS" --repeats "$NOISE_REPEATS" --n-allocs 3 --out "$DATA"
}

stage_drift() {
  [[ "$DRIFT_MIN" -gt 0 ]] || { echo "skipped in $MODE mode"; return; }
  tune drift --minutes "$DRIFT_MIN" --out "$DATA"
}

# Tune: stage-1 screen of the whole pruned Triton config space (every config compiled, validated and timed).
stage_tune_sweep() { tune sweep --split tune --per-suite "$TUNE_PS" --workers "$WORKERS" --out "$DATA"; }
stage_portfolio() { tune portfolio --data "$DATA" --report "$REPORTS"; }

# Held-out screen: shapes the selector never sees in training (heldout_region + heldout_random).
stage_heldout_sweep() { tune sweep --split heldout --per-suite "$HELD_PS" --workers "$WORKERS" --out "$DATA"; }
stage_portfolio_heldout() { tune portfolio --heldout --data "$DATA" --report "$REPORTS"; }

# Trivial selectors (b6, nn1, knn5, rule, nn1f, knnf, knnc): reference baselines; also writes selector_picks.json and the
# list of held-out shapes whose oracle is < 50 us (subset_lt50us.json), which the graph-timed validation needs.
stage_selectors() { tune selectors --data "$DATA" --report "$REPORTS"; }

# Train: grouped-CV hyperparameter search for all learned models (cached), refit lgbm_rank on every tune shape,
# fit the diversification rule on out-of-fold tune scores, save data/lgbm_rank.{txt,json}, write top-5 picks.
stage_train() { tune topk --data "$DATA" --report "$REPORTS"; }

# Validate: re-time every pick on the held-out shapes with the full protocol, in one session with the Triton oracle,
# torch.matmul and cuBLASLt (B3 = heuristic top-16, B3+ = enumerate-and-screen, DEEP_S s per shape).
validate_env() {
  export TUNE_SWEEP_DIR="$DATA/sweep_screen_heldout" TUNE_PICKS="$DATA/learned_topk_picks.json"
}
VAL_BACKENDS="torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:$DEEP_S"

stage_validate_cold() {
  validate_env
  tune benchmark --shapes-file "$DATA/sweep_screen_heldout/subset.json" --backends "$VAL_BACKENDS" \
    --cache-modes cold --out-name results_validate.jsonl --out "$DATA"
}

stage_validate_graph() {
  validate_env
  tune benchmark --shapes-file "$DATA/sweep_screen_heldout/subset_lt50us.json" --backends "$VAL_BACKENDS" \
    --cache-modes hot --graph-batch 50 --out-name results_validate_graph.jsonl --out "$DATA"
}

# First-use search: rank, then compile (empty Triton cache) + validate + short-time the top 5; keep the fastest.
# Choices for every k <= 5 are stored, so top-1..5 are all scored from one run.
stage_online_plain() {
  tune online --shapes-file "$DATA/sweep_screen_heldout/subset.json" --k 5 --order plain --fresh-cache --out "$DATA"
}
stage_online_diverse() {
  tune online --shapes-file "$DATA/sweep_screen_heldout/subset.json" --k 5 --order diverse --fresh-cache --out "$DATA"
}

stage_score() {
  tune topk --data "$DATA" --report "$REPORTS" --results results_validate.jsonl \
    --graph-results results_validate_graph.jsonl
}

stage_summary() {
  "$PY" - "$RUN_DIR" "$MODE" <<'EOF'
import json, sys
from pathlib import Path
run, mode = Path(sys.argv[1]), sys.argv[2]
rep = run / "reports"
out = [f"# Pipeline run `{run.name}` ({mode} mode)\n"]
env = json.loads((run / "data" / "env.json").read_text())
cl = env["clock_lock"]
out.append(f"GPU {env['gpu']['name']} ({env['hw_id']}), driver {env['driver']}, clock lock: {cl.get('state')} "
           f"{cl.get('lock_mhz') or ''}. Commit {env['harness_git']['commit'][:7]}"
           f"{' (dirty)' if env['harness_git']['dirty'] else ''}.\n")
md = rep / "learned_topk.md"
if md.exists():
    text = md.read_text()
    for title, tag in (("cold, all held-out shapes", "(cold)"), ("graph-amortized, held-out shapes < 50 us", "(graph)")):
        sec = text.split(f"## Stage 2 {tag}")
        if len(sec) < 2:
            continue
        body = sec[1].split("\n## ")[0]
        rows = [l for l in body.splitlines() if l.startswith("|") and any(
            k in l for k in ("| knnf |", "diverse online top-3", "plain online top-3", "diverse online top-1"))]
        n = next((l for l in body.splitlines() if "rows," in l), "")
        out.append(f"### Held-out stage 2, {title}\n\n{n}\n")
        out.append("Regret vs the per-shape Triton oracle, median / p95 (mean); last column = speed vs the searched cuBLASLt (B3+):\n")
        hdr = [l for l in body.splitlines() if l.startswith("|")][:2]
        out.extend(hdr)
        out.extend(rows)
        out.append("")
    out.append("The bar of the study is `diverse online top-3`: compile, check and short-time the ranker's 3 diversified picks and keep the fastest.\n")
else:
    out.append("No reports/learned_topk.md found; check logs/score.log.\n")
for name in ("sweep_portfolio", "sweep_portfolio_heldout", "selector_baselines", "learned_topk"):
    p = rep / f"{name}.md"
    if p.exists():
        out.append(f"- `reports/{p.name}`")
out.append("\nSaved ranker: `data/lgbm_rank.txt` + `data/lgbm_rank.json` (load with `analysis.learned.Ranker.load`).")
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

say "Pipeline mode: $MODE | run dir: $RUN_DIR | clock lock: $LOCK"
T_START=$SECONDS
for s in "${STAGES[@]}"; do
  selected "$s" || continue
  stamp="$RUN_DIR/stamps/$s.done"
  if [[ -f "$stamp" && "$FORCE" == 0 ]]; then echo "[skip] $s (done $(<"$stamp"))"; continue; fi
  # The install stage creates the interpreter the other stages use.
  if [[ "$s" != install && "$s" != preflight && ! -x "$PY" ]]; then die "$PY not found: run the install stage first"; fi
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
    echo "Fix the problem, then continue with: pipeline/run.sh --resume $RUN_DIR" >&2
    exit "$rc"
  fi
  date -Is > "$stamp"
  printf '[done] %s in %dm%02ds\n' "$s" $(((SECONDS - t0) / 60)) $(((SECONDS - t0) % 60))
done
printf '\nPipeline finished in %dh%02dm. Summary: %s/SUMMARY.md\n' $(((SECONDS - T_START) / 3600)) \
  $((((SECONDS - T_START) % 3600) / 60)) "$RUN_DIR"
