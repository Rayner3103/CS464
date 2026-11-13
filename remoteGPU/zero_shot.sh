#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=4           # CPUs: the sweep compiles Triton configs in parallel (one worker per CPU, ~1.25 GiB RAM each)
#SBATCH --mem=24GB                  # How much memory does your job require?
#SBATCH --gres=gpu:1                # One GPU: the measurements need the whole card to themselves
#SBATCH --time=01-00:00:00          # <DD-HH:MM:SS>. The oracle sweep dominates (see PER_SUITE); every stage resumes, so resubmit after a timeout
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=zero_shot_%j.out   # Log file, in the directory you submit from

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=student                 # The partition you've been assigned
#SBATCH --account=cs701                     # The account you've been assigned (normally student)
#SBATCH --qos=studentqos                    # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=rayner.sim.2024@computing.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=zero-shot-gemm           # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Zero-shot transfer of the RTX 3080 kernel selectors (lgbm_rank v1 and lgbm_rank_v2, no refit) to whatever GPU SLURM gives this job.
# Stages (docs/zero_shot_runbook.md has the long form and how to read the results):
#   sweep   the oracle: this GPU's own screen of the held-out shapes (hours; resumes per shape)
#   run     rank + compile + short-time on this GPU, v1/v2 x as_is/native, Triton top-5 + cuBLASLt top-16 (minutes)
#   score   regret vs the oracle at k = 3 and k = 5 (CPU only), reports under reports/zero_shot_<hw_id>/
#   stage2  times every pick with the full protocol next to B2/B3/B3+ and scores the joint choice (about 0.5 h + 20 s per shape)
#
# Once, on the login node:
#   module load Python/3.12.8-GCCcore-13.3.0
#   git clone <repo> ~/repos/CS464 && cd there      # a checkout that contains commit 9a2e9ab
#   python3.12 -m venv .venv && source .venv/bin/activate
#   pip install --upgrade pip && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
#   mkdir -p data/src3080     # and copy the four 3080 model files into it, by hand (data/ is gitignored, ~71 MB):
#     scp <3080 box>:.../data/rtx3080_sm86_68sms_10gb/{lgbm_rank.txt,lgbm_rank.json,lgbm_rank_v2.txt,lgbm_rank_v2.json} data/src3080/
#
# Submit from the repo:   sbatch remoteGPU/zero_shot.sh
# Settings, as environment variables (sbatch exports them):
#   PER_SUITE=12     held-out shapes per suite for the oracle: 12 = the 27-shape subset (the 2060 took 17.7 h for it), 1000 = all 86
#   STAGES=sweep,run,score,stage2   which stages to run (all by default; e.g. STAGES=run,score,stage2 when the screen already exists).
#                    ltscreen,ltfit (cuBLASLt ranker retraining, LT_MODE=retrain only) are added by LT_MODE, in front of sweep
#   LT_MODE=off      the learned cuBLASLt ranker in the joint search (baselines/cublaslt_rank.py): off | zero_shot | retrain.
#                      zero_shot  the 3080's lt_rank.{txt,json} (LT_MODELS=DIR, default $MODELS), no refit, as_is and native SM count.
#                                 Only meaningful on the same architecture: the RTX 3090 (sm_86), not the 2060 (sm_75)
#                      retrain    screen the cuBLASLt configs of the tune shapes on this GPU (ltscreen, LT_SHAPES=160 spread over the
#                                 652; 0 = all, ~70 min on a 3080), fit data/<hw>/lt_rank (ltfit, CPU) and use it: the choice for the 2060
#   LT_RANK=5        cuBLASLt ranker picks per search (with LT_MODE != off)
#   LT_SHAPES=160    tune shapes of the cuBLASLt screen for LT_MODE=retrain (0 = all 652)
#   e.g.  LT_MODE=retrain sbatch remoteGPU/zero_shot.sh            (2060)       LT_MODE=zero_shot sbatch remoteGPU/zero_shot.sh   (3090)
#   CLOCK_LOCK=none  declared SM clock lock in MHz. A job cannot run `nvidia-smi -lgc` (no sudo), so the default is "none" = unlocked:
#                    results are NOT comparable to the 3080's 1755 MHz numbers in absolute time, only in regret / ratios within this GPU
#   WORKERS=4        parallel compile workers of the sweep (default: the CPUs of the job)
#   MODELS=DIR       where the four model files are (default <repo>/data/src3080)
#   REPO=DIR         the repo checkout (default ~/repos/CS464; its .venv is used)
#   ALLOW_VERSION_MISMATCH=1   continue when torch / triton / lightgbm differ from requirements.txt (kernels and numbers can change)
#   e.g.  PER_SUITE=1000 sbatch remoteGPU/zero_shot.sh
# Resubmitting after a timeout continues: the sweep skips finished shapes, `run` skips finished (shape, k) rows, `benchmark` skips finished rows.

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

# Purge the environment, load the modules we require (the Python module also brings gcc, which Triton needs to build its launchers).
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
module purge
module load Python/3.12.8-GCCcore-13.3.0

REPO="${REPO:-~/repos/CS464}"
[ -f "$REPO/.venv/bin/activate" ] || die "no virtualenv at $REPO/.venv: see the login-node steps at the top of this script"
source "$REPO/.venv/bin/activate"
cd "$REPO"
[ -f analysis/zero_shot.py ] || die "$REPO is not the CS464 repo (analysis/zero_shot.py missing)"
# 9a2e9ab keeps picks the screen cut as `slow` in the statistics; earlier checkouts silently drop the bad picks. 4319453 (v2 native on a smaller GPU) precedes it.
git merge-base --is-ancestor 9a2e9ab HEAD 2>/dev/null || die "the checkout does not contain commit 9a2e9ab (git pull)"
command -v gcc >/dev/null || die "no gcc on PATH: Triton cannot build its launchers"

# --- versions: the pinned ones, or the numbers may not be comparable
python - <<'EOF' || { [ "${ALLOW_VERSION_MISMATCH:-0}" = "1" ] || exit 1; echo "continuing despite the version mismatch (ALLOW_VERSION_MISMATCH=1)"; }
import re, sys
want = dict(l.split("#")[0].strip().split("==") for l in open("requirements.txt") if "==" in l)
import torch, triton, lightgbm
have = {"torch": torch.__version__.split("+")[0], "triton": triton.__version__, "lightgbm": lightgbm.__version__}
bad = {k: (have[k], want[k]) for k in have if k in want and have[k] != want[k]}
print("versions:", have)
if bad:
    print("ERROR: differs from requirements.txt (have, want):", bad)
    sys.exit(1)
EOF

# --- the GPU this job got
echo "job ${SLURM_JOB_ID:-local} on $(hostname), started $(date '+%F %T'), repo $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
[ "$DRIVER_MAJOR" -ge 580 ] 2>/dev/null || die "driver $DRIVER_MAJOR is older than 580: the pinned torch wheel is CUDA 13.0"
echo "other compute processes on the GPU (should be none):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

LOCK="${CLOCK_LOCK:-none}"
[ "$LOCK" = "none" ] && echo "WARNING: clocks are not locked (a job cannot lock them): compare regret and ratios within this GPU, not times with the 3080's 1755 MHz runs"
TUNE=(python -m tune --clock-lock "$LOCK")
ZS=(python -m analysis.zero_shot --clock-lock "$LOCK")
PER_SUITE="${PER_SUITE:-12}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
STAGES="${STAGES:-ltscreen,ltfit,sweep,run,score,stage2}"
stage() { [[ ",$STAGES," == *",$1,"* ]]; }

D="$(python -c 'from tune.env import default_data_dir; print(default_data_dir())' | tail -1)"
HW="$(basename "$D")"
mkdir -p "$D"
echo "hw_id $HW, data $D, stages $STAGES, per-suite $PER_SUITE, workers $WORKERS, clock lock $LOCK"

# --- the 3080 models: present, and really the 3080's
MODELS="${MODELS:-$REPO/data/src3080}"
for f in lgbm_rank.txt lgbm_rank.json lgbm_rank_v2.txt lgbm_rank_v2.json; do
    [ -f "$MODELS/$f" ] || die "$MODELS/$f is missing: copy the four 3080 model files (see the top of this script)"
done
md5sum "$MODELS"/lgbm_rank.txt "$MODELS"/lgbm_rank_v2.txt
python - "$MODELS" <<'EOF'
import json, sys
for stem in ("lgbm_rank", "lgbm_rank_v2"):
    print(stem, "was fit on:", json.load(open(f"{sys.argv[1]}/{stem}.json"))["gpu"]["name"])
EOF

LT_MODE="${LT_MODE:-off}"
LT_RANK="${LT_RANK:-5}"
LT_SHAPES="${LT_SHAPES:-160}"
case "$LT_MODE" in off|zero_shot|retrain) ;; *) die "LT_MODE must be off, zero_shot or retrain" ;; esac
[ "$LT_MODE" = "off" ] && LT_RANK=0
LT_ARGS=()
if [ "$LT_MODE" = "zero_shot" ]; then
    LT_MODELS="${LT_MODELS:-$MODELS}"
    for f in lt_rank.txt lt_rank.json; do
        [ -f "$LT_MODELS/$f" ] || die "$LT_MODELS/$f is missing: LT_MODE=zero_shot needs the 3080's cuBLASLt ranker (data/<hw>/ or reports/snapshot/lock1755/)"
    done
    md5sum "$LT_MODELS"/lt_rank.txt
    LT_ARGS=(--lt-rank "$LT_RANK" --lt-rank-model "$LT_MODELS/lt_rank")
elif [ "$LT_MODE" = "retrain" ]; then
    LT_ARGS=(--lt-rank "$LT_RANK")
    export TUNE_LT_RANK="$D/lt_rank"
fi
echo "cuBLASLt ranker: $LT_MODE (top-$LT_RANK)"

if [ "$LT_MODE" = "retrain" ] && stage ltscreen; then
    echo "=== ltscreen: every cuBLASLt config of the tune shapes, on this GPU ($(date '+%F %T'))"
    [ "$LOCK" = "none" ] || export TUNE_CLOCK_LOCK="$LOCK"
    LIM=(); [ "$LT_SHAPES" -gt 0 ] && LIM=(--limit "$LT_SHAPES" --spread)
    python -u -m baselines.cublaslt_screen --out "$D/lt_screen" "${LIM[@]}"
fi
if [ "$LT_MODE" = "retrain" ] && stage ltfit; then
    echo "=== ltfit: cuBLASLt ranker on this GPU's screen ($(date '+%F %T'))"
    python -u -m analysis.lt_rank --screen-dir "$D/lt_screen" --save "$D/lt_rank"
fi

if stage sweep; then
    echo "=== sweep: held-out oracle ($(date '+%F %T'))"
    "${TUNE[@]}" env
    "${TUNE[@]}" sweep --split heldout --per-suite "$PER_SUITE" --workers "$WORKERS"
fi

# The scoring needs a complete oracle; a partial one means the job ran out of time or a shape did not fit in memory.
[ -f "$D/sweep_screen_heldout/subset.json" ] || die "no $D/sweep_screen_heldout/subset.json: run the sweep stage first"
python - "$D/sweep_screen_heldout" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
want = json.loads((d / "subset.json").read_text())["n"]
done = sum(1 for p in d.glob("*_fp16.json") if json.loads(p.read_text()).get("complete"))
print(f"oracle: {done} of {want} held-out shapes complete")
if done < want:
    print("WARNING: incomplete oracle; shapes without a screen are not scored (resubmit to continue the sweep)")
EOF

if stage run; then
    echo "=== run: ranker -> compile -> short online search ($(date '+%F %T'))"
    "${ZS[@]}" run --ranker v1="$MODELS/lgbm_rank" v2="$MODELS/lgbm_rank_v2" --k 5 --lt-n 16 --fresh-cache "${LT_ARGS[@]}"
fi

# Each `score` call writes one fixed file name per data directory, so k = 3 and k = 5 go to separate report folders
REPORTS="$REPO/reports/zero_shot_$HW"
if stage score; then
    echo "=== score ($(date '+%F %T'))"
    "${ZS[@]}" score --k 3 --report "$REPORTS/k3"
    "${ZS[@]}" score --k 5 --report "$REPORTS/k5"
fi

if stage stage2; then
    echo "=== stage 2: every pick next to B2/B3/B3+ ($(date '+%F %T'))"
    "${ZS[@]}" picks
    export TUNE_SWEEP_DIR="$D/sweep_screen_heldout" TUNE_PICKS="$D/zero_shot_stage2_picks.json"
    BACKENDS="torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20"
    [ "$LT_RANK" -gt 0 ] && BACKENDS="torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_rank:$LT_RANK:skip16,cublaslt_deep:20"
    # `cublaslt_rank:M` expands to what this process's ranker proposes: a zero-shot model needs a pass per SM-count variant (as_is, native);
    # `benchmark` resumes per (shape, candidate key), so a second pass only times the picks the first did not propose.
    VARIANTS=(default); [ "$LT_MODE" = "zero_shot" ] && VARIANTS=(as_is native)
    bench() {
        for v in "${VARIANTS[@]}"; do
            if [ "$v" != default ]; then
                export TUNE_LT_RANK="$LT_MODELS/lt_rank" TUNE_LT_RANK_FOREIGN=1
                if [ "$v" = native ]; then export TUNE_LT_RANK_SMS=native; else unset TUNE_LT_RANK_SMS; fi
                echo "--- cuBLASLt ranker variant: $v"
            fi
            "${TUNE[@]}" benchmark "$@"
        done
    }
    bench --shapes-file "$D/sweep_screen_heldout/subset.json" --backends "$BACKENDS" --cache-modes cold \
        --out-name results_zero_shot.jsonl
    N_SMALL="$(python -c "import json; print(json.load(open('$D/zero_shot_lt50us.json'))['n'])")"
    if [ "$N_SMALL" -gt 0 ]; then
        bench --shapes-file "$D/zero_shot_lt50us.json" --backends "$BACKENDS" --cache-modes hot --graph-batch 50 \
            --out-name results_zero_shot_graph.jsonl
    else
        echo "no held-out shape under 50 us on this GPU: no graph pass"
        : > "$D/results_zero_shot_graph.jsonl"
    fi
    # --report: the default (reports/) would overwrite the 3080's committed ranker_compare.md
    ONLINE=()
    for m in v1 v2; do for v in as_is native; do ONLINE+=("${m}_${v}=zero_shot_${m}_${v}.jsonl"); done; done
    python -m analysis.ranker_compare --data "$D" --report "$REPORTS" --results results_zero_shot.jsonl \
        --graph-results results_zero_shot_graph.jsonl --online "${ONLINE[@]}"
fi

echo "finished $(date '+%F %T'). Rows: $D/zero_shot_*.jsonl; reports: $REPORTS"
echo "next: copy $D and $REPORTS back (scp -r), then add a dated update to reports/week1.md (docs/zero_shot_runbook.md, section 7)"
