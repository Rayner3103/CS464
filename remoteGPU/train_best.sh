#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=4           # CPUs: the Triton sweeps compile configs in parallel (one worker per CPU, ~1.25 GiB RAM each); lower it if the QOS refuses
#SBATCH --mem=16GB                  # How much memory does your job require? (8 workers x ~1.25 GiB + the fits)
#SBATCH --gres=gpu:1                # One GPU: the measurements need the whole card to themselves (this script refuses a node that is not EXPECT_GPU)
#SBATCH --time=01-00:00:00          # <DD-HH:MM:SS>. A full run is ~35 h on a 3080: it will not fit one job. Every stage resumes (see CHAIN / resubmit)
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=train_best_%j.out  # Log file, in the directory you submit from

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=student                 # The partition you've been assigned
#SBATCH --account=cs701                     # The account you've been assigned (normally student)
#SBATCH --qos=studentqos                    # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=rayner.sim.2024@computing.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=train-best-gemm          # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Train the SHIPPED kernel selector from scratch on whatever GPU SLURM gives this job (meant for the RTX 3090), to compare it with the
# RTX 3080 one. This wraps pipeline/train_best.sh, which fits every learned component from this GPU's own measurements, borrowing nothing
# from the 3080:
#   rank v1 + rank v2   LightGBM rankers over the Triton family (v2 = analytic features + near-top weights + the `tail` suite as extra tune data)
#   cuBLASLt ranker     LightGBM regressor over every valid cuBLASLt config (top-5 on top of cuBLASLt's heuristic top-16)
#   family router       P(Triton still beats the cuBLASLt value); skipped by the pipeline if Triton (almost) never wins on this GPU
# then checks them on the 86 held-out shapes against this GPU's own oracle and against B2 / B3 / B3+. Stage list, what each stage runs and the
# outputs: `pipeline/train_best.sh --help` and `--list-stages`; the models end up in <RUN_DIR>/models/ with an INSTALL.md.
#
# Once, on the login node (the compute node may have no network, so install here, not in the job):
#   module load Python/3.12.8-GCCcore-13.3.0
#   git clone <repo> ~/repos/CS464 && cd ~/repos/CS464      # a checkout that contains commit 8a7cec1 (train_best.sh --quick and the router guard)
#   python3.12 -m venv .venv && source .venv/bin/activate
#   pip install --upgrade pip && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
#   chmod +x pipeline/train_best.sh remoteGPU/train_best.sh
# Optional, for the automatic 3080 comparison at the end (data/ is gitignored, copy by hand; see COMPARE below):
#   scp <3080 box>:.../data/rtx3080_sm86_68sms_10gb/{sweep_screen_heldout,online_joint_diverse_v2.jsonl,results_lock1755_v2.jsonl,results_lock1755_v2_graph.jsonl} data/src3080/
#   (sweep_screen_heldout is a directory: scp -r; the .raw.jsonl files are not needed)
#
# Submit from the repo:   MODE=full sbatch remoteGPU/train_best.sh
# Settings, as environment variables (sbatch exports them):
#   MODE=full          full | quick | smoke. full = the study's own sizes (477 tune + 150 tail shapes, 86 held-out, 652 cuBLASLt screen shapes):
#                      the only mode whose held-out set (86) is the 3080's, so the only one that compares directly. quick = ~56 / 27 / 30 shapes
#                      (smaller fits, 27 of the 86 held-out shapes), smoke = plumbing check (meaningless fits)
#   EXPECT_GPU=3090    substring the GPU name must contain, or the job stops before spending hours on the wrong card. EXPECT_GPU=any disables it
#   CHAIN=3            on a start (not a resume that already finished), queue up to this many follow-up jobs that run only if this one
#                      fails or times out (sbatch --dependency=afternotok), so a ~40 h run continues by itself across 1-day jobs. Each follow-up
#                      resumes the same RUN_DIR. Needs sbatch on the compute node; if it is refused the job runs anyway and you resubmit by hand.
#                      CHAIN=0 (default) = no follow-ups: after a timeout just `MODE=full sbatch remoteGPU/train_best.sh` again
#   CLOCK_LOCK=none    declared SM clock lock in MHz. A job cannot run `nvidia-smi -lgc` (no sudo), so the default is "none" = unlocked:
#                      times are NOT comparable with the 3080's 1755 MHz numbers in absolute terms, only regret / ratios within this GPU.
#                      If the node has the clock locked by the admins, declare it here (the env stage checks it against the real state)
#   WITH_BUNDLE=1      also build/verify/pack the prebuilt Triton kernel bundle of the new rankers (~10 min on 10 workers, ~2 GB)
#   SKIP_VALIDATION=1  stop after training and packaging the models (skips the held-out validation, ~1.5 h on a 3080)
#   WORKERS=8          parallel compile workers of the sweeps (default: the CPUs of the job; "auto" would read the node's RAM, not the job's cgroup)
#   STAGES=a,b         run only these stages (pipeline/train_best.sh --list-stages), e.g. STAGES=router_fit,package,summary
#   FROM=stage         start at this stage instead (earlier ones assumed done)
#   FORCE=1            re-run stages even if marked done
#   RUN_DIR=DIR        output directory (default $REPO/pipeline/runs/train_best-<MODE>-slurm; stable across resubmissions on purpose)
#   REPO=DIR           the repo checkout (default ~/repos/CS464; its .venv is used)
#   COMPARE=1          after a successful run, compare with the 3080 (CPU, analysis.cross_gpu -> <RUN_DIR>/reports/cross_gpu_3080.md) when
#     SRC3080=DIR      has the 3080's held-out files (default <repo>/data/src3080, see the scp line above). COMPARE=only does just this on
#                      an existing RUN_DIR and needs no GPU (run it with `bash remoteGPU/train_best.sh`, or sbatch it). Skipped with a note
#                      if the files are missing or the run was not --full (the shape sets must match to be comparable).
#   e.g.  MODE=full CHAIN=3 sbatch remoteGPU/train_best.sh
#
# How to read the result against the 3080 (reports/week1.md, AGENTS.md section 6; same protocol, a different card; MODE=full, held-out 86 shapes):
#   Triton-only online top-3 regret vs the GPU's own screen oracle (median / p95 / p99 / mean), v2 on the 3080: 0.2% / 4.6% / 9.6% / 1.2%.
#   Joint pool (v2 top-3 + LT16 + cuBLASLt ranker top-5) vs the extended oracle on the 3080: 0.5% / 5.5% / 8.4% / 1.4%; family router 0.4 / 4.9 / 8.8 / 1.3%.
#   Read SUMMARY.md first, then the `regret > 50%` and `unscored` columns of <RUN_DIR>/reports/k3 (docs/zero_shot_runbook.md section 5).
#   Confounders to name in any write-up: unlocked clocks (a job cannot lock them), a different driver and Triton/cuBLASLt build if the node differs,
#   no noise study on this GPU (so no tie margin of its own), and a fit on this card's 82 SMs vs the 3080's 68.
#
# Resubmitting after a timeout continues: the sweeps skip finished shapes, `benchmark` skips finished rows, finished stages are stamped.

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

# Purge the environment, load the modules we require (the Python module also brings gcc, which Triton needs to build its launchers).
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
if command -v module >/dev/null 2>&1 || type module >/dev/null 2>&1; then
    module purge
    module load Python/3.12.8-GCCcore-13.3.0
fi

REPO="${REPO:-$HOME/repos/CS464}"
[ -f "$REPO/.venv/bin/activate" ] || die "no virtualenv at $REPO/.venv: see the login-node steps at the top of this script"
source "$REPO/.venv/bin/activate"
cd "$REPO"
[ -x pipeline/train_best.sh ] || die "$REPO is not the CS464 repo with pipeline/train_best.sh (missing or not executable: git pull; chmod +x pipeline/train_best.sh)"
command -v gcc >/dev/null || die "no gcc on PATH: Triton cannot build its launchers"
command -v flock >/dev/null || die "no flock on PATH: pipeline/train_best.sh takes pipeline/.lock with it"

MODE="${MODE:-full}"
case "$MODE" in smoke|quick|full) ;; *) die "MODE must be smoke, quick or full (got '$MODE')" ;; esac
LOCK="${CLOCK_LOCK:-none}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
RUN_DIR="${RUN_DIR:-$REPO/pipeline/runs/train_best-$MODE-slurm}"
SRC3080="${SRC3080:-$REPO/data/src3080}"
EXPECT_GPU="${EXPECT_GPU:-3090}"
COMPARE="${COMPARE:-1}"

# --- 3080 comparison (CPU only): the same held-out shapes scored on both GPUs with the repo's own functions.
compare_with_3080() {
    [ -f "$RUN_DIR/mode" ] || { echo "compare: $RUN_DIR is not a train_best run directory"; return 0; }
    if [ "$(cat "$RUN_DIR/mode")" != "full" ]; then
        echo "compare: skipped, the run is '$(cat "$RUN_DIR/mode")' and only a full run has the 3080's 86 held-out shapes (for a subset use analysis.cross_gpu by hand)"
        return 0
    fi
    local f hw d out
    for f in sweep_screen_heldout/subset.json online_joint_diverse_v2.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl; do
        [ -e "$SRC3080/$f" ] || { echo "compare: skipped, $SRC3080/$f is missing (copy the 3080 files, see the top of this script)"; return 0; }
    done
    hw="$(cat "$RUN_DIR/hw_id" 2>/dev/null)" || { echo "compare: no hw_id in $RUN_DIR"; return 0; }
    d="$RUN_DIR/data/$hw"
    for f in zero_shot_v2_as_is.jsonl results_zero_shot.jsonl results_zero_shot_graph.jsonl; do
        [ -e "$d/$f" ] || { echo "compare: skipped, $d/$f is missing (the validation stages did not finish, or SKIP_VALIDATION=1)"; return 0; }
    done
    out="$RUN_DIR/reports/cross_gpu_3080.md"
    # Side 1 is the reference of the paired bootstrap. The 3080's v2 rows are the Triton-only choice "3_lt0" of the joint run; this card's v2 rows
    # carry the same key (analysis.zero_shot run --lt-n), and its stage 2 is the same-session B2 / B3 / B3+ of the validate stages.
    python -m analysis.cross_gpu --out "$out" \
        --side 3080-v2 "$SRC3080" online_joint_diverse_v2.jsonl online_joint_diverse_v2.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
        --side "$EXPECT_GPU-v2" "$d" zero_shot_v2_as_is.jsonl zero_shot_v2_as_is.jsonl results_zero_shot.jsonl results_zero_shot_graph.jsonl \
        && echo "compare: wrote $out" \
        || echo "compare: analysis.cross_gpu failed (see above); the run itself is fine, re-run with COMPARE=only"
}

if [ "$COMPARE" = "only" ]; then
    compare_with_3080
    exit 0
fi

# --- versions: the pinned ones, or the numbers may not be comparable
python - <<'EOF' || { [ "${ALLOW_VERSION_MISMATCH:-0}" = "1" ] || exit 1; echo "continuing despite the version mismatch (ALLOW_VERSION_MISMATCH=1)"; }
import sys
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
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [ "$EXPECT_GPU" != "any" ] && [[ "$GPU_NAME" != *"$EXPECT_GPU"* ]]; then
    die "this node's GPU is '$GPU_NAME', not a $EXPECT_GPU: request the right node (e.g. sbatch --nodelist=<node> or --gres=gpu:<type>:1), or EXPECT_GPU=any to train on it anyway"
fi
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
[ "$DRIVER_MAJOR" -ge 580 ] 2>/dev/null || die "driver $DRIVER_MAJOR is older than 580: the pinned torch wheel is CUDA 13.0"
echo "other compute processes on the GPU (should be none):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# --- the run directory: one per MODE, reused by every resubmission. Resume it with its own mode, never start a second full run next to it.
ARGS=(--skip-install --workers "$WORKERS" --clock-lock "$LOCK")
if [ -f "$RUN_DIR/mode" ]; then
    [ "$(cat "$RUN_DIR/mode")" = "$MODE" ] || die "$RUN_DIR holds a '$(cat "$RUN_DIR/mode")' run but MODE=$MODE: set RUN_DIR or MODE"
    echo "resuming $RUN_DIR (finished stages are skipped):"; ls "$RUN_DIR/stamps" 2>/dev/null | sed 's/\.done$//' | tr '\n' ' '; echo
    ARGS+=(--resume "$RUN_DIR")
    FRESH=0
else
    ARGS+=("--$MODE" --run-dir "$RUN_DIR")
    FRESH=1
fi
if [ "$LOCK" = "none" ]; then
    echo "WARNING: clocks are not locked (a job cannot lock them): compare regret and ratios within this GPU, not times with the 3080's 1755 MHz runs"
    ARGS+=(--allow-unlocked)
fi
[ "${WITH_BUNDLE:-0}" = "1" ] && ARGS+=(--with-bundle)
[ "${SKIP_VALIDATION:-0}" = "1" ] && ARGS+=(--skip-validation)
[ "${FORCE:-0}" = "1" ] && ARGS+=(--force)
[ -n "${STAGES:-}" ] && ARGS+=(--only "$STAGES")
[ -n "${FROM:-}" ] && ARGS+=(--from "$FROM")

# --- follow-up jobs for a run longer than the time limit. Queued only when this job fails or times out, so a finished run leaves them pending
# (never started) and the last one can simply be cancelled: squeue -u $USER, scancel <id>.
CHAIN="${CHAIN:-0}"
if [ "$CHAIN" -gt 0 ] && [ -n "${SLURM_JOB_ID:-}" ] && [ ! -f "$RUN_DIR/SUMMARY.md" ]; then
    if command -v sbatch >/dev/null 2>&1; then
        NEXT="$(sbatch --parsable --dependency="afternotok:$SLURM_JOB_ID" --export=ALL,CHAIN="$((CHAIN - 1))",MODE="$MODE",RUN_DIR="$RUN_DIR",REPO="$REPO" \
                "$REPO/remoteGPU/train_best.sh" 2>&1)" \
            && echo "queued follow-up job $NEXT (runs only if this one fails or times out; $((CHAIN - 1)) more after it)" \
            || echo "WARNING: could not queue a follow-up job ($NEXT): resubmit by hand after a timeout"
    else
        echo "WARNING: CHAIN=$CHAIN but sbatch is not available on this node: resubmit by hand after a timeout"
    fi
fi

echo "pipeline/train_best.sh ${ARGS[*]}"
# Not exec: the comparison runs afterwards. SLURM's SIGTERM at the time limit is forwarded to the pipeline's own trap, which stops the sweep
# workers (no orphans), and the job then exits non-zero so the afternotok follow-up starts.
pipeline/train_best.sh "${ARGS[@]}" &
PID=$!
trap 'echo "SIGTERM/INT: stopping the pipeline" >&2; kill -TERM "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; exit 143' TERM INT
RC=0
wait "$PID" || RC=$?
trap - TERM INT
if [ "$RC" -ne 0 ]; then
    echo "pipeline/train_best.sh exited with $RC; resubmit to continue: MODE=$MODE RUN_DIR=$RUN_DIR sbatch remoteGPU/train_best.sh" >&2
    exit "$RC"
fi

if [ -f "$RUN_DIR/SUMMARY.md" ]; then
    echo; echo "=== $RUN_DIR/SUMMARY.md ==="; cat "$RUN_DIR/SUMMARY.md"
    [ "$COMPARE" = "0" ] || compare_with_3080
    echo; echo "models: $RUN_DIR/models/ (INSTALL.md says where they go); copy the run directory back with:"
    echo "  rsync -a <cluster>:$RUN_DIR/ pipeline/runs/$(basename "$RUN_DIR")/   (add --exclude 'data/*/sweep_screen*/' for a small copy)"
fi
