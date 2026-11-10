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
#SBATCH --time=01-00:00:00          # <DD-HH:MM:SS>. The tune sweep dominates (see MODE); every stage resumes, so resubmit after a timeout
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=pipeline_%j.out    # Log file, in the directory you submit from

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=student                 # The partition you've been assigned
#SBATCH --account=cs701                     # The account you've been assigned (normally student)
#SBATCH --qos=studentqos                    # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=rayner.sim.2024@computing.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=gemm-pipeline            # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# The full pipeline (pipeline/run.sh: checks -> tune sweep -> train the learned selector -> validate on held-out shapes) on
# whatever GPU SLURM gives this job. Stages and outputs: pipeline/README.md. Everything goes to
# $REPO/pipeline/runs/<mode>-slurm/ (data/, reports/, logs/, SUMMARY.md), which is stable across resubmissions on purpose:
# finished stages are skipped and the sweep / benchmarks resume per shape, so after a timeout just `sbatch` again.
#
# Once, on the login node (the compute node may have no network, so install here, not in the job):
#   module load Python/3.12.8-GCCcore-13.3.0
#   git clone <repo> ~/repos/CS464 && cd there      # a checkout that contains the pipeline/ directory
#   python3.12 -m venv .venv && source .venv/bin/activate
#   pip install --upgrade pip && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
#
# Submit from the repo:   sbatch remoteGPU/run.sh
# Settings, as environment variables (sbatch exports them):
#   MODE=quick       smoke | quick | full. quick = ~50 tune + ~27 held-out shapes (4-5 h on the 3080, much more on a slower card),
#                    full = 477 + 86 shapes (a day or more: raise --time and check the QOS limit, or resubmit; it resumes)
#   CLOCK_LOCK=none  declared SM clock lock in MHz. A job cannot run `nvidia-smi -lgc` (no sudo), so the default is "none" = unlocked:
#                    results are NOT comparable to the 3080's 1755 MHz numbers in absolute time, only in regret / ratios within this GPU
#   WORKERS=4        parallel compile workers of the sweep (default: the CPUs of the job; "auto" would read the node's RAM, not the job's cgroup)
#   SKIP_TESTS=1     skip the pytest stage (it takes a while and needs the GPU)
#   STAGES=a,b       run only these stages (pipeline/run.sh --list-stages), e.g. STAGES=selectors,train,score once the sweeps are done
#   FROM=stage       start at this stage instead (earlier ones assumed done)
#   FORCE=1          re-run stages even if marked done
#   RUN_DIR=DIR      output directory (default $REPO/pipeline/runs/<mode>-slurm)
#   REPO=DIR         the repo checkout (default ~/repos/CS464; its .venv is used)
#   e.g.  MODE=full sbatch remoteGPU/run.sh

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

# Purge the environment, load the modules we require (the Python module also brings gcc, which Triton needs to build its launchers).
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
module purge
module load Python/3.12.8-GCCcore-13.3.0

REPO="${REPO:-$HOME/repos/CS464}"
[ -f "$REPO/.venv/bin/activate" ] || die "no virtualenv at $REPO/.venv: see the login-node steps at the top of this script"
source "$REPO/.venv/bin/activate"
cd "$REPO"
[ -x pipeline/run.sh ] || die "$REPO is not the CS464 repo with the pipeline (pipeline/run.sh missing or not executable)"
command -v gcc >/dev/null || die "no gcc on PATH: Triton cannot build its launchers"

# --- the GPU this job got
echo "job ${SLURM_JOB_ID:-local} on $(hostname), started $(date '+%F %T'), repo $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
echo "other compute processes on the GPU (should be none):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

MODE="${MODE:-quick}"
case "$MODE" in smoke|quick|full) ;; *) die "MODE must be smoke, quick or full (got '$MODE')" ;; esac
LOCK="${CLOCK_LOCK:-none}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
RUN_DIR="${RUN_DIR:-$REPO/pipeline/runs/$MODE-slurm}"

ARGS=("--$MODE" --run-dir "$RUN_DIR" --skip-install --workers "$WORKERS" --clock-lock "$LOCK")
if [ "$LOCK" = "none" ]; then
    echo "WARNING: clocks are not locked (a job cannot lock them): compare regret and ratios within this GPU, not times with the 3080's 1755 MHz runs"
    ARGS+=(--allow-unlocked)
fi
[ "${SKIP_TESTS:-0}" = "1" ] && ARGS+=(--skip-tests)
[ "${FORCE:-0}" = "1" ] && ARGS+=(--force)
[ -n "${STAGES:-}" ] && ARGS+=(--only "$STAGES")
[ -n "${FROM:-}" ] && ARGS+=(--from "$FROM")

echo "pipeline/run.sh ${ARGS[*]}"
# exec: SLURM's SIGTERM at the time limit goes straight to the pipeline's trap, which stops the sweep workers (no orphans).
exec pipeline/run.sh "${ARGS[@]}"
