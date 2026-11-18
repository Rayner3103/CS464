#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=2           # CPUs: the noise study times a handful of kernels, no parallel compile
#SBATCH --mem=8GB                   # How much memory does your job require?
#SBATCH --gres=gpu:1                # One GPU: the measurements need the whole card to themselves
#SBATCH --time=01-00:00:00          # <DD-HH:MM:SS>. Generous: the study is 30 shapes x 2 backends x cold/hot x 3 rounds
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=noise_%j.out       # Log file, in the directory you submit from

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=student                 # The partition you've been assigned
#SBATCH --account=cs701                     # The account you've been assigned (normally student)
#SBATCH --qos=studentqos                    # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=rayner.sim.2024@computing.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=noise-gemm               # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Noise-floor study (python -m tune noise --rounds 3 --repeats 50 --n-allocs 3, AGENTS.md section 3) on whatever GPU SLURM gives this job.
# It gives the GPU its own tie margin; without it the 3090's regret numbers have no noise floor to be read against.
# Output: $REPO/data/<hw_id>/noise.jsonl (+ .raw.jsonl); read it afterwards on the login node / locally with `python -m tune analyze`-style tooling.
#
# Same login-node setup as remoteGPU/train_best.sh (venv at $REPO/.venv). Submit from the repo:   sbatch remoteGPU/noise.sh
#   EXPECT_GPU=3090   substring the GPU name must contain, or the job stops (EXPECT_GPU=any disables it)
#   CLOCK_LOCK=none   declared SM clock lock in MHz; a job cannot run `nvidia-smi -lgc`, so the default is unlocked
#   ROUNDS=3 REPEATS=50 N_ALLOCS=3   study size (the AGENTS.md invocation)
#   REPO=DIR          the repo checkout (default ~/repos/CS464)
#   e.g.  EXPECT_GPU=3090 sbatch remoteGPU/noise.sh
#
# Caveat: one job is one session. The study's rounds are inside one process; a second job (another day, ideally after a node reboot) is
# what gives a true cross-session number. Unlocked clocks make the floor larger than the 3080's locked 1755 MHz one: compare with that in mind.

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

if command -v module >/dev/null 2>&1 || type module >/dev/null 2>&1; then
    module purge
    module load Python/3.12.8-GCCcore-13.3.0
fi

REPO="${REPO:-$HOME/repos/CS464}"
[ -f "$REPO/.venv/bin/activate" ] || die "no virtualenv at $REPO/.venv: see the login-node steps in remoteGPU/train_best.sh"
source "$REPO/.venv/bin/activate"
cd "$REPO"
command -v gcc >/dev/null || die "no gcc on PATH: Triton cannot build its launchers"

EXPECT_GPU="${EXPECT_GPU:-3090}"
LOCK="${CLOCK_LOCK:-none}"
ROUNDS="${ROUNDS:-3}"
REPEATS="${REPEATS:-50}"
N_ALLOCS="${N_ALLOCS:-3}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname), started $(date '+%F %T'), repo $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [ "$EXPECT_GPU" != "any" ] && [[ "$GPU_NAME" != *"$EXPECT_GPU"* ]]; then
    die "this node has '$GPU_NAME', expected a GPU containing '$EXPECT_GPU' (EXPECT_GPU=any disables the check)"
fi
echo "other compute processes on the GPU (should be none):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

[ "$LOCK" = "none" ] && echo "WARNING: clocks are not locked (a job cannot lock them): the floor measured here includes clock wander"

echo "python -m tune --clock-lock $LOCK noise --rounds $ROUNDS --repeats $REPEATS --n-allocs $N_ALLOCS"
exec python -m tune --clock-lock "$LOCK" noise --rounds "$ROUNDS" --repeats "$REPEATS" --n-allocs "$N_ALLOCS"
