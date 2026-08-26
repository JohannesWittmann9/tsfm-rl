#!/bin/bash
#SBATCH --job-name=dyna_standard
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# One array task per environment: each writes its own
# experiments/dyna_standard/<env>/results/*.csv, so the four never touch the
# same file and a failure in one leaves the others alone. Every stage resumes
# from its CSV, so a task that hits the time limit can simply be resubmitted.
#
# Submit via scripts/slurm/submit_dyna.sh, which runs the setup job first.
set -e
cd "${SLURM_SUBMIT_DIR:?submit via scripts/slurm/submit_dyna.sh from the repo root}"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/var/tmp/${USER}_uv_cache"
export HF_HOME="/var/tmp/${USER}_hf_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# The array index picks the folder; keep this list in step with envs.py, which
# is what actually decides which environments exist.
ENVS=(pendulum mountaincar acrobot cartpole)
env=${ENVS[SLURM_ARRAY_TASK_ID]}

echo "task ${SLURM_ARRAY_TASK_ID}: ${env} on $(hostname)"
uv run --no-sync python -c "import torch; print('cuda:', torch.cuda.is_available())"

uv run --no-sync python "experiments/dyna_standard/${env}/${env}.py"
