#!/bin/bash
#SBATCH --job-name=dyna_ppo
#SBATCH --partition=All
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# One array task per (environment, model, N, seed) cell of the sweep. Each task
# writes only its own experiments/dyna_standard_ppo/logs/<run>/, so the tasks
# never touch the same file, and a finished run is skipped via its
# best_model.zip -- the array can simply be resubmitted after failures.
#
# The array range comes from the notebook's own grid, so it is set by
# scripts/slurm/submit_dyna_ppo.sh rather than pinned here.
#
# The slowest cell is Chronos-2 S (diff, r=6) at N=1024: ~1.7 h on CPU at
# TOTAL_TIMESTEPS=20_000.

set -e
cd "${SLURM_SUBMIT_DIR:?submit via scripts/slurm/submit_dyna_ppo.sh from the repo root}"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/var/tmp/${USER}_uv_cache"
export HF_HOME="/var/tmp/${USER}_hf_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# CPU only. This partition allocates no GPU, but the nodes have one and
# torch will take it if it can see it -- with several array tasks per node
# that means several Chronos models on one unallocated device, which is what
# killed job 218117. Hiding the device is what makes the run reproducible.
export CUDA_VISIBLE_DEVICES=""

RUNNER=experiments/dyna_standard_ppo/run_train.py

cell=$(uv run --no-sync python "$RUNNER" --list | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$cell" ]; then
    echo "no grid cell at index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi
IFS='|' read -r env model n seed <<<"$cell"

echo "task ${SLURM_ARRAY_TASK_ID}: ${env} | ${model} | N=${n} | seed=${seed} on $(hostname)"
uv run --no-sync python -c "import torch; print('cuda:', torch.cuda.is_available(), '| threads:', torch.get_num_threads())"

uv run --no-sync python "$RUNNER" "$env" "$model" "$n" "$seed"
