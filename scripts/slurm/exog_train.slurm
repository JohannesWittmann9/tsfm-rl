#!/bin/bash
#SBATCH --job-name=exog_train
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --array=0-20%14
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Step 2: one array task per SAC run (7 arms x 3 seeds = tasks 0-20).
# Finished runs are skipped via the models/sac_<arm>_s<seed>.pt cache, so
# the array can simply be resubmitted after failures.
set -e
cd "${SLURM_SUBMIT_DIR:?submit via scripts/slurm/submit_exog.sh from the repo root}"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/var/tmp/${USER}_uv_cache"
export HF_HOME="/var/tmp/${USER}_hf_cache"

ARMS=(baseline price weather demand demand_oracle all all_oracle)
arm=${ARMS[SLURM_ARRAY_TASK_ID / 3]}
seed=$((SLURM_ARRAY_TASK_ID % 3))

cd experiments/citylearn_exog_forecast
echo "task ${SLURM_ARRAY_TASK_ID}: training ${arm} seed ${seed} on $(hostname)"
THREADS=2 uv run --no-sync python run_train.py "$arm" "$seed"
