#!/bin/bash
#SBATCH --job-name=forecast_train
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --array=0-26%14
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Step 2 of the single-building forecast experiment: one array task per SAC
# run (9 arms x 3 seeds = tasks 0-26). Finished runs are skipped via the
# models/sac_b1_<arm>_s<seed>.pt cache, so the array can simply be
# resubmitted after failures. Needs data/forecast_chronos_b1.csv from step 1.
set -e
cd "${SLURM_SUBMIT_DIR:?submit via scripts/slurm/submit_forecast.sh from the repo root}"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/var/tmp/${USER}_uv_cache"
export HF_HOME="/var/tmp/${USER}_hf_cache"

ARMS=(baseline price weather demand_points demand_volumes seasonal_naive_points seasonal_naive_volumes all all_oracle)
arm=${ARMS[SLURM_ARRAY_TASK_ID / 3]}
seed=$((SLURM_ARRAY_TASK_ID % 3))

cd experiments/citylearn_exog_forecast
echo "task ${SLURM_ARRAY_TASK_ID}: training ${arm} seed ${seed} on $(hostname)"
THREADS=2 uv run --no-sync python forecast_train.py "$arm" "$seed"
