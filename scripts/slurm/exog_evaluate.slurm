#!/bin/bash
#SBATCH --job-name=exog_evaluate
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Step 3: roll out all trained policies + reference agents, write
# results/kpis.csv and the figures.
set -e
cd "${SLURM_SUBMIT_DIR:?submit via scripts/slurm/submit_exog.sh from the repo root}"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/var/tmp/${USER}_uv_cache"
export HF_HOME="/var/tmp/${USER}_hf_cache"

cd experiments/citylearn_exog_forecast
uv run --no-sync python run_evaluate.py
