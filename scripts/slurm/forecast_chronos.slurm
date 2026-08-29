#!/bin/bash
#SBATCH --job-name=forecast_chronos
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Step 1 of the single-building forecast experiment: rolling Chronos-2
# forecasts -> data/forecast_chronos_b1.csv. Skips itself if the CSV exists.
set -e
cd "${SLURM_SUBMIT_DIR:?submit via scripts/slurm/submit_forecast.sh from the repo root}"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/var/tmp/${USER}_uv_cache"
export HF_HOME="/var/tmp/${USER}_hf_cache"

cd experiments/citylearn_exog_forecast
uv run --no-sync python forecast_chronos.py
