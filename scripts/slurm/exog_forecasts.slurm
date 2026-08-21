#!/bin/bash
#SBATCH --job-name=exog_forecasts
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Step 1: precompute the Chronos forecast parquet (skipped if it already
# exists). Also does the one-time `uv sync` so the training array can run
# with --no-sync. Submitted via scripts/slurm/submit_exog.sh.
set -e
cd "${SLURM_SUBMIT_DIR:?submit via scripts/slurm/submit_exog.sh from the repo root}"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Caches auf die lokale Platte statt ins Home (Fileserver-Regeln der RBG)
export UV_CACHE_DIR="/var/tmp/${USER}_uv_cache"
export HF_HOME="/var/tmp/${USER}_hf_cache"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

uv sync
cd experiments/citylearn_exog_forecast

# CityLearn-Datensatz in ~/.cache/citylearn holen, falls noch nicht vorhanden
uv run --no-sync python - <<'EOF'
from pathlib import Path

source_dir = (
    Path.home() / ".cache/citylearn/v2.5.0/datasets/citylearn_challenge_2022_phase_1"
)
if not (source_dir / "Building_1.csv").exists():
    from citylearn.data import DataSet

    DataSet().get_dataset("citylearn_challenge_2022_phase_1")
EOF

uv run --no-sync python run_forecasts.py
