#!/bin/bash
#SBATCH --job-name=citylearn_exog_forecast
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Runs the citylearn_exog_forecast experiment end to end (forecasts -> 7 arms
# x 3 seeds SAC training -> evaluation) via the existing parallel runner.
# Submit from the repo root: sbatch scripts/slurm/run_citylearn.slurm
set -e
cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repo root}"

# uv bereitstellen (einmalige Installation nach ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Caches auf die lokale Platte statt ins Home (Fileserver-Regeln der RBG)
export UV_CACHE_DIR="/var/tmp/${USER}_uv_cache"
export HF_HOME="/var/tmp/${USER}_hf_cache"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

echo "=== citylearn_exog_forecast on $(hostname), job ${SLURM_JOB_ID} ==="
bash experiments/citylearn_exog_forecast/run_all_parallel.sh
echo "Job ${SLURM_JOB_ID} finished."
