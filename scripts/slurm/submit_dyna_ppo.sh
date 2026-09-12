#!/bin/bash
# Submit the dyna_standard_ppo sweep as one Slurm array task per grid cell.
# The array size is read from the notebook's own grid, so changing ENV_IDS,
# N_VALUES, MODEL_NAMES or POLICY_SEEDS there is the whole edit.
# Run from anywhere: bash scripts/slurm/submit_dyna_ppo.sh
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

RUNNER=experiments/dyna_standard_ppo/run_train.py
n_cells=$(uv run --no-sync python "$RUNNER" --list | wc -l)
if [ "$n_cells" -lt 1 ]; then
    echo "no cells in the grid -- check ENV_IDS / N_VALUES / MODEL_NAMES"
    exit 1
fi

# %6 keeps six cells running at once; raise it if the partition is free.
job=$(sbatch --parsable --array=0-$((n_cells - 1))%6 scripts/slurm/dyna_ppo.slurm)
echo "submitted ${n_cells} cells as job ${job}"
echo "evaluation stays in the notebook: run the cells from 'Load best policy' down"
