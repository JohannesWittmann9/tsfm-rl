#!/bin/bash
# Submit the single-building forecast experiment as a Slurm dependency chain:
# Chronos forecasts -> training array (21 tasks).
# Run from anywhere: bash scripts/slurm/submit_forecast.sh
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

fc=$(sbatch --parsable scripts/slurm/forecast_chronos.slurm)
tr=$(sbatch --parsable --dependency=afterok:"$fc" scripts/slurm/forecast_train.slurm)
echo "submitted: forecasts=$fc train=$tr"
echo "afterwards: run the evaluation cells in experiments/citylearn_exog_forecast/forecast.ipynb"
