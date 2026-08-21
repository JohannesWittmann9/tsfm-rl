#!/bin/bash
# Submit the citylearn_exog_forecast pipeline as a Slurm dependency chain:
# forecasts -> training array (21 tasks) -> evaluate.
# Run from anywhere: bash scripts/slurm/submit_exog.sh
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm legt das Log-Verzeichnis nicht selbst an

fc=$(sbatch --parsable scripts/slurm/exog_forecasts.slurm)
tr=$(sbatch --parsable --dependency=afterok:"$fc" scripts/slurm/exog_train.slurm)
ev=$(sbatch --parsable --dependency=afterok:"$tr" scripts/slurm/exog_evaluate.slurm)
echo "submitted: forecasts=$fc train=$tr evaluate=$ev"
