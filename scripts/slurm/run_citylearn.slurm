#!/bin/bash
#SBATCH --job-name=citylearn_e2e_multiseed
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Load Python virtual environment
source ~/rl-env/bin/activate

export WANDB_DIR="/var/tmp/${USER}_wandb"
mkdir -p "$WANDB_DIR"

echo "=== Launching Python Experiment ==="

python3 -u train_citylearn_tsfm.py \
    --timesteps 100000 \
    --context-length 16 \
    --model-name "amazon/chronos-2" \
    --device cuda \
    --wandb-project "CityLearn-TSFM-RL" \
    --wandb-group "e2e-multiseed-annual" \
    --wandb-run-name "slurm_job_${SLURM_JOB_ID}" \
    --output-dir "./experiments/run_${SLURM_JOB_ID}"

echo "Job $SLURM_JOB_ID finished."