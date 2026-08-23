#!/bin/bash
#SBATCH --job-name=tsfm_context_sweep
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source ~/rl-env/bin/activate

export WANDB_DIR="/var/tmp/${USER}_wandb"
mkdir -p "$WANDB_DIR"

python3 -u sweep_context_length.py \
    --timesteps 50000 \
    --offline-steps 2000 \
    --model-name "amazon/chronos-2" \
    --device cuda \
    --train-schema "citylearn_challenge_2023_phase_1" \
    --eval-schema "citylearn_challenge_2023_phase_1" \
    --wandb-project "CityLearn-TSFM-RL" \
    --wandb-group "context-length-sweep" \
    --wandb-run-name "slurm_context_sweep_${SLURM_JOB_ID}" \
    --output-dir "./experiments/run_${SLURM_JOB_ID}"

echo "Job $SLURM_JOB_ID finished successfully."