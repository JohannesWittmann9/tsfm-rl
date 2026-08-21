#!/bin/bash
#SBATCH --job-name=citylearn_modelfree
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source ~/rl-env/bin/activate

export WANDB_DIR="/var/tmp/${USER}_wandb"
mkdir -p "$WANDB_DIR"

python3 -u train_citylearn_modelfree.py \
    --timesteps 100000 \
    --device cuda \
    --train-schema "citylearn_challenge_2023_phase_1" \
    --eval-schema "citylearn_challenge_2023_phase_1" \
    --wandb-project "CityLearn-ModelFree-PPO" \
    --wandb-group "modelfree-multiseed-annual-job-${SLURM_JOB_ID}" \
    --wandb-run-name "slurm_job_${SLURM_JOB_ID}" \
    --output-dir "./results_modelfree/run_${SLURM_JOB_ID}"

EXIT_CODE=$?
echo "=== Slurm Job $SLURM_JOB_ID completed with exit code $EXIT_CODE ==="
exit $EXIT_CODE