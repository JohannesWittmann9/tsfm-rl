#!/bin/bash
#SBATCH --job-name=citylearn_tsfm
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Load Python virtual environment
source ~/rl-env/bin/activate

# Direct Hugging Face & W&B caching to scratch storage to avoid NFS quotas
export WANDB_DIR="/var/tmp/${USER}_wandb"
mkdir -p "$WANDB_DIR"

# Launch training with structured project/group tracking
python3 -u train_citylearn_tsfm.py \
    --timesteps 100000 \
    --context-length 16 \
    --device cuda \
    --eval-steps 1000 \
    --wandb-project "CityLearn-TSFM-RL" \
    --wandb-group "chronos2-context16" \
    --wandb-run-name "slurm_job_${SLURM_JOB_ID}" \
    --output-dir "./experiments/run_${SLURM_JOB_ID}"

echo "Job $SLURM_JOB_ID finished."