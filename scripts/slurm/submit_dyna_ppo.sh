#!/bin/bash
# Submit the dyna_standard_ppo sweep as one Slurm array task per grid cell.
# The grid is read from the notebook, so changing ENV_IDS, N_VALUES,
# MODEL_NAMES or POLICY_SEEDS there is the whole edit.
#
#   bash scripts/slurm/submit_dyna_ppo.sh            # drip-feed the whole grid
#   bash scripts/slurm/submit_dyna_ppo.sh 0 24       # one explicit chunk, now
#   nohup bash scripts/slurm/submit_dyna_ppo.sh &    # unattended, overnight
#
# Slurm counts every array task against AssocMaxSubmitJobLimit, so a grid larger
# than that limit cannot be submitted in one go -- and the `%N` throttle does not
# help, it caps what runs, not what is queued. With no arguments this submits
# CHUNK cells at a time and waits for the queue to drain before the next chunk.
# Finished cells skip themselves, so chunks may overlap and the whole thing can
# be rerun safely.
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

CHUNK=${CHUNK:-25}        # cells per array; keep under MaxSubmit
CONCURRENT=${CONCURRENT:-15}  # cells running at once; keep at or under MaxJobs
HEADROOM=${HEADROOM:-2}   # queued jobs tolerated before submitting the next chunk

RUNNER=experiments/dyna_standard_ppo/run_train.py
n_cells=$(uv run --no-sync python "$RUNNER" --list | wc -l)
if [ "$n_cells" -lt 1 ]; then
    echo "no cells in the grid -- check ENV_IDS / N_VALUES / MODEL_NAMES"
    exit 1
fi

submit_range() {
    local start=$1 end=$2
    [ "$end" -ge "$n_cells" ] && end=$((n_cells - 1))
    local job
    job=$(sbatch --parsable --array="${start}-${end}%${CONCURRENT}" \
          scripts/slurm/dyna_ppo.slurm)
    echo "$(date +%H:%M)  submitted cells ${start}-${end} as job ${job}"
}

if [ $# -ge 2 ]; then
    submit_range "$1" "$2"
    exit 0
fi

echo "grid: ${n_cells} cells, ${CHUNK} per chunk, ${CONCURRENT} concurrent"
start=0
while [ "$start" -lt "$n_cells" ]; do
    # wait for the queue to drain before adding the next chunk
    while [ "$(squeue -u "$USER" -h | wc -l)" -gt "$HEADROOM" ]; do
        sleep 60
    done
    submit_range "$start" $((start + CHUNK - 1))
    sleep 30  # let the scheduler register the new array before re-checking
    start=$((start + CHUNK))
done
echo "$(date +%H:%M)  all ${n_cells} cells submitted"
echo "evaluation stays in the notebook: run the cells from 'Load best policy' down"
