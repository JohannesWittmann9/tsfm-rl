#!/bin/bash
# Run the citylearn_exog_forecast experiment with parallelized training.
set -e
cd "$(dirname "$0")/.."

uv sync --quiet

for arm in baseline demand_oracle all_oracle; do
    for s in 0 1 2; do echo "$arm $s"; done
done | THREADS=2 xargs -P 4 -L1 uv run --no-sync python run_train.py &
TRAIN_PID=$!

uv run --no-sync python run_forecasts.py
wait $TRAIN_PID

for arm in price weather demand all; do
    for s in 0 1 2; do echo "$arm $s"; done
done | THREADS=2 xargs -P 5 -L1 uv run --no-sync python run_train.py

uv run --no-sync python run_evaluate.py
