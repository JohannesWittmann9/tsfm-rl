"""Step 2: train the SAC runs (7 arms x 3 seeds).

python run_train.py
python run_train.py demand 1         # one specific run
THREADS=4 python run_train.py ...    # cap torch CPU threads

the loop can be parallelized across shells, e.g.:

for arm in baseline price weather demand demand_oracle all all_oracle; do
    for s in 0 1 2; do THREADS=2 python run_train.py $arm $s; done &
done; wait

Chronos arms need data/chronos_forecasts.parquet from run_forecasts.py.
"""

import os
import sys
import time

import torch

if os.environ.get("THREADS"):
    torch.set_num_threads(int(os.environ["THREADS"]))

import common


def main():
    if len(sys.argv) == 3:
        runs = [(sys.argv[1], int(sys.argv[2]))]
    else:
        runs = [(arm, seed) for arm in common.RL_ARMS for seed in common.SEEDS]
    for arm, seed in runs:
        tag = f"{arm}_s{seed}"
        if (common.MODELS_DIR / f"sac_{tag}.pt").exists():
            print(f"{tag}: cached, skipping")
            continue
        t0 = time.time()
        common.train_sac(arm, seed)
        print(f"{tag}: done in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
