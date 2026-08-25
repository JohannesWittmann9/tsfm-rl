"""Run every dyna_standard stage for CartPole-v1.

Compute lives here, not in the notebook: this writes cartpole/results/*.csv and
``cartpole_experiment.ipynb`` only reads them.

    python experiments/dyna_standard/cartpole/cartpole.py                 # all stages
    python experiments/dyna_standard/cartpole/cartpole.py --stages grid   # just one
    python experiments/dyna_standard/cartpole/cartpole.py --windows 4 --budgets 16 64         --results-dir results_smoke                                   # a smoke run

Stages resume from the CSVs, so an interrupted run picks up where it stopped and
adding a model to ``config.MODELS`` computes only the new rows.

``../run_all.py`` runs this for every environment in one command.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline

ENV_ID = "CartPole-v1"

if __name__ == "__main__":
    pipeline.cli(ENV_ID)
