"""Run every dyna_standard stage for every environment, one after another.

The whole study in one command::

    python experiments/dyna_standard/run_all.py

Everything ``<env>/<env>.py`` accepts works here too, plus ``--envs``::

    python experiments/dyna_standard/run_all.py --stages grid
    python experiments/dyna_standard/run_all.py --envs Pendulum-v1 Acrobot-v1
    python experiments/dyna_standard/run_all.py --windows 4 --budgets 16 64 --models "Chronos-2 S" MLP VARX

Each environment writes its own ``<env>/results/*.csv`` and every stage resumes,
so re-running after an interruption picks up where it stopped. An environment
that fails is reported at the end; the others still run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline

if __name__ == "__main__":
    pipeline.cli()
