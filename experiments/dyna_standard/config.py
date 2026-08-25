"""Global configuration for the dyna_standard study.

Everything that is a *choice* rather than a mechanism lives here: which
environments run, which models exist, which models appear in which figure, the
sweeps, and the protocol constants. The stages in ``pipeline.py`` and the figures
in ``plots.py`` read this module at call time, so changing a value here is the
whole edit -- add a line to :data:`MODELS` and it enters every stage, every table
and every legend.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- environments
# Discovered from the subfolders: every ``<env>/env.py`` defines one environment
# and its ordering, so adding an environment is a folder, not an edit here.
from envs import DEFAULT_R, ENV_DIRS, ENV_IDS, SHORT  # noqa: F401

ENVS = list(ENV_IDS)

# The environment the two illustration figures (1b and 4) are drawn on. They
# explain a transform, so they need one concrete trajectory rather than all four.
ILLUSTRATION_ENV = "Pendulum-v1"

# --------------------------------------------------------------------- models
# kind      how the model is built (see models/__init__.py)
# color     fixed per model, so a model keeps its colour in every figure
# marker    identity is carried by shape too, never by colour alone
# lags      trained models only: the history lengths swept in the grid
MODELS = {
    "Chronos-2 S": dict(
        kind="chronos",
        model_id="autogluon/chronos-2-small",  # 27.9M
        color="#2a78d6",
        marker="o",
        cost=1.0,
    ),
    "Chronos-2 L": dict(
        kind="chronos",
        model_id="amazon/chronos-2",  # 119.5M
        color="#e34948",
        marker="s",
        cost=3.4,
    ),
    "Chronos-2 L-syn": dict(
        kind="chronos",
        model_id="autogluon/chronos-2-synth",  # 119.0M, synthetic
        color="#4a3aa7",
        marker="D",
        cost=3.4,
    ),
    # Chronos-2 small with neither transform: the out-of-the-box reference the
    # rest of the study is measured against. `fixed` pins its variant, so it is
    # not swept -- there is nothing to sweep.
    "Chronos-2 S (level)": dict(
        kind="chronos",
        model_id="autogluon/chronos-2-small",
        fixed=dict(presentation="level", r=1),
        color="#6b6b6b",
        marker="o",
        cost=1.0,
    ),
    "Moirai": dict(
        kind="moirai",
        model_id="Salesforce/moirai-1.1-R-small",
        color="#eb6834",
        marker="s",
        cost=8.6,
    ),
    "MLP": dict(kind="mlp", lags=[1, 4, 16], color="#1baf7a", marker="^"),
    "VARX": dict(kind="varx", lags=[1, 4, 16], color="#eda100", marker="v"),
}

TSFM_KINDS = ("chronos", "moirai")
TSFM_MODELS = [m for m, c in MODELS.items() if c["kind"] in TSFM_KINDS]

# ---- which models appear where ----------------------------------------------
PROBE_MODELS = list(TSFM_MODELS)  # section 2b (the presentation probe)
GRID_MODELS = list(MODELS)  # section 5 (the hyperparameter tables)
GRID_FULL = ["Chronos-2 S", "Moirai"]  # get the full presentation x r grid;
# the rest get the differenced sweep
# plus the two `level` reference points
PLOT_MODELS = [
    "Chronos-2 S",
    "Chronos-2 S (level)",
    "Chronos-2 L",
    "Moirai",
    "MLP",
    "VARX",
]
"""Models drawn in the context-budget plot (§6) and the rollout plot (§7)."""

STRETCH_BUDGET = 256
"""The budget §5's stretch figure is drawn at."""

# --------------------------------------------------------------------- sweeps
PRESENTATIONS = ["diff", "level"]
STRETCH_R = [1, 2, 4, 8, 16]
# 1 and 2 are the zero-shot end: no context to speak of for a foundation model,
# nothing to fit for a trained one. Most of those columns come out blank, and
# what does not is the point of including them.
HP_BUDGETS = [1, 2, 16, 64, 256, 1024, 4096]
SCALE_BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
ROLL_BUDGETS = [64, 512, 4096]
ROLL_H = 20
TRAJ_BUDGET = 4096  # the budget the rollout trajectories (section 8) are drawn at
TRAJ_WINDOWS = 3  # example windows kept per environment

# The stretch factor the section 2b probe uses (it runs before the grid has
# measured anything) comes from each `<env>/env.py` as `default_r`, imported
# above as DEFAULT_R. Section 5 reports what the sweep actually prefers.

# How section 5 picks the variant carried into sections 6 and 7. "mean_rank"
# averages each variant's rank across budgets (the errors span six decades, so a
# mean error would be decided by the largest budget alone) and breaks ties on the
# geometric mean. "geo_nmse" selects on the geometric mean directly.
SELECT_RULE = "mean_rank"

# -------------------------------------------------------------------- protocol
SEED = 0
EPISODE_LEN = 200  # short episodes: figures 1, 1b, 4 and the 2b probe
N_EPISODES = 30
L = 64  # context window for the 2b probe, identical per model

PROBE_WINDOWS = 96  # evaluation windows behind the probe
PROBE_CTX = 96  # contexts the probe is averaged over
HP_WINDOWS = 24  # evaluation windows for the grid -- the main cost lever
ROLL_WINDOWS = 24

HP_EVAL_EPISODES = 48  # long contexts overlap, so one episode per window
HP_POOL_EPISODES = 4  # only the first is used; the rest are slack
HP_MARGIN = 32  # headroom between episode length and context

CHRONOS_BATCH = 16  # series per pipeline call (compute only, not a result)
MOIRAI_SAMPLES = 20  # Moirai is a sampling forecaster; median over N draws
CHRONOS_CTX = 8192  # Chronos-2 token limit
MOIRAI_CTX = 5000  # no published limit; kept well inside memory

# ----------------------------------------------------------------------- paths
RESULTS_DIRNAME = os.environ.get("DYNA_RESULTS", "results")
"""Subfolder the stage CSVs live in. Overridden by ``--results-dir`` on the
command line, or by the ``DYNA_RESULTS`` environment variable for a notebook."""


def env_dir(env_id: str) -> Path:
    """The subfolder holding one environment's script, notebook and results."""
    return ROOT / ENV_DIRS[env_id]


def results_dir(env_id: str | None = None) -> Path:
    """Where a stage's CSV lives. ``None`` is the study-level directory."""
    base = ROOT if env_id is None else env_dir(env_id)
    return base / RESULTS_DIRNAME
