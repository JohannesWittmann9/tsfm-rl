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
#
MODELS = {
    "Chronos-2 S": dict(
        kind="chronos",
        model_id="autogluon/chronos-2-small",  # 27.9M
        color="#007BFF",
        marker="o",
        cost=1.0,
    ),
    "Chronos-2 L": dict(
        kind="chronos",
        model_id="amazon/chronos-2",  # 119.5M
        color="#FF6347",
        marker="s",
        cost=3.4,
    ),
    "Chronos-2 L-syn": dict(
        kind="chronos",
        model_id="autogluon/chronos-2-synth",  # 119.0M, synthetic
        color="#28A745",
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
        color="#FFC107",
        marker="P",
        cost=8.6,
    ),
    "MLP": dict(kind="mlp", lags=[1, 4, 16], color="#046307", marker="^"),
    "VARX": dict(kind="varx", lags=[1, 4, 16], color="#d76fa8", marker="v"),
}

TSFM_KINDS = ("chronos", "moirai")
TSFM_MODELS = [m for m, c in MODELS.items() if c["kind"] in TSFM_KINDS]

# ---- which models appear where ----------------------------------------------
PROBE_MODELS = list(TSFM_MODELS)  # section 2b (the presentation probe)
GRID_MODELS = list(MODELS)  # section 5 (the hyperparameter tables)
GRID_FULL = list(TSFM_MODELS)  # get the full presentation x r grid; anything
# left out gets the differenced sweep plus
# the two `level` reference points
PLOT_MODELS = [
    "Chronos-2 S",
    "Chronos-2 S (level)",
    "Chronos-2 L",
    "Chronos-2 L-syn",
    "Moirai",
    "MLP",
    "VARX",
]
"""Models drawn in the context-budget plot (§6) and the rollout plot (§7)."""

STRETCH_BUDGET = 256
"""The budget §5's stretch figure is drawn at."""

ROLL_CONFIG_BUDGET = 512
"""The budget §7a's per-configuration rollout figure is drawn at. It already
spends a panel per model per environment, so the budget axis stays a single
value rather than becoming a third dimension."""

# ---- how the last two figures are drawn -------------------------------------
# Everything in this block is read when the figure is drawn, never when a stage
# computes. Change one, re-run the notebook cell, and the figure changes: none of
# it costs a recompute. A model left out here is still measured and still in the
# CSV, it is only absent from the panel.
SCALING_MODELS = None
"""§6's models. None follows PLOT_MODELS."""

ROLLOUT_MODELS = None
"""§7's models. None follows PLOT_MODELS."""

PLOT_VARIANTS = {}
"""Which *configuration* each model is drawn in, in §6 and §7.

`SCALING_MODELS` and `ROLLOUT_MODELS` say which models appear; this says which of
their variants. Empty means "whatever the section 5 grid selects". An entry
overrides that pick, in the same dict shape `MODELS[...]["fixed"]` uses, so there
is one way to name a variant in this file::

    PLOT_VARIANTS = {
        # None: applies to every environment
        None: {"Chronos-2 S": dict(presentation="diff", r=6)},
        # a named environment overrides the None block, for that environment only
        "Pendulum-v1": {"Chronos-2 L": dict(presentation="diff", r=12)},
        "CartPole-v1": {"MLP": dict(lag=4)},
    }

A model listed in neither block keeps the automatic selection. A variant that the
model does not have raises, naming the ones it does -- a typo must not quietly
redraw a different curve.

Free in §7: `rollout.csv` holds every variant, so the override is applied when the
figure is drawn -- in **both** §6 and §7, since the grid now covers the whole
budget axis and neither figure has any measurement of its own left to redo.

Pick the configuration off §6a and §7a, name it here, re-run the two cells.
"""

NMSE_YLIM = (1e-4, 2.0)
"""The log band §6 and §7 are drawn in. Errors span six decades -- a linear axis
collapses everything interesting and an unclipped log axis is mostly empty -- so
a curve outside this band is pinned to the edge and flagged with a triangle."""

SCALING_YLIM = None
"""§6's own band, when it wants one. None follows NMSE_YLIM."""

ROLLOUT_YLIM = None
"""§7's own band, when it wants one. None follows NMSE_YLIM."""

SCALE_PLOT_BUDGETS = None
"""Which of the computed SCALE_BUDGETS §6 draws. None draws all of them."""

ROLL_PLOT_H = 20
"""How far along the horizon §7 and §8 draw. None draws the whole computed
ROLL_H, so raising ROLL_H can stay a compute decision and this stays the
presentation one."""

# --------------------------------------------------------------------- sweeps
PRESENTATIONS = ["diff", "level"]
STRETCH_R = [1, 2, 4, 6, 8, 12, 16]
# 1 and 2 are the zero-shot end: no context to speak of for a foundation model,
# nothing to fit for a trained one. Most of those columns come out blank, and
# what does not is the point of including them.
SCALE_BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
HP_BUDGETS = SCALE_BUDGETS

ROLL_BUDGETS = SCALE_BUDGETS
ROLL_PLOT_BUDGETS = [64, 512, 1024, 2048, 4096]  # every one of them in ROLL_BUDGETS
"""Which of the computed ROLL_BUDGETS §7 draws -- it spends a row per budget and
gets tall fast, so the sweep is wider than the figure. None draws all of them."""
ROLL_H = 50
TRAJ_WINDOWS = 3  # example windows kept per environment

TRAJ_BUDGETS = [64, 512, 4096]


TRAJ_BUDGET = 512
"""Which of TRAJ_BUDGETS section 8 draws. Read when the figure is drawn, so
switching N is an edit and a re-run of the cell."""

SELECT_RULE = "mean_rank"

# -------------------------------------------------------------------- protocol
SEED = 0
EPISODE_LEN = 200  # short episodes: figures 1, 1b, 4 and the 2b probe
N_EPISODES = 50
L = 64  # context window for the 2b probe, identical per model

PROBE_WINDOWS = 96  # evaluation windows behind the probe
PROBE_CTX = 96  # contexts the probe is averaged over
HP_WINDOWS = 96  # evaluation windows for the grid -- the main cost lever
ROLL_WINDOWS = 32  # evaluation windows for the rollout trajectories

HP_EVAL_EPISODES = 48  # long contexts overlap, so one episode per window
HP_POOL_EPISODES = 4  # only the first is used; the rest are slack
HP_MARGIN = 32  # headroom between episode length and context

CHRONOS_BATCH = 16  # series per pipeline call (compute only, not a result)
MOIRAI_SAMPLES = 10  # Moirai is a sampling forecaster; median over N draws
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
