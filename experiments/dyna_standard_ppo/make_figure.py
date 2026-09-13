"""The paper figures for the PPO experiments.

Writes ppo_policy.{pdf,png} from ppo_results.csv and ppo_learning.{pdf,png}
from the monitor logs. Colours and markers come from dyna_standard's
config.MODELS so both match the forecasting figures of the same study.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dyna_standard"))
import config

# notebook label -> (registry entry supplying the colour, legend label, marker).
# The r=1 variant borrows Moirai's amber: it is the same checkpoint as r=6 but a
# different presentation, and it needs to be separable at a glance. Moirai does
# not appear in these experiments, so the colour is free.
STYLE = {
    "MLP": ("MLP", "MLP", None),
    "VARX": ("VARX", "VARX", None),
    "Chronos-2 S (level)": ("Chronos-2 S (level)", "Chronos-2 S (level)", None),
    "Chronos-2 S (diff, r=6)": ("Chronos-2 S", "Chronos-2 S, $r=6$", "o"),
    "Chronos-2 S (diff, r=1)": ("Moirai", "Chronos-2 S, $r=1$", "s"),
}
ENVS = ["CartPole-v1", "MountainCar-v0"]
TITLES = {"CartPole-v1": "CartPole", "MountainCar-v0": "MountainCar"}
REFS = {  # random-policy and solved levels, named in the caption
    "CartPole-v1": [22.0, 475.0],
    "MountainCar-v0": [-200.0, -110.0],
}
N_VALUES = [64, 256, 1024]
TOTAL_TIMESTEPS = 20_000
# Both environments: the in-model return on MountainCar is trustworthy again now
# that a clipped prediction can no longer register as goal-reached.
LEARN_ENVS = ENVS


def style(label):
    key, lab, mk = STYLE[label]
    cfg = config.MODELS[key]
    return cfg["color"], mk or cfg["marker"], lab


def legend(fig, extra=(), ncol=3, y=-0.02):
    handles = [
        plt.Line2D([], [], color=style(k)[0], marker=style(k)[1], ms=5, lw=1.6,
                   label=style(k)[2])
        for k in STYLE
    ]
    fig.legend(handles=list(handles) + list(extra), loc="lower center", ncol=ncol,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, y))


# --------------------------------------------------------------- policy figure
results = pd.read_csv(HERE / "ppo_results.csv")
by = ["environment", "model", "N"]
agg = results.groupby(by).mean(numeric_only=True).reset_index()
if results.seed.nunique() > 1:
    # several policy seeds: the error bar is the spread across seeds, which is
    # the uncertainty that matters, not evaluation noise within one seed.
    sd = results.groupby(by)[["real_reward", "model_reward"]].std().reset_index()
    agg["real_reward_std"] = sd.real_reward.values
    agg["model_reward_std"] = sd.model_reward.values

fig, axes = plt.subplots(1, len(ENVS), figsize=(3.7 * len(ENVS), 3.1))
for ax, env_id in zip(axes, ENVS):
    sub = agg[agg.environment == env_id]
    for level in REFS[env_id]:
        ax.axhline(level, color="0.75", lw=0.7, ls=":", zorder=0)
    for label in STYLE:
        c = sub[sub.model == label].sort_values("N")
        if c.empty:
            continue
        colour, marker, _ = style(label)
        # Real-environment return only. The in-model return is a diagnostic and
        # is not reported: on MountainCar the observation clip can turn an
        # out-of-range prediction into a goal-reached terminal state, so that
        # column is not trustworthy there.
        ax.errorbar(c["N"], c.real_reward, yerr=c.real_reward_std, color=colour,
                    marker=marker, ms=5, lw=1.6, capsize=2, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_VALUES, labels=[str(v) for v in N_VALUES])
    ax.minorticks_off()
    ax.set_title(TITLES[env_id], fontsize=10)
    ax.set_xlabel("$N$: context steps / transitions", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("return, real environment", fontsize=8)
legend(fig, ncol=5)
fig.tight_layout(rect=(0, 0.12, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"ppo_policy.{ext}", dpi=200, bbox_inches="tight")
print("wrote ppo_policy.pdf/.png")


# ------------------------------------------------------------- learning curves
def curve(log_dir, grid):
    """One run's episode reward, interpolated onto a common timestep grid.

    Monitor logs one row per episode, so runs have different numbers of rows at
    different timesteps; interpolating first is what makes them averageable.
    """
    m = log_dir / "monitor.csv"
    if not m.exists():
        return None
    df = pd.read_csv(m, skiprows=1)
    if len(df) < 3:
        return None
    t = df["l"].cumsum().to_numpy()
    y = df["r"].to_numpy()
    # rolling mean over available points, not np.convolve: zero-padded edges
    # bias the ends toward 0, which on MountainCar (values near -200) invents a
    # rise at the end of training that is not there.
    k = max(1, len(y) // 12)
    y = pd.Series(y).rolling(k, min_periods=1, center=True).mean().to_numpy()
    return np.interp(grid, t, y, left=np.nan, right=y[-1])


def run_dir(env_id, model, n, seed):
    slug = model.lower().replace(" ", "_").replace(",", "")
    return HERE / "logs" / f"{env_id}_{slug}_N{n}_s{seed}"


seeds = sorted(results.seed.unique())
# start past the first episode, or the whole column is NaN
grid = np.linspace(TOTAL_TIMESTEPS / 40, TOTAL_TIMESTEPS, 60)
fig, axes = plt.subplots(len(LEARN_ENVS), len(N_VALUES),
                         figsize=(3.3 * len(N_VALUES), 2.7 * len(LEARN_ENVS)),
                         squeeze=False, sharex=True, sharey=True)
for row, env_id in enumerate(LEARN_ENVS):
    for col, n in enumerate(N_VALUES):
        ax = axes[row][col]
        for label in STYLE:
            cs = [curve(run_dir(env_id, label, n, s), grid) for s in seeds]
            cs = [c for c in cs if c is not None]
            if not cs:
                continue
            a = np.vstack(cs)
            colour, _, _ = style(label)
            # mean across seeds, with the seed-to-seed spread as a band: one
            # line per model instead of one per run.
            ax.fill_between(grid, np.nanmin(a, 0), np.nanmax(a, 0),
                            color=colour, alpha=0.13, lw=0)
            ax.plot(grid, np.nanmean(a, 0), color=colour, lw=1.5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        if row == 0:
            ax.set_title(f"$N={n}$", fontsize=9)
        if row == len(LEARN_ENVS) - 1:
            ax.set_xlabel("PPO timesteps", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{TITLES[env_id]}
return inside the model",
                          fontsize=8)
legend(fig, ncol=5, y=-0.04)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"ppo_learning.{ext}", dpi=200, bbox_inches="tight")
print("wrote ppo_learning.pdf/.png")
