"""The paper figure for the PPO experiments.

Reads ppo_results.csv and writes ppo_policy.{pdf,png}. Colours and markers are
taken from dyna_standard's config.MODELS so the figure matches the forecasting
figures of the same study.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dyna_standard"))
import config

# notebook label -> (registry entry supplying colour, legend label, marker).
# The two differenced variants are the same model, so they share its colour and
# are separated by marker -- identity carried by shape, as in config.MODELS.
STYLE = {
    "MLP": ("MLP", "MLP", None),
    "VARX": ("VARX", "VARX", None),
    "Chronos-2 S (level)": ("Chronos-2 S (level)", "Chronos-2 S (level)", None),
    "Chronos-2 S (diff, r=6)": ("Chronos-2 S", "Chronos-2 S, $r=6$", "o"),
    "Chronos-2 S (diff, r=1)": ("Chronos-2 S", "Chronos-2 S, $r=1$", "s"),
}
ENVS = ["CartPole-v1", "MountainCar-v0", "Pendulum-v1"]
TITLES = {"CartPole-v1": "CartPole", "MountainCar-v0": "MountainCar",
          "Pendulum-v1": "Pendulum"}
# random-policy and solved thresholds, for scale
REFS = {
    "CartPole-v1": [(22.0, "random"), (475.0, "solved")],
    "MountainCar-v0": [(-200.0, "random"), (-110.0, "solved")],
    "Pendulum-v1": [(-1179.0, "random"), (-150.0, "solved")],
}

results = pd.read_csv(HERE / "ppo_results.csv")
by = ["environment", "model", "N"]
agg = results.groupby(by).mean(numeric_only=True).reset_index()
if results.seed.nunique() > 1:
    # several policy seeds: the error bar is the spread across seeds, which
    # is the uncertainty that matters, not evaluation noise within one seed.
    sd = results.groupby(by)[["real_reward", "model_reward"]].std().reset_index()
    agg["real_reward_std"] = sd.real_reward.values
    agg["model_reward_std"] = sd.model_reward.values

fig, axes = plt.subplots(1, len(ENVS), figsize=(3.7 * len(ENVS), 3.1))
for ax, env_id in zip(axes, ENVS):
    sub = agg[agg.environment == env_id]
    # random-policy and solved levels, named in the caption rather than inline
    for level, _ in REFS[env_id]:
        ax.axhline(level, color="0.75", lw=0.7, ls=":", zorder=0)
    for label, (key, _, mk) in STYLE.items():
        c = sub[sub.model == label].sort_values("N")
        if c.empty:
            continue
        cfg = config.MODELS[key]
        marker = mk or cfg["marker"]
        # solid + filled: the real environment. dashed + hollow: inside the model.
        ax.errorbar(c["N"], c.real_reward, yerr=c.real_reward_std,
                    color=cfg["color"], marker=marker, ms=5, lw=1.6,
                    capsize=2, zorder=3)
        ax.plot(c["N"], c.model_reward, color=cfg["color"], marker=marker,
                ms=4, lw=1.0, ls="--", mfc="white", alpha=0.55, zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 256, 1024], labels=["64", "256", "1024"])
    ax.minorticks_off()
    ax.set_title(TITLES[env_id], fontsize=10)
    ax.set_xlabel("$N$: context steps / transitions", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("episode return", fontsize=8)

handles = [
    plt.Line2D([], [], color=config.MODELS[k]["color"],
               marker=mk or config.MODELS[k]["marker"], ms=5, lw=1.6, label=lab)
    for k, lab, mk in STYLE.values()
]
handles += [
    plt.Line2D([], [], color="0.3", lw=1.6, label="real environment"),
    plt.Line2D([], [], color="0.3", lw=1.1, ls="--", mfc="white",
               marker="o", ms=5, label="inside the model"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"ppo_policy.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {HERE / 'ppo_policy.pdf'} and .png")
