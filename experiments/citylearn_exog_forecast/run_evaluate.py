import common
import matplotlib.pyplot as plt
import pandas as pd
from citylearn.agents.base import BaselineAgent
from citylearn.agents.rbc import BasicRBC

KPI_COLS = ["cost_total", "carbon_emissions_total"] + common.GRID_KPIS


def main():
    rows = []
    for name, cls in [("no_control", BaselineAgent), ("rbc", BasicRBC)]:
        kpi, ret = common.run_reference_agent(cls)
        rows.append(
            {
                "arm": name,
                "seed": None,
                "score": common.headline_score(kpi),
                "return": ret,
            }
            | kpi["District"].loc[KPI_COLS].to_dict()
        )
        print(f"{name:14s} score {rows[-1]['score']:.3f}  return {ret:9.1f}")

    for arm in common.RL_ARMS:
        for seed in common.SEEDS:
            if not (common.MODELS_DIR / f"sac_{arm}_s{seed}.pt").exists():
                print(f"{arm}_s{seed}: no trained policy, skipping")
                continue
            model, vecnorm = common.load_sac(arm, seed)
            kpi, ret = common.rollout_policy(model, vecnorm.obs_rms, arm, seed)
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "score": common.headline_score(kpi),
                    "return": ret,
                }
                | kpi["District"].loc[KPI_COLS].to_dict()
            )
            print(f"{arm}_s{seed:<8d} score {rows[-1]['score']:.3f}  return {ret:9.1f}")

    results = pd.DataFrame(rows)
    common.RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(common.RESULTS_DIR / "kpis.csv", index=False)

    rl = results[results.seed.notna()]
    if not rl.empty:
        print("\nmean +- std over seeds:")
        print(
            rl.groupby("arm")[
                ["score", "cost_total", "carbon_emissions_total", "return"]
            ]
            .agg(["mean", "std"])
            .round(3)
            .to_string()
        )

    plot_learning_curves()
    plot_headline(results)
    print("\nresults/kpis.csv and figs/ written")


def plot_learning_curves():
    files = sorted(common.RESULTS_DIR.glob("learning_curve_*.csv"))
    if not files:
        return
    curves = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    _, ax = plt.subplots(figsize=(9, 4.5))
    for i, arm in enumerate([a for a in common.RL_ARMS if a in set(curves.arm)]):
        g = curves[curves.arm == arm]
        piv = g.pivot_table(index="timestep", columns="seed", values="episode_return")
        mean, std = piv.mean(axis=1), piv.std(axis=1)
        color = common.PALETTE[i % len(common.PALETTE)]
        ax.plot(mean.index, mean, label=arm, color=color)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("environment step")
    ax.set_ylabel("train episode return")
    ax.legend(ncol=2)
    ax.set_title("SAC training (mean +- std over seeds)")
    common.savefig("learning_curves")


def plot_headline(results: pd.DataFrame):
    rl = results[results.seed.notna()]
    if rl.empty:
        return
    arms = [a for a in common.RL_ARMS if a in set(rl.arm)]
    _, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in [
        (
            axes[0],
            "score",
            "cost ratio vs no battery (<1 beats no storage)",
        ),
        (axes[1], "carbon_emissions_total", "district emissions"),
    ]:
        agg = rl.groupby("arm")[metric].agg(["mean", "std"]).reindex(arms)
        ax.bar(
            agg.index,
            agg["mean"],
            yerr=agg["std"],
            capsize=4,
            color=common.PALETTE[: len(agg)],
            alpha=0.85,
        )
        for j, arm in enumerate(agg.index):
            pts = rl.loc[rl.arm == arm, metric]
            ax.scatter([j] * len(pts), pts, color="black", zorder=3, s=18)
        for name, ls in [("no_control", "--"), ("rbc", ":")]:
            ref = results.loc[results.arm == name, metric]
            if not ref.empty:
                ax.axhline(float(ref.iloc[0]), color="gray", ls=ls, lw=1.2, label=name)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="x")
    axes[0].legend()
    common.savefig("headline")


if __name__ == "__main__":
    main()
