"""Draws the context-budget figure and the summary table from
``results/results.csv``. Styled to match ``experiments/dyna_standard/plots.py``,
so the two sections look like one study: no in-axes title (the caption names
the panels), panel letters (a)/(b)/(c), muted grid, NMSE clipped to a shared
band with out-of-band points marked by a triangle, best value per column in
the table set in bold.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

RESULTS_CSV = "results/results.csv"
FIG_PATH = "results/fig_forecast_scaling.pdf"
EXAMPLE_FIG_PATH = "results/fig_series_example.pdf"
TRAJ_FIG_PATH = "results/fig_forecast_trajectories.pdf"
TABLE_PATH = "results/table_forecast_scaling.tex"

EXAMPLE_START = 24 * 30  # one week from day 30, clear of the New Year's Eve edge
EXAMPLE_HOURS = 24 * 7
EXAMPLE_LONG_HOURS = 1024  # the largest budget in the sweep, same start point

TRAJ_BUDGET = 1024
TRAJ_CONTEXT_SHOWN = 18  # hours of context drawn before the forecast origin
TRAJ_DODGE = 0.16  # hours each model is nudged apart, so markers do not stack

mpl.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "grid.color": "#c9c9c4",
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.axisbelow": True,
    }
)
INK = "#1a1a1a"

SERIES_TITLE = {
    "non_shiftable_load": "Load",
    "solar_generation": "Solar",
    "electricity_pricing": "Price",
}
SERIES_ORDER = list(SERIES_TITLE)

MODEL_LABEL = {
    "Chronos-2 S (diff)": "Chronos-2 S, differenced",
    "Chronos-2 S (level)": "Chronos-2 S, levels",
    "MLP": "MLP",
    "Seasonal-naive": "Seasonal-naive",
}
MODEL_STYLE: dict[str, dict[str, str | None]] = {
    "Chronos-2 S (diff)": {"color": "#007BFF", "marker": "o"},
    "Chronos-2 S (level)": {"color": "#6b6b6b", "marker": "o", "linestyle": "--"},
    "MLP": {"color": "#046307", "marker": "^"},
    "Seasonal-naive": {"color": "#D55E00", "marker": None, "linestyle": ":"},
}
MODEL_ORDER = list(MODEL_STYLE)
BUDGET_TICKS = [16, 64, 256, 1024]

YLIM = (1e-2, 3e1)


def _clip(y):
    y = np.asarray(y, float)
    above = y > YLIM[1]
    return np.clip(y, YLIM[0], YLIM[1]), above


def panel_label(ax, letter):
    ax.text(
        0.04,
        0.94,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def make_figure(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for letter, ax, series in zip("abc", axes, SERIES_ORDER):
        sub = df[df["series"] == series]
        for model in MODEL_ORDER:
            row = sub[sub["model"] == model].sort_values("budget")
            if row.empty:
                continue
            style = MODEL_STYLE[model]
            y, above = _clip(row["nmse"].to_numpy())
            ax.plot(row["budget"], y, label=MODEL_LABEL[model], **style)
            if above.any():
                ax.scatter(
                    row["budget"][above],
                    y[above],
                    marker="^",
                    s=28,
                    color=style["color"],
                    zorder=5,
                )
        ax.axhline(1.0, color=INK, lw=0.7, alpha=0.35)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(*YLIM)
        ax.set_xticks(BUDGET_TICKS)
        ax.set_xticklabels([str(b) for b in BUDGET_TICKS])
        ax.grid(True, lw=0.5)
        ax.set_xlabel("Context budget $N$ [hours]")
        panel_label(ax, letter)
    axes[0].set_ylabel("NMSE, 24h ahead")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout()
    fig.savefig(FIG_PATH, bbox_extra_artists=(fig.legends[0],), bbox_inches="tight")
    print(f"wrote {FIG_PATH}")


SERIES_UNIT = {
    "non_shiftable_load": "kWh",
    "solar_generation": "W/kW",
    "electricity_pricing": "\\$/kWh",
}
SERIES_COLOR = {
    "non_shiftable_load": "#007BFF",
    "solar_generation": "#E69F00",
    "electricity_pricing": "#046307",
}


def make_example_figure(series_dict):
    """Two rows per series: one week (top), and the largest context budget in
    the sweep, 1024 hours from the same start point (bottom). The scaling
    figure only turns good from around $N=256$ on -- this shows why: a single
    week barely repeats, 1024 hours is enough cycles for a zero-shot model to
    match against."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6))
    hours = np.arange(EXAMPLE_HOURS)
    days_long = np.arange(EXAMPLE_LONG_HOURS) / 24

    for letter, ax, series in zip("abc", axes[0], SERIES_ORDER):
        y = series_dict[series][EXAMPLE_START : EXAMPLE_START + EXAMPLE_HOURS]
        ax.plot(hours, y, color=SERIES_COLOR[series], lw=1.2)
        ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
        ax.set_xlim(0, EXAMPLE_HOURS)
        ax.grid(True, lw=0.5)
        ax.set_xlabel("Hour of week")
        ax.set_ylabel(f"{SERIES_TITLE[series]} [{SERIES_UNIT[series]}]")
        panel_label(ax, letter)

    for letter, ax, series in zip("def", axes[1], SERIES_ORDER):
        y = series_dict[series][EXAMPLE_START : EXAMPLE_START + EXAMPLE_LONG_HOURS]
        ax.plot(days_long, y, color=SERIES_COLOR[series], lw=0.7)
        ax.set_xticks([0, 7, 14, 21, 28, 35, 42])
        ax.set_xlim(0, EXAMPLE_LONG_HOURS / 24)
        ax.grid(True, lw=0.5)
        ax.set_xlabel("Day (same start point, $N=1024$h)")
        ax.set_ylabel(f"{SERIES_TITLE[series]} [{SERIES_UNIT[series]}]")
        panel_label(ax, letter)

    fig.tight_layout()
    fig.savefig(EXAMPLE_FIG_PATH)
    print(f"wrote {EXAMPLE_FIG_PATH}")


def make_trajectory_figure(series_dict):
    """One example forecast origin per series, at the largest budget: the true
    context and future as a black line, each model's 24h forecast drawn on top
    of it. This is what the NMSE numbers in \\autoref{fig:forecast_component}
    are scoring, made visible.

    One row per series (not one column), a short context window, and a small
    horizontal offset per model in the forecast segment -- with four models on
    top of each other every hour, points that are close in value are
    otherwise unreadable.
    """
    from data import eval_origins, split_index
    from models import chronos_forecast, mlp_forecast, seasonal_naive
    from run import CHRONOS_MODEL, LAG, N_WINDOWS, SEED, H

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 4.6))
    for letter, ax, series in zip("abc", axes, SERIES_ORDER):
        s = series_dict[series]
        t = len(s)
        origins = eval_origins(t, H, N_WINDOWS, SEED)
        origin = int(origins[len(origins) // 2])
        split_idx = split_index(t)

        x_ctx = np.arange(-TRAJ_CONTEXT_SHOWN + 1, 1)
        x_fut = np.arange(1, H + 1)
        true_ctx = s[origin - TRAJ_CONTEXT_SHOWN + 1 : origin + 1]
        true_fut = s[origin + 1 : origin + 1 + H]
        origin_pt = s[origin]  # shared by every line, so nothing gaps at hour 0

        true_color = "#2f2f2f"  # softer than pure black, dark enough to stay clear of the models' mid-gray
        ax.plot(x_ctx, true_ctx, color=true_color, lw=1.1, label="True", zorder=3)
        ax.plot(
            np.concatenate([[0], x_fut]),
            np.concatenate([[origin_pt], true_fut]),
            color=true_color,
            lw=1.5,
            marker="o",
            markersize=2.5,
            zorder=3,
        )

        preds = {
            "Seasonal-naive": seasonal_naive(s, [origin], H)[0],
            "Chronos-2 S (level)": chronos_forecast(
                s, [origin], H, TRAJ_BUDGET, CHRONOS_MODEL, difference=False
            )[0],
            "Chronos-2 S (diff)": chronos_forecast(
                s, [origin], H, TRAJ_BUDGET, CHRONOS_MODEL, difference=True
            )[0],
        }
        mlp_pred = mlp_forecast(
            s, split_idx, [origin], H, TRAJ_BUDGET, lag=LAG, seed=SEED
        )
        if mlp_pred is not None:
            preds["MLP"] = mlp_pred[0]

        n = len(preds)
        for i, (model, pred) in enumerate(preds.items()):
            style = MODEL_STYLE[model]
            offset = (i - (n - 1) / 2) * TRAJ_DODGE
            ax.plot(
                np.concatenate([[0], x_fut + offset]),
                np.concatenate([[origin_pt], pred]),
                label=MODEL_LABEL[model],
                markersize=5,
                **style,
            )

        ax.axvline(0, color=INK, lw=0.7, alpha=0.4, linestyle="--")
        ax.set_xlim(-TRAJ_CONTEXT_SHOWN, H + 1)
        ax.grid(True, lw=0.5)
        ax.set_ylabel(f"{SERIES_TITLE[series]} [{SERIES_UNIT[series]}]")
        panel_label(ax, letter)
        if letter == "c":
            ax.set_xlabel(
                "Hours from forecast origin (context left, forecast right of the dashed line)",
                labelpad=6,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.09),
    )
    fig.tight_layout()
    fig.savefig(
        TRAJ_FIG_PATH, bbox_extra_artists=(fig.legends[0],), bbox_inches="tight"
    )
    print(f"wrote {TRAJ_FIG_PATH}")


def _fmt(col):
    """Bold the smallest (best) NMSE in a column, three significant figures."""
    best = col.min()
    out = []
    for v in col:
        s = "--" if pd.isna(v) else f"{v:.3g}"
        out.append(f"\\textbf{{{s}}}" if pd.notna(v) and v == best else s)
    return out


def make_table(df: pd.DataFrame, budget=1024):
    sub = df[df["budget"] == budget].pivot(
        index="model", columns="series", values="nmse"
    )
    sub = sub.reindex(MODEL_ORDER)[SERIES_ORDER]
    sub.index = [MODEL_LABEL[m] for m in sub.index]
    sub.columns = [SERIES_TITLE[c] for c in sub.columns]
    fmt = sub.apply(_fmt, axis=0)
    fmt.index = sub.index

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{24h-ahead NMSE at the largest context budget ($N=1024$). "
            r"Best (lowest) per column in bold. Persistence scores 1.0.}"
        ),
        r"\label{tab:forecast_component}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        " & " + " & ".join(sub.columns) + r" \\",
        r"\midrule",
    ]
    for name, row in fmt.iterrows():
        lines.append(f"{name} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(TABLE_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {TABLE_PATH}")
    print(sub)


def main():
    from data import load_series

    df = pd.read_csv(RESULTS_CSV)
    series_dict = load_series()
    make_example_figure(series_dict)
    make_trajectory_figure(series_dict)
    make_figure(df)
    make_table(df)


if __name__ == "__main__":
    main()
