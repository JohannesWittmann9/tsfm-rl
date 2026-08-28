"""Every figure in the study, one function each.

All of them take a list of environments, so a single-environment notebook and
``all_experiments.ipynb`` draw the *same* figure with a different number of
panels -- that is what keeps the notebooks structurally identical.

Sizing follows a journal column: 3.4 in single, 7.0 in double. Figures carry no
title (the caption does that in a paper), so every function prints a
ready-to-paste caption and writes PDF + PNG into the notebook's own ``figures/``.
"""

from collections.abc import Mapping

import config
import matplotlib as mpl
import matplotlib.pyplot as plt
import models
import numpy as np
import pandas as pd
import pipeline
from matplotlib.ticker import (
    FixedLocator,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    NullLocator,
)

COL, WIDE = 3.4, 7.0

mpl.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
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

INK = {"text": "#1a1a1a", "muted": "#6b6b6b", "rule": "#b0b0ac"}
ACCENT = "#2a78d6"  # the neutral single-series colour (figures 1, 1b, 4)
MARK = "#eb6834"  # the second colour those figures contrast it with
HILITE = "#eda100"

# One colour and one marker per model, straight out of config.MODELS, so a model
# keeps its identity in every figure and nothing depends on colour alone.


class _Style(Mapping):
    """``STYLE[name]`` -> that model's colour and marker, read at draw time.

    A dict comprehension here would snapshot config.MODELS at *import*, so
    editing a colour and re-running the figure cell in a live kernel changed
    nothing -- the one place in this module that broke the "config is read when
    the figure is drawn" rule. Each lookup returns a fresh dict, so callers can
    keep mutating it (``st = dict(STYLE[name])``, ``**STYLE[name]``).
    """

    def __getitem__(self, name):
        cfg = config.MODELS[name]
        return dict(color=cfg["color"], marker=cfg["marker"])

    def __iter__(self):
        return iter(config.MODELS)

    def __len__(self):
        return len(config.MODELS)


STYLE = _Style()


# The r ramp for the per-configuration figures (§6a, §7a). Inside one of those
# panels every curve is the *same model*, so colour cannot carry model identity
# any more: lightness carries the stretch factor and linestyle the presentation.
# One hue, light to dark, because r is an ordered quantity -- and no step lighter
# than the first, which is the lightest that still clears 2:1 on white.
R_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281"]
PRES_LS = {"diff": "-", "level": (0, (4, 2)), "-": "-"}


def model_label(name, presentation=None, r=None, lag=None):
    """The name a legend gives one *configuration* of a model.

    ``Chronos-2 S`` drawn as diff at r=12 is ``Chronos-2 S (diff, r=12)``: which
    variant a curve is has to be readable off the figure, not looked up in a
    table. A model whose name already ends in a bracket is a pinned variant with
    nothing to disambiguate, so it is left alone -- as is everything when
    ``config.LEGEND_VARIANTS`` is off.
    """
    if not config.LEGEND_VARIANTS or name.endswith(")"):
        return name
    if presentation in (None, "-") or r in (None, 0):
        return f"{name} (lag {lag})" if lag not in (None, -1, 0) else name
    return f"{name} ({presentation}, r={int(r)})"


def variant_fields(name, env_id, variant):
    """``presentation``/``r``/``lag`` for a named variant, from the model itself.

    §8's table is stored lean -- it is one row per channel per step per window
    per variant, so it carries `variant` and nothing derivable from it. The
    fields come back from ``models.variants`` rather than from parsing that
    string, so the two can never drift apart.
    """
    for v in models.variants(name, env_id):
        if v["variant"] == variant:
            return v
    return {}


def label_of(row):
    """``model_label`` straight off a results row, whatever columns it carries."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    return model_label(get("model"), get("presentation"), get("r"), get("lag"))


def _sweep_key(name):
    """The column whose value a model's variants differ along."""
    return "r" if models.is_tsfm(name) else "lag"


def _sweep_value(row):
    """The value this row's variant sits at along its model's sweep."""
    return int(row.get(_sweep_key(str(row.get("model"))), 0) or 0)


def _variant_style(row, steps):
    """Colour and linestyle for one variant inside a per-configuration panel.

    ``steps`` is the whole figure's sweep values, not one panel's: the ramp has
    to mean the same thing in every panel or the legend is a lie. r and lag share
    it -- they are both "how much history the variant is given", and the two
    lists coincide in practice.
    """
    val = _sweep_value(row)
    if len(steps) < 2:
        colour = R_RAMP[3]
    else:
        i = steps.index(val) if val in steps else 0
        colour = R_RAMP[round(i * (len(R_RAMP) - 1) / (len(steps) - 1))]
    return dict(color=colour, ls=PRES_LS.get(str(row.get("presentation", "-")), "-"))


def _config_legend(fig, model_names, steps, n_panels):
    """What the lightness means, what the dash means, which curve is the pick."""
    tsfm = any(models.is_tsfm(m) for m in model_names)
    lab = "$r$" if tsfm else "lag"
    h = [
        plt.Line2D(
            [],
            [],
            color=R_RAMP[
                round(i * (len(R_RAMP) - 1) / max(len(steps) - 1, 1))
                if len(steps) > 1
                else 3
            ],
            lw=1.8,
            label=f"{lab}={v}",
        )
        for i, v in enumerate(steps)
    ]
    if tsfm:
        h += [
            plt.Line2D([], [], color=INK["muted"], ls=PRES_LS["diff"], label="diff"),
            plt.Line2D([], [], color=INK["muted"], ls=PRES_LS["level"], label="level"),
        ]
    h.append(
        plt.Line2D(
            [],
            [],
            color=INK["text"],
            lw=2.4,
            marker="o",
            ms=4,
            label="drawn in §6/§7",
        )
    )
    _bottom_legend(fig, h, n_panels)


def _configs_grid(df, env_ids, model_names, chosen, draw, ylabel, xlabel, height=1.7):
    """The shared skeleton of §6a and §7a: a panel per model per environment.

    Every curve in a panel is the same model, so the model palette is not used at
    all here -- ``_variant_style`` carries the variant instead. ``draw`` plots one
    variant into one axis; everything else is layout.
    """
    steps = sorted(
        {
            int(v)
            for name in model_names
            for v in df[df.model == name][_sweep_key(name)].dropna().unique()
            if int(v) > 0
        }
    )
    fig, axes = plt.subplots(
        len(model_names),
        len(env_ids),
        figsize=(_width(len(env_ids)), height * len(model_names) + 0.7),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for r_i, name in enumerate(model_names):
        d_model = df[df.model == name]
        for c_i, env_id in enumerate(env_ids):
            ax = axes[r_i][c_i]
            d_env = d_model[d_model.env == env_id]
            for variant in sorted(d_env.variant.unique()):
                d = d_env[d_env.variant == variant]
                if d.empty:
                    continue
                st = _variant_style(d.iloc[0], steps)
                is_pick = chosen and chosen.get((env_id, name)) == variant
                draw(
                    ax,
                    d,
                    **st,
                    lw=2.2 if is_pick else 1.0,
                    marker="o" if is_pick else None,
                    ms=3,
                    markevery=0.25,
                    zorder=4 if is_pick else 2,
                )
            ax.set(yscale="log", ylim=ylim())
            ax.tick_params(labelsize=7)
            axgrid(ax)
            if r_i == 0:
                ax.set_title(config.SHORT[env_id], fontsize=8, pad=3)
            if c_i == 0:
                ax.set_ylabel(f"{name}\n{ylabel}", fontsize=7.5)
            if d_env.empty:
                ax.text(
                    0.5,
                    0.5,
                    "not measured",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7,
                    color=INK["muted"],
                )
    for c_i in range(len(env_ids)):
        axes[-1][c_i].set_xlabel(xlabel if c_i == len(env_ids) // 2 else "", fontsize=8)
    return fig, steps


def _sweep_models(df, model_names=None):
    """Models worth a per-configuration panel: the ones with a variant to sweep."""
    names = list(model_names or config.GRID_MODELS)
    return [n for n in names if df[df.model == n].variant.nunique() > 1]


def ylim(which=None):
    """The log band an error figure is drawn in, read from config at draw time.

    ``which`` names a per-figure override (``SCALING_YLIM``, ``ROLLOUT_YLIM``);
    unset, it falls back to the shared ``NMSE_YLIM``.
    """
    return (which and getattr(config, which)) or config.NMSE_YLIM


# Which excerpt the two illustration figures draw. An environment overrides this
# with an `illustration` entry in its `<env>/env.py`, because which window shows
# the motion depends on the environment, not on the figure.
DEFAULT_ILLUSTRATION = dict(
    episode=0,
    input_t0=40,
    input_steps=24,
    input_r=4,
    highlight=8,
    int_t0=20,
    int_steps=40,
    patch=16,
    int_r=4,
)


# ---------------------------------------------------------------- small helpers
def axgrid(ax, which="both"):
    ax.grid(True, axis=which, lw=0.5)
    ax.tick_params(length=2.5, colors=INK["muted"])
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_color(INK["text"])
    return ax


def panel_label(ax, letter, x=0.04, y=0.93, va="top"):
    """(a), (b), ... -- the convention captions refer to."""
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va=va,
        ha="left",
        color=INK["text"],
    )


def save(fig, name, outdir, caption=""):
    """Vector + raster into ``outdir``, and the caption printed underneath."""
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.show()
    if caption:
        print(f"Figure {name}. {caption}\n")
    return fig


def clipped_plot(ax, x, y, lo=None, hi=None, **kw):
    """Line plot with out-of-range points pinned to the axis and flagged."""
    lo, hi = (ylim()[0] if lo is None else lo), (ylim()[1] if hi is None else hi)
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if not len(x):
        return
    ax.plot(x, np.clip(y, lo, hi), **kw)
    colour = kw.get("color", kw.get("c", INK["muted"]))
    for mask, mark in ((y > hi, "^"), (y < lo, "v")):
        if mask.any():
            # A curve out of range for its whole length would draw a solid bar of
            # markers, so thin them to a handful of flags.
            xs = x[mask]
            xs = xs[:: max(1, int(np.ceil(len(xs) / 6)))]
            ax.plot(
                xs,
                np.full(len(xs), hi if mark == "^" else lo),
                ls="",
                marker=mark,
                ms=4.5,
                color=colour,
                markerfacecolor="none",
                markeredgewidth=1.0,
            )


def _tick_label(v):
    """These axes carry powers of two, so `k` means 1024 -- 4096 is `4k`, not
    `4.096k`, which is what a decimal `k` produced and it was unreadable."""
    v = float(v)
    if v >= 1024:
        return f"{v / 1024:g}k"
    return f"{v:g}"


def logticks(ax, values, axis="x", max_ticks=5):
    """Label these values on a log axis, thinned to at most ``max_ticks``.

    matplotlib's minor labels turn into mush when the values sit close together,
    and a full budget sweep (1 … 8192) has more labels than a panel this narrow
    can show -- at four panels the right-hand ones ran into each other. So drop
    every other label until they fit. Thinning by halving keeps the survivors
    evenly spaced on the log axis, which is why the last value is not pinned:
    forcing it back in put 4k and 8k side by side.
    """
    values = sorted(dict.fromkeys(float(v) for v in values))
    while len(values) > max_ticks:
        values = values[::2]
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(values))
    a.set_minor_locator(NullLocator())
    lab = [_tick_label(v) for v in values]
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)(lab)


def _width(n_panels, per=1.75, floor=COL):
    return float(np.clip(per * n_panels + 1.0, floor, WIDE))


def _model_line(name, presentation=None, r=None, lag=None, **kw):
    """Legend handle for one model: dashed + hollow for the foundation models,
    solid + filled for the trained ones. The label states the configuration, so
    the reader never has to guess which variant the curve is."""
    st = dict(STYLE[name])
    if models.is_tsfm(name):
        kw.setdefault("ls", (0, (4, 2)))
        kw.setdefault("markerfacecolor", "white")
    label = model_label(name, presentation, r, lag)
    return plt.Line2D([], [], ms=3, label=label, **st, **kw)


def _line_for(name, drawn):
    """Legend handle for a model, labelled with the configuration actually drawn."""
    row = drawn.get(name)
    if row is None:
        return _model_line(name)
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    return _model_line(name, get("presentation"), get("r"), get("lag"))


def _bottom_legend(fig, handles, n_panels):
    """One legend under the whole figure, anchored to its bottom edge so a narrow
    single-panel figure does not end up with a strip of white space."""
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=min(len(handles), 3 if n_panels == 1 else 7),
        fontsize=7.5,
    )


def _chosen(df, env_ids):
    """``(env, model) -> variant`` for the two combined figures.

    ``rollout.csv`` holds every variant, so a figure that draws one curve per
    model has to say which. It asks :func:`pipeline.selection`, which is the
    section 5 pick with ``config.PLOT_VARIANTS`` already folded in -- so choosing
    a different configuration for the paper is an edit to that dict and a re-run
    of the cell, with no recompute behind it.

    Falls back to whatever the data holds when the variant column is absent or
    the grid is unavailable, so an older CSV still draws.
    """
    if "variant" not in df.columns:
        return None
    try:
        sel = pipeline.selection(list(env_ids))
    except FileNotFoundError:
        return None
    return {(r.env, r.model): r.variant for r in sel.itertuples()}


def _one_variant(d, env_id, name, chosen):
    """One model's rows, cut to the configuration ``chosen`` names."""
    if not chosen or (env_id, name) not in chosen:
        return d
    want = chosen[(env_id, name)]
    cut = d[d.variant == want]
    if cut.empty and not d.empty:
        print(
            f"   note: {name} on {env_id} has no rows for {want!r} "
            f"(have: {sorted(d.variant.unique())}) -- not drawn"
        )
    return cut


def _plot_models(models_arg, df=None):
    """The models a figure draws, in config order, minus any the data has none
    of -- a model that was never run must not claim a legend entry."""
    names = list(models_arg if models_arg is not None else config.PLOT_MODELS)
    return [n for n in names if df is None or (df.model == n).any()]


# ------------------------------------------------------- 1  the environments
def fig_environments(env_ids, outdir, episode=0):
    """One episode per environment: every observation channel plus the action."""
    specs = {e: pipeline.spec_of(e) for e in env_ids}
    data = {e: pipeline.episodes(e) for e in env_ids}
    nrow = max(specs[e].n_obs for e in env_ids) + 1
    fig, axes = plt.subplots(
        nrow,
        len(env_ids),
        figsize=(_width(len(env_ids), 1.55), 1.05 * nrow),
        sharex=True,
        squeeze=False,
    )
    for col, env_id in enumerate(env_ids):
        spec, d = specs[env_id], data[env_id]
        series = [(lb, d["states"][episode, :, c]) for c, lb in enumerate(spec.labels)]
        series.append((spec["action_label"], d["actions"][episode, :, 0]))
        for r in range(nrow):
            ax = axes[r][col]
            if r >= len(series):
                ax.axis("off")
                continue
            lb, y = series[r]
            is_action = r == len(series) - 1
            ax.plot(y, lw=0.8, color=INK["muted"] if is_action else ACCENT)
            ax.set_ylabel(lb, fontsize=7.5)
            axgrid(ax, "y")
            ax.tick_params(labelsize=7)
        axes[0][col].set_title(config.SHORT[env_id], fontsize=8.5, pad=4)
        axes[len(series) - 1][col].set_xlabel("step", fontsize=8)
    fig.align_ylabels()
    fig.tight_layout(h_pad=0.35, w_pad=0.9)
    policies = {
        ("eps-greedy stabiliser" if specs[e].cfg.get("policy") else "uniform-random")
        for e in env_ids
    }
    return save(
        fig,
        "fig1_environments",
        outdir,
        f"One episode per environment under a {' / '.join(sorted(policies))} "
        "policy. Rows are observation channels, the bottom row in each "
        "column is the action. The integrated channels (velocities, "
        "angular rates) are the ones that must be passed as increments.",
    )


# ------------------------------------------- 1b  what the model is handed
def fig_model_input(env_id, outdir, **over):
    """The same window before and after the two transforms, on every channel.

    Section 1 plots the environment as the simulator emits it; this plots the
    array the forecaster actually receives, because the two are not the same and
    every claim about presentation is about the difference.
    """
    spec, d = pipeline.spec_of(env_id), pipeline.episodes(env_id)
    ill = {**DEFAULT_ILLUSTRATION, **spec.cfg.get("illustration", {}), **over}
    episode, t0, n_steps = ill["episode"], ill["input_t0"], ill["input_steps"]
    r, highlight = ill["input_r"], ill["highlight"]
    raw_s = d["states"][episode, t0 : t0 + n_steps + 1]
    raw_a = d["actions"][episode, t0 - 1 : t0 + n_steps, 0]  # a[i] produced s[i]

    L = len(raw_s)  # exactly UpsampledDynamics' arithmetic
    dst = np.arange((L - 1) * r + 1)
    idx = np.minimum(dst // r, L - 2)
    frac = ((dst - idx * r) / r)[:, None]
    up_s = (1 - frac) * raw_s[idx] + frac * raw_s[idx + 1]
    up_a = raw_a[np.ceil(dst / r).astype(int)]
    model_s = up_s.copy()  # then the differencing step
    for c in spec["difference"]:
        model_s[1:, c] = np.diff(up_s[:, c])
    model_s, model_a = model_s[1:], up_a[1:]  # first row has no increment

    fig, axes = plt.subplots(
        spec.n_obs + 1,
        2,
        figsize=(WIDE, 1.15 * (spec.n_obs + 1)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1, r]},
    )
    for row in range(spec.n_obs + 1):
        is_act = row == spec.n_obs
        lab = spec["action_label"] if is_act else spec.labels[row]
        diffed = (not is_act) and row in spec["difference"]

        ax = axes[row][0]  # ---- as observed
        y = raw_a if is_act else raw_s[:, row]
        ax.plot(
            np.arange(len(y)),
            y,
            lw=1.0,
            marker="o",
            ms=2.6,
            color=INK["muted"] if is_act else ACCENT,
            drawstyle="steps-post" if is_act else "default",
        )
        ax.set_ylabel(lab, fontsize=7.5)

        ax = axes[row][1]  # ---- as handed over
        y2 = model_a if is_act else model_s[:, row]
        x2 = np.arange(len(y2))
        colour = INK["muted"] if is_act else (MARK if diffed else ACCENT)
        ax.plot(
            x2,
            y2,
            lw=0.9,
            color=colour,
            drawstyle="steps-post" if is_act else "default",
        )
        if not is_act:  # mark the real samples
            keep = np.arange(1, len(dst)) % r == 0
            ax.plot(x2[keep], y2[keep], ls="", marker="o", ms=2.6, color=colour)
        ax.set_ylabel((r"$\Delta$ " if diffed else "") + lab, fontsize=7.5)
        if diffed:
            ax.axhline(0, color=INK["rule"], lw=0.6)
        # Shade one real transition. On the differenced row and the action row it
        # covers the *same* r sub-steps, which is the whole argument in one band:
        # the held action and the 1/r-sized increment step together.
        if diffed or is_act:
            ax.axvspan(
                highlight * r - 0.5,
                (highlight + 1) * r - 0.5,
                color=HILITE,
                alpha=0.18,
                lw=0,
                zorder=0,
            )
        for a_ in (axes[row][0], axes[row][1]):
            axgrid(a_, "y")
            a_.tick_params(labelsize=7)
    axes[0][0].set_title("as observed  ($r=1$, levels)", fontsize=8.5, pad=4)
    axes[0][1].set_title(
        f"as the model receives it  ($r={r}$, integrated channels differenced)",
        fontsize=8.5,
        pad=4,
    )
    axes[-1][0].set_xlabel("step", fontsize=8)
    axes[-1][1].set_xlabel(
        f"model position  ({len(model_a)} rows from {n_steps} real steps)", fontsize=8
    )
    fig.align_ylabels()
    fig.tight_layout(h_pad=0.35, w_pad=0.9)
    out = save(
        fig,
        "fig1b_model_input",
        outdir,
        f"The same {n_steps}-step {config.SHORT[env_id]} window before and "
        f"after the two transforms, on every channel. Left: the observation "
        f"as the simulator emits it. Right: the array handed to the "
        f"forecaster -- stretched by $r={r}$ (filled markers are real "
        f"samples, the line between them is interpolated) and with the "
        f"integrated channels passed as increments. The action is held "
        f"across the sub-steps of the transition it caused, which is a "
        f"zero-order hold, not a repeated application: the sub-step "
        f"increment shrinks by the same factor. The shaded band is one real "
        f"transition, spanning the same {r} rows in both panels.",
    )

    ch = spec["probe_channel"]
    d_real = np.abs(np.diff(raw_s[:, ch])).mean()
    d_sub = np.abs(np.diff(up_s[:, ch])).mean()
    print(
        f"Channel {spec.labels[ch]}: mean |increment| {d_real:.4f} per real step, "
        f"{d_sub:.4f} per model sub-step (ratio {d_real / d_sub:.2f}, r = {r})."
    )
    print(
        f"Action: {np.abs(raw_a).mean():.4f} mean |a| before, "
        f"{np.abs(model_a).mean():.4f} after -- unchanged by construction."
    )
    print(
        f"So the model reads 'this action accompanies a {r}x smaller increment', "
        f"and it reads that from the context rather than from an assumption "
        f"about units."
    )
    return out


# ------------------------------------------------------ 2b  the action probe
def _centred(d):
    """A stored probe curve, re-centred on its own mean.

    ``evaluate._centre`` writes it that way and doing it twice changes nothing,
    but a probe.csv written before that centred on the middle probe action --
    which for an even grid like CartPole's two is the *last* action, leaving the
    whole curve hanging below the zero line instead of straddling it.
    """
    v = d.response.to_numpy(float)
    return v - v.mean()


def fig_probe(probe, env_ids, outdir):
    """The counterfactual probe under four presentations of the same trajectory.

    ``probe`` is the ``probe`` stage's DataFrame (see ``pipeline.run_probe``).
    """
    conds = [c for c, _, _ in pipeline.CONDITIONS]
    tags = [
        m
        for m in config.PROBE_MODELS
        if m in set(probe[probe.model != "environment"].model)
    ]
    fig, axes = plt.subplots(
        len(env_ids),
        len(conds),
        figsize=(WIDE, 2.15 * len(env_ids)),
        squeeze=False,
        sharey="row",
    )
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for r_i, env_id in enumerate(env_ids):
        spec = pipeline.spec_of(env_id)
        d_env = probe[probe.env == env_id]
        ref = d_env[d_env.model == "environment"].sort_values("action")
        for c_i, cond in enumerate(conds):
            ax = axes[r_i][c_i]
            ax.axhline(0, color=INK["rule"], lw=0.6)
            ax.plot(ref.action, _centred(ref), "k--", lw=1.3, zorder=5)
            d_cond = d_env[d_env.condition == cond]
            for j, tag in enumerate(tags):
                d = d_cond[d_cond.model == tag].sort_values("action")
                if d.empty:
                    continue
                ax.plot(d.action, _centred(d), ms=3.5, **STYLE[tag])
                # the slope, in the panel, in the model's own colour: the number
                # the curve is making a claim about
                ax.text(
                    0.04,
                    0.95 - 0.085 * j,
                    f"{d.slope_pct.iloc[0]:+.0f}%",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    va="top",
                    ha="left",
                    color=STYLE[tag]["color"],
                )
            ax.set_xlabel(spec["action_label"], fontsize=8)
            if c_i == 0:
                ax.set_ylabel(
                    f"{config.SHORT[env_id]}\n"
                    f"$\\Delta$ {spec.labels[spec['probe_channel']]}",
                    fontsize=8,
                )
            if r_i == 0:
                ax.set_title(cond, fontsize=8.5, pad=4)
            if "stretch" in cond or cond == "both":  # r is per environment
                r_used = (
                    int(d_cond.r.max()) if len(d_cond) else config.DEFAULT_R[env_id]
                )
                ax.text(
                    0.97,
                    0.04,
                    f"$r$={r_used}",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    ha="right",
                    va="bottom",
                    color=INK["muted"],
                )
            ax.tick_params(labelsize=7)
            axgrid(ax)
            panel_label(ax, next(letters), x=-0.02, y=1.03, va="bottom")
        lo, hi = axes[r_i][0].get_ylim()  # headroom for the in-panel numbers
        axes[r_i][0].set_ylim(lo, hi + 0.45 * (hi - lo))
    h = [plt.Line2D([], [], color="k", ls="--", label="environment")]
    h += [plt.Line2D([], [], ms=3.5, **STYLE[t], label=t) for t in tags]
    fig.tight_layout(w_pad=1.0, h_pad=1.1)
    _bottom_legend(fig, h, len(conds))
    return save(
        fig,
        "fig2b_transforms",
        outdir,
        "The same probe, the same contexts, the same models -- only the "
        "presentation of the trajectory changes. Each panel sweeps the "
        "next action and plots the mean shift in the predicted next value, "
        "centred on the middle action; the dashed line is the real "
        "environment stepped from the same states, so a flat curve means "
        "the action was ignored. Percentages are the recovered slope as a "
        "share of the environment's, in each model's colour.",
    )


# --------------------------------------------- 4  what the transforms do
def fig_interventions(env_id, outdir, **over):
    """The two interventions on real data, not a sketch.

    Top row: differencing -- on levels the previous value already reproduces the
    series, so the action is a percent or two of what a forecaster sees; on
    increments the same term is a substantial share. Bottom row: stretching -- the
    same motion across more of the patch grid.
    """
    spec, d = pipeline.spec_of(env_id), pipeline.episodes(env_id)
    ill = {**DEFAULT_ILLUSTRATION, **spec.cfg.get("illustration", {}), **over}
    episode, t0, n_steps = ill["episode"], ill["int_t0"], ill["int_steps"]
    patch, r = ill["patch"], ill["int_r"]
    ch = spec["probe_channel"]
    lvl = d["states"][episode, t0 : t0 + n_steps + 1, ch]
    act = d["actions"][episode, t0 : t0 + n_steps, 0]
    inc = np.diff(lvl)
    label = spec.labels[ch]

    # the action's share of the increment, fitted on the whole draw
    dd = (d["states"][:, 1:, ch] - d["states"][:, :-1, ch]).ravel()
    aa = d["actions"][..., 0].ravel()
    alpha, _ = np.linalg.lstsq(np.stack([aa, np.ones_like(aa)], 1), dd, rcond=None)[0]

    fig, axes = plt.subplots(2, 2, figsize=(WIDE, 3.9))
    t = np.arange(len(lvl))

    ax = axes[0][0]  # (a) levels: the carry-over dominates
    ax.plot(t, lvl, color=ACCENT, lw=1.4, label=f"{label} (level)")
    ax.plot(
        t[1:],
        lvl[:-1],
        color=INK["muted"],
        lw=1.0,
        ls=(0, (4, 2)),
        label="previous value",
    )
    ax.set_ylabel(label, fontsize=8)
    ax.set_title("levels: the action is lost in the carry-over", fontsize=8, pad=3)
    ax.legend(
        fontsize=6.5,
        loc="upper right",
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
    )
    axgrid(ax)

    ax = axes[0][1]  # (b) increments: the action is visible
    ax.axhline(0, color=INK["rule"], lw=0.7)
    ax.plot(t[1:], inc, color=ACCENT, lw=1.4, label=rf"$\Delta$ {label}")
    ax.plot(
        t[1:],
        alpha * act,
        color=MARK,
        lw=1.4,
        ls=(0, (3, 2)),
        label=rf"action term ${alpha:.2f}\,a$",
    )
    ax.set_ylabel(rf"$\Delta$ {label}", fontsize=8)
    ax.set_title("increments: the action term is a real fraction", fontsize=8, pad=3)
    ax.legend(
        fontsize=6.5,
        loc="upper right",
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
    )
    axgrid(ax)

    seg = lvl[: 2 * patch + 1]  # (c)/(d) same motion, more patches
    for col, rr in enumerate((1, r)):
        ax = axes[1][col]
        n_real = len(seg)
        if rr == 1:
            x, y, interp_x, interp_y = np.arange(n_real), seg, [], []
        else:
            dst = np.arange((n_real - 1) * rr + 1)
            idx = np.minimum(dst // rr, n_real - 2)
            frac = (dst - idx * rr) / rr
            x, y = dst, (1 - frac) * seg[idx] + frac * seg[idx + 1]
            keep = dst % rr != 0
            interp_x, interp_y = dst[keep], y[keep]
        n_patch = int(np.ceil(len(x) / patch))
        for p in range(n_patch):  # alternating patch bands
            if p % 2 == 0:
                ax.axvspan(
                    p * patch - 0.5,
                    (p + 1) * patch - 0.5,
                    color=ACCENT,
                    alpha=0.06,
                    lw=0,
                )
            ax.axvline(p * patch - 0.5, color=INK["rule"], lw=0.6)
        ax.plot(x, y, color=ACCENT, lw=1.0, zorder=2)
        if len(interp_x):
            ax.plot(
                interp_x,
                interp_y,
                ls="",
                marker="o",
                ms=2.2,
                mfc="white",
                mec=ACCENT,
                mew=0.7,
                zorder=3,
                label="interpolated",
            )
        ax.plot(
            np.arange(n_real) * rr,
            seg,
            ls="",
            marker="o",
            ms=3.2,
            color=ACCENT,
            zorder=4,
            label="real step",
        )
        spans = [
            np.ptp(y[p * patch : (p + 1) * patch])
            for p in range(n_patch)
            if len(y[p * patch : (p + 1) * patch]) > 1
        ]
        ax.set_title(
            rf"$r={rr}$: {n_patch} patches, span/patch {np.mean(spans):.2f}",
            fontsize=8,
            pad=3,
        )
        ax.set_xlabel(
            f"position in the context (patch = {patch}, schematic)", fontsize=7.5
        )
        ax.set_xlim(-1, x[-1] + 1)  # do not draw empty patches past the data
        if col == 0:
            ax.set_ylabel(label, fontsize=8)
        else:  # only this panel has interpolated points
            ax.legend(
                fontsize=6.5,
                loc="upper center",
                ncol=2,
                framealpha=0.85,
                facecolor="white",
                edgecolor="none",
            )
        axgrid(ax, "y")

    for ax, letter in zip(axes.ravel(), "abcd"):
        ax.tick_params(labelsize=7)
        panel_label(ax, letter, x=-0.10, y=1.02, va="bottom")
    fig.tight_layout(w_pad=1.0, h_pad=0.9)
    out = save(
        fig,
        "fig4_interventions",
        outdir,
        f"The two interventions on real {config.SHORT[env_id]} data. (a) On "
        f"levels the previous value already reproduces the series, so the "
        f"action changes almost nothing a forecaster can see. (b) On "
        f"increments the same action term is a substantial share of the "
        f"signal. (c, d) Stretching inserts interpolated steps, so the same "
        f"physical motion spans more of the patch grid and each patch covers "
        f"a smaller, smoother excursion.",
    )
    print(
        f"Action term in the increment: {alpha:.3f}*a, contributing "
        f"{np.abs(alpha * act).mean():.3f} against {np.abs(inc).mean():.3f} mean "
        f"|increment| ({100 * np.abs(alpha * act).mean() / np.abs(inc).mean():.0f}%)."
    )
    return out


# ------------------------------------------------ 5  the stretch-factor sweep
def fig_stretch(grid, env_ids, outdir, budget=None):
    """One-step error against the stretch factor, one curve per model.

    The dashed open-marker curve is the reference model on levels: the gap
    between it and its own differenced curve is what differencing buys, and the
    slope along r is what stretching buys.
    """
    budget = budget or config.STRETCH_BUDGET
    d_all = grid[grid.budget == budget].dropna(subset=["nmse"])
    series = [
        (m, "diff") for m in config.TSFM_MODELS if not config.MODELS[m].get("fixed")
    ]
    series += [(config.GRID_FULL[0], "level")]

    fig, axes = plt.subplots(
        1, len(env_ids), figsize=(_width(len(env_ids)), 2.7), sharey=True, squeeze=False
    )
    for i, (ax, env_id) in enumerate(zip(axes[0], env_ids)):
        d_env = d_all[d_all.env == env_id]
        for name, pres in series:
            d = d_env[(d_env.model == name) & (d_env.presentation == pres)]
            d = d.sort_values("r")
            if d.empty:
                continue
            st = dict(STYLE[name])
            filled = pres == "diff"
            ax.plot(
                d.r,
                d.nmse,
                ms=3.5,
                **st,
                ls="-" if filled else (0, (4, 2)),
                markerfacecolor=st["color"] if filled else "white",
            )
        ax.set(xscale="log", yscale="log")
        logticks(ax, config.STRETCH_R)
        ax.yaxis.set_major_locator(LogLocator(numticks=4))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if i == len(env_ids) // 2:
            ax.set_xlabel("stretch factor $r$", fontsize=8)
        ax.set_title(config.SHORT[env_id], fontsize=8, pad=3)
        ax.tick_params(labelsize=7)
        axgrid(ax)
        if len(env_ids) > 1:
            panel_label(ax, "abcdefgh"[i])
    axes[0][0].set_ylabel("one-step NMSE", fontsize=8)
    h = [
        plt.Line2D([], [], ms=3.5, **STYLE[m], label=f"{m}, diff")
        for m, pres in series
        if pres == "diff"
    ]
    ref = config.GRID_FULL[0]
    h += [
        plt.Line2D(
            [],
            [],
            ls=(0, (4, 2)),
            ms=3.5,
            markerfacecolor="white",
            **STYLE[ref],
            label=f"{ref}, levels",
        )
    ]
    fig.tight_layout(w_pad=0.6)
    _bottom_legend(fig, h, len(env_ids))
    return save(
        fig,
        "fig5_stretch",
        outdir,
        f"One-step error against the stretch factor at $N$={budget}. Filled "
        f"markers are the differenced presentation, the open dashed curve is the "
        f"same model on levels. Stretching is a change of resolution, not of "
        f"information: N real samples become $(N-1)r+1$ tokens.",
    )


# --------------------------------------------- 6  context and data budget
def fig_scaling(grid, env_ids, outdir, plot_models=None):
    """One-step error against the amount of data each model was given.

    N is the whole context for the foundation models and the number of fitted
    transitions for the others, both drawn from episodes of the same length under
    the same policy. Read it for **where the curves cross**: that is the number of
    transitions the pretrained model saves you.
    """
    names = _plot_models(plot_models or config.SCALING_MODELS, grid)
    band = ylim("SCALING_YLIM")
    chosen, drawn = _chosen(grid, env_ids), {}
    scaling = grid.dropna(subset=["nmse"])
    if config.SCALE_PLOT_BUDGETS:
        scaling = scaling[scaling.budget.isin(config.SCALE_PLOT_BUDGETS)]
    fig, axes = plt.subplots(
        1, len(env_ids), figsize=(_width(len(env_ids)), 2.9), sharey=True, squeeze=False
    )
    for i, (ax, env_id) in enumerate(zip(axes[0], env_ids)):
        s = scaling[scaling.env == env_id]
        for name in names:
            d = _one_variant(s[s.model == name], env_id, name, chosen).sort_values(
                "budget"
            )
            if d.empty:
                continue
            drawn[name] = d.iloc[0]
            st = dict(STYLE[name])
            tsfm = models.is_tsfm(name)
            clipped_plot(
                ax,
                d.budget,
                d.nmse,
                ms=3,
                lo=band[0],
                hi=band[1],
                **st,
                markerfacecolor="white" if tsfm else st["color"],
                ls=(0, (4, 2)) if tsfm else "-",
            )
        # No "r reduced" rule any more. The grid *skips* a variant that does
        # not fit rather than quietly running a smaller r, so a curve labelled
        # r=12 is r=12 for its whole length and simply stops where it stops.
        ax.set(xscale="log", yscale="log", ylim=band)
        b_all = sorted(s.budget.unique())
        ax.set_xlim(min(b_all) * 0.75, max(b_all) * 1.35)
        logticks(
            ax,
            [
                b
                for b in sorted(s.budget.unique())
                if b in (1, 4, 16, 64, 256, 1024, 4096, 8192)
            ],
        )
        if i == len(env_ids) // 2:
            ax.set_xlabel("$N$: context steps / transitions", fontsize=8)
        ax.set_title(config.SHORT[env_id], fontsize=8, pad=3)
        ax.tick_params(labelsize=7)
        axgrid(ax)
        if len(env_ids) > 1:
            panel_label(ax, "abcdefgh"[i], x=0.03, y=0.06, va="bottom")
    axes[0][0].set_ylabel("one-step NMSE", fontsize=8)
    h = [_line_for(n, drawn) for n in names if n in drawn]
    fig.tight_layout(w_pad=0.6)
    _bottom_legend(fig, h, len(env_ids))
    return save(
        fig,
        "fig6_scaling",
        outdir,
        "How much data each model gets, on one axis, each model in the "
        "configuration named in its legend entry -- chosen off \u00a76a, and "
        "changed by editing config.PLOT_VARIANTS rather than by re-measuring "
        "anything. A curve ends where its stretched context stops fitting. "
        "For the "
        "foundation models $N$ is the whole context; for the others it is "
        "the number of transitions fitted on, drawn from episodes of the "
        "same length under the same policy.",
    )


# ---------------------------------------- 6a  every configuration, on N
def fig_scaling_configs(grid, env_ids, outdir, model_names=None):
    """One-step error against N, every variant of every model, a panel each.

    §6 draws one configuration per model and answers "which model"; this answers
    "does the configuration matter", which is the question you have to settle
    before the other one means anything. It reads ``grid.csv`` -- the same metric
    §6 plots, on the coarser budget axis the hyperparameter sweep already paid
    for -- so changing what it shows never costs a recompute.
    """
    names = _sweep_models(grid, model_names)
    chosen = _chosen(grid, env_ids)

    def draw(ax, d, **kw):
        d = d.sort_values("budget")
        clipped_plot(ax, d.budget, d.nmse, **kw)

    fig, steps = _configs_grid(
        grid,
        env_ids,
        names,
        chosen,
        draw,
        ylabel="one-step NMSE",
        xlabel="$N$: context steps / transitions",
    )
    for row in fig.axes:
        row.set_xscale("log")
    logticks(fig.axes[-1], [b for b in sorted(grid.budget.unique()) if b > 0])
    fig.tight_layout(w_pad=0.6, h_pad=0.4)
    _config_legend(fig, names, steps, len(env_ids))
    return save(
        fig,
        "fig6a_scaling_configs",
        outdir,
        "Every configuration of every model against the data budget, one panel "
        "per model. "
        "Lightness is the sweep value -- the stretch factor $r$ for the foundation models, the history length for the trained ones -- and the dash is the presentation; the heavy marked curve is the one the combined figure draws. "
        "Read it for whether the spread between a model's own variants is larger "
        "or smaller than the gap between models.",
    )


# ------------------------------------------------------- 7  multi-step rollout
def fig_rollout(rollout, env_ids, outdir, plot_models=None, budgets=None):
    """Open-loop rollout with the true action sequence known throughout.

    Rows are data budgets, columns are environments. Persistence sits at 1 for
    every horizon, so where a curve crosses that line the model has stopped being
    useful for planning.
    """
    names = _plot_models(plot_models or config.ROLLOUT_MODELS, rollout)
    band = ylim("ROLLOUT_YLIM")
    chosen, drawn = _chosen(rollout, env_ids), {}
    have = sorted(rollout.budget.unique())
    if budgets is None:
        budgets = config.ROLL_PLOT_BUDGETS or have
    budgets = [n for n in budgets if n in have] or have
    H = int(rollout.h.max())
    if config.ROLL_PLOT_H:
        H = min(H, int(config.ROLL_PLOT_H))
        rollout = rollout[rollout.h <= H]
    mark = max(1, round(H / 4))
    fig, axes = plt.subplots(
        len(budgets),
        len(env_ids),
        figsize=(_width(len(env_ids)), 2.0 * len(budgets) + 0.5),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for r_i, n in enumerate(budgets):
        for c_i, env_id in enumerate(env_ids):
            ax = axes[r_i][c_i]
            d_all = rollout[(rollout.env == env_id) & (rollout.budget == n)]
            for name in names:
                d = _one_variant(
                    d_all[d_all.model == name], env_id, name, chosen
                ).sort_values("h")
                if d.empty:
                    continue
                drawn[name] = d.iloc[0]
                st = dict(STYLE[name])
                tsfm = models.is_tsfm(name)
                clipped_plot(
                    ax,
                    d.h,
                    d.nmse,
                    ms=3,
                    markevery=mark,
                    lo=band[0],
                    hi=band[1],
                    **st,
                    markerfacecolor="white" if tsfm else st["color"],
                    ls=(0, (4, 2)) if tsfm else "-",
                )
            ax.set(yscale="log", ylim=band, xlim=(1, H))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(labelsize=7)
            axgrid(ax)
            if r_i == 0:
                ax.set_title(config.SHORT[env_id], fontsize=8, pad=3)
            if r_i == len(budgets) - 1 and c_i == len(env_ids) // 2:
                ax.set_xlabel("horizon $h$", fontsize=8)
            if c_i == 0:
                ax.set_ylabel(f"$N={n}$\nNMSE$(h)$", fontsize=8)
            if d_all.empty:
                ax.text(
                    0.5,
                    0.5,
                    "budget exceeds\nthis env's context",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7,
                    color=INK["muted"],
                )
            # No per-panel r any more: the file holds every variant, so the max
            # over the panel is a property of the sweep rather than of anything
            # drawn. Each legend entry carries its own r instead.
    h = [_line_for(n, drawn) for n in names if n in drawn]
    fig.tight_layout(w_pad=0.6, h_pad=0.5)
    _bottom_legend(fig, h, len(env_ids))
    return save(
        fig,
        "fig7_rollout",
        outdir,
        f"Open-loop rollout to $h={H}$ with the true actions known, "
        f"repeated at {len(budgets)} data budgets. Every model is drawn in one "
        f"configuration, named in the legend; the full sweep behind it is §7a. "
        f"A row is one budget: the trained models were fitted on $N$ "
        f"transitions, the foundation models given $N$ steps of context, "
        f"stretched by the $r$ its legend entry names.",
    )


# ------------------------------------ 7a  every configuration, over the horizon
def fig_rollout_configs(rollout, env_ids, outdir, budget=None, model_names=None):
    """Rollout error against the horizon, every variant of every model.

    The figure the study was missing: a variant that wins at h=1 need not win at
    h=20, because differencing is an integration and the error it leaves is
    accumulated rather than re-anchored. One budget only -- the panel grid is
    already model x environment.
    """
    budget = budget or config.ROLL_CONFIG_BUDGET
    have = sorted(rollout.budget.unique())
    if budget not in have:  # the configured budget was never computed
        budget = min(have, key=lambda b: abs(b - budget))
    d_all = rollout[rollout.budget == budget]
    if config.ROLL_PLOT_H:
        d_all = d_all[d_all.h <= config.ROLL_PLOT_H]
    names = _sweep_models(d_all, model_names)
    chosen = _chosen(rollout, env_ids)

    def draw(ax, d, **kw):
        d = d.sort_values("h")
        clipped_plot(ax, d.h, d.nmse, **kw)

    fig, steps = _configs_grid(
        d_all, env_ids, names, chosen, draw, ylabel="NMSE$(h)$", xlabel="horizon $h$"
    )
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(w_pad=0.6, h_pad=0.4)
    _config_legend(fig, names, steps, len(env_ids))
    return save(
        fig,
        "fig7a_rollout_configs",
        outdir,
        f"Every configuration of every model over the horizon, at $N$={budget}, "
        f"one panel per model. "
        f"Lightness is the sweep value -- the stretch factor $r$ for the foundation models, the history length for the trained ones -- and the dash is the presentation; the heavy marked curve is the one the combined figure draws. "
        f"Persistence sits at 1 at every $h$, so a curve crossing that line has "
        f"stopped being useful for planning; where a model's own variants cross "
        f"each other is where the configuration the one-step sweep prefers stops "
        f"being the right one.",
    )


# ------------------------------------------------- 8  the rollout, as states
def fig_trajectories(traj, env_id, outdir, budget=None, plot_models=None):
    """Ground truth and each model's prediction over the horizon, on the states
    themselves rather than on an error.

    An error curve says how wrong a model is; this says *how* it is wrong -- a
    phase lag, a damped amplitude, a drift off the manifold all look identical in
    NMSE and different here. Rows are observation channels, columns are example
    windows from the same evaluation set section 7 scores.

    Two knobs, both read here rather than when anything was measured, because the
    rollout kept these states for every variant it ran:

    ``budget``            which N, defaulting to ``config.TRAJ_BUDGET``
    ``PLOT_VARIANTS``     which configuration of each model, as in §6 and §7

    A combination the rollout never measured raises, naming what is available,
    rather than quietly dropping a model out of the panel.
    """
    d_env = traj[traj.env == env_id]
    if d_env.empty:
        raise ValueError(
            f"no trajectory rows for {env_id}. The rollout stage writes these "
            f"for the budgets in config.TRAJ_BUDGETS -- run it, or add the "
            f"budget you want there and re-run it with --force."
        )

    budget = int(budget or config.TRAJ_BUDGET)
    have_budgets = sorted(int(b) for b in d_env.budget.unique() if b)
    if budget not in have_budgets:
        raise ValueError(
            f"{env_id}: no trajectory states at N={budget}. Measured budgets are "
            f"{have_budgets} (config.TRAJ_BUDGETS was {config.TRAJ_BUDGETS} when "
            f"the rollout ran). Set config.TRAJ_BUDGET to one of them, or add "
            f"N={budget} to TRAJ_BUDGETS and re-run the rollout with --force."
        )
    # truth is stored once, under budget 0, because it does not depend on either
    d_env = d_env[(d_env.budget == budget) | (d_env.model == "truth")]

    if config.ROLL_PLOT_H:
        d_env = d_env[d_env.h <= config.ROLL_PLOT_H]

    # one variant per model, the same choice §6 and §7 make
    chosen = _chosen(d_env, [env_id])
    names, missing = [], []
    for name in plot_models or config.PLOT_MODELS:
        d = d_env[d_env.model == name]
        if d.empty:
            continue
        want = (chosen or {}).get((env_id, name))
        if want is not None and want not in set(d.variant):
            missing.append((name, want, sorted(d.variant.unique())))
            continue
        names.append(name)
    if missing:
        detail = "\n".join(
            f"    {m}: asked for {w!r}, measured at N={budget}: {got}"
            for m, w, got in missing
        )
        raise ValueError(
            f"{env_id}: config.PLOT_VARIANTS names configurations the rollout "
            f"did not measure at N={budget}:\n{detail}\n"
            f"  Stretching eats the context, so a large r may only exist at the "
            f"smaller budgets -- pick another N, or another variant."
        )
    d_env = pd.concat(
        [d_env[d_env.model == "truth"]]
        + [_one_variant(d_env[d_env.model == n], env_id, n, chosen) for n in names]
    )

    mark = max(1, round(d_env.h.max() / 4))
    windows = sorted(d_env.window.unique())
    channels = sorted(d_env.channel.unique())
    spec = pipeline.spec_of(env_id)
    labels = {c: spec.labels[c] for c in channels}
    drawn = {
        n: {
            "model": n,
            **variant_fields(n, env_id, d_env[d_env.model == n].variant.iloc[0]),
        }
        for n in names
    }

    fig, axes = plt.subplots(
        len(channels),
        len(windows),
        figsize=(_width(len(windows), 2.0), 1.25 * len(channels)),
        squeeze=False,
        sharex=True,
    )
    for r_i, c in enumerate(channels):
        for c_i, w in enumerate(windows):
            ax = axes[r_i][c_i]
            d = d_env[(d_env.channel == c) & (d_env.window == w)]
            truth = d[d.model == "truth"].sort_values("h")
            ax.plot(truth.h, truth.value, color=INK["text"], lw=1.6, zorder=5)
            for name in names:
                dm = d[d.model == name].sort_values("h")
                if dm.empty:
                    continue
                ax.plot(
                    dm.h,
                    dm.value,
                    ms=2.5,
                    markevery=mark,
                    **STYLE[name],
                    ls=(0, (4, 2)),
                    markerfacecolor="white",
                )
            ax.tick_params(labelsize=7)
            axgrid(ax)
            if c_i == 0:
                ax.set_ylabel(labels[c], fontsize=7.5)
            if r_i == 0:
                ax.set_title(f"window {w + 1}", fontsize=8, pad=3)
            if r_i == len(channels) - 1 and c_i == len(windows) // 2:
                ax.set_xlabel("horizon $h$", fontsize=8)
    h = [plt.Line2D([], [], color=INK["text"], lw=1.6, label="ground truth")]
    h += [_line_for(n, drawn) for n in names]
    fig.align_ylabels()
    fig.tight_layout(w_pad=0.8, h_pad=0.4)
    _bottom_legend(fig, h, len(windows))
    return save(
        fig,
        f"fig8_trajectories_{config.ENV_DIRS[env_id]}",
        outdir,
        f"Open-loop rollout on {config.SHORT[env_id]} as states rather than as an "
        f"error, at $N$={budget} with the true actions known throughout. Black is "
        f"the environment, dashed is each model. Rows are observation channels, "
        f"columns are windows from the same evaluation set section 7 scores.",
    )
