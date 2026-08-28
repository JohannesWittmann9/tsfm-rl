"""The hyperparameter study (section 5) as one table.

``grid.csv`` holds one one-step NMSE per (model, variant, budget); ``table_grid``
lays all of it out and marks the variant sections 6-8 use. ``latex_grid`` is the
same content as a LaTeX table for the appendix.
"""

import config
import models
import pandas as pd
import pipeline


def _variant_order(env_id):
    """Variants in the order ``config.MODELS`` and the sweeps define them, so the
    table never falls back on alphabetical order (which puts r=16 before r=2)."""
    return [
        (name, v["variant"])
        for name in config.MODELS
        for v in models.variants(name, env_id)
    ]


def table_grid(grid, env_id, mark="<-"):
    """Every variant against every budget, grouped by model.

    A blank cell is a configuration *not defined* at that budget -- a stretched
    context that would overflow the window, a VARX with more parameters than design
    rows, a Moirai whose context is shorter than its horizon -- not a bad score.
    """
    d = grid[grid.env == env_id]
    piv = d.pivot_table(index=["model", "variant"], columns="budget", values="nmse")
    order = [k for k in _variant_order(env_id) if k in piv.index]
    piv = piv.reindex(order)
    piv.columns = [int(c) for c in piv.columns]
    chosen = {
        (m, b["variant"]) for m, b in pipeline.best_variants(grid, env_id).items()
    }
    piv[""] = [mark if k in chosen else "" for k in piv.index]
    return piv


def show(df, digits=4, best="min"):
    """A table formatted for a notebook: fixed decimals on the numeric columns,
    missing cells as ``--``, and the best entry per budget in bold."""
    num = list(df.select_dtypes("number").columns)
    st = df.style.format(f"{{:.{digits}f}}", subset=num, na_rep="--")
    if best == "min" and num and len(df):
        st = st.highlight_min(subset=num, axis=0, props="font-weight:bold")
    return st


# ------------------------------------------------------------------- LaTeX
def _marked_column(col, digits):
    """One budget column, formatted, best in bold and second best underlined.

    Ties share a place, so two equal minima are both bold and nothing is
    underlined -- better than breaking the tie on row order, which would make the
    table depend on the sweep's iteration order.
    """
    ranked = sorted(dict.fromkeys(col.dropna()))
    best = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    out = []
    for v in col:
        if pd.isna(v):  # undefined at this budget, not a bad score
            out.append("")
        elif v == best:
            out.append(rf"\textbf{{{v:.{digits}g}}}")
        elif v == second:
            out.append(rf"\underline{{{v:.{digits}g}}}")
        else:
            out.append(f"{v:.{digits}g}")
    return out


def _row_label(model, variant):
    """``Chronos-2 S diff r=6`` -> ``diff, $r{=}6$``; the model is a group header,
    so the row only carries what distinguishes it."""
    tail = variant[len(model) :].strip()
    if tail.startswith(("diff", "level")):
        kind, _, rest = tail.partition(" ")
        return f"{kind}, ${rest.replace('r=', 'r{=}')}$"
    return tail  # "lag 4"


def latex_grid(grid, env_id, digits=3, caption=None, label=None):
    """The whole section 5 grid as one LaTeX table, for the appendix.

    Best value in each budget column is bold, second best underlined. Needs
    ``booktabs``; it is wide, so wrap it in ``sidewaystable`` (the ``rotating``
    package) or a ``\\resizebox`` unless the page is generous.
    """
    d = grid[grid.env == env_id]
    piv = d.pivot_table(index=["model", "variant"], columns="budget", values="nmse")
    piv = piv.reindex([k for k in _variant_order(env_id) if k in piv.index])
    budgets = [int(c) for c in piv.columns]
    marked = piv.apply(lambda c: _marked_column(c, digits))

    short = config.SHORT.get(env_id, env_id)
    caption = caption or (
        f"One-step NMSE on {short} for every configuration at every data budget "
        f"$N$. Best per column in bold, second best underlined. A blank cell is a "
        f"configuration undefined at that budget -- a stretched context that would "
        f"overflow the window, a VARX with more parameters than design rows -- not "
        f"a bad score. Persistence scores 1.0."
    )
    label = label or f"tab:grid-{config.ENV_DIRS[env_id]}"

    head = " & ".join(f"${b}$" for b in budgets)
    lines = [
        r"\begin{table}[t]",
        r"\centering\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{l{'r' * len(budgets)}}}",
        r"\toprule",
        rf"configuration & {head} \\",
        r"\midrule",
    ]
    last = None
    for (model, variant), row in zip(marked.index, marked.to_numpy()):
        if model != last:
            if last is not None:
                lines.append(r"\addlinespace")
            lines.append(
                rf"\multicolumn{{{len(budgets) + 1}}}{{l}}{{\textit{{{model}}}}} \\"
            )
            last = model
        cells = " & ".join(row)
        lines.append(rf"\quad {_row_label(model, variant)} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
