"""The hyperparameter study (section 5) as one table.

``grid.csv`` holds one one-step NMSE per (model, variant, budget); ``table_grid``
lays all of it out and marks the variant sections 6-8 use.
"""

import config
import models
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
