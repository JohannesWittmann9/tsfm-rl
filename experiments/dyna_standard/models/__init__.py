"""The model registry: one interface, four families.

Every model exposes the same call::

    predict(context_states, context_actions, future_actions) -> (batch, H, n_obs)

always on *levels*, so every table compares like with like. Which models exist,
and with which weights and colours, is decided in ``config.MODELS`` -- this module
only knows how to turn one of those entries into an object.

A model's hyperparameter is its *variant*:

===================  ===========================================
Chronos-2, Moirai    presentation (level | diff) x stretch factor
MLP, VARX            history length (lag)
===================  ===========================================
"""

import config

from .chronos import ChronosDynamics
from .mlp import MLPDynamics
from .moirai import MoiraiDynamics
from .varx import VARDynamics
from .wrappers import FROM_CFG, UpsampledDynamics

__all__ = [
    "FROM_CFG",
    "ChronosDynamics",
    "MLPDynamics",
    "MoiraiDynamics",
    "UpsampledDynamics",
    "VARDynamics",
    "build",
    "context_limit",
    "is_tsfm",
    "usable_r",
    "variant_name",
    "variants",
]


def is_tsfm(name):
    return config.MODELS[name]["kind"] in config.TSFM_KINDS


def context_limit(name):
    """Longest context this model can be handed, in tokens."""
    kind = config.MODELS[name]["kind"]
    if kind == "chronos":
        return config.CHRONOS_CTX
    if kind == "moirai":
        return config.MOIRAI_CTX
    return None  # trained models read their last `lag` rows only


def variant_name(name, presentation=None, r=None, lag=None):
    """The label a variant carries in every results file and every table."""
    if is_tsfm(name):
        return f"{name} {presentation} r={r}"
    return f"{name} lag {lag}"


def variants(name, env_id):
    """The variant grid for one model, as dicts ready to be rows of ``grid.csv``.

    The presentation x r grid *is* the four presentations of section 2b, not a
    separate experiment: ``level, r=1`` is the model out of the box, ``diff, r=1``
    is differencing alone, ``level, r>1`` is stretching alone, ``diff, r>1`` is
    both. Only ``config.GRID_FULL`` models pay for the full factorial; the others
    get the differenced sweep plus the two ``level`` points the presentation table
    needs.
    """
    if not is_tsfm(name):
        return [
            dict(
                model=name,
                variant=variant_name(name, lag=lag),
                presentation="-",
                r=0,
                lag=lag,
            )
            for lag in config.MODELS[name]["lags"]
        ]
    fixed = config.MODELS[name].get("fixed")
    if fixed:  # a pinned variant: nothing to sweep
        return [
            dict(
                model=name,
                variant=variant_name(name, **fixed),
                presentation=fixed["presentation"],
                r=fixed["r"],
                lag=-1,
            )
        ]
    if name in config.GRID_FULL:
        pairs = [(p, r) for p in config.PRESENTATIONS for r in config.STRETCH_R]
    else:
        pairs = sorted(
            {("diff", r) for r in config.STRETCH_R}
            | {("level", 1), ("level", config.DEFAULT_R[env_id])}
        )
    return [
        dict(
            model=name,
            variant=variant_name(name, presentation=p, r=r),
            presentation=p,
            r=r,
            lag=-1,
        )
        for p, r in pairs
    ]


def usable_r(name, n, prefer):
    """Largest stretch factor <= ``prefer`` that keeps N real steps inside the
    context. Used by the sweeps that must show a curve at every budget; the
    section 5 grid instead *skips* what does not fit, so a row labelled r=16 is
    never quietly a different r."""
    limit = context_limit(name)
    r = int(prefer)
    while r > 1 and (n - 1) * r + 1 > limit:
        r //= 2
    return r


def fits_context(name, n, r, h=0):
    """Does N real steps at stretch ``r`` fit, with an ``h``-step forecast after it?

    ``h`` defaults to 0 because the one-step stages only have to fit the context.
    A rollout does not: stretching multiplies the *prediction* length too, so an
    r that fits the context alone can still ask for an r*h-step forecast on top
    of it, and that has to be checked rather than assumed.
    """
    limit = context_limit(name)
    return limit is None or (n - 1) * r + 1 + h * r <= limit


def build(name, spec, presentation="diff", r=1, lag=1, fit=None):
    """One model in one variant. ``fit`` is ``(states, actions)`` and is required
    for the trained families only."""
    cfg = config.MODELS[name]
    kind = cfg["kind"]
    if kind in config.TSFM_KINDS:
        presentation, r = (
            (cfg.get("fixed") or {}).get("presentation", presentation),
            (cfg.get("fixed") or {}).get("r", r),
        )
        difference = FROM_CFG if presentation == "diff" else ()
        if kind == "chronos":
            m = ChronosDynamics(
                spec,
                model_id=cfg["model_id"],
                difference=difference,
                batch_size=config.CHRONOS_BATCH,
            )
        else:
            m = MoiraiDynamics(
                spec,
                model_id=cfg["model_id"],
                difference=difference,
                num_samples=config.MOIRAI_SAMPLES,
                batch_size=config.CHRONOS_BATCH,
            )
        return UpsampledDynamics(m, factor=r) if r > 1 else m
    if fit is None:
        raise ValueError(f"{name} is a trained model and needs fit=(states, actions)")
    states, actions = fit
    if kind == "mlp":
        return MLPDynamics(spec, states, actions, lag=lag, seed=config.SEED)
    if kind == "varx":
        return VARDynamics(spec, states, actions, lag=lag)
    raise ValueError(f"unknown model kind {kind!r} for {name!r}")
