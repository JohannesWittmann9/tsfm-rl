"""The four experiment stages, each cached to one CSV per environment.

    probe     section 2b  -- the counterfactual action probe under four
                             presentations of the same trajectory
    grid      section 5+6 -- every model, every variant, every budget; §6 is
                             this table filtered to one variant per model
    rollout   section 7+8 -- every model, every variant, over the horizon;
                             §8 is the raw states it keeps at TRAJ_BUDGETS

Compute lives here and in ``<env>/<env>.py``; the notebooks only read the CSVs.
Every stage resumes: rows already in the file are skipped, so adding a model to
``config.MODELS`` computes only the new rows.

    python experiments/dyna_standard/pendulum/pendulum.py --stages grid scaling
"""

import argparse
import time

import config
import models
import numpy as np
import pandas as pd
from envs import EnvSpec, collect_rollouts, longest_rollouts, take_transitions
from evaluate import (
    EvalSet,
    model_response,
    nmse,
    probe_indices,
    reference_response,
    score,
)

STAGES = ["probe", "grid", "rollout"]

# The four presentations of section 2b: (label, difference?, stretch?)
CONDITIONS = [
    ("levels", False, False),
    ("+ differencing", True, False),
    ("+ stretching", False, True),
    ("both", True, True),
]

_SPECS, _EPISODES, _STUDY = {}, {}, {}


# ------------------------------------------------------------------ shared data
def spec_of(env_id):
    if env_id not in _SPECS:
        _SPECS[env_id] = EnvSpec(env_id, episode_len=config.EPISODE_LEN)
    return _SPECS[env_id]


def episodes(env_id):
    """The short-episode draw behind figures 1, 1b, 4 and the section 2b probe."""
    if env_id not in _EPISODES:
        _EPISODES[env_id] = collect_rollouts(
            spec_of(env_id), n_episodes=config.N_EPISODES, seed=config.SEED
        )
    return _EPISODES[env_id]


def study(env_id):
    """The long-episode setup behind sections 5-7.

    Fit pool and evaluation windows come from episodes of the **same length under
    the same policy**, differing only in seed. That is a fairness fix, not a
    detail: fitting the trained baselines on short episodes and scoring them on
    long ones puts a quarter of some environments' evaluation states outside the
    range those models ever saw, while the foundation models read their context
    straight out of the evaluation episodes.
    """
    if env_id in _STUDY:
        return _STUDY[env_id]
    spec = spec_of(env_id)
    cap, ev_data = longest_rollouts(
        spec,
        max(config.SCALE_BUDGETS) + config.HP_MARGIN,
        config.HP_EVAL_EPISODES,
        config.SEED + 200,
    )
    ctx = cap - config.HP_MARGIN
    try:
        pool = collect_rollouts(
            spec,
            n_episodes=config.HP_POOL_EPISODES,
            seed=config.SEED + 100,
            episode_len=cap,
        )
    except RuntimeError:
        _, pool = longest_rollouts(
            spec, cap, config.HP_POOL_EPISODES, config.SEED + 100
        )
    pool_T = pool["actions"].shape[1]
    top = min(ctx, pool_T)
    budgets = [n for n in config.HP_BUDGETS if n <= top]
    st = dict(
        cap=cap,
        ctx=ctx,
        pool=pool,
        budgets=budgets,
        scale_budgets=[n for n in config.SCALE_BUDGETS if n <= top],
        ev_raw=ev_data,
        ev=EvalSet(
            spec, ev_data["states"], ev_data["actions"], L=ctx, H=1, n=config.HP_WINDOWS
        ),
    )
    # Every budget must arrive exactly, or the trained curves sit at different x
    # positions from the foundation-model ones and the comparison is not
    # like-for-like. Checked here rather than assumed.
    for n in budgets:
        _, a = take_transitions(pool, n)
        got = int(a.shape[0] * a.shape[1])
        if got != n:
            raise AssertionError(f"{env_id}: budget {n} delivered {got}")
    _STUDY[env_id] = st
    return st


# ----------------------------------------------------------------------- cache
def stage_path(env_id, stage):
    return config.results_dir(env_id) / f"{stage}.csv"


def load(env_id, stage, compute=False):
    """One environment's stage results. Missing results are computed only when
    asked for; otherwise the error names the command that produces them."""
    path = stage_path(env_id, stage)
    if path.exists():
        return pd.read_csv(path)
    if compute:
        print(f"{path} is missing -- running the {stage} stage for {env_id} now")
        return run(env_id, [stage])[stage]
    script = f"experiments/dyna_standard/{config.ENV_DIRS[env_id]}/{config.ENV_DIRS[env_id]}.py"
    raise FileNotFoundError(
        f"{path} is missing. Run:\n    python {script} --stages {stage}"
    )


def load_many(env_ids, stage, compute=False):
    return pd.concat([load(e, stage, compute) for e in env_ids], ignore_index=True)


def _write(df, env_id, stage):
    path = stage_path(env_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _existing(env_id, stage, keys, force, evalset):
    """Rows already computed, and the set of key tuples they cover.

    ``evalset`` names what the numbers were measured on -- how many evaluation
    windows, at what context and horizon. Rows carrying a different one were
    measured against a different set and are recomputed rather than resumed,
    because otherwise a changed window count leaves two protocols on one axis.
    """
    path = stage_path(env_id, stage)
    if force or not path.exists():
        return pd.DataFrame(), set()
    done = pd.read_csv(path)
    if not len(done) or not set(keys) <= set(done.columns):
        return pd.DataFrame(), set()
    fresh = (
        done[done.evalset.astype(str) == evalset]
        if "evalset" in done.columns
        else done.iloc[:0]
    )
    if len(fresh) < len(done):
        print(
            f"{env_id}: {stage} -- {len(done) - len(fresh)} cached rows were "
            f"measured on a different evaluation set ({evalset} now): recomputing"
        )
    return fresh, set(map(tuple, fresh[list(keys)].astype(str).values))


# ------------------------------------------------------- stage 1: the 2b probe
def run_probe(env_id, force=False):
    """Section 2b: the same probe, the same contexts, the same models -- only the
    presentation of the trajectory changes, so whatever the panels show is a fact
    about the model rather than about the environment."""
    spec = spec_of(env_id)
    d = episodes(env_id)
    ev = EvalSet(spec, d["states"], d["actions"], n=config.PROBE_WINDOWS)
    idx = probe_indices(ev)
    probe = spec["probe_actions"]
    r_env = config.DEFAULT_R[env_id]

    ek = f"{len(idx)}of{len(ev)}w@{config.L}/{config.N_EPISODES}x{config.EPISODE_LEN}"
    done, have = _existing(env_id, "probe", ("condition", "model"), force, ek)
    rows = []
    ref_curve, ref_slope = reference_response(ev, idx)
    if ("reference", "environment") not in have:
        rows += [
            dict(
                env=env_id,
                condition="reference",
                model="environment",
                r=1,
                action=float(a),
                response=float(v),
                slope_pct=100.0,
                evalset=ek,
            )
            for a, v in zip(probe, ref_curve)
        ]
    print(f"{env_id}: probe over {len(idx)} contexts, reference slope {ref_slope:.4f}")

    for cond, diff, stretch in CONDITIONS:
        for name in config.PROBE_MODELS:
            if (cond, name) in have:
                continue
            t0 = time.time()
            r = r_env if stretch else 1
            m = models.build(name, spec, presentation="diff" if diff else "level", r=r)
            curve, slope = model_response(m, ev, idx)
            pct = 100 * slope / ref_slope if ref_slope else np.nan
            rows += [
                dict(
                    env=env_id,
                    condition=cond,
                    model=name,
                    r=r,
                    action=float(a),
                    response=float(v),
                    slope_pct=pct,
                    evalset=ek,
                )
                for a, v in zip(probe, curve)
            ]
            print(
                f"   {cond:<15s} {name:<16s} slope {pct:7.1f}%"
                f"   [{time.time() - t0:5.1f}s]",
                flush=True,
            )
    return _write(
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True), env_id, "probe"
    )


# ------------------------------------------------- stage 2: the variant grid
def _grid_plan(env_id, budgets, h=0):
    """Every model x variant x budget worth running, as rows ready for a stage.

    ``h`` is the forecast length the cell will ask for: 0 for the one-step grid,
    ``config.ROLL_H`` for the rollout, where stretching multiplies the prediction
    length too and a variant can fit the context but not the context plus its own
    forecast.
    """
    plan = []
    for name in config.GRID_MODELS:
        for v in models.variants(name, env_id):
            for n in budgets:
                if models.is_tsfm(name) and not models.fits_context(name, n, v["r"], h):
                    continue  # would overflow the window: skip, not fall back
                plan.append({**v, "env": env_id, "budget": n})
    return plan


def _estimate_minutes(plan, h=0, windows=None):
    """Rough cost, so a multi-hour run is a decision rather than a surprise. The
    constant is measured: ~2.0 ms per context token per window for Chronos-2 S on
    CPU at batch 16, near-linear in tokens.

    ``h`` adds the forecast the cell asks for, which stretching multiplies too:
    at r=12 a 50-step horizon costs 600 more tokens per window, not 50.
    """
    per_token = 2.02e-3 / 24
    windows = (
        (config.ROLL_WINDOWS if h else config.HP_WINDOWS)
        if windows is None
        else windows
    )
    total = 0.0
    for p in plan:
        cost = config.MODELS[p["model"]].get("cost", 0.0)
        r = max(p["r"], 1)
        tokens = (p["budget"] - 1) * r + 1 + h * r
        total += tokens * per_token * windows * cost
    return total / 60


def run_grid(env_id, force=False):
    """Section 5: every model, every variant, every budget, one number per cell.

    N means the same thing on both sides -- context steps for a foundation model,
    training transitions for a fitted one -- and both are drawn from the matched
    distributions built in :func:`study`. Stretching does not count against the
    budget (N real samples become (N-1)r+1 tokens, a processing choice) but it does
    consume the context, so combinations that would overflow are skipped rather
    than quietly run at a smaller r.
    """
    spec, st = spec_of(env_id), study(env_id)
    ev, pool = st["ev"], st["pool"]
    plan = _grid_plan(env_id, st["budgets"])
    ek = f"{len(ev)}w@{st['ctx']}"
    done, have = _existing(env_id, "grid", ("variant", "budget"), force, ek)
    todo = [p for p in plan if (p["variant"], str(p["budget"])) not in have]
    print(
        f"{env_id}: grid {len(plan)} cells, {len(todo)} to run, "
        f"context {st['ctx']}, {len(ev)} windows, budgets {st['budgets']}"
    )
    print(
        f"   estimated {_estimate_minutes(todo):.0f} min of foundation-model "
        f"calls on CPU (levers: HP_WINDOWS, HP_BUDGETS, GRID_FULL, MODELS)"
    )

    rows, t0_all, k = [], time.time(), 0
    for n in st["budgets"]:
        block = [p for p in todo if p["budget"] == n]
        if not block:
            continue
        fit = take_transitions(pool, n)
        print(f"=== {env_id}  N={n}  ({len(block)} cells)")
        for p in block:
            t0 = time.time()
            if models.is_tsfm(p["model"]):
                m = models.build(
                    p["model"], spec, presentation=p["presentation"], r=p["r"]
                )
                val = score(m, ev, n)
            else:
                try:
                    m = models.build(p["model"], spec, lag=p["lag"], fit=fit)
                    val = score(m, ev, None)
                except Exception:
                    val = np.nan  # undefined at this budget
            rows.append({**p, "nmse": val, "seconds": time.time() - t0, "evalset": ek})
            k += 1
            eta = (time.time() - t0_all) / k * (len(todo) - k)
            print(
                f"   {p['variant']:<26s} NMSE "
                + (f"{val:9.4f}" if np.isfinite(val) else "      n/a")
                + f"   [{rows[-1]['seconds']:5.1f}s]  {k}/{len(todo)} "
                f"eta {eta / 60:.0f}m",
                flush=True,
            )
        # checkpoint after every budget block, so a long run survives a stop
        _write(pd.concat([done, pd.DataFrame(rows)], ignore_index=True), env_id, "grid")
    return _write(
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True), env_id, "grid"
    )


# ------------------------------------------------------- the selection rule
def variant_scores(grid, env_id, model):
    """Rank each variant of one model across budgets.

    Mean rank, not mean NMSE: the errors span six decades, so an average would be
    decided entirely by the largest budget. A variant that cannot run at a budget
    takes the worst rank there rather than being dropped -- a configuration that
    overflows the context at large N is genuinely less useful, and excusing it
    would let r=16 win a comparison it never entered.

    N=1 is the exception, and is dropped: differencing has no increment to hand
    over at a single sample, so it is *structurally* undefined there rather than
    too expensive. Left in, that one column charges every differenced variant the
    worst rank and decides the comparison on its own -- on CartPole it is the
    whole reason `level` won a sweep `diff` leads at seven of the nine budgets
    both can run.
    """
    d = grid[(grid.env == env_id) & (grid.model == model)]
    piv = d.pivot_table(index="variant", columns="budget", values="nmse")
    piv = piv.drop(columns=[c for c in piv.columns if c <= 1])
    ranks = piv.rank(axis=0, method="min").fillna(len(piv) + 1)
    with np.errstate(divide="ignore"):
        geo = np.exp(np.log(piv.clip(lower=1e-12)).mean(axis=1, skipna=True))
    return pd.DataFrame(
        {"mean_rank": ranks.mean(1), "geo_nmse": geo, "budgets_ok": piv.notna().sum(1)}
    )


def _override(env_id, name):
    """``config.PLOT_VARIANTS`` for one model, or None. A named environment's
    entry wins over the ``None`` block that applies to all of them."""
    for key in (env_id, None):
        block = config.PLOT_VARIANTS.get(key) or {}
        if name in block:
            return dict(block[name])
    return None


def _apply_override(env_id, name, chosen, sc):
    """Replace an automatic pick with the one ``config.PLOT_VARIANTS`` asks for.

    A variant the model does not have raises rather than falling back: the knob
    exists to put a *named* curve in the paper figure, and a typo that silently
    drew a different one would be worse than no knob at all. ``sc`` is the score
    table, so the row reports the chosen variant's own rank rather than keeping
    the rank of the variant it replaced.
    """
    want = _override(env_id, name)
    if want is None:
        return {**chosen, "source": "selected"}
    have = {v["variant"]: v for v in models.variants(name, env_id)}
    variant = models.variant_name(name, **want)
    if variant not in have:
        raise ValueError(
            f"config.PLOT_VARIANTS asks for {variant!r} on {env_id}, which "
            f"{name} does not have. Its variants are: {sorted(have)}"
        )
    v = have[variant]
    return {
        **chosen,
        "variant": variant,
        "presentation": v["presentation"],
        "r": int(v["r"]),
        "lag": int(v["lag"]),
        # NaN, not the replaced variant's score, when the grid never ran it
        "mean_rank": float(sc.mean_rank.get(variant, np.nan)),
        "geo_nmse": float(sc.geo_nmse.get(variant, np.nan)),
        "source": "manual",
    }


def best_variants(grid, env_id):
    """The configuration each model is carried into sections 6 and 7 in.

    Selected on error alone, by ``config.SELECT_RULE``, then overridden by
    ``config.PLOT_VARIANTS`` where that names a variant -- so the figures can be
    drawn in a configuration you chose rather than one a one-step rule chose.
    Each entry carries ``source``, "selected" or "manual", so the notebook table
    says which it was.

    Worth knowing what selecting on error does not do: on a low-signal
    environment the lowest error can belong to a configuration that has stopped
    reading the action, which forecasts well and plans not at all.
    """
    out = {}
    for name in grid[grid.env == env_id].model.unique():
        d = grid[(grid.env == env_id) & (grid.model == name)]
        if not d.nmse.notna().any():
            continue
        sc = variant_scores(grid, env_id, name)
        order = (
            ["mean_rank", "geo_nmse"]
            if config.SELECT_RULE == "mean_rank"
            else ["geo_nmse", "mean_rank"]
        )
        pick = sc.sort_values(order).index[0]
        row = d[d.variant == pick].iloc[0]
        out[name] = dict(
            model=name,
            variant=pick,
            presentation=row.presentation,
            r=int(row.r),
            lag=int(row.lag),
            mean_rank=float(sc.mean_rank[pick]),
            geo_nmse=float(sc.geo_nmse[pick]),
        )
        out[name] = _apply_override(env_id, name, out[name], sc)
    return out


def selection(env_ids, grid=None):
    """The configuration table sections 6-8 are drawn in, one row per model.

    The explicit form of a step the stages otherwise take silently: it reads
    ``grid.csv``, applies ``config.SELECT_RULE`` and then ``config.PLOT_VARIANTS``,
    and hands back something a notebook can display and a figure can filter on.
    """
    if isinstance(env_ids, str):
        env_ids = [env_ids]
    rows = []
    for env_id in env_ids:
        g = grid if grid is not None else load(env_id, "grid")
        g = g[g.env == env_id]
        for b in best_variants(g, env_id).values():  # each entry carries its model
            rows.append(dict(env=env_id, **b))
    cols = [
        "env",
        "model",
        "variant",
        "presentation",
        "r",
        "lag",
        "source",
        "mean_rank",
        "geo_nmse",
    ]
    return pd.DataFrame(rows, columns=cols)


# -------------------------------------------------- stage 4: multi-step rollout
def _state_rows(env_id, spec, model, variant, budget, arr, evalset):
    """``(window, horizon, channel)`` predictions as tidy rows for section 8.

    Deliberately lean: no presentation/lag/label columns, and values rounded --
    this table is one row per channel per step per window per variant, so its
    width is what decides whether keeping it is affordable. Everything dropped
    is recoverable from `variant` or from the env spec at draw time.
    """
    return [
        dict(
            env=env_id,
            model=model,
            variant=variant,
            budget=budget,
            window=w,
            h=h + 1,
            channel=c,
            value=round(float(arr[w, h, c]), 5),
            evalset=evalset,
        )
        for w in range(arr.shape[0])
        for h in range(arr.shape[1])
        for c in range(spec.n_obs)
    ]


def run_rollout(env_id, force=False):
    """Section 7: open loop to ``config.ROLL_H``, the true action sequence known
    throughout -- every variant of every model, at every budget.

    Normalisation is per horizon by the mean square of s_{t+h} - s_t, so
    persistence sits at 1 for every h: where a curve crosses that line the model
    has stopped being useful for planning.

    The whole presentation x r grid, not one configuration per model, because
    what a transform costs is a property of the *horizon*: differencing is an
    integration, so a variant that wins at h=1 can drift away by h=20, and a
    one-step winner extrapolated says nothing about that. Which variant then
    reaches the combined figure is a drawing decision (``config.PLOT_VARIANTS``)
    rather than a compute one.
    """
    spec, st = spec_of(env_id), study(env_id)
    pool, raw = st["pool"], st["ev_raw"]
    H = config.ROLL_H
    cap = st["ctx"] - H
    budgets = [n for n in config.ROLL_BUDGETS if n <= cap]
    ev = EvalSet(spec, raw["states"], raw["actions"], L=cap, H=H, n=config.ROLL_WINDOWS)
    plan = _grid_plan(env_id, budgets, h=H)
    ek = f"{len(ev)}w@{cap}h{H}"
    done, have = _existing(env_id, "rollout", ("variant", "budget"), force, ek)
    todo = [p for p in plan if (p["variant"], str(p["budget"])) not in have]
    # §8's raw states ride along on the same resume key, so the two files can
    # never disagree about which cells have been measured
    done_s, _ = _existing(env_id, "traj", ("variant", "budget"), force, ek)
    print(
        f"{env_id}: rollout H={H}, context to {cap}, {len(ev)} windows, "
        f"budgets {budgets} -- {len(plan)} cells, {len(todo)} to run"
    )
    print(
        f"   estimated {_estimate_minutes(todo, h=H):.0f} min of foundation-model "
        f"calls on CPU (levers: ROLL_WINDOWS, ROLL_BUDGETS, GRID_FULL, MODELS)"
    )

    keep = min(config.TRAJ_WINDOWS, len(ev))
    rows, states = [], []
    if config.TRAJ_BUDGETS:
        states += _state_rows(env_id, spec, "truth", "-", 0, ev.fs[:keep], ek)
    for n in budgets:
        block = [p for p in todo if p["budget"] == n]
        if not block:
            continue
        fit = take_transitions(pool, n)
        print(f"=== {env_id}  N={n}  ({len(block)} cells)")
        for cell in block:
            name, r = cell["model"], cell["r"]
            t0 = time.time()
            # A cell that cannot run is recorded as NaN rather than skipped, as
            # the grid does: skipping leaves no row, so every resume retries it
            # and a stage with an undefined cell never reports itself finished.
            curve = [np.nan] * H
            try:
                if models.is_tsfm(name):
                    m = models.build(name, spec, presentation=cell["presentation"], r=r)
                    cs, ca = ev.cs[:, -n:], ev.ca[:, -n:]
                else:
                    m = models.build(name, spec, lag=cell["lag"], fit=fit)
                    cs, ca = ev.cs, ev.ca
                pred = m.predict(cs, ca, ev.fa)
                curve = [nmse(pred[:, h], ev.fs[:, h], ev.scale_h[h]) for h in range(H)]
            except Exception:
                pass  # undefined at this budget
            rows += [
                dict(
                    env=env_id,
                    model=name,
                    variant=cell["variant"],
                    presentation=cell["presentation"],
                    r=r,
                    lag=cell["lag"],
                    budget=n,
                    h=h + 1,
                    nmse=v,
                    evalset=ek,
                )
                for h, v in enumerate(curve)
            ]
            # The states behind the curve, for the budgets §8 draws. Free:
            # `pred` is already here and is about to be reduced to an NMSE.
            if n in (config.TRAJ_BUDGETS or ()) and np.all(np.isfinite(curve)):
                states += _state_rows(
                    env_id, spec, name, cell["variant"], n, pred[:keep], ek
                )
            ok = np.all(np.isfinite(curve))
            print(
                f"   {cell['variant']:<30s} "
                + (
                    f"h=1 {curve[0]:9.4f}  h={H} {curve[-1]:9.4f}"
                    if ok
                    else f"h=1 {'n/a':>9s}  h={H} {'n/a':>9s}"
                )
                + f"   [{time.time() - t0:5.1f}s]",
                flush=True,
            )
        # checkpoint after every budget block, so a long run survives a stop
        _write(
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True), env_id, "rollout"
        )
        if states:
            _write(
                pd.concat([done_s, pd.DataFrame(states)], ignore_index=True),
                env_id,
                "traj",
            )
    out = _write(
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True), env_id, "rollout"
    )
    _write(pd.concat([done_s, pd.DataFrame(states)], ignore_index=True), env_id, "traj")
    return out


RUNNERS = {
    "probe": run_probe,
    "grid": run_grid,
    "rollout": run_rollout,
}


def run(env_id, stages=None, force=False):
    """Run stages for one environment, in dependency order."""
    out = {}
    for stage in stages or STAGES:
        t0 = time.time()
        out[stage] = RUNNERS[stage](env_id, force=force)
        print(
            f"-- {env_id} {stage}: {len(out[stage])} rows, "
            f"{time.time() - t0:.0f}s -> {stage_path(env_id, stage)}\n"
        )
    return out


# ------------------------------------------------------------------------- CLI
def cli(env_id=None, argv=None):
    """Entry point of ``<env>/<env>.py`` and of ``run_all.py``.

    With ``env_id`` given the environment is fixed; without it the CLI takes
    ``--envs`` and defaults to every environment in ``config.ENVS``.
    """
    p = argparse.ArgumentParser(
        description=f"dyna_standard stages for {env_id or 'every environment'}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if env_id is None:
        p.add_argument(
            "--envs",
            nargs="+",
            default=config.ENVS,
            choices=config.ENVS,
            metavar="ENV",
            help="environments to run, in order",
        )
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument(
        "--force",
        action="store_true",
        help="recompute instead of resuming from the CSV",
    )
    p.add_argument(
        "--results-dir",
        default=config.RESULTS_DIRNAME,
        help="subfolder the CSVs are written to",
    )
    p.add_argument(
        "--windows",
        type=int,
        default=None,
        help="evaluation windows per cell (the main cost lever)",
    )
    p.add_argument(
        "--probe-ctx",
        type=int,
        default=None,
        help="contexts the section 2b probe averages over",
    )
    p.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=None,
        help="override the grid budgets; scaling and rollout are "
        "filtered to what is <= the largest of them",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="restrict every stage to these models (names from config.MODELS)",
    )
    a = p.parse_args(argv)

    config.RESULTS_DIRNAME = a.results_dir
    if a.windows:
        config.HP_WINDOWS = config.ROLL_WINDOWS = config.PROBE_WINDOWS = a.windows
        config.PROBE_CTX = min(config.PROBE_CTX, a.windows)
    if a.probe_ctx:
        config.PROBE_CTX = a.probe_ctx
    if a.budgets:
        top = max(a.budgets)
        config.HP_BUDGETS = sorted(a.budgets)
        config.SCALE_BUDGETS = [n for n in config.SCALE_BUDGETS if n <= top]
        config.ROLL_BUDGETS = [n for n in config.ROLL_BUDGETS if n <= top]
    if a.models:
        unknown = set(a.models) - set(config.MODELS)
        if unknown:
            p.error(
                f"unknown models {sorted(unknown)}; available: {list(config.MODELS)}"
            )
        keep = [m for m in config.MODELS if m in a.models]
        config.PROBE_MODELS = [m for m in keep if m in config.PROBE_MODELS]
        config.GRID_MODELS = keep
        config.PLOT_MODELS = keep

    envs = [env_id] if env_id else list(a.envs)
    t0 = time.time()
    failed = {}
    for e in envs:
        try:
            run(e, a.stages, force=a.force)
        except Exception as exc:
            # One environment failing must not throw away the others: every stage
            # checkpoints as it goes, so the rest of the sweep is still worth
            # having and the failure is reported at the end rather than lost in
            # the scrollback.
            failed[e] = exc
            print(f"!! {e} failed: {type(exc).__name__}: {exc}", flush=True)
    print(
        f"== {len(envs) - len(failed)}/{len(envs)} environments done in "
        f"{(time.time() - t0) / 60:.0f} min"
    )
    for e, exc in failed.items():
        print(f"   {e}: {type(exc).__name__}: {exc}")
    if failed:
        raise SystemExit(1)
