"""The four experiment stages, each cached to one CSV per environment.

    probe     section 2b  -- the counterfactual action probe under four
                             presentations of the same trajectory
    grid      section 5   -- every model, every variant, every budget
    scaling   section 6   -- one configuration per model against the data budget
    rollout   section 7   -- open-loop rollout to H=20 at three budgets

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

STAGES = ["probe", "grid", "scaling", "rollout", "traj"]

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


def _drop_stale(done, have, best):
    """Forget cached rows whose variant is no longer the one the grid selects.

    Without this, re-running the grid with more variants would leave a curve in
    ``scaling.csv`` that no table any longer claims, and the figure would show a
    model in a configuration nothing selected.
    """
    if not len(done):
        return done, have
    current = {(m, b["variant"]) for m, b in best.items()}
    keep = [tuple(x) in current for x in done[["model", "variant"]].values]
    done = done[keep]
    return done, set(map(tuple, done[["model", "budget"]].astype(str).values))


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
def _grid_plan(env_id, budgets):
    plan = []
    for name in config.GRID_MODELS:
        for v in models.variants(name, env_id):
            for n in budgets:
                if models.is_tsfm(name) and not models.fits_context(name, n, v["r"]):
                    continue  # would overflow the window: skip, not fall back
                plan.append({**v, "env": env_id, "budget": n})
    return plan


def _estimate_minutes(plan):
    """Rough cost, so a multi-hour run is a decision rather than a surprise. The
    constant is measured: ~2.0 ms per context token per window for Chronos-2 S on
    CPU at batch 16, near-linear in tokens."""
    per_token = 2.02e-3 / 24
    total = 0.0
    for p in plan:
        cost = config.MODELS[p["model"]].get("cost", 0.0)
        tokens = (p["budget"] - 1) * max(p["r"], 1) + 1
        total += tokens * per_token * config.HP_WINDOWS * cost
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
    """
    d = grid[(grid.env == env_id) & (grid.model == model)]
    piv = d.pivot_table(index="variant", columns="budget", values="nmse")
    ranks = piv.rank(axis=0, method="min").fillna(len(piv) + 1)
    with np.errstate(divide="ignore"):
        geo = np.exp(np.log(piv.clip(lower=1e-12)).mean(axis=1, skipna=True))
    return pd.DataFrame(
        {"mean_rank": ranks.mean(1), "geo_nmse": geo, "budgets_ok": piv.notna().sum(1)}
    )


def best_variants(grid, env_id):
    """The configuration each model is carried into sections 6 and 7 in.

    Selected on error alone, by ``config.SELECT_RULE``. Worth knowing what that
    does not do: on a low-signal environment the lowest error can belong to a
    configuration that has stopped reading the action, which forecasts well and
    plans not at all.
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
    return out


# ---------------------------------------------- stage 3: the data-budget sweep
def run_scaling(env_id, force=False):
    """Section 6: N on a fine grid, each model in the configuration section 5
    selected, so a panel carries a handful of curves instead of forty.

    Stretching consumes the context, so above roughly 8192/r the preferred r no
    longer fits. Rather than let those curves disappear, r falls back to the
    largest factor that does; the figure marks where that first happens.
    """
    spec, st = spec_of(env_id), study(env_id)
    ev, pool = st["ev"], st["pool"]
    best = best_variants(load(env_id, "grid", compute=True), env_id)
    ek = f"{len(ev)}w@{st['ctx']}"
    done, have = _existing(env_id, "scaling", ("model", "budget"), force, ek)
    done, have = _drop_stale(done, have, best)
    print(f"{env_id}: scaling over {st['scale_budgets']}")

    rows = []
    for n in st["scale_budgets"]:
        fit = take_transitions(pool, n)
        out = []
        for name in config.PLOT_MODELS:
            if name not in best or (name, str(n)) in have:
                continue
            b = best[name]
            if models.is_tsfm(name):
                r = models.usable_r(name, n, b["r"])
                val = score(
                    models.build(name, spec, presentation=b["presentation"], r=r), ev, n
                )
            else:
                r = 0
                try:
                    val = score(models.build(name, spec, lag=b["lag"], fit=fit), ev)
                except Exception:
                    val = np.nan  # undefined at this budget: too few rows
            rows.append(
                dict(
                    env=env_id,
                    model=name,
                    variant=b["variant"],
                    budget=n,
                    r=r,
                    nmse=val,
                    evalset=ek,
                )
            )
            out.append(f"{name} " + (f"{val:.4f}" if np.isfinite(val) else "n/a"))
        if out:
            print(f"   N={n:>5d}  " + "  ".join(out), flush=True)
    return _write(
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True), env_id, "scaling"
    )


# -------------------------------------------------- stage 4: multi-step rollout
def run_rollout(env_id, force=False):
    """Section 7: open loop to ``config.ROLL_H``, the true action sequence known
    throughout, at three data budgets.

    Normalisation is per horizon by the mean square of s_{t+h} - s_t, so
    persistence sits at 1 for every h: where a curve crosses that line the model
    has stopped being useful for planning.
    """
    spec, st = spec_of(env_id), study(env_id)
    pool, raw = st["pool"], st["ev_raw"]
    H = config.ROLL_H
    cap = st["ctx"] - H
    budgets = [n for n in config.ROLL_BUDGETS if n <= cap]
    ev = EvalSet(spec, raw["states"], raw["actions"], L=cap, H=H, n=config.ROLL_WINDOWS)
    best = best_variants(load(env_id, "grid", compute=True), env_id)
    ek = f"{len(ev)}w@{cap}h{H}"
    done, have = _existing(env_id, "rollout", ("model", "budget"), force, ek)
    done, have = _drop_stale(done, have, best)
    print(
        f"{env_id}: rollout H={H}, context to {cap}, {len(ev)} windows, "
        f"budgets {budgets}"
    )

    rows = []
    for n in budgets:
        fit = take_transitions(pool, n)
        for name in config.PLOT_MODELS:
            if name not in best or (name, str(n)) in have:
                continue
            b = best[name]
            t0 = time.time()
            if models.is_tsfm(name):
                r = models.usable_r(name, n, b["r"])
                m = models.build(name, spec, presentation=b["presentation"], r=r)
                cs, ca = ev.cs[:, -n:], ev.ca[:, -n:]
            else:
                r = 0
                try:
                    m = models.build(name, spec, lag=b["lag"], fit=fit)
                except Exception:
                    continue  # undefined at this budget
                cs, ca = ev.cs, ev.ca
            try:
                p = m.predict(cs, ca, ev.fa)
            except Exception:
                continue
            curve = [nmse(p[:, h], ev.fs[:, h], ev.scale_h[h]) for h in range(H)]
            if not np.all(np.isfinite(curve)):
                continue
            rows += [
                dict(
                    env=env_id,
                    model=name,
                    variant=b["variant"],
                    budget=n,
                    r=r,
                    h=h + 1,
                    nmse=v,
                    evalset=ek,
                )
                for h, v in enumerate(curve)
            ]
            print(
                f"   N={n:>5d} r={r:<2d} {name:<16s} h=1 {curve[0]:.4f}  "
                f"h={H} {curve[-1]:.4f}   [{time.time() - t0:5.1f}s]",
                flush=True,
            )
    return _write(
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True), env_id, "rollout"
    )


# ------------------------------------------- stage 5: example rollout states
def run_traj(env_id, force=False):
    """The states behind section 7: ground truth and each model's prediction over
    the horizon, for a handful of windows at one budget."""
    spec, st = spec_of(env_id), study(env_id)
    H = config.ROLL_H
    cap = st["ctx"] - H
    # clamped, so a panel labelled N is a panel the environment can actually
    # deliver: Acrobot terminates early and its context stops short of 4096.
    n = min(config.TRAJ_BUDGET, cap)
    ev = EvalSet(
        spec,
        st["ev_raw"]["states"],
        st["ev_raw"]["actions"],
        L=cap,
        H=H,
        n=config.ROLL_WINDOWS,
    )
    keep = min(config.TRAJ_WINDOWS, len(ev))
    best = best_variants(load(env_id, "grid", compute=True), env_id)
    fit = take_transitions(st["pool"], n)
    ek = f"{keep}w@{cap}h{H}N{n}"
    done, have = _existing(env_id, "traj", ("model",), force, ek)
    print(f"{env_id}: trajectories at N={n}, {keep} windows, H={H}")

    rows = []
    if ("truth",) not in have:
        for w in range(keep):
            for h in range(H):
                for c, lab in enumerate(spec.labels):
                    rows.append(
                        dict(
                            env=env_id,
                            model="truth",
                            variant="-",
                            budget=n,
                            window=w,
                            h=h + 1,
                            channel=c,
                            label=lab,
                            value=float(ev.fs[w, h, c]),
                            evalset=ek,
                        )
                    )
    for name in config.PLOT_MODELS:
        if name not in best or (name,) in have:
            continue
        b = best[name]
        if models.is_tsfm(name):
            r = models.usable_r(name, n, b["r"])
            m = models.build(name, spec, presentation=b["presentation"], r=r)
            cs, ca = ev.cs[:keep, -n:], ev.ca[:keep, -n:]
        else:
            try:
                m = models.build(name, spec, lag=b["lag"], fit=fit)
            except Exception:
                continue
            cs, ca = ev.cs[:keep], ev.ca[:keep]
        try:
            p = m.predict(cs, ca, ev.fa[:keep])
        except Exception:
            continue
        for w in range(keep):
            for h in range(H):
                for c, lab in enumerate(spec.labels):
                    rows.append(
                        dict(
                            env=env_id,
                            model=name,
                            variant=b["variant"],
                            budget=n,
                            window=w,
                            h=h + 1,
                            channel=c,
                            label=lab,
                            value=float(p[w, h, c]),
                            evalset=ek,
                        )
                    )
        print(f"   {name:<20s} {b['variant']}", flush=True)
    return _write(
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True), env_id, "traj"
    )


RUNNERS = {
    "probe": run_probe,
    "grid": run_grid,
    "scaling": run_scaling,
    "rollout": run_rollout,
    "traj": run_traj,
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
