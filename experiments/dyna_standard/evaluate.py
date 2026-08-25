"""Evaluation windows, the error metric, and the counterfactual action probe.

Raw MSE is not comparable across environments, so errors are normalised per
channel by the mean square of the one-step increment: predicting "no change"
scores exactly 1.0 everywhere.
"""

import config
import numpy as np
from envs import env_step_reference, make_windows


def nmse(pred, true, scale):
    err = np.asarray(pred, np.float64) - np.asarray(true, np.float64)
    return float(np.mean(np.mean(err**2, 0) / scale))


class EvalSet:
    """One fixed set of evaluation windows, shared by every model and budget."""

    def __init__(self, spec, states, actions, L=None, H=1, n=None, seed=None):
        self.spec = spec
        seed = config.SEED if seed is None else seed
        self.cs, self.ca, self.fa, self.fs = make_windows(
            states,
            actions,
            L=config.L if L is None else L,
            H=H,
            max_windows=n,
            seed=seed,
        )
        self.true = self.fs[:, 0]
        self.scale = np.maximum(((self.true - self.cs[:, -1]) ** 2).mean(0), 1e-12)
        self.scale_h = np.maximum(
            np.stack(
                [
                    ((self.fs[:, h] - self.cs[:, -1]) ** 2).mean(0)
                    for h in range(self.fs.shape[1])
                ]
            ),
            1e-12,
        )

    def __len__(self):
        return len(self.cs)

    def nmse_of(self, pred):
        return nmse(pred, self.true, self.scale)


def score(model, ev, n=None):
    """One-step NMSE with the context cut to its last ``n`` steps (``None``: all).

    The trained models ignore ``n`` -- they read their last ``lag`` rows only --
    which is why their budget reaches them through the fit pool instead.
    """
    cs, ca = (ev.cs, ev.ca) if n is None else (ev.cs[:, -n:], ev.ca[:, -n:])
    try:
        v = ev.nmse_of(model.predict(cs, ca, ev.fa)[:, 0])
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan


# --------------------------------------------------------------- action probe
def probe_indices(ev, n_ctx=None, seed=None):
    """The contexts the probe is averaged over -- the same ones for every model."""
    n_ctx = config.PROBE_CTX if n_ctx is None else n_ctx
    seed = config.SEED if seed is None else seed
    return np.random.default_rng(seed).choice(
        len(ev), min(n_ctx, len(ev)), replace=False
    )


def _centre(curves, probe):
    """Mean response across contexts, centred on the middle probe action, plus the
    slope of the response in the action."""
    mid = len(probe) // 2
    return (
        (curves - curves[:, mid : mid + 1]).mean(0),
        float(np.polyfit(probe, curves.T, 1)[0].mean()),
    )


def model_response(model, ev, idx):
    """Hold each context fixed, sweep the next action over the probe grid, and
    record how far the predicted next value moves on the channel the action
    drives. A flat curve means the action was ignored."""
    spec = ev.spec
    probe, ch = spec["probe_actions"], spec["probe_channel"]
    k = len(probe)
    rs = np.repeat(ev.cs[idx], k, 0)
    ra = np.repeat(ev.ca[idx], k, 0)
    rf = np.tile(probe.reshape(k, 1, 1), (len(idx), 1, 1)).astype(np.float32)
    pred = model.predict(rs, ra, rf).reshape(len(idx), k, spec.n_obs)[:, :, ch]
    return _centre(pred, probe)


def reference_response(ev, idx):
    """The same sweep on the real environment, stepped from the same states, so
    clipping and saturation are handled exactly rather than assumed away."""
    spec = ev.spec
    probe, ch = spec["probe_actions"], spec["probe_channel"]
    ref = env_step_reference(spec, ev.cs[idx, -1], probe)
    return _centre(ref[:, :, ch], probe)
