"""Three forecasters, all with the signature
``predict(series, origins, h, budget) -> (n_windows, h) or None``.

``None`` means the budget is too small for that model to produce a forecast at
all (e.g. the MLP needs more training windows than the budget provides) --
returned rather than raising, so a sweep can leave the cell blank the way
section 3.1 does at its low-budget end.
"""

import numpy as np
import torch
from torch import nn

_CHRONOS_CACHE: dict[str, object] = {}


def seasonal_naive(series, origins, h, budget=None, period=24):
    """Repeats the value from ``period`` steps ago. Ignores ``budget`` -- it is
    not data-hungry, so it is drawn as one flat reference line across the sweep."""
    return np.stack([series[o + 1 - period : o + 1 - period + h] for o in origins])


def chronos_forecast(series, origins, h, budget, model_id, difference, batch_size=16):
    """Chronos-2 zero-shot, univariate, no covariates.

    ``difference``: the context is handed over as first differences and the
    forecast is re-integrated, the transform section 3.1 found to matter most
    for action sensitivity there. Needs ``budget >= 2``.
    """
    from chronos import BaseChronosPipeline  # type: ignore[import-untyped]

    if difference and budget < 2:
        return None
    if model_id not in _CHRONOS_CACHE:
        _CHRONOS_CACHE[model_id] = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
    pipe = _CHRONOS_CACHE[model_id]
    q = pipe.quantiles.index(0.5)

    tasks, lasts = [], []
    for o in origins:
        ctx = series[max(0, o - budget + 1) : o + 1].astype(np.float32)
        lasts.append(series[o])
        tasks.append(np.diff(ctx) if difference else ctx)

    out = pipe.predict(tasks, prediction_length=h, batch_size=batch_size)
    pred = np.stack([t[0, q, :].numpy() for t in out]).astype(np.float64)
    if difference:
        pred = np.asarray(lasts, np.float64)[:, None] + np.cumsum(pred, axis=1)
    return pred


class _MLP(nn.Module):
    def __init__(self, lag, h, hidden=(64, 32)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(lag, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h),
        )

    def forward(self, x):
        return self.net(x)


def mlp_forecast(
    series,
    split_idx,
    origins,
    h,
    budget,
    lag=24,
    seed=0,
    lr=1e-3,
    batch_size=64,
    max_epochs=200,
    patience=15,
    val_frac=0.2,
    min_windows=8,
):
    """Trained baseline: direct multi-step regression from the last ``lag``
    hours to the next ``h``, fit on the last ``budget`` hours of the training
    region (``series[:split_idx]``) -- ``budget`` is training-set size here,
    the same convention section 3.1 uses for its MLP/VARX baselines.
    """
    pool = series[max(0, split_idx - budget) : split_idx].astype(np.float32)
    n_win = len(pool) - lag - h + 1
    if n_win < min_windows:
        return None

    idx = np.arange(n_win)
    x = np.stack([pool[i : i + lag] for i in idx])
    y = np.stack([pool[i + lag : i + lag + h] for i in idx])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_win)
    n_val = max(1, round(val_frac * n_win))
    va, tr = perm[:n_val], perm[n_val:]
    xt, yt, xv, yv = x[tr], y[tr], x[va], y[va]

    xm, xs = xt.mean(0), np.where(xt.std(0) > 1e-8, xt.std(0), 1.0)
    ym, ys = yt.mean(0), np.where(yt.std(0) > 1e-8, yt.std(0), 1.0)

    def norm(a, m, s):
        return torch.as_tensor(((a - m) / s).astype(np.float32))

    xt_t, yt_t = norm(xt, xm, xs), norm(yt, ym, ys)
    xv_t, yv_t = norm(xv, xm, xs), norm(yv, ym, ys)

    torch.manual_seed(seed)
    net = _MLP(lag, h)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, state, bad = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(max_epochs):
        net.train()
        for i in range(0, len(xt_t), batch_size):
            b = torch.randperm(len(xt_t), generator=g)[i : i + batch_size]
            opt.zero_grad()
            loss_fn(net(xt_t[b]), yt_t[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(xv_t), yv_t))
        if v < best - 1e-9:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state:
        net.load_state_dict(state)
    net.eval()

    ctx = np.stack([series[o - lag + 1 : o + 1] for o in origins]).astype(np.float32)
    with torch.no_grad():
        pred_n = net(norm(ctx, xm, xs)).numpy()
    return pred_n * ys + ym
