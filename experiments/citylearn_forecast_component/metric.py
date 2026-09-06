"""The error metric, normalised per horizon step so that persistence -- "the
series stays where it was at the forecast origin" -- scores exactly 1.0 at
every step. Identical convention to the rollout metric in
``experiments/dyna_standard/evaluate.py``, so the two sections' NMSE numbers
read on the same scale.
"""

import numpy as np


def persistence_scale(series: np.ndarray, origins: np.ndarray, h: int) -> np.ndarray:
    """Per-horizon-step mean square of the persistence error, shape ``(h,)``."""
    true = np.stack([series[o + 1 : o + 1 + h] for o in origins])
    last = series[origins]
    err2 = ((true - last[:, None]) ** 2).mean(axis=0)
    return np.maximum(err2, 1e-12)


def true_values(series: np.ndarray, origins: np.ndarray, h: int) -> np.ndarray:
    """Shape ``(len(origins), h)``."""
    return np.stack([series[o + 1 : o + 1 + h] for o in origins])


def nmse(pred: np.ndarray, true: np.ndarray, scale_h: np.ndarray) -> float:
    """Mean over windows and horizon steps of the per-step normalised error.

    ``pred``/``true``: ``(n_windows, h)``. ``nan`` in ``pred`` (a model that
    could not run at this budget) propagates to ``nan``, which the caller
    reports as a blank cell rather than a score.
    """
    err2 = ((np.asarray(pred, np.float64) - true) ** 2).mean(axis=0)
    return float((err2 / scale_h).mean())
