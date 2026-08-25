"""The least-squares baseline: VAR(p) with the action as exogenous input.

Ported from ``notebooks/pendulum/pendulum.py`` so the two studies run the same
baseline: statsmodels' ``VAR``, episodes concatenated along time, the action as
``exog``.
"""

import numpy as np
from statsmodels.tsa.api import VAR


class VARDynamics:
    """VAR on the observation channels with the action as exogenous input.

    ``endog`` starts at s_1 and ``exog`` at a_0, so the exog row for s_t is
    a_{t-1}: the action that produced it, this study's alignment.

    Episodes are concatenated, so ~one row per episode pairs the last state of one
    with the first of the next. Where the linear fit is nearly exact those rows
    dominate the residual -- on MountainCar 19 seam rows out of 4000 moved NMSE
    from 0.000061 to 0.008560. An accepted bias, as in ``notebooks/pendulum``.

    ``lag="aic"`` selects p over 1..maxlags with statsmodels' own AIC.
    """

    def __init__(self, spec, states, actions, lag=1, maxlags=8):
        self.spec = spec
        n = spec.n_obs
        # exog row for s_t is a_{t-1}, the action that produced it
        endog = np.asarray(states, np.float64)[:, 1:].reshape(-1, n)
        exog = np.asarray(actions, np.float64).reshape(-1, 1)
        model = VAR(endog, exog=exog)
        if lag == "aic":
            self.lag = max(int(model.select_order(maxlags=maxlags).aic), 1)
        else:
            self.lag = int(lag)
        self.results = model.fit(self.lag)
        self.name = f"VARX(p={self.lag})"

    def predict(self, cs, ca, fa):
        horizon = fa.shape[1]
        preds = np.empty((len(cs), horizon, self.spec.n_obs), np.float32)
        for i in range(len(cs)):
            preds[i] = self.results.forecast(
                np.asarray(cs[i, -self.lag :], np.float64),
                steps=horizon,
                exog_future=np.asarray(fa[i], np.float64),
            )
        return preds
