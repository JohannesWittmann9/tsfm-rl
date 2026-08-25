"""Chronos-2 as a dynamics model, with the action as a known-future covariate."""

import numpy as np
import torch
from chronos import BaseChronosPipeline

from .wrappers import FROM_CFG, resolve_difference, to_increments

_CACHE = {}


class ChronosDynamics:
    """Chronos-2, action as past + known-future covariate.

    ``difference``: channels handed over as increments and re-integrated after.
    Chronos conditions covariates on the *level* of a series, so an integrated
    channel passed as a level is dominated by autoregressive continuation and the
    action gets shrunk away -- which is what the level/diff columns of section 2b
    show.
    """

    def __init__(
        self,
        spec=None,
        model_id="autogluon/chronos-2-small",
        device=None,
        batch_size=16,
        difference=FROM_CFG,
    ):
        self.spec, self.batch_size = spec, batch_size
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        key = (model_id, device)
        if key not in _CACHE:
            _CACHE[key] = BaseChronosPipeline.from_pretrained(
                model_id, device_map=device
            )
        self.pipeline = _CACHE[key]
        self._q = self.pipeline.quantiles.index(0.5)
        self.difference = resolve_difference(spec, difference)
        self.name = "Chronos-2"

    def predict(self, context_states, context_actions, future_actions):
        H = future_actions.shape[1]
        cs = np.asarray(context_states, np.float32)
        ca = np.asarray(context_actions, np.float32)
        last = cs[:, -1]
        if self.difference:
            model_s = to_increments(cs, self.difference)
            model_a = ca[:, 1:]  # delta s_t was produced by a_t
        else:
            model_s, model_a = cs, ca

        tasks = []
        for s, a, f in zip(model_s, model_a, future_actions):
            tasks.append(
                {
                    "target": np.ascontiguousarray(s.T, np.float32),
                    "past_covariates": {
                        "action": np.ascontiguousarray(a[:, 0], np.float32)
                    },
                    "future_covariates": {
                        "action": np.ascontiguousarray(f[:, 0], np.float32)
                    },
                }
            )
        out = self.pipeline.predict(
            tasks, prediction_length=H, batch_size=self.batch_size
        )
        preds = np.stack([t[:, self._q, :].T.numpy() for t in out]).astype(np.float32)
        for c in self.difference:
            preds[:, :, c] = last[:, None, c] + np.cumsum(preds[:, :, c], axis=1)
        return preds
