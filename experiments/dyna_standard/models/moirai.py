"""Moirai-1.1-R as a dynamics model, action as a known-future covariate."""

import numpy as np
import torch

from .wrappers import FROM_CFG, resolve_difference, to_increments

_CACHE = {}


class MoiraiDynamics:
    """Moirai-1.1-R with the action as a known-future covariate.

    Pinned to ``patch_size="auto"``: on the explicit-patch path uni2ts 1.1.1 leaves
    the future covariate block empty, so the action driving the predicted step
    never reaches the model. ``auto`` needs
    ``past_length = context_length + prediction_length``, hence
    ``context_length = L - H`` below -- which also means the context has to be
    longer than the horizon, so the smallest budgets have no Moirai number at all.

    A sampling forecaster: the prediction is a median over ``num_samples`` draws,
    seeded per call. Moirai 2.0 dropped covariate support and is unusable here.
    """

    def __init__(
        self,
        spec,
        model_id="Salesforce/moirai-1.1-R-small",
        patch_size="auto",
        num_samples=20,
        batch_size=16,
        difference=FROM_CFG,
        seed=0,
    ):
        self.spec, self.patch_size, self.num_samples = spec, patch_size, num_samples
        self.batch_size, self.seed = batch_size, seed
        self.model_id = model_id
        key = ("moirai-module", model_id)
        if key not in _CACHE:
            from uni2ts.model.moirai import MoiraiModule

            _CACHE[key] = MoiraiModule.from_pretrained(model_id)
        self.module = _CACHE[key]
        self.difference = resolve_difference(spec, difference)
        self.name = "Moirai" if patch_size == "auto" else f"Moirai (patch {patch_size})"

    def _forecaster(self, L, H):
        """One cached forecaster per shape: MoiraiForecast fixes context and
        horizon at construction."""
        ctx = L - H if self.patch_size == "auto" else L
        key = (
            "moirai",
            self.model_id,
            ctx,
            H,
            self.patch_size,
            self.num_samples,
            self.spec.n_obs,
        )
        if key not in _CACHE:
            from uni2ts.model.moirai import MoiraiForecast

            fc = MoiraiForecast(
                module=self.module,
                prediction_length=H,
                context_length=ctx,
                target_dim=self.spec.n_obs,
                feat_dynamic_real_dim=1,
                past_feat_dynamic_real_dim=0,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
            )
            fc.eval()
            _CACHE[key] = fc
        return _CACHE[key]

    def predict(self, context_states, context_actions, future_actions):
        H = future_actions.shape[1]
        cs = np.asarray(context_states, np.float32)
        ca = np.asarray(context_actions, np.float32)
        last = cs[:, -1]
        if self.difference:
            model_s = to_increments(cs, self.difference)
            model_a = ca[:, 1:]
        else:
            model_s, model_a = cs, ca

        L = model_s.shape[1]
        fc = self._forecaster(L, H)
        cov = np.concatenate([model_a[:, :, 0], future_actions[:, :, 0]], 1)[..., None]
        preds = np.empty((len(model_s), H, self.spec.n_obs), np.float32)
        for i in range(0, len(model_s), self.batch_size):
            past = torch.as_tensor(model_s[i : i + self.batch_size])
            c = torch.as_tensor(cov[i : i + self.batch_size], dtype=torch.float32)
            kw = dict(
                feat_dynamic_real=c,
                observed_feat_dynamic_real=torch.ones_like(c, dtype=torch.bool),
            )
            torch.manual_seed(self.seed)  # sampling forecaster -> pin it
            with torch.no_grad():
                o = fc(
                    past_target=past,
                    past_observed_target=torch.ones_like(past, dtype=torch.bool),
                    past_is_pad=torch.zeros(past.shape[:2], dtype=torch.bool),
                    **kw,
                )
            preds[i : i + self.batch_size] = o.median(dim=1).values.numpy()
        for c in self.difference:
            preds[:, :, c] = last[:, None, c] + np.cumsum(preds[:, :, c], axis=1)
        return preds
