"""Dynamics models for Pendulum-v1 with a unified batched interface, plus the
few environment helpers the notebooks share (analytic reward, state/obs
conversion, random-action data collection).

All models predict future states s = (theta, theta_dot) given a context of past
states and actions plus a known future action sequence:

    predict(context_states (B, L, 2), context_actions (B, L, 1),
            future_actions (B, H, 1)) -> (B, H, 2)

Alignment convention: the action covariate at time t is the action that
*produced* the state at time t (applied one step earlier). So
context_actions[:, t] produced context_states[:, t] (index 0 may be zero
padding at an episode start), and future_actions[:, h] produces the h-th
predicted state. This makes the covariate contemporaneous with its effect,
which is what both Chronos-2 future covariates and VARX exog assume.
"""

import gymnasium as gym
import numpy as np
import torch
from chronos import BaseChronosPipeline
from statsmodels.tsa.api import VAR

MAX_TORQUE = 2.0
MAX_SPEED = 8.0
EPISODE_LEN = 200


def angle_normalize(theta: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi)."""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def reward_fn(
    theta: np.ndarray, theta_dot: np.ndarray, action: np.ndarray
) -> np.ndarray:
    """Pendulum-v1 reward, analytic in (theta, theta_dot, action)."""
    return -(
        angle_normalize(theta) ** 2
        + 0.1 * theta_dot**2
        + 0.001 * np.asarray(action) ** 2
    )


def state_to_obs(state: np.ndarray) -> np.ndarray:
    """(..., 2) (theta, theta_dot) -> (..., 3) (cos, sin, theta_dot)."""
    theta, theta_dot = state[..., 0], state[..., 1]
    return np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=-1).astype(
        np.float32
    )


def collect_random_dataset(
    n_episodes: int, seed: int = 0, episode_len: int = EPISODE_LEN
) -> dict:
    """Roll out Pendulum-v1 with uniform random actions.

    Returns states (n_episodes, T+1, 2) from ``env.unwrapped.state`` (theta is
    never wrapped there, so the series are smooth) and actions
    (n_episodes, T, 1). Deterministic given seed; T = episode_len (default 200).
    """
    rng = np.random.default_rng(seed)
    env = gym.make("Pendulum-v1", max_episode_steps=episode_len)
    states = np.zeros((n_episodes, episode_len + 1, 2), dtype=np.float32)
    actions = np.zeros((n_episodes, episode_len, 1), dtype=np.float32)
    for ep in range(n_episodes):
        env.reset(seed=int(rng.integers(2**31)))
        states[ep, 0] = env.unwrapped.state
        for t in range(episode_len):
            a = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=(1,)).astype(np.float32)
            env.step(a)
            actions[ep, t] = a
            states[ep, t + 1] = env.unwrapped.state
    env.close()
    return {"states": states, "actions": actions}


class ChronosDynamics:
    """Zero-shot Chronos-2 with the action as past + known-future covariate."""

    FALLBACK_MODEL_ID = "amazon/chronos-2"

    def __init__(
        self,
        model_id: str = "autogluon/chronos-2-small",
        device: str | None = None,
        batch_size: int = 256,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_id, device_map=device
            )
        except Exception:  # noqa: BLE001 - any load/download failure -> fallback model
            model_id = self.FALLBACK_MODEL_ID
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_id, device_map=device
            )
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._median_idx = self.pipeline.quantiles.index(0.5)
        self.name = f"Chronos-2 ({model_id.split('/')[-1]})"

    def predict(
        self,
        context_states: np.ndarray,
        context_actions: np.ndarray,
        future_actions: np.ndarray,
    ) -> np.ndarray:
        horizon = future_actions.shape[1]
        tasks = [
            {
                "target": np.ascontiguousarray(s.T, dtype=np.float32),
                "past_covariates": {
                    "action": np.ascontiguousarray(a[:, 0], dtype=np.float32)
                },
                "future_covariates": {
                    "action": np.ascontiguousarray(f[:, 0], dtype=np.float32)
                },
            }
            for s, a, f in zip(context_states, context_actions, future_actions)
        ]
        preds = []
        for start in range(0, len(tasks), self.batch_size):
            out = self.pipeline.predict(
                tasks[start : start + self.batch_size], prediction_length=horizon
            )
            # each element: (2, n_quantiles, H) -> median -> (H, 2)
            preds.extend(t[:, self._median_idx, :].T.numpy() for t in out)
        return np.stack(preds).astype(np.float32)


class VARDynamics:
    """VARX baseline: VAR on (theta, theta_dot) with the action as exogenous
    input, fit once on episodes concatenated along time (the ~n_episodes seam
    rows are an accepted bias)."""

    def __init__(self, states: np.ndarray, actions: np.ndarray, maxlags: int = 8):
        # exog row for s_t is a_{t-1}, the action that produced it
        endog = states[:, 1:].reshape(-1, 2).astype(np.float64)
        exog = actions.reshape(-1, 1).astype(np.float64)
        model = VAR(endog, exog=exog)
        self.lag = max(int(model.select_order(maxlags=maxlags).aic), 1)
        self.results = model.fit(self.lag)
        self.name = f"VARX(p={self.lag})"

    def predict(
        self,
        context_states: np.ndarray,
        context_actions: np.ndarray,
        future_actions: np.ndarray,
    ) -> np.ndarray:
        horizon = future_actions.shape[1]
        preds = np.empty((len(context_states), horizon, 2), dtype=np.float32)
        for i in range(len(context_states)):
            preds[i] = self.results.forecast(
                context_states[i, -self.lag :].astype(np.float64),
                steps=horizon,
                exog_future=future_actions[i].astype(np.float64),
            )
        return preds


class LinearDynamics:
    """One-step linear model s' = A s + B a + c, fit by least squares on all
    transitions; multi-step prediction iterates the one-step map."""

    def __init__(self, states: np.ndarray, actions: np.ndarray):
        s = states[:, :-1].reshape(-1, 2)
        s_next = states[:, 1:].reshape(-1, 2)
        a = actions.reshape(-1, 1)
        x = np.concatenate([s, a, np.ones((len(s), 1))], axis=1)
        w, *_ = np.linalg.lstsq(x, s_next, rcond=None)
        self.A, self.B, self.c = w[:2].T, w[2:3].T, w[3]
        self.name = "Linear"

    def predict(
        self,
        context_states: np.ndarray,
        context_actions: np.ndarray,
        future_actions: np.ndarray,
    ) -> np.ndarray:
        horizon = future_actions.shape[1]
        s = context_states[:, -1].astype(np.float64)
        preds = np.empty((len(s), horizon, 2), dtype=np.float32)
        for h in range(horizon):
            s = (
                s @ self.A.T
                + future_actions[:, h].astype(np.float64) @ self.B.T
                + self.c
            )
            preds[:, h] = s
        return preds
