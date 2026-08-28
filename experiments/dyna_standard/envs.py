"""Generic machinery for turning environments into training data.

The environments themselves are **not** defined here: each one lives in its own
folder as ``<env>/env.py``, imported by name just below. This module only knows
how to roll episodes out, cut windows, and step an environment for the probe's
ground truth.

An ``ENV`` entry supplies:

``labels``          one per observation channel
``to_obs``          simulator state -> model observation
``to_state``        the inverse, so the probe can replay counterfactual actions
``difference``      channels that are integrals and must be passed as increments
``probe_channel``   the channel the action acts on
``probe_actions``   the grid the section 2b probe sweeps
``to_env_action``   probe value -> what ``env.step`` expects
``policy``          optional; a data-collection policy (only CartPole needs one)
``default_r``       stretch factor for the probe, before section 5 sweeps it
"""

import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np

_HERE = Path(__file__).resolve().parent


def _load(folder):
    """Load ``<folder>/env.py``.

    By file path rather than ``from pendulum.env import ENV``, because the folder
    also holds ``pendulum.py``: run that script, or the notebook next to it, and
    the name ``pendulum`` resolves to the file instead of to the folder.
    """
    path = _HERE / folder / "env.py"
    spec = importlib.util.spec_from_file_location(f"dyna_env_{folder}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {**module.ENV, "dir": folder}


# The environments of the study, in the order every figure and table uses them.
# Adding one is a folder with an ``env.py`` in it plus a name in this list.
ENV_LIST = [_load(d) for d in ("pendulum", "mountaincar", "cartpole")] #"acrobot"

ENV_CONFIG = {c["env_id"]: c for c in ENV_LIST}
ENV_IDS = list(ENV_CONFIG)
ENV_DIRS = {c["env_id"]: c["dir"] for c in ENV_LIST}
SHORT = {c["env_id"]: c["short"] for c in ENV_LIST}
DEFAULT_R = {c["env_id"]: c["default_r"] for c in ENV_LIST}


class EnvSpec:
    """An environment plus its config, passed explicitly so several can coexist
    in one kernel."""

    def __init__(self, env_id, episode_len=200):
        self.env_id, self.episode_len = env_id, episode_len
        self.cfg = ENV_CONFIG[env_id]
        self.n_obs = len(self.cfg["labels"])

    def __getitem__(self, k):
        return self.cfg[k]

    def __repr__(self):
        return f"EnvSpec({self.env_id}, n_obs={self.n_obs})"

    @property
    def labels(self):
        return self.cfg["labels"]

    def to_obs(self, state):
        return self.cfg["to_obs"](np.asarray(state, np.float32)).astype(np.float32)


def collect_rollouts(spec, n_episodes, seed=0, episode_len=None):
    """Rollouts -> states (n, T+1, n_obs), actions (n, T, 1).

    Episodes that terminate early are dropped and replaced by the next seed, so
    no window ever straddles an episode boundary. The policy is uniform-random
    unless the env config supplies one (only CartPole does -- see CARTPOLE_GAIN).
    """
    T = episode_len or spec.episode_len
    policy = spec.cfg.get("policy")
    env = gym.make(spec.env_id, max_episode_steps=T)
    states = np.zeros((n_episodes, T + 1, spec.n_obs), np.float32)
    actions = np.zeros((n_episodes, T, 1), np.float32)
    bs, ba = np.zeros_like(states[0]), np.zeros_like(actions[0])
    ep = attempt = 0
    while ep < n_episodes and attempt < 20 * n_episodes:
        s = seed + attempt
        attempt += 1
        env.reset(seed=s)
        env.action_space.seed(s)
        rng = np.random.default_rng(s)
        bs[0] = spec.to_obs(env.unwrapped.state)
        full = True
        for t in range(T):
            a = policy(bs[t], rng) if policy else env.action_space.sample()
            _, _, terminated, _, _ = env.step(a)
            ba[t], bs[t + 1] = a, spec.to_obs(env.unwrapped.state)
            if terminated:
                full = False
                break
        if full:
            states[ep], actions[ep] = bs, ba
            ep += 1
    env.close()
    if ep < n_episodes:
        raise RuntimeError(f"{spec.env_id}: only {ep}/{n_episodes} full episodes")
    return {"states": states, "actions": actions}


def longest_rollouts(spec, want, n_episodes, seed, floor=64):
    """Longest episodes (<= want) this env can sustain, plus the data.

    ``collect_rollouts`` raises when too many episodes terminate early, so halve
    the request until it succeeds: the ceiling is a property of the env under its
    data-collection policy, measured rather than hand-set.
    """
    n = int(want)
    while n >= floor:
        try:
            return n, collect_rollouts(
                spec, n_episodes=n_episodes, seed=seed, episode_len=n
            )
        except RuntimeError:
            n //= 2
    raise RuntimeError(f"{spec.env_id}: cannot sustain {floor}-step episodes")


def make_windows(states, actions, L, H, max_windows=None, seed=0):
    """Cut (context, future) windows. Alignment: the action covariate at t is the
    action that *produced* s_t."""
    cs, ca, fa, fs = [], [], [], []
    for ep in range(len(actions)):
        for t0 in range(1, actions.shape[1] - L - H + 2):
            t1 = t0 + L
            cs.append(states[ep, t0:t1])
            ca.append(actions[ep, t0 - 1 : t1 - 1])
            fa.append(actions[ep, t1 - 1 : t1 - 1 + H])
            fs.append(states[ep, t1 : t1 + H])
    if max_windows and len(cs) > max_windows:
        idx = np.random.default_rng(seed).choice(len(cs), max_windows, replace=False)
        cs, ca, fa, fs = ([x[i] for i in idx] for x in (cs, ca, fa, fs))
    return [np.stack(x) for x in (cs, ca, fa, fs)]


def take_transitions(pool, n_steps):
    """Exactly ``n_steps`` transitions from a rollout pool, for the trained
    models. Below one episode this is a single truncated episode; above it, whole
    episodes (so the budget snaps to a multiple of the episode length)."""
    T = pool["actions"].shape[1]
    if n_steps <= T:
        return pool["states"][:1, : n_steps + 1], pool["actions"][:1, :n_steps]
    k = min(n_steps // T, len(pool["actions"]))
    return pool["states"][:k], pool["actions"][:k]


def env_step_reference(spec, last_obs, probe_actions):
    """Ground truth for the probe: set env.state to the end of each context and
    take one real step per probe action. (n_ctx, n_probe, n_obs)."""
    env = gym.make(spec.env_id).unwrapped
    env.reset(seed=0)
    out = np.empty((len(last_obs), len(probe_actions), spec.n_obs), np.float32)
    for i, obs in enumerate(last_obs):
        st = spec["to_state"](obs)
        for j, a in enumerate(probe_actions):
            env.state = st.copy()
            # CartPole refuses to step on from a state it once called terminal and
            # warns on every call. The probe states are non-terminal, so clearing
            # the flag is correct, not a workaround.
            if hasattr(env, "steps_beyond_terminated"):
                env.steps_beyond_terminated = None
            env.step(spec["to_env_action"](a))
            out[i, j] = spec.to_obs(env.state)
    env.close()
    return out
