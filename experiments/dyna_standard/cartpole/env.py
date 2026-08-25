"""CartPole-v1.

Everything specific to this environment lives here: how the simulator's state
becomes the model observation and back, which channels are integrals of the
dynamics, and the action grid the section 2b probe sweeps. ``envs.py`` discovers
this file, so adding an environment is a folder with one of these in it.
"""

import numpy as np

# CartPole needs its own data-collection policy, and the reason is worth stating
# because it is the one asymmetry in this study. Under the uniform-random policy
# the other three environments use, the pole falls in ~25 steps; if termination is
# ignored so the episode can continue, the state runs away (theta past 1000 rad
# within a few thousand steps). Neither produces the long, stationary episodes the
# budget sweeps need.
#
# Data therefore comes from an epsilon-greedy stabiliser: fixed linear feedback on
# (x, x_dot, theta, theta_dot), overridden by a uniform random action with
# probability EPS. At eps=0.15 the pole survives 8224 steps on 39 of 40 seeds
# while ~15% of actions are pure noise. The cost is that CartPole's action is then
# partly a function of its state, so its numbers are not drawn from the same
# policy as the other three.
GAIN = np.array([0.6, 1.4, 14.0, 2.5])
EPS = 0.15


def policy(obs, rng):
    if rng.random() < EPS:
        return int(rng.integers(2))
    return int(float(GAIN @ np.asarray(obs, np.float64)) > 0)


ENV = dict(
    env_id="CartPole-v1",
    short="CartPole",
    labels=["$x$", r"$\dot x$", r"$\theta$", r"$\dot\theta$"],
    to_obs=lambda s: np.stack([s[..., 0], s[..., 1], s[..., 2], s[..., 3]], -1),
    to_state=lambda o: np.asarray(o, float),
    difference=(0, 1, 2, 3),
    probe_channel=1,
    probe_actions=np.array([0.0, 1.0], np.float32),
    to_env_action=lambda a: int(round(a)),
    action_label="force (0/1)",
    action_space="Discrete(2)",
    policy=policy,  # the one env that is not sampled uniformly at random
    default_r=4,
)
