"""Acrobot-v1.

Everything specific to this environment lives here: how the simulator's state
becomes the model observation and back, which channels are integrals of the
dynamics, and the action grid the section 2b probe sweeps. ``envs.py`` discovers
this file, so adding an environment is a folder with one of these in it.
"""

import numpy as np


def to_obs(s):
    t1, t2 = s[..., 0], s[..., 1]
    return np.stack(
        [np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2), s[..., 2], s[..., 3]], -1
    )


ENV = dict(
    env_id="Acrobot-v1",
    short="Acrobot",
    labels=[
        r"$\cos\theta_1$",
        r"$\sin\theta_1$",
        r"$\cos\theta_2$",
        r"$\sin\theta_2$",
        r"$\dot\theta_1$",
        r"$\dot\theta_2$",
    ],
    to_obs=to_obs,
    to_state=lambda o: np.array(
        [np.arctan2(o[1], o[0]), np.arctan2(o[3], o[2]), o[4], o[5]], float
    ),
    difference=(4, 5),  # the two angular rates; the trig channels are bounded
    probe_channel=5,
    probe_actions=np.array([0.0, 1.0, 2.0], np.float32),
    to_env_action=lambda a: int(round(a)),
    action_label="torque (0/1/2)",
    action_space="Discrete(3)",
    default_r=2,
)
