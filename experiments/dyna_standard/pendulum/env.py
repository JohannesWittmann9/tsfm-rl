"""Pendulum-v1.

Everything specific to this environment lives here: how the simulator's state
becomes the model observation and back, which channels are integrals of the
dynamics, and the action grid the section 2b probe sweeps. ``envs.py`` discovers
this file, so adding an environment is a folder with one of these in it.
"""

import numpy as np


def to_obs(s):
    th, thd = s[..., 0], s[..., 1]
    return np.stack([np.cos(th), np.sin(th), thd], -1)


ENV = dict(
    env_id="Pendulum-v1",
    short="Pendulum",
    labels=[r"$\cos\theta$", r"$\sin\theta$", r"$\dot\theta$"],
    to_obs=to_obs,
    to_state=lambda o: np.array([np.arctan2(o[1], o[0]), o[2]], float),
    difference=(2,),  # theta_dot is the integrated channel
    probe_channel=2,
    probe_actions=np.array([-2.0, -1.0, 0.0, 1.0, 2.0], np.float32),
    to_env_action=lambda a: np.array([a], np.float32),
    action_label="torque",
    action_space="Box[-2, 2]",
    default_r=8,  # seed for the section 2b probe; section 5 sweeps it
)
