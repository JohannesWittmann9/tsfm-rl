"""MountainCar-v0.

Everything specific to this environment lives here: how the simulator's state
becomes the model observation and back, which channels are integrals of the
dynamics, and the action grid the section 2b probe sweeps. ``envs.py`` discovers
this file, so adding an environment is a folder with one of these in it.
"""

import numpy as np

ENV = dict(
    env_id="MountainCar-v0",
    short="MountainCar",
    labels=["position", "velocity"],
    to_obs=lambda s: np.stack([s[..., 0], s[..., 1]], -1),
    to_state=lambda o: np.asarray(o, float),
    difference=(0, 1),  # position and velocity are both integrals here
    probe_channel=1,
    probe_actions=np.array([0.0, 1.0, 2.0], np.float32),
    to_env_action=lambda a: int(round(a)),
    action_label="thrust (0/1/2)",
    action_space="Discrete(3)",
    default_r=4,
)
