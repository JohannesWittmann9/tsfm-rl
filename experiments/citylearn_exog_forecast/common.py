"""Shared pieces for the forecast experiment (forecast.ipynb,
forecast_chronos.py, forecast_train.py). Reward classes must live in an
importable module, CityLearn re-imports them."""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from stable_baselines3.common.callbacks import BaseCallback

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
MODELS_DIR = EXP_DIR / "models"
RESULTS_DIR = EXP_DIR / "results"

SOURCE_DIR = (
    Path.home() / ".cache/citylearn/v2.5.0/datasets/citylearn_challenge_2022_phase_1"
)
N_TIME_STEPS = 8760
EVAL_HOLDOUT_HOURS = 168  # last 7 days of every month are held out


def _month_blocks() -> list[tuple[int, int]]:
    """Month blocks in the data; too-short ones are merged into a neighbor."""
    month = pd.read_csv(SOURCE_DIR / "Building_1.csv")["month"].to_numpy()
    changes = np.flatnonzero(np.diff(month) != 0) + 1
    starts = np.concatenate([[0], changes]).tolist()
    ends = np.concatenate([changes - 1, [len(month) - 1]]).tolist()
    merged = []
    for start, end in zip(starts, ends):
        if merged and merged[-1][1] - merged[-1][0] + 1 < 2 * EVAL_HOLDOUT_HOURS:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) > 1 and merged[0][1] - merged[0][0] + 1 < 2 * EVAL_HOLDOUT_HOURS:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return [(int(s), int(e)) for s, e in merged]


MONTH_BLOCKS = _month_blocks()
TRAIN_BLOCKS = [(s, e - EVAL_HOLDOUT_HOURS) for s, e in MONTH_BLOCKS]
EVAL_BLOCKS = [(e - EVAL_HOLDOUT_HOURS + 1, e) for s, e in MONTH_BLOCKS]
TRAIN_STEPS_PER_PASS = sum(e - s + 1 for s, e in TRAIN_BLOCKS)

# the dataset's built-in (perfect) forecast observations, always off
DATASET_PREDICTED_COLS = [
    f"{s}_predicted_{k}"
    for s in (
        "outdoor_dry_bulb_temperature",
        "outdoor_relative_humidity",
        "diffuse_solar_irradiance",
        "direct_solar_irradiance",
        "electricity_pricing",
    )
    for k in (1, 2, 3)
]

TOTAL_TIMESTEPS = 131_040  # ~19 passes over the train blocks
GAMMA = 0.99

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#999999"]
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "lines.linewidth": 2.0,
        "font.size": 10,
        "legend.frameon": False,
    }
)


class CostReward(RewardFunction):
    """Eval reward: the negative electricity bill."""

    def calculate(self, observations):
        reward_list = [
            -o["electricity_pricing"] * max(o["net_electricity_consumption"], 0.0)
            for o in observations
        ]
        return [sum(reward_list)] if self.central_agent else reward_list


class DifferenceCostReward(RewardFunction):
    """Training reward: bill without battery minus bill with battery.

    Everything the agent cannot influence cancels out. Doing nothing is
    worth exactly 0, stored solar surplus pays off as positive reward at
    discharge time. Same optimum as CostReward. (At the reset step
    CityLearn reports net = 3x load, an init artifact; action-independent,
    so harmless.)"""

    def calculate(self, observations):
        reward_list = [
            o["electricity_pricing"]
            * (
                max(o["non_shiftable_load"] - o["solar_generation"], 0.0)
                - max(o["net_electricity_consumption"], 0.0)
            )
            for o in observations
        ]
        return [sum(reward_list)] if self.central_agent else reward_list


class ForecastWrapper(gym.ObservationWrapper):
    """Appends forecast features from a lookup table to the observation,
    indexed by the absolute data row (episode start + time step)."""

    def __init__(self, env, table: np.ndarray):
        super().__init__(env)
        self.table = table.astype(np.float32)
        n = table.shape[1]
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [env.observation_space.low, np.full(n, -np.inf, np.float32)]
            ),
            high=np.concatenate(
                [env.observation_space.high, np.full(n, np.inf, np.float32)]
            ),
            dtype=np.float32,
        )

    def observation(self, observation):
        base = self.env.unwrapped
        t = base.episode_tracker.episode_start_time_step + base.time_step
        return np.concatenate([observation, self.table[t]]).astype(np.float32)


class EpisodeLogger(BaseCallback):
    """Collects (timestep, episode_return) from the Monitor infos."""

    def __init__(self):
        super().__init__()
        self.records = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.records.append((self.num_timesteps, info["episode"]["r"]))
        return True


def normalize_obs(obs: np.ndarray, obs_rms, epsilon: float = 1e-8, clip: float = 10.0):
    """VecNormalize.normalize_obs, replicated for raw (non-vec) envs."""
    return np.clip(
        (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -clip, clip
    ).astype(np.float32)


def kpi_table(env: CityLearnEnv) -> pd.DataFrame:
    kpis = env.evaluate()
    return kpis.pivot(index="cost_function", columns="name", values="value").dropna(
        how="all"
    )


def headline_score(kpi_df: pd.DataFrame) -> float:
    """Electricity cost relative to no battery (<1 beats no battery)."""
    return float(kpi_df["District"].loc["cost_total"])
