"""Shared code for the CityLearn exogenous-forecast experiment.

Question: does giving a model-free SAC agent forecasts of the exogenous
series as extra observations improve battery control and how close do
zero-shot Chronos-2 forecasts get to perfect foresight?
"""

import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import StableBaselines3Wrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
MODELS_DIR = EXP_DIR / "models"
RESULTS_DIR = EXP_DIR / "results"
FIGS_DIR = EXP_DIR / "figs"

SOURCE_DIR = (
    Path.home() / ".cache/citylearn/v2.5.0/datasets/citylearn_challenge_2022_phase_1"
)
N_TIME_STEPS = 8760
EVAL_HOLDOUT_HOURS = 168  # last 7 days of every month are held out


def _month_blocks() -> list[tuple[int, int]]:
    """Consecutive same-month runs in the data; runs too short to split off a
    holdout week (the year starts mid-month) are merged into their neighbor."""
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
EVAL_ORIGINS = np.concatenate([np.arange(s, e + 1) for s, e in EVAL_BLOCKS])
TRAIN_STEPS_PER_PASS = sum(e - s + 1 for s, e in TRAIN_BLOCKS)

HORIZONS = {1: 6, 2: 12, 3: 24}
WEATHER_SERIES = [
    "outdoor_dry_bulb_temperature",
    "outdoor_relative_humidity",
    "diffuse_solar_irradiance",
    "direct_solar_irradiance",
]
PRICE_SERIES = ["electricity_pricing"]
BUILDINGS = [f"Building_{i}" for i in range(1, 6)]
BUILDING_SERIES = ["non_shiftable_load", "solar_generation"]
DEMAND_SERIES = [f"{b}:{s}" for b in BUILDINGS for s in BUILDING_SERIES]
GROUP_SERIES = {
    "price": PRICE_SERIES,
    "weather": WEATHER_SERIES,
    "demand": DEMAND_SERIES,
}
ALL_SERIES = WEATHER_SERIES + PRICE_SERIES + DEMAND_SERIES  # 15 series

# the dataset's built-in (perfect-foresight) forecast observations, always off
DATASET_PREDICTED_COLS = [
    f"{s}_predicted_{k}" for s in WEATHER_SERIES + PRICE_SERIES for k in HORIZONS
]

CHRONOS_MODEL_ID = "autogluon/chronos-2-small"
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 24
FORECAST_PARQUET = DATA_DIR / "chronos_forecasts.parquet"

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 131_040  # ~19 passes over the interleaved train blocks

ARM_SPECS = {
    "baseline": ((), None),
    "price": (("price",), "chronos"),
    "weather": (("weather",), "chronos"),
    "demand": (("demand",), "chronos"),
    "demand_oracle": (("demand",), "perfect"),
    "all": (("price", "weather", "demand"), "chronos"),
    "all_oracle": (("price", "weather", "demand"), "perfect"),
}
RL_ARMS = tuple(ARM_SPECS)


def predicted_cols(series: list[str]) -> list[str]:
    return [f"{s}_predicted_{k}" for s in series for k in HORIZONS]


def arm_columns(arm: str) -> list[str]:
    groups, _ = ARM_SPECS[arm]
    return [c for g in groups for c in predicted_cols(GROUP_SERIES[g])]


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


def savefig(name: str):
    FIGS_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGS_DIR / f"{name}.png")
    plt.close()


class CostReward(RewardFunction):
    def calculate(self, observations):
        reward_list = [
            -o["electricity_pricing"] * max(o["net_electricity_consumption"], 0.0)
            for o in observations
        ]
        return [sum(reward_list)] if self.central_agent else reward_list


def load_series() -> pd.DataFrame:
    """The 16 exogenous actual series, 8760 rows, columns in ALL_SERIES order."""
    parts = [
        pd.read_csv(SOURCE_DIR / "weather.csv")[WEATHER_SERIES],
        pd.read_csv(SOURCE_DIR / "pricing.csv")[PRICE_SERIES],
        pd.read_csv(SOURCE_DIR / "carbon_intensity.csv")[["carbon_intensity"]],
    ]
    for b in BUILDINGS:
        bdf = pd.read_csv(SOURCE_DIR / f"{b}.csv")[BUILDING_SERIES]
        parts.append(bdf.rename(columns={s: f"{b}:{s}" for s in BUILDING_SERIES}))
    return pd.concat(parts, axis=1)[ALL_SERIES]


def load_chronos(model_id: str = CHRONOS_MODEL_ID, device: str | None = None):
    from chronos import BaseChronosPipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return BaseChronosPipeline.from_pretrained(model_id, device_map=device)


def rolling_chronos_forecasts(
    series_df: pd.DataFrame,
    pipeline,
    context_length: int = CONTEXT_LENGTH,
    batch_size: int = 256,
    origins: np.ndarray | None = None,
    progress_every: int | None = 1000,
) -> pd.DataFrame:
    """Rolling-origin median forecasts for every series at horizons 6/12/24 h.

    One multivariate task per origin t with target shape (15, min(t+1, C)) —
    history up to and including t, never beyond (no leakage; early origins
    simply get short contexts). Returns a frame indexed by origin with 45
    """
    values = series_df[ALL_SERIES].to_numpy(dtype=np.float32).T  # (15, T)
    if origins is None:
        origins = np.arange(values.shape[1])
    origins = np.asarray(origins)
    median_idx = pipeline.quantiles.index(0.5)

    tasks = [
        {
            "target": np.ascontiguousarray(
                values[:, max(0, t + 1 - context_length) : t + 1]
            )
        }
        for t in origins
    ]
    rows = np.empty((len(tasks), len(ALL_SERIES), len(HORIZONS)), dtype=np.float32)
    horizon_idx = np.array(list(HORIZONS.values())) - 1  # steps 6/12/24 -> 5/11/23
    for start in range(0, len(tasks), batch_size):
        out = pipeline.predict(
            tasks[start : start + batch_size], prediction_length=PREDICTION_LENGTH
        )
        for i, t in enumerate(out):  # t: (16, n_quantiles, 24)
            rows[start + i] = t[:, median_idx, :].numpy()[:, horizon_idx]
        if (
            progress_every
            and (start // batch_size) % max(progress_every // batch_size, 1) == 0
        ):
            print(
                f"  {min(start + batch_size, len(tasks))}/{len(tasks)} origins",
                flush=True,
            )

    data = {
        f"{s}_predicted_{k}": rows[:, i, j]
        for i, s in enumerate(ALL_SERIES)
        for j, k in enumerate(HORIZONS)
    }
    return pd.DataFrame(data, index=origins)[predicted_cols(ALL_SERIES)]


def perfect_forecast_table(series_df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth forecasts: the realized value h hours ahead (wrapping at
    the year boundary, like the dataset's own predicted columns)."""
    data = {}
    for s in ALL_SERIES:
        actual = series_df[s].to_numpy()
        for k, h in HORIZONS.items():
            data[f"{s}_predicted_{k}"] = np.roll(actual, -h)
    return pd.DataFrame(data)[predicted_cols(ALL_SERIES)]


def naive_forecasts(series_df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Reference forecasters. seasonal: value 24 h before the target time
    (always in the past since h <= 24). persistence: the value at issue time."""
    assert kind in ("seasonal", "persistence")
    data = {}
    for s in ALL_SERIES:
        actual = series_df[s].to_numpy()
        for k, h in HORIZONS.items():
            if kind == "persistence":
                data[f"{s}_predicted_{k}"] = actual.copy()
            else:
                data[f"{s}_predicted_{k}"] = np.roll(actual, 24 - h)
    return pd.DataFrame(data)[predicted_cols(ALL_SERIES)]


def forecast_metrics(
    forecasts: pd.DataFrame, series_df: pd.DataFrame, skip_first: int = 48
) -> pd.DataFrame:
    """MAE/RMSE per series and horizon against the realized future values."""
    records = []
    for s in ALL_SERIES:
        actual = series_df[s].to_numpy()
        for k, h in HORIZONS.items():
            target = np.roll(actual, -h)
            idx = forecasts.index.to_numpy()
            valid = idx[(idx >= skip_first) & (idx < N_TIME_STEPS - h)]
            err = forecasts.loc[valid, f"{s}_predicted_{k}"].to_numpy() - target[valid]
            records.append(
                {
                    "series": s,
                    "horizon_h": h,
                    "mae": np.abs(err).mean(),
                    "rmse": np.sqrt((err**2).mean()),
                }
            )
    return pd.DataFrame(records)


def forecast_table(source: str) -> pd.DataFrame:
    if source == "perfect":
        return perfect_forecast_table(load_series())
    if not FORECAST_PARQUET.exists():
        raise FileNotFoundError(
            f"run run_forecasts.py first (writes {FORECAST_PARQUET})"
        )
    return pd.read_parquet(FORECAST_PARQUET)


def save_forecasts(forecasts_df: pd.DataFrame, path: Path = FORECAST_PARQUET):
    import chronos

    DATA_DIR.mkdir(exist_ok=True)
    forecasts_df.to_parquet(path)
    meta = {
        "model_id": CHRONOS_MODEL_ID,
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "quantile": 0.5,
        "chronos_version": chronos.__version__,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def make_env(split: str, seed: int = 0, eval_block: int | None = None) -> CityLearnEnv:
    """Raw CityLearn env, identical for every arm (built-in forecast
    observations off). split='train' covers the full year but episodes cycle
    through the ~3-week TRAIN_BLOCKS (the holdout weeks are never visited);
    split='eval' simulates exactly one held-out week (eval_block)."""
    assert split in ("train", "eval")
    if split == "train":
        window = {
            "simulation_start_time_step": 0,
            "simulation_end_time_step": N_TIME_STEPS - 1,
            "episode_time_steps": TRAIN_BLOCKS,
        }
    else:
        assert eval_block is not None, "split='eval' needs an eval_block index"
        start, end = EVAL_BLOCKS[eval_block]
        window = {"simulation_start_time_step": start, "simulation_end_time_step": end}
    return CityLearnEnv(
        schema=str(SOURCE_DIR / "schema.json"),
        root_directory=str(SOURCE_DIR),
        central_agent=True,
        reward_function=CostReward,
        inactive_observations=DATASET_PREDICTED_COLS,
        random_seed=seed,
        **window,
    )


class ForecastWrapper(gym.ObservationWrapper):
    """Appends forecast features from a lookup table, indexed by the absolute
    data row (episode start + env time step) so it aligns for any episode
    split or simulation window."""

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


def make_agent_env(arm: str, split: str, seed: int = 0, eval_block: int | None = None):
    """SB3-compatible env for an arm: flattened obs + the arm's forecast features."""
    env = StableBaselines3Wrapper(make_env(split, seed, eval_block))
    cols = arm_columns(arm)
    if cols:
        _, source = ARM_SPECS[arm]
        env = ForecastWrapper(env, forecast_table(source)[cols].to_numpy(np.float32))
    return env


def make_sb3_env(
    arm: str, seed: int = 0, training: bool = True, gamma: float = 0.99
) -> VecNormalize:
    venv = VecMonitor(DummyVecEnv([lambda: make_agent_env(arm, "train", seed)]))
    return VecNormalize(
        venv, training=training, norm_obs=True, norm_reward=training, gamma=gamma
    )


def unwrap_citylearn(env) -> CityLearnEnv:
    while hasattr(env, "venv"):
        env = env.venv
    if hasattr(env, "envs"):
        env = env.envs[0]
    return env.unwrapped


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


def train_sac(
    arm: str,
    seed: int,
    total_timesteps: int | None = None,
    verbose: int = 0,
) -> tuple[SAC, VecNormalize, pd.DataFrame]:
    """Train one SAC run and persist policy state_dict + VecNormalize + curve.

    Persisting state_dict instead of SAC.save avoids the torch>=2.6
    weights_only load failure with SB3 2.3.x.
    """
    total_timesteps = total_timesteps or TOTAL_TIMESTEPS
    env = make_sb3_env(arm, seed, training=True)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=total_timesteps,
        learning_starts=TRAIN_STEPS_PER_PASS,
        batch_size=256,
        seed=seed,
        device="cpu",
        verbose=verbose,
    )
    logger = EpisodeLogger()
    model.learn(total_timesteps=total_timesteps, callback=logger)

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"{arm}_s{seed}"
    torch.save(model.policy.state_dict(), MODELS_DIR / f"sac_{tag}.pt")
    env.save(str(MODELS_DIR / f"vecnorm_{tag}.pkl"))
    curve = pd.DataFrame(logger.records, columns=["timestep", "episode_return"])
    curve.insert(0, "arm", arm)
    curve.insert(1, "seed", seed)
    curve.to_csv(RESULTS_DIR / f"learning_curve_{tag}.csv", index=False)
    return model, env, curve


def load_sac(arm: str, seed: int) -> tuple[SAC, VecNormalize]:
    """Rebuild a trained run from persisted state_dict + VecNormalize stats."""
    tag = f"{arm}_s{seed}"
    venv = VecMonitor(DummyVecEnv([lambda: make_agent_env(arm, "train", seed)]))
    vecnorm = VecNormalize.load(str(MODELS_DIR / f"vecnorm_{tag}.pkl"), venv)
    vecnorm.training = False
    model = SAC("MlpPolicy", vecnorm, buffer_size=1, device="cpu", seed=seed)
    model.policy.load_state_dict(
        torch.load(MODELS_DIR / f"sac_{tag}.pt", weights_only=True)
    )
    return model, vecnorm


def normalize_obs(obs: np.ndarray, obs_rms, epsilon: float = 1e-8, clip: float = 10.0):
    """VecNormalize.normalize_obs, replicated for use on a raw (non-vec) env."""
    return np.clip(
        (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -clip, clip
    ).astype(np.float32)


def mean_kpis(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Average KPI pivots over the held-out weeks (each already normalized
    against its own no-storage baseline)."""
    return pd.concat(tables).groupby(level=0).mean()


def rollout_policy(
    model: SAC, obs_rms, arm: str, seed: int = 0
) -> tuple[pd.DataFrame, float]:
    """One deterministic episode per held-out week on raw envs (no auto-reset,
    so each finished env can be evaluated). Returns (mean KPI table, total
    return over all 12 weeks)."""
    tables, total_return = [], 0.0
    for block in range(len(EVAL_BLOCKS)):
        env = make_agent_env(arm, "eval", seed, eval_block=block)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(normalize_obs(obs, obs_rms), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            done = terminated or truncated
        tables.append(kpi_table(env.unwrapped))
    return mean_kpis(tables), total_return


def run_reference_agent(agent_cls, seed: int = 0) -> tuple[pd.DataFrame, float]:
    """Deterministic citylearn-native agent (RBC/baseline) on every held-out
    week. Returns (mean KPI table, total return over all 12 weeks)."""
    tables, total_return = [], 0.0
    for block in range(len(EVAL_BLOCKS)):
        env = make_env("eval", seed, eval_block=block)
        agent_cls(env).learn(episodes=1, deterministic=True)
        total_return += float(np.sum([np.sum(r) for r in env.rewards[1:]]))
        tables.append(kpi_table(env))
    return mean_kpis(tables), total_return


def kpi_table(env: CityLearnEnv) -> pd.DataFrame:
    kpis = env.evaluate()
    return kpis.pivot(index="cost_function", columns="name", values="value").dropna(
        how="all"
    )


GRID_KPIS = [
    "ramping_average",
    "daily_one_minus_load_factor_average",
    "daily_peak_average",
    "all_time_peak_average",
]


def headline_score(kpi_df: pd.DataFrame) -> float:
    """District electricity cost normalized against the no-battery baseline
    (<1 beats no storage)"""
    return float(kpi_df["District"].loc["cost_total"])
