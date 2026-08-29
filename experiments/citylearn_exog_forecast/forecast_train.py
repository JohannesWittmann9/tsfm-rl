"""Train the SAC policies for the single-building forecast experiment (forecast.ipynb).

    python forecast_train.py            # every missing run (9 arms x 3 seeds)
    python forecast_train.py demand 0   # one specific run
    THREADS=2 python forecast_train.py  # cap torch CPU threads

Artifacts per run: models/sac_b1_<arm>_s<seed>.pt, models/vecnorm_b1_<arm>_s<seed>.pkl,
results/learning_curve_b1_<arm>_s<seed>.csv. Finished runs are skipped, so the
script can simply be rerun after failures. Needs data/forecast_chronos_b1.csv
from forecast_chronos.py for the Chronos arms.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

if os.environ.get("THREADS"):
    torch.set_num_threads(int(os.environ["THREADS"]))

import common
import forecast_chronos as sf
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

PRICE_COLS = [f"price_h{h}" for h in sf.HORIZONS]
WEATHER_COLS = [
    f"{s}_h{h}"
    for s in ("temperature", "humidity", "diffuse", "direct")
    for h in sf.HORIZONS
]
DEMAND_POINT_COLS = [f"{s}_h{h}" for s in ("load", "solar") for h in (6, 12, 24)]
DEMAND_VOLUME_COLS = [f"{s}_sum{h}" for s in ("load", "solar") for h in (6, 12, 24)]
ALL_COLS = PRICE_COLS + WEATHER_COLS + DEMAND_POINT_COLS + DEMAND_VOLUME_COLS

ARMS = {
    "baseline": (None, []),
    "price": ("chronos", PRICE_COLS),
    "weather": ("chronos", WEATHER_COLS),
    "demand_points": ("chronos", DEMAND_POINT_COLS),
    "demand_volumes": ("chronos", DEMAND_VOLUME_COLS),
    "seasonal_naive_points": ("naive", DEMAND_POINT_COLS),
    "seasonal_naive_volumes": ("naive", DEMAND_VOLUME_COLS),
    "all": ("chronos", ALL_COLS),
    "all_oracle": ("oracle", ALL_COLS),
}
SEEDS = (0, 1, 2)

_sources: dict[str, pd.DataFrame] = {}


def sources() -> dict[str, pd.DataFrame]:
    """Forecast feature tables by source, built once on first use."""
    if not _sources:
        if not sf.FORECAST_CSV.exists():
            raise FileNotFoundError(
                f"run forecast_chronos.py first ({sf.FORECAST_CSV})"
            )
        series = sf.load_b1_series()
        _sources["chronos"] = pd.read_csv(sf.FORECAST_CSV, index_col=0)
        _sources["oracle"] = sf.build_table(sf.shifted_paths(series, "oracle"))
        _sources["naive"] = sf.build_table(sf.shifted_paths(series, "naive"))
    return _sources


def make_b1_env(split: str, seed: int = 0, eval_block: int | None = None):
    """Single-building CityLearn env. Training uses the difference reward,
    evaluation the plain cost reward (both live in common.py because
    CityLearn re-imports reward classes)."""
    from citylearn.citylearn import CityLearnEnv

    if split == "train":
        window = {
            "simulation_start_time_step": 0,
            "simulation_end_time_step": common.N_TIME_STEPS - 1,
            "episode_time_steps": common.TRAIN_BLOCKS,
        }
    else:
        s, e = common.EVAL_BLOCKS[eval_block]
        window = {"simulation_start_time_step": s, "simulation_end_time_step": e}
    return CityLearnEnv(
        schema=str(common.SOURCE_DIR / "schema.json"),
        root_directory=str(common.SOURCE_DIR),
        buildings=["Building_1"],
        central_agent=True,
        reward_function=common.ShapedDifferenceReward
        if split == "train"
        else common.CostReward,
        inactive_observations=common.DATASET_PREDICTED_COLS,
        random_seed=seed,
        **window,
    )


def make_arm_env(arm: str, split: str, seed: int = 0, eval_block: int | None = None):
    """SB3-compatible env with the arm's forecast features appended."""
    from citylearn.wrappers import StableBaselines3Wrapper

    env = StableBaselines3Wrapper(make_b1_env(split, seed, eval_block))
    source, cols = ARMS[arm]
    if cols:
        env = common.ForecastWrapper(env, sources()[source][cols].to_numpy(np.float32))
    return env


def train_one(
    arm: str,
    seed: int,
    total_timesteps: int = common.TOTAL_TIMESTEPS,
    tag: str | None = None,
):
    tag = tag or f"b1_{arm}_s{seed}"
    venv = VecNormalize(
        VecMonitor(DummyVecEnv([lambda: make_arm_env(arm, "train", seed)])),
        norm_obs=True,
        norm_reward=True,
        gamma=common.GAMMA,
    )
    model = SAC(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        gamma=common.GAMMA,
        buffer_size=total_timesteps,
        learning_starts=min(common.TRAIN_STEPS_PER_PASS, total_timesteps // 2),
        batch_size=256,
        seed=seed,
        device="cpu",
    )
    logger = common.EpisodeLogger()
    model.learn(total_timesteps=total_timesteps, callback=logger)
    common.MODELS_DIR.mkdir(exist_ok=True)
    common.RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(model.policy.state_dict(), common.MODELS_DIR / f"sac_{tag}.pt")
    venv.save(str(common.MODELS_DIR / f"vecnorm_{tag}.pkl"))
    curve = pd.DataFrame(logger.records, columns=["timestep", "episode_return"])
    curve.to_csv(common.RESULTS_DIR / f"learning_curve_{tag}.csv", index=False)


def load_one(arm: str, seed: int, tag: str | None = None):
    """Rebuild a trained run from state_dict + VecNormalize stats."""
    tag = tag or f"b1_{arm}_s{seed}"
    venv = VecMonitor(DummyVecEnv([lambda: make_arm_env(arm, "train", seed)]))
    vecnorm = VecNormalize.load(str(common.MODELS_DIR / f"vecnorm_{tag}.pkl"), venv)
    vecnorm.training = False
    model = SAC("MlpPolicy", vecnorm, buffer_size=1, device="cpu", seed=seed)
    model.policy.load_state_dict(
        torch.load(common.MODELS_DIR / f"sac_{tag}.pt", weights_only=True)
    )
    return model, vecnorm


def evaluate_one(arm: str, seed: int) -> float:
    """Deterministic rollout on each held-out week; mean cost ratio."""
    model, vecnorm = load_one(arm, seed)
    scores = []
    for block in range(len(common.EVAL_BLOCKS)):
        env = make_arm_env(arm, "eval", seed, eval_block=block)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(
                common.normalize_obs(obs, vecnorm.obs_rms), deterministic=True
            )
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        scores.append(common.headline_score(common.kpi_table(env.unwrapped)))
    return float(np.mean(scores))


def main():
    if len(sys.argv) == 3:
        runs = [(sys.argv[1], int(sys.argv[2]))]
    else:
        runs = [(arm, seed) for arm in ARMS for seed in SEEDS]
    for arm, seed in runs:
        tag = f"b1_{arm}_s{seed}"
        if (common.MODELS_DIR / f"sac_{tag}.pt").exists():
            print(f"{tag}: cached, skipping")
            continue
        t0 = time.time()
        train_one(arm, seed)
        print(f"{tag}: done in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
