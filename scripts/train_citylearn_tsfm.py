import os
import argparse
import collections
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces

import wandb
from wandb.integration.sb3 import WandbCallback

from chronos import Chronos2Pipeline
from stable_baselines3 import PPO
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import StableBaselines3Wrapper


class CityLearnTSFMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        context_length=16,
        model_name="amazon/chronos-2",
        device="cuda",
        max_steps=720,
    ):
        super().__init__()
        self.context_length = context_length
        self.device = device
        self.max_steps = max_steps
        self.current_step = 0

        self._real_env = CityLearnEnv(
            "citylearn_challenge_2023_phase_2_local_evaluation", central_agent=True
        )

        self.observation_space = self._real_env.observation_space[0]
        self.action_space = self._real_env.action_space[0]
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]

        print(f"[CityLearnTSFMEnv] Loading TSFM World Model ({model_name}) on {device}...")
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=self.device,
        )

        self.obs_history = collections.deque(maxlen=self.context_length)
        self.action_history = collections.deque(maxlen=self.context_length)

        raw_targets = self._real_env.observation_names[0]
        raw_actions = self._real_env.action_names[0]

        flat_targets = [
            str(col)
            for sublist in (raw_targets if isinstance(raw_targets[0], list) else [raw_targets])
            for col in (sublist if isinstance(sublist, list) else [sublist])
        ]
        flat_actions = [
            str(col)
            for sublist in (raw_actions if isinstance(raw_actions[0], list) else [raw_actions])
            for col in (sublist if isinstance(sublist, list) else [sublist])
        ]

        self.target_column_names = [f"target_{i}_{name}" for i, name in enumerate(flat_targets)]
        self.action_column_names = [f"action_{i}_{name}" for i, name in enumerate(flat_actions)]
        self.all_feature_column_names = self.target_column_names + self.action_column_names

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.obs_history.clear()
        self.action_history.clear()

        real_obs_list, _ = self._real_env.reset(seed=seed)
        current_real_obs = np.array(real_obs_list[0], dtype=np.float32)

        for _ in range(self.context_length - 1):
            self.obs_history.append(current_real_obs.copy())
            action = self.action_space.sample()
            self.action_history.append(action.astype(np.float32))
            step_obs, _, _, _, _ = self._real_env.step([action])
            current_real_obs = np.array(step_obs[0], dtype=np.float32)

        self.obs_history.append(current_real_obs.copy())
        self.action_history.append(np.zeros(self.act_dim, dtype=np.float32))
        return current_real_obs, {}

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32)
        prev_obs = self.obs_history[-1].copy()

        deltas = [
            self.obs_history[i + 1] - self.obs_history[i]
            for i in range(self.context_length - 1)
        ]
        deltas.append(np.zeros(self.obs_dim, dtype=np.float32))

        context_data = [
            np.concatenate([deltas[i], self.action_history[i]])
            for i in range(self.context_length)
        ]
        context_df = pd.DataFrame(context_data, columns=self.all_feature_column_names)
        context_df["id"] = 0
        context_df["timestamp"] = pd.to_datetime(np.arange(self.context_length), unit="s")

        future_df = pd.DataFrame([action], columns=self.action_column_names)
        future_df["id"] = 0
        future_df["timestamp"] = pd.to_datetime([self.context_length], unit="s")

        with torch.no_grad():
            pred_df = self.pipeline.predict_df(
                context_df,
                future_df=future_df,
                prediction_length=1,
                id_column="id",
                timestamp_column="timestamp",
                target=self.target_column_names,
            )
            target_id_col = "target_name" if "target_name" in pred_df.columns else "target"
            value_col = 0.5 if 0.5 in pred_df.columns else "predictions"

            pivoted_df = pred_df.pivot(
                index="timestamp",
                columns=target_id_col,
                values=value_col,
            )
            raw_pred_delta = pivoted_df[self.target_column_names].values[0].astype(np.float32)

        next_obs_pred = prev_obs + raw_pred_delta

        self._real_env.next_time_step()
        terminated = False
        dataset_truncated = self._real_env.time_step >= (self._real_env.time_steps - 1)
        step_truncated = self.current_step >= self.max_steps
        truncated = bool(dataset_truncated or step_truncated)

        reward = self._compute_reward(next_obs_pred)
        self.obs_history.append(next_obs_pred.copy())
        self.action_history.append(action.copy())

        info = {}
        if truncated:
            info["kpis"] = self._real_env.evaluate()

        return next_obs_pred, reward, terminated, truncated, info

    def _compute_reward(self, predicted_obs):
        raw_targets = self._real_env.observation_names[0]
        obs_dict = {
            name: float(
                val.item()
                if hasattr(val, "item") and getattr(val, "size", 1) == 1
                else (val[0] if isinstance(val, (list, np.ndarray)) else val)
            )
            for name, val in zip(raw_targets, predicted_obs)
        }

        per_building_obs_dicts = []

        def _get_scalar_attr(obj, attr_name, default):
            val = getattr(obj, attr_name, default)
            if val is None:
                return default
            if isinstance(val, (int, float, bool, str)):
                return val
            if hasattr(val, "size"):
                return val.item() if val.size == 1 else float(val[-1])
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return float(val[-1])
            return default

        for bldg in self._real_env.buildings:
            raw_b_dict = bldg.observations()
            b_dict = {}
            for k, v in raw_b_dict.items():
                if hasattr(v, "item") and getattr(v, "size", 1) == 1:
                    b_dict[k] = float(v.item())
                elif isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    b_dict[k] = float(v[0])
                else:
                    b_dict[k] = float(v) if isinstance(v, (int, float, np.number)) else v

            for key in list(b_dict.keys()):
                if key in obs_dict:
                    b_dict[key] = float(obs_dict[key])

            required_reward_defaults = {
                "hvac_mode": int(_get_scalar_attr(bldg, "hvac_mode", 1)),
                "indoor_dry_bulb_temperature_cooling_set_point": float(
                    _get_scalar_attr(bldg, "indoor_dry_bulb_temperature_cooling_set_point", 22.0)
                ),
                "indoor_dry_bulb_temperature_heating_set_point": float(
                    _get_scalar_attr(bldg, "indoor_dry_bulb_temperature_heating_set_point", 18.0)
                ),
                "comfort_band": float(_get_scalar_attr(bldg, "comfort_band", 2.0)),
                "cooling_demand": float(_get_scalar_attr(bldg, "cooling_demand", 0.0)),
                "heating_demand": float(_get_scalar_attr(bldg, "heating_demand", 0.0)),
                "dhw_demand": float(_get_scalar_attr(bldg, "dhw_demand", 0.0)),
                "cooling_storage_soc": float(_get_scalar_attr(bldg, "cooling_storage_soc", 0.0)),
                "heating_storage_soc": float(_get_scalar_attr(bldg, "heating_storage_soc", 0.0)),
                "dhw_storage_soc": float(_get_scalar_attr(bldg, "dhw_storage_soc", 0.0)),
                "electrical_storage_soc": float(_get_scalar_attr(bldg, "electrical_storage_soc", 0.0)),
            }

            for key, default_val in required_reward_defaults.items():
                if key not in b_dict or b_dict[key] is None:
                    b_dict[key] = default_val

            if "net_electricity_consumption" not in b_dict:
                non_shift = b_dict.get("non_shiftable_load", 0.0)
                solar = b_dict.get("solar_generation", 0.0)
                cool = b_dict.get("cooling_demand", 0.0)
                heat = b_dict.get("heating_demand", 0.0)
                dhw = b_dict.get("dhw_demand", 0.0)
                b_dict["net_electricity_consumption"] = float(non_shift + cool + heat + dhw - solar)

            per_building_obs_dicts.append(b_dict)

        reward_list = self._real_env.reward_function.calculate(per_building_obs_dicts)
        return float(reward_list[0]) if isinstance(reward_list, list) else float(reward_list)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on CityLearn with Chronos-2 & W&B Tracking")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total PPO training timesteps")
    parser.add_argument("--context-length", type=int, default=16, help="TSFM context history window")
    parser.add_argument("--model-name", type=str, default="amazon/chronos-2", help="Hugging Face repo or path")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Steps for real env evaluation")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory for artifacts")

    # W&B Configuration Arguments
    parser.add_argument("--wandb-project", type=str, default="CityLearn-TSFM-RL", help="W&B project name")
    parser.add_argument("--wandb-group", type=str, default="chronos2-experiments", help="W&B group name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional specific run name")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable. Falling back to CPU.")
        device = "cpu"

    # Initialize Weights & Biases
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_run_name,
        config=vars(args),
        sync_tensorboard=True,  # Syncs SB3 internal metrics automatically
        monitor_gym=True,
        save_code=True,
    )

    # Build TSFM World Model Environment
    env_tsfm = CityLearnTSFMEnv(
        context_length=args.context_length,
        model_name=args.model_name,
        device=device,
    )

    # Train PPO Agent with WandbCallback
    ppo_model_tsfm = PPO("MlpPolicy", env_tsfm, device="cpu", verbose=1, tensorboard_log=f"runs/{run.id}")
    
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=os.path.join(args.output_dir, f"models/{run.id}"),
        verbose=2,
    )

    print(f"Starting PPO training for {args.timesteps} timesteps...")
    ppo_model_tsfm.learn(
        total_timesteps=args.timesteps,
        callback=wandb_callback,
        progress_bar=False,
    )
    print("\n--- PPO training on CityLearnTSFMEnv complete ---")

    # Save final model
    final_model_path = os.path.join(args.output_dir, "ppo_citylearn_tsfm_final.zip")
    ppo_model_tsfm.save(final_model_path)
    wandb.save(final_model_path)

    # Evaluate Agent in the Real Environment
    print("\n--- Evaluating PPO agent on real CityLearn environment ---")
    real_env = CityLearnEnv("citylearn_challenge_2023_phase_2_local_evaluation", central_agent=True)
    real_env = StableBaselines3Wrapper(real_env)

    observations_real, _ = real_env.reset()
    done = False
    step_count = 0

    while not done and step_count < args.eval_steps:
        actions_tsfm, _ = ppo_model_tsfm.predict(observations_real, deterministic=True)
        observations_real, _, terminated, truncated, _ = real_env.step(actions_tsfm)
        step_count += 1
        done = terminated or truncated

    print(f"Evaluation finished after {step_count} steps.")

    # Extract KPIs and Log as a W&B Table
    print("\n--- Real Environment KPIs ---")
    kpis_raw = real_env.unwrapped.evaluate()
    kpis_ppo = kpis_raw.pivot(index="cost_function", columns="name", values="value").round(3)
    kpis_ppo = kpis_ppo.dropna(how="all")
    print(kpis_ppo.to_string())

    # Format table for W&B
    kpis_reset = kpis_ppo.reset_index()
    wandb_kpi_table = wandb.Table(dataframe=kpis_reset)
    
    # Log the table and individual aggregate metrics to the W&B run dashboard
    wandb.log({"evaluation/kpis_table": wandb_kpi_table})

    # Log summary aggregate scalars if present
    if "District" in kpis_ppo.columns:
        for cost_fn, val in kpis_ppo["District"].items():
            wandb.summary[f"eval_district/{cost_fn}"] = val

    wandb.finish()


if __name__ == "__main__":
    main()