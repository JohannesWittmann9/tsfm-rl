import os
import argparse
import collections
import numpy as np
import pandas as pd
import torch
import gymnasium as gym

import wandb
from wandb.integration.sb3 import WandbCallback

from chronos import Chronos2Pipeline
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper


class CityLearnTSFMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        schema="citylearn_challenge_2023_phase_1",
        context_length=16,
        pipeline=None,
        device="cuda",
        max_steps=720,
    ):
        super().__init__()
        self.context_length = context_length
        self.device = device
        self.max_steps = max_steps
        self.current_step = 0

        self._real_env = CityLearnEnv(schema, central_agent=True)
        self.total_time_steps = self._real_env.time_steps

        self.observation_space = self._real_env.observation_space[0]
        self.action_space = self._real_env.action_space[0]
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]

        self.pipeline = pipeline

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

        # Clean standard reset of the base simulator
        real_obs_list, _ = self._real_env.reset(seed=seed)
        current_real_obs = np.array(real_obs_list[0], dtype=np.float32)

        # Seed the sliding context window with K-1 warm-up steps
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

        self._real_env.next_time_step() # only advances the clock counter
        terminated = False
        dataset_truncated = self._real_env.time_step >= (self._real_env.time_steps - 1)
        step_truncated = self.current_step >= self.max_steps
        truncated = bool(dataset_truncated or step_truncated)

        reward = self._compute_reward(next_obs_pred)
        self.obs_history.append(next_obs_pred.copy())
        self.action_history.append(action.copy())

        info = {}
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
    parser = argparse.ArgumentParser(description="End-to-End Multi-Seed Training on TSFM Dream & Annual Deployment")
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps per seed")
    parser.add_argument("--context-length", type=int, default=16, help="TSFM context history window")
    parser.add_argument("--model-name", type=str, default="amazon/chronos-2")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--train-schema", type=str, default="citylearn_challenge_2023_phase_1")
    parser.add_argument("--eval-schema", type=str, default="citylearn_challenge_2023_phase_1")
    parser.add_argument(
        "--seeds",
        type=lambda s: [int(item.strip()) for item in s.split(",") if item.strip()],
        default=[42, 123, 456, 789, 2026],
        help="Comma-separated training/eval seeds (e.g., 42,123,456)",
    )
    parser.add_argument("--output-dir", type=str, default="./results")

    # W&B Arguments
    parser.add_argument("--wandb-project", type=str, default="CityLearn-TSFM-RL")
    parser.add_argument("--wandb-group", type=str, default="e2e-multiseed-annual")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()

def run_multiseed_training(args, device, group_name):
    # Preload the shared Chronos pipeline onto GPU once
    print(f"\n[Chronos Setup] Pre-loading TSFM Pipeline ({args.model_name}) on {device}...")
    pipeline = Chronos2Pipeline.from_pretrained(
        args.model_name,
        device_map=device,
    )

    seed_tables_dict = {}

    for seed in args.seeds:
        print(f"\n============================================================")
        print(f"       STARTING END-TO-END TRAINING RUN: SEED {seed}       ")
        print(f"============================================================")

        run_name = f"seed_{seed}"
        if args.wandb_run_name:
            run_name = f"{args.wandb_run_name}_seed_{seed}"

        run = wandb.init(
            project=args.wandb_project,
            group=group_name,
            name=run_name,
            config=dict(vars(args), current_seed=seed),
            sync_tensorboard=True,
            reinit=True,  # Allows re-initializing in a loop
        )

        set_random_seed(seed)

        # Instantiate fresh TSFM World Model training environment
        env_tsfm = CityLearnTSFMEnv(
            schema=args.train_schema,
            context_length=args.context_length,
            pipeline=pipeline,
            device=device,
            max_steps=720,
        )

        # Instantiate and train independent PPO agent
        tb_log_dir = f"runs/{run.id}_seed_{seed}"
        ppo_model = PPO(
            policy="MlpPolicy",
            env=env_tsfm,
            device="cpu",
            seed=seed,
            verbose=1,
            tensorboard_log=tb_log_dir,
        )

        wandb_callback = WandbCallback(
            gradient_save_freq=0, 
            verbose=2,
        )

        print(f"Training PPO policy from scratch for {args.timesteps} steps...")
        ppo_model.learn(total_timesteps=args.timesteps, callback=wandb_callback,)

        checkpoint_path = os.path.join(args.output_dir, f"ppo_tsfm_seed_{seed}.zip")
        ppo_model.save(checkpoint_path)

        print(f"\nDeploying Seed {seed} Policy on Full-Horizon Simulator...")
        eval_env = CityLearnEnv(args.eval_schema, central_agent=True)
        # eval_env = NormalizedObservationWrapper(eval_env)
        eval_env = StableBaselines3Wrapper(eval_env)

        observations, _ = eval_env.reset(seed=seed)
        total_horizon = eval_env.unwrapped.time_steps
        print(f"  Simulating Horizon: {total_horizon} hours")

        done = False
        step_count = 0

        while not done:
            action, _ = ppo_model.predict(observations, deterministic=True)
            observations, _, terminated, truncated, _ = eval_env.step(action)
            step_count += 1
            done = terminated or truncated

        print(f"  Seed {seed} deployment completed in {step_count} steps.")

        # Extract Full Evaluation KPI Table
        raw_kpis = eval_env.unwrapped.evaluate()
        if "cost_function" in raw_kpis.columns and "name" in raw_kpis.columns and "value" in raw_kpis.columns:
            pivoted_kpi = raw_kpis.pivot(index="cost_function", columns="name", values="value").astype(float)
        else:
            pivoted_kpi = raw_kpis.copy().astype(float)

        seed_tables_dict[seed] = pivoted_kpi

        wandb.log({f"eval_tables/seed_{seed}_kpis": wandb.Table(dataframe=pivoted_kpi.reset_index())})
        
        if "District" in pivoted_kpi.columns:
            for cost_fn, val in pivoted_kpi["District"].items():
                wandb.summary[f"eval_district/{cost_fn}"] = val
        run.finish()

    # Aggregate Metrics Across Seeds
    all_seed_series = [df.stack(dropna=False) for df in seed_tables_dict.values()]
    stacked_matrix = pd.concat(all_seed_series, axis=1)

    mu_series = stacked_matrix.mean(axis=1)
    sigma_series = stacked_matrix.std(axis=1, ddof=1).fillna(0.0)

    summary_df = pd.DataFrame({
        "Mean (μ)": mu_series.round(4),
        "Std (σ)": sigma_series.round(4),
        "μ ± σ": [f"{m:.4f} ± {s:.4f}" for m, s in zip(mu_series, sigma_series)]
    }).reset_index()

    first_col = summary_df.columns[0]
    second_col = summary_df.columns[1]
    summary_df.rename(columns={first_col: "cost_function", second_col: "entity"}, inplace=True)

    return seed_tables_dict, summary_df

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable. Falling back to CPU.")
        device = "cpu"

    group_name = args.wandb_group or f"e2e-multiseed-{os.getpid()}"

    seed_tables, summary_df = run_multiseed_training(args, device, group_name)

    print("\n============================================================")
    print("      END-TO-END MULTI-SEED ANNUAL AGGREGATE RESULTS (μ ± σ)")
    print("============================================================")
    print(summary_df.to_string(index=False))
    print("============================================================")

    # Dedicated Summary Run for the entire group
    summary_run_name = (
        f"{args.wandb_run_name}_AGGREGATE"
        if args.wandb_run_name
        else "multiseed_aggregate_summary"
    )

    with wandb.init(
        project=args.wandb_project,
        group=group_name,
        name=summary_run_name,
        job_type="evaluation_summary",
        config=vars(args),
    ) as agg_run:
        
        agg_run.log({"multiseed/aggregate_summary_table": wandb.Table(dataframe=summary_df)})

        for _, row in summary_df.iterrows():
            entity = str(row.get("entity", ""))
            cost_fn = str(row.get("cost_function", ""))
            if entity.lower() == "district":
                agg_run.summary[f"district_mean/{cost_fn}"] = row["Mean (μ)"]
                agg_run.summary[f"district_std/{cost_fn}"] = row["Std (σ)"]

    print("Aggregate metrics successfully uploaded to Weights & Biases.")


if __name__ == "__main__":
    main()