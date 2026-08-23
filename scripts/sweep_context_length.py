import os
import time
import argparse
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import gymnasium as gym
from gymnasium import spaces

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
        schema="citylearn_challenge_2022_phase_1",
        context_length=16,
        pipeline=None,
        device="cuda",
        max_steps=720,
        randomize_start_offset=True,
    ):
        super().__init__()
        self.context_length = context_length
        self.device = device
        self.max_steps = max_steps
        self.randomize_start_offset = randomize_start_offset
        self.current_step = 0
        self.start_offset = 0

        raw_env = CityLearnEnv(schema, central_agent=True)
        self._real_env = NormalizedObservationWrapper(raw_env)

        self.observation_space = self._real_env.observation_space[0]
        self.action_space = self._real_env.action_space[0]
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]

        self.pipeline = pipeline

        self.obs_history = collections.deque(maxlen=self.context_length)
        self.action_history = collections.deque(maxlen=self.context_length)

        raw_targets = self._real_env.observation_names[0]
        raw_actions = self._real_env.unwrapped.action_names[0]

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

        # Apply randomized start offset
        total_time_steps = self._real_env.unwrapped.time_steps
        max_possible_t0 = max(0, total_time_steps - self.max_steps - self.context_length - 10)

        if self.randomize_start_offset and max_possible_t0 > 0:
            self.start_offset = int(np.random.randint(0, max_possible_t0))
            self._real_env.unwrapped.time_step = self.start_offset
        else:
            self.start_offset = 0

        action_sample = self.action_space.sample().astype(np.float32)
        step_obs, _, _, _, _ = self._real_env.step([action_sample])
        current_real_obs = np.array(step_obs[0], dtype=np.float32)

        for _ in range(self.context_length - 1):
            self.obs_history.append(current_real_obs.copy())
            action = self.action_space.sample().astype(np.float32)
            self.action_history.append(action.copy())
            step_obs, _, _, _, _ = self._real_env.step([action])
            current_real_obs = np.array(step_obs[0], dtype=np.float32)

        self.obs_history.append(current_real_obs.copy())
        self.action_history.append(np.zeros(self.act_dim, dtype=np.float32))
        return current_real_obs, {}

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32).flatten()
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
        context_df = pd.DataFrame(context_data, columns=self.all_feature_column_names, dtype=np.float32)
        context_df["id"] = 0
        context_df["timestamp"] = pd.to_datetime(np.arange(self.context_length), unit="s")

        future_df = pd.DataFrame([action], columns=self.action_column_names, dtype=np.float32)
        future_df["id"] = 0
        future_df["timestamp"] = pd.to_datetime([self.context_length], unit="s")

        with torch.no_grad():
            pred_df = self.pipeline.predict_df(
                df=context_df,
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

        self._real_env.unwrapped.next_time_step()
        terminated = False
        dataset_truncated = self._real_env.unwrapped.time_step >= (self._real_env.unwrapped.time_steps - 1)
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
                val.item() if hasattr(val, "item") and getattr(val, "size", 1) == 1
                else (val[0] if isinstance(val, (list, np.ndarray)) else val)
            )
            for name, val in zip(raw_targets, predicted_obs)
        }

        per_building_obs_dicts = []

        def _get_scalar(obj, attr, default):
            val = getattr(obj, attr, default)
            if val is None:
                return default
            if isinstance(val, (int, float, bool, str)):
                return val
            if hasattr(val, "size"):
                return val.item() if val.size == 1 else float(val[-1])
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return float(val[-1])
            return default

        for bldg in self._real_env.unwrapped.buildings:
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

            # Inject fallback keys required by multi-objective reward classes
            defaults = {
                "hvac_mode": int(_get_scalar(bldg, "hvac_mode", 1)),
                "indoor_dry_bulb_temperature_cooling_set_point": float(_get_scalar(bldg, "indoor_dry_bulb_temperature_cooling_set_point", 22.0)),
                "indoor_dry_bulb_temperature_heating_set_point": float(_get_scalar(bldg, "indoor_dry_bulb_temperature_heating_set_point", 18.0)),
                "comfort_band": float(_get_scalar(bldg, "comfort_band", 2.0)),
                "cooling_demand": float(_get_scalar(bldg, "cooling_demand", 0.0)),
                "heating_demand": float(_get_scalar(bldg, "heating_demand", 0.0)),
                "dhw_demand": float(_get_scalar(bldg, "dhw_demand", 0.0)),
                "cooling_storage_soc": float(_get_scalar(bldg, "cooling_storage_soc", 0.0)),
                "heating_storage_soc": float(_get_scalar(bldg, "heating_storage_soc", 0.0)),
                "dhw_storage_soc": float(_get_scalar(bldg, "dhw_storage_soc", 0.0)),
                "electrical_storage_soc": float(_get_scalar(bldg, "electrical_storage_soc", 0.0)),
            }
            for k, def_val in defaults.items():
                if k not in b_dict or b_dict[k] is None:
                    b_dict[k] = def_val

            if "net_electricity_consumption" not in b_dict:
                non_shift = b_dict.get("non_shiftable_load", 0.0)
                solar = b_dict.get("solar_generation", 0.0)
                cool = b_dict.get("cooling_demand", 0.0)
                heat = b_dict.get("heating_demand", 0.0)
                dhw = b_dict.get("dhw_demand", 0.0)
                b_dict["net_electricity_consumption"] = float(non_shift + cool + heat + dhw - solar)

            per_building_obs_dicts.append(b_dict)

        reward_list = self._real_env.unwrapped.reward_function.calculate(per_building_obs_dicts)
        return float(reward_list[0]) if isinstance(reward_list, list) else float(reward_list)


# Phase 1: Pure Dynamics Benchmark (Offline Sweep)
def collect_offline_dataset(schema: str, num_steps: int = 2000, seed: int = 42):
    print(f"\n[Offline Data] Collecting {num_steps} transitions from {schema} (Seed {seed})...")
    raw_env = CityLearnEnv(schema, central_agent=True)
    env = NormalizedObservationWrapper(raw_env)

    obs_list, _ = env.reset(seed=seed)
    curr_obs = np.array(obs_list[0], dtype=np.float32)

    obs_records = [curr_obs]
    act_records = []
    delta_records = []
    hour_records = []

    for step in range(num_steps):
        hour = int(env.unwrapped.time_step % 24)
        hour_records.append(hour)

        action = env.action_space[0].sample().astype(np.float32)
        nxt_obs_list, _, done, trunc, _ = env.step([action])
        nxt_obs = np.array(nxt_obs_list[0], dtype=np.float32)

        delta = nxt_obs - curr_obs
        act_records.append(action)
        delta_records.append(delta)
        obs_records.append(nxt_obs)

        curr_obs = nxt_obs
        if done or trunc:
            obs_list, _ = env.reset(seed=seed + step)
            curr_obs = np.array(obs_list[0], dtype=np.float32)

    # Convert to NumPy arrays
    dataset = {
        "observations": np.array(obs_records[:-1], dtype=np.float32),  # Shape: (N, obs_dim)
        "next_observations": np.array(obs_records[1:], dtype=np.float32),
        "actions": np.array(act_records, dtype=np.float32),            # Shape: (N, act_dim)
        "deltas": np.array(delta_records, dtype=np.float32),           # Shape: (N, obs_dim)
        "hours": np.array(hour_records, dtype=np.int32),               # Shape: (N,)
        "obs_names": env.observation_names[0],
        "act_names": env.unwrapped.action_names[0],
    }
    print(f"[Offline Data] Collected matrix: {dataset['observations'].shape[0]} steps, "
          f"Obs Dim: {dataset['observations'].shape[1]}, Act Dim: {dataset['actions'].shape[1]}")
    return dataset


def run_dynamics_benchmark(pipeline, dataset, context_lengths=[4, 8, 16, 24, 48, 72], eval_steps=500):
    print("\n============================================================")
    print("      PHASE 1: DYNAMICS FIDELITY & LATENCY BENCHMARK        ")
    print("============================================================")

    flat_targets = [str(col) for col in dataset["obs_names"]]
    flat_actions = [str(col) for col in dataset["act_names"]]
    target_cols = [f"target_{i}_{name}" for i, name in enumerate(flat_targets)]
    action_cols = [f"action_{i}_{name}" for i, name in enumerate(flat_actions)]
    all_cols = target_cols + action_cols

    # Identify indoor temperature and net load indices for diurnal error decomposition
    temp_indices = [i for i, name in enumerate(flat_targets) if "indoor_dry_bulb_temperature" in name]
    load_indices = [i for i, name in enumerate(flat_targets) if "net_electricity_consumption" in name or "load" in name]
    if len(temp_indices) == 0:
        temp_indices = [0]
    if len(load_indices) == 0:
        load_indices = [min(1, len(flat_targets) - 1)]

    dynamics_summary = []
    diurnal_records = []
    autoregressive_drift = {}

    max_k = max(context_lengths)
    N = min(eval_steps, dataset["observations"].shape[0] - max_k - 30)

    for K in context_lengths:
        print(f"\nEvaluating Dynamics Model with Context Length K={K}...")
        step_times = []
        one_step_errors = []
        hour_temp_errors = collections.defaultdict(list)
        hour_load_errors = collections.defaultdict(list)

        # 1-Step Prediction Loop & Latency Profiling
        for t in range(max_k, max_k + N):
            # Assemble historical context window of length K
            hist_deltas = dataset["deltas"][t - K : t]
            hist_actions = dataset["actions"][t - K : t]
            curr_action = dataset["actions"][t]
            true_delta = dataset["deltas"][t]
            hour = dataset["hours"][t]

            context_data = [np.concatenate([hist_deltas[i], hist_actions[i]]) for i in range(K)]
            context_df = pd.DataFrame(context_data, columns=all_cols, dtype=np.float32)
            context_df["id"] = 0
            context_df["timestamp"] = pd.to_datetime(np.arange(K), unit="s")

            future_df = pd.DataFrame([curr_action], columns=action_cols, dtype=np.float32)
            future_df["id"] = 0
            future_df["timestamp"] = pd.to_datetime([K], unit="s")

            t_start = time.perf_counter()
            with torch.no_grad():
                pred_df = pipeline.predict_df(
                    df=context_df,
                    future_df=future_df,
                    prediction_length=1,
                    id_column="id",
                    timestamp_column="timestamp",
                    target=target_cols,
                )
                t_col = "target_name" if "target_name" in pred_df.columns else "target"
                v_col = 0.5 if 0.5 in pred_df.columns else "predictions"
                pivoted = pred_df.pivot(index="timestamp", columns=t_col, values=v_col)
                pred_delta = pivoted[target_cols].values[0].astype(np.float32)

            t_elapsed = time.perf_counter() - t_start
            step_times.append(t_elapsed)

            mse = float(np.mean((true_delta - pred_delta) ** 2))
            one_step_errors.append(mse)

            # Record hourly errors for Diurnal Decomposition
            temp_err = float(np.mean(np.abs(true_delta[temp_indices] - pred_delta[temp_indices])))
            load_err = float(np.mean(np.abs(true_delta[load_indices] - pred_delta[load_indices])))
            hour_temp_errors[hour].append(temp_err)
            hour_load_errors[hour].append(load_err)

        mean_step_sec = float(np.mean(step_times))
        fps = 1.0 / mean_step_sec if mean_step_sec > 0 else 0.0
        mean_1step_mse = float(np.mean(one_step_errors))

        print(f"  -> K={K:2d} | 1-Step MSE: {mean_1step_mse:.6f} | Step Time: {mean_step_sec*1000:.2f} ms ({fps:.1f} FPS)")

        dynamics_summary.append({
            "Context Length (K)": K,
            "1-Step MSE": round(mean_1step_mse, 6),
            "Step Latency (s)": round(mean_step_sec, 4),
            "Throughput (FPS)": round(fps, 2),
        })

        # Save diurnal error profile
        for h in range(24):
            diurnal_records.append({
                "Context Length (K)": K,
                "Hour": h,
                "Indoor Temp MAE": np.mean(hour_temp_errors[h]) if len(hour_temp_errors[h]) > 0 else 0.0,
                "Net Load MAE": np.mean(hour_load_errors[h]) if len(hour_load_errors[h]) > 0 else 0.0,
            })

        # Multi-Step Autoregressive Rollout Drift (H = 1..24)
        print(f"  -> Evaluating 24-step autoregressive drift for K={K}...")
        drift_rollout_errors = collections.defaultdict(list)
        rollout_trials = 10
        horizon_H = 24

        for trial in range(rollout_trials):
            start_t = max_k + trial * 25
            sim_obs = dataset["observations"][start_t].copy()
            sim_obs_hist = collections.deque(maxlen=K)
            sim_act_hist = collections.deque(maxlen=K)

            for hist_i in range(start_t - K + 1, start_t):
                sim_obs_hist.append(dataset["observations"][hist_i].copy())
                sim_act_hist.append(dataset["actions"][hist_i].copy())

            for h in range(horizon_H):
                t_curr = start_t + h
                action = dataset["actions"][t_curr]
                true_obs = dataset["next_observations"][t_curr]

                deltas = [sim_obs_hist[i + 1] - sim_obs_hist[i] for i in range(len(sim_obs_hist) - 1)]
                deltas.append(np.zeros(len(target_cols), dtype=np.float32))

                c_data = [np.concatenate([deltas[i], sim_act_hist[i]]) for i in range(len(deltas))]
                c_df = pd.DataFrame(c_data, columns=all_cols, dtype=np.float32)
                c_df["id"] = 0
                c_df["timestamp"] = pd.to_datetime(np.arange(len(deltas)), unit="s")

                f_df = pd.DataFrame([action], columns=action_cols, dtype=np.float32)
                f_df["id"] = 0
                f_df["timestamp"] = pd.to_datetime([len(deltas)], unit="s")

                with torch.no_grad():
                    p_df = pipeline.predict_df(
                        df=c_df, future_df=f_df, prediction_length=1,
                        id_column="id", timestamp_column="timestamp", target=target_cols
                    )
                    t_col = "target_name" if "target_name" in p_df.columns else "target"
                    v_col = 0.5 if 0.5 in p_df.columns else "predictions"
                    piv = p_df.pivot(index="timestamp", columns=t_col, values=v_col)
                    d_pred = piv[target_cols].values[0].astype(np.float32)

                sim_obs = sim_obs + d_pred
                drift_mse = float(np.mean((true_obs - sim_obs) ** 2))
                drift_rollout_errors[h + 1].append(drift_mse)

                sim_obs_hist.append(sim_obs.copy())
                sim_act_hist.append(action.copy())

        autoregressive_drift[K] = [np.mean(drift_rollout_errors[h]) for h in range(1, horizon_H + 1)]

    dynamics_summary_df = pd.DataFrame(dynamics_summary)
    diurnal_df = pd.DataFrame(diurnal_records)
    return dynamics_summary_df, diurnal_df, autoregressive_drift


def run_rl_sweep(args, pipeline, sweep_k_list=[4, 8, 16, 24, 48, 72]):
    print("\n============================================================")
    print("      PHASE 2: DOWNSTREAM PPO TRAINING & ANNUAL DEPLOYMENT   ")
    print("============================================================")

    results_across_k = []

    for K in sweep_k_list:
        print(f"\n============================================================")
        print(f"      STARTING PPO DREAM TRAINING FOR CONTEXT LENGTH K={K}  ")
        print(f"============================================================")

        seed_kpis = collections.defaultdict(list)

        for seed in args.seeds:
            print(f"\n[Training] Context K={K} | Seed {seed}...")
            set_random_seed(seed)

            env_dream = CityLearnTSFMEnv(
                schema=args.train_schema,
                context_length=K,
                pipeline=pipeline,
                device=args.device,
                max_steps=720,
                randomize_start_offset=True,
            )

            ppo_agent = PPO(
                policy="MlpPolicy",
                env=env_dream,
                device="cpu",
                seed=seed,
                verbose=0,
                tensorboard_log=f"runs/sweep_k_{K}_seed_{seed}",
            )

            ppo_agent.learn(total_timesteps=args.timesteps, progress_bar=False)

            ckpt_path = os.path.join(args.output_dir, f"ppo_dream_k{K}_seed{seed}.zip")
            ppo_agent.save(ckpt_path)

            print(f"  [Evaluation] Deploying Seed {seed} on Real Simulator (8,760 Hours)...")
            eval_env = CityLearnEnv(args.eval_schema, central_agent=True)
            eval_env = NormalizedObservationWrapper(eval_env)
            eval_env = StableBaselines3Wrapper(eval_env)

            obs, _ = eval_env.reset(seed=seed)
            done = False
            steps = 0

            while not done:
                action, _ = ppo_agent.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = eval_env.step(action)
                steps += 1
                done = term or trunc

            # 4. Extract official CityLearn KPI table
            kpis_raw = eval_env.unwrapped.evaluate()
            if "cost_function" in kpis_raw.columns and "name" in kpis_raw.columns and "value" in kpis_raw.columns:
                pivoted_kpi = kpis_raw.pivot(index="cost_function", columns="name", values="value").astype(float)
            else:
                pivoted_kpi = kpis_raw.copy().astype(float)

            # Extract key District metrics
            district_col = "District" if "District" in pivoted_kpi.columns else pivoted_kpi.columns[-1]
            for metric_name in pivoted_kpi.index:
                seed_kpis[metric_name].append(float(pivoted_kpi.loc[metric_name, district_col]))

        # Aggregate across seeds (Mean and Std) for this context length K
        k_summary = {"Context Length (K)": K}
        for metric, vals in seed_kpis.items():
            k_summary[f"{metric}_mean"] = float(np.mean(vals))
            k_summary[f"{metric}_std"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        results_across_k.append(k_summary)

    rl_sweep_df = pd.DataFrame(results_across_k)
    return rl_sweep_df


def generate_experiment_plots(dynamics_df, diurnal_df, rl_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    saved_figures = {}

    # Plot A: Downstream Policy Return vs. K (with Multi-Seed Error Bands)
    if rl_df is not None and not rl_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        k_vals = rl_df["Context Length (K)"].values

        palette = {"cost_total": "#1f77b4", "carbon_emissions_total": "#2ca02c", "daily_peak_average": "#d62728"}
        labels = {"cost_total": "District Cost", "carbon_emissions_total": "Carbon Emissions", "daily_peak_average": "Daily Peak Demand"}

        for metric_key, color in palette.items():
            mean_col = f"{metric_key}_mean"
            std_col = f"{metric_key}_std"

            if mean_col in rl_df.columns:
                mu = rl_df[mean_col].values
                sigma = rl_df[std_col].values if std_col in rl_df.columns else np.zeros_like(mu)

                ax.plot(k_vals, mu, marker="o", color=color, linewidth=2.2, label=labels.get(metric_key, metric_key))
                ax.fill_between(k_vals, mu - sigma, mu + sigma, color=color, alpha=0.18)

        ax.set_title("Downstream Annual Policy Performance vs. Context Length $K$", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Historical Context Window Length $K$ (Hours)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized District KPI (Lower is Better)", fontsize=11, fontweight="bold")
        ax.set_xticks(k_vals)
        ax.legend(frameon=True, fontsize=10, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        plot_a_path = os.path.join(output_dir, "plot_A_return_vs_k.png")
        fig.savefig(plot_a_path, dpi=300)
        saved_figures["plot_A"] = fig
        plt.close(fig)
        print(f"[Visualization] Saved Plot A to {plot_a_path}")

    # Plot B: The Efficiency Pareto Frontier (Prediction MSE vs. Step Latency)
    if dynamics_df is not None and not dynamics_df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

        latency = dynamics_df["Step Latency (s)"].values
        mse = dynamics_df["1-Step MSE"].values
        k_vals = dynamics_df["Context Length (K)"].values

        ax.plot(latency, mse, linestyle="--", color="#4b0082", alpha=0.5, zorder=1)
        scatter = ax.scatter(latency, mse, c=k_vals, cmap="viridis", s=130, edgecolor="black", linewidth=1.2, zorder=2)

        # Annotate points with K values
        for i, txt in enumerate(k_vals):
            ax.annotate(
                f" $K={txt}$",
                (latency[i], mse[i]),
                textcoords="offset points",
                xytext=(8, 4),
                fontweight="bold",
                fontsize=10,
            )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Context Length $K$", fontweight="bold", fontsize=10)

        ax.set_title("Efficiency Pareto Frontier: Prediction Fidelity vs. Step Latency", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Wall-Clock Execution Time per Step (Seconds / Step)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Dynamics 1-Step MSE ($Delta s$)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        plot_b_path = os.path.join(output_dir, "plot_B_pareto_frontier.png")
        fig.savefig(plot_b_path, dpi=300)
        saved_figures["plot_B"] = fig
        plt.close(fig)
        print(f"[Visualization] Saved Plot B to {plot_b_path}")

    # Plot C: Diurnal Error Decomposition Across 24-Hour Solar Cycles
    if diurnal_df is not None and not diurnal_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300, sharex=True)
        representative_ks = [k for k in [4, 24, 48] if k in diurnal_df["Context Length (K)"].unique()]
        colors = {4: "#e41a1c", 24: "#377eb8", 48: "#4daf4a"}

        for k in representative_ks:
            sub = diurnal_df[diurnal_df["Context Length (K)"] == k].sort_values("Hour")
            ax1.plot(sub["Hour"], sub["Indoor Temp MAE"], marker="o", label=f"$K={k}$", color=colors.get(k, "gray"), linewidth=2)
            ax2.plot(sub["Hour"], sub["Net Load MAE"], marker="s", label=f"$K={k}$", color=colors.get(k, "gray"), linewidth=2)

        # Highlight solar peak window (10:00 - 15:00)
        for ax in (ax1, ax2):
            ax.axvspan(10, 15, color="#ffbb00", alpha=0.15, label="Peak Solar Window (10-15h)")
            ax.set_xlabel("Hour of Day (0 → 23)", fontsize=10, fontweight="bold")
            ax.set_xticks(np.arange(0, 24, 3))
            ax.grid(True, linestyle=":", alpha=0.6)

        ax1.set_title("Indoor Temperature MAE Across Diurnal Cycle", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Mean Absolute Error (°C)", fontsize=10, fontweight="bold")
        ax1.legend(frameon=True, fontsize=9)

        ax2.set_title("Net Load MAE Across Diurnal Cycle", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean Absolute Error (Normalized)", fontsize=10, fontweight="bold")
        ax2.legend(frameon=True, fontsize=9)

        plt.suptitle("Diurnal Error Decomposition Across 24-Hour Solar & Occupancy Cycles", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        plot_c_path = os.path.join(output_dir, "plot_C_diurnal_error.png")
        fig.savefig(plot_c_path, dpi=300)
        saved_figures["plot_C"] = fig
        plt.close(fig)
        print(f"[Visualization] Saved Plot C to {plot_c_path}")

    return saved_figures


def parse_args():
    parser = argparse.ArgumentParser(description="Context Length Hyperparameter Sweep on Chronos 2.0 World Model")
    parser.add_argument(
        "--context-lengths",
        type=lambda s: [int(item.strip()) for item in s.split(",") if item.strip()],
        default=[4, 8, 16, 24, 48, 72],
        help="Comma-separated candidate context lengths (e.g. 4,8,16,24,48,72)",
    )
    parser.add_argument(
        "--rl-context-lengths",
        type=lambda s: [int(item.strip()) for item in s.split(",") if item.strip()],
        default=[4, 16, 24, 48],
        help="Subset of context lengths to train PPO policies on (to save compute)",
    )
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps per PPO seed training")
    parser.add_argument("--offline-steps", type=int, default=2000, help="Offline transition steps for Phase 1")
    parser.add_argument("--model-name", type=str, default="amazon/chronos-2")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--train-schema", type=str, default="citylearn_challenge_2022_phase_1")
    parser.add_argument("--eval-schema", type=str, default="citylearn_challenge_2022_phase_1")
    parser.add_argument(
        "--seeds",
        type=lambda s: [int(item.strip()) for item in s.split(",") if item.strip()],
        default=[42, 123, 456],
        help="Random seeds for multi-seed RL evaluation",
    )
    parser.add_argument("--skip-rl", action="store_true", help="Run only Phase 1 Dynamics Benchmark")
    parser.add_argument("--output-dir", type=str, default="./results_context_sweep")

    # W&B Configuration
    parser.add_argument("--wandb-project", type=str, default="CityLearn-TSFM-RL")
    parser.add_argument("--wandb-group", type=str, default="context-length-sweep")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable. Falling back to CPU.")
        device = "cpu"

    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_run_name,
        config=vars(args),
    )

    print(f"\n[Chronos Setup] Loading TSFM Pipeline ({args.model_name}) on {device}...")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_name, device_map=device)

    dataset = collect_offline_dataset(schema=args.train_schema, num_steps=args.offline_steps)
    dynamics_df, diurnal_df, drift_dict = run_dynamics_benchmark(
        pipeline=pipeline,
        dataset=dataset,
        context_lengths=args.context_lengths,
        eval_steps=min(600, args.offline_steps - 100),
    )

    wandb.log({
        "dynamics/summary_table": wandb.Table(dataframe=dynamics_df),
        "dynamics/diurnal_error_table": wandb.Table(dataframe=diurnal_df),
    })

    rl_df = None
    if not args.skip_rl:
        rl_df = run_rl_sweep(
            args=args,
            pipeline=pipeline,
            sweep_k_list=args.rl_context_lengths,
        )
        wandb.log({"evaluation/rl_sweep_summary": wandb.Table(dataframe=rl_df)})

    plots = generate_experiment_plots(dynamics_df, diurnal_df, rl_df, args.output_dir)
    for plot_name, fig in plots.items():
        wandb.log({f"visualizations/{plot_name}": wandb.Image(fig)})

    print("\n============================================================")
    print("      CONTEXT LENGTH HYPERPARAMETER SWEEP COMPLETE           ")
    print("============================================================")
    if dynamics_df is not None:
        print("\n--- Dynamics Fidelity & Throughput Summary ---")
        print(dynamics_df.to_string(index=False))
    if rl_df is not None:
        print("\n--- Downstream Multi-Seed RL Transfer Summary ---")
        print(rl_df.to_string(index=False))

    wandb.finish()


if __name__ == "__main__":
    main()