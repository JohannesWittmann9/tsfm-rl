#!/usr/bin/env python3
"""
TSFM Causal Structure, Feature Attribution, LOCO Ablation & Action Sweep Suite for CityLearn

Evaluates:
  1. Leave-One-Channel-Out (LOCO) Dynamics Fidelity (nMAE, MAE, RMSE, CRPS per dimension)
  2. Subsystem-Isolated Action Sweeps (Battery, HVAC/Cooling, Thermal Storage)
  3. Step-Synchronized Real CityLearn Counterfactual Action Response Reference
  4. Decoupled Active Action Contribution vs. Exogenous Environmental Drift
  5. Cross-Building Spatial Attention Leakage & Multi-Agent Isolation Audit vs. Real Baseline
  6. Downstream PPO Policy Retraining & Sim-to-Real Exploitation Diagnostics
  7. Multi-Panel Diagnostic Visualizations & W&B Artifact Upload
"""

import os
import argparse
import collections
# [Removed 4.1] Removed `import copy` — CityLearn environments contain PyTorch tensors that forbid deepcopy
import numpy as np
import pandas as pd
import torch
import gymnasium as gym

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless cluster execution
import matplotlib.pyplot as plt
import seaborn as sns

import wandb
from wandb.integration.sb3 import WandbCallback

from chronos import Chronos2Pipeline
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import StableBaselines3Wrapper


# =============================================================================
# 1. Environment: CityLearn TSFM World Model with Channel Ablation Support
# =============================================================================
class CityLearnTSFMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        schema: str = "citylearn_challenge_2023_phase_1",
        context_length: int = 16,
        pipeline=None,
        device: str = "cuda",
        max_steps: int = 720,
        ablation_channel: str = None,
        marginal_means: dict = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.device = device
        self.max_steps = max_steps
        self.current_step = 0
        self.ablation_channel = ablation_channel
        self.marginal_means = marginal_means or {}

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

    def _apply_channel_ablation(self, context_df: pd.DataFrame, future_df: pd.DataFrame):
        if not self.ablation_channel or str(self.ablation_channel).strip().lower() in ["none", ""]:
            return context_df, future_df

        ab_lower = str(self.ablation_channel).strip().lower()
        target_cols_to_mask = []

        for col in self.all_feature_column_names:
            col_lower = col.lower()
            mask = False

            if ab_lower == "weather":
                if any(w in col_lower for w in [
                    "outdoor_dry_bulb_temperature",
                    "outdoor_relative_humidity",
                    "relative_humidity",
                    "diffuse_solar_irradiance",
                    "direct_solar_irradiance",
                ]):
                    mask = True

            elif ab_lower == "indoor_thermal":
                if "indoor_dry_bulb_temperature" in col_lower:
                    mask = True

            elif ab_lower == "price_carbon":
                if any(p in col_lower for p in ["electricity_pricing", "carbon_intensity", "pricing", "cost_function"]):
                    mask = True

            elif ab_lower == "occupancy_load":
                if any(o in col_lower for o in ["occupant_counter", "non_shiftable_load"]):
                    mask = True

            elif ab_lower == "storage_soc":
                if any(s in col_lower for s in [
                    "electrical_storage_soc",
                    "cooling_storage_soc",
                    "heating_storage_soc",
                    "dhw_storage_soc",
                    "storage_soc",
                ]):
                    mask = True

            elif ab_lower == "past_actions":
                if col.startswith("action_"):
                    mask = True

            elif ab_lower in ["current_action", "all_actions"]:
                pass

            elif ab_lower == col_lower or ab_lower == col_lower.replace("target_", "").replace("action_", ""):
                mask = True

            if mask:
                target_cols_to_mask.append(col)

        for col in target_cols_to_mask:
            mean_val = self.marginal_means.get(col, 0.0)
            if col in context_df.columns:
                context_df[col] = mean_val

        if ab_lower in ["current_action", "all_actions"]:
            for col in self.action_column_names:
                mean_val = self.marginal_means.get(col, 0.0)
                if col in future_df.columns:
                    future_df[col] = mean_val

        return context_df, future_df

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
        context_df = pd.DataFrame(context_data, columns=self.all_feature_column_names, dtype=np.float32)
        context_df["id"] = 0
        context_df["timestamp"] = pd.to_datetime(np.arange(self.context_length), unit="s")

        future_df = pd.DataFrame([action], columns=self.action_column_names, dtype=np.float32)
        future_df["id"] = 0
        future_df["timestamp"] = pd.to_datetime([self.context_length], unit="s")

        context_df, future_df = self._apply_channel_ablation(context_df, future_df)

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

        return next_obs_pred, reward, terminated, truncated, {}

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
            b_dict = {
                k: float(v.item()) if hasattr(v, "item") and getattr(v, "size", 1) == 1
                else (float(v[0]) if isinstance(v, (list, np.ndarray)) and len(v) > 0
                      else (float(v) if isinstance(v, (int, float, np.number)) else v))
                for k, v in raw_b_dict.items()
            }

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


# =============================================================================
# 2. Offline Dataset Harvester & Marginal Statistics Estimator
# =============================================================================
def collect_offline_data_and_statistics(schema: str, num_steps: int = 2000, seed: int = 42):
    print(f"\n[Data Collection] Gathering {num_steps} transitions to compute marginal statistics...")
    raw_env = CityLearnEnv(schema, central_agent=True)

    obs_list, _ = raw_env.reset(seed=seed)
    curr_obs = np.array(obs_list[0], dtype=np.float32)

    obs_records, act_records, delta_records = [], [], []

    for step in range(num_steps):
        action = raw_env.action_space[0].sample().astype(np.float32)
        nxt_obs_list, _, done, trunc, _ = raw_env.step([action])
        nxt_obs = np.array(nxt_obs_list[0], dtype=np.float32)

        delta = nxt_obs - curr_obs
        obs_records.append(curr_obs)
        act_records.append(action)
        delta_records.append(delta)

        curr_obs = nxt_obs
        if done or trunc:
            obs_list, _ = raw_env.reset(seed=seed + step)
            curr_obs = np.array(obs_list[0], dtype=np.float32)

    obs_arr = np.array(obs_records, dtype=np.float32)
    act_arr = np.array(act_records, dtype=np.float32)
    delta_arr = np.array(delta_records, dtype=np.float32)

    raw_targets = raw_env.observation_names[0]
    raw_actions = raw_env.action_names[0]
    flat_targets = [str(c) for s in (raw_targets if isinstance(raw_targets[0], list) else [raw_targets]) for c in (s if isinstance(s, list) else [s])]
    flat_actions = [str(c) for s in (raw_actions if isinstance(raw_actions[0], list) else [raw_actions]) for c in (s if isinstance(s, list) else [s])]

    target_cols = [f"target_{i}_{name}" for i, name in enumerate(flat_targets)]
    action_cols = [f"action_{i}_{name}" for i, name in enumerate(flat_actions)]
    all_cols = target_cols + action_cols

    marginal_means = {}
    for i, col in enumerate(target_cols):
        marginal_means[col] = float(np.mean(delta_arr[:, i]))
    for i, col in enumerate(action_cols):
        marginal_means[col] = float(np.mean(act_arr[:, i]))

    num_buildings = len(raw_env.buildings)
    building_actions = {b: [] for b in range(num_buildings)}
    act_ptr = 0
    for b_idx, bldg in enumerate(raw_env.buildings):
        num_bldg_acts = bldg.action_space.shape[0]
        building_actions[b_idx] = action_cols[act_ptr : act_ptr + num_bldg_acts]
        act_ptr += num_bldg_acts

    building_targets = {b: [] for b in range(num_buildings)}
    shared_targets = []
    shared_keywords = ["day_type", "hour", "outdoor_dry_bulb_temperature", "diffuse_solar", "direct_solar", "carbon_intensity", "pricing"]

    bldg_var_counts = collections.defaultdict(int)
    for t_idx, col_name in enumerate(target_cols):
        col_clean = col_name.split("_", 2)[-1]
        if any(k in col_clean.lower() for k in shared_keywords) and "cooling_set_point" not in col_clean.lower():
            shared_targets.append(t_idx)
        else:
            b_idx = bldg_var_counts[col_clean] % num_buildings
            building_targets[b_idx].append(t_idx)
            bldg_var_counts[col_clean] += 1

    target_stds = np.std(delta_arr, axis=0) + 1e-6

    dataset = {
        "observations": obs_arr,
        "actions": act_arr,
        "deltas": delta_arr,
        "target_cols": target_cols,
        "action_cols": action_cols,
        "all_cols": all_cols,
        "marginal_means": marginal_means,
        "num_buildings": num_buildings,
        "building_actions": building_actions,
        "building_targets": building_targets,
        "shared_targets": shared_targets,
        "target_stds": target_stds,
        "schema": schema,
        "seed": seed,
    }
    print(f"[Data Collection] Successfully extracted {len(all_cols)} channel statistics across {num_buildings} buildings.")
    return dataset


# =============================================================================
# 3. Leave-One-Channel-Out (LOCO) Dynamics Error Evaluator
# =============================================================================
def evaluate_dynamics_fidelity(pipeline, dataset, context_length=16, ablation_channel=None, eval_steps=720):
    all_cols = dataset["all_cols"]
    target_cols = dataset["target_cols"]
    action_cols = dataset["action_cols"]
    marginal_means = dataset["marginal_means"]

    deltas = dataset["deltas"]
    actions = dataset["actions"]
    N = min(eval_steps, len(deltas) - context_length - 5)

    y_true_list = []
    y_pred_list = []
    crps_scores = []

    for t in range(context_length, context_length + N):
        hist_deltas = deltas[t - context_length : t].copy()
        hist_actions = actions[t - context_length : t].copy()
        curr_action = actions[t].copy()
        true_delta = deltas[t].copy()

        context_data = [np.concatenate([hist_deltas[i], hist_actions[i]]) for i in range(context_length)]
        context_df = pd.DataFrame(context_data, columns=all_cols, dtype=np.float32)
        context_df["id"] = 0
        context_df["timestamp"] = pd.to_datetime(np.arange(context_length), unit="s")

        future_df = pd.DataFrame([curr_action], columns=action_cols, dtype=np.float32)
        future_df["id"] = 0
        future_df["timestamp"] = pd.to_datetime([context_length], unit="s")

        if ablation_channel and str(ablation_channel).strip().lower() != "none":
            ab_lower = str(ablation_channel).strip().lower()

            for col in all_cols:
                col_lower = col.lower()
                mask = False

                if ab_lower == "weather":
                    if any(w in col_lower for w in [
                        "outdoor_dry_bulb_temperature", "outdoor_relative_humidity",
                        "relative_humidity", "diffuse_solar_irradiance", "direct_solar_irradiance"
                    ]):
                        mask = True
                elif ab_lower == "indoor_thermal":
                    if "indoor_dry_bulb_temperature" in col_lower:
                        mask = True
                elif ab_lower == "price_carbon":
                    if any(p in col_lower for p in ["electricity_pricing", "carbon_intensity", "pricing", "cost_function"]):
                        mask = True
                elif ab_lower == "occupancy_load":
                    if any(o in col_lower for o in ["occupant_counter", "non_shiftable_load"]):
                        mask = True
                elif ab_lower == "storage_soc":
                    if any(s in col_lower for s in [
                        "electrical_storage_soc", "cooling_storage_soc", "heating_storage_soc",
                        "dhw_storage_soc", "storage_soc"
                    ]):
                        mask = True
                elif ab_lower == "past_actions" and col.startswith("action_"):
                    mask = True
                elif ab_lower == col_lower or ab_lower == col_lower.replace("target_", "").replace("action_", ""):
                    mask = True

                if mask and col in context_df.columns:
                    context_df[col] = marginal_means.get(col, 0.0)

            if ab_lower in ["current_action", "all_actions"]:
                for col in action_cols:
                    future_df[col] = marginal_means.get(col, 0.0)

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

            available_quantiles = [q for q in [0.1, 0.25, 0.5, 0.75, 0.9] if q in pred_df.columns]
            step_pinball = []
            if len(available_quantiles) >= 3:
                for q in available_quantiles:
                    q_piv = pred_df.pivot(index="timestamp", columns=t_col, values=q)
                    q_val = q_piv[target_cols].values[0].astype(np.float32)
                    err = true_delta - q_val
                    loss = np.maximum(q * err, (q - 1) * err)
                    step_pinball.append(np.mean(loss))
                crps_scores.append(np.mean(step_pinball))
            else:
                crps_scores.append(np.mean(np.abs(true_delta - pred_delta)))

        y_true_list.append(true_delta)
        y_pred_list.append(pred_delta)

    Y_true = np.array(y_true_list)
    Y_pred = np.array(y_pred_list)

    std_per_dim = np.std(Y_true, axis=0) + 1e-6
    nmae_per_dim = np.mean(np.abs(Y_true - Y_pred), axis=0) / std_per_dim
    overall_nmae = float(np.mean(nmae_per_dim))

    overall_mae = float(np.mean(np.abs(Y_true - Y_pred)))
    overall_rmse = float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))
    overall_crps = float(np.mean(crps_scores))

    per_dim_metrics = {}
    for idx, name in enumerate(target_cols):
        clean_name = name.replace("target_", "")
        per_dim_metrics[f"nMAE_{clean_name}"] = float(nmae_per_dim[idx])
        per_dim_metrics[f"MAE_{clean_name}"] = float(np.mean(np.abs(Y_true[:, idx] - Y_pred[:, idx])))

    summary_metrics = {
        "Ablation Channel": ablation_channel or "None (Baseline)",
        "Overall nMAE (Scale-Normalized)": round(overall_nmae, 6),
        "Overall MAE": round(overall_mae, 6),
        "Overall RMSE": round(overall_rmse, 6),
        "Overall CRPS": round(overall_crps, 6),
        **per_dim_metrics,
    }
    return summary_metrics


# =============================================================================
# 4A. Subsystem-Isolated Action Sweeps with Synchronized Real Environment Ground Truth [Fixed 4.1]
# =============================================================================
def evaluate_isolated_device_action_sweep(
    pipeline,
    dataset,
    context_length: int = 16,
    num_samples: int = 15,  # [Fixed 4.1] Default to 15 synchronized sample points for fast replay
    test_action_values: list = None,
):
    """
    [Fixed 4.1] Evaluates isolated actuator sweeps across TSFM and CityLearn ground truth:
      - Uses deterministic replay up to s_idx rather than copy.deepcopy (avoids PyTorch tensor crash)
      - Caps s_idx strictly within episode horizon (<= min(time_steps - 2, 718)) to avoid IndexError
      - Clips counterfactual actions to physical simulator bounds (avoids negative power assertion)
    """
    if test_action_values is None:
        test_action_values = [-1.0, -0.5, 0.0, 0.5, 1.0]

    all_cols = dataset["all_cols"]
    target_cols = dataset["target_cols"]
    action_cols = dataset["action_cols"]
    deltas = dataset["deltas"]
    actions = dataset["actions"]
    schema = dataset.get("schema", "citylearn_challenge_2023_phase_1")
    seed = dataset.get("seed", 42)

    device_groups = {
        "Electrical_Battery": [
            c for c in action_cols if "electrical_storage" in c.lower() or ("storage" in c.lower() and "dhw" not in c.lower() and "cool" not in c.lower() and "heat" not in c.lower())
        ],
        "HVAC_Cooling": [
            c for c in action_cols if any(k in c.lower() for k in ["cooling_device", "cooling", "hvac", "heat_pump"]) and "storage" not in c.lower()
        ],
        "Thermal_Storage": [
            c for c in action_cols if any(k in c.lower() for k in ["dhw_storage", "cooling_storage", "heating_storage", "dhw"])
        ],
    }

    print("\n" + "=" * 60)
    print(" PHASE 2A: SYNCHRONIZED REAL VS. TSFM DEVICE ACTION SWEEPS [Fixed 4.1] ")
    print("=" * 60)
    for dev_name, cols in device_groups.items():
        print(f" -> Device Group [{dev_name}]: {len(cols)} active channels mapped -> {cols}")

    # [Fixed 4.1] Instantiate persistent simulator once outside evaluation loops
    sim_env = CityLearnEnv(schema, central_agent=True)
    sim_obs_init, _ = sim_env.reset(seed=seed)

    # [Fixed 4.1] Strictly cap sample indices to prevent episode index overflow (IndexError on hvac_mode)
    max_eval_step = min(int(sim_env.time_steps) - 2, len(deltas) - 5, 718)
    sample_indices = np.unique(np.linspace(context_length, max_eval_step, num_samples, dtype=int))

    device_sweep_results = {}

    for dev_name, active_act_cols in device_groups.items():
        if not active_act_cols:
            print(f" [Warning] No action columns found matching device group: {dev_name}. Skipping.")
            continue

        tsfm_sweep_records = {act_val: [] for act_val in test_action_values}
        real_sweep_records = {act_val: [] for act_val in test_action_values}

        for s_idx in sample_indices:
            # Historical context for TSFM
            hist_deltas = deltas[s_idx - context_length : s_idx]
            hist_actions = actions[s_idx - context_length : s_idx]

            c_data = [np.concatenate([hist_deltas[i], hist_actions[i]]) for i in range(context_length)]
            c_df = pd.DataFrame(c_data, columns=all_cols, dtype=np.float32)
            c_df["id"] = 0
            c_df["timestamp"] = pd.to_datetime(np.arange(context_length), unit="s")

            for act_val in test_action_values:
                forced_act_dict = {col: 0.0 for col in action_cols}
                for col in active_act_cols:
                    forced_act_dict[col] = float(act_val)

                # 1. TSFM World Model Forecast
                f_df = pd.DataFrame([forced_act_dict], dtype=np.float32)
                f_df["id"] = 0
                f_df["timestamp"] = pd.to_datetime([context_length], unit="s")

                with torch.no_grad():
                    pred_df = pipeline.predict_df(
                        df=c_df, future_df=f_df, prediction_length=1,
                        id_column="id", timestamp_column="timestamp", target=target_cols
                    )
                    t_col = "target_name" if "target_name" in pred_df.columns else "target"
                    v_col = 0.5 if 0.5 in pred_df.columns else "predictions"
                    piv = pred_df.pivot(index="timestamp", columns=t_col, values=v_col)
                    tsfm_delta = piv[target_cols].values[0].astype(np.float32)
                    tsfm_sweep_records[act_val].append(tsfm_delta)

                # 2. [Fixed 4.1] Ground-Truth Real Environment Deterministic Replay (No deepcopy)
                forced_act_vector = np.array([forced_act_dict[col] for col in action_cols], dtype=np.float32)
                # [Fixed 4.1] Clip action vector to physical simulator bounds (prevents negative chiller power crash)
                sim_action = np.clip(
                    forced_act_vector,
                    sim_env.action_space[0].low,
                    sim_env.action_space[0].high,
                ).astype(np.float32)

                sim_obs_init, _ = sim_env.reset(seed=seed)
                last_step_obs = sim_obs_init
                for t in range(s_idx):
                    last_step_obs, _, _, _, _ = sim_env.step([actions[t]])
                curr_real_obs = np.array(last_step_obs[0] if s_idx > 0 else sim_obs_init[0], dtype=np.float32)

                nxt_real_obs_list, _, _, _, _ = sim_env.step([sim_action])
                real_delta = np.array(nxt_real_obs_list[0], dtype=np.float32) - curr_real_obs
                real_sweep_records[act_val].append(real_delta)

        # Empirical expectations across samples
        tsfm_mean_matrix = np.array([np.mean(tsfm_sweep_records[act_val], axis=0) for act_val in test_action_values])
        real_mean_matrix = np.array([np.mean(real_sweep_records[act_val], axis=0) for act_val in test_action_values])

        zero_act_idx = test_action_values.index(0.0) if 0.0 in test_action_values else len(test_action_values) // 2

        # Decoupled passive drift vs active control authority
        tsfm_baseline_drift = tsfm_mean_matrix[zero_act_idx].copy()
        real_baseline_drift = real_mean_matrix[zero_act_idx].copy()

        tsfm_action_contribution = tsfm_mean_matrix - tsfm_baseline_drift
        real_action_contribution = real_mean_matrix - real_baseline_drift

        summary_dict = {}
        for idx, col in enumerate(target_cols):
            clean_name = col.replace("target_", "")
            row_data = {
                "Target": clean_name,
                "TSFM_Idle_Drift": round(float(tsfm_baseline_drift[idx]), 5),
                "Real_Idle_Drift": round(float(real_baseline_drift[idx]), 5),
                "Drift_Error": round(float(abs(tsfm_baseline_drift[idx] - real_baseline_drift[idx])), 5),
            }
            for v_i, val in enumerate(test_action_values):
                row_data[f"TSFM_Act_{val:+.1f}"] = round(float(tsfm_mean_matrix[v_i, idx]), 5)
                row_data[f"Real_Act_{val:+.1f}"] = round(float(real_mean_matrix[v_i, idx]), 5)
                row_data[f"TSFM_Contrib_{val:+.1f}"] = round(float(tsfm_action_contribution[v_i, idx]), 5)
                row_data[f"Real_Contrib_{val:+.1f}"] = round(float(real_action_contribution[v_i, idx]), 5)
            summary_dict[clean_name] = row_data

        sweep_summary_df = pd.DataFrame.from_dict(summary_dict, orient="index")

        device_sweep_results[dev_name] = {
            "summary_df": sweep_summary_df,
            "tsfm_response_matrix": tsfm_mean_matrix,
            "real_response_matrix": real_mean_matrix,
            "tsfm_baseline_drift": tsfm_baseline_drift,
            "real_baseline_drift": real_baseline_drift,
            "tsfm_action_contribution": tsfm_action_contribution,
            "real_action_contribution": real_action_contribution,
            "active_channels": active_act_cols,
        }

    return device_sweep_results, test_action_values


# =============================================================================
# 4B. Cross-Building Spatial Attention Leakage Probe vs. Real Environment Baseline [Fixed 4.1]
# =============================================================================
def evaluate_cross_building_spatial_leakage(
    pipeline,
    dataset,
    context_length: int = 16,
    num_samples: int = 15,  # [Fixed 4.1]
    target_building_idx: int = 0,
    test_action_values: list = None,
):
    """
    [Fixed 4.1] Evaluates spatial leakage comparing TSFM multi-agent coupling against
    deterministic real simulator ground-truth replay.
    """
    if test_action_values is None:
        test_action_values = [-1.0, -0.5, 0.0, 0.5, 1.0]

    all_cols = dataset["all_cols"]
    target_cols = dataset["target_cols"]
    action_cols = dataset["action_cols"]
    deltas = dataset["deltas"]
    actions = dataset["actions"]
    schema = dataset.get("schema", "citylearn_challenge_2023_phase_1")
    seed = dataset.get("seed", 42)

    building_targets = dataset["building_targets"]
    building_actions = dataset["building_actions"]
    target_stds = dataset["target_stds"]

    own_action_cols = building_actions.get(target_building_idx, [])
    own_target_indices = building_targets.get(target_building_idx, [])

    other_target_indices = [
        idx for b_idx, idxs in building_targets.items() if b_idx != target_building_idx for idx in idxs
    ]

    print("\n" + "=" * 60)
    print(f" PHASE 2B: CROSS-BUILDING SPATIAL LEAKAGE PROBE (Target: Bldg {target_building_idx}) [Fixed 4.1] ")
    print("=" * 60)
    print(f" -> Perturbing Building {target_building_idx} Actions ({len(own_action_cols)} channels): {own_action_cols}")
    print(f" -> Comparing TSFM predictions against real decoupled CityLearn ground truth")

    sim_env = CityLearnEnv(schema, central_agent=True)
    sim_obs_init, _ = sim_env.reset(seed=seed)

    # [Fixed 4.1] Strictly cap sample indices to prevent episode index overflow
    max_eval_step = min(int(sim_env.time_steps) - 2, len(deltas) - 5, 718)
    sample_indices = np.unique(np.linspace(context_length, max_eval_step, num_samples, dtype=int))

    tsfm_sweep_records = {act_val: [] for act_val in test_action_values}
    real_sweep_records = {act_val: [] for act_val in test_action_values}

    for s_idx in sample_indices:
        hist_deltas = deltas[s_idx - context_length : s_idx]
        hist_actions = actions[s_idx - context_length : s_idx]

        c_data = [np.concatenate([hist_deltas[i], hist_actions[i]]) for i in range(context_length)]
        c_df = pd.DataFrame(c_data, columns=all_cols, dtype=np.float32)
        c_df["id"] = 0
        c_df["timestamp"] = pd.to_datetime(np.arange(context_length), unit="s")

        for act_val in test_action_values:
            forced_act_dict = {col: 0.0 for col in action_cols}
            for col in own_action_cols:
                forced_act_dict[col] = float(act_val)

            # 1. TSFM World Model Forecast
            f_df = pd.DataFrame([forced_act_dict], dtype=np.float32)
            f_df["id"] = 0
            f_df["timestamp"] = pd.to_datetime([context_length], unit="s")

            with torch.no_grad():
                pred_df = pipeline.predict_df(
                    df=c_df, future_df=f_df, prediction_length=1,
                    id_column="id", timestamp_column="timestamp", target=target_cols
                )
                t_col = "target_name" if "target_name" in pred_df.columns else "target"
                v_col = 0.5 if 0.5 in pred_df.columns else "predictions"
                piv = pred_df.pivot(index="timestamp", columns=t_col, values=v_col)
                tsfm_delta = piv[target_cols].values[0].astype(np.float32)
                tsfm_sweep_records[act_val].append(tsfm_delta)

            # 2. [Fixed 4.1] Ground-Truth Real Environment Deterministic Replay
            forced_act_vector = np.array([forced_act_dict[col] for col in action_cols], dtype=np.float32)
            sim_action = np.clip(
                forced_act_vector,
                sim_env.action_space[0].low,
                sim_env.action_space[0].high,
            ).astype(np.float32)

            sim_obs_init, _ = sim_env.reset(seed=seed)
            last_step_obs = sim_obs_init
            for t in range(s_idx):
                last_step_obs, _, _, _, _ = sim_env.step([actions[t]])
            curr_real_obs = np.array(last_step_obs[0] if s_idx > 0 else sim_obs_init[0], dtype=np.float32)

            nxt_real_obs_list, _, _, _, _ = sim_env.step([sim_action])
            real_delta = np.array(nxt_real_obs_list[0], dtype=np.float32) - curr_real_obs
            real_sweep_records[act_val].append(real_delta)

    tsfm_mean_matrix = np.array([np.mean(tsfm_sweep_records[act_val], axis=0) for act_val in test_action_values])
    real_mean_matrix = np.array([np.mean(real_sweep_records[act_val], axis=0) for act_val in test_action_values])

    tsfm_dynamic_range = np.ptp(tsfm_mean_matrix, axis=0) / target_stds
    real_dynamic_range = np.ptp(real_mean_matrix, axis=0) / target_stds

    tsfm_own_swing = np.mean(tsfm_dynamic_range[own_target_indices]) if own_target_indices else 1e-6
    tsfm_other_swing = np.mean(tsfm_dynamic_range[other_target_indices]) if other_target_indices else 0.0
    tsfm_leakage_pct = (tsfm_other_swing / (tsfm_own_swing + 1e-8)) * 100.0

    real_own_swing = np.mean(real_dynamic_range[own_target_indices]) if own_target_indices else 1e-6
    real_other_swing = np.mean(real_dynamic_range[other_target_indices]) if other_target_indices else 0.0
    real_leakage_pct = (real_other_swing / (real_own_swing + 1e-8)) * 100.0

    leakage_report = {
        "Target Building": target_building_idx,
        "TSFM Own Dynamic Swing": round(float(tsfm_own_swing), 6),
        "Real Own Dynamic Swing": round(float(real_own_swing), 6),
        "TSFM Other Leakage Swing": round(float(tsfm_other_swing), 6),
        "Real Other Leakage Swing (Ground Truth)": round(float(real_other_swing), 6),
        "TSFM Spatial Leakage Ratio (%)": round(float(tsfm_leakage_pct), 2),
        "Real Spatial Leakage Ratio (%)": round(float(real_leakage_pct), 2),
        "Spatial Integrity Pass": bool(tsfm_leakage_pct < 5.0),
        "Diagnostic Status": (
            "Pass (Causally Isolated Buildings)" if tsfm_leakage_pct < 5.0
            else f"FLAG: Cross-Building Attention Bleed ({tsfm_leakage_pct:.1f}%)"
        ),
    }

    leakage_df = pd.DataFrame([leakage_report])
    print("\n--- Spatial Isolation & Cross-Building Leakage Report ---")
    print(leakage_df.to_string(index=False))

    return leakage_df, tsfm_mean_matrix, real_mean_matrix, own_target_indices, other_target_indices


# =============================================================================
# 5. Downstream PPO Training & Annual Ground-Truth Evaluation
# =============================================================================
def run_downstream_ppo_ablation(args, pipeline, marginal_means, ablation_channels, seed=42):
    print("\n============================================================")
    print("      PHASE 3: DOWNSTREAM PPO RETRAINING ON ABLATED TSFM    ")
    print("============================================================")

    ablation_rl_results = []

    for ab_channel in ablation_channels:
        print(f"\n[RL Ablation] Retraining PPO under Ablated World Model: [{ab_channel}]...")
        set_random_seed(seed)

        env_dream = CityLearnTSFMEnv(
            schema=args.train_schema,
            context_length=args.context_length,
            pipeline=pipeline,
            device=args.device,
            max_steps=720,
            ablation_channel=ab_channel,
            marginal_means=marginal_means,
        )

        ppo_model = PPO(
            policy="MlpPolicy",
            env=env_dream,
            device="cpu",
            seed=seed,
            verbose=0,
            tensorboard_log=f"runs/ablation_{ab_channel}_seed_{seed}",
        )

        ppo_model.learn(total_timesteps=args.timesteps)

        print(f"  [Sim-to-Real Deployment] Evaluating Policy in Ground-Truth Env...")
        eval_env = CityLearnEnv(args.eval_schema, central_agent=True)
        eval_env = StableBaselines3Wrapper(eval_env)

        obs, _ = eval_env.reset(seed=seed)
        done = False
        steps = 0

        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = eval_env.step(action)
            steps += 1
            done = term or trunc

        print(f"  [Sim-to-Real Deployment] Evaluated PPO over {steps} steps")

        raw_kpis = eval_env.unwrapped.evaluate()
        if "cost_function" in raw_kpis.columns and "name" in raw_kpis.columns and "value" in raw_kpis.columns:
            pivoted_kpi = raw_kpis.pivot(index="cost_function", columns="name", values="value").astype(float)
        else:
            pivoted_kpi = raw_kpis.copy().astype(float)

        district_col = "District" if "District" in pivoted_kpi.columns else pivoted_kpi.columns[-1]

        cost_total = float(pivoted_kpi.loc["cost_total", district_col]) if "cost_total" in pivoted_kpi.index else 1.0
        carbon_total = float(pivoted_kpi.loc["carbon_emissions_total", district_col]) if "carbon_emissions_total" in pivoted_kpi.index else 1.0
        discomfort = float(pivoted_kpi.loc["discomfort_proportion", district_col]) if "discomfort_proportion" in pivoted_kpi.index else 0.0

        ablation_rl_results.append({
            "Ablation Channel": ab_channel or "None (Baseline)",
            "District Cost": round(cost_total, 4),
            "District Carbon": round(carbon_total, 4),
            "Discomfort Proportion": round(discomfort, 4),
            "Total Steps Deployed": steps,
        })

    return pd.DataFrame(ablation_rl_results)


# =============================================================================
# 6. Exploitation vs. Signal Diagnostic Cross-Analyzer
# =============================================================================
def analyze_exploitation_vs_signal(dynamics_df: pd.DataFrame, rl_df: pd.DataFrame):
    merged = pd.merge(dynamics_df, rl_df, on="Ablation Channel")

    nmae_col = "Overall nMAE (Scale-Normalized)" if "Overall nMAE (Scale-Normalized)" in merged.columns else "Overall MAE"
    base_nmae = float(merged.loc[merged["Ablation Channel"] == "None (Baseline)", nmae_col].iloc[0])
    base_cost = float(merged.loc[merged["Ablation Channel"] == "None (Baseline)", "District Cost"].iloc[0])

    merged["Delta_nMAE (%)"] = ((merged[nmae_col] - base_nmae) / base_nmae * 100.0).round(2)
    merged["Delta_Cost (%)"] = ((merged["District Cost"] - base_cost) / base_cost * 100.0).round(2)

    diagnostics = []
    for _, row in merged.iterrows():
        d_nmae = row["Delta_nMAE (%)"]
        d_cost = row["Delta_Cost (%)"]

        if row["Ablation Channel"] == "None (Baseline)":
            diagnostics.append("Reference Baseline")
        elif abs(d_nmae) < 2.0 and abs(d_cost) > 5.0:
            diagnostics.append("EXPLOITATION DETECTED (Policy relies on artifact, not physical dynamics)")
        elif d_nmae > 5.0 and d_cost > 3.0:
            diagnostics.append("CRITICAL PHYSICAL SIGNAL (High causal fidelity for dynamics & control)")
        elif d_nmae > 5.0 and abs(d_cost) <= 3.0:
            diagnostics.append("PASSIVE EXOGENOUS (High dynamics impact, but policy control is robust)")
        else:
            diagnostics.append("INFORMATIVE (Balanced dynamics & control contribution)")

    merged["Diagnostic Classification"] = diagnostics
    return merged


# =============================================================================
# 7. Comprehensive Diagnostic Visualization Suite [Fixed 4.1]
# =============================================================================
def generate_all_experiment_visualizations(
    target_cols: list,
    feature_cols: list,
    dynamics_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
    device_sweep_results: dict,
    spatial_leakage_bundle: tuple,
    test_action_vals: list,
    output_dir: str = "./visualizations",
):
    os.makedirs(output_dir, exist_ok=True)
    generated_figures = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------------------
    # Plot A1: Subsystem-Isolated Actuator Sweeps (TSFM vs. Real Ground Truth)
    # -------------------------------------------------------------------------
    if device_sweep_results:
        fig_a1, axes_a1 = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
        configs = [
            ("Electrical_Battery", "electrical_storage_soc", "Battery ΔSoC", "#9467bd", axes_a1[0]),
            ("HVAC_Cooling", "indoor_dry_bulb_temperature", "Indoor ΔT_in (°C)", "#2ca02c", axes_a1[1]),
            ("Thermal_Storage", "dhw_storage_soc", "Thermal / DHW ΔSoC", "#d62728", axes_a1[2]),
        ]

        for dev_name, target_key, y_label, color, ax in configs:
            if dev_name in device_sweep_results:
                tsfm_mat = device_sweep_results[dev_name]["tsfm_response_matrix"]
                real_mat = device_sweep_results[dev_name]["real_response_matrix"]
                tsfm_base = device_sweep_results[dev_name]["tsfm_baseline_drift"]
                real_base = device_sweep_results[dev_name]["real_baseline_drift"]

                # [Fixed 4.1] Robust target index search avoiding false matches between electrical and thermal storage
                t_idx = next((i for i, c in enumerate(target_cols) if target_key in c.lower()), None)
                if t_idx is None and dev_name == "Thermal_Storage":
                    t_idx = next((i for i, c in enumerate(target_cols) if ("storage_soc" in c.lower() and "electrical" not in c.lower())), None)

                if t_idx is not None:
                    tsfm_curve = tsfm_mat[:, t_idx]
                    real_curve = real_mat[:, t_idx]
                    tsfm_idle = tsfm_base[t_idx]
                    real_idle = real_base[t_idx]

                    ax.plot(test_action_vals, real_curve, marker="s", markersize=6, linewidth=2.4, color="#1f77b4", linestyle="--", label="CityLearn Ground Truth", zorder=4)
                    ax.plot(test_action_vals, tsfm_curve, marker="o", markersize=6, linewidth=2.4, color=color, label="TSFM Predicted Response", zorder=5)

                    ax.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.5, label="Neutral Zero Line (Δs = 0)", zorder=1)
                    ax.axhline(real_idle, color="#1f77b4", linestyle=":", linewidth=1.6, alpha=0.8, label=f"Real Idle ({real_idle:+.4f})", zorder=2)
                    ax.axhline(tsfm_idle, color=color, linestyle=":", linewidth=1.6, alpha=0.8, label=f"TSFM Idle ({tsfm_idle:+.4f})", zorder=3)

                    ax.fill_between(
                        test_action_vals, real_curve, real_idle,
                        color="#1f77b4", alpha=0.10, label="Real Action Contribution δ_real(a)", zorder=0
                    )
                    ax.fill_between(
                        test_action_vals, tsfm_curve, tsfm_idle,
                        color=color, alpha=0.15, label="TSFM Action Contribution δ_tsfm(a)", zorder=1
                    )

                    ax.set_title(f"Isolated {dev_name.replace('_', ' ')}: TSFM vs. Real", fontsize=11, fontweight="bold", pad=8)
                    ax.set_xlabel("Isolated Action Value $a_t \\in [-1.0, 1.0]$", fontsize=9.5, fontweight="bold")
                    ax.set_ylabel(y_label, fontsize=9.5, fontweight="bold")
                    ax.grid(True, linestyle=":", alpha=0.6)
                    ax.legend(frameon=True, fontsize=7.5, loc="best")
                else:
                    ax.text(0.5, 0.5, f"Key '{target_key}' not in targets", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, f"Device '{dev_name}' not found", ha="center", va="center")

        fig_a1.suptitle("Subsystem-Isolated Actuator Sweeps: TSFM Forecast vs. Real CityLearn Ground Truth", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        path_a1 = os.path.join(output_dir, "plot_A1_isolated_device_sweeps.png")
        fig_a1.savefig(path_a1, dpi=300, bbox_inches="tight")
        generated_figures["plot_A1_isolated_device_sweeps"] = fig_a1

    # -------------------------------------------------------------------------
    # Plot A2: Cross-Building Spatial Isolation Audit vs. Real Baseline
    # -------------------------------------------------------------------------
    if spatial_leakage_bundle is not None:
        leakage_df, tsfm_spatial_mat, real_spatial_mat, own_indices, other_indices, target_bldg = spatial_leakage_bundle
        fig_a2, (ax_sp1, ax_sp2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)

        for idx in own_indices[:4]:
            lbl = target_cols[idx].split("_", 2)[-1]
            ax_sp1.plot(test_action_vals, real_spatial_mat[:, idx], marker="s", linestyle="--", linewidth=1.8, label=f"Real: {lbl}", alpha=0.8, zorder=3)
            ax_sp1.plot(test_action_vals, tsfm_spatial_mat[:, idx], marker="o", linewidth=2.0, label=f"TSFM: {lbl}", zorder=4)
        ax_sp1.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.5, label="Neutral Zero Line", zorder=1)
        ax_sp1.set_title(f"Building {target_bldg} Internal States (Action Source): TSFM vs. Real", fontsize=11, fontweight="bold")
        ax_sp1.set_xlabel(f"Building {target_bldg} Forced Action $a_{target_bldg}$", fontsize=9.5, fontweight="bold")
        ax_sp1.set_ylabel("State Delta (Δs)", fontsize=9.5, fontweight="bold")
        ax_sp1.legend(frameon=True, fontsize=7.5, loc="best")
        ax_sp1.grid(True, linestyle=":", alpha=0.6)

        for idx in other_indices[:4]:
            lbl = target_cols[idx].split("_", 2)[-1] + f" (Col {idx})"
            ax_sp2.plot(test_action_vals, real_spatial_mat[:, idx], marker="s", linestyle="--", linewidth=1.8, color="#1f77b4", label=f"Real Ground Truth: {lbl}", alpha=0.7, zorder=3)
            ax_sp2.plot(test_action_vals, tsfm_spatial_mat[:, idx], marker="o", linewidth=1.8, label=f"TSFM Prediction: {lbl}", zorder=4)
        ax_sp2.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.5, label="Ideal Zero Leakage Line", zorder=1)
        ax_sp2.axhspan(-0.005, 0.005, color="gray", alpha=0.15, label="Permissible Isolation Tolerance (±0.005)", zorder=0)

        ax_sp2.set_title("Other Buildings External States (TSFM Leakage vs. Real Zero Response)", fontsize=11, fontweight="bold")
        ax_sp2.set_xlabel(f"Building {target_bldg} Forced Action $a_{target_bldg}$", fontsize=9.5, fontweight="bold")
        ax_sp2.set_ylabel("State Delta (Δs)", fontsize=9.5, fontweight="bold")
        ax_sp2.legend(frameon=True, fontsize=7.5, loc="best")
        ax_sp2.grid(True, linestyle=":", alpha=0.6)

        leak_pct = leakage_df["TSFM Spatial Leakage Ratio (%)"].iloc[0]
        real_leak_pct = leakage_df["Real Spatial Leakage Ratio (%)"].iloc[0]
        fig_a2.suptitle(
            f"Cross-Building Spatial Isolation Audit (Bldg {target_bldg} | TSFM Leakage: {leak_pct:.2f}% | Real Ref: {real_leak_pct:.2f}%)",
            fontsize=13, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        path_a2 = os.path.join(output_dir, "plot_A2_spatial_leakage.png")
        fig_a2.savefig(path_a2, dpi=300, bbox_inches="tight")
        generated_figures["plot_A2_spatial_leakage"] = fig_a2

    # -------------------------------------------------------------------------
    # Plot D: LOCO Dynamics Error Degradation Bar Chart
    # -------------------------------------------------------------------------
    if dynamics_df is not None and not dynamics_df.empty:
        fig_d, ax_d = plt.subplots(figsize=(10, 4.5), dpi=300)
        nmae_col = "Overall nMAE (Scale-Normalized)" if "Overall nMAE (Scale-Normalized)" in dynamics_df.columns else "Overall MAE"

        sns.barplot(
            data=dynamics_df,
            x="Ablation Channel",
            y=nmae_col,
            palette="Blues_d",
            ax=ax_d,
        )
        ax_d.set_title("Leave-One-Channel-Out (LOCO) Dynamics Forecast Error (Scale-Normalized nMAE)", fontsize=12, fontweight="bold", pad=10)
        ax_d.set_xlabel("Ablated Input Channel", fontsize=10, fontweight="bold")
        ax_d.set_ylabel("Overall nMAE", fontsize=10, fontweight="bold")
        plt.xticks(rotation=25, ha="right", fontsize=9)
        plt.tight_layout()
        path_d = os.path.join(output_dir, "plot_D_loco_dynamics_error.png")
        fig_d.savefig(path_d, dpi=300, bbox_inches="tight")
        generated_figures["plot_D_loco_dynamics_error"] = fig_d

    # -------------------------------------------------------------------------
    # Plot E: Signal vs. Exploitation Diagnostic Quadrant Frontier
    # -------------------------------------------------------------------------
    if diagnostic_df is not None and not diagnostic_df.empty:
        fig_e, ax_e = plt.subplots(figsize=(8.5, 6), dpi=300)
        df_plot = diagnostic_df[diagnostic_df["Ablation Channel"] != "None (Baseline)"].copy()

        cat_palette = {
            "CRITICAL PHYSICAL SIGNAL": "#2ca02c",
            "EXPLOITATION DETECTED": "#d62728",
            "PASSIVE EXOGENOUS": "#1f77b4",
            "INFORMATIVE": "#ff7f0e",
        }

        ax_e.axvline(2.0, color="gray", linestyle="--", alpha=0.6)
        ax_e.axhline(5.0, color="gray", linestyle="--", alpha=0.6)

        for _, row in df_plot.iterrows():
            x = row["Delta_nMAE (%)"]
            y = row["Delta_Cost (%)"]
            label = row["Ablation Channel"]
            classification = row.get("Diagnostic Classification", "INFORMATIVE")

            color = next((v for k, v in cat_palette.items() if k in classification.upper()), "#7f7f7f")
            ax_e.scatter(x, y, color=color, s=140, edgecolors="black", linewidth=1.2, zorder=3)
            ax_e.annotate(label, (x, y), textcoords="offset points", xytext=(8, 4), fontweight="bold", fontsize=9)

        ax_e.text(0.98, 0.98, "CRITICAL PHYSICAL SIGNAL\n(High ΔnMAE, High ΔCost)", transform=ax_e.transAxes, ha="right", va="top", fontsize=8, color="#2ca02c", fontweight="bold")
        ax_e.text(0.02, 0.98, "MODEL EXPLOITATION\n(Low ΔnMAE, High ΔCost)", transform=ax_e.transAxes, ha="left", va="top", fontsize=8, color="#d62728", fontweight="bold")
        ax_e.text(0.98, 0.02, "PASSIVE EXOGENOUS\n(High ΔnMAE, Low ΔCost)", transform=ax_e.transAxes, ha="right", va="bottom", fontsize=8, color="#1f77b4", fontweight="bold")

        ax_e.set_title("Signal vs. Artifact Exploitation Diagnostic Frontier", fontsize=12, fontweight="bold", pad=12)
        ax_e.set_xlabel("Dynamics Prediction Degradation $\\Delta$nMAE (%)", fontsize=10, fontweight="bold")
        ax_e.set_ylabel("Downstream Real Policy Degradation $\\Delta$Cost (%)", fontsize=10, fontweight="bold")
        ax_e.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        path_e = os.path.join(output_dir, "plot_E_exploitation_frontier.png")
        fig_e.savefig(path_e, dpi=300, bbox_inches="tight")
        generated_figures["plot_E_exploitation_frontier"] = fig_e

    print(f"[Visualization Engine] All {len(generated_figures)} diagnostic figures generated and saved to {output_dir}")
    return generated_figures


# =============================================================================
# 8. Argument Parsing & Main Orchestrator
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="TSFM Input Channel Attribution, LOCO Ablation & Action Sweep Suite")
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps per PPO ablation training")
    parser.add_argument("--context-length", type=int, default=16, help="TSFM historical context window (K)")
    parser.add_argument("--model-name", type=str, default="amazon/chronos-2")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--train-schema", type=str, default="citylearn_challenge_2023_phase_1")
    parser.add_argument("--eval-schema", type=str, default="citylearn_challenge_2023_phase_1")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation and training random seed")
    parser.add_argument(
        "--ablation-channels",
        type=lambda s: [item.strip() for item in s.replace(",", " ").split() if item.strip()],
        default=[
            "None",
            "weather",
            "indoor_thermal",
            "price_carbon",
            "occupancy_load",
            "storage_soc",
            "past_actions",
            "current_action",
        ],
        help="Comma- or space-separated candidate ablation targets",
    )
    parser.add_argument(
        "--action-sweep-values",
        type=lambda s: [float(item.strip()) for item in s.replace(",", " ").split() if item.strip()],
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
        help="Comma-separated actions to sweep on future_df",
    )
    parser.add_argument("--action-sweep-samples", type=int, default=15, help="Synchronized sample steps for real vs. TSFM action sweeps")
    parser.add_argument("--target-building-leakage-idx", type=int, default=0, help="Building index to perturb for spatial attention audit")
    parser.add_argument("--skip-rl-retraining", action="store_true", help="Run only dynamics ablation and action sweeps")
    parser.add_argument("--output-dir", type=str, default="./results_causal_attribution")

    # W&B Configuration
    parser.add_argument("--wandb-project", type=str, default="CityLearn-TSFM-Causal")
    parser.add_argument("--wandb-group", type=str, default="input-attribution-loco")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable. Falling back to CPU.")
        device = "cpu"

    set_random_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_run_name or f"causal_loco_seed_{args.seed}",
        config=vars(args),
    )

    # 1. Preload Chronos Pipeline
    print(f"\n[Chronos Setup] Pre-loading TSFM Pipeline ({args.model_name}) on {device}...")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_name, device_map=device)

    # 2. Collect Reference Dataset & Marginal Statistics
    dataset = collect_offline_data_and_statistics(schema=args.train_schema, num_steps=2000, seed=args.seed)

    # 3. Leave-One-Channel-Out Dynamics Fidelity Benchmark
    print("\n============================================================")
    print("      PHASE 1: LEAVE-ONE-CHANNEL-OUT DYNAMICS BENCHMARK     ")
    print("============================================================")
    dynamics_records = []
    for ab_chan in args.ablation_channels:
        print(f"Evaluating Dynamics Forecast under Ablation: [{ab_chan}]...")
        ab_name = None if ab_chan.lower() == "none" else ab_chan
        metrics = evaluate_dynamics_fidelity(
            pipeline=pipeline,
            dataset=dataset,
            context_length=args.context_length,
            ablation_channel=ab_name,
            eval_steps=720,
        )
        dynamics_records.append(metrics)

    dynamics_df = pd.DataFrame(dynamics_records)
    print("\n--- Dynamics Error Metrics (nMAE / MAE / RMSE / CRPS) ---")
    print(dynamics_df[["Ablation Channel", "Overall nMAE (Scale-Normalized)", "Overall MAE", "Overall RMSE", "Overall CRPS"]].to_string(index=False))
    wandb.log({"loco/dynamics_fidelity_table": wandb.Table(dataframe=dynamics_df)})

    # 4. Subsystem-Isolated Action Sweeps [Fixed 4.1]
    device_sweep_results, test_act_vals = evaluate_isolated_device_action_sweep(
        pipeline=pipeline,
        dataset=dataset,
        context_length=args.context_length,
        num_samples=args.action_sweep_samples,
        test_action_values=args.action_sweep_values,
    )
    for dev_name, res in device_sweep_results.items():
        print(f"\n--- Isolated Action Sensitivity & Real Comparison: [{dev_name}] ---")
        print(res["summary_df"].head(100).to_string())
        wandb.log({f"action_sweep/isolated_{dev_name.lower()}_table": wandb.Table(dataframe=res["summary_df"].reset_index())})

    # 5. Cross-Building Spatial Attention Leakage Probe [Fixed 4.1]
    leakage_df, tsfm_spatial_mat, real_spatial_mat, own_indices, other_indices = evaluate_cross_building_spatial_leakage(
        pipeline=pipeline,
        dataset=dataset,
        context_length=args.context_length,
        num_samples=args.action_sweep_samples,
        target_building_idx=args.target_building_leakage_idx,
        test_action_values=args.action_sweep_values,
    )
    wandb.log({"spatial_leakage/isolation_report": wandb.Table(dataframe=leakage_df)})
    spatial_leakage_bundle = (leakage_df, tsfm_spatial_mat, real_spatial_mat, own_indices, other_indices, args.target_building_leakage_idx)

    # 6. Downstream PPO Policy Retraining & Sim-to-Real Evaluation
    rl_df = None
    diagnostic_df = None
    if not args.skip_rl_retraining:
        clean_ab_channels = [None if c.lower() == "none" else c for c in args.ablation_channels]
        rl_df = run_downstream_ppo_ablation(
            args=args,
            pipeline=pipeline,
            marginal_means=dataset["marginal_means"],
            ablation_channels=clean_ab_channels,
            seed=args.seed,
        )
        wandb.log({"loco/downstream_rl_performance": wandb.Table(dataframe=rl_df)})

        diagnostic_df = analyze_exploitation_vs_signal(dynamics_df, rl_df)
        print("\n============================================================")
        print("      DIAGNOSTIC ANALYSIS: SIGNAL VS. ARTIFACT EXPLOITATION  ")
        print("============================================================")
        print(diagnostic_df[["Ablation Channel", "Delta_nMAE (%)", "Delta_Cost (%)", "Diagnostic Classification"]].to_string(index=False))
        wandb.log({"diagnostic/exploitation_summary": wandb.Table(dataframe=diagnostic_df)})

    # 7. Generate Diagnostic Visualizations & Upload to W&B
    print("\n============================================================")
    print("      PHASE 4: RENDERING DIAGNOSTIC VISUALIZATION SUITE     ")
    print("============================================================")
    generated_figures = generate_all_experiment_visualizations(
        target_cols=dataset["target_cols"],
        feature_cols=dataset["all_cols"],
        dynamics_df=dynamics_df,
        rl_df=rl_df,
        diagnostic_df=diagnostic_df,
        device_sweep_results=device_sweep_results,
        spatial_leakage_bundle=spatial_leakage_bundle,
        test_action_vals=test_act_vals,
        output_dir=vis_dir,
    )

    for fig_name, fig_obj in generated_figures.items():
        wandb.log({f"visualizations/{fig_name}": wandb.Image(fig_obj)})
        plt.close(fig_obj)

    print("\n[Complete] All causal experiments and visualizations successfully finished and logged to W&B.")
    wandb.finish()


if __name__ == "__main__":
    main()