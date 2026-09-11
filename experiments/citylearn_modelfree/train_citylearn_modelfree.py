import os
import argparse
import numpy as np
import pandas as pd
import torch
import gymnasium as gym

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper


def make_citylearn_env(schema: str, seed: int = None) -> gym.Env:
    """Instantiates and wraps CityLearn with standard normalization and SB3 interfaces."""
    env = CityLearnEnv(schema, central_agent=True)
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Seed Model-Free PPO Training and Annual Deployment on CityLearn"
    )
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps per seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--train-schema", type=str, default="citylearn_challenge_2023_phase_1")
    parser.add_argument("--eval-schema", type=str, default="citylearn_challenge_2023_phase_1")
    parser.add_argument(
        "--seeds",
        type=lambda s: [int(item.strip()) for item in s.replace(",", " ").split() if item.strip()],
        default=[42, 123, 456, 789, 2026],
        help="Comma or space-separated training/eval seeds (e.g., '42,123,456')",
    )
    parser.add_argument("--output-dir", type=str, default="./results_modelfree")

    # W&B Configuration
    parser.add_argument("--wandb-project", type=str, default="CityLearn-ModelFree-PPO")
    parser.add_argument("--wandb-group", type=str, default="modelfree-multiseed-annual")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def run_multiseed_model_free_ppo(args, device: str, group_name: str):
    seed_tables_dict = {}

    for seed in args.seeds:
        print("\n" + "=" * 60)
        print(f"       STARTING MODEL-FREE PPO RUN: SEED {seed} (Group: {group_name})")
        print("=" * 60)

        # Initialize dedicated W&B run per seed
        run_name = f"seed_{seed}"
        if args.wandb_run_name:
            run_name = f"{args.wandb_run_name}_seed_{seed}"

        run = wandb.init(
            project=args.wandb_project,
            group=group_name,
            name=run_name,
            config=dict(vars(args), current_seed=seed),
            sync_tensorboard=True,
            reinit=True,
        )

        # Hard reset random seeds across all libraries
        set_random_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate fresh real simulator training environment
        train_env = make_citylearn_env(args.train_schema, seed=seed)

        tb_log_dir = f"runs/{run.id}_seed_{seed}"
        ppo_model = PPO(
            policy="MlpPolicy",
            env=train_env,
            device=device if device == "cuda" else "cpu",
            seed=seed,
            verbose=1,
            tensorboard_log=tb_log_dir,
        )

        wandb_callback = WandbCallback(
            gradient_save_freq=0,
            verbose=2,
        )

        print(f"Training Model-Free PPO policy for {args.timesteps} steps on ground-truth simulator...")
        ppo_model.learn(
            total_timesteps=args.timesteps,
            callback=wandb_callback,
            progress_bar=False,
        )

        checkpoint_path = os.path.join(args.output_dir, f"ppo_modelfree_seed_{seed}.zip")
        ppo_model.save(checkpoint_path)

        print(f"\nDeploying Seed {seed} Policy on Full-Horizon Evaluation Simulator...")
        eval_env = make_citylearn_env(args.eval_schema, seed=seed)
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

        raw_kpis = eval_env.unwrapped.evaluate()
        if "cost_function" in raw_kpis.columns and "name" in raw_kpis.columns and "value" in raw_kpis.columns:
            pivoted_kpi = raw_kpis.pivot(index="cost_function", columns="name", values="value").astype(float)
        else:
            pivoted_kpi = raw_kpis.copy().astype(float)

        seed_tables_dict[seed] = pivoted_kpi

        # Log individual seed table to this seed's run
        wandb.log({f"eval_tables/seed_{seed}_kpis": wandb.Table(dataframe=pivoted_kpi.reset_index())})

        # Log district metrics into summary for W&B group aggregations
        if "District" in pivoted_kpi.columns:
            for cost_fn, val in pivoted_kpi["District"].items():
                wandb.summary[f"eval_district/{cost_fn}"] = val

        run.finish()

    # This section was fixed for Pandas 2.0 / 2.1 / 2.2+
    combined_df = pd.concat(
        seed_tables_dict.values(),
        keys=seed_tables_dict.keys(),
        names=["seed", "cost_function"],
    )

    grouped = combined_df.groupby("cost_function")
    mean_df = grouped.mean()
    std_df = grouped.std(ddof=1).fillna(0.0)

    mean_long = mean_df.reset_index().melt(
        id_vars="cost_function", var_name="entity", value_name="Mean (μ)"
    )
    std_long = std_df.reset_index().melt(
        id_vars="cost_function", var_name="entity", value_name="Std (σ)"
    )

    summary_df = pd.merge(mean_long, std_long, on=["cost_function", "entity"])
    summary_df["Mean (μ)"] = summary_df["Mean (μ)"].round(4)
    summary_df["Std (σ)"] = summary_df["Std (σ)"].round(4)
    summary_df["μ ± σ"] = summary_df.apply(
        lambda row: f"{row['Mean (μ)']:.4f} ± {row['Std (σ)']:.4f}"
        if pd.notnull(row["Mean (μ)"])
        else "N/A",
        axis=1,
    )

    summary_df = summary_df[pd.notnull(summary_df["Mean (μ)"])].reset_index(drop=True)
    summary_csv_path = os.path.join(args.output_dir, "modelfree_multiseed_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved aggregate summary table to {summary_csv_path}")

    return seed_tables_dict, summary_df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable. Falling back to CPU.")
        device = "cpu"

    group_name = args.wandb_group or f"modelfree-multiseed-{os.getpid()}"

    seed_tables, summary_df = run_multiseed_model_free_ppo(args, device, group_name)

    print("\n" + "=" * 60)
    print("      MODEL-FREE PPO MULTI-SEED ANNUAL AGGREGATE RESULTS (μ ± σ)")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)

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