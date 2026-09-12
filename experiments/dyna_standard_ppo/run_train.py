"""Train one (environment, model, N, seed) cell of the sweep.

The notebook stays the single source of truth: this executes its setup cells and
then trains one cell, so there is no second copy of ModelEnv to drift out of
step. Used by scripts/slurm/dyna_ppo.slurm, one array task per cell.

    python experiments/dyna_standard_ppo/run_train.py --list
    python experiments/dyna_standard_ppo/run_train.py CartPole-v1 MLP 64 0
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "dyna_standard_ppo.ipynb"
SETUP_CELLS = 4  # imports, ModelEnv, grid, callback -- everything but the loops
NEEDED = (
    "PPO",
    "ENV_IDS",
    "N_VALUES",
    "MODEL_NAMES",
    "POLICY_SEEDS",
    "TOTAL_TIMESTEPS",
    "CHECK_FREQ",
    "AVG_EPISODES",
    "make_dynamics",
    "make_train_env",
    "run_dir",
    "SaveOnBestTrainingRewardCallback",
)


def setup():
    """Run the notebook's definition cells and hand back their namespace."""
    cells = [
        c
        for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    ]
    ns = {"__name__": "__main__"}
    for i in range(SETUP_CELLS):
        src = "".join(cells[i]["source"])
        # exec is the point: running the notebook's own cells is what keeps
        # this from becoming a second, drifting copy of ModelEnv.
        exec(compile(src, f"{NOTEBOOK.name}:cell{i}", "exec"), ns)  # noqa: S102
    missing = [n for n in NEEDED if n not in ns]
    if missing:
        raise RuntimeError(
            f"{NOTEBOOK.name}: first {SETUP_CELLS} code cells did not define "
            f"{missing} -- the notebook has been restructured, update SETUP_CELLS"
        )
    return ns


def cells(ns):
    """The sweep grid, in the order the array indexes it."""
    return [
        (env_id, model_name, n, seed)
        for env_id in ns["ENV_IDS"]
        for n in ns["N_VALUES"]
        for model_name in ns["MODEL_NAMES"]
        for seed in ns["POLICY_SEEDS"]
    ]


def train(ns, env_id, model_name, n, seed):
    log_dir = ns["run_dir"](env_id, model_name, n, seed)
    # The marker, not best_model.zip: the callback writes that on the first
    # improvement, so a run killed midway would otherwise look finished and be
    # skipped with a half-trained policy.
    if (log_dir / "done").exists():
        print(f"skip (already trained): {log_dir.name}", flush=True)
        return
    print(f"=== {env_id} | {model_name} | N={n} | seed={seed}", flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    dynamics, context_len = ns["make_dynamics"](env_id, model_name, n)
    env = ns["make_train_env"](env_id, dynamics, context_len, seed, log_dir)
    agent = ns["PPO"]("MlpPolicy", env, verbose=0, seed=seed)
    agent.learn(
        total_timesteps=ns["TOTAL_TIMESTEPS"],
        callback=ns["SaveOnBestTrainingRewardCallback"](
            check_freq=ns["CHECK_FREQ"],
            log_dir=log_dir,
            avg_episodes=ns["AVG_EPISODES"],
            vec_norm_env=env,
        ),
    )
    env.close()
    (log_dir / "done").touch()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("env", nargs="?", help="e.g. CartPole-v1")
    p.add_argument("model", nargs="?", help="e.g. 'Chronos-2 S (diff, r=6)'")
    p.add_argument("n", nargs="?", type=int, help="the budget N")
    p.add_argument("seed", nargs="?", type=int, default=0)
    p.add_argument(
        "--list",
        action="store_true",
        help="print the grid as env|model|N|seed, one cell per line",
    )
    p.add_argument("--timesteps", type=int, help="override TOTAL_TIMESTEPS")
    args = p.parse_args()

    ns = setup()
    if args.list:
        for env_id, model_name, n, seed in cells(ns):
            print(f"{env_id}|{model_name}|{n}|{seed}")
        return
    if not (args.env and args.model and args.n):
        p.error("give env, model and N -- or --list")
    if args.timesteps:
        ns["TOTAL_TIMESTEPS"] = args.timesteps
    train(ns, args.env, args.model, args.n, args.seed)


if __name__ == "__main__":
    main()
