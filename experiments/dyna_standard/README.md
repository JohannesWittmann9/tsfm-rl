# dyna_standard

Chronos-2 and Moirai as dynamics models for control, against an MLP and a VARX,
on Pendulum, MountainCar, Acrobot and CartPole.

Scripts compute and cache to `<env>/results/*.csv`; the notebooks only plot.
All knobs — models, budgets, sweeps, which models the last two figures draw —
live in `config.py`. [README_DEV.md](README_DEV.md) explains the procedure and
every knob in it.

## Full Results
Full results of parameter sweeps are provided within the notebooks. Just run one or all experiments.

## Running it

```bash
uv sync                                                   # from the repo root
python experiments/dyna_standard/run_all.py               # every env, every stage
python experiments/dyna_standard/pendulum/pendulum.py     # a single env
python experiments/dyna_standard/run_all.py --stages grid # a single stage
```


Flags: `--envs --stages {probe,grid,rollout} --models --budgets
--windows --probe-ctx --results-dir --force`. A two-minute end-to-end check:

```bash
python experiments/dyna_standard/run_all.py --windows 4 --probe-ctx 4 \
    --budgets 16 64 256 --models "Chronos-2 S" MLP VARX --results-dir results_smoke
```


## Using Chronos-2 with differencing and upsampling elsewhere

```python
import sys

sys.path.insert(0, "<repo>/experiments/dyna_standard")
from models import ChronosDynamics, UpsampledDynamics

# ctx_s (B, L, n_obs)  context states, on levels
# ctx_a (B, L, 1)      ctx_a[:, t] is the action that PRODUCED ctx_s[:, t]
# fut_a (B, H, 1)      the actions to forecast under
model = UpsampledDynamics(
    ChronosDynamics(difference=(1,)),  # channel indices that are integrals
    factor=4,  # stretch factor r; 1 disables stretching
)
pred = model.predict(ctx_s, ctx_a, fut_a)  # (B, H, n_obs), on levels
```

* **Which channels.** Difference anything that is an integral of the dynamics;
  leave bounded channels ($\cos\theta$, a temperature) as levels. `difference=()`
  turns it off.

