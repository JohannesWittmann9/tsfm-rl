# dyna_standard, in detail

What the study measures, how the numbers are produced, and which knob in
`config.py` moves which part of it.

---

## 1. The data

Three separate draws, from three different seeds. Nothing is shared between them
except the environment and the policy.

| draw | episodes | length | seed | used by |
|---|---|---|---|---|
| **short** | `N_EPISODES` = 50 | `EPISODE_LEN` = 200 | 0 | figures 1, 1b, 4 and the §2b probe |
| **evaluation** | `HP_EVAL_EPISODES` = 48 | `cap` (below) | 200 | every window every model is scored on |
| **fit pool** | `HP_POOL_EPISODES` = 4 | `cap` | 100 | the transitions MLP and VARX are fitted on |

The fit pool is **not** the training-set size -- the budget N is. `take_transitions`
hands a trained model the first N transitions of the pool's *first* episode, and
since the largest budget (8192) is smaller than one episode (8224), the other three
episodes are never touched. They are headroom for a budget that outgrows an
episode, nothing more. Fitting on one contiguous trajectory rather than several
short ones is also the better choice here: split N evenly over four episodes and
every baseline gets worse, VARX by up to 40x (more episode seams, and a seam is a
row pairing the end of one trajectory with the start of the next).

Actions are uniform-random, with one exception: CartPole falls over in ~25 steps
under random actions, so it is collected under an ε-greedy stabiliser (`GAIN`,
`EPS = 0.15` in [cartpole/env.py](cartpole/env.py)). Roughly 15% of its actions
are pure noise, the rest are linear feedback. That keeps the pole up for thousands
of steps, at the price that CartPole's action is partly a function of its state
and its states never leave a small neighbourhood of upright. Read every CartPole
number with that in mind — see the FAQ.

Episodes that terminate early are **dropped and replaced by the next seed**, so no
window ever straddles an episode boundary. That is also why episode length is
measured rather than chosen: `longest_rollouts` asks for
`max(SCALE_BUDGETS) + HP_MARGIN` = 8224 steps and halves the request until every
episode survives.

| env | cap | ctx = cap − 32 | largest budget it can serve |
|---|---|---|---|
| Pendulum-v1 | 8224 | 8192 | 8192 |
| MountainCar-v0 | 8224 | 8192 | 8192 |
| CartPole-v1 | 8224 | 8192 | 8192 |
| Acrobot-v1 *(disabled in `envs.py`)* | 4112 | 4080 | 2048 |

`HP_MARGIN` = 32 is headroom between the episode and the context, so a window has
somewhere to start. Budgets larger than `ctx` are dropped from the sweep rather
than run at a smaller size, which is why Acrobot has no 4096 column anywhere.

## 2. Evaluation windows

A window is `(context states, context actions, future actions) -> future states`,
with this study's alignment throughout: **`ctx_a[:, t]` is the action that
produced `ctx_s[:, t]`**, and `fut_a[:, 0]` drives the first predicted step.

| used by | context length | horizon | windows |
|---|---|---|---|
| §2b probe | `L` = 64 | 1 | `PROBE_WINDOWS` = 96, averaged over `PROBE_CTX` = 96 contexts |
| grid (§5), scaling (§6) | `ctx` | 1 | `HP_WINDOWS` = 96 |
| rollout (§7), trajectories (§8) | `ctx − ROLL_H` | `ROLL_H` = 20 | `ROLL_WINDOWS` = 96 |

All windows are drawn once, with `SEED` = 0, and every model and every budget sees
**the same ones**. The one-step and the 20-step sets are different draws, though —
they have to be, the 20-step windows need 19 more steps of future. Two figures
reading the same budget are not reading the same windows.

**The metric.** NMSE, normalised per channel by the mean square of the one-step
increment, so predicting "no change" scores exactly 1.0 in every environment. The
rollout normalises per horizon instead, by the mean square of `s(t+h) − s(t)`, so
persistence sits at 1.0 at every `h`. Errors are a mean of squares over windows,
which means one bad window out of 96 can carry a panel — see the FAQ.

**The budget N** means the same thing on both sides of the comparison: context
steps for a foundation model, fitted transitions for a trained one.
`take_transitions` delivers exactly N, and `study()` asserts it.

## 3. The models

| model | what it is |
|---|---|
| `Chronos-2 S` | `autogluon/chronos-2-small`, 27.9M, action as past + known-future covariate, median quantile |
| `Chronos-2 L` | `amazon/chronos-2`, 119.5M |
| `Chronos-2 L-syn` | `autogluon/chronos-2-synth`, 119.0M, synthetic pretraining |
| `Chronos-2 S (level)` | the small model with **both transforms off** — the out-of-the-box reference. Pinned by `fixed=`, never swept |
| `Moirai` | `Salesforce/moirai-1.1-R-small`, median over `MOIRAI_SAMPLES` = 10 draws |
| `MLP` | 3-layer (128/64), Adam, early stopping, trained on the increment |
| `VARX` | statsmodels `VAR` with the action as `exog`, same implementation as `notebooks/pendulum` |

### The two transforms

**Differencing** hands the integrated channels over as increments and re-integrates
after. Which channels is per environment (`difference` in `<env>/env.py`): the
velocities and angular rates, never the bounded trig channels. Chronos conditions
covariates on the *level* of a series, so an integrated channel passed as a level
is dominated by autoregressive continuation and the action gets shrunk away —
that is what §2b's first two columns show.

**Upsampling** (stretching) inserts `r − 1` interpolated steps between real steps,
holds the action across them (a zero-order hold, so each sub-step increment shrinks
by the same factor), forecasts `r·H` sub-steps and keeps every r-th.

`r` is a **multiplier**: `r = 1` reproduces the input exactly and is the
untransformed case. There is no `r = 0`. The `r` column reads 0 only on MLP and
VARX rows, where stretching has no meaning.

### The variant grid

| model kind | variants swept |
|---|---|
| in `GRID_FULL` | `PRESENTATIONS` × `STRETCH_R` = {diff, level} × {1, 2, 4, 8, 16} = 10 |
| other TSFMs | the differenced sweep plus `level r=1` and `level r=DEFAULT_R` |
| `fixed=` models | exactly one, never swept |
| MLP, VARX | one per entry in `lags` = [1, 4, 16] |

Stretching does not count against the budget — N real samples become `(N−1)r + 1`
tokens, which is a processing choice, not more data — but it does consume context.
Combinations that would overflow `CHRONOS_CTX` / `MOIRAI_CTX` are **skipped** in
the grid, so a row labelled `r=16` is never quietly a different r. The sweeps in §6
and §7 instead fall back to the largest r that fits (`usable_r`) and mark where
that happened.

### Which variant carries into §6, §7 and §8

`SELECT_RULE = "mean_rank"`: each variant's rank across budgets, averaged, ties
broken on the geometric mean. Ranks rather than mean error because the errors span
six decades and a mean would be decided by the largest budget alone. The selection
is per environment, so **"Chronos-2 S" means a different configuration in each
panel** — the `variant` column of every CSV says which, and §5's table marks it.

## 4. The stages

Each writes one CSV per environment into `<env>/results/`. Every stage resumes.

| stage | what it computes | key columns |
|---|---|---|
| `probe` | §2b: same contexts, same probe actions, four presentations | `condition, model, r, action, response, slope_pct` |
| `grid` | §5: every model, every variant, every budget in `HP_BUDGETS` | `variant, presentation, r, lag, budget, nmse, seconds` |
| `scaling` | §6: the selected variant per model over `SCALE_BUDGETS` | `model, variant, budget, r, nmse` |
| `rollout` | §7: open loop to `ROLL_H` at each of `ROLL_BUDGETS` | `model, budget, r, h, nmse` |
| `traj` | §8: the predicted states themselves, `TRAJ_WINDOWS` windows at `TRAJ_BUDGET` | `model, window, h, channel, label, value` |

The **probe** holds each context fixed, sweeps the next action over the
environment's `probe_actions` grid, and records how far the predicted next value
moves on `probe_channel`. A flat curve means the action was ignored. `slope_pct` is
that slope as a percentage of the real environment's, stepped from the same states
so clipping and saturation are handled exactly rather than assumed away. The curve
itself is only meaningful up to an offset and is centred on its own mean.


## 5. Config reference

Everything below is `config.py`. The two groups are split by what changing them
costs.

### Costs a recompute

| knob | now | what it does |
|---|---|---|
| `MODELS` | 7 entries | the registry: kind, checkpoint, colour, marker, `lags`, `fixed` |
| `GRID_MODELS` | all 7 | which models §5 measures |
| `GRID_FULL` | all 5 TSFMs | which get the full presentation × r factorial |
| `PROBE_MODELS` | all 5 TSFMs | which models §2b probes |
| `PRESENTATIONS`, `STRETCH_R` | {diff, level}, [1, 2, 4, 8, 16] | the variant grid |
| `HP_BUDGETS` | [1, 2, 16, 64, 256, 1024, 4096] | §5's columns. 1 and 2 are the zero-shot end; most of those cells come out blank and what does not is the point |
| `SCALE_BUDGETS` | [1 … 8192] | §6's x axis. Also sets the context length, so changing it invalidates the grid |
| `ROLL_BUDGETS` | [64, 512, 1024, 2048, 4096] | which budgets §7 is computed at |
| `ROLL_H` | 20 | rollout horizon. Changing it re-stamps `rollout` and `traj` only -- the grid and scaling survive, since their windows do not depend on it |
| `TRAJ_BUDGET`, `TRAJ_WINDOWS` | 4096, 3 | §8's budget (clamped to what the env can serve) and how many windows it keeps |
| `HP_WINDOWS`, `ROLL_WINDOWS` | 96, 96 | evaluation windows. **The main cost lever** |
| `PROBE_WINDOWS`, `PROBE_CTX` | 96, 96 | the probe's window pool and how many contexts it averages |
| `N_EPISODES`, `EPISODE_LEN`, `L` | 50, 200, 64 | the short draw and the probe's context |
| `HP_EVAL_EPISODES`, `HP_POOL_EPISODES`, `HP_MARGIN` | 48, 4, 32 | the long draws |
| `SELECT_RULE` | `mean_rank` | how §5 picks the variant §6–§8 use |
| `MOIRAI_SAMPLES` | 20 | draws Moirai's median is taken over. Moirai is ~70% of the grid's runtime; this is the lever on it |
| `CHRONOS_BATCH`, `CHRONOS_CTX`, `MOIRAI_CTX` | 16, 8192, 5000 | batch size and context limits |
| `SEED` | 0 | every draw and every window sample |


Change one of these, re-run the notebook cell, done. Nothing recomputes. A model
left out is still measured and still in the CSV, it is only absent from the panel.

| knob | now | what it does |
|---|---|---|
| `PLOT_MODELS` | 6 models | default model list for §6 and §7 |
| `SCALING_MODELS` | `None` | §6's models; `None` follows `PLOT_MODELS` |
| `ROLLOUT_MODELS` | `None` | §7's models |
| `NMSE_YLIM` | `(1e-4, 2.0)` | the log band both are drawn in; curves outside are pinned to the edge and flagged with a triangle |
| `SCALING_YLIM`, `ROLLOUT_YLIM` | `None` | per-figure band; `None` follows `NMSE_YLIM` |
| `SCALE_PLOT_BUDGETS` | `None` | which budgets §6 draws; `None` is all of them |
| `ROLL_PLOT_BUDGETS` | `[64, 512, 2048]` | which budgets §7 draws — it spends a row per budget, so the sweep is wider than the figure |
| `ROLL_PLOT_H` | `None` | how far along the horizon §7 and §8 draw; `None` is the whole computed `ROLL_H` |
| `STRETCH_BUDGET` | 256 | the budget §5's r figure is drawn at |
| `ILLUSTRATION_ENV` | Pendulum-v1 | which env figures 1b and 4 use |
| `SHORT`, colours, markers | per env / per model | from `<env>/env.py` and `MODELS` |

Environments come from `envs.py`'s hardcoded list, one folder each. Comment one
out and it leaves every figure and table; its CSVs stay on disk.
