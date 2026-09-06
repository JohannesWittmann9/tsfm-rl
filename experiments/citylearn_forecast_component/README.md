# CityLearn forecast component

Answers section 3.2 of the report ("TSFM as a Forecasting Component"): does
Chronos-2 zero-shot forecasting compete with classical baselines on real
CityLearn time series, and how does that compare to section 3.1's result on
synthetic control-environment dynamics?

Pure forecasting, no RL, no battery model, no environment stepping. Three
series from `citylearn_challenge_2022_phase_1` (Building 1 + the shared
district price): `non_shiftable_load` (noisy, occupant-driven),
`solar_generation` (weather-driven, strongly periodic), `electricity_pricing`
(exactly periodic). 24-hour-ahead forecasts, scored by NMSE per horizon step
relative to persistence (same convention as `experiments/dyna_standard`),
averaged over the horizon, on 48 fixed evaluation windows drawn from the last
20% of the year.

Four models: seasonal-naive (24h persistence, budget-independent), Chronos-2
small zero-shot on levels, Chronos-2 small zero-shot differenced (the
transform that helped most in section 3.1), and a trained MLP baseline
(direct 24-hour regression from a 24-hour lookback, budget = training-set
size). Swept over context budgets N in {16, 64, 256, 1024}.

## Run

```bash
uv run python run.py    # ~3 min on CPU, writes results/results.csv
uv run python plot.py   # writes the two figures and the table below
```

## Result

At the largest budget, Chronos-2 on levels beats both baselines on all three
series (Load 0.433 vs. MLP 0.606 vs. seasonal-naive 0.797; Solar 0.077 vs.
0.094 vs. 0.148; Price 0.012 vs. 0.094 vs. 0.076). Differencing, which was
essential in section 3.1, *hurts* here throughout: these series are already
periodic rather than inertia-dominated, so differencing removes the very
pattern Chronos can match against, instead of removing noise. The MLP is
unstable below roughly 200 training windows (Price at N=64 explodes to
NMSE > 1e10, a single unstable fit for a 24-to-24 regression on 14 training
examples), which the plot clips to the shared y-axis band and marks with a
triangle, following `dyna_standard`'s convention.
