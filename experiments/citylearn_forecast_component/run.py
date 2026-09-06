"""Runs the full sweep and writes ``results/results.csv``.

For each of the three series, four context budgets, and four models
(seasonal-naive, Chronos-2 S level, Chronos-2 S differenced, MLP), scores one
NMSE number -- averaged over the 24-hour forecast horizon -- on the same 48
fixed evaluation windows. Takes a few minutes on CPU.
"""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from data import eval_origins, load_series, split_index
from metric import nmse, persistence_scale, true_values
from models import chronos_forecast, mlp_forecast, seasonal_naive

BUDGETS = [16, 64, 256, 1024]
H = 24
N_WINDOWS = 48
SEED = 0
LAG = 24  # MLP's fixed lookback window (one day)
CHRONOS_MODEL = "autogluon/chronos-2-small"
RESULTS_CSV = "results/results.csv"


def main():
    series_dict = load_series()
    rows = []

    for name, series in series_dict.items():
        t = len(series)
        split_idx = split_index(t)
        origins = eval_origins(t, H, N_WINDOWS, SEED)
        scale_h = persistence_scale(series, origins, H)
        true = true_values(series, origins, H)

        def add(model, budget, pred, name=name, true=true, scale_h=scale_h):
            score = np.nan if pred is None else nmse(pred, true, scale_h)
            rows.append(
                {"series": name, "model": model, "budget": budget, "nmse": score}
            )
            print(f"{name:22s} {model:22s} N={budget:<5d} NMSE={score:.4f}")

        naive_pred = seasonal_naive(series, origins, H)
        for budget in BUDGETS:
            add("Seasonal-naive", budget, naive_pred)

        for budget in BUDGETS:
            pred = chronos_forecast(
                series, origins, H, budget, CHRONOS_MODEL, difference=False
            )
            add("Chronos-2 S (level)", budget, pred)

        for budget in BUDGETS:
            pred = chronos_forecast(
                series, origins, H, budget, CHRONOS_MODEL, difference=True
            )
            add("Chronos-2 S (diff)", budget, pred)

        for budget in BUDGETS:
            pred = mlp_forecast(
                series, split_idx, origins, H, budget, lag=LAG, seed=SEED
            )
            add("MLP", budget, pred)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
