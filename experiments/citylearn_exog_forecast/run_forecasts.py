"""Step 1: precompute the Chronos forecast table and document forecast quality.

Writes
    data/chronos_forecasts.parquet   rolling forecasts, 8760 rows x 48 columns
    results/forecast_metrics.csv     MAE/RMSE per series/horizon/model
    figs/forecast_week.png           actual vs forecast, one held-out week
    figs/forecast_error.png          normalized error by horizon and model
"""

import time

import common
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    series = common.load_series()

    # the dataset's own predicted columns are exactly the future values (wrapping at the year boundary)
    stock = pd.concat(
        [
            pd.read_csv(common.SOURCE_DIR / "weather.csv"),
            pd.read_csv(common.SOURCE_DIR / "pricing.csv"),
        ],
        axis=1,
    )
    for s in common.WEATHER_SERIES + common.PRICE_SERIES:
        for k, h in common.HORIZONS.items():
            assert np.allclose(stock[f"{s}_predicted_{k}"], np.roll(stock[s], -h)), (
                s,
                k,
            )

    if common.FORECAST_PARQUET.exists():
        forecasts = pd.read_parquet(common.FORECAST_PARQUET)
    else:
        print(
            f"computing rolling Chronos forecasts for all {common.N_TIME_STEPS} origins ..."
        )
        t0 = time.time()
        forecasts = common.rolling_chronos_forecasts(series, common.load_chronos())
        common.save_forecasts(forecasts)
        print(f"saved {common.FORECAST_PARQUET} after {time.time() - t0:.0f}s")

    metrics = []
    for name, table in [
        ("chronos", forecasts),
        ("seasonal_naive", common.naive_forecasts(series, "seasonal")),
        ("persistence", common.naive_forecasts(series, "persistence")),
    ]:
        m = common.forecast_metrics(table, series)
        m.insert(0, "model", name)
        metrics.append(m)
    metrics = pd.concat(metrics, ignore_index=True)
    common.RESULTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(common.RESULTS_DIR / "forecast_metrics.csv", index=False)

    tbl = metrics.pivot_table(
        index=["series", "horizon_h"], columns="model", values="mae"
    )
    tbl["skill_vs_snaive"] = 1 - tbl["chronos"] / tbl["seasonal_naive"]
    print("\nMAE per series and horizon (skill > 0 beats seasonal-naive):")
    print(
        tbl.round(3)[
            ["chronos", "seasonal_naive", "persistence", "skill_vs_snaive"]
        ].to_string()
    )

    plot_series = [
        "outdoor_dry_bulb_temperature",
        "direct_solar_irradiance",
        "electricity_pricing",
        "carbon_intensity",
        "Building_1:non_shiftable_load",
        "Building_1:solar_generation",
    ]
    t0, t1 = common.EVAL_BLOCKS[8][0], common.EVAL_BLOCKS[8][1] + 1
    t = np.arange(t0, t1)
    fig, axes = plt.subplots(len(plot_series), 1, figsize=(10, 13), sharex=True)
    for ax, s in zip(axes, plot_series):
        ax.plot(t, series[s].iloc[t0:t1], color="black", lw=1.4, label="actual")
        for k, h in common.HORIZONS.items():
            pred = forecasts[f"{s}_predicted_{k}"].to_numpy()
            ax.plot(t, pred[t - h], lw=1.2, alpha=0.85, label=f"forecast h={h}")
        ax.set_ylabel(s.replace("_", " ").replace(":", "\n"), fontsize=8)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("hour of year")
    fig.suptitle("Chronos-2-small rolling forecasts, one held-out week", y=0.995)
    common.savefig("forecast_week")

    scale = series.std()
    metrics["nmae"] = metrics.apply(lambda r: r["mae"] / scale[r["series"]], axis=1)
    piv = metrics.pivot_table(index="horizon_h", columns="model", values="nmae")
    piv[["chronos", "seasonal_naive", "persistence"]].plot.bar(rot=0, figsize=(6, 4))
    plt.ylabel("MAE / series std (mean over 16 series)")
    plt.xlabel("horizon (h)")
    common.savefig("forecast_error")


if __name__ == "__main__":
    main()
