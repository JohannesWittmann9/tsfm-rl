"""Forecast features for the single-building forecast experiment (forecast.ipynb).

Rolling-origin Chronos-2 forecasts for Building_1's 7 exogenous series: at
every hour t of the year the model sees history up to t (max 512 h, no
leakage) and predicts the next 24 h for all series jointly. The 24-step
median paths are reduced to point features at 6/12/24 h plus, for the demand
series (load, solar), cumulative volumes over the next 6/12/24 h.

    python forecast_chronos.py    # ~10-20 min on CPU, writes data/forecast_chronos_b1.csv

The seasonal-naive and oracle reference tables are cheap and deterministic;
build them on demand via shifted_paths() + build_table().
"""

import time

import common
import numpy as np
import pandas as pd

FORECAST_CSV = common.DATA_DIR / "forecast_chronos_b1.csv"
FC_SERIES = ["load", "solar", "price", "temperature", "humidity", "diffuse", "direct"]
HORIZONS = (6, 12, 24)
VOLUME_SERIES = ("load", "solar")
CONTEXT = 512
PV_KW = 4.0  # schema: Building_1 pv nominal_power; the CSV stores W per kW installed


def load_b1_series() -> pd.DataFrame:
    """The 7 exogenous series for Building_1, one year hourly, kWh units."""
    b1 = pd.read_csv(common.SOURCE_DIR / "Building_1.csv")
    weather = pd.read_csv(common.SOURCE_DIR / "weather.csv")
    return pd.DataFrame(
        {
            "load": b1["non_shiftable_load"],
            "solar": b1["solar_generation"] * PV_KW / 1000.0,
            "price": pd.read_csv(common.SOURCE_DIR / "pricing.csv")[
                "electricity_pricing"
            ],
            "temperature": weather["outdoor_dry_bulb_temperature"],
            "humidity": weather["outdoor_relative_humidity"],
            "diffuse": weather["diffuse_solar_irradiance"],
            "direct": weather["direct_solar_irradiance"],
        }
    )


def build_table(paths: dict[str, np.ndarray]) -> pd.DataFrame:
    """paths: {series: (T, 24) forecast paths}. Extracts the point features at
    6/12/24 h and, for the demand series, the cumulative volumes."""
    data = {}
    for name, p in paths.items():
        for h in HORIZONS:
            data[f"{name}_h{h}"] = p[:, h - 1]
        if name in VOLUME_SERIES:
            csum = p.cumsum(axis=1)
            for h in HORIZONS:
                data[f"{name}_sum{h}"] = csum[:, h - 1]
    return pd.DataFrame(data)


def shifted_paths(series_df: pd.DataFrame, kind: str) -> dict[str, np.ndarray]:
    """Reference forecast paths. oracle: the realized future values (wrapping
    at the year boundary). naive: the value at the same hour yesterday."""
    assert kind in ("oracle", "naive")
    n = len(series_df)
    idx = np.arange(n)[:, None] + np.arange(1, 25)[None, :]  # target times t+1 .. t+24
    src = idx % n if kind == "oracle" else (idx - 24) % n
    return {name: series_df[name].to_numpy(np.float32)[src] for name in FC_SERIES}


def chronos_paths(
    series_df: pd.DataFrame, batch_size: int = 256
) -> dict[str, np.ndarray]:
    """Rolling-origin median forecast paths for all series, (T, 24) each."""
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        "autogluon/chronos-2-small", device_map="cpu"
    )
    values = series_df[FC_SERIES].to_numpy(np.float32).T  # (7, T)
    n = values.shape[1]
    tasks = [  # history up to and including t, never beyond: no leakage
        {"target": np.ascontiguousarray(values[:, max(0, t + 1 - CONTEXT) : t + 1])}
        for t in range(n)
    ]
    median = pipeline.quantiles.index(0.5)
    paths = np.empty((n, len(FC_SERIES), 24), np.float32)
    t0 = time.time()
    for start in range(0, n, batch_size):
        out = pipeline.predict(tasks[start : start + batch_size], prediction_length=24)
        for i, q in enumerate(out):  # q: (series, quantiles, 24)
            paths[start + i] = q[:, median, :].numpy()
        if start % 2048 == 0:
            print(
                f"  {min(start + batch_size, n)}/{n} origins, {time.time() - t0:.0f}s",
                flush=True,
            )
    return {name: paths[:, i, :] for i, name in enumerate(FC_SERIES)}


def main():
    if FORECAST_CSV.exists():
        print(f"{FORECAST_CSV} exists, nothing to do")
        return
    table = build_table(chronos_paths(load_b1_series()))
    common.DATA_DIR.mkdir(exist_ok=True)
    table.to_csv(FORECAST_CSV)
    print(f"wrote {FORECAST_CSV}: {table.shape[0]} origins x {table.shape[1]} features")


if __name__ == "__main__":
    main()
