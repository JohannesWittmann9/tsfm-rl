"""Loads the three CityLearn series this experiment forecasts, and defines the
train/eval split.

Series come from ``citylearn_challenge_2022_phase_1``, Building 1: the district
price is shared by every building, so any one building's file carries it too.

- ``non_shiftable_load``  occupant electricity demand, noisy, weakly periodic
- ``solar_generation``    PV output, weather-driven, strongly periodic (daylight)
- ``electricity_pricing`` five-level tariff, exactly periodic

Train/eval is a single chronological split, not windows: the last 20% of the
year is held out for evaluation, the first 80% is what the MLP baseline trains
on. Chronos is zero-shot, so it never touches the train/eval distinction --
its context is just "however many hours precede the forecast origin", which
for the largest budget reaches back past the eval region's start into the
training period. That is real history, not leakage, since nothing is fit on it.
"""

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from citylearn.data import DataSet  # type: ignore[import-untyped]

SERIES = ("non_shiftable_load", "solar_generation", "electricity_pricing")
EVAL_FRACTION = 0.2


def _dataset_dir() -> Path:
    return Path(DataSet().get_dataset("citylearn_challenge_2022_phase_1")).parent


def load_series() -> dict[str, np.ndarray]:
    """One float32 array per series in :data:`SERIES`, aligned to the same hours."""
    d = _dataset_dir()
    building = pd.read_csv(d / "Building_1.csv")
    pricing = pd.read_csv(d / "pricing.csv")
    return {
        "non_shiftable_load": building["non_shiftable_load"].to_numpy(np.float32),
        "solar_generation": building["solar_generation"].to_numpy(np.float32),
        "electricity_pricing": pricing["electricity_pricing"].to_numpy(np.float32),
    }


def split_index(t: int) -> int:
    """The first index of the held-out evaluation region, for a series of length t."""
    return round(t * (1 - EVAL_FRACTION))


def eval_origins(t: int, h: int, n_windows: int, seed: int) -> np.ndarray:
    """``n_windows`` fixed forecast origins in the eval region.

    An origin ``o`` means: context ends at ``o`` (inclusive), the forecast covers
    ``o+1 .. o+h``. Same origins for every budget and every model, so a budget
    sweep changes only how far back the context reaches.
    """
    lo, hi = split_index(t), t - h - 1
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(lo, hi + 1), size=n_windows, replace=False))
