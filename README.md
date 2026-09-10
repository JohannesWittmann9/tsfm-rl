# Time Series Foundation Models in Reinforcement Learning

This project studies how time-series foundation models can be used in reinforcement learning.
Chronos and Moirai are evaluated as dynamics models for control tasks and as forecasting components for building energy management with [CityLearn](https://www.citylearn.net/).
The experiments compare them with smaller task-specific baselines such as MLP and VARX models.

## Setup

The project requires Git, Python 3.10 or newer, and [uv](https://docs.astral.sh/uv/).
Package dependencies are declared in `pyproject.toml`.
The exact resolved versions are stored in `uv.lock`.

Install `uv` on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, run `winget install --id=astral-sh.uv -e`.

Clone the repository and install all dependencies:

```bash
git clone https://github.com/JohannesWittmann9/tsfm-rl.git
cd tsfm-rl
uv sync --locked
```

`uv sync --locked` creates a local virtual environment and installs the versions from the lockfile.
Run the smoke test to check the CityLearn installation:

```bash
uv run python scripts/smoke.py
```

Model weights are downloaded automatically when Chronos or Moirai is used for the first time.
The initial run therefore requires an internet connection and may take longer.

## Running the experiments

Run these commands from the repository root.

### Dynamics models

Run the standard dynamics-model experiments on Pendulum, MountainCar, Acrobot, and CartPole:

```bash
uv run python experiments/dyna_standard/run_all.py
```

### CityLearn forecasting component

Run the CityLearn forecasting comparison:

```bash
cd experiments/citylearn_forecast_component
uv run python run.py
uv run python plot.py
```

### CityLearn exogenous forecasts

Generate the Chronos forecast features and train the CityLearn agents that use them:

```bash
cd experiments/citylearn_exog_forecast
uv run python forecast_chronos.py
uv run python forecast_train.py
```

`forecast_chronos.py` writes the forecast data used during training.
The trained models and learning curves are saved under `models/` and `results/`.
The evaluation and plots are available in `forecast.ipynb`.

Results are written to the corresponding experiment directories.
The README files inside `dyna_standard` and `citylearn_forecast_component` describe their available options and shorter test runs.

## Repository layout

```text
experiments/   experiment code, cached results, and figures
notebooks/     exploratory notebooks
scripts/       training, smoke-test, and cluster entry points
```

Use `uv run` for project commands so they run with the locked environment.
Run `uv run ruff check .` to check the Python code.