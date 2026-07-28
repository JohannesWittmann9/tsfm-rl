# tsfm-rl

Time-series foundation models (TSFMs) for reinforcement learning.

We investigate whether TSFMs such as Chronos can serve as a world model or forecasting
component in RL, using [CityLearn](https://www.citylearn.net/), a Gymnasium environment
for building energy coordination and demand response, as the test scenario.

**Research questions**

- *TBD*

---

## Setup

Requires Python >= 3.10.

**1. Install uv**

Windows:

```powershell
winget install --id=astral-sh.uv -e
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone and sync**

```bash
git clone https://github.com/JohannesWittmann9/tsfm-rl.git
cd tsfm-rl
uv sync --locked
```

`uv sync --locked` creates the virtualenv, installs the exact versions from `uv.lock`,
and fails if the lockfile is out of date. That failure is intentional, it means someone
changed dependencies without committing the lockfile.

**3. Verify**

```bash
uv run pytest # TBD: Currently no tests
uv run python scripts/smoke.py # Minimal script to check setup (Maybe enhance?)
```

### Platform notes

- **Windows.** CityLearn pins `openstudio<=3.3.0`, which has no Windows wheels. We
  override it to `>=3.10.0` in `pyproject.toml`; openstudio is unused in our code path.

---

## Project structure

```
src/tsfmrl/
├── TBD
scripts/          entry points: collect_data.py, run_experiment.py, figures.py
tests/            maybe we will need some testing
notebooks/        exploration only, never imported by src/
docs/             decisions.md and design notes (TBD)
```

---

## Contributing

### Dependencies

- Add packages with `uv add <package>`
- Commit `pyproject.toml` **and** `uv.lock` together in the same PR.
- **Never `pip install` into the project venv.** It installs packages the lockfile
  doesn't record, the environment silently diverges, and results stop being reproducible.
- Prefix commands with `uv run` rather than activating the venv manually, e.g. `uv run pytest` or `uv run ruff check`.

### Code style

`ruff` handles linting and formatting, enforced by pre-commit and CI. Set it up once:

```bash
uv run pre-commit install
```

## Citation

CityLearn:

```bibtex
@article{doi:10.1080/19401493.2024.2418813,
   author = {Nweye, Kingsley and Kaspar, Kathryn and Buscemi, Giacomo and Fonseca, Tiago
             and Pinto, Giuseppe and Ghose, Dipanjan and Duddukuru, Satvik and Pratapa, Pavani
             and Li, Han and Mohammadi, Javad and Lino Ferreira, Luis and Hong, Tianzhen
             and Ouf, Mohamed and Capozzoli, Alfonso and Nagy, Zoltan},
   title = {CityLearn v2: energy-flexible, resilient, occupant-centric, and carbon-aware
            management of grid-interactive communities},
   journal = {Journal of Building Performance Simulation},
   volume = {0},
   number = {0},
   pages = {1--22},
   year = {2024},
   publisher = {Taylor \& Francis},
   doi = {10.1080/19401493.2024.2418813},
   url = {https://doi.org/10.1080/19401493.2024.2418813},
}
```