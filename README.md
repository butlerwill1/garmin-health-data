# Garmin–Daylio Association Analysis

A private, association-first notebook project for combining Garmin daily measures with Daylio mood check-ins. It is designed to be safe to publish: the repository holds code and invented demo data only.

## Privacy boundary

Do not add personal exports, source folders, local configuration, derived tables, charts, or executed notebooks to Git. The loaders use allowlists: free text, names, identifiers, locations, and unapproved source fields never enter analysis tables. Daylio activity tags are analysed only when explicitly listed in the untracked local configuration.

The example configuration and synthetic demo are fictional. Create `config.local.toml` from `config.example.toml`, point it at local files outside the repository, and keep it untracked.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

Run the synthetic notebooks with JupyterLab:

```powershell
jupyter lab
```

For local inputs, copy `config.example.toml` to `config.local.toml`, then use the loading cell in `notebooks/01_data_inventory.ipynb`. The initial configuration should include only activity tags you are comfortable analysing.

```python
from garmin_daylio_analysis.workflow import load_local_analysis

config, analysis_daily = load_local_analysis()
```

## Analysis approach

- Daily mean mood is the primary outcome; median and final check-in are sensitivity outcomes.
- Same-day relationships use Spearman correlations, seven-day block-bootstrap confidence intervals, and false-discovery-rate adjustment.
- Lag plots test Garmin(t-lag) against mood(t); positive lag means the Garmin measure precedes the mood date.
- Adjusted ordinal models account for weekday, long-term trend, and annual seasonality.
- Results are associations, not causal conclusions. Missing values remain missing and no statistical-test imputation is used.

## Quality checks

```powershell
pytest
python scripts/smoke_test_notebooks.py
python scripts/verify_public_repo.py
```

Before publishing, revoke any exposed GitHub credential through GitHub, ensure `origin` contains no credential, run the repository safety check, and inspect `git status` plus `git ls-files`.
