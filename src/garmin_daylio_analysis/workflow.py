"""One safe entry point for building a local analysis table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AnalysisConfig, load_config
from .loaders import load_daylio_entries, load_garmin_activities, load_garmin_daily
from .prepare import build_analysis_daily, make_feature_variants


def load_local_analysis(config_path: str | Path = "config.local.toml") -> tuple[AnalysisConfig, pd.DataFrame]:
    """Build private analysis data from the paths in an untracked config file."""
    config = load_config(config_path)
    daily = load_garmin_daily(config.garmin_export_dir)
    activities = load_garmin_activities(config.garmin_export_dir, config.timezone)
    entries = load_daylio_entries(
        config.daylio_export_csv,
        config.timezone,
        config.approved_activity_tags,
    )
    analysis_daily = build_analysis_daily(daily, activities, entries)
    return config, make_feature_variants(analysis_daily)
