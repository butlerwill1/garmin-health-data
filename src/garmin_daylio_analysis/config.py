"""Configuration for local, untracked data sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AnalysisConfig:
    garmin_export_dir: Path
    daylio_export_csv: Path
    timezone: str = "Europe/London"
    approved_activity_tags: tuple[str, ...] = field(default_factory=tuple)
    min_observations: int = 60
    min_group_size: int = 20
    max_lag_days: int = 7
    bootstrap_resamples: int = 2000
    block_size_days: int = 7


def load_config(path: str | Path = "config.local.toml") -> AnalysisConfig:
    """Read a local TOML file without ever loading free-text export fields."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    paths = raw["paths"]
    analysis = raw.get("analysis", {})
    daylio = raw.get("daylio", {})
    return AnalysisConfig(
        garmin_export_dir=Path(paths["garmin_export_dir"]),
        daylio_export_csv=Path(paths["daylio_export_csv"]),
        timezone=analysis.get("timezone", "Europe/London"),
        approved_activity_tags=tuple(daylio.get("approved_activity_tags", [])),
        min_observations=int(analysis.get("min_observations", 60)),
        min_group_size=int(analysis.get("min_group_size", 20)),
        max_lag_days=int(analysis.get("max_lag_days", 7)),
        bootstrap_resamples=int(analysis.get("bootstrap_resamples", 2000)),
        block_size_days=int(analysis.get("block_size_days", 7)),
    )
