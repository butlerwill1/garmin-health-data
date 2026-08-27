"""Invented, deterministic data for demos and tests; never based on an export."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .prepare import build_analysis_daily, make_feature_variants


def make_synthetic_tables(days: int = 140, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create safe demo tables with positive, null, and lagged relationships."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D").date
    movement = rng.normal(7000, 1700, days).clip(1000, None)
    prior_movement = np.r_[movement[0], movement[:-1]]
    latent_mood = 3 + 0.00013 * prior_movement + rng.normal(0, 0.45, days)
    score = np.clip(np.rint(latent_mood), 1, 5).astype(int)
    labels = np.array(["awful", "bad", "meh", "good", "rad"])[score - 1]
    garmin_daily = pd.DataFrame(
        {
            "calendar_date": dates,
            "steps": movement,
            "distance_m": movement * rng.uniform(0.55, 0.8, days),
            "active_calories": movement * rng.uniform(0.025, 0.04, days),
            "moderate_intensity_minutes": rng.poisson(25, days),
            "vigorous_intensity_minutes": rng.poisson(6, days),
            "resting_heart_rate": rng.normal(58, 4, days),
            "minimum_heart_rate": rng.normal(50, 4, days),
            "maximum_heart_rate": rng.normal(130, 16, days),
            "stress_average": rng.normal(30, 8, days),
            "stress_maximum": rng.normal(75, 10, days),
            "stress_minutes": rng.normal(110, 35, days).clip(0),
            "stress_rest_minutes": rng.normal(350, 60, days).clip(0),
            "energy_charged": rng.normal(35, 8, days),
            "energy_drained": rng.normal(42, 9, days),
            "energy_start": rng.normal(55, 12, days),
            "energy_end": rng.normal(45, 12, days),
            "energy_high": rng.normal(70, 10, days),
            "energy_low": rng.normal(30, 9, days),
            "respiration_average": rng.normal(14, 1.2, days),
            "respiration_high": rng.normal(17, 1.5, days),
            "respiration_low": rng.normal(10, 1.0, days),
            "hydration_ml": rng.normal(1800, 400, days).clip(400),
        }
    )
    activities = pd.DataFrame(
        {
            "calendar_date": dates[::3],
            "activity_type": "demo_activity",
            "activity_count": 1,
            "duration_seconds": rng.integers(1200, 3600, len(dates[::3])),
            "distance_m": rng.uniform(2000, 8000, len(dates[::3])),
            "calories": rng.uniform(150, 600, len(dates[::3])),
            "average_heart_rate": rng.uniform(110, 150, len(dates[::3])),
            "maximum_heart_rate": rng.uniform(140, 180, len(dates[::3])),
        }
    )
    timestamps = pd.to_datetime(dates).tz_localize("Europe/London") + pd.Timedelta(hours=18)
    entries = pd.DataFrame(
        {
            "local_timestamp": timestamps,
            "calendar_date": dates,
            "mood_label": labels,
            "mood_score": score,
            "approved_activity_tags": [("Demo outdoors",) if index % 4 == 0 else () for index in range(days)],
        }
    )
    return garmin_daily, activities, entries


def make_synthetic_analysis_daily(days: int = 140, seed: int = 7) -> pd.DataFrame:
    """Build the same analysis table shape used by the notebooks."""
    daily, activities, entries = make_synthetic_tables(days, seed)
    return make_feature_variants(build_analysis_daily(daily, activities, entries))
