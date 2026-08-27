"""Aggregate-only data quality summaries for notebooks."""

from __future__ import annotations

import pandas as pd


PLAUSIBLE_RANGES = {
    "steps": (0, 100_000),
    "distance_m": (0, 200_000),
    "active_calories": (0, 10_000),
    "moderate_intensity_minutes": (0, 1_440),
    "vigorous_intensity_minutes": (0, 1_440),
    "resting_heart_rate": (20, 250),
    "minimum_heart_rate": (20, 250),
    "maximum_heart_rate": (20, 250),
    "stress_average": (0, 100),
    "stress_maximum": (0, 100),
    "energy_start": (0, 100),
    "energy_end": (0, 100),
    "energy_high": (0, 100),
    "energy_low": (0, 100),
    "respiration_average": (3, 60),
    "respiration_high": (3, 80),
    "respiration_low": (1, 60),
    "hydration_ml": (0, 15_000),
    "mood_mean": (1, 5),
}


def inventory_summary(frame: pd.DataFrame) -> pd.Series:
    """Return coverage and overlap counts without exposing row values."""
    has_garmin = frame.get("steps", pd.Series(False, index=frame.index)).notna()
    has_daylio = frame.get("mood_mean", pd.Series(False, index=frame.index)).notna()
    return pd.Series(
        {
            "rows": len(frame),
            "unique_dates": frame["calendar_date"].nunique(),
            "duplicate_dates": int(frame["calendar_date"].duplicated().sum()),
            "garmin_dates": int(has_garmin.sum()),
            "daylio_dates": int(has_daylio.sum()),
            "overlap_dates": int((has_garmin & has_daylio).sum()),
        }
    )


def implausible_value_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Count values outside conservative physical ranges; do not remove them."""
    rows = []
    for column, (minimum, maximum) in PLAUSIBLE_RANGES.items():
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "observed": int(values.notna().sum()),
                "below_range": int((values < minimum).sum()),
                "above_range": int((values > maximum).sum()),
            }
        )
    return pd.DataFrame(rows)
