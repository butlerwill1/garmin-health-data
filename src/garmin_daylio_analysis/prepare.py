"""Safe daily aggregation and feature preparation."""

from __future__ import annotations

import re

import pandas as pd

from .privacy import assert_public_frame_safe


def _tag_column(tag: str) -> str:
    return "tag_" + re.sub(r"[^a-z0-9]+", "_", tag.casefold()).strip("_")


def aggregate_daylio_daily(entries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated mood check-ins without retaining text or unapproved tags."""
    required = {"calendar_date", "local_timestamp", "mood_score", "approved_activity_tags"}
    missing = required.difference(entries.columns)
    if missing:
        raise ValueError(f"Daylio entry table is missing: {sorted(missing)}")

    ordered = entries.sort_values("local_timestamp", kind="stable")
    daily = (
        ordered.groupby("calendar_date", as_index=False)
        .agg(
            mood_mean=("mood_score", "mean"),
            mood_median=("mood_score", "median"),
            mood_last=("mood_score", "last"),
            mood_range=("mood_score", lambda values: values.max() - values.min()),
            checkin_count=("mood_score", "size"),
        )
    )
    known_tags = sorted({tag for tags in ordered["approved_activity_tags"] for tag in tags})
    for tag in known_tags:
        column = _tag_column(tag)
        counts = ordered.assign(**{column: ordered["approved_activity_tags"].map(lambda tags: int(tag in tags))})
        daily = daily.merge(counts.groupby("calendar_date", as_index=False)[column].sum(), on="calendar_date", how="left")
    assert_public_frame_safe(daily)
    return daily.sort_values("calendar_date").reset_index(drop=True)


def aggregate_activity_daily(activities: pd.DataFrame) -> pd.DataFrame:
    """Create generic daily workout totals from the safe activity table."""
    if activities.empty:
        return pd.DataFrame(columns=["calendar_date", "workout_count", "workout_duration_seconds", "workout_distance_m", "workout_calories"])
    daily = (
        activities.groupby("calendar_date", as_index=False)
        .agg(
            workout_count=("activity_count", "sum"),
            workout_duration_seconds=("duration_seconds", "sum"),
            workout_distance_m=("distance_m", "sum"),
            workout_calories=("calories", "sum"),
        )
    )
    assert_public_frame_safe(daily)
    return daily


def build_analysis_daily(garmin_daily: pd.DataFrame, activities: pd.DataFrame, daylio_entries: pd.DataFrame) -> pd.DataFrame:
    """Return one date-indexed table for descriptive and association analyses."""
    mood_daily = aggregate_daylio_daily(daylio_entries)
    activity_daily = aggregate_activity_daily(activities)
    for frame, label in ((garmin_daily, "Garmin daily"), (mood_daily, "Daylio daily"), (activity_daily, "activity daily")):
        if frame["calendar_date"].duplicated().any():
            raise ValueError(f"{label} table contains duplicate calendar dates.")
    merged = garmin_daily.merge(mood_daily, how="outer", on="calendar_date")
    merged = merged.merge(activity_daily, how="left", on="calendar_date")
    merged = merged.sort_values("calendar_date").reset_index(drop=True)
    assert_public_frame_safe(merged)
    return merged


def numeric_exposure_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric Garmin and workout features, excluding outcomes and metadata."""
    excluded = {"mood_mean", "mood_median", "mood_last", "mood_range", "checkin_count"}
    return [
        column
        for column in frame.columns
        if column not in excluded
        and column != "calendar_date"
        and not column.startswith("tag_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def make_feature_variants(frame: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Add strictly-prior seven-day and 28-day personal-baseline features."""
    result = frame.sort_values("calendar_date").copy()
    features = features or numeric_exposure_columns(result)
    for feature in features:
        prior = result[feature].shift(1)
        trailing = prior.rolling(window=7, min_periods=3).mean()
        baseline = prior.rolling(window=28, min_periods=14).mean()
        result[f"{feature}_trailing_7d"] = trailing
        result[f"{feature}_deviation_28d"] = result[feature] - baseline
    assert_public_frame_safe(result)
    return result
