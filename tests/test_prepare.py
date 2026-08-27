from __future__ import annotations

import pandas as pd
import pytest

from garmin_daylio_analysis.prepare import aggregate_daylio_daily, build_analysis_daily, make_feature_variants


def _entries():
    return pd.DataFrame(
        {
            "calendar_date": [pd.Timestamp("2024-01-01").date()] * 2,
            "local_timestamp": pd.to_datetime(["2024-01-01 09:00", "2024-01-01 18:00"], utc=True),
            "mood_score": [2, 4],
            "approved_activity_tags": [("Demo outdoors",), ()],
        }
    )


def test_aggregate_daylio_retains_sensitivity_outcomes():
    daily = aggregate_daylio_daily(_entries())
    assert daily.loc[0, "mood_mean"] == 3
    assert daily.loc[0, "mood_median"] == 3
    assert daily.loc[0, "mood_last"] == 4
    assert daily.loc[0, "mood_range"] == 2
    assert daily.loc[0, "checkin_count"] == 2
    assert daily.loc[0, "tag_demo_outdoors"] == 1


def test_make_feature_variants_use_only_prior_days():
    dates = pd.date_range("2024-01-01", periods=30, freq="D").date
    frame = pd.DataFrame({"calendar_date": dates, "steps": range(30), "mood_mean": [3] * 30})
    prepared = make_feature_variants(frame, ["steps"])
    assert pd.isna(prepared.loc[0, "steps_trailing_7d"])
    assert prepared.loc[3, "steps_trailing_7d"] == pytest.approx(1.0)
    assert prepared.loc[28, "steps_deviation_28d"] == pytest.approx(14.5)


def test_build_analysis_daily_rejects_duplicate_source_dates():
    garmin = pd.DataFrame({"calendar_date": [pd.Timestamp("2024-01-01").date()] * 2, "steps": [1, 2]})
    activities = pd.DataFrame(columns=["calendar_date", "activity_count", "duration_seconds", "distance_m", "calories"])
    with pytest.raises(ValueError, match="duplicate"):
        build_analysis_daily(garmin, activities, _entries())
