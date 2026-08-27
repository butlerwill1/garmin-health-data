from __future__ import annotations

import pandas as pd

from garmin_daylio_analysis.quality import implausible_value_summary, inventory_summary


def test_quality_summaries_report_counts_only():
    frame = pd.DataFrame(
        {
            "calendar_date": pd.date_range("2024-01-01", periods=3).date,
            "steps": [1000, -1, 5000],
            "mood_mean": [3, 4, None],
        }
    )
    inventory = inventory_summary(frame)
    quality = implausible_value_summary(frame).set_index("feature")
    assert inventory["overlap_dates"] == 2
    assert quality.loc["steps", "below_range"] == 1
