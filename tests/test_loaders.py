from __future__ import annotations

import json

import pandas as pd
import pytest

from garmin_daylio_analysis.loaders import load_daylio_entries, load_garmin_activities, load_garmin_daily


def test_load_daylio_keeps_only_safe_fields_and_approved_tags(tmp_path):
    source = tmp_path / "daylio.csv"
    source.write_text(
        "full_date,time,mood,activities,note_title,note\n"
        "2024-03-31,01:30,good,Keep | Drop,private title,private text\n",
        encoding="utf-8",
    )
    entries = load_daylio_entries(source, approved_activity_tags=["Keep"])

    assert entries.columns.tolist() == ["local_timestamp", "calendar_date", "mood_label", "mood_score", "approved_activity_tags"]
    assert entries.loc[0, "approved_activity_tags"] == ("Keep",)
    assert entries.loc[0, "local_timestamp"].tzinfo is not None


def test_load_daylio_rejects_unknown_mood(tmp_path):
    source = tmp_path / "daylio.csv"
    source.write_text("full_date,time,mood\n2024-01-01,12:00,unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown Daylio mood"):
        load_daylio_entries(source)


def test_load_garmin_daily_flattens_allowlisted_nested_values(tmp_path):
    daily_dir = tmp_path / "export"
    daily_dir.mkdir()
    (daily_dir / "UDSFile_fixture.json").write_text(
        json.dumps(
            [
                {
                    "calendarDate": "2024-01-02",
                    "totalSteps": 4321,
                    "totalDistanceMeters": 3200,
                    "activeKilocalories": 280,
                    "moderateIntensityMinutes": 30,
                    "vigorousIntensityMinutes": 10,
                    "restingHeartRate": 57,
                    "allDayStress": {"aggregatorList": [{"type": "TOTAL", "averageStressLevel": 25, "maxStressLevel": 73, "stressDuration": 1200, "restDuration": 600}]},
                    "bodyBattery": {"chargedValue": 32, "drainedValue": 40, "bodyBatteryStatList": [{"bodyBatteryStatType": "STARTOFDAY", "statsValue": 62}, {"bodyBatteryStatType": "ENDOFDAY", "statsValue": 42}]},
                    "respiration": {"avgWakingRespirationValue": 14, "highestRespirationValue": 17, "lowestRespirationValue": 11},
                    "startLatitude": 1.0,
                    "userProfilePK": 99,
                }
            ]
        ),
        encoding="utf-8",
    )
    (daily_dir / "HydrationLogFile_fixture.json").write_text(
        json.dumps([{"calendarDate": "2024-01-02", "valueInML": 750, "userProfilePK": 99}]), encoding="utf-8"
    )

    daily = load_garmin_daily(daily_dir)

    assert daily.loc[0, "stress_average"] == 25
    assert daily.loc[0, "stress_minutes"] == 20
    assert daily.loc[0, "energy_start"] == 62
    assert daily.loc[0, "hydration_ml"] == 750
    assert "startLatitude" not in daily.columns
    assert "userProfilePK" not in daily.columns


def test_load_garmin_daily_rejects_duplicate_dates(tmp_path):
    source = tmp_path / "UDSFile_fixture.json"
    source.write_text(json.dumps([{"calendarDate": "2024-01-01"}, {"calendarDate": "2024-01-01"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate Garmin"):
        load_garmin_daily(tmp_path)


def test_load_garmin_activities_aggregates_generic_types(tmp_path):
    source = tmp_path / "person_0_summarizedActivities.json"
    source.write_text(
        json.dumps(
            {
                "summarizedActivitiesExport": [
                    {"beginTimestamp": 1704110400000, "activityType": {"typeKey": "running"}, "duration": 1200, "distance": 2500, "calories": 190, "avgHr": 130, "maxHr": 160},
                    {"beginTimestamp": 1704114000000, "activityType": {"typeKey": "running"}, "duration": 600, "distance": 1200, "calories": 90, "avgHr": 125, "maxHr": 150},
                ]
            }
        ),
        encoding="utf-8",
    )
    activities = load_garmin_activities(tmp_path)
    assert activities.loc[0, "activity_count"] == 2
    assert activities.loc[0, "activity_type"] == "running"
    assert activities.loc[0, "duration_seconds"] == 1800


def test_load_garmin_activities_accepts_single_export_wrapper(tmp_path):
    source = tmp_path / "person_0_summarizedActivities.json"
    source.write_text(
        json.dumps(
            [
                {
                    "summarizedActivitiesExport": [
                        {"beginTimestamp": 1704110400000, "activityType": {"typeKey": "cycling"}, "duration": 900}
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    activities = load_garmin_activities(tmp_path)
    assert len(activities) == 1
    assert activities.loc[0, "activity_type"] == "cycling"
