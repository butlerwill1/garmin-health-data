"""Allowlist-only readers for the two local export formats."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .privacy import assert_public_frame_safe


DAILY_COLUMNS = [
    "calendar_date",
    "steps",
    "distance_m",
    "active_calories",
    "moderate_intensity_minutes",
    "vigorous_intensity_minutes",
    "resting_heart_rate",
    "minimum_heart_rate",
    "maximum_heart_rate",
    "stress_average",
    "stress_maximum",
    "stress_minutes",
    "stress_rest_minutes",
    "energy_charged",
    "energy_drained",
    "energy_start",
    "energy_end",
    "energy_high",
    "energy_low",
    "respiration_average",
    "respiration_high",
    "respiration_low",
    "hydration_ml",
]

ACTIVITY_COLUMNS = [
    "calendar_date",
    "activity_type",
    "activity_count",
    "duration_seconds",
    "distance_m",
    "calories",
    "average_heart_rate",
    "maximum_heart_rate",
]

MOOD_ORDER = {"awful": 1, "bad": 2, "meh": 3, "good": 4, "rad": 5}


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected an array in {path.name}")
    return [row for row in value if isinstance(row, dict)]


def _first_stat(stats: Any, stat_type: str) -> float | None:
    if not isinstance(stats, list):
        return None
    for stat in stats:
        if isinstance(stat, dict) and stat.get("bodyBatteryStatType") == stat_type:
            return stat.get("statsValue")
    return None


def _stress_total(record: dict[str, Any]) -> dict[str, float | None]:
    stress = record.get("allDayStress")
    entries = stress.get("aggregatorList", []) if isinstance(stress, dict) else []
    total = next((entry for entry in entries if entry.get("type") == "TOTAL"), {})
    return {
        "stress_average": total.get("averageStressLevel"),
        "stress_maximum": total.get("maxStressLevel"),
        "stress_minutes": _minutes(total.get("stressDuration")),
        "stress_rest_minutes": _minutes(total.get("restDuration")),
    }


def _minutes(value: Any) -> float | None:
    return value / 60 if isinstance(value, (int, float)) else None


def _daily_row(record: dict[str, Any]) -> dict[str, Any]:
    energy = record.get("bodyBattery") if isinstance(record.get("bodyBattery"), dict) else {}
    respiration = record.get("respiration") if isinstance(record.get("respiration"), dict) else {}
    row = {
        "calendar_date": record.get("calendarDate"),
        "steps": record.get("totalSteps"),
        "distance_m": record.get("totalDistanceMeters"),
        "active_calories": record.get("activeKilocalories"),
        "moderate_intensity_minutes": record.get("moderateIntensityMinutes"),
        "vigorous_intensity_minutes": record.get("vigorousIntensityMinutes"),
        "resting_heart_rate": record.get("restingHeartRate"),
        "minimum_heart_rate": record.get("minHeartRate"),
        "maximum_heart_rate": record.get("maxHeartRate"),
        "energy_charged": energy.get("chargedValue"),
        "energy_drained": energy.get("drainedValue"),
        "energy_start": _first_stat(energy.get("bodyBatteryStatList"), "STARTOFDAY"),
        "energy_end": _first_stat(energy.get("bodyBatteryStatList"), "ENDOFDAY"),
        "energy_high": _first_stat(energy.get("bodyBatteryStatList"), "HIGHEST"),
        "energy_low": _first_stat(energy.get("bodyBatteryStatList"), "LOWEST"),
        "respiration_average": respiration.get("avgWakingRespirationValue"),
        "respiration_high": respiration.get("highestRespirationValue"),
        "respiration_low": respiration.get("lowestRespirationValue"),
        "hydration_ml": None,
    }
    row.update(_stress_total(record))
    return row


def _coerce_daily(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reindex(columns=DAILY_COLUMNS)
    frame["calendar_date"] = pd.to_datetime(frame["calendar_date"], errors="raise").dt.date
    if frame["calendar_date"].duplicated().any():
        duplicate_dates = frame.loc[frame["calendar_date"].duplicated(), "calendar_date"].astype(str).tolist()
        raise ValueError(f"Duplicate Garmin daily records: {duplicate_dates}")
    for column in DAILY_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("calendar_date").reset_index(drop=True)


def load_garmin_daily(export_dir: str | Path) -> pd.DataFrame:
    """Load only approved daily Garmin fields from an export directory."""
    export_path = Path(export_dir)
    daily_paths = sorted(export_path.rglob("UDSFile*.json"))
    if not daily_paths:
        raise FileNotFoundError("No Garmin daily-summary files were found.")

    records = [record for path in daily_paths for record in _read_json_array(path)]
    daily = _coerce_daily(pd.DataFrame([_daily_row(record) for record in records]))

    hydration_paths = sorted(export_path.rglob("HydrationLogFile*.json"))
    if hydration_paths:
        hydration_records = [record for path in hydration_paths for record in _read_json_array(path)]
        hydration = pd.DataFrame(hydration_records)
        if {"calendarDate", "valueInML"}.issubset(hydration.columns):
            hydration = hydration.assign(
                calendar_date=pd.to_datetime(hydration["calendarDate"], errors="coerce").dt.date,
                value=pd.to_numeric(hydration["valueInML"], errors="coerce"),
            )
            hydration = hydration.groupby("calendar_date", as_index=False)["value"].sum(min_count=1)
            daily = daily.drop(columns="hydration_ml").merge(hydration, how="left", on="calendar_date")
            daily = daily.rename(columns={"value": "hydration_ml"})

    daily = daily.reindex(columns=DAILY_COLUMNS)
    assert_public_frame_safe(daily)
    return daily


def _activity_type(value: Any) -> str:
    if isinstance(value, str) and value:
        return value.lower()
    if isinstance(value, dict):
        candidate = value.get("typeKey") or value.get("typeId") or value.get("sportTypeId")
        if candidate is not None:
            return str(candidate).lower()
    return "other"


def _find_activity_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.rglob("*_0_summarizedActivities.json"))
    if not candidates:
        raise FileNotFoundError("No Garmin summarised-activities file was found.")
    return candidates[0]


def load_garmin_activities(export_dir: str | Path, timezone: str = "Europe/London") -> pd.DataFrame:
    """Load generic daily activity aggregates without activity names or locations."""
    activity_file = _find_activity_file(Path(export_dir))
    with activity_file.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        records = raw.get("summarizedActivitiesExport", [])
    elif isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        records = raw[0].get("summarizedActivitiesExport", [])
    elif isinstance(raw, list):
        records = raw
    else:
        records = []

    rows: list[dict[str, Any]] = []
    for record in records:
        timestamp = record.get("beginTimestamp")
        if timestamp is None:
            continue
        local_date = pd.to_datetime(timestamp, unit="ms", utc=True).tz_convert(ZoneInfo(timezone)).date()
        rows.append(
            {
                "calendar_date": local_date,
                "activity_type": _activity_type(record.get("activityType") or record.get("sportType")),
                "duration_seconds": record.get("duration"),
                "distance_m": record.get("distance"),
                "calories": record.get("calories"),
                "average_heart_rate": record.get("avgHr"),
                "maximum_heart_rate": record.get("maxHr"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=ACTIVITY_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in ACTIVITY_COLUMNS[2:]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = (
        frame.groupby(["calendar_date", "activity_type"], as_index=False)
        .agg(
            activity_count=("activity_type", "size"),
            duration_seconds=("duration_seconds", "sum"),
            distance_m=("distance_m", "sum"),
            calories=("calories", "sum"),
            average_heart_rate=("average_heart_rate", "mean"),
            maximum_heart_rate=("maximum_heart_rate", "max"),
        )
        .reindex(columns=ACTIVITY_COLUMNS)
        .sort_values(["calendar_date", "activity_type"])
        .reset_index(drop=True)
    )
    assert_public_frame_safe(grouped)
    return grouped


def _split_approved_tags(value: Any, approved_tags: Iterable[str]) -> tuple[str, ...]:
    approved = {tag.casefold(): tag for tag in approved_tags}
    if not isinstance(value, str) or not value.strip():
        return ()
    tags = [tag.strip() for tag in value.split("|") if tag.strip()]
    return tuple(approved[tag.casefold()] for tag in tags if tag.casefold() in approved)


def load_daylio_entries(
    csv_path: str | Path,
    timezone: str = "Europe/London",
    approved_activity_tags: Iterable[str] = (),
) -> pd.DataFrame:
    """Load Daylio timestamps, ordered mood, and explicitly approved activity tags only."""
    source = pd.read_csv(
        csv_path,
        usecols=lambda name: name in {"full_date", "time", "mood", "activities"},
        dtype="string",
    )
    required = {"full_date", "time", "mood"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Daylio export is missing required columns: {sorted(missing)}")

    timestamp = pd.to_datetime(source["full_date"] + " " + source["time"], format="%Y-%m-%d %H:%M", errors="raise")
    local_timestamp = timestamp.dt.tz_localize(timezone, ambiguous=False, nonexistent="shift_forward")
    mood_label = source["mood"].str.strip().str.casefold()
    unknown = sorted(set(mood_label.dropna()) - set(MOOD_ORDER))
    if unknown:
        raise ValueError(f"Unknown Daylio mood labels: {unknown}")

    entries = pd.DataFrame(
        {
            "local_timestamp": local_timestamp,
            "calendar_date": local_timestamp.dt.date,
            "mood_label": mood_label,
            "mood_score": mood_label.map(MOOD_ORDER).astype("Int64"),
            "approved_activity_tags": [
                _split_approved_tags(value, approved_activity_tags)
                for value in source.get("activities", pd.Series(pd.NA, index=source.index))
            ],
        }
    ).sort_values("local_timestamp", kind="stable").reset_index(drop=True)
    assert_public_frame_safe(entries.drop(columns="approved_activity_tags"))
    return entries
