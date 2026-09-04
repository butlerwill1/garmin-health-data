from datetime import date

import pandas as pd

from garmin_daylio_analysis.sleep import (
    derive_awakenings,
    reconstruct_timestamp_16,
    scenario_segments,
    sleep_tag_associations,
    transition_messages_to_segments,
)


UTC = "UTC"


def stamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=UTC)


def test_timestamp_16_reconstruction_handles_rollover_and_missing_anchor():
    assert reconstruct_timestamp_16(2, 0x1FFFE) == 0x20002
    assert reconstruct_timestamp_16(100, None) is None


def test_transition_messages_make_clipped_segments_and_unknown_stage_safe():
    summaries = pd.DataFrame([{
        "sleep_date": date(2026, 1, 2), "sleep_start_utc": stamp("2026-01-01 23:00"),
        "sleep_end_utc": stamp("2026-01-02 00:00"),
    }])
    segments, _ = transition_messages_to_segments([
        {"timestamp": stamp("2026-01-01 22:50"), "sleep_level": "awake"},
        {"timestamp": stamp("2026-01-01 23:10"), "sleep_level": "mystery"},
        {"timestamp": stamp("2026-01-01 23:10"), "sleep_level": "light"},
    ], summaries, "Europe/London", 3)
    assert [item["raw_stage"] for item in segments] == ["awake", "light"]
    assert segments[0]["start_utc"] == stamp("2026-01-01 23:00")
    assert segments[-1]["end_utc"] == stamp("2026-01-02 00:00")


def timeline() -> pd.DataFrame:
    return pd.DataFrame([
        {"sleep_date": date(2026, 1, 2), "segment_id": "a", "start_utc": stamp("2026-01-01 23:00"), "end_utc": stamp("2026-01-01 23:10"), "raw_stage": "awake"},
        {"sleep_date": date(2026, 1, 2), "segment_id": "b", "start_utc": stamp("2026-01-01 23:10"), "end_utc": stamp("2026-01-01 23:50"), "raw_stage": "light"},
        {"sleep_date": date(2026, 1, 2), "segment_id": "c", "start_utc": stamp("2026-01-01 23:50"), "end_utc": stamp("2026-01-02 00:10"), "raw_stage": "awake"},
        {"sleep_date": date(2026, 1, 2), "segment_id": "d", "start_utc": stamp("2026-01-02 00:10"), "end_utc": stamp("2026-01-02 00:30"), "raw_stage": "deep"},
    ])


def test_scenarios_and_manual_overrides_do_not_change_raw_labels():
    segments = timeline()
    islands = pd.DataFrame([{"segment_id": "b", "evidence_adjusted_candidate": True}])
    overrides = pd.DataFrame([{"sleep_date": date(2026, 1, 2), "start_utc": stamp("2026-01-01 23:15"), "end_utc": stamp("2026-01-01 23:25"), "state": "awake"}])
    raw = scenario_segments(segments, islands, overrides, "raw")
    adjusted = scenario_segments(segments, islands, overrides, "evidence_adjusted")
    assert set(raw.loc[raw.segment_id == "b", "stage"]) == {"light"}
    assert set(adjusted.loc[adjusted.segment_id == "b", "raw_stage"]) == {"light"}
    assert set(adjusted.loc[adjusted.segment_id == "b", "stage"]) == {"awake"}


def test_prolonged_thresholds_and_fifteen_minute_return_to_sleep():
    frame = timeline()
    frame["stage"] = frame.raw_stage
    frame["scenario"] = "raw"
    events, nights = derive_awakenings(frame)
    final = events.iloc[-1]
    assert final.duration_minutes == 20
    assert final.return_to_sleep
    assert not bool(nights.iloc[0].prolonged_wake_30_minutes)


def test_tag_associations_keep_low_samples_descriptive_only():
    nights = pd.DataFrame({
        "tag_medication": [1, 0, 0, 0],
        "raw_wake_after_sleep_onset_minutes": [30, 5, 10, 2],
        "raw_prolonged_wake_30_minutes": [True, False, False, False],
    })
    result = sleep_tag_associations(nights)
    assert len(result) == 2
    assert result.descriptive_only.all()
