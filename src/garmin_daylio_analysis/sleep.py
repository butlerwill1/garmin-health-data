"""Private, export-first Garmin sleep analysis.

The module deliberately keeps only derived sleep and physiological measurements.
It never returns source filenames, device identifiers, locations, or account data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
from garmin_fit_sdk.decoder import Decoder
from garmin_fit_sdk.stream import Stream
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .config import AnalysisConfig, load_config
from .loaders import load_daylio_entries
from .privacy import assert_public_frame_safe


FIT_EPOCH_UNIX_SECONDS = 631065600
STAGES = frozenset({"awake", "light", "deep", "rem", "unmeasurable"})
SCENARIOS = ("raw", "evidence_adjusted", "bracketed_upper_bound")


@dataclass(frozen=True)
class SleepAnalysis:
    """Privacy-safe tables used by the sleep notebooks."""

    nights: pd.DataFrame
    segments: pd.DataFrame
    signals: pd.DataFrame
    awakenings: pd.DataFrame
    light_islands: pd.DataFrame


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _as_utc(value: object) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _fit_seconds(value: object) -> int | None:
    stamp = _as_utc(value)
    if pd.isna(stamp):
        return None
    return int(stamp.timestamp()) - FIT_EPOCH_UNIX_SECONDS


def reconstruct_timestamp_16(timestamp_16: int, reference_fit_seconds: int | None) -> int | None:
    """Expand a FIT ``timestamp_16`` using a preceding full FIT timestamp.

    The modulo operation is intentional: it advances across a 16-bit rollover.
    ``None`` is returned when an anchor is unavailable rather than inventing time.
    """
    if reference_fit_seconds is None or pd.isna(timestamp_16):
        return None
    return reference_fit_seconds + ((int(timestamp_16) - (reference_fit_seconds & 0xFFFF)) & 0xFFFF)


def _from_fit_seconds(value: int | None) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    return pd.Timestamp(value + FIT_EPOCH_UNIX_SECONDS, unit="s", tz="UTC")


def _local(stamp: pd.Timestamp, timezone_name: str) -> pd.Timestamp:
    return stamp.tz_convert(timezone_name)


def _parse_summary_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    return _as_utc(value)


def load_sleep_summaries(export_dir: str | Path, timezone_name: str) -> pd.DataFrame:
    """Load Garmin JSON sleep summaries, retaining only analysis-relevant fields."""
    records: list[dict[str, object]] = []
    fields = {
        "deepSleepSeconds": "summary_deep_seconds",
        "lightSleepSeconds": "summary_light_seconds",
        "remSleepSeconds": "summary_rem_seconds",
        "awakeSleepSeconds": "summary_awake_seconds",
        "unmeasurableSeconds": "summary_unmeasurable_seconds",
        "averageRespiration": "summary_average_respiration",
        "lowestRespiration": "summary_lowest_respiration",
        "highestRespiration": "summary_highest_respiration",
        "awakeCount": "summary_awake_count",
        "avgSleepStress": "summary_average_stress",
        "restlessMomentCount": "summary_restless_moment_count",
    }
    for path in Path(export_dir).rglob("*_sleepData.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in payload if isinstance(payload, list) else [payload]:
            start = _parse_summary_timestamp(item.get("sleepStartTimestampGMT"))
            end = _parse_summary_timestamp(item.get("sleepEndTimestampGMT"))
            if pd.isna(start) or pd.isna(end) or end <= start:
                continue
            row: dict[str, object] = {
                "sleep_date": _local(end, timezone_name).date(),
                "sleep_start_utc": start,
                "sleep_end_utc": end,
                "sleep_start_local": _local(start, timezone_name),
                "sleep_end_local": _local(end, timezone_name),
                "summary_coverage": True,
            }
            for source, target in fields.items():
                row[target] = pd.to_numeric(item.get(source), errors="coerce")
            records.append(row)
    if not records:
        return _empty(["sleep_date", "sleep_start_utc", "sleep_end_utc", "summary_coverage"])
    frame = pd.DataFrame(records).sort_values("sleep_end_utc").drop_duplicates("sleep_date", keep="last")
    frame["summary_total_seconds"] = frame.filter(regex=r"^summary_(?:deep|light|rem|awake|unmeasurable)_seconds$").sum(axis=1, min_count=1)
    return frame.reset_index(drop=True)


def _decode_fit(data: bytes) -> dict[str, list[dict[str, object]]]:
    result, errors = Decoder(Stream.from_bytes_io(BytesIO(data))).read(
        convert_types_to_strings=True, convert_datetimes_to_dates=True,
    )
    if errors:
        raise ValueError("FIT decoder reported an error")
    return result


class _FileIdReadComplete(Exception):
    """Internal control flow used to cheaply classify a FIT stream."""


def _read_file_id(data: bytes) -> dict[str, object] | None:
    """Read only the first FIT file-id message before doing a full decode."""
    found: dict[str, object] | None = None

    def listener(global_message_number: int, message: dict[str, object]) -> None:
        nonlocal found
        if global_message_number == 0:
            found = message
            raise _FileIdReadComplete()

    Decoder(Stream.from_bytes_io(BytesIO(data))).read(
        convert_types_to_strings=True, convert_datetimes_to_dates=True,
        merge_heart_rates=False, mesg_listener=listener,
    )
    return found


def _fit_type(messages: dict[str, list[dict[str, object]]]) -> str | None:
    for message in messages.get("file_id_mesgs", []):
        value = message.get("type")
        if value is not None:
            return str(value)
    return None


def _uploaded_archives(export_dir: str | Path) -> list[Path]:
    return sorted(Path(export_dir).rglob("UploadedFiles*.zip"))


def _stage_from(value: object) -> str:
    stage = str(value).lower()
    return stage if stage in STAGES else "unmeasurable"


def _match_summary(transitions: pd.DataFrame, summaries: pd.DataFrame, timezone_name: str) -> pd.Series | None:
    if transitions.empty or summaries.empty:
        return None
    first, last = transitions.timestamp_utc.min(), transitions.timestamp_utc.max()
    overlap = (summaries.sleep_start_utc <= last) & (summaries.sleep_end_utc >= first)
    candidates = summaries.loc[overlap].copy()
    if candidates.empty:
        return None
    candidates["overlap"] = np.minimum(candidates.sleep_end_utc, last) - np.maximum(candidates.sleep_start_utc, first)
    return candidates.sort_values("overlap", ascending=False).iloc[0]


def transition_messages_to_segments(
    messages: list[dict[str, object]], summaries: pd.DataFrame, timezone_name: str, sleep_id: int,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Turn one sleep FIT's transition messages into clipped labelled segments."""
    rows = [
        {"timestamp_utc": _as_utc(message.get("timestamp")), "raw_stage": _stage_from(message.get("sleep_level"))}
        for message in messages
    ]
    transitions = pd.DataFrame(rows).dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    transitions = transitions.drop_duplicates("timestamp_utc", keep="last")
    if transitions.empty:
        return [], None
    summary = _match_summary(transitions, summaries, timezone_name)
    if summary is not None:
        start, end = summary.sleep_start_utc, summary.sleep_end_utc
        sleep_date = summary.sleep_date
    else:
        start, end = transitions.timestamp_utc.iloc[0], transitions.timestamp_utc.iloc[-1]
        sleep_date = _local(transitions.timestamp_utc.iloc[-1], timezone_name).date()
    records: list[dict[str, object]] = []
    points = transitions.to_dict("records")
    for index, point in enumerate(points):
        segment_start = max(point["timestamp_utc"], start)
        segment_end = points[index + 1]["timestamp_utc"] if index + 1 < len(points) else end
        segment_end = min(segment_end, end)
        if segment_end <= segment_start:
            continue
        records.append({
            "segment_id": f"{sleep_id}:{len(records)}", "sleep_id": sleep_id, "sleep_date": sleep_date,
            "start_utc": segment_start, "end_utc": segment_end,
            "start_local": _local(segment_start, timezone_name), "end_local": _local(segment_end, timezone_name),
            "raw_stage": point["raw_stage"], "duration_minutes": (segment_end - segment_start).total_seconds() / 60,
            "fit_coverage": True, "summary_coverage": summary is not None,
        })
    return records, summary.to_dict() if summary is not None else None


def _monitoring_timestamp(message: dict[str, object], anchor: int | None) -> tuple[pd.Timestamp | pd.NaT, int | None]:
    full = _fit_seconds(message.get("timestamp")) if message.get("timestamp") is not None else None
    current = full if full is not None else reconstruct_timestamp_16(message.get("timestamp_16"), anchor)
    return _from_fit_seconds(current), current if current is not None else anchor


def decode_monitoring_messages(messages: list[dict[str, object]]) -> pd.DataFrame:
    """Decode monitoring messages and reconstruct compressed timestamps in order."""
    anchor: int | None = None
    rows: list[dict[str, object]] = []
    for message in messages:
        timestamp, anchor = _monitoring_timestamp(message, anchor)
        if pd.isna(timestamp):
            continue
        rows.append({
            "timestamp_utc": timestamp,
            "heart_rate": pd.to_numeric(message.get("heart_rate"), errors="coerce"),
            "activity_type": message.get("activity_type"),
            "intensity": pd.to_numeric(message.get("intensity", message.get("current_activity_type_intensity")), errors="coerce"),
            "steps": pd.to_numeric(message.get("steps"), errors="coerce"),
        })
    return pd.DataFrame(rows)


def _timestamped_signal(messages: list[dict[str, object]], value_field: str, output_field: str) -> pd.DataFrame:
    rows = []
    for message in messages:
        raw_timestamp = message.get("timestamp") or message.get("stress_level_time")
        stamp = _from_fit_seconds(int(raw_timestamp)) if isinstance(raw_timestamp, (int, np.integer)) else _as_utc(raw_timestamp)
        if pd.isna(stamp):
            continue
        rows.append({"timestamp_utc": stamp, output_field: pd.to_numeric(message.get(value_field), errors="coerce")})
    return pd.DataFrame(rows)


def decode_monitoring_fit(messages: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    """Return the permitted physiological signals from a monitoring_b FIT file."""
    frames = [
        decode_monitoring_messages(messages.get("monitoring_mesgs", [])),
        _timestamped_signal(messages.get("respiration_rate_mesgs", []), "respiration_rate", "respiration_rate"),
        _timestamped_signal(messages.get("stress_level_mesgs", []), "stress_level_value", "stress_level"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty(["timestamp_utc", "heart_rate", "respiration_rate", "stress_level"])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    aggregations = {column: "last" for column in combined.columns if column != "timestamp_utc"}
    return combined.groupby("timestamp_utc", as_index=False).agg(aggregations).sort_values("timestamp_utc")


def _decode_uploaded_fit(export_dir: str | Path, summaries: pd.DataFrame, timezone_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    segment_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    fit_nights: list[dict[str, object]] = []
    sleep_id = 0
    relevant_dates = set(summaries.sleep_date) if not summaries.empty else set()
    for archive_path in _uploaded_archives(export_dir):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".fit"):
                    continue
                try:
                    data = archive.read(member)
                    file_id = _read_file_id(data)
                except Exception:  # Corrupt/non-FIT entries must not block an export analysis.
                    continue
                kind = str(file_id.get("type")) if file_id else None
                created = _as_utc(file_id.get("time_created")) if file_id else pd.NaT
                created_date = _local(created, timezone_name).date() if not pd.isna(created) else None
                if kind not in {"sleep", "monitoring_b"}:
                    continue
                # Monitoring files are daily chunks. A one-day margin captures an overnight
                # window while avoiding a costly decode of unrelated all-day files.
                if kind == "monitoring_b" and relevant_dates and created_date not in relevant_dates and (created_date + timedelta(days=1) if created_date else None) not in relevant_dates:
                    continue
                try:
                    messages = _decode_fit(data)
                except Exception:
                    continue
                if kind == "sleep":
                    rows, summary = transition_messages_to_segments(messages.get("sleep_level_mesgs", []), summaries, timezone_name, sleep_id)
                    if rows:
                        segment_rows.extend(rows)
                        fit_nights.append({
                            "sleep_id": sleep_id, "sleep_date": rows[0]["sleep_date"],
                            "fit_coverage": True, "summary_coverage": summary is not None,
                        })
                        sleep_id += 1
                elif kind == "monitoring_b":
                    frame = decode_monitoring_fit(messages)
                    if not frame.empty:
                        signal_frames.append(frame)
    segments = pd.DataFrame(segment_rows)
    signals = pd.concat(signal_frames, ignore_index=True, sort=False) if signal_frames else _empty(["timestamp_utc", "heart_rate", "respiration_rate", "stress_level"])
    if not signals.empty:
        signals = signals.groupby("timestamp_utc", as_index=False).last().sort_values("timestamp_utc")
    return segments, signals, pd.DataFrame(fit_nights)


def _assign_signals_to_nights(signals: pd.DataFrame, nights: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    if signals.empty or nights.empty:
        return _empty(["sleep_date", "timestamp_utc", "timestamp_local", "heart_rate", "respiration_rate", "stress_level", "activity_type", "intensity", "steps"])
    rows = []
    for night in nights.itertuples():
        match = signals[(signals.timestamp_utc >= night.sleep_start_utc) & (signals.timestamp_utc <= night.sleep_end_utc)].copy()
        if match.empty:
            continue
        match["sleep_date"] = night.sleep_date
        match["timestamp_local"] = match.timestamp_utc.dt.tz_convert(timezone_name)
        rows.append(match)
    return pd.concat(rows, ignore_index=True) if rows else _empty(["sleep_date", "timestamp_utc"])


def identify_light_islands(segments: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Describe every Garmin awake-light-awake interval without changing its label."""
    records = []
    for sleep_date, group in segments.sort_values("start_utc").groupby("sleep_date"):
        group = group.reset_index(drop=True)
        stable_light = group[(group.raw_stage == "light") & ~((group.raw_stage.shift(1) == "awake") & (group.raw_stage.shift(-1) == "awake"))]
        for index in range(1, len(group) - 1):
            before, island, after = group.iloc[index - 1], group.iloc[index], group.iloc[index + 1]
            if not (before.raw_stage == "awake" and island.raw_stage == "light" and after.raw_stage == "awake"):
                continue
            record: dict[str, object] = {
                "sleep_date": sleep_date, "segment_id": island.segment_id,
                "start_utc": island.start_utc, "end_utc": island.end_utc,
                "duration_minutes": island.duration_minutes, "short_10_minutes_or_less": island.duration_minutes <= 10,
                "preceding_stage": before.raw_stage, "following_stage": after.raw_stage,
            }
            available = 0
            awake_like = 0
            for channel in ("heart_rate", "respiration_rate", "stress_level"):
                island_values = signals[(signals.sleep_date == sleep_date) & (signals.timestamp_utc >= island.start_utc) & (signals.timestamp_utc < island.end_utc)][channel].dropna()
                awake_values = signals[(signals.sleep_date == sleep_date) & (((signals.timestamp_utc >= before.start_utc) & (signals.timestamp_utc < before.end_utc)) | ((signals.timestamp_utc >= after.start_utc) & (signals.timestamp_utc < after.end_utc)))][channel].dropna()
                stable_values = pd.Series(dtype=float)
                for light in stable_light.itertuples():
                    stable_values = pd.concat([stable_values, signals[(signals.sleep_date == sleep_date) & (signals.timestamp_utc >= light.start_utc) & (signals.timestamp_utc < light.end_utc)][channel].dropna()])
                record[f"{channel}_samples"] = len(island_values)
                record[f"{channel}_island_median"] = island_values.median() if len(island_values) else np.nan
                if len(island_values) >= 3 and len(awake_values) >= 3 and len(stable_values) >= 3:
                    available += 1
                    island_median = island_values.median()
                    if abs(island_median - awake_values.median()) < abs(island_median - stable_values.median()):
                        awake_like += 1
            record["physiological_channels_available"] = available
            record["physiological_channels_awake_like"] = awake_like
            record["evidence_adjusted_candidate"] = awake_like >= 2
            records.append(record)
    return pd.DataFrame(records)


def load_awake_overrides(path: str | Path, timezone_name: str) -> pd.DataFrame:
    """Load ignored private manual intervals; missing files intentionally mean no overrides."""
    path = Path(path)
    if not path.exists():
        return _empty(["sleep_date", "start_utc", "end_utc", "state"])
    frame = pd.read_csv(path, usecols=lambda column: column in {"sleep_date", "start_local", "end_local", "state"})
    required = {"sleep_date", "start_local", "end_local", "state"}
    if not required <= set(frame):
        raise ValueError("sleep_awake_overrides.csv requires sleep_date, start_local, end_local, and state")
    frame["sleep_date"] = pd.to_datetime(frame.sleep_date).dt.date
    for source, target in (("start_local", "start_utc"), ("end_local", "end_utc")):
        frame[target] = pd.to_datetime(frame[source]).dt.tz_localize(timezone_name, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    frame["state"] = frame.state.str.lower()
    return frame.loc[frame.state.isin(STAGES) & (frame.end_utc > frame.start_utc), ["sleep_date", "start_utc", "end_utc", "state"]]


def scenario_segments(segments: pd.DataFrame, islands: pd.DataFrame, overrides: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Create a labelled sensitivity timeline. Raw labels are always retained."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    candidates = set(islands.loc[islands.evidence_adjusted_candidate, "segment_id"]) if not islands.empty else set()
    all_islands = set(islands.segment_id) if not islands.empty else set()
    rows = []
    for segment in segments.itertuples():
        cuts = {segment.start_utc, segment.end_utc}
        matching = overrides[(overrides.sleep_date == segment.sleep_date) & (overrides.end_utc > segment.start_utc) & (overrides.start_utc < segment.end_utc)] if scenario != "raw" else overrides.iloc[0:0]
        for override in matching.itertuples():
            cuts.update({max(segment.start_utc, override.start_utc), min(segment.end_utc, override.end_utc)})
        cuts = sorted(cuts)
        derived = segment.raw_stage
        if scenario == "evidence_adjusted" and segment.segment_id in candidates:
            derived = "awake"
        if scenario == "bracketed_upper_bound" and segment.segment_id in all_islands:
            derived = "awake"
        for start, end in zip(cuts, cuts[1:]):
            state = derived
            for override in matching.itertuples():
                if override.start_utc <= start and override.end_utc >= end:
                    state = override.state
            rows.append({"sleep_date": segment.sleep_date, "segment_id": segment.segment_id, "start_utc": start, "end_utc": end, "raw_stage": segment.raw_stage, "stage": state, "scenario": scenario, "duration_minutes": (end - start).total_seconds() / 60})
    return pd.DataFrame(rows)


def derive_awakenings(scenario_timeline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate awakenings, return-to-sleep, and nightly outcomes for one scenario."""
    scenario_timeline = scenario_timeline.copy()
    if "duration_minutes" not in scenario_timeline:
        scenario_timeline["duration_minutes"] = (scenario_timeline.end_utc - scenario_timeline.start_utc).dt.total_seconds() / 60
    events, metrics = [], []
    for sleep_date, group in scenario_timeline.sort_values("start_utc").groupby("sleep_date"):
        group = group.reset_index(drop=True)
        awake = group[group.stage == "awake"]
        for event_no, event in enumerate(awake.itertuples(), start=1):
            following = group[group.start_utc >= event.end_utc]
            consecutive = 0.0
            for item in following.itertuples():
                if item.stage == "awake":
                    break
                consecutive += item.duration_minutes
                if consecutive >= 15:
                    break
            events.append({"sleep_date": sleep_date, "scenario": event.scenario, "awakening_number": event_no,
                "start_utc": event.start_utc, "end_utc": event.end_utc, "duration_minutes": event.duration_minutes,
                "return_to_sleep": consecutive >= 15, "post_awakening_sleep_minutes": consecutive,
                "prolonged_30_minutes": event.duration_minutes >= 30, "prolonged_60_minutes": event.duration_minutes >= 60,
                "prolonged_90_minutes": event.duration_minutes >= 90})
        awake_minutes = awake.duration_minutes.sum()
        asleep_minutes = group.loc[group.stage != "awake", "duration_minutes"].sum()
        window_minutes = group.duration_minutes.sum()
        row = {"sleep_date": sleep_date, "scenario": group.scenario.iloc[0], "awakening_count": len(awake),
               "wake_after_sleep_onset_minutes": awake_minutes, "longest_awakening_minutes": awake.duration_minutes.max() if len(awake) else 0.0,
               "sleep_minutes": asleep_minutes, "window_minutes": window_minutes,
               "sleep_efficiency": asleep_minutes / window_minutes if window_minutes else np.nan}
        for threshold in (30, 60, 90):
            row[f"prolonged_wake_{threshold}_minutes"] = bool((awake.duration_minutes >= threshold).any())
        metrics.append(row)
    return pd.DataFrame(events), pd.DataFrame(metrics)


def _join_daylio(nights: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    entries = load_daylio_entries(config.daylio_export_csv, config.timezone, config.approved_activity_tags)
    tag_columns = [column for column in entries if column.startswith("tag_")]
    daily = entries.groupby("calendar_date", as_index=False)[tag_columns].max()
    joined = nights.merge(daily, how="left", left_on="sleep_date", right_on="calendar_date").drop(columns="calendar_date", errors="ignore")
    prior = daily.copy()
    prior.calendar_date = prior.calendar_date + pd.Timedelta(days=1)
    prior = prior.rename(columns={column: f"prior_day_{column}" for column in tag_columns})
    return joined.merge(prior, how="left", left_on="sleep_date", right_on="calendar_date").drop(columns="calendar_date", errors="ignore")


def load_local_sleep_analysis(config_path: str | Path = "config.local.toml") -> SleepAnalysis:
    """Load local Garmin and Daylio exports into the five safe sleep-analysis tables."""
    config = load_config(config_path)
    summaries = load_sleep_summaries(config.garmin_export_dir, config.timezone)
    segments, all_signals, fit_nights = _decode_uploaded_fit(config.garmin_export_dir, summaries, config.timezone)
    if not summaries.empty:
        summary_nights = summaries.copy()
        summary_nights["fit_coverage"] = summary_nights.sleep_date.isin(fit_nights.sleep_date if not fit_nights.empty else [])
        nights = summary_nights.merge(fit_nights[["sleep_date", "sleep_id"]] if not fit_nights.empty else _empty(["sleep_date", "sleep_id"]), how="left", on="sleep_date")
    else:
        nights = fit_nights.copy()
    if not segments.empty:
        fit_only = segments.loc[~segments.sleep_date.isin(nights.sleep_date), ["sleep_date", "start_utc", "end_utc"]].groupby("sleep_date", as_index=False).agg(sleep_start_utc=("start_utc", "min"), sleep_end_utc=("end_utc", "max"))
        fit_only["fit_coverage"], fit_only["summary_coverage"] = True, False
        nights = pd.concat([nights, fit_only], ignore_index=True, sort=False)
    nights = nights.sort_values("sleep_date").drop_duplicates("sleep_date", keep="last").reset_index(drop=True)
    signals = _assign_signals_to_nights(all_signals, nights.dropna(subset=["sleep_start_utc", "sleep_end_utc"]), config.timezone)
    islands = identify_light_islands(segments, signals) if not segments.empty else _empty(["sleep_date", "segment_id"])
    overrides = load_awake_overrides(Path(config.daylio_export_csv).parent / "sleep_awake_overrides.csv", config.timezone)
    awakening_frames, metric_frames = [], []
    for scenario in SCENARIOS:
        timeline = scenario_segments(segments, islands, overrides, scenario) if not segments.empty else pd.DataFrame()
        events, metrics = derive_awakenings(timeline) if not timeline.empty else (pd.DataFrame(), pd.DataFrame())
        awakening_frames.append(events)
        metric_frames.append(metrics)
    awakenings = pd.concat(awakening_frames, ignore_index=True) if any(not frame.empty for frame in awakening_frames) else _empty(["sleep_date", "scenario"])
    for metrics in metric_frames:
        if metrics.empty:
            continue
        scenario = metrics.scenario.iloc[0]
        renamed = metrics.drop(columns="scenario").rename(columns={column: f"{scenario}_{column}" for column in metrics.columns if column != "sleep_date"})
        nights = nights.merge(renamed, on="sleep_date", how="left")
    nights = _join_daylio(nights, config)
    for frame in (nights, segments, signals, awakenings, islands):
        assert_public_frame_safe(frame)
    return SleepAnalysis(nights=nights, segments=segments, signals=signals, awakenings=awakenings, light_islands=islands)


def sleep_tag_associations(nights: pd.DataFrame, tag_columns: list[str] | None = None) -> pd.DataFrame:
    """Descriptive and inferential comparisons for every Daylio tag and sleep metric."""
    tag_columns = tag_columns or [column for column in nights if column.startswith("tag_")]
    metrics = [column for column in nights if column.startswith("raw_") and column.endswith(("wake_after_sleep_onset_minutes", "longest_awakening_minutes", "sleep_efficiency", "sleep_minutes", "prolonged_wake_30_minutes", "prolonged_wake_60_minutes", "prolonged_wake_90_minutes"))]
    rows = []
    for tag in tag_columns:
        present = pd.to_numeric(nights[tag], errors="coerce").fillna(0).astype(bool)
        for metric in metrics:
            values = pd.to_numeric(nights[metric], errors="coerce")
            yes, no = values[present].dropna(), values[~present].dropna()
            descriptive = len(yes) < 5 or len(no) < 5
            row = {"tag": tag.removeprefix("tag_"), "metric": metric.removeprefix("raw_"), "tag_present_n": len(yes), "tag_absent_n": len(no), "median_difference": yes.median() - no.median(), "descriptive_only": descriptive}
            if "prolonged_wake_" in metric:
                table = [[int(yes.sum()), int((~yes.astype(bool)).sum())], [int(no.sum()), int((~no.astype(bool)).sum())]]
                row["effect_size"] = (yes.mean() - no.mean()) if len(yes) and len(no) else np.nan
                row["test"] = "Fisher exact"
                row["p_value"] = fisher_exact(table).pvalue if not descriptive else np.nan
            else:
                row["effect_size"] = (yes.median() - no.median()) / values.std(ddof=0) if values.std(ddof=0) else np.nan
                row["test"] = "Mann–Whitney U"
                row["p_value"] = mannwhitneyu(yes, no, alternative="two-sided").pvalue if not descriptive else np.nan
            rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        valid = result.p_value.notna()
        result["fdr_q_value"] = np.nan
        result.loc[valid, "fdr_q_value"] = multipletests(result.loc[valid, "p_value"], method="fdr_bh")[1]
    return result
