"""Association-first statistical routines with temporal uncertainty estimates."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests


def _paired(frame: pd.DataFrame, feature: str, outcome: str) -> pd.DataFrame:
    return frame[[feature, outcome]].dropna().astype(float)


def block_bootstrap_spearman(
    pairs: pd.DataFrame,
    block_size_days: int = 7,
    resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percentile interval using contiguous observations as blocks."""
    if len(pairs) < 4:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(pairs)
    starts = np.arange(n)
    for _ in range(resamples):
        chosen: list[int] = []
        while len(chosen) < n:
            start = int(rng.choice(starts))
            chosen.extend((start + offset) % n for offset in range(block_size_days))
        sample = pairs.iloc[chosen[:n]]
        coefficient = spearmanr(sample.iloc[:, 0], sample.iloc[:, 1]).statistic
        if np.isfinite(coefficient):
            values.append(float(coefficient))
    if not values:
        return (np.nan, np.nan)
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def spearman_associations(
    frame: pd.DataFrame,
    features: Iterable[str],
    outcome: str = "mood_mean",
    min_observations: int = 60,
    block_size_days: int = 7,
    bootstrap_resamples: int = 2000,
) -> pd.DataFrame:
    """Test same-day rank associations and adjust the full feature family for FDR."""
    rows: list[dict[str, object]] = []
    for feature in features:
        pairs = _paired(frame, feature, outcome)
        if len(pairs) < min_observations or pairs[feature].nunique() < 2 or pairs[outcome].nunique() < 2:
            rows.append({"feature": feature, "n": len(pairs), "status": "insufficient_data"})
            continue
        test = spearmanr(pairs[feature], pairs[outcome])
        ci_low, ci_high = block_bootstrap_spearman(pairs, block_size_days, bootstrap_resamples)
        rows.append(
            {
                "feature": feature,
                "n": len(pairs),
                "effect_size": float(test.statistic),
                "p_value": float(test.pvalue),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "status": "tested",
            }
        )
    result = pd.DataFrame(rows)
    result["p_value_fdr"] = np.nan
    tested = result["status"].eq("tested")
    if tested.any():
        result.loc[tested, "p_value_fdr"] = multipletests(result.loc[tested, "p_value"], method="fdr_bh")[1]
    return result.sort_values(["status", "p_value_fdr", "feature"], na_position="last").reset_index(drop=True)


def group_comparisons(
    frame: pd.DataFrame,
    feature: str,
    outcome: str = "mood_mean",
    min_group_size: int = 20,
) -> pd.DataFrame:
    """Compare adjacent observed mood groups with a non-parametric test."""
    pairs = _paired(frame, feature, outcome)
    groups = [(value, values[feature].to_numpy()) for value, values in pairs.groupby(outcome)]
    rows: list[dict[str, object]] = []
    for (low_value, low), (high_value, high) in zip(groups, groups[1:]):
        if len(low) < min_group_size or len(high) < min_group_size:
            rows.append(
                {
                    "feature": feature,
                    "lower_mood": low_value,
                    "higher_mood": high_value,
                    "n_lower": len(low),
                    "n_higher": len(high),
                    "status": "insufficient_group_size",
                }
            )
            continue
        test = mannwhitneyu(low, high, alternative="two-sided")
        rows.append(
            {
                "feature": feature,
                "lower_mood": low_value,
                "higher_mood": high_value,
                "n_lower": len(low),
                "n_higher": len(high),
                "effect_size": float(2 * test.statistic / (len(low) * len(high)) - 1),
                "p_value": float(test.pvalue),
                "status": "tested",
            }
        )
    return pd.DataFrame(rows)


def binary_activity_associations(
    frame: pd.DataFrame,
    indicators: Iterable[str],
    outcome: str = "mood_mean",
    min_group_size: int = 20,
) -> pd.DataFrame:
    """Compare mood on dates with and without approved binary activity indicators."""
    rows: list[dict[str, object]] = []
    for indicator in indicators:
        pairs = frame[[indicator, outcome]].dropna()
        absent = pairs.loc[pairs[indicator].eq(0), outcome].astype(float)
        present = pairs.loc[pairs[indicator].gt(0), outcome].astype(float)
        if len(absent) < min_group_size or len(present) < min_group_size:
            rows.append(
                {
                    "indicator": indicator,
                    "n_absent": len(absent),
                    "n_present": len(present),
                    "status": "insufficient_group_size",
                }
            )
            continue
        test = mannwhitneyu(present, absent, alternative="two-sided")
        rows.append(
            {
                "indicator": indicator,
                "n_absent": len(absent),
                "n_present": len(present),
                "effect_size": float(2 * test.statistic / (len(present) * len(absent)) - 1),
                "p_value": float(test.pvalue),
                "status": "tested",
            }
        )
    result = pd.DataFrame(rows)
    result["p_value_fdr"] = np.nan
    tested = result["status"].eq("tested")
    if tested.any():
        result.loc[tested, "p_value_fdr"] = multipletests(result.loc[tested, "p_value"], method="fdr_bh")[1]
    return result


def lagged_associations(
    frame: pd.DataFrame,
    features: Iterable[str],
    outcome: str = "mood_mean",
    max_lag_days: int = 7,
    min_observations: int = 60,
) -> pd.DataFrame:
    """Test Garmin(t-lag) against mood(t); positive lag means Garmin precedes mood."""
    rows: list[dict[str, object]] = []
    for feature in features:
        family_rows: list[dict[str, object]] = []
        for lag in range(-max_lag_days, max_lag_days + 1):
            pairs = pd.DataFrame({feature: frame[feature].shift(lag), outcome: frame[outcome]}).dropna()
            if len(pairs) < min_observations or pairs[feature].nunique() < 2 or pairs[outcome].nunique() < 2:
                family_rows.append({"feature": feature, "lag_days": lag, "n": len(pairs), "status": "insufficient_data"})
                continue
            test = spearmanr(pairs[feature], pairs[outcome])
            family_rows.append(
                {
                    "feature": feature,
                    "lag_days": lag,
                    "n": len(pairs),
                    "effect_size": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "status": "tested",
                }
            )
        family = pd.DataFrame(family_rows)
        family["p_value_fdr"] = np.nan
        tested = family["status"].eq("tested")
        if tested.any():
            family.loc[tested, "p_value_fdr"] = multipletests(family.loc[tested, "p_value"], method="fdr_bh")[1]
        rows.extend(family.to_dict("records"))
    return pd.DataFrame(rows)


def _adjusted_design(frame: pd.DataFrame, feature: str, outcome: str) -> tuple[pd.Series, pd.DataFrame]:
    data = frame[["calendar_date", feature, outcome]].dropna().copy()
    dates = pd.to_datetime(data["calendar_date"])
    ordinal = data[outcome].round().clip(1, 5).astype(int)
    day_index = (dates - dates.min()).dt.days
    day_of_year = dates.dt.dayofyear
    weekday = pd.get_dummies(dates.dt.weekday, prefix="weekday", drop_first=True, dtype=float)
    design = pd.DataFrame(
        {
            feature: (data[feature] - data[feature].mean()) / data[feature].std(ddof=0),
            "time_trend": (day_index - day_index.mean()) / max(day_index.std(ddof=0), 1),
            "annual_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "annual_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        },
        index=data.index,
    )
    weekday.index = data.index
    return ordinal, pd.concat([design, weekday], axis=1)


def adjusted_ordinal_association(
    frame: pd.DataFrame,
    feature: str,
    outcome: str = "mood_mean",
    bootstrap_resamples: int = 0,
    block_size_days: int = 7,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Fit an adjusted ordered-logistic association model for one feature."""
    outcome_values, design = _adjusted_design(frame, feature, outcome)
    if len(outcome_values) < 60 or outcome_values.nunique() < 2 or design[feature].nunique() < 2:
        return {"feature": feature, "n": len(outcome_values), "status": "insufficient_data"}
    try:
        model = OrderedModel(outcome_values, design, distr="logit")
        fitted = model.fit(method="bfgs", disp=False)
    except (ValueError, np.linalg.LinAlgError) as error:
        return {"feature": feature, "n": len(outcome_values), "status": f"model_failed: {error}"}
    result: dict[str, float | int | str] = {
        "feature": feature,
        "n": len(outcome_values),
        "odds_ratio_per_sd": float(np.exp(fitted.params[feature])),
        "p_value": float(fitted.pvalues[feature]),
        "status": "tested",
    }
    if bootstrap_resamples <= 0:
        return result

    source = frame[["calendar_date", feature, outcome]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    coefficients: list[float] = []
    for _ in range(bootstrap_resamples):
        chosen: list[int] = []
        while len(chosen) < len(source):
            start = int(rng.integers(0, len(source)))
            chosen.extend((start + offset) % len(source) for offset in range(block_size_days))
        sample = source.iloc[chosen[: len(source)]].reset_index(drop=True)
        try:
            sample_outcome, sample_design = _adjusted_design(sample, feature, outcome)
            sample_model = OrderedModel(sample_outcome, sample_design, distr="logit")
            sample_fit = sample_model.fit(method="bfgs", disp=False)
        except (ValueError, np.linalg.LinAlgError):
            continue
        coefficient = sample_fit.params.get(feature)
        if coefficient is not None and np.isfinite(coefficient):
            coefficients.append(float(coefficient))
    if coefficients:
        low, high = np.quantile(coefficients, [0.025, 0.975])
        result["odds_ratio_ci_low"] = float(np.exp(low))
        result["odds_ratio_ci_high"] = float(np.exp(high))
        result["bootstrap_successes"] = len(coefficients)
    return result


def classify_findings(results: pd.DataFrame) -> pd.DataFrame:
    """Classify FDR-controlled association evidence without causal language."""
    ranked = results.copy()
    robust = ranked["status"].eq("tested") & ranked["p_value_fdr"].le(0.05) & (
        (ranked["ci_low"] > 0) | (ranked["ci_high"] < 0)
    )
    tentative = ranked["status"].eq("tested") & ranked["p_value_fdr"].le(0.10) & ~robust
    ranked["evidence"] = np.select([robust, tentative], ["robust", "tentative"], default="unsupported")
    return ranked


def summarize_variant_stability(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise whether raw, trailing, and baseline-deviation variants agree."""
    classified = classify_findings(results)
    classified["base_feature"] = classified["feature"].str.replace(
        r"_(trailing_7d|deviation_28d)$", "", regex=True
    )
    classified["variant"] = np.select(
        [
            classified["feature"].str.endswith("_trailing_7d"),
            classified["feature"].str.endswith("_deviation_28d"),
        ],
        ["trailing_7d", "deviation_28d"],
        default="raw",
    )
    rows = []
    for base_feature, group in classified.groupby("base_feature"):
        robust = group.loc[group["evidence"].eq("robust")]
        directions = set(np.sign(robust["effect_size"].dropna()))
        if len(robust) >= 2 and len(directions) == 1:
            stability = "stable"
        elif len(robust) == 1:
            stability = "single_variant"
        else:
            stability = "unsupported"
        rows.append(
            {
                "base_feature": base_feature,
                "tested_variants": int(group["status"].eq("tested").sum()),
                "robust_variants": len(robust),
                "direction_consistent": len(directions) <= 1,
                "stability": stability,
            }
        )
    return pd.DataFrame(rows).sort_values(["stability", "robust_variants", "base_feature"], ascending=[True, False, True])
