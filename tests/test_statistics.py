from __future__ import annotations

from garmin_daylio_analysis.statistics import binary_activity_associations, classify_findings, lagged_associations, spearman_associations, summarize_variant_stability
from garmin_daylio_analysis.synthetic import make_synthetic_analysis_daily


def test_same_day_associations_return_fdr_controlled_results():
    frame = make_synthetic_analysis_daily(days=120, seed=11)
    results = spearman_associations(
        frame,
        features=["steps", "resting_heart_rate"],
        min_observations=60,
        bootstrap_resamples=80,
    )
    assert set(results["feature"]) == {"steps", "resting_heart_rate"}
    assert results["p_value_fdr"].notna().all()
    assert classify_findings(results)["evidence"].notna().all()


def test_positive_lag_uses_prior_garmin_value():
    frame = make_synthetic_analysis_daily(days=140, seed=4)
    lags = lagged_associations(frame, ["steps"], max_lag_days=2, min_observations=60)
    one_day = lags.loc[lags["lag_days"].eq(1)].iloc[0]
    assert one_day["status"] == "tested"
    assert one_day["effect_size"] > 0


def test_approved_activity_indicator_analysis_enforces_group_sizes():
    frame = make_synthetic_analysis_daily(days=140, seed=4)
    result = binary_activity_associations(frame, ["tag_demo_outdoors"], min_group_size=20)
    assert result.loc[0, "status"] == "tested"
    assert result.loc[0, "p_value_fdr"] >= 0


def test_variant_stability_groups_feature_forms():
    frame = make_synthetic_analysis_daily(days=140, seed=4)
    results = spearman_associations(
        frame,
        ["steps", "steps_trailing_7d", "steps_deviation_28d"],
        min_observations=60,
        bootstrap_resamples=30,
    )
    stability = summarize_variant_stability(results)
    assert stability["base_feature"].tolist() == ["steps"]
    assert stability.loc[0, "tested_variants"] == 3
