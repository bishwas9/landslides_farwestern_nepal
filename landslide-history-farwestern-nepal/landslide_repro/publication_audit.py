from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .common import LOGGER, output_path, write_csv, write_json


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _add(checks: list[dict], section: str, check: str, passed: bool, observed, requirement: str, blocking: bool = True) -> None:
    checks.append(
        {
            "section": section,
            "check": check,
            "passed": bool(passed),
            "blocking": bool(blocking),
            "observed": observed,
            "requirement": requirement,
        }
    )


def run(cfg: Mapping, quick: bool = False) -> bool:
    """Audit the completed outputs used by the manuscript and figures."""
    root = Path(cfg["paths"]["output_dir"])
    checks: list[dict] = []
    results: list[dict] = []

    recurrence = _read(root / "tables" / "recurrence_focal_effects.csv")
    required_terms = ["prior_active_years_z", "[T.1_year]", "[T.2_years]", "[T.3_5_years]", "[T.6_10_years]"]
    recurrence_ok = recurrence is not None and all(recurrence["term"].astype(str).str.contains(term, regex=False).any() for term in required_terms)
    _add(checks, "recurrence", "focal effects and uncertainty", recurrence_ok, 0 if recurrence is None else len(recurrence), "all prespecified focal effects with intervals")
    if recurrence is not None:
        for _, row in recurrence.iterrows():
            results.append(
                {
                    "analysis": "recurrence",
                    "metric": str(row.get("display_term", row["term"])),
                    "estimate": float(row["odds_ratio"]),
                    "lower": float(row["interval_low"]),
                    "upper": float(row["interval_high"]),
                    "p_value": np.nan,
                    "units": "odds ratio",
                }
            )

    temporal_null = _read(root / "randomization" / "temporal_fixed_margin_null.csv")
    temporal_summary = _read(root / "tables" / "temporal_fixed_margin_summary.csv")
    expected_temporal = 99 if quick else 9999
    temporal_count = 0 if temporal_null is None else len(temporal_null)
    _add(checks, "temporal", "retained fixed-margin draws", temporal_count == expected_temporal, temporal_count, str(expected_temporal))
    if not quick and temporal_null is not None and "chain" in temporal_null:
        counts = temporal_null.groupby("chain").size().to_dict()
        _add(checks, "temporal", "balanced chains", counts == {1: 3333, 2: 3333, 3: 3333}, str(counts), "{1:3333, 2:3333, 3:3333}")
    mixing = _read(root / "randomization" / "temporal_chain_mixing_diagnostics.csv")
    _add(checks, "temporal", "mixing diagnostics", mixing is not None and len(mixing) >= 7, 0 if mixing is None else len(mixing), "at least seven prespecified statistics")
    if temporal_summary is not None:
        for _, row in temporal_summary.iterrows():
            results.append(
                {
                    "analysis": "temporal fixed-margin null",
                    "metric": str(row["statistic"]),
                    "estimate": float(row["observed"]),
                    "lower": float(row["null_q025"]),
                    "upper": float(row["null_q975"]),
                    "p_value": float(row["p_upper"]),
                    "units": "probability or risk ratio; interval is null 95% interval",
                }
            )

    ame = _read(root / "tables" / "Figure4_probability_scale_AME_two_way_cluster.csv")
    contrasts = _read(root / "tables" / "Figure4_AME_contrasts_vs_gt10_two_way_cluster.csv")
    joint = _read(root / "tables" / "Figure4_joint_interaction_tests_two_way_cluster.csv")
    durations = [] if ame is None else sorted(pd.to_numeric(ame["duration_days"]).astype(int).unique().tolist())
    _add(checks, "rainfall", "seven durations", durations == [1, 3, 5, 7, 10, 15, 30], str(durations), "[1, 3, 5, 7, 10, 15, 30]")
    _add(checks, "rainfall", "AME contrast table", contrasts is not None and len(contrasts) == 28, 0 if contrasts is None else len(contrasts), "28 history-by-duration contrasts")
    _add(checks, "rainfall", "joint interaction tests", joint is not None and len(joint) == 7, 0 if joint is None else len(joint), "7 duration-level joint tests")
    if contrasts is not None:
        focal = contrasts.loc[contrasts["duration_days"].eq(10)]
        for _, row in focal.iterrows():
            results.append(
                {
                    "analysis": "rainfall-history interaction",
                    "metric": f"Rx10 {row['history_state']} vs >10 years",
                    "estimate": float(row["AME_contrast_percentage_points"]),
                    "lower": float(row["CI95_low_percentage_points"]),
                    "upper": float(row["CI95_high_percentage_points"]),
                    "p_value": float(row["p_contrast"]),
                    "units": "percentage-point AME contrast per +1 local-SD rainfall",
                }
            )

    spatial_null = _read(root / "randomization" / ("time_label_permutation_null_quick.csv" if quick else "time_label_permutation_null_9999.csv"))
    spatial_summary = _read(root / "tables" / ("observed_vs_time_permutation_quick.csv" if quick else "observed_vs_time_permutation_9999.csv"))
    expected_spatial = 99 if quick else 9999
    counts = {} if spatial_null is None else spatial_null.groupby("gap_model").size().to_dict()
    expected_categories = {"overall", "1_year", "2_years", "3_5_years", "6_10_years", "gt10_years"}
    spatial_complete = set(counts) == expected_categories and all(value == expected_spatial for value in counts.values())
    _add(checks, "spatial", "time-label permutations per category", spatial_complete, str(counts), f"{expected_spatial} in each of six categories")
    spatial_spec_path = root / "models" / "spatial_randomization_specification.json"
    spatial_spec_ok = spatial_spec_path.exists()
    if spatial_spec_ok:
        spec = json.loads(spatial_spec_path.read_text(encoding="utf-8"))
        spatial_spec_ok = int(spec.get("permutations", -1)) == expected_spatial and bool(spec.get("quick_mode")) == bool(quick)
    _add(
        checks,
        "spatial",
        "spatial randomization specification",
        spatial_spec_ok,
        str(spatial_spec_path.relative_to(root)) if spatial_spec_path.exists() else "missing",
        "specification JSON written by the spatial stage",
    )
    if spatial_summary is not None:
        for _, row in spatial_summary.iterrows():
            results.append(
                {
                    "analysis": "spatial chronology null",
                    "metric": str(row["gap_model"]),
                    "estimate": float(row["observed_median_m"]),
                    "lower": float(row["null_CI_low_m"]),
                    "upper": float(row["null_CI_high_m"]),
                    "p_value": float(row["empirical_p_one_sided"]),
                    "units": "metres; interval is null 95% interval",
                }
            )

    pooled = _read(root / "prediction" / "forward_metrics_pooled.csv")
    yearly = _read(root / "prediction" / "forward_metrics_by_year.csv")
    capture = _read(root / "prediction" / "top_fraction_performance_pooled_annual_budget.csv")
    models = set() if pooled is None else set(pooled["model"])
    expected_models = {"P0_static", "P1_history", "P2_history_lagged_rain"}
    _add(checks, "prediction", "three prospective predictor sets", models == expected_models, str(sorted(models)), str(sorted(expected_models)))
    pooled_counts_ok = pooled is not None and pooled["n"].eq(265181).all() and pooled["positives"].eq(2812).all()
    _add(checks, "prediction", "held-out sample", pooled_counts_ok, "missing" if pooled is None else f"n={pooled['n'].unique().tolist()}, positives={pooled['positives'].unique().tolist()}", "n=265181 and positives=2812 for each model")
    years = [] if yearly is None else sorted(pd.to_numeric(yearly["test_year"]).astype(int).unique().tolist())
    expected_years = list(range(2006, 2019)) if not quick else years
    _add(checks, "prediction", "forward holdout years", years == expected_years, str(years), str(expected_years))
    if pooled is not None:
        for _, row in pooled.iterrows():
            results.append(
                {
                    "analysis": "forward prediction",
                    "metric": f"{row['model']} pooled average precision",
                    "estimate": float(row["average_precision"]),
                    "lower": np.nan,
                    "upper": np.nan,
                    "p_value": np.nan,
                    "units": "average precision",
                }
            )
    if capture is not None:
        for _, row in capture.loc[np.isclose(capture["top_fraction"], 0.05)].iterrows():
            results.append(
                {
                    "analysis": "forward prediction",
                    "metric": f"{row['model']} annual-budget top-5% capture",
                    "estimate": 100 * float(row["positive_capture_rate"]),
                    "lower": np.nan,
                    "upper": np.nan,
                    "p_value": np.nan,
                    "units": "percent of held-out active slope-unit years captured",
                }
            )

    frame = pd.DataFrame(checks)
    write_csv(frame, output_path(cfg, "audit", "publication_readiness_checks.csv"))
    write_csv(pd.DataFrame(results), output_path(cfg, "tables", "main_text_numerical_results.csv"))
    blockers = frame.loc[frame["blocking"] & ~frame["passed"]]
    status = {
        "publication_outputs_complete": bool(blockers.empty),
        "blocking_checks_failed": int(len(blockers)),
        "blocking_check_names": blockers.apply(lambda row: f"{row['section']}: {row['check']}", axis=1).tolist(),
        "reproducibility_audit": "complete" if blockers.empty else "incomplete",
        "quick_mode": bool(quick),
    }
    write_json(status, output_path(cfg, "audit", "publication_readiness_summary.json"))
    LOGGER.info("Publication audit complete: %d blocking checks failed", len(blockers))
    return bool(blockers.empty)
