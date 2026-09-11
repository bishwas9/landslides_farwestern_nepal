from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .common import LOGGER, output_path, write_csv, write_json


SUPPLEMENT_INDEX = [
    ("S1", "audit/input_manifest.csv", "Input file names, sizes, hashes, schemas, and provenance"),
    ("S2", "audit/panel_year_audit.csv", "Inventory support by observed year"),
    ("S3", "audit/history_exclusion_audit.csv", "Pre-first and 1994–1995 gap exclusions by analysis and year"),
    ("S4", "tables/recurrence_gap_support.csv", "Recurrence risk-set support and observed rates"),
    ("S5", "tables/recurrence_focal_effects.csv", "Mixed-effects recurrence-model focal odds ratios and uncertainty"),
    ("S6", "models/recurrence_all_coefficients.csv", "Complete recurrence model coefficients"),
    ("S7", "randomization/temporal_chain_diagnostics.csv", "Fixed-margin chain, burn-in, thinning, and margin diagnostics"),
    ("S7a", "randomization/temporal_chain_mixing_diagnostics.csv", "Split-Rhat, lag-1 autocorrelation, and approximate effective sample-size checks"),
    ("S8", "tables/temporal_fixed_margin_summary.csv", "Observed versus temporal fixed-margin null statistics"),
    ("S9", "randomization/temporal_fixed_margin_null.csv", "Full temporal-null draws"),
    ("S9a", "audit/rainfall_parent_material_exclusion.csv", "Audit confirming that valid UF1/UF2 records were retained in the Figure 4 sample"),
    ("S9b", "tables/rainfall_history_gap_support.csv", "History-state rows, outcomes, slope units, and observed rates for each rainfall duration"),
    ("S10", "tables/Figure4_probability_scale_AME_two_way_cluster.csv", "Probability-scale AMEs and pointwise intervals"),
    ("S11", "tables/Figure4_AME_contrasts_vs_gt10_two_way_cluster.csv", "AME contrasts versus >10 years, including multiplicity-adjusted p-values"),
    ("S12", "tables/Figure4_joint_interaction_tests_two_way_cluster.csv", "Joint rainfall × history interaction tests"),
    ("S12a", "tables/Figure4_multiplicity_adjusted_tests.csv", "Holm-adjusted p-values supplied separately from the reference-compatible Figure 4 tables"),
    ("S13", "models/rainfall_history_model_specification.json", "Complete rainfall-history estimator, sample, scaling, covariance, and harmonization specification"),
    ("S14", "audit/spatial_location_multiplicity.csv", "Spatial duplicate-location audit"),
    ("S14a", "audit/spatial_year_counts_before_after_dedup.csv", "Annual event-label totals before and after fixed-location de-duplication"),
    ("S15", "tables/observed_distinct_location_summary.csv", "Observed distinct-location distance summaries"),
    ("S16", "tables/observed_vs_time_permutation_9999.csv", "Observed-versus-spatial-null inference"),
    ("S17", "randomization/time_label_permutation_null_9999.csv", "Full time-label spatial-null draws"),
    ("S18", "audit/prediction_sample_exclusions.csv", "Explanation of prediction sample size by held-out year"),
    ("S18a", "audit/prediction_leakage_checks.csv", "Machine-readable expanding-window and lag-source leakage assertions"),
    ("S19", "prediction/forward_metrics_by_year.csv", "Year-specific forward-prediction metrics"),
    ("S20", "prediction/forward_metrics_pooled.csv", "Pooled held-out metrics"),
    ("S21", "prediction/incremental_performance.csv", "Incremental P0/P1/P2 performance"),
    ("S22", "prediction/top_fraction_performance.csv", "Operational top-fraction capture and precision"),
    ("S22a", "prediction/top_fraction_performance_pooled_annual_budget.csv", "Pooled capture after applying the inspection budget separately within every held-out year"),
    ("S23", "prediction/calibration_data.csv", "Calibration-bin data"),
    ("S24", "prediction/forward_predictions.csv", "All held-out probabilities and outcomes"),
    ("S25", "figures/figure_manifest.csv", "Analysis-derived Figures 3–6 and their numerical source files"),
    ("S26", "tables/main_text_numerical_results.csv", "Single checked table of numerical results used in the Results section and figure captions"),
    ("S27", "audit/publication_readiness_checks.csv", "Blocking reproducibility and publication-readiness checks"),
    ("S27a", "audit/publication_readiness_summary.json", "Machine-readable publication-readiness summary"),
    ("S29", "figures/FigureS1_recurrence_model_effects.pdf", "Supplementary logistic mixed-effects recurrence-model forest plot (vector)"),
    ("S29a", "figures/FigureS1_recurrence_model_effects.png", "Supplementary logistic mixed-effects recurrence-model forest plot (600 dpi)"),
    ("S30", "figures/FigureS2_temporal_chain_diagnostics.pdf", "Supplementary temporal-null mixing diagnostics (vector)"),
    ("S30a", "figures/FigureS2_temporal_chain_diagnostics.png", "Supplementary temporal-null mixing diagnostics (600 dpi)"),
    ("S31", "figures/FigureS3_prediction_calibration.pdf", "Supplementary forward-prediction calibration plot (vector)"),
    ("S31a", "figures/FigureS3_prediction_calibration.png", "Supplementary forward-prediction calibration plot (600 dpi)"),
]


def _comparison(reference: Path, candidate: Path, atol: float = 1e-8, rtol: float = 1e-6) -> dict:
    try:
        old = pd.read_csv(reference)
        new = pd.read_csv(candidate)
    except Exception as exc:
        return {"status": "read_error", "detail": str(exc)}
    if list(old.columns) != list(new.columns):
        return {
            "status": "schema_changed",
            "detail": f"reference columns={list(old.columns)}; new columns={list(new.columns)}",
            "reference_rows": len(old),
            "new_rows": len(new),
        }
    if len(old) != len(new):
        return {"status": "row_count_changed", "detail": "", "reference_rows": len(old), "new_rows": len(new)}
    key_columns = [column for column in old.columns if not pd.api.types.is_numeric_dtype(old[column])]
    if key_columns:
        old = old.sort_values(key_columns, kind="mergesort", na_position="first").reset_index(drop=True)
        new = new.sort_values(key_columns, kind="mergesort", na_position="first").reset_index(drop=True)
    numeric = [column for column in old.columns if pd.api.types.is_numeric_dtype(old[column]) and pd.api.types.is_numeric_dtype(new[column])]
    nonnumeric_equal = all(old[column].fillna("<NA>").astype(str).equals(new[column].fillna("<NA>").astype(str)) for column in key_columns)
    differences = []
    all_close = nonnumeric_equal
    for column in numeric:
        a = pd.to_numeric(old[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(new[column], errors="coerce").to_numpy(dtype=float)
        equal_nan = np.isnan(a) & np.isnan(b)
        finite = np.isfinite(a) & np.isfinite(b)
        column_close = bool(np.all(equal_nan | (finite & np.isclose(a, b, atol=atol, rtol=rtol))))
        all_close &= column_close
        if finite.any():
            differences.append(float(np.max(np.abs(a[finite] - b[finite]))))
    return {
        "status": "matches_within_tolerance" if all_close else "values_changed",
        "detail": "",
        "reference_rows": len(old),
        "new_rows": len(new),
        "max_absolute_numeric_difference": max(differences) if differences else 0.0,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
    }


def _predictor_dictionary(cfg: Mapping) -> pd.DataFrame:
    """Build the machine-readable predictor dictionary with explicit units.

    The journal-facing Table S1 with definitions, native resolution, aggregation,
    and source references is generated by ``build_presentation_supplementary_tables.py``.
    """
    unit_map = {
        "elev_min_m": "m",
        "elev_max_m": "m",
        "elev_std_m": "m",
        "elev_median_m": "m",
        "relief_m": "m",
        "slope_median_deg": "degrees",
        "slope_std_deg": "degrees",
        "slope_p90_deg": "degrees",
        "twi_median": "dimensionless index",
        "tri_median": "m",
        "distance_stream_median_m": "m",
        "plan_curvature_median": "m^-1",
        "profile_curvature_median": "m^-1",
        "northness_mean": "dimensionless (-1 to 1)",
        "eastness_mean": "dimensionless (-1 to 1)",
        "log_su_area": "log(m^2)",
    }
    history_units = {
        "prior_failure": "binary (0/1)",
        "prior_active_years_log1p": "log(1 + active years)",
        "prior_landslide_count_log1p": "log(1 + landslide count)",
    }
    records = []
    for variable in cfg["static_predictors"]:
        source = cfg["files"]["slope_units"] if variable == "log_su_area" else cfg["files"]["terrain"]
        rainfall_use = variable in cfg["rainfall_ame"]["static_predictors"]
        records.append({
            "variable": variable,
            "role": "static numeric predictor",
            "source_file": source,
            "transformation": "natural log of projected polygon area, then training-fold z-score" if variable == "log_su_area" else ("training-fold z-score in prediction; global model-sample z-score in rainfall-history models" if rainfall_use else "training-fold z-score in prediction"),
            "units": unit_map[variable],
        })
    for variable in cfg["categorical_predictors"]:
        records.append({"variable": variable, "role": "static categorical predictor", "source_file": cfg["files"]["parent_material"], "transformation": "source codes preserved in prediction; configured UF1/UF2-to-UF harmonization is applied only in the rainfall-history model; treatment coding", "units": "categorical class"})
    for variable in cfg["history_predictors"]:
        records.append({"variable": variable, "role": "pre-target history predictor", "source_file": "rebuilt history panel", "transformation": "defined from mapped activity strictly before the target year", "units": history_units[variable]})
    for variable in cfg["prediction"]["lagged_rainfall_columns"]:
        records.append({"variable": f"{variable}_lag1", "role": "pre-season lagged rainfall predictor", "source_file": cfg["files"]["rainfall"], "transformation": "source year = target year - 1; training-fold imputation and scaling", "units": "mm"})
    return pd.DataFrame(records)


def run(cfg: Mapping, quick: bool = False) -> None:
    del quick
    output_dir = Path(cfg["paths"]["output_dir"])
    supplement_dir = output_dir / "supplement"
    supplement_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for number, relative, description in SUPPLEMENT_INDEX:
        source = output_dir / relative
        target_name = f"{number}_{source.name}"
        target = supplement_dir / target_name
        status = "included" if source.exists() else "not_generated"
        if source.exists():
            shutil.copy2(source, target)
        index_rows.append(
            {
                "supplement_number": number,
                "file": target_name,
                "source_output": relative,
                "description": description,
                "status": status,
            }
        )
    predictor_dictionary = _predictor_dictionary(cfg)
    write_csv(predictor_dictionary, supplement_dir / "S0_predictor_dictionary.csv")
    index_rows.insert(
        0,
        {
            "supplement_number": "S0",
            "file": "S0_predictor_dictionary.csv",
            "source_output": "generated from config",
            "description": "Predictor role, source, units, and transformation dictionary",
            "status": "included",
        },
    )
    write_csv(pd.DataFrame(index_rows), supplement_dir / "Supplementary_File_Index.csv")

    write_json(
        {
            "recommended_submission": "The indexed machine-readable files reproduce the numerical analyses and figures. Main Figures 3–6 are generated separately as vector PDFs and high-resolution PNGs.",
            "do_not_include": ["exploratory notebooks with failed cells", "duplicated 999- and 9999-draw versions", "target-year-rainfall prediction outputs", "manual intermediate exports with no data dictionary"],
        },
        supplement_dir / "supplement_packaging_notes.json",
    )
    LOGGER.info("Supplementary package index created with %d entries", len(index_rows))
