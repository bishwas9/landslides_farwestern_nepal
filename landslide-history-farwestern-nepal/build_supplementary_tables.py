from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "supplement" / "presentation_tables"
OUT.mkdir(parents=True, exist_ok=True)

GAP_LABELS = {
    "1_year": "1 year",
    "2_years": "2 years",
    "3_5_years": "3–5 years",
    "6_10_years": "6–10 years",
    "gt10_years": ">10 years",
    "overall": "Overall",
}


def save(df: pd.DataFrame, name: str) -> None:
    path = OUT / name
    df.to_csv(path, index=False)
    print(path.relative_to(ROOT))


def table_s1() -> None:
    cop = "https://copernicus-dem-30m.s3.amazonaws.com/readme.html"
    narc = "https://soil.narc.gov.np/data/parentsoil/"
    tph = "https://doi.org/10.5194/essd-15-621-2023"
    inv = "https://doi.org/10.1007/s10346-021-01632-6"
    su = "https://doi.org/10.5194/gmd-9-3975-2016"

    rows = [
        ("elev_min_m", "Minimum elevation within each slope unit", "Copernicus DEM GLO-30", "1 arc-second (nominal 30 m)", "m", "Minimum of DEM cells intersecting the slope unit", "Training-fold median imputation and z-score in forward prediction", "P0/P1/P2 static predictor", cop),
        ("elev_max_m", "Maximum elevation within each slope unit", "Copernicus DEM GLO-30", "1 arc-second (nominal 30 m)", "m", "Maximum of DEM cells intersecting the slope unit", "Training-fold median imputation and z-score in forward prediction", "P0/P1/P2 static predictor", cop),
        ("elev_std_m", "Within-slope-unit standard deviation of elevation", "Copernicus DEM GLO-30", "1 arc-second (nominal 30 m)", "m", "Sample standard deviation of DEM cells within the slope unit", "Training-fold median imputation and z-score in forward prediction", "P0/P1/P2 static predictor", cop),
        ("elev_median_m", "Median elevation within each slope unit", "Copernicus DEM GLO-30", "1 arc-second (nominal 30 m)", "m", "Median of DEM cells within the slope unit", "Prediction: training-fold median imputation and z-score; rainfall-history models: model-sample z-score", "Static predictor; rainfall-history covariate", cop),
        ("relief_m", "Slope-unit relief (maximum elevation minus minimum elevation)", "Copernicus DEM GLO-30", "1 arc-second (nominal 30 m)", "m", "Difference between slope-unit maximum and minimum elevation", "Prediction: training-fold median imputation and z-score; rainfall-history models: model-sample z-score", "Static predictor; rainfall-history covariate", cop),
        ("slope_median_deg", "Median terrain slope angle", "Derived from Copernicus DEM GLO-30", "DEM source: 1 arc-second (nominal 30 m)", "degrees", "Median slope angle of raster cells within the slope unit", "Prediction: training-fold median imputation and z-score; rainfall-history models: model-sample z-score", "Static predictor; rainfall-history covariate", cop),
        ("slope_std_deg", "Within-slope-unit standard deviation of terrain slope", "Derived from Copernicus DEM GLO-30", "DEM source: 1 arc-second (nominal 30 m)", "degrees", "Standard deviation of slope raster cells within the slope unit", "Training-fold median imputation and z-score in forward prediction", "P0/P1/P2 static predictor", cop),
        ("slope_p90_deg", "90th percentile of terrain slope", "Derived from Copernicus DEM GLO-30", "DEM source: 1 arc-second (nominal 30 m)", "degrees", "90th percentile of slope raster cells within the slope unit", "Training-fold median imputation and z-score in forward prediction", "P0/P1/P2 static predictor", cop),
        ("twi_median", "Median topographic wetness index", "Derived from Copernicus DEM GLO-30", "DEM source: 1 arc-second (nominal 30 m)", "dimensionless index", "Median TWI value within the slope unit", "Prediction: training-fold median imputation and z-score; rainfall-history models: model-sample z-score", "Static predictor; rainfall-history covariate", cop),
        ("tri_median", "Median terrain ruggedness index, calculated as the mean absolute elevation difference between a cell and its valid 3 x 3 neighbors", "Derived from Copernicus DEM GLO-30", "DEM source: 1 arc-second (nominal 30 m)", "m", "Median TRI value within the slope unit", "Training-fold median imputation and z-score in forward prediction", "P0/P1/P2 static predictor", cop),
        ("distance_stream_median_m", "Median distance to the drainage network", "Drainage network and distance raster derived in the terrain workflow", "Terrain workflow based on projected GLO-30 DEM", "m", "Median cell distance to drainage within the slope unit", "Prediction: training-fold median imputation and z-score; rainfall-history models: model-sample z-score", "Static predictor; rainfall-history covariate", cop),
        ("plan_curvature_median", "Median plan curvature", "Derived from Copernicus DEM GLO-30", "DEM source: 1 arc-second (nominal 30 m)", "m^-1", "Median plan-curvature raster value within the slope unit", "Training-fold median imputation and z-score in forward prediction", "P0/P1/P2 static predictor", cop),
        ("profile_curvature_median", "Median profile curvature", "Derived from Copernicus DEM GLO-30", "DEM source: 1 arc-second (nominal 30 m)", "m^-1", "Median profile-curvature raster value within the slope unit", "Training-fold median imputation and z-score in forward prediction", "P0/P1/P2 static predictor", cop),
        ("northness_mean", "Mean northness of slope aspect", "Derived from Copernicus DEM GLO-30", "DEM source: 1 arc-second (nominal 30 m)", "dimensionless (-1 to 1)", "Mean northness within the slope unit", "Prediction: training-fold median imputation and z-score; rainfall-history models: model-sample z-score", "Static predictor; rainfall-history covariate", cop),
        ("eastness_mean", "Mean eastness of slope aspect", "Derived from Copernicus DEM GLO-30", "DEM source: 1 arc-second (nominal 30 m)", "dimensionless (-1 to 1)", "Mean eastness within the slope unit", "Prediction: training-fold median imputation and z-score; rainfall-history models: model-sample z-score", "Static predictor; rainfall-history covariate", cop),
        ("log_su_area", "Natural logarithm of slope-unit polygon area", "r.slopeunits partition derived from Copernicus DEM GLO-30", "Slope units derived from nominal 30 m DEM", "log(m^2)", "Area calculated from projected slope-unit polygons", "Natural log of polygon area; prediction uses training-fold z-score; rainfall-history models use model-sample z-score", "Static predictor; rainfall-history covariate", su),
        ("parent_material", "Dominant parent-material class within each slope unit", "NARC Parent Soil layer; source identified by NARC as SOTER Nepal", "Native scale not stated on the NARC parent-soil source page", "categorical class", "Dominant class assigned by area coverage within the slope unit", "Prediction preserves source codes; rainfall-history models harmonize UF1 and UF2 to UF only; treatment coding", "Static categorical predictor", narc),
        ("prior_failure", "Indicator that the slope unit had at least one earlier mapped active year", "Multi-temporal landslide inventory / rebuilt annual history panel", "Annual or monsoon-season timing; 1994–1995 treated as an observation gap", "binary (0/1)", "Computed per slope-unit-year using only earlier observed years", "No future activity used; rows crossing the 1994–1995 history gap excluded where required", "P1/P2 forward-prediction history predictor; risk-set definition", inv),
        ("prior_active_years_log1p", "Log-transformed cumulative number of earlier observed active years", "Multi-temporal landslide inventory / rebuilt annual history panel", "Annual or monsoon-season timing", "log(1 + active years)", "Count earlier active years for each slope-unit-year", "log(1 + count); prediction uses training-fold scaling; recurrence model standardizes over its risk set", "Recurrence burden effect; P1/P2 history predictor", inv),
        ("prior_landslide_count_log1p", "Log-transformed cumulative number of earlier mapped landslides", "Multi-temporal landslide inventory / rebuilt annual history panel", "Annual or monsoon-season timing", "log(1 + landslide count)", "Cumulative earlier mapped-landslide count within each slope unit", "log(1 + count), then training-fold scaling in forward prediction", "P1/P2 forward-prediction history predictor", inv),
        ("elapsed_time_category", "Elapsed calendar time since the most recent previous mapped active year", "Multi-temporal landslide inventory / rebuilt annual history panel", "Annual or monsoon-season timing", "categorical years", "Calculated for previously active slope-unit-years only", "Categories: 1 year, 2 years, 3–5 years, 6–10 years, >10 years; >10 years is reference; intervals crossing 1994–1995 excluded", "Recurrence and rainfall-history history-state predictor", inv),
        ("monsoon_within_z", "Within-slope-unit monsoon-rainfall anomaly used in the recurrence model", "TPHiPr daily precipitation", "1/30 degree (~3.3 km), daily; 1979–2020 product", "SD units", "Daily grid precipitation area-overlap weighted to slope units; monsoon total computed by year", "Subtract each slope unit's all-year monsoon mean and standardize over the recurrence-model risk set", "Recurrence-model rainfall covariate", tph),
        ("rx1day_mm / rx3day_mm / rx5day_mm / rx7day_mm / rx10day_mm / rx15day_mm / rx30day_mm", "Annual maximum accumulated precipitation over 1-, 3-, 5-, 7-, 10-, 15-, and 30-day windows", "TPHiPr daily precipitation", "1/30 degree (~3.3 km), daily; 1979–2020 product", "mm", "Daily grid precipitation area-overlap weighted to slope units before rolling accumulation and annual maximum", "For each duration, decomposed into within-slope anomaly (slope-unit centered and SD-scaled) and between-slope climatological component; separate model per duration", "Rainfall-history interaction exposure", tph),
        ("monsoon_rainfall_mm_lag1", "Previous-calendar-year monsoon rainfall", "TPHiPr daily precipitation", "1/30 degree (~3.3 km), daily; 1979–2020 product", "mm", "Area-overlap weighted slope-unit monsoon total from target year - 1", "Training-fold median imputation and scaling", "P2 lagged-rainfall predictor", tph),
        ("rx10day_mm_lag1", "Previous-calendar-year annual maximum 10-day accumulated rainfall", "TPHiPr daily precipitation", "1/30 degree (~3.3 km), daily; 1979–2020 product", "mm", "Area-overlap weighted slope-unit Rx10 from target year - 1", "Training-fold median imputation and scaling", "P2 lagged-rainfall predictor", tph),
        ("rx30day_mm_lag1", "Previous-calendar-year annual maximum 30-day accumulated rainfall", "TPHiPr daily precipitation", "1/30 degree (~3.3 km), daily; 1979–2020 product", "mm", "Area-overlap weighted slope-unit Rx30 from target year - 1", "Training-fold median imputation and scaling", "P2 lagged-rainfall predictor", tph),
    ]
    cols = ["variable", "scientific_definition", "source_product", "native_resolution_or_map_scale", "units", "slope_unit_aggregation", "transformation", "analysis_role", "source_reference"]
    save(pd.DataFrame(rows, columns=cols), "TableS1_predictor_definitions_and_provenance.csv")


def table_s2() -> None:
    panel = pd.read_csv(ROOT / "output/audit/panel_year_audit.csv")
    save(panel, "TableS2A_annual_panel_support.csv")

    support = pd.read_csv(ROOT / "output/tables/recurrence_gap_support.csv")
    support.insert(0, "history_state", support["gap_time_category"].map(GAP_LABELS))
    support["observed_rate_percent"] = 100 * support["rate"]
    support = support[["history_state", "n", "active", "slope_units", "observed_rate_percent"]]
    save(support, "TableS2B_recurrence_history_state_support.csv")

    h = pd.read_csv(ROOT / "output/audit/history_exclusion_audit.csv")
    agg = h.groupby("analysis", as_index=False).agg(
        potential_slope_unit_years=("slope_unit_years", "sum"),
        pre_first_excluded=("pre_first_excluded", "sum"),
        gap_crossing_excluded=("gap_crossing_excluded_from_risk", "sum"),
        retained_risk_rows=("retained_risk_rows", "sum"),
    )
    pred = pd.read_csv(ROOT / "output/audit/prediction_sample_exclusions.csv")
    pred_row = pd.DataFrame([{
        "analysis": "forward_prediction_test_set",
        "potential_slope_unit_years": int(pred["potential_slope_unit_years"].sum()),
        "pre_first_excluded": np.nan,
        "gap_crossing_excluded": int(pred["excluded_history_crosses_1994_1995_gap"].sum()),
        "retained_risk_rows": int(pred["retained_prediction_rows"].sum()),
    }])
    agg = pd.concat([agg, pred_row], ignore_index=True)
    save(agg, "TableS2C_analysis_exclusion_summary.csv")
    save(pred, "TableS2D_prediction_exclusions_by_year.csv")


def table_s3() -> None:
    focal = pd.read_csv(ROOT / "output/tables/recurrence_focal_effects.csv")
    def label(term: str) -> str:
        if "[T.1_year]" in term: return "1 year vs >10 years"
        if "[T.2_years]" in term: return "2 years vs >10 years"
        if "[T.3_5_years]" in term: return "3–5 years vs >10 years"
        if "[T.6_10_years]" in term: return "6–10 years vs >10 years"
        if term == "prior_active_years_z": return "Prior active-year burden (1 SD in log[1 + prior active years])"
        if term == "monsoon_z": return "Within-slope-unit monsoon rainfall (1 SD)"
        return term
    pub = focal.copy()
    pub["effect"] = pub["term"].map(label)
    pub["odds_ratio_display"] = pub["odds_ratio"].map(lambda x: f"{x:.2f}")
    pub["approx_95_posterior_CI"] = pub.apply(lambda r: f"{r.interval_low:.2f}–{r.interval_high:.2f}", axis=1)
    pub = pub[["effect", "log_odds_mean", "posterior_sd", "odds_ratio", "interval_low", "interval_high", "odds_ratio_display", "approx_95_posterior_CI"]]
    save(pub, "TableS3A_recurrence_focal_effects.csv")

    allcoef = pd.read_csv(ROOT / "output/models/recurrence_all_coefficients.csv")
    save(allcoef, "TableS3B_recurrence_all_fixed_effect_coefficients.csv")

    spec = json.load(open(ROOT / "output/models/recurrence_model_specification.json"))
    fields = [
        ("Estimator", spec.get("estimator")),
        ("Formula", spec.get("formula")),
        ("Random effect", spec.get("random_effect")),
        ("Risk-set rows", spec.get("n_rows")),
        ("Slope units", spec.get("n_slope_units")),
        ("Active outcomes", spec.get("active_outcomes")),
        ("vcp_p", spec.get("vcp_p")),
        ("fe_p", spec.get("fe_p")),
        ("Random-intercept SD", spec.get("random_intercept_sd")),
        ("Random-intercept variance", spec.get("random_intercept_variance")),
        ("Interval interpretation", spec.get("interpretation_note")),
    ]
    save(pd.DataFrame(fields, columns=["model_item", "value"]), "TableS3C_recurrence_model_specification.csv")


def table_s4() -> None:
    main = pd.read_csv(ROOT / "output/tables/temporal_fixed_margin_summary.csv")
    save(main, "TableS4A_temporal_fixed_margin_results.csv")
    chains = pd.read_csv(ROOT / "output/randomization/temporal_chain_diagnostics.csv")
    save(chains, "TableS4B_temporal_chain_settings_and_margin_checks.csv")
    mixing = pd.read_csv(ROOT / "output/randomization/temporal_chain_mixing_diagnostics.csv")
    save(mixing, "TableS4C_temporal_chain_mixing_diagnostics.csv")


def table_s5() -> None:
    ame = pd.read_csv(ROOT / "output/tables/Figure4_probability_scale_AME_two_way_cluster.csv")
    ame = ame[["duration_days", "history_label", "AME_percentage_points", "CI95_low_percentage_points", "CI95_high_percentage_points", "p_AME", "n_rows_model", "n_SU_clusters", "n_grid_year_clusters"]]
    save(ame, "TableS5A_state_specific_average_marginal_effects.csv")

    con = pd.read_csv(ROOT / "output/tables/Figure4_AME_contrasts_vs_gt10_two_way_cluster.csv")
    mult = pd.read_csv(ROOT / "output/tables/Figure4_multiplicity_adjusted_tests.csv")
    mult = mult[mult["history_state"].ne("all interactions")][["duration_days", "history_state", "p_holm"]]
    con = con.merge(mult, on=["duration_days", "history_state"], how="left", validate="one_to_one")
    keep = ["duration_days", "history_label", "AME_contrast_percentage_points", "CI95_low_percentage_points", "CI95_high_percentage_points", "p_contrast", "p_holm"]
    save(con[keep], "TableS5B_AME_contrasts_vs_gt10.csv")

    jt = pd.read_csv(ROOT / "output/tables/Figure4_joint_interaction_tests_two_way_cluster.csv")
    jholm = pd.read_csv(ROOT / "output/tables/Figure4_multiplicity_adjusted_tests.csv")
    jholm = jholm[jholm["history_state"].eq("all interactions")][["duration_days", "p_holm"]]
    jt = jt.merge(jholm, on="duration_days", how="left", validate="one_to_one")
    save(jt, "TableS5C_joint_rainfall_history_interaction_tests.csv")


def table_s6() -> None:
    s = pd.read_csv(ROOT / "output/tables/observed_vs_time_permutation_9999.csv")
    s.insert(1, "history_state", s["gap_model"].map(GAP_LABELS))
    save(s, "TableS6_spatial_chronology_randomization_results.csv")


def table_s7() -> None:
    pooled = pd.read_csv(ROOT / "output/prediction/forward_metrics_pooled.csv")
    top = pd.read_csv(ROOT / "output/prediction/top_fraction_performance_pooled_annual_budget.csv")
    top5 = top[np.isclose(top["top_fraction"], 0.05)][["model", "selected_rows", "captured_positives", "positive_capture_rate", "precision_at_fraction"]]
    top5 = top5.rename(columns={"positive_capture_rate": "top5_capture_rate", "precision_at_fraction": "top5_precision"})
    main = pooled.merge(top5, on="model", how="left", validate="one_to_one")
    save(main, "TableS7A_pooled_forward_prediction_performance.csv")
    yearly = pd.read_csv(ROOT / "output/prediction/forward_metrics_by_year.csv")
    save(yearly, "TableS7B_forward_prediction_performance_by_year.csv")
    inc = pd.read_csv(ROOT / "output/prediction/incremental_performance.csv")
    save(inc, "TableS7C_incremental_forward_prediction_performance.csv")


if __name__ == "__main__":
    table_s1(); table_s2(); table_s3(); table_s4(); table_s5(); table_s6(); table_s7()
