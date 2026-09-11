from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, norm

from .common import LOGGER, normal_interval, output_path, write_csv, write_json
from .data import add_within_between, complete_case_report, read_dominant_grid, read_history, read_rainfall, read_static


HISTORY_LEVELS = ["gt10_years", "1_year", "2_years", "3_5_years", "6_10_years"]
DISPLAY_ORDER = ["1_year", "2_years", "3_5_years", "6_10_years", "gt10_years"]
HISTORY_LABELS = {"1_year": "1 year", "2_years": "2 years", "3_5_years": "3–5 years", "6_10_years": "6–10 years", "gt10_years": ">10 years"}


def _zscore_sample(series: pd.Series) -> tuple[pd.Series, float, float]:
    x = pd.to_numeric(series, errors="coerce")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Cannot standardize {series.name!r}: sample SD={sd}")
    return (x - mean) / sd, mean, sd


def _design_matrix(result, frame: pd.DataFrame) -> np.ndarray:
    from patsy import build_design_matrices

    info = result.model.data.design_info
    matrix = build_design_matrices([info], frame, return_type="dataframe")[0]
    return np.asarray(matrix, dtype=float)


def _ame_and_gradient(result, cov: np.ndarray, standardization: pd.DataFrame, history_level: str) -> dict:
    beta = np.asarray(result.params, dtype=float)
    base = standardization.copy()
    base["gap_shallow"] = pd.Categorical([history_level] * len(base), categories=HISTORY_LEVELS, ordered=True)
    plus = base.copy()
    plus["rx_z"] = plus["rx_z"] + 1.0
    x0 = _design_matrix(result, base)
    x1 = _design_matrix(result, plus)
    p0 = expit(x0 @ beta)
    p1 = expit(x1 @ beta)
    ame = float(np.mean(p1 - p0))
    gradient = np.mean(p1[:, None] * (1 - p1[:, None]) * x1 - p0[:, None] * (1 - p0[:, None]) * x0, axis=0)
    variance = float(gradient @ cov @ gradient)
    se = float(np.sqrt(max(variance, 0.0)))
    low, high = normal_interval(ame, se)
    return {"ame": ame, "se": se, "ci_low": low, "ci_high": high, "gradient": gradient}


def _holm(p_values: pd.Series) -> np.ndarray:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy()
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        finite_values = values[valid]
        order = np.argsort(finite_values, kind="mergesort")
        ordered = finite_values[order]
        adjusted_ordered = np.maximum.accumulate((len(ordered) - np.arange(len(ordered))) * ordered)
        adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
        adjusted = np.empty_like(adjusted_ordered)
        adjusted[order] = adjusted_ordered
        result[valid] = adjusted
    return result


def _plot_figure(table: pd.DataFrame, cfg: Mapping) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "1_year": "1 year",
        "2_years": "2 years",
        "3_5_years": "3–5 years",
        "6_10_years": "6–10 years",
        "gt10_years": ">10 years",
    }
    colors = {
        "1_year": "#0072B2",
        "2_years": "#56B4E9",
        "3_5_years": "#009E73",
        "6_10_years": "#E69F00",
        "gt10_years": "#6A3D9A",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for level in DISPLAY_ORDER:
        part = table.loc[table["history_state"].eq(level)].sort_values("duration_days")
        ax.errorbar(
            part["duration_days"],
            part["AME_probability"],
            yerr=[part["AME_probability"] - part["CI95_low_probability"], part["CI95_high_probability"] - part["AME_probability"]],
            marker="o",
            linewidth=1.5,
            capsize=2.5,
            label=labels[level],
            color=colors[level],
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Rainfall accumulation duration (days)")
    ax.set_ylabel("Average change in annual probability\nfor a 1-SD rainfall increase")
    ax.set_xticks(sorted(table["duration_days"].unique()))
    ax.legend(title="Elapsed time since prior activity", frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    png = output_path(cfg, "figures", "Diagnostic_rainfall_history_AME.png")
    pdf = output_path(cfg, "figures", "Diagnostic_rainfall_history_AME.pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def run(cfg: Mapping, quick: bool = False) -> None:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

    history = read_history(cfg)
    rain_all = read_rainfall(cfg)
    static = read_static(cfg)
    dominant = read_dominant_grid(cfg)
    base = (
        history.merge(rain_all, on=["su_id", "year"], how="left", validate="one_to_one")
        .merge(static, on="su_id", how="left", validate="many_to_one")
        .merge(dominant, on="su_id", how="left", validate="many_to_one")
    )
    base = base.loc[
        base["prior_shallow_failure"].eq(1)
        & base["shallow_history_crosses_missing_gap"].eq(0)
        & base["gap_shallow_category"].notna()
    ].copy()
    parent_code_map = {
        str(source): str(target)
        for source, target in cfg["rainfall_ame"].get("parent_material_code_map", {}).items()
    }
    if parent_code_map:
        base["parent_material"] = base["parent_material"].astype("string").replace(parent_code_map)
    parent_exclusion_before = {"rows": len(base), "slope_units": int(base["su_id"].nunique()), "events": int(base["shallow_active"].sum())}
    excluded_codes = [str(x) for x in cfg["rainfall_ame"].get("exclude_parent_material_codes", [])]
    if excluded_codes:
        base = base.loc[~base["parent_material"].astype(str).isin(excluded_codes)].copy()
    parent_exclusion_after = {"rows": len(base), "slope_units": int(base["su_id"].nunique()), "events": int(base["shallow_active"].sum())}
    base["gap_shallow"] = pd.Categorical(base["gap_shallow_category"], categories=HISTORY_LEVELS, ordered=True)
    base["prior_shallow_active_years_log1p"] = np.log1p(base["prior_shallow_active_years"])
    base["prior_shallow_active_years_z"], prior_mean, prior_sd = _zscore_sample(base["prior_shallow_active_years_log1p"])

    static_predictors = list(cfg["rainfall_ame"]["static_predictors"])
    missing_static = [column for column in static_predictors if column not in base.columns]
    if missing_static:
        raise ValueError(f"Missing configured static predictors for rainfall model: {missing_static}")
    scaling_rows = []
    for column in static_predictors:
        z_name = f"{column}_z"
        base[z_name], mean, sd = _zscore_sample(base[column])
        scaling_rows.append({"variable": column, "mean": mean, "sd": sd, "transformation": "(x-mean)/sample_sd"})
    base["parent_material"] = base["parent_material"].astype("category")
    base["grid_year_cluster"] = base["dominant_grid_id"].astype(str) + "__" + base["year"].astype(str)

    duration_map = {int(k): v for k, v in cfg["rainfall_ame"]["rainfall_columns"].items()}
    durations = [int(x) for x in cfg["rainfall_ame"]["durations_days"]]
    if quick:
        durations = [7, 15]
    ame_rows = []
    contrast_rows = []
    joint_rows = []
    coefficient_rows = []
    model_specs = []
    missing_reports = []
    support_rows = []

    static_terms = " + ".join(f"{column}_z" for column in static_predictors)
    formula = (
        "shallow_active ~ rx_z * C(gap_shallow, Treatment(reference='gt10_years')) "
        "+ rx_between_z + prior_shallow_active_years_z"
        f" + {static_terms} + C(parent_material) + C(year)"
    )
    for duration in durations:
        rain_col = duration_map[duration]
        data, rainfall_scaling = add_within_between(base.copy(), rain_all, rain_col)
        required = [
            "shallow_active", "rx_z", "rx_between_z", "prior_shallow_active_years_z", "gap_shallow",
            "parent_material", "year", "su_id", "grid_year_cluster", *[f"{c}_z" for c in static_predictors],
        ]
        data, missing = complete_case_report(data, required, f"rainfall_history_rx{duration}")
        missing["duration_days"] = duration
        missing_reports.append(missing)
        data["gap_shallow"] = pd.Categorical(data["gap_shallow"], categories=HISTORY_LEVELS, ordered=True)
        support = (
            data.groupby("gap_shallow", observed=True)
            .agg(n_rows=("shallow_active", "size"), active_outcomes=("shallow_active", "sum"), n_slope_units=("su_id", "nunique"))
            .reset_index()
            .rename(columns={"gap_shallow": "history_state"})
        )
        support.insert(0, "rainfall", rain_col)
        support.insert(1, "duration_days", duration)
        support["observed_rate"] = support["active_outcomes"] / support["n_rows"]
        support_rows.append(support)
        result = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit()
        group_su = pd.factorize(data["su_id"], sort=True)[0]
        group_grid_year = pd.factorize(data["grid_year_cluster"], sort=True)[0]
        cov, cov_su, cov_grid_year = cov_cluster_2groups(
            result,
            group_su,
            group_grid_year,
            use_correction=bool(cfg["rainfall_ame"]["cluster_small_sample_correction"]),
        )
        cov = np.asarray(cov, dtype=float)

        coef = pd.DataFrame({"term": result.model.exog_names, "estimate": np.asarray(result.params, dtype=float), "se_two_way_cluster": np.sqrt(np.maximum(np.diag(cov), 0))})
        coef["rainfall"] = rain_col
        coef["duration_days"] = duration
        coefficient_rows.append(coef)

        interaction_indices = [i for i, name in enumerate(result.model.exog_names) if "rx_z:C(gap_shallow" in name]
        if len(interaction_indices) != 4:
            raise ValueError(f"Expected four rainfall × history interactions for Rx{duration}; found {[result.model.exog_names[i] for i in interaction_indices]}")
        b = np.asarray(result.params)[interaction_indices]
        v = cov[np.ix_(interaction_indices, interaction_indices)]
        wald = float(b @ np.linalg.pinv(v) @ b)
        p_joint = float(chi2.sf(wald, df=len(interaction_indices)))
        joint_rows.append(
            {
                "rainfall": rain_col,
                "duration_days": duration,
                "joint_interaction_wald_two_way": wald,
                "df": len(interaction_indices),
                "p_two_way": p_joint,
                "n_rows": len(data),
                "n_SU_clusters": int(data["su_id"].nunique()),
                "n_grid_year_clusters": int(data["grid_year_cluster"].nunique()),
            }
        )

        requested_n = cfg["rainfall_ame"].get("standardization_rows")
        if quick:
            requested_n = min(1000, len(data))
        if requested_n is not None and int(requested_n) < len(data):
            standardization = data.sample(n=int(requested_n), random_state=int(cfg["rainfall_ame"]["standardization_seed"]))
        else:
            standardization = data
        by_level = {}
        for level in DISPLAY_ORDER:
            estimate = _ame_and_gradient(result, cov, standardization, level)
            by_level[level] = estimate
            ame_rows.append(
                {
                    "rainfall": rain_col,
                    "duration_days": duration,
                    "history_state": level,
                    "history_label": HISTORY_LABELS[level],
                    "n_standardized": len(standardization),
                    "AME_probability": estimate["ame"],
                    "AME_percentage_points": 100 * estimate["ame"],
                    "SE_probability_two_way": estimate["se"],
                    "CI95_low_probability": estimate["ci_low"],
                    "CI95_high_probability": estimate["ci_high"],
                    "CI95_low_percentage_points": 100 * estimate["ci_low"],
                    "CI95_high_percentage_points": 100 * estimate["ci_high"],
                    "p_AME": float(2 * norm.sf(abs(estimate["ame"] / estimate["se"]))) if estimate["se"] > 0 else np.nan,
                    "n_rows_model": len(data),
                    "n_SU_clusters": int(data["su_id"].nunique()),
                    "n_grid_year_clusters": int(data["grid_year_cluster"].nunique()),
                }
            )
        reference = by_level["gt10_years"]
        for level in ["1_year", "2_years", "3_5_years", "6_10_years"]:
            delta = by_level[level]["ame"] - reference["ame"]
            gradient = by_level[level]["gradient"] - reference["gradient"]
            se = float(np.sqrt(max(float(gradient @ cov @ gradient), 0.0)))
            low, high = normal_interval(delta, se)
            z_value = delta / se if se > 0 else np.nan
            contrast_rows.append(
                {
                    "rainfall": rain_col,
                    "duration_days": duration,
                    "history_state": level,
                    "history_label": HISTORY_LABELS[level],
                    "reference_state": "gt10_years",
                    "AME_contrast_probability": delta,
                    "AME_contrast_percentage_points": 100 * delta,
                    "SE_contrast_two_way": se,
                    "CI95_low_probability": low,
                    "CI95_high_probability": high,
                    "CI95_low_percentage_points": 100 * low,
                    "CI95_high_percentage_points": 100 * high,
                    "p_contrast": float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
                }
            )
        model_specs.append(
            {
                "duration_days": duration,
                "rainfall_scaling": rainfall_scaling,
                "formula": formula,
                "n_rows": len(data),
                "n_standardization_rows": len(standardization),
                "covariance": "two-way cluster inclusion-exclusion covariance",
            }
        )

    ame = pd.DataFrame(ame_rows)
    contrasts = pd.DataFrame(contrast_rows)
    joint = pd.DataFrame(joint_rows)
    multiplicity = pd.concat(
        [
            contrasts[["rainfall", "duration_days", "history_state", "p_contrast"]].assign(test_family="AME contrast vs >10 years", p_holm=lambda d: _holm(d["p_contrast"])).rename(columns={"p_contrast": "p_pointwise"}),
            joint[["rainfall", "duration_days", "p_two_way"]].assign(history_state="all interactions", test_family="duration-level joint interaction", p_holm=lambda d: _holm(d["p_two_way"])).rename(columns={"p_two_way": "p_pointwise"}),
        ],
        ignore_index=True,
    )
    write_csv(ame, output_path(cfg, "tables", "Figure4_probability_scale_AME_two_way_cluster.csv"))
    write_csv(contrasts, output_path(cfg, "tables", "Figure4_AME_contrasts_vs_gt10_two_way_cluster.csv"))
    write_csv(joint, output_path(cfg, "tables", "Figure4_joint_interaction_tests_two_way_cluster.csv"))
    write_csv(multiplicity, output_path(cfg, "tables", "Figure4_multiplicity_adjusted_tests.csv"))
    write_csv(pd.concat(coefficient_rows, ignore_index=True), output_path(cfg, "models", "rainfall_history_all_coefficients.csv"))
    write_csv(pd.concat(missing_reports, ignore_index=True), output_path(cfg, "audit", "rainfall_history_missingness.csv"))
    write_csv(pd.concat(support_rows, ignore_index=True), output_path(cfg, "tables", "rainfall_history_gap_support.csv"))
    write_csv(
        pd.DataFrame(
            [
                {"stage": "before_parent_material_exclusion", **parent_exclusion_before},
                {"stage": "after_parent_material_exclusion", **parent_exclusion_after},
                {"stage": "removed", **{key: parent_exclusion_before[key] - parent_exclusion_after[key] for key in parent_exclusion_before}},
            ]
        ),
        output_path(cfg, "audit", "rainfall_parent_material_exclusion.csv"),
    )
    write_csv(pd.DataFrame(scaling_rows), output_path(cfg, "models", "rainfall_history_static_scaling.csv"))
    write_json(
        {
            "models": model_specs,
            "AME_definition": "Mean over the standardization rows of p(rx_z + 1, history=h) - p(rx_z, history=h).",
            "cluster_1": "slope unit",
            "cluster_2": "dominant rainfall grid cell crossed with year",
            "intervals": "pointwise, unadjusted 95% delta-method intervals; Holm-adjusted p-values are additionally reported",
            "dominant_grid_rule": "maximum slope-unit/grid overlap weight",
            "parent_material_code_map": parent_code_map,
            "excluded_parent_material_codes": excluded_codes,
        },
        output_path(cfg, "models", "rainfall_history_model_specification.json"),
    )
    _plot_figure(ame, cfg)
    LOGGER.info("Rainfall-history AMEs fitted for %d durations", len(durations))
