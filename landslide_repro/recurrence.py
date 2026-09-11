from __future__ import annotations

import json
from typing import Mapping

import numpy as np
import pandas as pd

from .common import LOGGER, output_path, write_csv, write_json, zscore
from .data import complete_case_report, read_history, read_rainfall


def run(cfg: Mapping, quick: bool = False) -> None:
    del quick
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    # BinomialBayesMixedGLM.fit_vb initializes its optimizer numerically.
    # Seeding here makes repeated fits within the locked environment stable.
    np.random.seed(int(cfg["analysis"]["seed"]))

    history = read_history(cfg)
    rain_all = read_rainfall(cfg)
    rain = rain_all[["su_id", "year", "monsoon_rainfall_mm"]]
    climatology = rain_all.groupby("su_id", observed=True)["monsoon_rainfall_mm"].mean().rename("monsoon_all_year_mean").reset_index()
    data = history.merge(rain, on=["su_id", "year"], how="left", validate="one_to_one")
    data = data.merge(climatology, on="su_id", how="left", validate="many_to_one")
    data["monsoon_within"] = data["monsoon_rainfall_mm"] - data["monsoon_all_year_mean"]

    risk = data.loc[
        data["prior_failure"].eq(1)
        & data["history_crosses_missing_gap"].eq(0)
        & data["gap_time_category"].notna()
    ].copy()
    risk["prior_active_years_log1p"] = np.log1p(risk["prior_active_years"])
    risk["prior_active_years_z"], burden_mean, burden_sd = zscore(risk["prior_active_years_log1p"])
    risk["monsoon_z"], rain_center, rain_sd = zscore(risk["monsoon_within"])
    risk["gap_time_category"] = pd.Categorical(
        risk["gap_time_category"],
        categories=["gt10_years", "1_year", "2_years", "3_5_years", "6_10_years"],
        ordered=True,
    )
    required = ["active", "gap_time_category", "prior_active_years_z", "monsoon_z", "year", "su_id"]
    risk, missing_report = complete_case_report(risk, required, "recurrence_mixed_effects")
    write_csv(missing_report, output_path(cfg, "audit", "recurrence_missingness.csv"))

    formula = (
        "active ~ C(gap_time_category, Treatment(reference='gt10_years')) "
        "+ prior_active_years_z + monsoon_z"
    )
    if cfg["recurrence"].get("include_year_fixed_effects", True):
        formula += " + C(year)"
    model = BinomialBayesMixedGLM.from_formula(
        formula,
        {"SU_intercept": "0 + C(su_id)"},
        risk,
        vcp_p=float(cfg["recurrence"]["vcp_p"]),
        fe_p=float(cfg["recurrence"]["fe_p"]),
    )
    result = model.fit_vb(
        fit_method="L-BFGS-B",
        minim_opts={
            "maxiter": int(cfg["recurrence"]["maxiter"]),
            "gtol": float(cfg["recurrence"]["gtol"]),
            "ftol": 1e-9,
            "maxls": 50,
        },
        scale_fe=True,
    )

    coef = pd.DataFrame(
        {
            "term": model.exog_names,
            "log_odds_mean": result.fe_mean,
            "posterior_sd": result.fe_sd,
        }
    )
    coef["odds_ratio"] = np.exp(coef["log_odds_mean"])
    coef["interval_low"] = np.exp(coef["log_odds_mean"] - 1.959963984540054 * coef["posterior_sd"])
    coef["interval_high"] = np.exp(coef["log_odds_mean"] + 1.959963984540054 * coef["posterior_sd"])
    coef["interval_type"] = "approximate 95% posterior credible interval"

    focal_mask = coef["term"].str.contains("gap_time_category") | coef["term"].isin(["prior_active_years_z", "monsoon_z"])
    focal = coef.loc[focal_mask].copy()
    focal["display_term"] = focal["term"].replace(
        {
            "C(gap_time_category, Treatment(reference='gt10_years'))[T.1_year]": "1 year vs >10 years",
            "C(gap_time_category, Treatment(reference='gt10_years'))[T.2_years]": "2 years vs >10 years",
            "C(gap_time_category, Treatment(reference='gt10_years'))[T.3_5_years]": "3–5 years vs >10 years",
            "C(gap_time_category, Treatment(reference='gt10_years'))[T.6_10_years]": "6–10 years vs >10 years",
            "prior_active_years_z": "Prior active-year burden (1 SD in log[1 + prior active years])",
            "monsoon_z": "Within-slope-unit monsoon rainfall (1 SD)",
        }
    )
    random_log_sd = float(result.vcp_mean[0])
    random_sd = float(np.exp(random_log_sd))
    random_var = random_sd**2
    optim = getattr(result, "optim_retvals", {})
    if hasattr(optim, "items"):
        # SciPy's OptimizeResult also contains vector/matrix fields such as
        # ``x``, ``jac`` and ``hess_inv``.  They are unnecessary for the
        # reproducibility metadata and cannot be converted with scalar
        # ``.item()``.  Retain the compact convergence diagnostics only.
        diagnostic_keys = {"success", "status", "message", "nit", "fun", "nfev", "njev"}
        optim = {
            str(k): (v.item() if isinstance(v, np.generic) else v)
            for k, v in optim.items()
            if k in diagnostic_keys
        }
    else:
        optim = {"raw": str(optim)}
    metadata = {
        "estimator": "statsmodels BinomialBayesMixedGLM variational Bayes",
        "formula": formula,
        "random_effect": "slope-unit random intercept",
        "n_rows": len(risk),
        "n_slope_units": int(risk["su_id"].nunique()),
        "active_outcomes": int(risk["active"].sum()),
        "vcp_p": float(cfg["recurrence"]["vcp_p"]),
        "fe_p": float(cfg["recurrence"]["fe_p"]),
        "prior_burden_log1p_mean": burden_mean,
        "prior_burden_log1p_sd": burden_sd,
        "monsoon_within_mean": rain_center,
        "monsoon_within_sd": rain_sd,
        "random_intercept_log_sd": random_log_sd,
        "random_intercept_sd": random_sd,
        "random_intercept_variance": random_var,
        "optimizer": optim,
        "numpy_seed": int(cfg["analysis"]["seed"]),
        "interpretation_note": "Intervals are approximate posterior credible intervals from the variational-Bayes fit, not frequentist confidence intervals.",
    }
    support = (
        risk.groupby("gap_time_category", observed=True)
        .agg(n=("active", "size"), active=("active", "sum"), slope_units=("su_id", "nunique"))
        .reset_index()
    )
    support["rate"] = support["active"] / support["n"]

    write_csv(risk, output_path(cfg, "data", "recurrence_model_dataset.csv"))
    write_csv(coef, output_path(cfg, "models", "recurrence_all_coefficients.csv"))
    write_csv(focal, output_path(cfg, "tables", "recurrence_focal_effects.csv"))
    write_csv(support, output_path(cfg, "tables", "recurrence_gap_support.csv"))
    write_json(metadata, output_path(cfg, "models", "recurrence_model_specification.json"))
    LOGGER.info("Logistic mixed-effects recurrence model fitted to %d slope-unit years", len(risk))
