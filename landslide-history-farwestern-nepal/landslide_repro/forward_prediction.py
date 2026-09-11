from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.special import logit

from .common import LOGGER, output_path, write_csv, write_json
from .data import read_history, read_rainfall, read_static


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    result = {
        "n": len(y),
        "positives": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
    }
    try:
        import statsmodels.api as sm

        x = sm.add_constant(logit(clipped))
        calibration = sm.GLM(y, x, family=sm.families.Binomial()).fit()
        result["calibration_intercept"] = float(calibration.params[0])
        result["calibration_slope"] = float(calibration.params[1])
    except Exception:
        # Keep this diagnostic available in lightweight environments where
        # statsmodels is not installed.  This is the same unpenalized logistic
        # calibration model, optimized directly with SciPy.
        from scipy.optimize import minimize
        from scipy.special import expit

        calibration_x = logit(clipped)

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            linear = beta[0] + beta[1] * calibration_x
            fitted = expit(linear)
            value = -float(np.sum(y * np.log(np.clip(fitted, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - fitted, 1e-12, 1))))
            residual = fitted - y
            gradient = np.array([residual.sum(), np.dot(residual, calibration_x)], dtype=float)
            return value, gradient

        calibration = minimize(
            lambda beta: objective(beta)[0],
            x0=np.array([0.0, 1.0]),
            jac=lambda beta: objective(beta)[1],
            method="BFGS",
        )
        result["calibration_intercept"] = float(calibration.x[0])
        result["calibration_slope"] = float(calibration.x[1])
    return result


def _top_fraction_rows(frame: pd.DataFrame, fractions: list[float]) -> list[dict]:
    rows = []
    ordered = frame.sort_values("predicted_probability", ascending=False, kind="mergesort")
    positives = int(ordered["observed"].sum())
    for fraction in fractions:
        k = max(1, int(np.ceil(len(ordered) * fraction)))
        selected = ordered.head(k)
        captured = int(selected["observed"].sum())
        rows.append(
            {
                "top_fraction": fraction,
                "selected_rows": k,
                "captured_positives": captured,
                "total_positives": positives,
                "positive_capture_rate": captured / positives if positives else np.nan,
                "precision_at_fraction": captured / k,
            }
        )
    return rows


def _pooled_annual_budget_rows(frame: pd.DataFrame, fractions: list[float]) -> list[dict]:
    """Pool captures after applying the same rank budget separately in each year."""
    rows = []
    total_positives = int(frame["observed"].sum())
    for fraction in fractions:
        selected_rows = 0
        captured = 0
        for _, part in frame.groupby("test_year", sort=True, observed=True):
            ordered = part.sort_values("predicted_probability", ascending=False, kind="mergesort")
            k = max(1, int(np.ceil(len(ordered) * fraction)))
            selected = ordered.head(k)
            selected_rows += k
            captured += int(selected["observed"].sum())
        rows.append(
            {
                "top_fraction": fraction,
                "selected_rows": selected_rows,
                "captured_positives": captured,
                "total_positives": total_positives,
                "positive_capture_rate": captured / total_positives if total_positives else np.nan,
                "precision_at_fraction": captured / selected_rows if selected_rows else np.nan,
                "budget_definition": "top fraction selected independently within each held-out year, then counts pooled",
            }
        )
    return rows


def _calibration_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, year), part in frame.groupby(["model", "test_year"], observed=True):
        bins = pd.qcut(part["predicted_probability"], q=10, labels=False, duplicates="drop")
        grouped = part.assign(calibration_bin=bins).groupby("calibration_bin", observed=True)
        table = grouped.agg(n=("observed", "size"), mean_predicted=("predicted_probability", "mean"), observed_rate=("observed", "mean")).reset_index()
        table.insert(0, "test_year", year)
        table.insert(0, "model", model)
        out.append(table)
    return pd.concat(out, ignore_index=True)


def _build_pipeline(numeric: list[str], categorical: list[str], settings: Mapping):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )
    model = LogisticRegression(
        C=float(settings["C"]),
        max_iter=int(settings["max_iter"]),
        solver="lbfgs",
        class_weight=settings.get("class_weight"),
        random_state=0,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def run(cfg: Mapping, quick: bool = False) -> None:
    history = read_history(cfg)
    rain = read_rainfall(cfg)
    static = read_static(cfg)
    settings = cfg["prediction"]
    lag_columns = list(settings["lagged_rainfall_columns"])
    missing_lag = [column for column in lag_columns if column not in rain.columns]
    if missing_lag:
        raise ValueError(f"Missing rainfall fields required for lagged prediction: {missing_lag}")
    lagged = rain[["su_id", "year", *lag_columns]].copy()
    lagged = lagged.rename(columns={column: f"{column}_lag1" for column in lag_columns})
    lagged["rainfall_source_year"] = lagged["year"]
    lagged["year"] = lagged["year"] + 1

    data = history.merge(static, on="su_id", how="left", validate="many_to_one")
    data = data.merge(lagged, on=["su_id", "year"], how="left", validate="one_to_one")
    valid_source = data["rainfall_source_year"].notna()
    if not (data.loc[valid_source, "rainfall_source_year"].astype(int) == data.loc[valid_source, "year"].astype(int) - 1).all():
        raise AssertionError("Lagged-rainfall leakage check failed")
    current_rainfall_dropped = [column for column in rain.columns if column not in {"su_id", "year"} and column in data.columns and not column.endswith("_lag1")]
    if current_rainfall_dropped:
        data = data.drop(columns=current_rainfall_dropped)

    data["prior_active_years_log1p"] = np.log1p(data["prior_active_years"])
    data["prior_landslide_count_log1p"] = np.log1p(data["prior_landslide_count"])
    data = data.loc[data["year"].ge(int(settings["first_training_year"]))].copy()

    all_test_years = [int(year) for year in settings["test_years"]]
    test_years = all_test_years[:2] if quick else all_test_years
    potential = data.loc[data["year"].isin(test_years)].copy()
    eligible = data.loc[data["history_crosses_missing_gap"].eq(0)].copy()

    static_numeric = list(cfg["static_predictors"])
    static_categorical = list(cfg["categorical_predictors"])
    missing_static = [column for column in static_numeric + static_categorical if column not in eligible.columns]
    if missing_static:
        raise ValueError(f"Missing prediction predictors: {missing_static}")
    history_numeric = ["prior_failure", "prior_active_years_log1p", "prior_landslide_count_log1p"]
    lag_numeric = [f"{column}_lag1" for column in lag_columns]
    feature_sets = {
        "P0_static": (static_numeric, static_categorical),
        "P1_history": (static_numeric + history_numeric, static_categorical),
        "P2_history_lagged_rain": (static_numeric + history_numeric + lag_numeric, static_categorical),
    }
    current_rainfall_names = set(rain.columns) - {"su_id", "year"}
    leakage_features = sorted(
        feature for numeric, categorical in feature_sets.values() for feature in numeric + categorical if feature in current_rainfall_names
    )
    if leakage_features:
        raise AssertionError(f"Current-year rainfall entered a prediction feature set: {leakage_features}")

    exclusion_rows = []
    for year in test_years:
        p = potential.loc[potential["year"].eq(year)]
        e = eligible.loc[eligible["year"].eq(year)]
        exclusion_rows.append(
            {
                "test_year": year,
                "potential_slope_unit_years": len(p),
                "excluded_history_crosses_1994_1995_gap": int(p["history_crosses_missing_gap"].eq(1).sum()),
                "retained_prediction_rows": len(e),
                "retained_slope_units": int(e["su_id"].nunique()),
                "missing_lagged_rainfall_rows": int(e[lag_numeric].isna().any(axis=1).sum()),
            }
        )

    predictions = []
    metrics_by_year = []
    coefficients = []
    top_rows = []
    for test_year in test_years:
        train = eligible.loc[eligible["year"].lt(test_year)].copy()
        test = eligible.loc[eligible["year"].eq(test_year)].copy()
        if train.empty or test.empty:
            raise ValueError(f"No training or testing data for {test_year}")
        if train["year"].max() >= test_year:
            raise AssertionError("Forward split leakage detected")
        for model_name, (numeric, categorical) in feature_sets.items():
            pipeline = _build_pipeline(numeric, categorical, settings)
            columns = numeric + categorical
            pipeline.fit(train[columns], train["active"].astype(int))
            probability = pipeline.predict_proba(test[columns])[:, 1]
            pred = pd.DataFrame(
                {
                    "su_id": test["su_id"].to_numpy(),
                    "test_year": test_year,
                    "model": model_name,
                    "observed": test["active"].astype(int).to_numpy(),
                    "predicted_probability": probability,
                    "max_training_year": int(train["year"].max()),
                    "rainfall_source_year": test["rainfall_source_year"].to_numpy(),
                }
            )
            predictions.append(pred)
            metric = _metrics(pred["observed"].to_numpy(), probability)
            metrics_by_year.append({"model": model_name, "test_year": test_year, **metric})
            for row in _top_fraction_rows(pred, [float(x) for x in settings["top_fractions"]]):
                top_rows.append({"model": model_name, "test_year": test_year, **row})
            preprocessor = pipeline.named_steps["preprocess"]
            names = preprocessor.get_feature_names_out()
            values = pipeline.named_steps["model"].coef_.ravel()
            coefficients.extend(
                {"model": model_name, "test_year": test_year, "feature": name, "coefficient": value}
                for name, value in zip(names, values)
            )

    pred_all = pd.concat(predictions, ignore_index=True)
    leakage_checks = pd.DataFrame(
        [
            {
                "check": "all_max_training_year_before_test_year",
                "passed": bool((pred_all["max_training_year"] < pred_all["test_year"]).all()),
                "detail": f"maximum offset={int((pred_all['max_training_year'] - pred_all['test_year']).max())}",
            },
            {
                "check": "first_test_year_uses_previous_year_as_latest_training_year",
                "passed": bool(pred_all.loc[pred_all["test_year"].eq(test_years[0]), "max_training_year"].eq(test_years[0] - 1).all()),
                "detail": f"test={test_years[0]}, expected max training={test_years[0] - 1}",
            },
            {
                "check": "all_nonmissing_rainfall_sources_are_t_minus_1",
                "passed": bool((pred_all.loc[pred_all["rainfall_source_year"].notna(), "rainfall_source_year"].astype(int) == pred_all.loc[pred_all["rainfall_source_year"].notna(), "test_year"].astype(int) - 1).all()),
                "detail": "source year is stored with every held-out prediction",
            },
            {
                "check": "no_current_year_rainfall_in_feature_sets",
                "passed": not leakage_features,
                "detail": str(leakage_features),
            },
        ]
    )
    if not leakage_checks["passed"].all():
        raise AssertionError("Prediction leakage QA failed:\n" + leakage_checks.to_string(index=False))
    pooled_rows = []
    for model_name, part in pred_all.groupby("model", observed=True):
        pooled_rows.append({"model": model_name, **_metrics(part["observed"].to_numpy(), part["predicted_probability"].to_numpy())})
    pooled = pd.DataFrame(pooled_rows)
    pooled_top_rows = []
    for model_name, part in pred_all.groupby("model", observed=True):
        for row in _pooled_annual_budget_rows(part, [float(x) for x in settings["top_fractions"]]):
            pooled_top_rows.append({"model": model_name, **row})
    wide = pooled.set_index("model")
    increments = []
    for richer, simpler in (("P1_history", "P0_static"), ("P2_history_lagged_rain", "P1_history"), ("P2_history_lagged_rain", "P0_static")):
        for metric in ("average_precision", "roc_auc", "brier_score", "log_loss"):
            increments.append(
                {
                    "richer_model": richer,
                    "simpler_model": simpler,
                    "metric": metric,
                    "difference_richer_minus_simpler": float(wide.loc[richer, metric] - wide.loc[simpler, metric]),
                }
            )

    model_spec = {
        "estimator": "scikit-learn LogisticRegression",
        "penalty": "L2",
        "C": float(settings["C"]),
        "solver": "lbfgs",
        "max_iter": int(settings["max_iter"]),
        "class_weight": settings.get("class_weight"),
        "numeric_preprocessing": "training-fold median imputation followed by training-fold StandardScaler",
        "categorical_preprocessing": "training-fold most-frequent imputation and OneHotEncoder(handle_unknown='ignore')",
        "validation": "expanding-window annual holdout",
        "test_years": test_years,
        "first_test_year_training_years": sorted(eligible.loc[eligible["year"].lt(test_years[0]), "year"].unique().astype(int).tolist()),
        "feature_sets": {name: {"numeric": numeric, "categorical": categorical} for name, (numeric, categorical) in feature_sets.items()},
        "lagged_rainfall_rule": "Every nonmissing rainfall_source_year equals target year minus one; target-year rainfall is absent from the model frame.",
        "history_definition": "Prior failure status, log1p cumulative prior active years, and log1p cumulative prior mapped-landslide count. Elapsed-time category is not a prediction feature.",
        "current_year_rainfall_columns_removed_before_modelling": current_rainfall_dropped,
        "risk_set_rule": "All slope units are eligible, including never-active units, except rows whose history crosses the 1994–1995 observation gap.",
        "quick_mode": quick,
    }
    write_csv(pred_all, output_path(cfg, "prediction", "forward_predictions.csv"))
    write_csv(pd.DataFrame(metrics_by_year), output_path(cfg, "prediction", "forward_metrics_by_year.csv"))
    write_csv(pooled, output_path(cfg, "prediction", "forward_metrics_pooled.csv"))
    write_csv(pd.DataFrame(increments), output_path(cfg, "prediction", "incremental_performance.csv"))
    write_csv(pd.DataFrame(top_rows), output_path(cfg, "prediction", "top_fraction_performance.csv"))
    write_csv(pd.DataFrame(pooled_top_rows), output_path(cfg, "prediction", "top_fraction_performance_pooled_annual_budget.csv"))
    write_csv(_calibration_bins(pred_all), output_path(cfg, "prediction", "calibration_data.csv"))
    write_csv(pd.DataFrame(coefficients), output_path(cfg, "models", "prediction_fold_coefficients.csv"))
    write_csv(pd.DataFrame(exclusion_rows), output_path(cfg, "audit", "prediction_sample_exclusions.csv"))
    write_csv(leakage_checks, output_path(cfg, "audit", "prediction_leakage_checks.csv"))
    write_json(model_spec, output_path(cfg, "models", "forward_prediction_specification.json"))
    LOGGER.info("Forward prediction completed for %d held-out years", len(test_years))
