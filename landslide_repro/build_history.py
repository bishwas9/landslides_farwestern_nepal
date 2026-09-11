from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .common import (
    LOGGER,
    assert_unique,
    canonicalize_ids,
    gap_category,
    input_file,
    observed_years,
    output_path,
    write_csv,
)


def _derive_event_count(cfg: Mapping, panel: pd.DataFrame, shallow_only: bool) -> pd.DataFrame:
    import geopandas as gpd

    events = canonicalize_ids(gpd.read_file(input_file(cfg, "events")), require_year=True)
    if shallow_only:
        type_candidates = [c for c in ("landslide_type", "type", "movement_type") if c in events.columns]
        if not type_candidates:
            raise ValueError("Response panel has no shallow response and event file has no landslide_type column")
        event_type = type_candidates[0]
        events = events.loc[events[event_type].astype(str).str.lower().str.strip().eq("shallow")].copy()
        count_name = "shallow_landslide_count"
    else:
        count_name = "landslide_count"
    counts = events.groupby(["su_id", "year"], observed=True).size().rename(count_name).reset_index()
    out = panel.merge(counts, on=["su_id", "year"], how="left", validate="one_to_one")
    out[count_name] = out[count_name].fillna(0).astype(int)
    return out


def _find_first(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _binary_and_count(panel: pd.DataFrame, prefix: str) -> tuple[pd.Series, pd.Series]:
    if prefix == "":
        count_col = _find_first(panel, ("landslide_count", "n_landslides", "count"))
        active_col = _find_first(panel, ("active", "landslide_occurrence", "occurrence"))
    else:
        count_col = _find_first(panel, ("shallow_count", "shallow_landslide_count", "n_shallow"))
        active_col = _find_first(panel, ("shallow_active", "shallow_occurrence"))
    if count_col is None and active_col is None:
        label = "overall" if prefix == "" else "shallow"
        raise ValueError(f"No {label} response column found")
    if count_col is not None:
        count = pd.to_numeric(panel[count_col], errors="coerce").fillna(0).clip(lower=0)
        active = count.gt(0).astype(np.int8)
    else:
        active = pd.to_numeric(panel[active_col], errors="coerce").fillna(0).gt(0).astype(np.int8)
        count = active.astype(float)
    return active, count


def _history_block(panel: pd.DataFrame, active_col: str, count_col: str, prefix: str, missing_years: list[int]) -> pd.DataFrame:
    p = f"{prefix}_" if prefix else ""
    g = panel.groupby("su_id", sort=False, observed=True)
    active = panel[active_col].astype(int)
    count = panel[count_col].astype(float)
    panel[f"prior_{p}landslide_count"] = g[count_col].cumsum() - count
    panel[f"prior_{p}active_years"] = g[active_col].cumsum() - active
    panel[f"prior_{p}failure"] = panel[f"prior_{p}active_years"].gt(0).astype(np.int8)

    event_year = panel["year"].where(active.eq(1))
    last_prior = event_year.groupby(panel["su_id"], observed=True).transform(lambda s: s.ffill().shift(1))
    panel[f"last_{p}active_year"] = last_prior
    panel[f"years_since_last_{p}activity"] = panel["year"] - last_prior

    crosses = np.zeros(len(panel), dtype=np.int8)
    valid = last_prior.notna().to_numpy()
    years = panel["year"].to_numpy()
    last = last_prior.fillna(-10_000).to_numpy()
    for missing in missing_years:
        crosses |= (valid & (last < missing) & (years > missing)).astype(np.int8)
    panel[f"{p}history_crosses_missing_gap"] = crosses
    panel[f"gap_{prefix or 'time'}_category"] = gap_category(panel[f"years_since_last_{p}activity"])
    return panel


def run(cfg: Mapping, quick: bool = False) -> None:
    del quick
    path = input_file(cfg, "response_panel")
    panel = canonicalize_ids(pd.read_csv(path), require_year=True)
    assert_unique(panel, ["su_id", "year"], "response panel")
    expected_years = observed_years(cfg)
    actual_years = sorted(panel["year"].unique().tolist())
    if actual_years != expected_years:
        raise ValueError(f"Observed years differ from config. actual={actual_years}; expected={expected_years}")

    unit_year_counts = panel.groupby("su_id", observed=True)["year"].nunique()
    unbalanced = unit_year_counts.ne(len(expected_years))
    if unbalanced.any():
        sample = unit_year_counts[unbalanced].head(10).to_dict()
        raise ValueError(f"Response panel is not balanced over observed years; examples: {sample}")

    panel = panel.sort_values(["su_id", "year"], kind="mergesort").reset_index(drop=True)
    if _find_first(panel, ("landslide_count", "n_landslides", "count")) is None:
        panel = _derive_event_count(cfg, panel, shallow_only=False)
    panel["active"], panel["landslide_count"] = _binary_and_count(panel, "")
    if _find_first(panel, ("shallow_count", "shallow_active", "shallow_landslide_count")) is None:
        panel = _derive_event_count(cfg, panel, shallow_only=True)
    panel["shallow_active"], panel["shallow_count"] = _binary_and_count(panel, "shallow")
    missing_years = [int(y) for y in cfg["analysis"]["missing_years"]]
    panel = _history_block(panel, "active", "landslide_count", "", missing_years)
    panel = _history_block(panel, "shallow_active", "shallow_count", "shallow", missing_years)

    panel["previously_active_state"] = np.where(panel["prior_failure"].eq(1), "previously_active", "never_previously_active")
    panel["activity_class"] = np.select(
        [
            panel["year"].eq(expected_years[0]) & panel["active"].eq(1),
            panel["active"].eq(1) & panel["prior_failure"].eq(0),
            panel["active"].eq(1) & panel["prior_failure"].eq(1),
        ],
        ["active_history_unknown", "first_observed_activation", "recurrent_activity"],
        default="inactive",
    )
    panel["recurrence_stage"] = pd.Categorical(
        np.select(
            [panel["prior_active_years"].eq(1), panel["prior_active_years"].eq(2), panel["prior_active_years"].ge(3)],
            ["1_prior_active_year", "2_prior_active_years", "3plus_prior_active_years"],
            default=None,
        ),
        categories=["1_prior_active_year", "2_prior_active_years", "3plus_prior_active_years"],
        ordered=True,
    )
    panel["prior_active_years_log1p"] = np.log1p(panel["prior_active_years"])
    panel["prior_landslide_count_log1p"] = np.log1p(panel["prior_landslide_count"])
    panel["prior_shallow_active_years_log1p"] = np.log1p(panel["prior_shallow_active_years"])

    n_units = panel["su_id"].nunique()
    expected_units = int(cfg["analysis"].get("expected_slope_units", n_units))
    if cfg["analysis"].get("strict_expected_counts", False) and n_units != expected_units:
        raise ValueError(f"Expected {expected_units} slope units but found {n_units}")

    recurrence_risk = panel["prior_failure"].eq(1)
    shallow_risk = panel["prior_shallow_failure"].eq(1)
    audit = []
    for analysis, risk, gap_flag in (
        ("recurrence", recurrence_risk, "history_crosses_missing_gap"),
        ("rainfall_history", shallow_risk, "shallow_history_crosses_missing_gap"),
    ):
        for year, frame in panel.groupby("year", observed=True):
            in_risk = risk.loc[frame.index]
            crossing = frame[gap_flag].eq(1)
            audit.append(
                {
                    "analysis": analysis,
                    "year": int(year),
                    "slope_unit_years": len(frame),
                    "pre_first_excluded": int((~in_risk).sum()),
                    "gap_crossing_excluded_from_risk": int((in_risk & crossing).sum()),
                    "retained_risk_rows": int((in_risk & ~crossing).sum()),
                }
            )

    year_audit = (
        panel.groupby("year", observed=True)
        .agg(slope_units=("su_id", "nunique"), active_slope_units=("active", "sum"), landslides=("landslide_count", "sum"), shallow_active_slope_units=("shallow_active", "sum"), shallow_landslides=("shallow_count", "sum"))
        .reset_index()
    )
    support = (
        panel.loc[recurrence_risk & panel["history_crosses_missing_gap"].eq(0)]
        .groupby("gap_time_category", observed=True)
        .agg(n_slope_unit_years=("active", "size"), active_outcomes=("active", "sum"), n_slope_units=("su_id", "nunique"))
        .reset_index()
    )
    support["observed_rate"] = support["active_outcomes"] / support["n_slope_unit_years"]

    write_csv(panel, output_path(cfg, "data", "history_panel.csv"))
    write_csv(pd.DataFrame(audit), output_path(cfg, "audit", "history_exclusion_audit.csv"))
    write_csv(year_audit, output_path(cfg, "audit", "panel_year_audit.csv"))
    write_csv(support, output_path(cfg, "tables", "recurrence_risk_support.csv"))
    LOGGER.info("History panel written: %d rows, %d slope units", len(panel), n_units)
