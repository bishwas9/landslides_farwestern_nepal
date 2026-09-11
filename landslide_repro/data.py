from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .common import SU_ALIASES, canonicalize_ids, find_column, input_file, weights_file


def read_history(cfg: Mapping) -> pd.DataFrame:
    path = __import__("pathlib").Path(cfg["paths"]["output_dir"]) / "data" / "history_panel.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python run_all.py --stage history` first")
    return canonicalize_ids(pd.read_csv(path), require_year=True)


def read_rainfall(cfg: Mapping) -> pd.DataFrame:
    rain = canonicalize_ids(pd.read_csv(input_file(cfg, "rainfall")), require_year=True)
    aliases = {
        "monsoon_rainfall_mm": ("monsoon_rainfall_mm", "monsoon_mm", "monsoon_rainfall"),
        "rx1day_mm": ("rx1day_mm", "Rx1day", "rx1"),
        "rx3day_mm": ("rx3day_mm", "Rx3day", "rx3"),
        "rx5day_mm": ("rx5day_mm", "Rx5day", "rx5"),
        "rx7day_mm": ("rx7day_mm", "Rx7day", "rx7"),
        "rx10day_mm": ("rx10day_mm", "Rx10day", "rx10"),
        "rx15day_mm": ("rx15day_mm", "Rx15day", "rx15"),
        "rx30day_mm": ("rx30day_mm", "Rx30day", "rx30"),
    }
    rename = {}
    for canonical, candidates in aliases.items():
        found = find_column(rain.columns, candidates, canonical, required=True)
        rename[found] = canonical
    rain = rain.rename(columns=rename)
    keep = ["su_id", "year", *aliases]
    return rain.loc[:, list(dict.fromkeys(keep))]


def read_static(cfg: Mapping) -> pd.DataFrame:
    terrain = canonicalize_ids(pd.read_csv(input_file(cfg, "terrain")))
    parent = canonicalize_ids(pd.read_csv(input_file(cfg, "parent_material")))
    parent_col = find_column(
        parent.columns,
        ("parent_material_code", "parent_material", "parent_material_name", "dominant_parent_material", "lithology"),
        "parent material",
    )
    if parent_col != "parent_material":
        parent = parent.rename(columns={parent_col: "parent_material"})
    # Preserve the source codes here.  In particular, UF1 and UF2 are valid
    # parent-material classes; collapsing them globally changed the prediction
    # design matrix.  Analysis-specific harmonization, when required, belongs
    # in that analysis stage and is recorded in its model specification.
    parent["parent_material"] = parent["parent_material"].astype("string").str.strip()
    parent_fields = ["su_id", "parent_material"]
    if "su_area_m2" in parent.columns:
        parent_fields.append("su_area_m2")
    static = terrain.merge(parent[parent_fields], on="su_id", how="left", validate="one_to_one")
    if "log_su_area" in cfg.get("static_predictors", []):
        if "su_area_m2" in static.columns:
            static["su_area_m2"] = pd.to_numeric(static["su_area_m2"], errors="raise")
        else:
            import geopandas as gpd

            slope_units = gpd.read_file(input_file(cfg, "slope_units"))
            slope_units = canonicalize_ids(slope_units)
            if slope_units.crs is None:
                raise ValueError("Slope-unit polygons have no CRS; log_su_area cannot be calculated")
            slope_units = slope_units.to_crs(cfg["analysis"]["projected_crs"])
            area = pd.DataFrame({"su_id": slope_units["su_id"], "su_area_m2": slope_units.geometry.area.to_numpy()})
            static = static.merge(area, on="su_id", how="left", validate="one_to_one")
        if static["su_area_m2"].isna().any():
            raise ValueError("Slope-unit area contains missing values")
        if static["su_area_m2"].le(0).any():
            raise ValueError("Slope-unit area contains non-positive values")
        static["log_su_area"] = np.log(static["su_area_m2"])
    return static


def read_dominant_grid(cfg: Mapping) -> pd.DataFrame:
    weights = canonicalize_ids(pd.read_csv(weights_file(cfg)))
    grid_col = find_column(weights.columns, ("grid_id", "gridcell_id", "cell_id", "grid"), "rainfall grid ID")
    weight_col = find_column(weights.columns, ("weight", "overlap_weight", "area_weight", "overlap_area", "area_m2"), "overlap weight")
    weights[weight_col] = pd.to_numeric(weights[weight_col], errors="raise")
    dominant = (
        weights.sort_values(["su_id", weight_col, grid_col], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates("su_id", keep="first")[["su_id", grid_col, weight_col]]
        .rename(columns={grid_col: "dominant_grid_id", weight_col: "dominant_grid_weight"})
    )
    dominant["dominant_grid_id"] = dominant["dominant_grid_id"].astype(str)
    return dominant


def add_within_between(data: pd.DataFrame, rain_all: pd.DataFrame, rain_col: str) -> tuple[pd.DataFrame, dict]:
    climatology = (
        rain_all.groupby("su_id", observed=True)[rain_col]
        .agg(rain_mean="mean", rain_sd=lambda x: x.std(ddof=1))
        .reset_index()
    )
    out = data.merge(climatology, on="su_id", how="left", validate="many_to_one")
    out["rx_z"] = (out[rain_col] - out["rain_mean"]) / out["rain_sd"]
    between = out[["su_id", "rain_mean"]].drop_duplicates("su_id").copy()
    between_mean = float(between["rain_mean"].mean())
    between_sd = float(between["rain_mean"].std(ddof=1))
    if between_sd <= 0 or out["rain_sd"].dropna().le(0).any():
        raise ValueError(f"Rainfall decomposition has non-positive SD for {rain_col}")
    between["rx_between_z"] = (between["rain_mean"] - between_mean) / between_sd
    out = out.merge(between[["su_id", "rx_between_z"]], on="su_id", how="left", validate="many_to_one")
    return out, {
        "rainfall_column": rain_col,
        "within_definition": "(annual rainfall - slope-unit all-year mean) / slope-unit all-year sample SD",
        "slope_unit_rain_sd_min": float(out["rain_sd"].min()),
        "slope_unit_rain_sd_max": float(out["rain_sd"].max()),
        "between_mean": between_mean,
        "between_sd": between_sd,
    }


def complete_case_report(data: pd.DataFrame, required: list[str], analysis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    keep = pd.Series(True, index=data.index)
    for column in required:
        missing = data[column].isna()
        records.append({"analysis": analysis, "variable": column, "missing_rows": int(missing.sum())})
        keep &= ~missing
    records.append({"analysis": analysis, "variable": "ANY_REQUIRED", "missing_rows": int((~keep).sum())})
    return data.loc[keep].copy(), pd.DataFrame(records)
