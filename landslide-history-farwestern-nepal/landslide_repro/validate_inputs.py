from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .common import (
    LOGGER,
    SU_ALIASES,
    YEAR_ALIASES,
    canonicalize_ids,
    find_column,
    input_file,
    observed_years,
    output_path,
    sha256_file,
    weights_file,
    write_csv,
    write_json,
)


RAINFALL_REQUIRED = (
    "monsoon_rainfall_mm",
    "rx1day_mm",
    "rx3day_mm",
    "rx5day_mm",
    "rx7day_mm",
    "rx10day_mm",
    "rx15day_mm",
    "rx30day_mm",
)


def _resolve_alias(columns, aliases, label) -> str:
    return find_column(columns, aliases, label, required=True)


def _csv_check(path: Path, kind: str, cfg: Mapping) -> tuple[list[dict], dict]:
    sample = pd.read_csv(path, nrows=1000)
    columns = list(sample.columns)
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"file": path.name, "check": name, "passed": bool(passed), "detail": detail})

    try:
        su = _resolve_alias(columns, SU_ALIASES, "slope-unit ID")
        add("slope_unit_id_column", True, su)
    except Exception as exc:
        add("slope_unit_id_column", False, str(exc))

    if kind in {"response", "rainfall"}:
        try:
            year = _resolve_alias(columns, YEAR_ALIASES, "year")
            add("year_column", True, year)
        except Exception as exc:
            add("year_column", False, str(exc))

    if kind == "response":
        available = [c for c in ("landslide_count", "active", "landslide_occurrence") if c in columns]
        add("response_column", bool(available), str(available) if available else "Need landslide_count or active")
        count_fields = [c for c in ("landslide_count", "n_landslides", "count") if c in columns]
        add("mapped_count_column_or_event_derivation", True, str(count_fields) if count_fields else "Count absent here; will be derived from event records")
        shallow = [c for c in ("shallow_count", "shallow_active", "shallow_landslide_count") if c in columns]
        add("shallow_response_column_or_event_derivation", True, str(shallow) if shallow else "Absent here; will be derived from shallow event records")
    elif kind == "rainfall":
        lower = {c.lower(): c for c in columns}
        missing = [c for c in RAINFALL_REQUIRED if c.lower() not in lower]
        add("rainfall_columns", not missing, "missing=" + str(missing))
    elif kind == "terrain":
        configured = [c for c in cfg["static_predictors"] if c != "log_su_area"]
        present = [c for c in configured if c in columns]
        missing = [c for c in configured if c not in columns]
        add("configured_static_predictors", not missing, f"present={len(present)}/{len(configured)}; missing={missing}")
    elif kind == "parent":
        candidates = [c for c in ("parent_material", "parent_material_code", "parent_material_name", "dominant_parent_material", "lithology") if c in columns]
        add("parent_material_column", bool(candidates), str(candidates))
    elif kind == "weights":
        grid_candidates = [c for c in ("grid_id", "gridcell_id", "cell_id", "grid") if c in columns]
        weight_candidates = [c for c in ("weight", "overlap_weight", "area_weight", "overlap_area", "area_m2") if c in columns]
        add("grid_id_column", bool(grid_candidates), str(grid_candidates))
        add("overlap_weight_column", bool(weight_candidates), str(weight_candidates))

    manifest = {
        "file": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(sum(1 for _ in path.open("rb")) - 1),
        "columns": "|".join(columns),
    }
    return checks, manifest


def _geopackage_check(path: Path, kind: str) -> tuple[list[dict], dict]:
    import geopandas as gpd

    gdf = gpd.read_file(path)
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"file": path.name, "check": name, "passed": bool(passed), "detail": detail})

    try:
        su = _resolve_alias(gdf.columns, SU_ALIASES, "slope-unit ID")
        add("slope_unit_id_column", True, su)
    except Exception as exc:
        add("slope_unit_id_column", False, str(exc))
    add("nonempty_geometry", bool(gdf.geometry.notna().all()), f"missing={int(gdf.geometry.isna().sum())}")
    add("valid_crs", gdf.crs is not None, str(gdf.crs))
    if kind == "events":
        try:
            year = _resolve_alias(gdf.columns, YEAR_ALIASES, "event year")
            add("event_year_column", True, year)
        except Exception as exc:
            add("event_year_column", False, str(exc))
        type_candidates = [c for c in ("landslide_type", "type", "movement_type") if c in gdf.columns]
        add("landslide_type_column", bool(type_candidates), str(type_candidates))
        if type_candidates:
            values = gdf[type_candidates[0]].dropna().astype(str).str.lower().str.strip()
            add("shallow_event_class", bool(values.eq("shallow").any()), f"shallow rows={int(values.eq('shallow').sum())}")
    if kind == "slope_units":
        add("polygon_geometry", set(gdf.geom_type.unique()).issubset({"Polygon", "MultiPolygon"}), str(gdf.geom_type.value_counts().to_dict()))
    manifest = {
        "file": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": len(gdf),
        "columns": "|".join(map(str, gdf.columns)),
        "crs": str(gdf.crs),
    }
    return checks, manifest


def run(cfg: Mapping, quick: bool = False) -> None:
    del quick
    input_dir = Path(cfg["paths"]["input_dir"])
    input_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "slope_units": input_file(cfg, "slope_units"),
        "events": input_file(cfg, "events"),
        "response": input_file(cfg, "response_panel"),
        "terrain": input_file(cfg, "terrain"),
        "parent": input_file(cfg, "parent_material"),
        "rainfall": input_file(cfg, "rainfall"),
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    try:
        weight_path = weights_file(cfg)
    except FileNotFoundError as exc:
        missing.append(str(exc))
        weight_path = None
    if missing:
        audit = pd.DataFrame([{"file": item, "check": "exists", "passed": False, "detail": "missing"} for item in missing])
        write_csv(audit, output_path(cfg, "audit", "input_validation.csv"))
        raise FileNotFoundError("Required inputs are missing. See input/README.md and output/audit/input_validation.csv")

    checks: list[dict] = []
    manifests: list[dict] = []
    for key, path in required.items():
        if path.suffix.lower() == ".gpkg":
            these_checks, manifest = _geopackage_check(path, key)
        else:
            these_checks, manifest = _csv_check(path, key, cfg)
        checks.extend(these_checks)
        manifests.append(manifest)
    assert weight_path is not None
    these_checks, manifest = _csv_check(weight_path, "weights", cfg)
    checks.extend(these_checks)
    manifests.append(manifest)

    checks_df = pd.DataFrame(checks)
    write_csv(checks_df, output_path(cfg, "audit", "input_validation.csv"))
    write_csv(pd.DataFrame(manifests), output_path(cfg, "audit", "input_manifest.csv"))
    write_json({"observed_years": observed_years(cfg), "all_checks_passed": bool(checks_df["passed"].all())}, output_path(cfg, "audit", "validation_summary.json"))
    if not checks_df["passed"].all():
        failed = checks_df.loc[~checks_df["passed"], ["file", "check", "detail"]]
        raise ValueError("Input validation failed:\n" + failed.to_string(index=False))
    LOGGER.info("Input validation passed for %d files", len(manifests))
