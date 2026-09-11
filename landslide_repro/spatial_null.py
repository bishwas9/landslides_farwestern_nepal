from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .common import LOGGER, SU_ALIASES, YEAR_ALIASES, find_column, gap_category, input_file, observed_years, output_path, write_csv, write_json


GAP_ORDER = ["overall", "1_year", "2_years", "3_5_years", "6_10_years", "gt10_years"]


def _recurring_locations(year_values: np.ndarray, coordinates: np.ndarray, groups: list[np.ndarray], missing_years: list[int]) -> pd.DataFrame:
    rows = []
    for indices in groups:
        group_years = year_values[indices]
        group_xy = coordinates[indices]
        for local_index, target_year in enumerate(group_years):
            prior_mask = group_years < target_year
            if not prior_mask.any():
                continue
            last_prior = int(group_years[prior_mask].max())
            if any(last_prior < missing < int(target_year) for missing in missing_years):
                continue
            distance = float(cdist(group_xy[[local_index]], group_xy[prior_mask]).min())
            elapsed = int(target_year) - last_prior
            rows.append(
                {
                    "event_index": int(indices[local_index]),
                    "target_year": int(target_year),
                    "last_prior_year": last_prior,
                    "gap_years": elapsed,
                    "gap_model": str(gap_category([elapsed])[0]),
                    "nearest_prior_distance_m": distance,
                }
            )
    return pd.DataFrame(rows)


def _distance_summary(recurrent: pd.DataFrame, count_name: str = "n_distinct") -> pd.DataFrame:
    rows = []
    rows.append({"gap_model": "overall", count_name: len(recurrent), "median_distance_m": float(recurrent["nearest_prior_distance_m"].median())})
    for category in GAP_ORDER[1:]:
        part = recurrent.loc[recurrent["gap_model"].eq(category), "nearest_prior_distance_m"]
        rows.append({"gap_model": category, count_name: len(part), "median_distance_m": float(part.median()) if len(part) else np.nan})
    return pd.DataFrame(rows)


def _one_permutation(seed: int, years: np.ndarray, coordinates: np.ndarray, groups: list[np.ndarray], missing_years: list[int], shuffle_scope: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    if shuffle_scope == "global":
        permuted = rng.permutation(years)
    elif shuffle_scope == "within_slope_unit":
        permuted = years.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
    else:
        raise ValueError("spatial_null.year_shuffle_scope must be global or within_slope_unit")
    recurrent = _recurring_locations(permuted, coordinates, groups, missing_years)
    return _distance_summary(recurrent).to_dict("records")


def run(cfg: Mapping, quick: bool = False) -> None:
    import geopandas as gpd
    from joblib import Parallel, delayed

    settings = cfg["spatial_null"]
    events = gpd.read_file(input_file(cfg, "events"))
    su_col = find_column(events.columns, SU_ALIASES, "slope-unit ID")
    year_col = find_column(events.columns, YEAR_ALIASES, "event year")
    events = events.rename(columns={su_col: "su_id", year_col: "event_year"})
    events["su_id"] = events["su_id"].astype(str).str.strip()
    events["event_year"] = pd.to_numeric(events["event_year"], errors="coerce")
    before = len(events)
    events = events.loc[
        events["event_year"].isin(observed_years(cfg)) & events["su_id"].notna() & events.geometry.notna()
    ].copy()
    events["event_year"] = events["event_year"].astype(int)
    if events.crs is None:
        raise ValueError("Event file has no CRS; spatial distances cannot be reproduced")
    events = events.to_crs(cfg["analysis"]["projected_crs"])
    mode = settings.get("event_coordinate", "representative_point")
    if set(events.geom_type.unique()).issubset({"Point"}):
        points = events.geometry
        actual_mode = "input point geometry"
    elif mode == "representative_point":
        points = events.geometry.representative_point()
        actual_mode = "representative_point of projected event geometry"
    elif mode == "centroid":
        points = events.geometry.centroid
        actual_mode = "centroid of projected event geometry"
    else:
        raise ValueError("event_coordinate must be representative_point or centroid")
    events["x_m"] = points.x
    events["y_m"] = points.y
    events["source_event_index"] = events.index.astype(str)
    tolerance = float(settings["coordinate_tolerance_m"])
    events["x_key"] = np.round(events["x_m"] / tolerance).astype(np.int64)
    events["y_key"] = np.round(events["y_m"] / tolerance).astype(np.int64)
    location_keys = ["su_id", "x_key", "y_key"]
    multiplicity = (
        events.groupby(location_keys, observed=True)
        .agg(records=("event_year", "size"), distinct_years=("event_year", "nunique"), earliest_year=("event_year", "min"), latest_year=("event_year", "max"))
        .reset_index()
    )
    if bool(settings.get("deduplicate_fixed_locations", True)):
        fixed = events.sort_values(["event_year", "source_event_index"], kind="mergesort").drop_duplicates(location_keys, keep="first").copy()
    else:
        fixed = events.copy()
    fixed = fixed.sort_values(["su_id", "event_year", "x_m", "y_m"], kind="mergesort").reset_index(drop=True)
    fixed["location_index"] = np.arange(len(fixed))
    years = fixed["event_year"].to_numpy(dtype=int)
    coordinates = fixed[["x_m", "y_m"]].to_numpy(dtype=float)
    groups = [indices.to_numpy(dtype=int) for _, indices in fixed.groupby("su_id", sort=True, observed=True).groups.items()]
    missing_years = [int(x) for x in cfg["analysis"]["missing_years"]]
    observed_recurrent = _recurring_locations(years, coordinates, groups, missing_years)
    observed_recurrent = observed_recurrent.merge(
        fixed[["location_index", "su_id", "source_event_index", "x_m", "y_m"]],
        left_on="event_index",
        right_on="location_index",
        how="left",
        validate="one_to_one",
    ).drop(columns="location_index")
    observed_summary = _distance_summary(observed_recurrent, count_name="n")

    n_permutations = 99 if quick else int(settings["permutations"])
    seed_sequence = np.random.SeedSequence(int(settings["seed"]))
    seeds = [int(s.generate_state(1)[0]) for s in seed_sequence.spawn(n_permutations)]
    checkpoint_every = int(settings["checkpoint_every"])
    workers = int(settings.get("workers", 1))
    all_rows: list[dict] = []
    for start in range(0, n_permutations, checkpoint_every):
        batch_seeds = seeds[start : start + checkpoint_every]
        batch = Parallel(n_jobs=workers, prefer="processes")(
            delayed(_one_permutation)(
                seed,
                years,
                coordinates,
                groups,
                missing_years,
                str(settings["year_shuffle_scope"]),
            )
            for seed in batch_seeds
        )
        for offset, summary_rows in enumerate(batch):
            permutation = start + offset + 1
            for row in summary_rows:
                all_rows.append({"perm": permutation, **row})
        write_csv(pd.DataFrame(all_rows), output_path(cfg, "randomization", "time_label_permutation_null_checkpoint.csv"))
    null = pd.DataFrame(all_rows)

    summary_rows = []
    for _, obs in observed_summary.iterrows():
        category = obs["gap_model"]
        values = null.loc[null["gap_model"].eq(category), "median_distance_m"].dropna().to_numpy()
        observed_median = float(obs["median_distance_m"])
        at_or_below = int(np.sum(values <= observed_median))
        null_median = float(np.median(values))
        summary_rows.append(
            {
                "gap_model": category,
                "observed_n": int(obs["n"]),
                "observed_median_m": observed_median,
                "null_median_m": null_median,
                "null_CI_low_m": float(np.quantile(values, 0.025)),
                "null_CI_high_m": float(np.quantile(values, 0.975)),
                "observed_null_ratio": observed_median / null_median,
                "percent_shorter_than_null": 100 * (1 - observed_median / null_median),
                "n_permutations": len(values),
                "n_null_at_or_below_observed": at_or_below,
                "empirical_p_one_sided": (at_or_below + 1) / (len(values) + 1),
            }
        )

    audit = {
        "input_event_rows": before,
        "valid_observed_event_rows": len(events),
        "fixed_location_rows": len(fixed),
        "duplicate_location_records_removed": len(events) - len(fixed),
        "location_groups_with_multiple_years": int(multiplicity["distinct_years"].gt(1).sum()),
        "coordinate_definition": actual_mode,
        "distance_crs": str(events.crs),
        "distance_units": "metres",
        "nearest_prior_scope": "any strictly earlier mapped fixed location in the same slope unit",
        "time_label_shuffle_scope": settings["year_shuffle_scope"],
        "annual_label_counts_preserved": "Preserved across permutations after the configured fixed-location de-duplication; compare spatial_year_counts_before_after_dedup.csv for any change from the raw event file.",
        "location_coordinates_preserved": True,
        "deduplication": "su_id plus representative-point coordinates rounded to configured tolerance; earliest label retained before permutations" if settings.get("deduplicate_fixed_locations", True) else "none",
        "coordinate_tolerance_m": tolerance,
        "seed": int(settings["seed"]),
        "permutations": n_permutations,
        "quick_mode": quick,
    }
    write_csv(observed_recurrent, output_path(cfg, "data", "observed_recurrent_distances_distinct.csv"))
    write_csv(observed_summary, output_path(cfg, "tables", "observed_distinct_location_summary.csv"))
    write_csv(null, output_path(cfg, "randomization", "time_label_permutation_null_9999.csv" if not quick else "time_label_permutation_null_quick.csv"))
    write_csv(pd.DataFrame(summary_rows), output_path(cfg, "tables", "observed_vs_time_permutation_9999.csv" if not quick else "observed_vs_time_permutation_quick.csv"))
    write_csv(multiplicity, output_path(cfg, "audit", "spatial_location_multiplicity.csv"))
    before_year = events.groupby("event_year", observed=True).size().rename("raw_valid_event_rows")
    after_year = fixed.groupby("event_year", observed=True).size().rename("fixed_location_rows")
    year_counts = pd.concat([before_year, after_year], axis=1).fillna(0).astype(int).reset_index().rename(columns={"event_year": "year"})
    year_counts["rows_removed_by_deduplication"] = year_counts["raw_valid_event_rows"] - year_counts["fixed_location_rows"]
    write_csv(year_counts, output_path(cfg, "audit", "spatial_year_counts_before_after_dedup.csv"))
    write_json(audit, output_path(cfg, "models", "spatial_randomization_specification.json"))
    LOGGER.info("Spatial time-label randomization completed: %d permutations", n_permutations)
