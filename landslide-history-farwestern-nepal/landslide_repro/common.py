from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger("landslide_repro")

SU_ALIASES = ("su_id", "SU_ID", "suid", "SUID", "slope_unit_id", "slopeunit_id", "cat")
YEAR_ALIASES = ("year", "event_year", "Year", "YEAR", "year_original")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    root = path.parent
    for key in ("input_dir", "output_dir"):
        cfg["paths"][key] = str((root / cfg["paths"][key]).resolve())
    return cfg, root


def output_path(cfg: Mapping, *parts: str, mkdir: bool = True) -> Path:
    path = Path(cfg["paths"]["output_dir"]).joinpath(*parts)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def find_column(columns: Iterable[str], aliases: Sequence[str], label: str, required: bool = True) -> str | None:
    columns = list(columns)
    exact = {str(c): c for c in columns}
    lower = {str(c).lower(): c for c in columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
        if alias.lower() in lower:
            return lower[alias.lower()]
    if required:
        raise ValueError(f"Missing {label}; accepted columns: {list(aliases)}")
    return None


def canonicalize_ids(df: pd.DataFrame, require_year: bool = False) -> pd.DataFrame:
    out = df.copy()
    su_col = find_column(out.columns, SU_ALIASES, "slope-unit ID")
    if su_col != "su_id":
        out = out.rename(columns={su_col: "su_id"})
    out["su_id"] = out["su_id"].astype(str).str.strip()
    if require_year:
        year_col = find_column(out.columns, YEAR_ALIASES, "year")
        if year_col != "year":
            out = out.rename(columns={year_col: "year"})
        out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    return out


def gap_category(values: pd.Series | np.ndarray) -> pd.Categorical:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    labels = np.select(
        [x.eq(1), x.eq(2), x.between(3, 5), x.between(6, 10), x.gt(10)],
        ["1_year", "2_years", "3_5_years", "6_10_years", "gt10_years"],
        default=None,
    )
    return pd.Categorical(
        labels,
        categories=["1_year", "2_years", "3_5_years", "6_10_years", "gt10_years"],
        ordered=True,
    )


def zscore(series: pd.Series) -> tuple[pd.Series, float, float]:
    x = pd.to_numeric(series, errors="coerce")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Cannot standardize {series.name!r}: SD={sd}")
    return (x - mean) / sd, mean, sd


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)
    return path


def write_json(data: Mapping | list, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def input_file(cfg: Mapping, key: str) -> Path:
    return Path(cfg["paths"]["input_dir"]) / cfg["files"][key]


def weights_file(cfg: Mapping) -> Path:
    directory = Path(cfg["paths"]["input_dir"])
    matches = [directory / name for name in cfg["files"]["weights_candidates"] if (directory / name).exists()]
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one rainfall-grid weight file; found " + str([str(p) for p in matches])
        )
    return matches[0]


def software_manifest() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "sklearn", "geopandas", "shapely", "yaml"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[name] = "not installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def assert_unique(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    duplicated = df.duplicated(list(columns), keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, list(columns)].head(10).to_dict("records")
        raise ValueError(f"{label} has duplicate keys {list(columns)}; examples: {sample}")


def normal_interval(estimate: float, se: float, z: float = 1.959963984540054) -> tuple[float, float]:
    return estimate - z * se, estimate + z * se


def observed_years(cfg: Mapping) -> list[int]:
    return [int(y) for y in cfg["analysis"]["observed_years"]]
