from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .common import LOGGER, gap_category, observed_years, output_path, write_csv, write_json
from .data import read_history


CATEGORIES = ["1_year", "2_years", "3_5_years", "6_10_years", "gt10_years"]


@dataclass
class SwapState:
    rows: list[set[int]]
    attempts: int = 0
    accepted: int = 0


def _state_from_matrix(matrix: np.ndarray) -> SwapState:
    return SwapState([set(np.flatnonzero(row).tolist()) for row in matrix])


def _attempt_swaps(state: SwapState, n: int, rng: np.random.Generator, n_years: int) -> None:
    n_rows = len(state.rows)
    for _ in range(int(n)):
        a = int(rng.integers(0, n_rows))
        b = int(rng.integers(0, n_rows - 1))
        if b >= a:
            b += 1
        ca = int(rng.integers(0, n_years))
        cb = int(rng.integers(0, n_years - 1))
        if cb >= ca:
            cb += 1
        state.attempts += 1
        ac = ca in state.rows[a]
        ad = cb in state.rows[a]
        bc = ca in state.rows[b]
        bd = cb in state.rows[b]
        if ac and not ad and not bc and bd:
            state.rows[a].remove(ca)
            state.rows[a].add(cb)
            state.rows[b].remove(cb)
            state.rows[b].add(ca)
            state.accepted += 1
        elif not ac and ad and bc and not bd:
            state.rows[a].remove(cb)
            state.rows[a].add(ca)
            state.rows[b].remove(ca)
            state.rows[b].add(cb)
            state.accepted += 1


def _matrix_from_state(state: SwapState, n_years: int) -> np.ndarray:
    matrix = np.zeros((len(state.rows), n_years), dtype=np.int8)
    for row_index, columns in enumerate(state.rows):
        if columns:
            matrix[row_index, list(columns)] = 1
    return matrix


def _recurrence_counts(matrix: np.ndarray, years: np.ndarray, missing_years: list[int]) -> pd.DataFrame:
    """Count recurrence outcomes with the same rules as the original row loop.

    The original implementation iterated through every slope-unit/year cell in
    Python for every retained null draw.  This version iterates over the 25
    observed years and performs all slope-unit operations in NumPy.  It is an
    algebraic/vectorized implementation of the same risk-set definition; it
    does not alter the swap chain, random seeds, gap categories, or exclusions.
    """

    activity = np.asarray(matrix, dtype=np.int8)
    year_values = np.asarray(years, dtype=np.int64)
    n_rows = activity.shape[0]
    last_active_year = np.full(n_rows, -1, dtype=np.int64)
    numerators = np.zeros(len(CATEGORIES), dtype=np.int64)
    denominators = np.zeros(len(CATEGORIES), dtype=np.int64)

    for col, target_year_value in enumerate(year_values):
        target_year = int(target_year_value)
        current_active = activity[:, col].astype(bool, copy=False)
        has_prior = last_active_year >= 0
        crosses_gap = np.zeros(n_rows, dtype=bool)
        for missing in missing_years:
            crosses_gap |= (last_active_year < int(missing)) & (int(missing) < target_year)
        eligible = has_prior & ~crosses_gap

        if eligible.any():
            elapsed = target_year - last_active_year
            category_index = np.full(n_rows, -1, dtype=np.int8)
            category_index[elapsed == 1] = 0
            category_index[elapsed == 2] = 1
            category_index[(elapsed >= 3) & (elapsed <= 5)] = 2
            category_index[(elapsed >= 6) & (elapsed <= 10)] = 3
            category_index[elapsed > 10] = 4
            eligible &= category_index >= 0
            if eligible.any():
                denominators += np.bincount(category_index[eligible], minlength=len(CATEGORIES))
                active_eligible = eligible & current_active
                if active_eligible.any():
                    numerators += np.bincount(category_index[active_eligible], minlength=len(CATEGORIES))

        last_active_year[current_active] = target_year

    return pd.DataFrame(
        [
            {
                "gap_time_category": category,
                "n_risk": int(denominators[index]),
                "active": int(numerators[index]),
                "rate": float(numerators[index] / denominators[index]) if denominators[index] else np.nan,
            }
            for index, category in enumerate(CATEGORIES)
        ]
    )


def _statistics(counts: pd.DataFrame) -> dict[str, float]:
    rates = counts.set_index("gap_time_category")["rate"]
    short = counts[counts["gap_time_category"].isin(["1_year", "2_years", "3_5_years"])]
    long = counts[counts["gap_time_category"].isin(["6_10_years", "gt10_years"])]
    short_rate = short["active"].sum() / short["n_risk"].sum()
    long_rate = long["active"].sum() / long["n_risk"].sum()
    result = {f"rate_{category}": float(rates.get(category, np.nan)) for category in CATEGORIES}
    result["short_1_5_rate"] = float(short_rate)
    result["long_gt5_rate"] = float(long_rate)
    result["short_vs_long_risk_ratio"] = float(short_rate / long_rate) if long_rate > 0 else np.nan
    result["one_vs_gt10_risk_ratio"] = float(rates["1_year"] / rates["gt10_years"]) if rates["gt10_years"] > 0 else np.nan
    return result


def _empirical_summary(observed: Mapping[str, float], null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for statistic, obs in observed.items():
        values = pd.to_numeric(null[statistic], errors="coerce").dropna().to_numpy()
        center = float(np.median(values))
        lower_tail = (1 + int(np.sum(values <= obs))) / (len(values) + 1)
        upper_tail = (1 + int(np.sum(values >= obs))) / (len(values) + 1)
        two_sided = min(1.0, 2 * min(lower_tail, upper_tail))
        rows.append(
            {
                "statistic": statistic,
                "observed": obs,
                "null_median": center,
                "null_q025": float(np.quantile(values, 0.025)),
                "null_q975": float(np.quantile(values, 0.975)),
                "p_lower": lower_tail,
                "p_upper": upper_tail,
                "p_two_sided": two_sided,
                "n_permutations": len(values),
            }
        )
    return pd.DataFrame(rows)


def _lag1_autocorrelation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.std(values) == 0:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _split_rhat(chain_values: list[np.ndarray]) -> float:
    """Classic split-Rhat diagnostic using equal-length half chains."""
    halves: list[np.ndarray] = []
    for values in chain_values:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        half = len(values) // 2
        if half >= 2:
            halves.extend([values[:half], values[-half:]])
    if len(halves) < 2:
        return np.nan
    n = min(len(values) for values in halves)
    samples = np.vstack([values[:n] for values in halves])
    within = float(np.mean(np.var(samples, axis=1, ddof=1)))
    if within <= 0:
        return np.nan
    between = float(n * np.var(np.mean(samples, axis=1), ddof=1))
    variance = ((n - 1) / n) * within + between / n
    return float(np.sqrt(variance / within))


def _mixing_diagnostics(null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    statistics = [
        "short_vs_long_risk_ratio",
        "one_vs_gt10_risk_ratio",
        *[f"rate_{category}" for category in CATEGORIES],
    ]
    for statistic in statistics:
        chain_values = [
            pd.to_numeric(part[statistic], errors="coerce").dropna().to_numpy()
            for _, part in null.groupby("chain", sort=True, observed=True)
        ]
        autocorrelations = [_lag1_autocorrelation(values) for values in chain_values]
        finite_rhos = np.asarray([value for value in autocorrelations if np.isfinite(value)], dtype=float)
        mean_rho = float(np.mean(finite_rhos)) if len(finite_rhos) else np.nan
        n_total = int(sum(len(values) for values in chain_values))
        approximate_ess = (
            float(n_total * (1 - mean_rho) / (1 + mean_rho))
            if np.isfinite(mean_rho) and mean_rho > -1
            else np.nan
        )
        rows.append(
            {
                "statistic": statistic,
                "split_rhat": _split_rhat(chain_values),
                "mean_chain_lag1_autocorrelation": mean_rho,
                "approximate_ess_from_lag1": min(float(n_total), approximate_ess) if np.isfinite(approximate_ess) else np.nan,
                "retained_draws": n_total,
                "chain_lag1_autocorrelations": "|".join("nan" if not np.isfinite(value) else f"{value:.8g}" for value in autocorrelations),
            }
        )
    return pd.DataFrame(rows)


def run(cfg: Mapping, quick: bool = False) -> None:
    history = read_history(cfg)
    years = np.asarray(observed_years(cfg), dtype=int)
    table = history.pivot(index="su_id", columns="year", values="active").reindex(columns=years)
    if table.isna().any().any():
        raise ValueError("The activity panel is not balanced; cannot run fixed-margin randomization")
    matrix_all = table.to_numpy(dtype=np.int8)
    nonzero = matrix_all.sum(axis=1) > 0
    matrix = matrix_all[nonzero]
    missing_years = [int(x) for x in cfg["analysis"]["missing_years"]]
    observed_counts = _recurrence_counts(matrix, years, missing_years)
    observed_stats = _statistics(observed_counts)

    settings = cfg["temporal_null"]
    chains = int(settings["chains"])
    retained = 99 if quick else int(settings["retained_draws"])
    burn = 1000 if quick else int(settings["burn_in_attempted_swaps"])
    thin = 100 if quick else int(settings["thinning_attempted_swaps"])
    seeds = [int(x) for x in settings["seeds"]]
    if len(seeds) != chains:
        raise ValueError("temporal_null.seeds must contain one seed per chain")

    allocations = [retained // chains + (1 if chain < retained % chains else 0) for chain in range(chains)]
    draws = []
    diagnostics = []
    observed_row_sums = matrix.sum(axis=1)
    observed_col_sums = matrix.sum(axis=0)
    permutation_id = 0
    for chain in range(chains):
        rng = np.random.default_rng(seeds[chain])
        state = _state_from_matrix(matrix)
        _attempt_swaps(state, burn, rng, len(years))
        attempts_after_burn = state.attempts
        for within_chain in range(allocations[chain]):
            _attempt_swaps(state, thin, rng, len(years))
            draw_matrix = _matrix_from_state(state, len(years))
            if not np.array_equal(draw_matrix.sum(axis=1), observed_row_sums):
                raise AssertionError("Row margins changed during temporal randomization")
            if not np.array_equal(draw_matrix.sum(axis=0), observed_col_sums):
                raise AssertionError("Year margins changed during temporal randomization")
            permutation_id += 1
            stats = _statistics(_recurrence_counts(draw_matrix, years, missing_years))
            draws.append({"permutation": permutation_id, "chain": chain + 1, "draw_within_chain": within_chain + 1, **stats})
            if (within_chain + 1) % int(settings["checkpoint_every"]) == 0:
                write_csv(pd.DataFrame(draws), output_path(cfg, "randomization", "temporal_fixed_margin_null_checkpoint.csv"))
        diagnostics.append(
            {
                "chain": chain + 1,
                "seed": seeds[chain],
                "retained_draws": allocations[chain],
                "burn_in_attempted_swaps": burn,
                "thinning_attempted_swaps": thin,
                "total_attempts": state.attempts,
                "total_accepted": state.accepted,
                "acceptance_rate": state.accepted / state.attempts,
                "attempts_after_burn": attempts_after_burn,
                "row_margins_preserved": True,
                "year_margins_preserved": True,
            }
        )

    null = pd.DataFrame(draws)
    summary = _empirical_summary(observed_stats, null)
    mixing = _mixing_diagnostics(null)
    write_csv(observed_counts, output_path(cfg, "tables", "temporal_observed_gap_rates.csv"))
    write_csv(pd.DataFrame([observed_stats]), output_path(cfg, "tables", "temporal_observed_statistics.csv"))
    write_csv(null, output_path(cfg, "randomization", "temporal_fixed_margin_null.csv"))
    write_csv(pd.DataFrame(diagnostics), output_path(cfg, "randomization", "temporal_chain_diagnostics.csv"))
    write_csv(mixing, output_path(cfg, "randomization", "temporal_chain_mixing_diagnostics.csv"))
    write_csv(summary, output_path(cfg, "tables", "temporal_fixed_margin_summary.csv"))
    write_json(
        {
            "algorithm": "Uniformly propose two distinct rows and two distinct years; swap only checkerboard 2x2 submatrices, retaining rejected proposals as self-transitions.",
            "preserved": ["slope-unit activity totals", "annual active-slope-unit totals"],
            "chains": chains,
            "retained_draws": retained,
            "burn_in_attempted_swaps": burn,
            "thinning_attempted_swaps": thin,
            "mixing_diagnostics": "Split-Rhat, per-chain lag-1 autocorrelation, and a lag-1 approximate effective sample size are written for both risk ratios and all five category rates.",
            "quick_mode": quick,
        },
        output_path(cfg, "models", "temporal_randomization_specification.json"),
    )
    LOGGER.info("Temporal fixed-margin randomization completed: %d draws", retained)
