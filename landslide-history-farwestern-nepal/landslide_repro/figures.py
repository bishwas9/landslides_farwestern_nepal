from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

from .common import LOGGER, output_path, write_csv, write_json


GAP_ORDER = ["1_year", "2_years", "3_5_years", "6_10_years", "gt10_years"]
GAP_LABELS = {
    "overall": "Overall",
    "1_year": "1 year",
    "2_years": "2 years",
    "3_5_years": "3–5 years",
    "6_10_years": "6–10 years",
    "gt10_years": ">10 years",
}
GAP_COLORS = {
    "overall": "#4D4D4D",
    "1_year": "#0072B2",
    "2_years": "#E69F00",
    "3_5_years": "#009E73",
    "6_10_years": "#D55E00",
    "gt10_years": "#7B3294",
}
MODEL_LABELS = {
    "P0_static": "Static",
    "P1_history": "History",
    "P2_history_lagged_rain": "History + lagged rainfall",
}
MODEL_COLORS = {
    "P0_static": "#6F6F6F",
    "P1_history": "#0072B2",
    "P2_history_lagged_rain": "#D55E00",
}


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )
    # FontTools otherwise emits thousands of INFO lines while subsetting PDFs.
    import logging

    logging.getLogger("fontTools").setLevel(logging.WARNING)


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot generate figures; required output is missing: {path}")
    return pd.read_csv(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_figure(fig, path: Path, *, bbox_inches="tight", dpi: int | None = None) -> None:
    """Write atomically and verify that Matplotlib can reopen raster output."""
    temporary = path.with_name(f".{path.name}.tmp")
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format=path.suffix.lstrip("."),
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=0.04,
        facecolor="white",
    )
    payload = buffer.getvalue()
    if path.suffix.lower() == ".png" and not payload.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"):
        raise RuntimeError(f"Incomplete PNG stream generated for {path}")
    temporary.write_bytes(payload)
    if not temporary.exists() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Figure write failed: {temporary}")
    if path.suffix.lower() == ".png":
        from PIL import Image

        with Image.open(temporary) as image:
            image.verify()
    os.replace(temporary, path)


def _record_figure(path: Path, stem: str, sources: list[Path], records: list[dict]) -> None:
    records.append(
        {
            "figure": stem.split("_", 1)[0],
            "file": str(path),
            "format": path.suffix.lstrip("."),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "source_files": "|".join(str(source) for source in sources),
            "source_sha256": "|".join(_sha256(source) for source in sources),
        }
    )


def _save(fig, cfg: Mapping, stem: str, sources: list[Path], records: list[dict]) -> None:
    png = output_path(cfg, "figures", f"{stem}.png")
    pdf = output_path(cfg, "figures", f"{stem}.pdf")
    _write_figure(fig, png, dpi=600)
    _write_figure(fig, pdf)
    for path in (png, pdf):
        _record_figure(path, stem, sources, records)


def _save_panel(
    fig,
    ax,
    cfg: Mapping,
    stem: str,
    sources: list[Path],
    records: list[dict],
    extra_axes: list | None = None,
) -> None:
    """Export one panel from a combined figure without rasterizing vector artwork."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    retained_axes = [ax, *(extra_axes or [])]
    boxes = [ax.get_tightbbox(renderer)]
    for extra_ax in extra_axes or []:
        boxes.append(extra_ax.get_tightbbox(renderer))
    bbox = Bbox.union(boxes).expanded(1.075, 1.095).transformed(fig.dpi_scale_trans.inverted())
    png = output_path(cfg, "figures", f"{stem}.png")
    pdf = output_path(cfg, "figures", f"{stem}.pdf")
    visibility = {item: item.get_visible() for item in fig.axes}
    layout_engine = fig.get_layout_engine()
    try:
        # Freeze the combined layout before hiding neighbouring panels; otherwise
        # constrained layout moves the retained panel after its crop box is measured.
        fig.set_layout_engine(None)
        for item in fig.axes:
            item.set_visible(item in retained_axes)
        _write_figure(fig, png, dpi=600, bbox_inches=bbox)
        _write_figure(fig, pdf, bbox_inches=bbox)
    finally:
        for item, visible in visibility.items():
            item.set_visible(visible)
        fig.set_layout_engine(layout_engine)
    for path in (png, pdf):
        _record_figure(path, stem, sources, records)


def _panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def _figure3(cfg: Mapping, records: list[dict], quick: bool) -> None:
    import matplotlib.pyplot as plt

    root = Path(cfg["paths"]["output_dir"])
    observed_path = root / "tables" / "temporal_observed_gap_rates.csv"
    stats_path = root / "tables" / "temporal_observed_statistics.csv"
    summary_path = root / "tables" / "temporal_fixed_margin_summary.csv"
    null_path = root / "randomization" / "temporal_fixed_margin_null.csv"
    observed = _read_required(observed_path).set_index("gap_time_category").reindex(GAP_ORDER)
    observed_stats = _read_required(stats_path).iloc[0]
    summary = _read_required(summary_path).set_index("statistic")
    null = _read_required(null_path)
    expected_draws = 99 if quick else 9999
    if len(null) != expected_draws:
        raise ValueError(f"Figure 3 requires {expected_draws:,} retained temporal-null draws; found {len(null):,}")
    if not quick and "chain" in null and null.groupby("chain").size().to_dict() != {1: 3333, 2: 3333, 3: 3333}:
        raise ValueError("Figure 3 requires 3 balanced temporal-null chains of 3,333 retained draws")

    x = np.arange(len(GAP_ORDER))
    rate_columns = [f"rate_{category}" for category in GAP_ORDER]
    null_rates = null[rate_columns].apply(pd.to_numeric, errors="coerce")
    rate_summary = summary.reindex(rate_columns)
    null_median = rate_summary["null_median"].to_numpy(dtype=float)
    null_low = rate_summary["null_q025"].to_numpy(dtype=float)
    null_high = rate_summary["null_q975"].to_numpy(dtype=float)
    observed_rate = observed["rate"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(7.2, 6.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    ax_a.fill_between(x, 100 * null_low, 100 * null_high, color="#56B4E9", alpha=0.28, label="Randomized 95% interval")
    ax_a.plot(x, 100 * null_median, color="#0072B2", marker="o", markerfacecolor="white", label="Randomized median")
    ax_a.plot(x, 100 * observed_rate, color="#D55E00", marker="D", markersize=4.5, label="Observed")
    ax_a.set_xticks(x, [GAP_LABELS[c] for c in GAP_ORDER])
    ax_a.tick_params(axis="x", labelrotation=25)
    ax_a.set_ylabel("Annual recurrence probability (%)")
    ax_a.set_xlabel("Elapsed time since previous mapped activity")
    ax_a.set_title("Observed recurrence and fixed-margin null", pad=8)
    ax_a.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax_a.legend(frameon=False, ncol=1, loc="upper right", fontsize=6.4)
    ax_a.text(-0.28, 1.10, "a", transform=ax_a.transAxes, fontsize=11, fontweight="bold", va="bottom")

    departure = 100 * (observed_rate - null_median)
    y = np.arange(len(GAP_ORDER))[::-1]
    ax_b.axvline(0, color="#777777", linewidth=0.8)
    for yi, value, category in zip(y, departure, GAP_ORDER):
        ax_b.hlines(yi, min(0, value), max(0, value), color=GAP_COLORS[category], linewidth=2)
        ax_b.plot(value, yi, "o", color=GAP_COLORS[category], markersize=5)
        ax_b.annotate(
            f"{value:+.2f}",
            (value, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35, "alpha": 0.86},
        )
    ax_b.set_yticks(y, [GAP_LABELS[c] for c in GAP_ORDER])
    ax_b.set_xlabel("Observed − randomized-median probability\n(percentage points)")
    ax_b.set_ylabel("Elapsed-time category")
    ax_b.set_title("Departure from the randomized chronology", pad=8)
    ax_b.set_xlim(min(-1.25, float(departure.min()) - 0.30), max(2.50, float(departure.max()) + 0.40))
    ax_b.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax_b.text(-0.28, 1.10, "b", transform=ax_b.transAxes, fontsize=11, fontweight="bold", va="bottom")

    ratio_specs = [
        ("one_vs_gt10_risk_ratio", "1 year vs >10 years"),
        ("short_vs_long_risk_ratio", "≤5 years vs >5 years"),
    ]
    datasets = [pd.to_numeric(null[column], errors="coerce").dropna().to_numpy() for column, _ in ratio_specs]
    positions = [1, 0]
    violin = ax_c.violinplot(datasets, positions=positions, vert=False, widths=0.65, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#9ECAE1")
        body.set_edgecolor("#2B8CBE")
        body.set_alpha(0.38)
    for position, (column, _), values in zip(positions, ratio_specs, datasets):
        row = summary.loc[column]
        median = float(row["null_median"])
        low = float(row["null_q025"])
        high = float(row["null_q975"])
        observed_value = float(row["observed"])
        ax_c.hlines(position, low, high, color="#0072B2", linewidth=2.0, zorder=2)
        ax_c.plot(median, position, "o", markerfacecolor="white", markeredgecolor="#0072B2", zorder=3)
        ax_c.plot(observed_value, position, "D", color="#D55E00", markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax_c.annotate(
            f"RR={observed_value:.2f}; p$_{{MC}}$={float(row['p_upper']):.4f}",
            (observed_value, position),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.2,
        )
    ax_c.set_yticks(positions, [label for _, label in ratio_specs])
    ax_c.set_xlabel("Recurrence risk ratio")
    ax_c.set_title("Prespecified aggregate recurrence contrasts", pad=8)
    ax_c.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    all_low = min(float(summary.loc[column, "null_q025"]) for column, _ in ratio_specs)
    all_observed = max(float(summary.loc[column, "observed"]) for column, _ in ratio_specs)
    ax_c.set_xlim(all_low - 0.20, all_observed + 0.82)
    _panel_label(ax_c, "c")

    sources = [observed_path, stats_path, summary_path, null_path]
    _save(fig, cfg, "Figure3_temporal_fixed_margin", sources, records)
    _save_panel(fig, ax_a, cfg, "Figure3a_observed_recurrence_vs_null", sources, records)
    _save_panel(fig, ax_b, cfg, "Figure3b_observed_minus_null", sources, records)
    _save_panel(fig, ax_c, cfg, "Figure3c_null_risk_ratios", sources, records)
    plt.close(fig)


def _figure4(cfg: Mapping, records: list[dict], quick: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    root = Path(cfg["paths"]["output_dir"])
    ame_path = root / "tables" / "Figure4_probability_scale_AME_two_way_cluster.csv"
    contrast_path = root / "tables" / "Figure4_AME_contrasts_vs_gt10_two_way_cluster.csv"
    multiplicity_path = root / "tables" / "Figure4_multiplicity_adjusted_tests.csv"
    ame = _read_required(ame_path)
    contrasts = _read_required(contrast_path)
    if multiplicity_path.exists():
        multiplicity = pd.read_csv(multiplicity_path)
        multiplicity = multiplicity.loc[multiplicity["history_state"].ne("all interactions"), ["duration_days", "history_state", "p_holm"]]
        contrasts = contrasts.merge(multiplicity, on=["duration_days", "history_state"], how="left", validate="one_to_one")
    else:
        contrasts["p_holm"] = np.nan
    durations = sorted(pd.to_numeric(ame["duration_days"], errors="raise").astype(int).unique().tolist())
    contrast_states = GAP_ORDER[:-1]

    fig = plt.figure(figsize=(10.4, 6.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    matrix = (
        contrasts.pivot(index="history_state", columns="duration_days", values="AME_contrast_percentage_points")
        .reindex(index=contrast_states, columns=durations)
    )
    limit = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
    image = ax_a.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    for row, state in enumerate(contrast_states):
        for col, duration in enumerate(durations):
            item = contrasts.loc[(contrasts["history_state"].eq(state)) & (contrasts["duration_days"].eq(duration))]
            if item.empty:
                continue
            value = float(item["AME_contrast_percentage_points"].iloc[0])
            low = float(item["CI95_low_percentage_points"].iloc[0])
            high = float(item["CI95_high_percentage_points"].iloc[0])
            pointwise = low > 0 or high < 0
            holm = bool(pd.notna(item["p_holm"].iloc[0]) and float(item["p_holm"].iloc[0]) < 0.05)
            star = "*" if holm else "†" if pointwise else ""
            color = "white" if abs(value) > 0.55 * limit else "#222222"
            ax_a.text(col, row, f"{value:+.1f}{star}", ha="center", va="center", fontsize=8, color=color)
    ax_a.set_xticks(np.arange(len(durations)), durations)
    ax_a.set_yticks(np.arange(len(contrast_states)), [GAP_LABELS[s] for s in contrast_states])
    ax_a.set_xlabel("Rainfall accumulation duration (days)")
    ax_a.set_ylabel("Time since previous shallow activity")
    colorbar = fig.colorbar(image, ax=ax_a, orientation="horizontal", fraction=0.045, pad=0.075)
    colorbar.set_label("AME contrast vs >10 years (percentage points per +1 local-SD rainfall)")
    ax_a.text(0.99, 1.015, "* Holm-adjusted p < 0.05;  † pointwise 95% CI excludes zero but does not survive Holm", transform=ax_a.transAxes, ha="right", va="bottom", fontsize=6.7, color="#555555")
    _panel_label(ax_a, "a")

    one_year = contrasts.loc[contrasts["history_state"].eq("1_year")].sort_values("duration_days")
    x = one_year["duration_days"].to_numpy(dtype=float)
    y = one_year["AME_contrast_percentage_points"].to_numpy(dtype=float)
    low = one_year["CI95_low_percentage_points"].to_numpy(dtype=float)
    high = one_year["CI95_high_percentage_points"].to_numpy(dtype=float)
    ax_b.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax_b.fill_between(x, low, high, color="#9ECAE1", alpha=0.45)
    ax_b.errorbar(x, y, yerr=[y - low, high - y], color="#2B8CBE", marker="o", markersize=3.5, linewidth=1.4, capsize=2)
    ax_b.set_xticks(durations)
    ax_b.set_xlabel("Rainfall accumulation duration (days)")
    ax_b.set_ylabel("AME contrast (percentage points)", fontsize=8, labelpad=3)
    ax_b.set_title("1-year response relative to >10-year state", pad=5)
    ax_b.grid(color="#E6E6E6", linewidth=0.6)
    ax_b.text(0.02, 0.96, "b", transform=ax_b.transAxes, fontsize=11, fontweight="bold", va="top", zorder=10, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    focal_duration = 10 if 10 in durations else min(durations, key=lambda value: abs(value - 10))
    focal = ame.loc[ame["duration_days"].eq(focal_duration)].set_index("history_state").reindex(GAP_ORDER)
    positions = np.arange(len(GAP_ORDER))[::-1]
    estimate = focal["AME_percentage_points"].to_numpy(dtype=float)
    lower = focal["CI95_low_percentage_points"].to_numpy(dtype=float)
    upper = focal["CI95_high_percentage_points"].to_numpy(dtype=float)
    ax_c.axvline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax_c.errorbar(estimate, positions, xerr=[estimate - lower, upper - estimate], fmt="o", color="#2B8CBE", capsize=2.5, markersize=4)
    for value, position in zip(estimate, positions):
        ax_c.annotate(f"{value:+.2f}", (value, position), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    ax_c.set_yticks(positions, [GAP_LABELS[state] for state in GAP_ORDER])
    ax_c.set_xlabel(f"Rx{focal_duration} AME per +1 local SD (percentage points)")
    if quick and focal_duration != 10:
        ax_c.set_title(f"Quick-mode diagnostic: Rx{focal_duration} (Rx10 not fitted)")
    ax_c.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax_c.text(0.02, 0.96, "c", transform=ax_c.transAxes, fontsize=11, fontweight="bold", va="top", zorder=10, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    sources = [ame_path, contrast_path] + ([multiplicity_path] if multiplicity_path.exists() else [])
    _save(fig, cfg, "Figure4_rainfall_history_interaction", sources, records)
    _save_panel(fig, ax_a, cfg, "Figure4a_ame_contrast_heatmap", sources, records, extra_axes=[colorbar.ax])
    _save_panel(fig, ax_b, cfg, "Figure4b_one_year_contrast_by_duration", sources, records)
    _save_panel(fig, ax_c, cfg, f"Figure4c_rx{focal_duration}_state_ame", sources, records)
    plt.close(fig)


def _figure5(cfg: Mapping, records: list[dict], quick: bool) -> None:
    import matplotlib.pyplot as plt

    root = Path(cfg["paths"]["output_dir"])
    suffix = "quick" if quick else "9999"
    summary_path = root / "tables" / f"observed_vs_time_permutation_{suffix}.csv"
    null_path = root / "randomization" / f"time_label_permutation_null_{suffix}.csv"
    summary = _read_required(summary_path).set_index("gap_model")
    null = _read_required(null_path)
    categories = ["overall", *GAP_ORDER]
    positions = np.arange(len(categories))[::-1]
    distributions = [
        pd.to_numeric(null.loc[null["gap_model"].eq(category), "median_distance_m"], errors="coerce").dropna().to_numpy()
        for category in categories
    ]
    expected_draws = 99 if quick else 9999
    if any(len(values) != expected_draws for values in distributions):
        counts = {category: len(values) for category, values in zip(categories, distributions)}
        raise ValueError(f"Figure 5 requires {expected_draws:,} spatial-null draws per category; found {counts}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.2, 4.8), constrained_layout=True)
    for position, category in zip(positions, categories):
        row = summary.loc[category]
        ax_a.hlines(
            position,
            float(row["null_CI_low_m"]),
            float(row["null_CI_high_m"]),
            color="#9A9A9A",
            linewidth=2.4,
            zorder=1,
        )
        ax_a.plot(float(row["null_median_m"]), position, "s", color="#777777", markersize=4.5, zorder=3)
        ax_a.plot(
            float(row["observed_median_m"]),
            position,
            "o",
            color="#0072B2",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.5,
            zorder=4,
        )
    ax_a.set_yticks(positions, [GAP_LABELS[category] for category in categories])
    ax_a.set_xlabel("Nearest earlier-event distance (m)")
    ax_a.set_ylabel("Elapsed-time category")
    ax_a.set_title("Observed and randomized distances", pad=8)
    ax_a.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax_a.legend(
        handles=[
            plt.Line2D([0], [0], marker="s", color="#777777", linestyle="-", label="Randomized median (95% interval)"),
            plt.Line2D([0], [0], marker="o", color="#0072B2", markeredgecolor="white", linestyle="none", label="Observed median"),
        ],
        frameon=False,
        loc="upper right",
        fontsize=7.0,
    )
    _panel_label(ax_a, "a")

    ax_b.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    for position, category in zip(positions, categories):
        row = summary.loc[category]
        observed = float(row["percent_shorter_than_null"])
        ax_b.hlines(position, min(0, observed), max(0, observed), color=GAP_COLORS[category], linewidth=2.0)
        ax_b.plot(observed, position, "D", color=GAP_COLORS[category], markersize=5)
        significant = float(row["empirical_p_one_sided"]) < 0.05
        ax_b.annotate(
            f"{observed:+.1f}%{'*' if significant else ''}",
            (observed, position),
            xytext=(5 if observed >= 0 else -5, 0),
            textcoords="offset points",
            ha="left" if observed >= 0 else "right",
            va="center",
            fontsize=8,
        )
    ax_b.set_yticks(positions, [GAP_LABELS[c] for c in categories])
    ax_b.set_xlabel("Observed distance reduction relative to null (%)")
    ax_b.set_ylabel("Elapsed-time category")
    ax_b.set_title("Localization relative to chronology null", pad=8)
    ax_b.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax_b.set_xlim(-7.0, 18.5)
    ax_b.text(
        0.99,
        0.015,
        "* one-sided empirical p < 0.05; exact p-values in Table S6",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )
    _panel_label(ax_b, "b")

    sources = [summary_path, null_path]
    _save(fig, cfg, "Figure5_spatial_localization", sources, records)
    _save_panel(fig, ax_a, cfg, "Figure5a_spatial_distance_null", sources, records)
    _save_panel(fig, ax_b, cfg, "Figure5b_spatial_localization_percent", sources, records)
    plt.close(fig)


def _downsample(x: np.ndarray, y: np.ndarray, maximum: int = 1800) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= maximum:
        return x, y
    indices = np.unique(np.linspace(0, len(x) - 1, maximum).astype(int))
    return x[indices], y[indices]


def _annual_budget_capture_curve(part: pd.DataFrame, fractions: np.ndarray) -> np.ndarray:
    total_positive = int(part["observed"].sum())
    if total_positive == 0:
        return np.full(len(fractions), np.nan)
    yearly = [
        frame.sort_values("predicted_probability", ascending=False, kind="mergesort")
        for _, frame in part.groupby("test_year", sort=True, observed=True)
    ]
    values = []
    for fraction in fractions:
        captured = 0
        for frame in yearly:
            k = max(1, int(np.ceil(len(frame) * float(fraction))))
            captured += int(frame.head(k)["observed"].sum())
        values.append(100 * captured / total_positive)
    return np.asarray(values, dtype=float)


def _figure6(cfg: Mapping, records: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    root = Path(cfg["paths"]["output_dir"])
    prediction_path = root / "prediction" / "forward_predictions.csv"
    pooled_path = root / "prediction" / "forward_metrics_pooled.csv"
    yearly_path = root / "prediction" / "forward_metrics_by_year.csv"
    predictions = _read_required(prediction_path)
    pooled = _read_required(pooled_path).set_index("model")
    yearly = _read_required(yearly_path)
    required_models = ["P0_static", "P1_history", "P2_history_lagged_rain"]
    missing_models = [model for model in required_models if model not in set(predictions["model"])]
    if missing_models:
        raise ValueError(f"Cannot generate Figure 6; missing prediction models: {missing_models}")

    fig = plt.figure(figsize=(10.4, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    line_styles = {"P0_static": "-", "P1_history": "-", "P2_history_lagged_rain": "--"}
    for model in required_models:
        part = predictions.loc[predictions["model"].eq(model)]
        precision, recall, _ = precision_recall_curve(part["observed"].to_numpy(), part["predicted_probability"].to_numpy())
        ap = float(pooled.loc[model, "average_precision"])
        ax_a.plot(recall, precision, color=MODEL_COLORS[model], linestyle=line_styles[model], linewidth=1.6, label=f"{MODEL_LABELS[model]} (AP = {ap:.3f})")
    prevalence = float(predictions["observed"].mean())
    ax_a.axhline(prevalence, color="#56B4E9", linewidth=1.0, linestyle="--", label=f"Prevalence = {prevalence:.3f}")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Recall")
    ax_a.set_ylabel("Precision")
    ax_a.set_title("Pre-season precision–recall performance")
    ax_a.grid(color="#E6E6E6", linewidth=0.6)
    ax_a.legend(frameon=False, loc="upper right")
    _panel_label(ax_a, "a")

    budget_fractions = np.linspace(0.001, 0.20, 200)
    for model in required_models:
        part = predictions.loc[predictions["model"].eq(model)].copy()
        captured = _annual_budget_capture_curve(part, budget_fractions)
        ax_b.plot(100 * budget_fractions, captured, color=MODEL_COLORS[model], linestyle=line_styles[model], linewidth=1.6, label=MODEL_LABELS[model])
        capture_5 = float(_annual_budget_capture_curve(part, np.asarray([0.05]))[0])
        ax_b.plot(5, capture_5, "o", color=MODEL_COLORS[model])
        offsets = {"P0_static": (7, -10), "P1_history": (7, 8), "P2_history_lagged_rain": (7, -14)}
        ax_b.annotate(f"{capture_5:.1f}%", (5, capture_5), xytext=offsets[model], textcoords="offset points", fontsize=7.3, color=MODEL_COLORS[model])
    ax_b.plot([0, 20], [0, 20], color="#009E73", linestyle="--", linewidth=1.1, label="Random ranking")
    ax_b.axvline(5, color="#56B4E9", linestyle=":", linewidth=1.0)
    ax_b.set_xlim(0, 20)
    ax_b.set_ylim(0, 100)
    ax_b.set_xlabel("Highest-ranked slope units selected within each year (%)")
    ax_b.set_ylabel("Future active slope-unit years captured (%)")
    ax_b.set_title("Concentration of future activity")
    ax_b.grid(color="#E6E6E6", linewidth=0.6)
    ax_b.legend(frameon=False, loc="lower right")
    _panel_label(ax_b, "b")

    for model in required_models:
        part = yearly.loc[yearly["model"].eq(model)].sort_values("test_year")
        ax_c.plot(part["test_year"], part["average_precision"], color=MODEL_COLORS[model], linestyle=line_styles[model], marker="o", markersize=3.5, linewidth=1.4, label=MODEL_LABELS[model])
    prevalence_by_year = yearly.loc[yearly["model"].eq("P0_static")].sort_values("test_year")
    ax_c.plot(prevalence_by_year["test_year"], prevalence_by_year["prevalence"], color="#009E73", linestyle="--", linewidth=1.2, label="Annual prevalence")
    years = sorted(yearly["test_year"].unique().astype(int).tolist())
    ax_c.set_xticks(years)
    ax_c.set_xlabel("Genuinely held-out year")
    ax_c.set_ylabel("Average precision")
    ax_c.set_title("Prediction performance across held-out years")
    ax_c.grid(color="#E6E6E6", linewidth=0.6)
    ax_c.legend(frameon=False, ncol=2, loc="upper left")
    _panel_label(ax_c, "c")

    sources = [prediction_path, pooled_path, yearly_path]
    _save(fig, cfg, "Figure6_forward_prediction", sources, records)
    _save_panel(fig, ax_a, cfg, "Figure6a_precision_recall", sources, records)
    _save_panel(fig, ax_b, cfg, "Figure6b_cumulative_capture", sources, records)
    _save_panel(fig, ax_c, cfg, "Figure6c_yearly_average_precision", sources, records)
    plt.close(fig)


def _figure_s1_recurrence(cfg: Mapping, records: list[dict]) -> None:
    """Supplementary forest plot for the logistic mixed-effects recurrence model."""
    import matplotlib.pyplot as plt

    root = Path(cfg["paths"]["output_dir"])
    source = root / "tables" / "recurrence_focal_effects.csv"
    data = _read_required(source)
    definitions = [
        ("prior_active_years_z", "Prior active-year burden\n(1 SD in log[1 + prior active years])", "#0072B2"),
        ("[T.1_year]", "1 year vs >10 years", GAP_COLORS["1_year"]),
        ("[T.2_years]", "2 years vs >10 years", GAP_COLORS["2_years"]),
        ("[T.3_5_years]", "3–5 years vs >10 years", GAP_COLORS["3_5_years"]),
        ("[T.6_10_years]", "6–10 years vs >10 years", GAP_COLORS["6_10_years"]),
        ("monsoon_z", "Within-slope monsoon rainfall\n(1 SD)", "#777777"),
    ]
    rows = []
    for pattern, label, color in definitions:
        match = data.loc[data["term"].astype(str).str.contains(pattern, regex=False)]
        if len(match) != 1:
            raise ValueError(f"Expected one recurrence term containing {pattern!r}; found {len(match)}")
        row = match.iloc[0]
        rows.append((label, float(row["odds_ratio"]), float(row["interval_low"]), float(row["interval_high"]), color))

    fig, ax = plt.subplots(figsize=(5.6, 3.65), constrained_layout=True)
    positions = np.arange(len(rows))
    ax.axvline(1, color="#555555", linewidth=0.8)
    for position, (_, estimate, low, high, color) in zip(positions, rows):
        ax.errorbar(estimate, position, xerr=[[estimate - low], [high - estimate]], fmt="o", color=color, capsize=2.5, markersize=5.8)
        ax.text(high * 1.025, position, f"{estimate:.2f} [{low:.2f}, {high:.2f}]", va="center", fontsize=6.8)
    ax.set_yticks(positions, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_title("Recurrence mixed-effects model: focal effects", pad=8)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.set_xlim(0.80, 2.55)
    _save(fig, cfg, "FigureS1_recurrence_model_effects", [source], records)
    plt.close(fig)


def _figure_s2_temporal_diagnostics(cfg: Mapping, records: list[dict]) -> None:
    """Supplementary mixing diagnostics for the fixed-margin null."""
    import matplotlib.pyplot as plt

    root = Path(cfg["paths"]["output_dir"])
    source = root / "randomization" / "temporal_chain_mixing_diagnostics.csv"
    data = _read_required(source).copy()
    label_map = {
        "short_vs_long_risk_ratio": "≤5 vs >5-year RR",
        "one_vs_gt10_risk_ratio": "1 vs >10-year RR",
        "rate_1_year": "1-year rate",
        "rate_2_years": "2-year rate",
        "rate_3_5_years": "3–5-year rate",
        "rate_6_10_years": "6–10-year rate",
        "rate_gt10_years": ">10-year rate",
    }
    order = list(label_map)
    data = data.set_index("statistic").reindex(order)
    positions = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True, sharey=True)
    axes[0].axvline(1.01, color="#D55E00", linestyle="--", linewidth=0.9, label="Reference = 1.01")
    axes[0].scatter(data["split_rhat"], positions, s=36, color="#0072B2", edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(positions, [label_map[key] for key in order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Split-R̂")
    axes[0].set_title("Between-chain agreement", pad=8)
    axes[0].grid(axis="x", color="#E6E6E6", linewidth=0.6)
    axes[0].legend(frameon=False, loc="lower right")
    bars = axes[1].barh(positions, data["approximate_ess_from_lag1"], color="#56B4E9", edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Approximate effective sample size")
    axes[1].set_title("Serial dependence after thinning", pad=8)
    axes[1].grid(axis="x", color="#E6E6E6", linewidth=0.6)
    axes[1].set_xlim(0, float(data["approximate_ess_from_lag1"].max()) * 1.42)
    for bar, (_, row) in zip(bars, data.iterrows()):
        axes[1].text(
            bar.get_width() + 7,
            bar.get_y() + bar.get_height() / 2,
            f"ESS≈{float(row['approximate_ess_from_lag1']):.0f}; ρ₁={float(row['mean_chain_lag1_autocorrelation']):.2f}",
            va="center",
            fontsize=6.5,
        )
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    _save(fig, cfg, "FigureS2_temporal_chain_diagnostics", [source], records)
    plt.close(fig)


def _figure_s3_prediction_calibration(cfg: Mapping, records: list[dict]) -> None:
    """Supplementary pooled calibration plot for the three forward models."""
    import matplotlib.pyplot as plt

    root = Path(cfg["paths"]["output_dir"])
    source = root / "prediction" / "calibration_data.csv"
    data = _read_required(source).copy()
    required = {"model", "calibration_bin", "n", "mean_predicted", "observed_rate"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Calibration data are missing columns: {sorted(missing)}")
    data["weighted_predicted"] = data["n"] * data["mean_predicted"]
    data["weighted_observed"] = data["n"] * data["observed_rate"]
    pooled = (
        data.groupby(["model", "calibration_bin"], observed=True, as_index=False)
        .agg(n=("n", "sum"), weighted_predicted=("weighted_predicted", "sum"), weighted_observed=("weighted_observed", "sum"))
    )
    pooled["mean_predicted"] = pooled["weighted_predicted"] / pooled["n"]
    pooled["observed_rate"] = pooled["weighted_observed"] / pooled["n"]

    models = ["P0_static", "P1_history", "P2_history_lagged_rain"]
    positive = pooled.loc[(pooled["mean_predicted"] > 0) & (pooled["observed_rate"] > 0)]
    lower = 0.75 * float(min(positive["mean_predicted"].min(), positive["observed_rate"].min()))
    upper = 1.30 * float(max(positive["mean_predicted"].max(), positive["observed_rate"].max()))
    styles = {"P0_static": "-", "P1_history": "-", "P2_history_lagged_rain": "--"}
    fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    ax.plot([lower, upper], [lower, upper], color="#555555", linestyle=":", linewidth=1.0, label="Perfect calibration")
    for model in models:
        part = pooled.loc[pooled["model"].eq(model)].sort_values("mean_predicted")
        ax.plot(
            part["mean_predicted"],
            part["observed_rate"],
            color=MODEL_COLORS[model],
            linestyle=styles[model],
            marker="o",
            markersize=3.8,
            linewidth=1.4,
            label=MODEL_LABELS[model],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Forward-prediction calibration", pad=8)
    ax.grid(color="#E6E6E6", linewidth=0.6, which="both")
    ax.legend(frameon=False, fontsize=7.0, loc="upper left")
    _save(fig, cfg, "FigureS3_prediction_calibration", [source], records)
    plt.close(fig)


def run(cfg: Mapping, quick: bool = False) -> None:
    _style()
    records: list[dict] = []
    _figure3(cfg, records, quick=quick)
    _figure4(cfg, records, quick=quick)
    _figure5(cfg, records, quick=quick)
    _figure6(cfg, records)
    _figure_s1_recurrence(cfg, records)
    _figure_s2_temporal_diagnostics(cfg, records)
    _figure_s3_prediction_calibration(cfg, records)
    manifest = pd.DataFrame(records)
    manifest["quick_mode"] = bool(quick)
    write_csv(manifest, output_path(cfg, "figures", "figure_manifest.csv"))
    write_json(
        {
            "generated_figures": sorted(manifest["figure"].unique().tolist()),
            "formats": ["png_600_dpi", "vector_pdf"],
            "combined_and_separate_panels": True,
            "raster_write_validation": True,
            "source_hashes_recorded": True,
            "quick_mode": bool(quick),
            "submission_warning": "Quick-mode figures are installation diagnostics and must not be used in the manuscript." if quick else "Figures were generated from full analysis outputs.",
        },
        output_path(cfg, "figures", "figure_generation_specification.json"),
    )
    LOGGER.info("Generated Figures 3–6 in PNG and PDF formats")
