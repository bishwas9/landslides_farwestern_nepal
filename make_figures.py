from __future__ import annotations

import argparse
from pathlib import Path

from landslide_repro import figures
from landslide_repro.common import configure_logging, load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate combined and separate manuscript figure panels from completed output tables"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Read output_quick and create diagnostic figures; never use these in the manuscript",
    )
    args = parser.parse_args()

    configure_logging()
    cfg, _ = load_config(args.config)
    if args.quick:
        full_output = Path(cfg["paths"]["output_dir"])
        cfg["paths"]["output_dir"] = str(full_output.with_name(full_output.name + "_quick"))

    figures.run(cfg, quick=args.quick)
    figure_dir = Path(cfg["paths"]["output_dir"]) / "figures"
    print(f"Figures written to: {figure_dir.resolve()}")
    print(f"Source manifest: {(figure_dir / 'figure_manifest.csv').resolve()}")
    for path in sorted(figure_dir.glob("Figure*.png")):
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
