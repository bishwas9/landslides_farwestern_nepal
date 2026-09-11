from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from landslide_repro import build_history, figures, forward_prediction, publication_audit, rainfall_ame, recurrence, spatial_null, supplement, temporal_null, validate_inputs
from landslide_repro.common import configure_logging, load_config, output_path, sha256_file, software_manifest, write_json


STAGES = {
    "validate": validate_inputs.run,
    "history": build_history.run,
    "recurrence": recurrence.run,
    "temporal": temporal_null.run,
    "rainfall": rainfall_ame.run,
    "spatial": spatial_null.run,
    "prediction": forward_prediction.run,
    "figures": figures.run,
    "publication_audit": publication_audit.run,
    "supplement": supplement.run,
}


def _manifest(cfg, config_path: Path, stage: str, quick: bool, status: str, error: str | None = None) -> dict:
    root = config_path.parent
    code_files = sorted(root.glob("landslide_repro/*.py")) + [root / "run_all.py", config_path]
    inputs = []
    input_dir = Path(cfg["paths"]["input_dir"])
    if input_dir.exists():
        for path in sorted(p for p in input_dir.iterdir() if p.is_file()):
            inputs.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_stage": stage,
        "quick_mode": quick,
        "status": status,
        "error": error,
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "code_files": [{"file": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in code_files if path.exists()],
        "inputs": inputs,
        "software": software_manifest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the landslide recurrence reproducibility workflow")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--quick", action="store_true", help="Use 99 randomizations and two prediction years for an installation test")
    args = parser.parse_args()
    configure_logging()
    cfg, _ = load_config(args.config)
    config_path = Path(args.config).resolve()
    if args.quick:
        full_output = Path(cfg["paths"]["output_dir"])
        cfg["paths"]["output_dir"] = str(full_output.with_name(full_output.name + "_quick"))
    status = "success"
    error = None
    try:
        if args.stage == "all":
            order = ["validate", "history", "recurrence", "rainfall", "prediction", "temporal", "spatial", "publication_audit", "figures", "supplement"]
            for name in order:
                print(f"\n=== {name.upper()} ===", flush=True)
                STAGES[name](cfg, quick=args.quick)
        else:
            STAGES[args.stage](cfg, quick=args.quick)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        write_json(_manifest(cfg, config_path, args.stage, args.quick, status, error), output_path(cfg, "run_manifest.json"))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
