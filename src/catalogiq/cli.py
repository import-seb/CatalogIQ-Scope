"""Command-line interface for the CatalogIQ cleaner."""
import argparse
import csv
import json
from pathlib import Path

from .paths import DATA_PATH, TARGET_PATH, TRAIN_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH, help="Training CSV")
    parser.add_argument("--target", type=Path, default=TARGET_PATH, help="Prediction target CSV")
    parser.add_argument("--mode", choices=("targets", "integrated", "structural"), default="targets",
                        help="Legacy target cleaner, combined decisions, or independent structural audit")
    parser.add_argument("--exclude-policy", choices=("hold", "keep", "quarantine"), default="hold",
                        help="Integrated mode: handling of the literal source Exclude marker")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory for outputs; existing directories are never overwritten")
    parser.add_argument("--config", type=Path, help="Structural mode only: JSON policy overrides")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DATA_PATH / "processed" / ("structural_check" if args.mode == "structural" else "target_cleaning")
    try:
        if args.config and args.mode != "structural":
            raise ValueError("--config applies only to --mode structural")
        if args.mode != "integrated" and args.exclude_policy != "hold":
            raise ValueError("--exclude-policy applies only to --mode integrated")
        if args.mode == "integrated":
            from .integration import run as integrate
            summary = integrate(args.train, args.target, args.output_dir, args.exclude_policy)
        elif args.mode == "structural":
            from .structural import StructuralConfig, run as structural_run
            config = (StructuralConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
                      if args.config else StructuralConfig())
            summary = structural_run(args.train, args.target, args.output_dir, config)
        else:
            from .cleaning import run
            summary = run(args.train, args.target, args.output_dir)
    except (ValueError, OSError, csv.Error, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    if args.mode == "structural":
        print(json.dumps({role: data["decisions"] for role, data in summary["datasets"].items()}, indent=2))
        return
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "rows": summary["rows"],
                      "changes_by_rule": summary["changes_by_rule"]}, indent=2))


def structural_main() -> None:
    """Independent structural audit; does not invoke target or feature cleaning."""
    from .structural import StructuralConfig, run as structural_run

    parser = argparse.ArgumentParser(description=structural_main.__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--target", type=Path, default=TARGET_PATH)
    parser.add_argument("--output-dir", type=Path,
                        default=DATA_PATH / "processed" / "structural_check")
    parser.add_argument("--config", type=Path, help="JSON policy overrides; recorded in summary.json")
    args = parser.parse_args()
    try:
        config = (StructuralConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
                  if args.config else StructuralConfig())
        summary = structural_run(args.train, args.target, args.output_dir, config)
    except (ValueError, OSError, csv.Error, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps({role: data["decisions"] for role, data in summary["datasets"].items()}, indent=2))

