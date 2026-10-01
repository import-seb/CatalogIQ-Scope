"""Command-line interface for the CatalogIQ cleaner."""
import argparse
import json
from pathlib import Path

from .cleaning import run
from .paths import DATA_PATH, TARGET_PATH, TRAIN_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH, help="Training CSV")
    parser.add_argument("--target", type=Path, default=TARGET_PATH, help="Prediction target CSV")
    parser.add_argument("--mode", choices=("targets", "integrated"), default="targets",
                        help="Legacy target cleaner or combined feature/target decisions")
    parser.add_argument("--exclude-policy", choices=("hold", "keep", "quarantine"), default="hold",
                        help="Integrated mode: handling of the literal source Exclude marker")
    parser.add_argument("--output-dir", type=Path, default=DATA_PATH / "processed" / "target_cleaning",
                        help="New directory for outputs; existing directories are never overwritten")
    args = parser.parse_args()
    try:
        if args.mode == "integrated":
            from .integration import run as integrate
            summary = integrate(args.train, args.target, args.output_dir, args.exclude_policy)
        else:
            if args.exclude_policy != "hold":
                raise ValueError("--exclude-policy applies only to --mode integrated")
            summary = run(args.train, args.target, args.output_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "rows": summary["rows"],
                      "changes_by_rule": summary["changes_by_rule"]}, indent=2))

