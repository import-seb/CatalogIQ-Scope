"""Command-line interface for the CatalogIQ cleaner."""
import argparse
import json
from pathlib import Path

from .cleaning import run
from .paths import DATA_PATH, TARGET_PATH, TRAIN_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH, help="Training CSV")
    parser.add_argument("--target", type=Path, default=TARGET_PATH, help="Target CSV (passed through)")
    parser.add_argument("--output-dir", type=Path, default=DATA_PATH / "processed" / "target_cleaning",
                        help="New directory for outputs; existing directories are never overwritten")
    args = parser.parse_args()
    try:
        summary = run(args.train, args.target, args.output_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "rows": summary["rows"],
                      "changes_by_rule": summary["changes_by_rule"]}, indent=2))

