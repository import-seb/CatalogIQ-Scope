"""Command-line interface for the CatalogIQ cleaner."""
import argparse
import csv
import json
from pathlib import Path

from .paths import DATA_PATH, TARGET_PATH, TRAIN_PATH


def segment_baseline_main(argv=None) -> int:
    """Prepare or train a word TF-IDF Segment baseline on explicit shared assignments."""
    parser = argparse.ArgumentParser(description=segment_baseline_main.__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Cleaned training_candidate.csv")
    parser.add_argument("--assignments", type=Path, required=True, help="Team-agreed assignments CSV; never regenerated")
    parser.add_argument("--assignments-sha256", required=True, help="Expected full SHA-256 of that assignment file")
    parser.add_argument("--input-sha256", help="Optional expected full SHA-256 of the cleaned candidate")
    parser.add_argument("--features", choices=("base", "category", "compare"), default="compare")
    parser.add_argument("--train", action="store_true", help="Fit and evaluate validation; otherwise check inputs only")
    parser.add_argument("--config", type=Path, help="Optional BaselineConfig JSON; never changes assignments")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory")
    args = parser.parse_args(argv)
    try:
        from .segment_baseline import BaselineConfig, run_baseline
        values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(values, dict):
            raise ValueError("Baseline configuration must be a JSON object")
        run_baseline(args.input, args.assignments, args.assignments_sha256, args.output_dir,
                     BaselineConfig(**values), expected_input_sha256=args.input_sha256,
                     features=args.features, train=args.train)
    except (ValueError, OSError, RuntimeError, TypeError, Warning) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH, help="Training CSV")
    parser.add_argument("--target", type=Path, default=TARGET_PATH, help="Prediction target CSV")
    parser.add_argument("--mode", choices=("targets", "integrated", "structural"), default="targets",
                        help="Legacy target cleaner, combined decisions, or independent structural audit")
    parser.add_argument("--exclude-policy", choices=("hold", "keep", "quarantine"), default=None,
                        help="Integrated mode: Exclude policy (agreed default: keep; other values are explicit overrides)")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory for outputs; existing directories are never overwritten")
    parser.add_argument("--config", type=Path, help="Structural mode only: JSON policy overrides")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DATA_PATH / "processed" / ("structural_check" if args.mode == "structural" else "target_cleaning")
    try:
        if args.config and args.mode != "structural":
            raise ValueError("--config applies only to --mode structural")
        if args.mode != "integrated" and args.exclude_policy is not None:
            raise ValueError("--exclude-policy applies only to --mode integrated")
        if args.mode == "integrated":
            from .integration import run as integrate
            summary = integrate(args.train, args.target, args.output_dir, args.exclude_policy or "keep")
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


def splitting_main() -> None:
    """Compare label-blind product groupings and Segment-stratified splits."""
    from .paths import PROCESSED_DATA_PATH
    from .splitting import SplitConfig
    from .splitting_experiment import display_results, run_experiment, verify_saved

    parser = argparse.ArgumentParser(description=splitting_main.__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH / "training_cleaned.csv")
    parser.add_argument("--identifiers", type=Path,
                        help="Audited identifier CSV; default: cleaned export's integrated candidate artifact")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_PATH / "split_comparison")
    parser.add_argument("--config", type=Path, help="JSON SplitConfig overrides; resolved settings saved with run")
    parser.add_argument("--seed", type=int, help="Override the configured random seed (default 42)")
    parser.add_argument("--skip-reproduction", action="store_true",
                        help="Explicitly omit the full grouping+allocation repeat check")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--show", type=Path, help="Display a completed run without recomputing")
    mode.add_argument("--verify", type=Path, help="Check artifact hashes, isolation, and saved allocation reproducibility")
    args = parser.parse_args()
    try:
        if args.show:
            display_results(args.show)
        elif args.verify:
            print(json.dumps(verify_saved(args.verify), indent=2))
        else:
            values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            if args.seed is not None:
                values["seed"] = args.seed
            run_experiment(args.input, args.output_dir, SplitConfig.from_dict(values), args.identifiers,
                           verify_reproducibility=not args.skip_reproduction)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


def split_data_main(argv=None) -> int:
    """Create full-record train/validation/test CSVs with rule-v3 grouping."""
    from .split_export import export_splits
    from .split_rules_v3 import RuleRefinementConfig
    from .splitting import SplitConfig

    parser = argparse.ArgumentParser(description=split_data_main.__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Cleaned CSV with source identity and Segment")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for split records and audits")
    parser.add_argument("--identifiers", type=Path, help="Optional source-keyed identifier CSV; otherwise use the export manifest")
    parser.add_argument("--seed", type=int, help="Override the configured random seed (default 42)")
    parser.add_argument("--config", type=Path, help="JSON SplitConfig overrides, including ratios and seed")
    parser.add_argument("--grouping-config", type=Path, help="JSON RuleRefinementConfig overrides for rule-v3")
    args = parser.parse_args(argv)

    def read_config(path):
        try:
            values = json.loads(path.read_text(encoding="utf-8")) if path else {}
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid configuration file {path}: {error}") from error
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        return values

    try:
        values = {"grouping_version": 2, **read_config(args.config)}
        if args.seed is not None:
            values["seed"] = args.seed
        config = SplitConfig.from_dict(values)
        grouping = RuleRefinementConfig.from_dict(read_config(args.grouping_config))
        export_splits(args.input, args.output_dir, config, args.identifiers, grouping,
                      config_paths=[path for path in (args.config, args.grouping_config) if path])
    except (ValueError, OSError, RuntimeError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0

