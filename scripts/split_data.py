import argparse
from pathlib import Path

from catalogiq.splitting import SplitConfig
from catalogiq.splitting_experiment import load_input, _assign
from catalogiq.split_rules_v3 import refine_rule_groups_v3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--identifiers", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error("Output directory already exists")

    frame, _ = load_input(args.input, args.identifiers)
    config = SplitConfig(grouping_version=2, seed=args.seed)

    groups, _, _ = refine_rule_groups_v3(
        frame, frame["record_id"].to_numpy(), config
    )

    assignments, _ = _assign(frame, groups, config)

    args.output_dir.mkdir(parents=True)
    assignments.to_csv(
        args.output_dir / "rule_assignments.csv", index=False
    )

    print(assignments["split"].value_counts())

if __name__ == "__main__":
    main()