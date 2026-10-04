"""Export retained training features, approved labels and provenance together."""
import argparse
import json
from pathlib import Path

import pandas as pd

from catalogiq import PROCESSED_DATA_PATH
from catalogiq.cleaning import TARGET_COLUMNS
from catalogiq.feature_decisions import KEY
from catalogiq.features import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROCESSED_DATA_PATH /
                        "integrated_policy_v4_final_20261003")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_PATH / "training_cleaned.csv")
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    sources = [args.run_dir / name for name in ("training_features.csv", "training_candidate.csv")]
    if args.output.resolve() in {p.resolve() for p in sources}:
        raise ValueError("Output must not overwrite an integration source artifact")
    fingerprints = {p.name: sha256(p) for p in sources}
    if any(fingerprints[p.name] != summary["output_sha256"][p.name] for p in sources):
        raise ValueError("Input artifact does not match its completed run manifest")
    features, candidates = [pd.read_csv(p, dtype="string", keep_default_na=False) for p in sources]
    inputs = summary["feature_exports"]["training"]["input_columns"]
    if list(features.columns) != inputs + KEY:
        raise ValueError("Unexpected feature export schema")
    if any(frame.duplicated(KEY).any() for frame in (features, candidates)):
        raise ValueError("Duplicate source identities")
    if not features[KEY].equals(candidates[KEY]):
        raise ValueError("Candidate and feature identities or ordering differ")
    if not features.equals(candidates[inputs + KEY]):
        raise ValueError("Feature values do not match retained candidates")
    if len(features) != summary["rows"]["training"]["candidate"]:
        raise ValueError("Candidate count does not match the run")
    result = candidates[inputs + TARGET_COLUMNS + KEY]
    if any(sha256(p) != fingerprints[p.name] for p in sources):
        raise ValueError("Input changed during export")
    result.to_csv(args.output, index=False)
    report = {"source_run": str(args.run_dir.resolve()), "input_sha256": fingerprints,
              "policy_version": summary["policy_version"], "rows": len(result),
              "feature_columns": inputs, "label_columns": TARGET_COLUMNS,
              "provenance_columns": KEY, "output_sha256": sha256(args.output),
              "values_preserved": True, "labels_imputed": False}
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(result):,} rows and {len(result.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
