"""Export a reproducible sample for investigating brand relationships."""

import argparse
from pathlib import Path

import pandas as pd

from catalogiq import PROCESSED_DATA_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH / "training_cleaned.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_PATH / "training_aryx_sample.csv")
    parser.add_argument("--size", type=int, default=10_000, help="Maximum sample rows (default: 10000)")
    args = parser.parse_args()
    if args.size < 2:
        parser.error("--size must be at least 2 to accommodate both reserved brand buckets")
    # Read strings so identifiers retain leading zeros.
    # Preserve literal cells; use the integrated cleaner's narrow missing policy.
    df = pd.read_csv(args.input, dtype="string", keep_default_na=False)
    populated = df[["ProductBrand", "Brand"]].apply(
        lambda column: ~column.str.strip().str.lower().isin(["", "null"])
    ).all(axis=1)
    eligible = df.loc[populated]
    if eligible.empty:
        raise ValueError("No rows have both ProductBrand and Brand populated.")

    # These are Brand values, not separate columns. Reserve one row per bucket
    # when available; sample the remaining rows without replacement.
    buckets = ["UnItemised brand", "Other Brands"]
    reserved = eligible.loc[eligible["Brand"].isin(buckets)].groupby(
        "Brand", group_keys=False
    ).sample(n=1, random_state=42)
    remaining = eligible.drop(index=reserved.index)
    sample = pd.concat([
        reserved,
        remaining.sample(n=min(args.size, len(eligible)) - len(reserved), random_state=42),
    ]).sample(frac=1, random_state=42)

    output_path = args.output
    sample.to_csv(output_path, index=False)
    print(f"Saved {len(sample):,} rows from {len(eligible):,} eligible rows to {output_path}")
    for bucket in buckets:
        print(f"{bucket}: {sample['Brand'].eq(bucket).sum():,} rows")


if __name__ == "__main__":
    main()
