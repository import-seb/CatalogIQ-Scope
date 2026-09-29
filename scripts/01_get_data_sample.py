"""Export a reproducible sample for investigating brand relationships."""

import pandas as pd

from catalogiq import PROCESSED_DATA_PATH


def main():
    # Read strings so identifiers retain leading zeros.
    df = pd.read_csv(
        PROCESSED_DATA_PATH / "target_cleaning/training_cleaned.csv", dtype="string"
    )
    populated = df[["ProductBrand", "Brand"]].apply(
        lambda column: column.str.strip().fillna("").ne("")
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
        remaining.sample(n=min(10_000, len(eligible)) - len(reserved), random_state=42),
    ]).sample(frac=1, random_state=42)

    output_path = PROCESSED_DATA_PATH / "training_aryx_sample.csv"
    sample.to_csv(output_path, index=False)
    print(f"Saved {len(sample):,} rows from {len(eligible):,} eligible rows to {output_path}")
    for bucket in buckets:
        print(f"{bucket}: {sample['Brand'].eq(bucket).sum():,} rows")


if __name__ == "__main__":
    main()
