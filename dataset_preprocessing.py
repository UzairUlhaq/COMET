"""LNPDB organ-classification preprocessing.

Stage 1 (this file): clean the raw LNPDB.csv into a slim CSV that keeps only
the 4 lipid components (SMILES + mol ratio) and the target organ.

  raw LNPDB.csv  ->  LNPDB_clean.csv  ->  (later) COMET JSON / LMDB

Cleaning rules:
  1. Drop rows whose target organ (Model_target) is NaN.
  2. Keep only the 4 lipids (IL / HL / PEG / CH, each SMILES + mol ratio) and
     Model_target; drop everything else.
  3. Do NOT require all 4 lipids -- formulations legitimately lack a component
     (e.g. no HL or no PEG). Missing components are left as NaN here and simply
     omitted when components are built downstream; the row is kept.
"""

import argparse
from pathlib import Path

import pandas as pd


# The 4 lipid components, in (SMILES column, mol-ratio column) pairs.
# CHL = cholesterol; emitted downstream as the "CH" component type.
LIPID_COLUMNS = [
    "IL_SMILES", "IL_molratio",
    "HL_SMILES", "HL_molratio",
    "PEG_SMILES", "PEG_molratio",
    "CHL_SMILES", "CHL_molratio",
]
TARGET_COLUMN = "Model_target"
KEEP_COLUMNS = LIPID_COLUMNS + [TARGET_COLUMN]

DEFAULT_INPUT_PATH = Path("experiments/data_raw/LNPDB.csv")
DEFAULT_OUTPUT_PATH = Path("experiments/data_processed/LNPDB_clean.csv")


def clean_lnpdb(df):
    """Return (clean_df, stats): slim CSV with only the 4 lipids + target organ
    and no NaNs in any kept column."""
    n_raw = len(df)

    # 1. Drop rows with a missing target organ.
    df = df[df[TARGET_COLUMN].notna()]
    n_after_target = len(df)

    # 2. Keep only the 4 lipids + target organ. Missing components stay NaN.
    df = df[KEEP_COLUMNS].copy()

    stats = {
        "raw_rows": n_raw,
        "after_drop_nan_target": n_after_target,
        "rows_missing_a_component": int(df[LIPID_COLUMNS].isna().any(axis=1).sum()),
    }
    return df, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean LNPDB.csv -> slim CSV of 4 lipids + target organ."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)

    df = pd.read_csv(args.input_path, low_memory=False)
    clean_df, stats = clean_lnpdb(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)

    print(f"Saved {len(clean_df)} clean rows to {output_path}")
    print(f"  raw rows                       : {stats['raw_rows']}")
    print(f"  after dropping NaN Model_target: {stats['after_drop_nan_target']}")
    print(f"  rows kept missing >=1 component: {stats['rows_missing_a_component']}")
    print("\nTarget organ counts:")
    print(clean_df[TARGET_COLUMN].value_counts().to_string())


if __name__ == "__main__":
    main()
