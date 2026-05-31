import pandas as pd
import json
from collections import Counter
from pathlib import Path
import argparse


# -----------------------
# Component mapping
# -----------------------
# Maps output component_type -> (smiles_col, mol_col).
# Note: cholesterol is emitted as "CH" (not "CHL") so it matches the
# component_type vocabulary used by the pretrained encoder / task schema.
COMPONENTS = {
    "IL": ("IL_SMILES", "IL_molratio"),
    "HL": ("HL_SMILES", "HL_molratio"),
    "PEG": ("PEG_SMILES", "PEG_molratio"),
    "CH": ("CHL_SMILES", "CHL_molratio"),
}

# Dataset name groups all LNPDB samples into a single task (required by the
# JSON->LMDB preprocessing step, which hard-accesses content['dataset_name']).
DATASET_NAME = "lnpdb"
DEFAULT_INPUT_PATH = Path("experiments/data_raw/LNPDB.csv")
DEFAULT_OUTPUT_PATH = Path("experiments/data_json/LNPDB_heart_kidney.json")
DEFAULT_TARGET_LABELS = ("heart", "kidney")


def build_lnpdb_json(df, target_labels=None):
    target_labels = set(target_labels or [])
    keep_all_labels = len(target_labels) == 0

    # Skip rows whose target/value is missing. Optionally keep only the chosen
    # label rows so small target-specific experiments do not process all LNPDB.
    return {
        str(row["Index"]): {
            "components": [
                {
                    "smi": row[smiles_col],
                    "component_type": comp_type,
                    "mol": row[mol_col],
                }
                for comp_type, (smiles_col, mol_col) in COMPONENTS.items()
                if pd.notna(row[smiles_col])
            ],
            "labels": {
                row["Model_target"]: row["Experiment_value"]
            },
            "dataset_name": DATASET_NAME,
        }
        for row in df.to_dict("records")
        if (
            pd.notna(row["Model_target"])
            and pd.notna(row["Experiment_value"])
            and (keep_all_labels or row["Model_target"] in target_labels)
        )
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LNPDB.csv into COMET JSON. Defaults to heart/kidney only."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--target-labels",
        nargs="*",
        default=list(DEFAULT_TARGET_LABELS),
        help="Model_target values to keep. Default: heart kidney.",
    )
    parser.add_argument(
        "--all-labels",
        action="store_true",
        help="Keep all non-missing Model_target rows. Use with --output-path data_json/LNPDB.json for full LNPDB.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    target_labels = None if args.all_labels else args.target_labels

    df = pd.read_csv(input_path)
    data_dict = build_lnpdb_json(df, target_labels=target_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

    label_counts = Counter(
        next(iter(sample["labels"].keys()))
        for sample in data_dict.values()
    )
    label_desc = "all labels" if args.all_labels else ", ".join(args.target_labels)
    print(f"Saved {len(data_dict)} {label_desc} samples to {output_path}")
    print("Label value counts:", label_counts)


if __name__ == "__main__":
    main()
