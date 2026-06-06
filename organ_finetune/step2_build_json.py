"""LNPDB organ-classification preprocessing -- Stage 2.

  LNPDB_clean.csv  ->  LNPDB_organ.json  (+ lnpdb_organ_schema.json)

Reads the slim CSV from stage 1 (../dataset_preprocessing.py: 4 lipids + the
target organ) and emits COMET JSON in the same shape as the in_house datasets:
each sample is {components: [...], labels: {...}, dataset_name: ...}.

Task framing -- SOFTMAX DISTRIBUTION over organs
-------------------------------------------------
Goal: given a formulation, predict a probability distribution over the 9 organs
(a single 9-way softmax head, probabilities sum to 1). NOT independent per-organ
sigmoids -- this is mutually-exclusive multiclass with SOFT targets.

The same physical formulation is measured across several organs (biodistribution
panels), and the only reliable per-formulation identity in this CSV is the
composition itself (the 4 SMILES + 4 mol ratios) -- LNP_ID / Formulation_ID are
unique per row. So we GROUP rows by composition and turn the set of organs a
formulation was measured in into a target DISTRIBUTION:

    labels = {"organ": [p_0, ..., p_8]}   # order == ORGAN_CLASSES, sums to 1

  * single-organ formulation       -> one-hot (e.g. liver = 1.0)
  * k-organ biodistribution panel  -> uniform 1/k over the measured organs

One sample per formulation (no duplicated inputs -> no train/val leakage). This
pairs with a single classification head (--num-classes 9) trained with a
SOFT-TARGET cross-entropy: loss = -sum(target * log_softmax(logits)). At
inference, softmax(logits) IS the predicted organ distribution.

NOTE on the uniform split: without Experiment_value (dropped in stage 1) a panel
becomes uniform over its organs. To weight the distribution by actual delivery
(e.g. softmax of per-organ values), re-add Experiment_value in stage 1 -- left
as a follow-up.
"""

import json
from collections import Counter
from pathlib import Path
import argparse

import pandas as pd


# Maps output component_type -> (smiles_col, mol_col). CHL (cholesterol) is
# emitted as "CH" to match the component_type vocabulary in the task schema.
COMPONENTS = {
    "IL": ("IL_SMILES", "IL_molratio"),
    "HL": ("HL_SMILES", "HL_molratio"),
    "PEG": ("PEG_SMILES", "PEG_molratio"),
    "CH": ("CHL_SMILES", "CHL_molratio"),
}

# The 9 organ classes. INDEX IN THIS LIST == position in the softmax/target
# vector -- frozen so JSON, schema, and inference all agree. Kept "as-is"
# (lung_epithelium and lung separate). in_vitro / multiorgan / whole_body are
# not specific organs and are excluded.
ORGAN_CLASSES = [
    "lung_epithelium", "liver", "muscle", "spleen", "bone_marrow",
    "heart", "lung", "kidney", "ear",
]

# Single softmax task name; the head outputs len(ORGAN_CLASSES) logits.
TASK_NAME = "organ"

# Columns whose combination uniquely identifies a formulation (the model input).
FORMULATION_KEY_COLS = [
    "IL_SMILES", "HL_SMILES", "PEG_SMILES", "CHL_SMILES",
    "IL_molratio", "HL_molratio", "PEG_molratio", "CHL_molratio",
]

DATASET_NAME = "lnpdb"
DEFAULT_INPUT_PATH = Path("experiments/data_processed/LNPDB_clean.csv")
DEFAULT_OUTPUT_PATH = Path("experiments/data_json/LNPDB_organ.json")
DEFAULT_SCHEMA_PATH = Path("experiments/task_schemas/lnpdb_organ_schema.json")

COMPONENT_TYPE_DICTIONARY = [
    "[PAD]", "[CLS]", "[SEP]", "[UNK]", "IL", "HL", "CH", "PEG", "Others",
]


def _formulation_key(row):
    """Stable per-formulation key; NaNs (absent components) stringify
    consistently so the same recipe groups together."""
    return "|".join(str(row[col]) for col in FORMULATION_KEY_COLS)


def _build_components(row):
    """Component dicts for one formulation, skipping components whose SMILES is
    absent (so 3-component LNPs without HL or PEG are handled gracefully)."""
    components = []
    for comp_type, (smiles_col, mol_col) in COMPONENTS.items():
        if pd.notna(row[smiles_col]):
            components.append({
                "smi": row[smiles_col],
                "component_type": comp_type,
                "mol": float(row[mol_col]) if pd.notna(row[mol_col]) else 0.0,
            })
    return components


def _organ_distribution(measured_organs):
    """Uniform probability distribution over the measured organs, as a vector
    aligned to ORGAN_CLASSES (sums to 1)."""
    k = len(measured_organs)
    return [1.0 / k if organ in measured_organs else 0.0
            for organ in ORGAN_CLASSES]


def build_organ_json(df):
    """Group rows by composition and build the soft-label JSON dict.

    Returns (data_dict, stats).
    """
    # Keep only rows whose target is one of our organ classes.
    df = df[df["Model_target"].isin(ORGAN_CLASSES)].copy()
    n_organ_rows = len(df)

    df["_fkey"] = df.apply(_formulation_key, axis=1)

    data_dict = {}
    n_multi = 0
    for idx, (_, group) in enumerate(df.groupby("_fkey", sort=False)):
        first = group.iloc[0]
        components = _build_components(first)
        if not components:
            continue

        measured = set(group["Model_target"])
        if len(measured) > 1:
            n_multi += 1

        data_dict[str(idx)] = {
            "components": components,
            "labels": {TASK_NAME: _organ_distribution(measured)},
            "dataset_name": DATASET_NAME,
        }

    stats = {
        "organ_rows": n_organ_rows,
        "unique_formulations": int(df["_fkey"].nunique()),
        "samples_written": len(data_dict),
        "multi_organ_samples": n_multi,
    }
    return data_dict, stats


def write_task_schema(path):
    """Single softmax head named `organ`; output width (num_classes) is set at
    train time to len(ORGAN_CLASSES)."""
    schema = {
        "datasets": {
            DATASET_NAME: {
                "labels": {TASK_NAME: 1.0},
                "np_props": {},
            }
        },
        "np_component_types": {
            "component_type": {
                "dictionary": COMPONENT_TYPE_DICTIONARY,
                "embed_dim": 128,
            }
        },
        # Reference only (not read by COMET): the organ order of the softmax.
        "_organ_classes": ORGAN_CLASSES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: LNPDB_clean.csv -> softmax-distribution organ JSON."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--no-schema", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)

    df = pd.read_csv(args.input_path, low_memory=False)
    data_dict, stats = build_organ_json(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

    if not args.no_schema:
        write_task_schema(Path(args.schema_path))

    print(f"Saved {stats['samples_written']} samples to {output_path}")
    print(f"  organ rows (excl in_vitro/multiorgan/whole_body): {stats['organ_rows']}")
    print(f"  unique formulations  : {stats['unique_formulations']}")
    print(f"  multi-organ (soft)   : {stats['multi_organ_samples']}")
    if not args.no_schema:
        print(f"  schema -> {args.schema_path}  (num_classes = {len(ORGAN_CLASSES)})")

    # Aggregate probability mass per organ (expected #samples per organ).
    mass = Counter()
    for s in data_dict.values():
        for organ, p in zip(ORGAN_CLASSES, s["labels"][TASK_NAME]):
            mass[organ] += p
    print("\nTotal probability mass per organ (sum of target probs):")
    for organ in ORGAN_CLASSES:
        print(f"  {organ:<16} {mass.get(organ, 0.0):.1f}")


if __name__ == "__main__":
    main()
