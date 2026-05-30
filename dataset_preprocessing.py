import pandas as pd
import json
from collections import Counter



# -----------------------
# Load data
# -----------------------
df = pd.read_csv(
    "/home/uzair/Cellrewire/Code/COMET/experiments/data_raw/LNPDB.csv"
)

# -----------------------
# Component mapping
# -----------------------
COMPONENTS = {
    "IL": ("IL_SMILES", "IL_molratio"),
    "HL": ("HL_SMILES", "HL_molratio"),
    "PEG": ("PEG_SMILES", "PEG_molratio"),
    "CHL": ("CHL_SMILES", "CHL_molratio"),
}

# -----------------------
# Build dataset
# -----------------------
data_dict = {
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
        }
    }
    for row in df.to_dict("records")
}

# -----------------------
# Save JSON
# -----------------------
output_path = "/home/uzair/Cellrewire/Code/COMET/experiments/data_json/LNPDB.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4, ensure_ascii=False)

print(f"Saved {len(data_dict)} samples to {output_path}")

# -----------------------
# Count label values (per assay)
# -----------------------
label_counts = Counter(
    next(iter(sample["labels"].keys()))
    for sample in data_dict.values()
)

print("Label value counts:", label_counts)