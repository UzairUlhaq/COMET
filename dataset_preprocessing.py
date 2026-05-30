import pandas as pd
import json
from collections import Counter

# Load data
df = pd.read_csv(
    "/home/uzair/Cellrewire/Code/COMET/experiments/data_raw/LNPDB.csv"
)

# Define component mappings once
COMPONENTS = {
    "IL": ("IL_SMILES", "IL_molratio"),
    "HL": ("HL_SMILES", "HL_molratio"),
    "PEG": ("PEG_SMILES", "PEG_molratio"),
    "CHL": ("CHL_SMILES", "CHL_molratio"),
}

# Build dictionary
data_dict = {
    row["Index"]: {
        "components": [
            {
                "smi": row[smiles_col],
                "component_type": component_type,
                "mol": row[molratio_col],
            }
            for component_type, (smiles_col, molratio_col) in COMPONENTS.items()
            if pd.notna(row[smiles_col])
        ],
        "labels": {
            "labels": row["Model_target"]
        },
    }
    for row in df.to_dict("records")
}


# %%

label_counts = Counter(
    sample["labels"]["labels"]
    for sample in data_dict.values()
)

print(label_counts)


import json

with open("/home/uzair/Cellrewire/Code/COMET/experiments/data_json/LNPDB.json", "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4, ensure_ascii=False)