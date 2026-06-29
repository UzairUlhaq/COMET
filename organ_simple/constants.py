ORGAN_CLASSES = [
    "lung_epithelium",
    "liver",
    "muscle",
    "spleen",
    "bone_marrow",
    "heart",
    "lung",
    "kidney",
    "ear",
]

COMPONENT_TYPES = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "IL", "HL", "CH", "PEG", "Others"]

COMPONENT_TYPE_TO_ID = {name: i for i, name in enumerate(COMPONENT_TYPES)}
