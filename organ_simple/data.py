from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LNPEmbeddingDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        self.component_embeddings = torch.tensor(
            data["component_embeddings"], dtype=torch.float32
        )
        self.percents = torch.tensor(data["percents"], dtype=torch.float32)
        self.component_types = torch.tensor(data["component_types"], dtype=torch.long)
        self.mask = torch.tensor(data["mask"], dtype=torch.bool)
        self.target = torch.tensor(data["target"], dtype=torch.float32)
        self.lnp_ids = data["lnp_ids"].tolist()

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        return {
            "component_embeddings": self.component_embeddings[idx],
            "percents": self.percents[idx],
            "component_types": self.component_types[idx],
            "mask": self.mask[idx],
            "target": self.target[idx],
            "lnp_id": self.lnp_ids[idx],
        }
