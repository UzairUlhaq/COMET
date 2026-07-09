from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LNPEmbeddingDataset(Dataset):
    def __init__(
        self,
        path,
        single_organ_only=False,
        target_threshold=0.0,
        class_indices=None,
        renormalize_target=True,
    ):
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        component_embeddings = data["component_embeddings"]
        percents = data["percents"]
        component_types = data["component_types"]
        mask = data["mask"]
        target = data["target"]
        lnp_ids = data["lnp_ids"]
        original_n = len(target)

        if single_organ_only:
            keep = (target > target_threshold).sum(axis=1) == 1
            component_embeddings = component_embeddings[keep]
            percents = percents[keep]
            component_types = component_types[keep]
            mask = mask[keep]
            target = target[keep]
            lnp_ids = lnp_ids[keep]

        if class_indices is not None:
            class_indices = np.asarray(class_indices, dtype=int)
            target_subset = target[:, class_indices]
            keep = target_subset.sum(axis=1) > target_threshold
            component_embeddings = component_embeddings[keep]
            percents = percents[keep]
            component_types = component_types[keep]
            mask = mask[keep]
            target = target_subset[keep]
            lnp_ids = lnp_ids[keep]
            if renormalize_target:
                target_sum = target.sum(axis=1, keepdims=True)
                target = target / np.clip(target_sum, a_min=1e-12, a_max=None)

        self.component_embeddings = torch.tensor(
            component_embeddings, dtype=torch.float32
        )
        self.percents = torch.tensor(percents, dtype=torch.float32)
        self.component_types = torch.tensor(component_types, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.lnp_ids = lnp_ids.tolist()
        self.num_filtered = int(original_n - len(target))

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
