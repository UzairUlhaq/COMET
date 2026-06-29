import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pyprojroot import here as project_root

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
os.chdir(EXPERIMENTS)

sys.path.insert(0, str(project_root()))
importlib.import_module("unimol")

from unimol.core import checkpoint_utils, options, tasks, utils  # noqa: E402

from organ_simple.constants import COMPONENT_TYPE_TO_ID  # noqa: E402

TASK_ROOT = "processed_data_dirs/lnpdb_organ_gen"
SCHEMA = "task_schemas/lnpdb_organ_schema.json"
HEAD_NAME = "organ"
PRETRAINED = "../ckp/mol_pre_no_h_220816.pt"


def build_unimol_args(fold, batch_size, num_workers, cpu):
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    input_args = [
        "./",
        "--task-name", f"{TASK_ROOT}/fold_V{fold}",
        "--valid-subset", "valid",
        "--num-workers", str(num_workers),
        "--ddp-backend=c10d",
        "--batch-size", str(batch_size),
        "--required-batch-size-multiple", "1",
        "--task", "mol_np_finetune",
        "--loss", "np_finetune_soft_cross_entropy",
        "--arch", "np_unimol",
        "--classification-head-name", HEAD_NAME,
        "--num-classes", "9",
        "--dict-name", "dict.txt",
        "--conf-size", "11",
        "--only-polar", "0",
        "--path", PRETRAINED,
        "--results-path", "./infer_results/organ_simple_precompute",
        "--full-dataset-task-schema-path", SCHEMA,
        "--concat-datasets",
        "--lnp-encoder-layers", "1",
        "--lnp-encoder-embed-dim", "256",
        "--lnp-encoder-ffn-embed-dim", "256",
        "--lnp-encoder-attention-heads", "8",
    ]
    if cpu:
        input_args.append("--cpu")
    return options.parse_args_and_arch(parser, input_args=input_args)


def target_tensor(sample):
    target = sample["target"]["finetune_target"]
    if isinstance(target, dict):
        return target[HEAD_NAME] if HEAD_NAME in target else next(iter(target.values()))
    return target


def component_type_ids(sample, mask):
    raw = sample["net_input"].get("component_types")
    if raw is None:
        return torch.zeros_like(mask, dtype=torch.long)
    if torch.is_tensor(raw):
        ids = raw.long()
        ids = ids.clamp_min(0)
        ids = ids.masked_fill(~mask, 0)
        return ids

    encoded = []
    for row in raw:
        encoded.append([COMPONENT_TYPE_TO_ID.get(str(x), 0) for x in row])
    return torch.tensor(encoded, dtype=torch.long, device=mask.device)


@torch.no_grad()
def encode_split(task, args, model, split, device):
    task.load_concat_dataset(split, combine=False, epoch=1)
    dataset = task.dataset(split)
    itr = task.get_batch_iterator(
        dataset=dataset,
        batch_size=args.batch_size,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    all_embeddings, all_percents, all_types, all_masks, all_targets, all_ids = [], [], [], [], [], []
    for sample in itr:
        sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
        if len(sample) == 0:
            continue

        encoder_rep, _ = model.mol_model(
            **sample["mol_features"],
            features_only=True,
            output_rep_only=True,
        )
        mol_rep = encoder_rep[:, 0, :]
        mol_batch_ids = sample["mol_batch_ids"].clone()
        original_shape = mol_batch_ids.shape
        flat_ids = mol_batch_ids.flatten()

        pad_rep = torch.zeros_like(mol_rep[0]).unsqueeze(0)
        mol_rep = torch.cat([mol_rep, pad_rep], dim=0)
        pad_idx = mol_rep.shape[0] - 1
        mask = flat_ids.ne(-1).view(original_shape)
        flat_ids[flat_ids == -1] = pad_idx

        flat_component_rep = torch.index_select(mol_rep, 0, flat_ids)
        component_rep = torch.unflatten(flat_component_rep, 0, original_shape)

        percents = sample["net_input"]["percents"].float()
        if percents.dim() == 3 and percents.size(-1) == 1:
            percents = percents.squeeze(-1)

        all_embeddings.append(component_rep.float().cpu().numpy())
        all_percents.append(percents.float().cpu().numpy())
        all_types.append(component_type_ids(sample, mask).cpu().numpy())
        all_masks.append(mask.cpu().numpy())
        all_targets.append(target_tensor(sample).float().cpu().numpy())
        all_ids.extend(list(sample["net_input"].get("lnp_ids", [])))

    return {
        "component_embeddings": np.concatenate(all_embeddings),
        "percents": np.concatenate(all_percents),
        "component_types": np.concatenate(all_types),
        "mask": np.concatenate(all_masks),
        "target": np.concatenate(all_targets),
        "lnp_ids": np.asarray(all_ids, dtype=object),
    }


def save_npz(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--splits", nargs="*", default=["train", "valid", "test"])
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "organ_simple/cache"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    for fold in args.folds:
        unimol_args = build_unimol_args(fold, args.batch_size, args.num_workers, args.cpu)
        task = tasks.setup_task(unimol_args)
        model = task.build_model(unimol_args)
        checkpoint = checkpoint_utils.load_checkpoint_to_cpu(PRETRAINED)
        model.mol_model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device)
        model.eval()

        for split in args.splits:
            payload = encode_split(task, unimol_args, model, split, device)
            save_npz(Path(args.output_dir) / f"fold_V{fold}" / f"{split}.npz", payload)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
