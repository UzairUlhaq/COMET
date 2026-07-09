"""Predict the 9-organ distribution for one raw LNP with organ_simple.

This script accepts a single formulation as SMILES + component ratios, builds a
temporary one-sample LMDB, encodes each component with the frozen UniMol encoder,
then feeds those component embeddings into the trained SimpleLNPTransformer.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from pyprojroot import here as project_root
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"

sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(REPO_ROOT))

from preprocess_data_LNPDB import inner_lnp2data, inner_smi2coords_onlymol  # noqa: E402

from organ_simple.constants import (  # noqa: E402
    COMPONENT_TYPE_TO_ID,
    COMPONENT_TYPES,
    ORGAN_CLASSES,
)
from organ_simple.model import SimpleLNPTransformer  # noqa: E402

os.chdir(EXPERIMENTS)
sys.path.insert(0, str(project_root()))
importlib.import_module("unimol")

from unimol.core import checkpoint_utils, options, tasks, utils  # noqa: E402

DATASET_NAME = "lnpdb"
HEAD_NAME = "organ"
NUM_CLASSES = len(ORGAN_CLASSES)
SCHEMA = "task_schemas/lnpdb_organ_schema.json"
PRETRAINED = "../ckp/mol_pre_no_h_220816.pt"
WORK_DIR = "processed_data_dirs/lnpdb_organ_simple_single/fold_V0"

COMPONENTS = [
    ("IL", "il", "il_mol", "IL_SMILES", "IL_molratio"),
    ("HL", "hl", "hl_mol", "HL_SMILES", "HL_molratio"),
    ("PEG", "peg", "peg_mol", "PEG_SMILES", "PEG_molratio"),
    ("CH", "chl", "chl_mol", "CHL_SMILES", "CHL_molratio"),
]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_components(components):
    if not components:
        raise ValueError("provide at least one component with a SMILES string")
    total = sum(max(float(component["mol"]), 0.0) for component in components)
    if total <= 0:
        raise ValueError("component mol ratios must sum to a positive value")
    normalized = []
    for component in components:
        mol = max(float(component["mol"]), 0.0) / total
        normalized.append({
            "smi": component["smi"],
            "component_type": component["component_type"],
            "mol": mol,
            "percent": mol,
        })
    return normalized


def components_from_sample(obj):
    components = []
    for component in obj["components"]:
        mol = float(component.get("mol", component.get("percent", 0.0)))
        components.append({
            "smi": component["smi"],
            "component_type": component["component_type"],
            "mol": mol,
        })
    true_label = (obj.get("labels") or {}).get(HEAD_NAME)
    return normalize_components(components), true_label


def build_components(args):
    if args.sample is not None:
        path = REPO_ROOT / args.dataset_json
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if args.sample not in data:
            raise KeyError(f"sample {args.sample!r} not found in {path}")
        print(f"[predict_single] using dataset sample {args.sample}")
        return components_from_sample(data[args.sample])

    if args.input:
        with repo_path(args.input).open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if "components" not in obj and len(obj) == 1:
            only_value = next(iter(obj.values()))
            if isinstance(only_value, dict) and "components" in only_value:
                obj = only_value
        if "components" in obj:
            return components_from_sample(obj)

        components = []
        for component_type, _, _, smi_key, mol_key in COMPONENTS:
            smi = obj.get(smi_key)
            if smi and str(smi).strip():
                components.append({
                    "smi": smi,
                    "component_type": component_type,
                    "mol": float(obj.get(mol_key) or 0.0),
                })
        return normalize_components(components), None

    components = []
    for component_type, smi_attr, mol_attr, _, _ in COMPONENTS:
        smi = getattr(args, smi_attr)
        if smi and smi.strip():
            components.append({
                "smi": smi,
                "component_type": component_type,
                "mol": float(getattr(args, mol_attr) or 0.0),
            })
    if components:
        return normalize_components(components), None

    path = REPO_ROOT / args.dataset_json
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    print("[predict_single] no input given; using dataset sample 0")
    return components_from_sample(data["0"])


def validate_components(components):
    valid_types = {"IL", "HL", "PEG", "CH", "Others"}
    for index, component in enumerate(components):
        component_type = component["component_type"]
        smi = component["smi"]
        if component_type not in valid_types:
            raise ValueError(
                f"component {index} has unknown component_type={component_type!r}; "
                f"expected one of {sorted(valid_types)}"
            )
        if smi in {
            "IL_SMILES",
            "HL_SMILES",
            "PEG_SMILES",
            "CH_SMILES",
            "CHL_SMILES",
        }:
            raise ValueError(
                f"component {component_type} still has placeholder SMILES {smi!r}; "
                "replace it with the real SMILES string"
            )
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            preview = smi[:120] + ("..." if len(smi) > 120 else "")
            raise ValueError(
                f"RDKit could not parse the {component_type} SMILES at component "
                f"{index}: {preview!r}. Check JSON escaping for backslashes and "
                "make sure this is a valid SMILES string."
            )


def write_lmdb(path, entries):
    path = Path(path)
    if path.exists():
        path.unlink()
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(10e9),
    )
    with env.begin(write=True) as txn:
        for i, entry in enumerate(entries):
            txn.put(str(i).encode("ascii"), entry)
    env.close()


def build_single_lmdb(components, work_dir):
    work_dir = Path(work_dir)
    split_dir = work_dir / DATASET_NAME
    split_dir.mkdir(parents=True, exist_ok=True)

    validate_components(components)
    unique_smis = []
    for component in components:
        if component["smi"] not in unique_smis:
            unique_smis.append(component["smi"])

    print(f"[predict_single] generating conformers for {len(unique_smis)} molecule(s)")
    mol_entries = [inner_smi2coords_onlymol(smi) for smi in unique_smis]
    write_lmdb(work_dir / "mol.lmdb", mol_entries)
    smi_to_mol_id = {smi: i for i, smi in enumerate(unique_smis)}

    sample = {
        "components": components,
        "labels": {HEAD_NAME: [1.0 / NUM_CLASSES] * NUM_CLASSES},
        "dataset_name": DATASET_NAME,
        "lnp_id": "single_0",
    }
    write_lmdb(split_dir / "test.lmdb", [inner_lnp2data(smi_to_mol_id, sample)])
    return work_dir


def build_unimol_args(task_name, batch_size, num_workers, cpu):
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    input_args = [
        "./",
        "--task-name", task_name,
        "--valid-subset", "test",
        "--num-workers", str(num_workers),
        "--ddp-backend=c10d",
        "--batch-size", str(batch_size),
        "--required-batch-size-multiple", "1",
        "--task", "mol_np_finetune",
        "--loss", "np_finetune_soft_cross_entropy",
        "--arch", "np_unimol",
        "--classification-head-name", HEAD_NAME,
        "--num-classes", str(NUM_CLASSES),
        "--dict-name", "dict.txt",
        "--conf-size", "11",
        "--only-polar", "0",
        "--path", PRETRAINED,
        "--results-path", "./infer_results/organ_simple_single_precompute",
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


def component_type_ids(sample, mask):
    raw = sample["net_input"].get("component_types")
    if raw is None:
        return torch.zeros_like(mask, dtype=torch.long)
    if torch.is_tensor(raw):
        ids = raw.long().clamp_min(0)
        return ids.masked_fill(~mask, 0)

    encoded = []
    for row in raw:
        encoded.append([COMPONENT_TYPE_TO_ID.get(str(value), 0) for value in row])
    return torch.tensor(encoded, dtype=torch.long, device=mask.device)


@torch.no_grad()
def encode_single_lnp(task_name, batch_size, num_workers, device, cpu):
    unimol_args = build_unimol_args(task_name, batch_size, num_workers, cpu)
    task = tasks.setup_task(unimol_args)
    model = task.build_model(unimol_args)
    checkpoint = checkpoint_utils.load_checkpoint_to_cpu(PRETRAINED)
    model.mol_model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    task.load_concat_dataset("test", combine=False, epoch=1)
    dataset = task.dataset("test")
    itr = task.get_batch_iterator(
        dataset=dataset,
        batch_size=unimol_args.batch_size,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=unimol_args.required_batch_size_multiple,
        seed=unimol_args.seed,
        num_workers=unimol_args.num_workers,
        data_buffer_size=unimol_args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    for sample in itr:
        sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
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

        return {
            "component_embeddings": component_rep.float(),
            "percents": percents.float(),
            "component_types": component_type_ids(sample, mask),
            "mask": mask,
        }

    raise RuntimeError("no sample was loaded from the one-sample LMDB")


def load_simple_model(checkpoint_path, component_embedding_dim, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    model = SimpleLNPTransformer(
        component_embedding_dim=component_embedding_dim,
        num_component_types=len(COMPONENT_TYPES),
        num_classes=len(ORGAN_CLASSES),
        embed_dim=saved_args.get("embed_dim", 256),
        layers=saved_args.get("layers", 2),
        heads=saved_args.get("heads", 4),
        dropout=saved_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def predict(simple_model, batch):
    logits, cls_rep = simple_model(
        batch["component_embeddings"],
        batch["percents"],
        batch["component_types"],
        batch["mask"],
    )
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return probs, cls_rep[0].cpu().numpy()


def write_prediction(path, components, probs, true_label=None):
    ranked = sorted(zip(ORGAN_CLASSES, probs.tolist()), key=lambda item: item[1], reverse=True)
    payload = {
        "formulation": components,
        "distribution": {organ: float(prob) for organ, prob in zip(ORGAN_CLASSES, probs)},
        "top_organ": ranked[0][0],
        "top_prob": float(ranked[0][1]),
    }
    if true_label is not None:
        payload["true_distribution"] = {
            organ: float(value) for organ, value in zip(ORGAN_CLASSES, true_label)
        }
        payload["true_top_organ"] = ORGAN_CLASSES[int(np.asarray(true_label).argmax())]

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def print_prediction(probs, true_label=None):
    true = np.asarray(true_label, dtype=float) if true_label is not None else None
    ranked = sorted(enumerate(probs.tolist()), key=lambda item: item[1], reverse=True)
    print("\n=== organ_simple prediction ===")
    print(f"  {'organ':<16} {'prob':>8}" + (f" {'true':>8}" if true is not None else ""))
    for idx, prob in ranked:
        organ = ORGAN_CLASSES[idx]
        bar = "#" * int(round(prob * 40))
        suffix = f" {true[idx]:8.3f}" if true is not None else ""
        print(f"  {organ:<16} {prob:8.4f}{suffix}  {bar}")
    print(f"\nTop prediction: {ORGAN_CLASSES[ranked[0][0]]} ({ranked[0][1]:.2%})")
    if true is not None:
        true_top = int(true.argmax())
        print(f"True top organ: {ORGAN_CLASSES[true_top]} ({true[true_top]:.2%})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict the 9-organ distribution for one LNP using organ_simple.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", help="JSON file: COMET sample shape or flat recipe")
    for _, smi_attr, mol_attr, _, _ in COMPONENTS:
        parser.add_argument(f"--{smi_attr}", help=f"{smi_attr.upper()} SMILES")
        parser.add_argument(
            f"--{mol_attr.replace('_', '-')}",
            dest=mol_attr,
            type=float,
            help=f"{smi_attr.upper()} mol ratio",
        )
    parser.add_argument("--sample", help="sample key from --dataset-json")
    parser.add_argument("--dataset-json", default="experiments/data_json/LNPDB_organ.json")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--work-dir", default=WORK_DIR)
    parser.add_argument("--output-json", default="organ_simple_single_prediction.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--keep-files", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint_path = Path(
        repo_path(args.checkpoint)
        if args.checkpoint
        else REPO_ROOT / f"organ_simple/runs/fold_V{args.fold}/best.pt"
    )

    components, true_label = build_components(args)
    print("[predict_single] normalized formulation:")
    for component in components:
        smi = component["smi"]
        print(
            f"  {component['component_type']:<4} "
            f"percent={component['percent']:.6g}  "
            f"{smi[:70]}{'...' if len(smi) > 70 else ''}"
        )

    work_dir = EXPERIMENTS / args.work_dir
    try:
        build_single_lmdb(components, work_dir)
        batch = encode_single_lnp(
            args.work_dir,
            args.batch_size,
            args.num_workers,
            device,
            args.cpu,
        )
        simple_model = load_simple_model(
            checkpoint_path,
            batch["component_embeddings"].shape[-1],
            device,
        )
        probs, _ = predict(simple_model, batch)
        print_prediction(probs, true_label)
        out_path = write_prediction(args.output_json, components, probs, true_label)
        print(f"saved prediction -> {out_path}")
    finally:
        if not args.keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
