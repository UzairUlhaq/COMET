import argparse
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import lmdb


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path):
    with open(repo_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def rel_to(path, root):
    return Path(os.path.relpath(repo_path(path), repo_path(root)))


def normalize_input(payload, dataset_name):
    if isinstance(payload, dict) and "components" in payload:
        payload = {"new_lnp": payload}
    elif isinstance(payload, list):
        payload = {f"new_lnp_{idx}": sample for idx, sample in enumerate(payload)}
    elif not isinstance(payload, dict):
        raise ValueError("Input must be a sample object, list of samples, or id -> sample mapping.")

    normalized = {}
    for lnp_id, sample in payload.items():
        if "components" not in sample:
            raise ValueError(f"Sample {lnp_id} is missing 'components'.")

        components = []
        for component in sample["components"]:
            item = dict(component)
            if "mol" not in item and "percent" in item:
                item["mol"] = item["percent"]
            if "percent" not in item and "mol" in item:
                item["percent"] = item["mol"]
            for required in ("smi", "component_type", "mol", "percent"):
                if required not in item:
                    raise ValueError(f"Sample {lnp_id} component is missing '{required}'.")
            components.append(item)

        normalized[str(lnp_id)] = {
            "components": components,
            "labels": sample.get("labels", {}),
            "dataset_name": sample.get("dataset_name", dataset_name),
            "lnp_id": str(lnp_id),
        }
    return normalized


def write_lmdb(path, payloads):
    path.parent.mkdir(parents=True, exist_ok=True)
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
        map_size=int(100e9),
    )
    txn = env.begin(write=True)
    for idx, payload in enumerate(payloads):
        txn.put(str(idx).encode("ascii"), payload)
    txn.commit()
    env.close()


def build_infer_lmdb(samples, output_dir, dataset_name):
    sys.path.insert(0, str(repo_path("experiments")))
    from preprocess_data_LNPDB import inner_lnp2data, smi2coords_onlymol

    output_dir = repo_path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    unique_smiles = []
    for sample in samples.values():
        for component in sample["components"]:
            if component["smi"] not in unique_smiles:
                unique_smiles.append(component["smi"])

    smi2mol_id = {smi: idx for idx, smi in enumerate(unique_smiles)}
    mol_payloads = []
    for smi in unique_smiles:
        payload = smi2coords_onlymol(smi)
        if payload is None:
            raise RuntimeError(f"Failed to generate conformers for SMILES: {smi}")
        mol_payloads.append(payload)
    write_lmdb(output_dir / "mol.lmdb", mol_payloads)

    infer_payloads = []
    infer_json = []
    for sample in samples.values():
        infer_json.append(sample)
        infer_payloads.append(inner_lnp2data(smi2mol_id, sample, pickle_output=True))
    write_lmdb(dataset_dir / "infer.lmdb", infer_payloads)

    with open(dataset_dir / "infer.json", "w", encoding="utf-8") as handle:
        json.dump(infer_json, handle, indent=2, ensure_ascii=False)

    return output_dir


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch

            return lambda payload: torch.load(io.BytesIO(payload), map_location="cpu")
        return super().find_class(module, name)


def load_pickle_cpu(path):
    with open(path, "rb") as handle:
        return CPUUnpickler(handle).load()


def collect_predictions(result_pickle):
    metrics = load_pickle_cpu(result_pickle)
    lnp_ids = [str(item) for item in metrics.get("lnp_ids", [])]
    predictions = {lnp_id: {"lnp_id": lnp_id} for lnp_id in lnp_ids}

    for key, value in metrics.items():
        if not key.endswith("infer_predict"):
            continue
        task_name = key[: -len("infer_predict")]
        values = value.detach().cpu().view(-1).float().tolist()
        for idx, score in enumerate(values):
            lnp_id = lnp_ids[idx] if idx < len(lnp_ids) else str(idx)
            predictions.setdefault(lnp_id, {"lnp_id": lnp_id})[f"{task_name}_score"] = score

    for row in predictions.values():
        heart = row.get("heart_score")
        kidney = row.get("kidney_score")
        if heart is not None and kidney is not None:
            row["higher_score"] = "heart" if heart >= kidney else "kidney"
            row["score_margin"] = abs(heart - kidney)
    return list(predictions.values())


def default_checkpoint(cfg):
    train = cfg["training"]
    exp_name = (
        f"{cfg['name']}_fold_V{train['fold']}_lnp_{train['loss']}"
        f"-bs{train['batch_size']}-lr{train['lr']}"
        f"-lnpmodparams{train['lnp_encoder_layers']}-{train['lnp_encoder_embed_dim']}"
        f"-{train['lnp_encoder_ffn_embed_dim']}-{train['lnp_encoder_attention_heads']}"
        f"-trainrat{train['train_data_ratio']}-ep{train['epochs']}"
        f"-pat{train['patience']}-metric{train['metric']}-cagrad{train['cagrad_c']}"
        f"-percentnoise{train['percent_noise']}-labelmargin{train['contrast_margin_coeff']}"
        f"-seed{train['seed']}"
    )
    return repo_path(train["save_root"]) / f"save_{exp_name}" / "checkpoint_best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Score new LNP compositions with a heart/kidney model.")
    parser.add_argument("--config", default="configs/lnpdb_heartkidney.json")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--checkpoint", help="Defaults to configured fold checkpoint_best.pt.")
    parser.add_argument("--output-dir", default="experiments/inference_inputs/new_lnp_lmdb")
    parser.add_argument("--results-dir", default="experiments/infer_results/new_lnp_predictions")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train = cfg["training"]
    dataset_name = cfg["data"]["dataset_name"]
    run_dir = repo_path(cfg["run_dir"])
    output_dir = repo_path(args.output_dir)
    results_dir = repo_path(args.results_dir)
    checkpoint = repo_path(args.checkpoint) if args.checkpoint else default_checkpoint(cfg)
    if not checkpoint.exists() and not args.dry_run:
        raise SystemExit(f"Missing checkpoint: {checkpoint}")

    with open(repo_path(args.input_json), "r", encoding="utf-8") as handle:
        samples = normalize_input(json.load(handle), dataset_name)

    if not args.dry_run:
        build_infer_lmdb(samples, output_dir, dataset_name)

    cmd = [
        "python", "../unimol/infer_np.py",
        "--user-dir", "../unimol",
        "./",
        "--task-name", str(rel_to(output_dir, run_dir)),
        "--valid-subset", "infer",
        "--num-workers", str(train["num_workers"]),
        "--ddp-backend=c10d",
        "--batch-size", str(train["eval_batch_size"]),
        "--required-batch-size-multiple", str(train["required_batch_size_multiple"]),
        "--task", "mol_np_finetune",
        "--loss", train["loss"],
        "--arch", "np_unimol",
        "--classification-head-name", str(rel_to(output_dir, run_dir)),
        "--num-classes", "1",
        "--dict-name", train["dict_name"],
        "--conf-size", str(train["conf_size"]),
        "--only-polar", str(train["only_polar"]),
        "--path", str(rel_to(checkpoint, run_dir)),
        "--log-interval", "50",
        "--log-format", "simple",
        "--results-path", str(rel_to(results_dir, run_dir)),
        "--lnp-encoder-layers", str(train["lnp_encoder_layers"]),
        "--lnp-encoder-embed-dim", str(train["lnp_encoder_embed_dim"]),
        "--lnp-encoder-ffn-embed-dim", str(train["lnp_encoder_ffn_embed_dim"]),
        "--lnp-encoder-attention-heads", str(train["lnp_encoder_attention_heads"]),
        "--full-dataset-task-schema-path", str(rel_to(cfg["schema"], run_dir)),
        "--load-full-np-model",
        "--concat-datasets",
    ]
    if train.get("fp16", False):
        cmd.extend(["--fp16", "--fp16-init-scale", "4", "--fp16-scale-window", "256"])

    print("cwd:", run_dir)
    print(" ".join(str(part) for part in cmd))
    if args.dry_run:
        return
    result = subprocess.run(cmd, cwd=run_dir, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    result_pickles = sorted(results_dir.glob("*_infer.out.pkl"))
    if not result_pickles:
        raise SystemExit(f"No inference pickle found in {results_dir}")
    predictions = collect_predictions(result_pickles[-1])
    output_json = results_dir / "predictions.json"
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)

    print(f"Saved predictions to {output_json}")
    for row in predictions:
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
