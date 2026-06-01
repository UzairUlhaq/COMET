import argparse
import copy
import hashlib
import io
import itertools
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_TYPE_DICTIONARY = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "IL", "HL", "CH", "PEG", "Others"]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path

def rel_to(path, root):
    return Path(os.path.relpath(repo_path(path), repo_path(root)))

def load_config(path):
    with open(repo_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)

def save_json(path, payload):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

def get_wandb_cfg(cfg, args):
    configured = cfg.get("wandb", {})
    project = args.wandb_project or configured.get("project") or os.environ.get("WANDB_PROJECT") or "comet-lnpdb"
    entity = args.wandb_entity or configured.get("entity") or os.environ.get("WANDB_ENTITY")
    mode = args.wandb_mode or configured.get("mode") or os.environ.get("WANDB_MODE")
    return {
        "project": project,
        "entity": entity,
        "mode": mode,
        "tags": configured.get("tags", []),
    }

def run(cmd, cwd=None, dry_run=False):
    cwd = repo_path(cwd) if cwd else None
    if cwd:
        print(f"cwd: {cwd}")
    print(" ".join(str(part) for part in cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0)
    return subprocess.run(cmd, cwd=cwd, check=False)

def load_tensorboard_events(log_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise SystemExit("Install tensorboard to read loss/metric event logs.") from exc

    events = []
    for path in sorted(repo_path(log_dir).glob("**/events.out.tfevents.*")):
        split = path.parent.name
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in sorted(acc.Tags().get("scalars", [])):
            for event in acc.Scalars(tag):
                events.append(
                    {
                        "step": event.step,
                        "tag": f"{split}/{tag}",
                        "value": event.value,
                    }
                )
    return events

def load_test_result_metrics(infer_dir):
    metrics = {}
    for path in sorted(repo_path(infer_dir).glob("*_test.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                metrics[f"test/{key}"] = float(value)
    return metrics

def log_run_to_wandb(cfg, fold, args, source="train"):
    prepare_wandb_environment(cfg, fold, args, live=False)

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("Install wandb first, e.g. `pip install wandb`, then rerun with --wandb.") from exc

    paths = run_paths(cfg, fold=fold)
    wandb_cfg = get_wandb_cfg(cfg, args)
    wandb_root = repo_path(cfg.get("outputs", {}).get("wandb_root", "experiments/wandb"))
    run_config = {
        "source": source,
        "fold": fold,
        "experiment_name": paths["exp_name"],
        "dataset_name": cfg.get("dataset_name"),
        "target_labels": cfg.get("target_labels"),
        **{f"model/{key}": value for key, value in cfg.get("model", {}).items()},
        **{f"train/{key}": value for key, value in cfg.get("train", {}).items()},
    }
    init_kwargs = {
        "project": wandb_cfg["project"],
        "name": paths["exp_name"],
        "id": os.environ.get("COMET_WANDB_RUN_ID"),
        "resume": "allow",
        "config": run_config,
        "tags": wandb_cfg["tags"],
        "dir": str(wandb_root),
        "reinit": True,
    }
    if wandb_cfg["entity"]:
        init_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg["mode"]:
        init_kwargs["mode"] = wandb_cfg["mode"]

    with wandb.init(**init_kwargs) as wb_run:
        events_by_step = defaultdict(dict)
        if source != "train":
            for event in load_tensorboard_events(paths["log_dir"]):
                events_by_step[event["step"]][event["tag"]] = event["value"]
        event_count = sum(len(payload) for payload in events_by_step.values())
        for step in sorted(events_by_step):
            wb_run.log(events_by_step[step], step=step)
        result_metrics = load_test_result_metrics(paths["infer_dir"])
        if result_metrics:
            wb_run.log(result_metrics)
        checkpoint = checkpoint_path(cfg, fold=fold)
        if checkpoint.exists():
            artifact_name = safe_wandb_artifact_name(f"{paths['exp_name']}-checkpoint")
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(str(checkpoint))
            wb_run.log_artifact(artifact)
        print(f"Logged {event_count} TensorBoard scalar events and {len(result_metrics)} test metrics to W&B.")


def labels_from_json(json_path):
    with open(repo_path(json_path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    labels = []
    for sample in payload.values() if isinstance(payload, dict) else payload:
        for label in sample.get("labels", {}):
            if label not in labels:
                labels.append(label)
    return labels


def build_experiment_name(cfg, fold=None):
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    fold = train_cfg["fold"] if fold is None else fold
    return (
        f"{cfg['name']}_fold_V{fold}_lnp_{model_cfg['loss']}"
        f"-bs{train_cfg['batch_size']}"
        f"-lr{train_cfg['lr']}"
        f"-lnpmodparams{model_cfg['lnp_encoder_layers']}-{model_cfg['lnp_encoder_embed_dim']}"
        f"-{model_cfg['lnp_encoder_ffn_embed_dim']}-{model_cfg['lnp_encoder_attention_heads']}"
        f"-trainrat{train_cfg['train_data_ratio']}"
        f"-ep{train_cfg['epochs']}"
        f"-pat{train_cfg['patience']}"
        f"-metric{train_cfg['metric']}"
        f"-cagrad{train_cfg['cagrad_c']}"
        f"-percentnoise{train_cfg['percent_noise']}"
        f"-labelmargin{train_cfg['contrast_margin_coeff']}"
        f"-seed{train_cfg['seed']}"
    )


def run_paths(cfg, fold=None):
    exp_name = build_experiment_name(cfg, fold=fold)
    outputs = cfg["outputs"]
    return {
        "exp_name": exp_name,
        "save_dir": repo_path(outputs["save_root"]) / f"save_{exp_name}",
        "tmp_save_dir": repo_path(outputs["tmp_save_root"]) / f"save_{exp_name}",
        "log_dir": repo_path(outputs["log_root"]) / f"log_{exp_name}",
        "infer_dir": repo_path(outputs["infer_root"]) / f"infer_{exp_name}",
    }


def checkpoint_path(cfg, fold=None):
    return run_paths(cfg, fold=fold)["save_dir"] / "checkpoint_best.pt"


def task_dir(cfg, fold=None):
    fold = cfg["train"]["fold"] if fold is None else fold
    return repo_path(cfg["processed_root"]) / f"fold_V{fold}"


def compact_value(value):
    text = str(value).replace("-", "m").replace(".", "p")
    return text.replace("+", "").replace("e", "e")


def sweep_label(overrides):
    return "_".join(f"{key}{compact_value(value)}" for key, value in overrides.items())


def safe_wandb_artifact_name(name, max_len=120):
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    if len(safe) <= max_len:
        return safe
    digest = hashlib.md5(safe.encode("utf-8")).hexdigest()[:10]
    return f"{safe[: max_len - len(digest) - 1]}-{digest}"


def prepare_wandb_environment(cfg, fold, args, live=False):
    paths = run_paths(cfg, fold=fold)
    wandb_cfg = get_wandb_cfg(cfg, args)
    wandb_root = repo_path(cfg.get("outputs", {}).get("wandb_root", "experiments/wandb"))
    wandb_root.mkdir(parents=True, exist_ok=True)
    for subdir in ("data", "cache", "artifacts"):
        (wandb_root / subdir).mkdir(parents=True, exist_ok=True)
    run_id = hashlib.md5(paths["exp_name"].encode("utf-8")).hexdigest()
    env_updates = {
        "COMET_WANDB_LIVE": "1" if live else "0",
        "COMET_WANDB_RUN_NAME": paths["exp_name"],
        "COMET_WANDB_RUN_ID": run_id,
        "WANDB_PROJECT": wandb_cfg["project"],
        "WANDB_DIR": str(wandb_root),
        "WANDB_DATA_DIR": str(wandb_root / "data"),
        "WANDB_CACHE_DIR": str(wandb_root / "cache"),
        "WANDB_ARTIFACT_DIR": str(wandb_root / "artifacts"),
    }
    if wandb_cfg["entity"]:
        env_updates["WANDB_ENTITY"] = wandb_cfg["entity"]
    if wandb_cfg["mode"]:
        env_updates["WANDB_MODE"] = wandb_cfg["mode"]
    os.environ.update(env_updates)
    return env_updates


def build_base_unimol_args(cfg, task_name):
    model_cfg = cfg["model"]
    run_dir = cfg["run_dir"]
    return [
        "--user-dir", "../unimol",
        "./",
        "--task-name", str(task_name),
        "--num-workers", str(cfg["train"]["num_workers"]),
        "--ddp-backend=c10d",
        "--task", "mol_np_finetune",
        "--loss", model_cfg["loss"],
        "--arch", model_cfg["arch"],
        "--classification-head-name", str(task_name),
        "--num-classes", "1",
        "--dict-name", model_cfg["dict_name"],
        "--conf-size", str(model_cfg["conf_size"]),
        "--only-polar", str(model_cfg["only_polar"]),
        "--full-dataset-task-schema-path", str(rel_to(cfg["schema_path"], run_dir)),
        "--lnp-encoder-layers", str(model_cfg["lnp_encoder_layers"]),
        "--lnp-encoder-embed-dim", str(model_cfg["lnp_encoder_embed_dim"]),
        "--lnp-encoder-ffn-embed-dim", str(model_cfg["lnp_encoder_ffn_embed_dim"]),
        "--lnp-encoder-attention-heads", str(model_cfg["lnp_encoder_attention_heads"]),
        "--concat-datasets",
    ]


def build_train_cmd(cfg, fold=None):
    fold = cfg["train"]["fold"] if fold is None else fold
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    paths = run_paths(cfg, fold=fold)
    task_name = rel_to(task_dir(cfg, fold=fold), cfg["run_dir"])
    cmd = [
        "python", "../unimol/train_np.py",
        *build_base_unimol_args(cfg, task_name),
        "--train-subset", "train",
        "--valid-subset", "valid",
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.99)",
        "--adam-eps", "1e-6",
        "--clip-norm", "1.0",
        "--lr-scheduler", "polynomial_decay",
        "--lr", str(train_cfg["lr"]),
        "--warmup-ratio", str(train_cfg["warmup_ratio"]),
        "--max-epoch", str(train_cfg["epochs"]),
        "--batch-size", str(train_cfg["batch_size"]),
        "--required-batch-size-multiple", str(train_cfg["required_batch_size_multiple"]),
        "--pooler-dropout", str(train_cfg["dropout"]),
        "--loss-sample-dropout", "0",
        "--update-freq", "1",
        "--seed", str(train_cfg["seed"]),
        "--log-interval", "100",
        "--log-format", "simple",
        "--validate-interval", "1",
        "--keep-last-epochs", "10",
        "--best-checkpoint-metric", train_cfg["metric"],
        "--patience", str(train_cfg["patience"]),
        "--maximize-best-checkpoint-metric",
        "--save-dir", str(rel_to(paths["save_dir"], cfg["run_dir"])),
        "--tmp-save-dir", str(rel_to(paths["tmp_save_dir"], cfg["run_dir"])),
        "--tensorboard-logdir", str(rel_to(paths["log_dir"], cfg["run_dir"])),
        "--multitask-reg",
        "--cagrad-c", str(train_cfg["cagrad_c"]),
        "--train-data-ratio", str(train_cfg["train_data_ratio"]),
        "--noise-augment-percent",
        "--percent-noise", str(train_cfg["percent_noise"]),
        "--percent-noise-type", train_cfg["percent_noise_type"],
        "--contrast-margin-coeff", str(train_cfg["contrast_margin_coeff"]),
    ]
    if train_cfg.get("fp16", False):
        cmd.extend(["--fp16", "--fp16-init-scale", "4", "--fp16-scale-window", "256"])
    if model_cfg.get("pretrained_mol_encoder"):
        cmd.extend(["--finetune-from-model", str(rel_to(model_cfg["pretrained_mol_encoder"], cfg["run_dir"]))])
    if train_cfg.get("freeze_molecule_encoder", False):
        cmd.append("--freeze-molecule-encoder")
    return cmd


def build_infer_cmd(cfg, fold=None, checkpoint=None, subset="test", results_dir=None, task_override=None):
    fold = cfg["train"]["fold"] if fold is None else fold
    train_cfg = cfg["train"]
    checkpoint = repo_path(checkpoint) if checkpoint else checkpoint_path(cfg, fold=fold)
    paths = run_paths(cfg, fold=fold)
    results_dir = repo_path(results_dir) if results_dir else paths["infer_dir"]
    task_name = task_override or rel_to(task_dir(cfg, fold=fold), cfg["run_dir"])
    cmd = [
        "python", "../unimol/infer_np.py",
        *build_base_unimol_args(cfg, task_name),
        "--valid-subset", subset,
        "--batch-size", str(train_cfg["eval_batch_size"]),
        "--required-batch-size-multiple", str(train_cfg["required_batch_size_multiple"]),
        "--path", str(rel_to(checkpoint, cfg["run_dir"])),
        "--log-interval", "50",
        "--log-format", "simple",
        "--results-path", str(rel_to(results_dir, cfg["run_dir"])),
        "--load-full-np-model",
    ]
    if train_cfg.get("fp16", False):
        cmd.extend(["--fp16", "--fp16-init-scale", "4", "--fp16-scale-window", "256"])
    return cmd


def command_prepare_json(args):
    import pandas as pd

    cfg = load_config(args.config)
    raw_csv = repo_path(args.raw_csv or cfg["raw_csv"])
    output = repo_path(args.output or cfg["json_path"])
    columns = cfg["csv_columns"]
    target_labels = [] if args.all_labels else list(args.labels or cfg["target_labels"])
    keep_all = len(target_labels) == 0

    df = pd.read_csv(raw_csv, low_memory=False)
    records = {}
    for row in df.to_dict("records"):
        label = row.get(columns["target"])
        value = row.get(columns["value"])
        if pd.isna(label) or pd.isna(value):
            continue
        if not keep_all and label not in target_labels:
            continue
        components = []
        for component_type, (smiles_col, mol_col) in columns["components"].items():
            if pd.notna(row.get(smiles_col)):
                components.append(
                    {
                        "smi": row[smiles_col],
                        "component_type": component_type,
                        "mol": row[mol_col],
                    }
                )
        records[str(row[columns["id"]])] = {
            "components": components,
            "labels": {label: value},
            "dataset_name": cfg["dataset_name"],
        }

    save_json(output, records)
    label_counts = Counter()
    for sample in records.values():
        label_counts.update(sample["labels"].keys())
    print(f"Saved {len(records)} samples to {output}")
    print("Label counts:", dict(label_counts))


def command_make_schema(args):
    cfg = load_config(args.config)
    labels = list(args.labels or cfg["target_labels"])
    if args.from_json:
        labels = labels_from_json(cfg["json_path"])
    schema = {
        "datasets": {
            cfg["dataset_name"]: {
                "labels": {label: 1.0 for label in labels},
                "np_props": {},
            }
        },
        "np_component_types": {
            "component_type": {
                "dictionary": COMPONENT_TYPE_DICTIONARY,
                "embed_dim": 128,
            }
        },
    }
    save_json(args.output or cfg["schema_path"], schema)
    print("Schema labels:", labels)
    print("Saved schema to:", repo_path(args.output or cfg["schema_path"]))


def command_preprocess_lmdb(args):
    cfg = load_config(args.config)
    sys.path.insert(0, str(repo_path("experiments")))
    from preprocess_data_LNPDB import write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling

    split_cfg = cfg["split"]
    train_ids, valid_ids, test_ids = write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=str(repo_path(args.input or cfg["json_path"])),
        outpath=str(repo_path(args.output or cfg["processed_root"])),
        nthreads=args.nthreads or split_cfg["nthreads"],
        kfold_valid=args.kfold_valid or split_cfg["kfold_valid"],
        test_ratio=args.test_ratio if args.test_ratio is not None else split_cfg["test_ratio"],
        top_heldout_ratio=0,
        bottom_heldout_ratio=0,
        shuffle=True,
        debug=not args.no_debug,
    )
    print("Done. Processed data:", repo_path(args.output or cfg["processed_root"]))
    print("Train sizes:", {fold: len(ids) for fold, ids in train_ids.items()})
    print("Valid sizes:", {fold: len(ids) for fold, ids in valid_ids.items()})
    print("Test size:", len(test_ids))


def count_samples(payload):
    rows = payload.values() if isinstance(payload, dict) else payload
    labels = Counter()
    component_types = Counter()
    component_count = Counter()
    for sample in rows:
        labels.update(sample.get("labels", {}).keys())
        components = sample.get("components", [])
        component_count[len(components)] += 1
        component_types.update(component.get("component_type") for component in components)
    return {
        "samples": len(payload),
        "labels": dict(labels),
        "component_types": dict(sorted(component_types.items())),
        "components_per_sample": dict(sorted(component_count.items())),
    }


def command_inspect(args):
    cfg = load_config(args.config)
    paths = [repo_path(cfg["json_path"])]
    fold_root = task_dir(cfg, fold=args.fold) / cfg["dataset_name"]
    paths.extend(fold_root / f"{split}.json" for split in ("train", "valid", "test"))
    for path in paths:
        print(f"\n{path}")
        if not path.exists():
            print("missing")
            continue
        with open(path, "r", encoding="utf-8") as handle:
            print(json.dumps(count_samples(json.load(handle)), indent=2))


def run_training_workflow(cfg, fold, args):
    paths = run_paths(cfg, fold=fold)
    last_checkpoint = paths["save_dir"] / "checkpoint_last.pt"
    if last_checkpoint.exists() and not args.clean and not args.resume:
        raise SystemExit(
            "Found an existing checkpoint_last.pt in this experiment directory:\n"
            f"  {last_checkpoint}\n\n"
            "UniMol will try to resume that checkpoint automatically. If you changed model/data settings, "
            "that can cause architecture mismatch errors.\n\n"
            "Use one of:\n"
            "  --clean   start this experiment from the pretrained molecular encoder and delete old run outputs\n"
            "  --resume  intentionally continue from checkpoint_last.pt"
        )
    if args.clean and not args.dry_run:
        for key in ("save_dir", "tmp_save_dir", "log_dir"):
            if paths[key].exists():
                shutil.rmtree(paths[key])
    if args.wandb and not args.dry_run:
        prepare_wandb_environment(cfg, fold, args, live=True)
    result = run(build_train_cmd(cfg, fold=fold), cwd=cfg["run_dir"], dry_run=args.dry_run)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    if args.no_infer or not cfg["train"].get("run_inference_after_train", True):
        if args.wandb and not args.dry_run:
            log_run_to_wandb(cfg, fold, args, source="train")
        return
    if not checkpoint_path(cfg, fold=fold).exists() and not args.dry_run:
        raise SystemExit(f"Missing checkpoint: {checkpoint_path(cfg, fold=fold)}")
    result = run(build_infer_cmd(cfg, fold=fold), cwd=cfg["run_dir"], dry_run=args.dry_run)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    if args.wandb and not args.dry_run:
        log_run_to_wandb(cfg, fold, args, source="train")


def command_train(args):
    cfg = load_config(args.config)
    fold = args.fold if args.fold is not None else cfg["train"]["fold"]
    run_training_workflow(cfg, fold, args)


def command_infer(args):
    cfg = load_config(args.config)
    fold = args.fold if args.fold is not None else cfg["train"]["fold"]
    ckpt = repo_path(args.checkpoint) if args.checkpoint else checkpoint_path(cfg, fold=fold)
    if not ckpt.exists() and not args.dry_run:
        raise SystemExit(f"Missing checkpoint: {ckpt}")
    result = run(
        build_infer_cmd(cfg, fold=fold, checkpoint=ckpt, subset=args.subset),
        cwd=cfg["run_dir"],
        dry_run=args.dry_run,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    if args.wandb and not args.dry_run:
        log_run_to_wandb(cfg, fold, args, source="infer")


def command_wandb_log(args):
    cfg = load_config(args.config)
    fold = args.fold if args.fold is not None else cfg["train"]["fold"]
    log_run_to_wandb(cfg, fold, args, source="existing")


def sweep_values(args, cfg):
    train_cfg = cfg["train"]
    return {
        "lr": args.lr if args.lr is not None else [train_cfg["lr"]],
        "batch_size": args.batch_size if args.batch_size is not None else [train_cfg["batch_size"]],
        "cagrad_c": args.cagrad_c if args.cagrad_c is not None else [train_cfg["cagrad_c"]],
        "percent_noise": args.percent_noise if args.percent_noise is not None else [train_cfg["percent_noise"]],
        "contrast_margin_coeff": args.contrast_margin_coeff if args.contrast_margin_coeff is not None else [train_cfg["contrast_margin_coeff"]],
        "seed": args.seed if args.seed is not None else [train_cfg["seed"]],
        "dropout": args.dropout if args.dropout is not None else [train_cfg["dropout"]],
    }


def iter_sweep_runs(args, cfg):
    values = sweep_values(args, cfg)
    keys = list(values)
    for raw_combo in itertools.product(*(values[key] for key in keys)):
        combo = dict(zip(keys, raw_combo))
        changed = {key: value for key, value in combo.items() if value != cfg["train"].get(key)}
        label = sweep_label(changed or {"base": 1})
        run_cfg = copy.deepcopy(cfg)
        run_cfg["name"] = f"{cfg['name']}_{args.name}_{label}"
        for key, value in combo.items():
            run_cfg["train"][key] = value
        if "batch_size" in combo and args.eval_batch_size_follows_batch:
            run_cfg["train"]["eval_batch_size"] = combo["batch_size"]
        yield label, changed, run_cfg


def command_sweep(args):
    cfg = load_config(args.config)
    folds = args.folds if args.folds is not None else [cfg["train"]["fold"]]
    sweep_root = repo_path(args.output_dir or f"pipeline/sweeps/{args.name}")
    config_dir = sweep_root / "configs"
    manifest_path = repo_path(args.manifest) if args.manifest else sweep_root / "manifest.json"
    manifest = {
        "sweep_name": args.name,
        "base_config": str(repo_path(args.config)),
        "runs": [],
    }

    run_index = 0
    for label, overrides, run_cfg in iter_sweep_runs(args, cfg):
        for fold in folds:
            if args.max_runs is not None and run_index >= args.max_runs:
                save_json(manifest_path, manifest)
                print(f"Reached --max-runs={args.max_runs}. Manifest: {manifest_path}")
                return

            paths = run_paths(run_cfg, fold=fold)
            config_path = config_dir / f"{paths['exp_name']}.json"
            save_json(config_path, run_cfg)
            record = {
                "index": run_index,
                "fold": fold,
                "label": label,
                "overrides": overrides,
                "experiment_name": paths["exp_name"],
                "config_path": str(config_path),
                "save_dir": str(paths["save_dir"]),
                "log_dir": str(paths["log_dir"]),
                "infer_dir": str(paths["infer_dir"]),
                "checkpoint": str(checkpoint_path(run_cfg, fold=fold)),
                "status": "pending",
            }
            manifest["runs"].append(record)
            save_json(manifest_path, manifest)

            if args.skip_existing and checkpoint_path(run_cfg, fold=fold).exists():
                print(f"Skipping existing checkpoint: {checkpoint_path(run_cfg, fold=fold)}")
                record["status"] = "skipped_existing"
                record["test_metrics"] = load_test_result_metrics(paths["infer_dir"])
                save_json(manifest_path, manifest)
                run_index += 1
                continue

            print(f"\n=== Sweep run {run_index}: fold={fold} {label} ===")
            try:
                run_training_workflow(run_cfg, fold, args)
                record["status"] = "dry_run" if args.dry_run else "completed"
                record["test_metrics"] = {} if args.dry_run else load_test_result_metrics(paths["infer_dir"])
            except SystemExit as exc:
                record["status"] = "failed"
                record["returncode"] = exc.code
                save_json(manifest_path, manifest)
                if not args.keep_going:
                    raise
            save_json(manifest_path, manifest)
            run_index += 1

    print(f"Sweep complete. Manifest: {manifest_path}")


def command_summarize_sweep(args):
    with open(repo_path(args.manifest), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = manifest.get("runs", [])
    if args.status:
        rows = [row for row in rows if row.get("status") == args.status]
    if args.metric:
        rows = [row for row in rows if args.metric in row.get("test_metrics", {})]
        rows = sorted(
            rows,
            key=lambda row: row["test_metrics"][args.metric],
            reverse=not args.minimize,
        )
    print(f"Sweep: {manifest.get('sweep_name')} | runs: {len(rows)}")
    for row in rows[: args.top]:
        metrics = row.get("test_metrics", {})
        metric_text = ""
        if args.metric and args.metric in metrics:
            metric_text = f" {args.metric}={metrics[args.metric]:.6g}"
        print(
            f"#{row.get('index')} {row.get('status')} fold={row.get('fold')} "
            f"{row.get('label')}{metric_text}"
        )
        if args.show_path:
            print(f"  config: {row.get('config_path')}")
            print(f"  checkpoint: {row.get('checkpoint')}")


def normalize_predict_input(payload, dataset_name):
    if isinstance(payload, dict) and "components" in payload:
        payload = {"new_lnp": payload}
    elif isinstance(payload, list):
        payload = {f"new_lnp_{idx}": sample for idx, sample in enumerate(payload)}
    normalized = {}
    for lnp_id, sample in payload.items():
        components = []
        for component in sample["components"]:
            item = dict(component)
            if "mol" not in item and "percent" in item:
                item["mol"] = item["percent"]
            if "percent" not in item and "mol" in item:
                item["percent"] = item["mol"]
            components.append(item)
        normalized[str(lnp_id)] = {
            "components": components,
            "labels": sample.get("labels", {}),
            "dataset_name": sample.get("dataset_name", dataset_name),
            "lnp_id": str(lnp_id),
        }
    return normalized


def write_lmdb(path, payloads):
    import lmdb

    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    env = lmdb.open(str(path), subdir=False, readonly=False, lock=False, readahead=False, meminit=False, max_readers=1, map_size=int(100e9))
    txn = env.begin(write=True)
    for idx, payload in enumerate(payloads):
        txn.put(str(idx).encode("ascii"), payload)
    txn.commit()
    env.close()


def build_predict_lmdb(cfg, input_json, output_dir):
    sys.path.insert(0, str(repo_path("experiments")))
    from preprocess_data_LNPDB import inner_lnp2data, smi2coords_onlymol

    with open(repo_path(input_json), "r", encoding="utf-8") as handle:
        samples = normalize_predict_input(json.load(handle), cfg["dataset_name"])

    output_dir = repo_path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset_dir = output_dir / cfg["dataset_name"]
    dataset_dir.mkdir(parents=True, exist_ok=True)

    smiles = []
    for sample in samples.values():
        for component in sample["components"]:
            if component["smi"] not in smiles:
                smiles.append(component["smi"])
    smi2mol_id = {smi: idx for idx, smi in enumerate(smiles)}

    mol_payloads = []
    for smi in smiles:
        payload = smi2coords_onlymol(smi)
        if payload is None:
            raise RuntimeError(f"Failed conformer generation for {smi}")
        mol_payloads.append(payload)
    write_lmdb(output_dir / "mol.lmdb", mol_payloads)

    infer_payloads = []
    infer_json = []
    for sample in samples.values():
        infer_json.append(sample)
        infer_payloads.append(inner_lnp2data(smi2mol_id, sample, pickle_output=True))
    write_lmdb(dataset_dir / "infer.lmdb", infer_payloads)
    save_json(dataset_dir / "infer.json", infer_json)
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

def collect_prediction_rows(result_pickle):
    metrics = load_pickle_cpu(result_pickle)
    lnp_ids = [str(item) for item in metrics.get("lnp_ids", [])]
    if not lnp_ids:
        for key, value in metrics.items():
            if key.endswith("infer_predict"):
                lnp_ids = [str(idx) for idx in range(value.detach().cpu().view(-1).numel())]
                break
    rows = {lnp_id: {"lnp_id": lnp_id} for lnp_id in lnp_ids}
    for key, value in metrics.items():
        if not key.endswith("infer_predict"):
            continue
        task = key[: -len("infer_predict")].rstrip("_")
        scores = value.detach().cpu().view(-1).float().tolist()
        for idx, score in enumerate(scores):
            lnp_id = lnp_ids[idx] if idx < len(lnp_ids) else str(idx)
            rows.setdefault(lnp_id, {"lnp_id": lnp_id})[f"{task}_score"] = score
    return list(rows.values())


def command_predict(args):
    cfg = load_config(args.config)
    output_dir = repo_path(args.output_dir or cfg["outputs"]["new_lnp_lmdb"])
    results_dir = repo_path(args.results_dir or cfg["outputs"]["new_lnp_results"])
    ckpt = repo_path(args.checkpoint) if args.checkpoint else checkpoint_path(cfg, fold=args.fold)
    if not ckpt.exists() and not args.dry_run:
        raise SystemExit(f"Missing checkpoint: {ckpt}")
    if not args.dry_run:
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        build_predict_lmdb(cfg, args.input_json, output_dir)
    task_name = rel_to(output_dir, cfg["run_dir"])
    cmd = build_infer_cmd(
        cfg,
        fold=args.fold,
        checkpoint=ckpt,
        subset="infer",
        results_dir=results_dir,
        task_override=task_name,
    )
    result = run(cmd, cwd=cfg["run_dir"], dry_run=args.dry_run)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    if args.dry_run:
        return
    pickles = sorted(results_dir.glob("*_infer.out.pkl"))
    if not pickles:
        raise SystemExit(f"No inference pickle found under {results_dir}")
    rows = collect_prediction_rows(pickles[-1])
    for row in rows:
        scores = {key: value for key, value in row.items() if key.endswith("_score")}
        if scores:
            best = max(scores.items(), key=lambda item: item[1])
            row["raw_score_argmax"] = best[0].replace("_score", "")
    output_json = results_dir / "predictions.json"
    save_json(output_json, rows)
    print(f"Saved predictions: {output_json}")
    for row in rows:
        print(json.dumps(row, indent=2))


def command_summarize_results(args):
    cfg = load_config(args.config)
    root = repo_path(args.results_root or cfg["outputs"]["infer_root"])
    files = sorted(root.glob("infer_*/*_test.json"))
    if args.contains:
        files = [path for path in files if args.contains in str(path)]
    if not files:
        raise SystemExit(f"No result JSON files found under {root}")
    grouped = defaultdict(list)
    for path in files:
        print(path)
        with open(path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                grouped[key].append(float(value))
    print("\nSummary:")
    for key in sorted(grouped):
        values = grouped[key]
        row = {"n": len(values), "mean": mean(values), "min": min(values), "max": max(values)}
        if len(values) > 1:
            row["std"] = stdev(values)
        print(f"{key}: {json.dumps(row)}")


def command_summarize_logs(args):
    cfg = load_config(args.config)
    log_root = repo_path(args.log_root or cfg["outputs"]["log_root"])
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise SystemExit("Install tensorboard to summarize event logs.") from exc
    files = sorted(log_root.glob("**/events.out.tfevents.*"))
    if args.contains:
        files = [path for path in files if args.contains in str(path)]
    if not files:
        raise SystemExit(f"No TensorBoard event files found under {log_root}")
    for path in files:
        print(f"\n{path}")
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in sorted(acc.Tags().get("scalars", [])):
            values = [event.value for event in acc.Scalars(tag)]
            if values:
                print(f"  {tag}: n={len(values)} first={values[0]:.6g} last={values[-1]:.6g} min={min(values):.6g} max={max(values):.6g}")


def build_parser():
    parser = argparse.ArgumentParser(description="Config-driven COMET LNP pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-json")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--raw-csv")
    p.add_argument("--output")
    p.add_argument("--labels", nargs="*")
    p.add_argument("--all-labels", action="store_true")
    p.set_defaults(func=command_prepare_json)

    p = sub.add_parser("make-schema")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--output")
    p.add_argument("--labels", nargs="*")
    p.add_argument("--from-json", action="store_true")
    p.set_defaults(func=command_make_schema)

    p = sub.add_parser("preprocess-lmdb")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--nthreads", type=int)
    p.add_argument("--kfold-valid", type=int)
    p.add_argument("--test-ratio", type=float)
    p.add_argument("--no-debug", action="store_true")
    p.set_defaults(func=command_preprocess_lmdb)

    p = sub.add_parser("inspect")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--fold", type=int, default=0)
    p.set_defaults(func=command_inspect)

    p = sub.add_parser("train")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--fold", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-infer", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log TensorBoard losses/metrics and test metrics to Weights & Biases after the run.")
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    p.set_defaults(func=command_train)

    p = sub.add_parser("sweep")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--name", default="sweep")
    p.add_argument("--folds", nargs="*", type=int)
    p.add_argument("--lr", nargs="*", type=float)
    p.add_argument("--batch-size", nargs="*", type=int)
    p.add_argument("--cagrad-c", nargs="*", type=float)
    p.add_argument("--percent-noise", nargs="*", type=float)
    p.add_argument("--contrast-margin-coeff", nargs="*", type=float)
    p.add_argument("--seed", nargs="*", type=int)
    p.add_argument("--dropout", nargs="*", type=float)
    p.add_argument("--max-runs", type=int)
    p.add_argument("--output-dir")
    p.add_argument("--manifest")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-infer", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--eval-batch-size-follows-batch", action="store_true", default=True)
    p.add_argument("--wandb", action="store_true", help="Log each sweep run to Weights & Biases after it finishes.")
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    p.set_defaults(func=command_sweep)

    p = sub.add_parser("summarize-sweep")
    p.add_argument("--manifest", required=True)
    p.add_argument("--metric", default="test/heart_test_spearmanr_coeff")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--status", default="completed")
    p.add_argument("--minimize", action="store_true")
    p.add_argument("--show-path", action="store_true")
    p.set_defaults(func=command_summarize_sweep)

    p = sub.add_parser("infer")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--fold", type=int)
    p.add_argument("--checkpoint")
    p.add_argument("--subset", default="test", choices=["train", "valid", "test", "infer"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log TensorBoard losses/metrics and test metrics to Weights & Biases after inference.")
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    p.set_defaults(func=command_infer)

    p = sub.add_parser("wandb-log")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--fold", type=int)
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    p.set_defaults(func=command_wandb_log)

    p = sub.add_parser("predict")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--input-json", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint")
    p.add_argument("--output-dir")
    p.add_argument("--results-dir")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=command_predict)

    p = sub.add_parser("summarize-results")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--results-root")
    p.add_argument("--contains", default="")
    p.set_defaults(func=command_summarize_results)

    p = sub.add_parser("summarize-logs")
    p.add_argument("--config", default="pipeline/configs/lnpdb_heartkidney.json")
    p.add_argument("--log-root")
    p.add_argument("--contains", default="")
    p.set_defaults(func=command_summarize_logs)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
