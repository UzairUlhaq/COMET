"""Paper-style organ fine-tune sweep across folds and hyperparameters.

Runs train_np.py + infer_np.py for each fold/grid combination, records a manifest,
and copies the best checkpoint into the sweep's `best_model/` folder.

Example:
    python organ_finetune/sweep_train.py --wandb --folds 0 1 2 3 4

    python organ_finetune/sweep_train.py \
      --name lr_bs_search --folds 0 1 2 3 4 \
      --lr 3e-5 1e-4 3e-4 --batch-size 4 8 --dropout 0.1 0.2 \
      --wandb --embedding-subsets train,valid,test
"""

import argparse
import itertools
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"

TASK_ROOT = "processed_data_dirs/lnpdb_organ_gen"
SCHEMA = "task_schemas/lnpdb_organ_schema.json"
HEAD_NAME = "organ"
NUM_CLASSES = 9
LOSS = "np_finetune_soft_cross_entropy"
PRETRAINED = "../ckp/mol_pre_no_h_220816.pt"
DICT_NAME = "dict.txt"

CONF_SIZE = 11
ONLY_POLAR = 0
WARMUP = 0.06
METRIC = "valid_top1_acc"
TEST_METRIC = "test_top1_accuracy"
ORGAN_CLASSES = [
    "lung_epithelium", "liver", "muscle", "spleen", "bone_marrow",
    "heart", "lung", "kidney", "ear",
]

DEFAULT_LNP_LAYERS = 8
DEFAULT_LNP_EMBED = 256
DEFAULT_LNP_FFN = 256
DEFAULT_LNP_HEADS = 8


def slug_value(value):
    return str(value).replace(".", "p").replace("-", "m")


def exp_name(args, fold, combo):
    return (
        f"{args.name}_fold_V{fold}"
        f"_lr{slug_value(combo['lr'])}"
        f"_bs{combo['batch_size']}"
        f"_drop{slug_value(combo['dropout'])}"
        f"_lnp{combo['lnp_layers']}-{combo['lnp_embed']}-{combo['lnp_ffn']}-{combo['lnp_heads']}"
        f"_seed{combo['seed']}"
    )


def rel_to_experiments(path):
    path = Path(path)
    if path.is_absolute():
        try:
            return str(path.relative_to(EXPERIMENTS))
        except ValueError:
            return str(path)
    return str(path)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_cmd(cmd, env, dry_run):
    print("\n[sweep] cwd=", EXPERIMENTS)
    print("[sweep]", " ".join(str(part) for part in cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=EXPERIMENTS, env=env).returncode


def build_env(args, name):
    env = dict(os.environ)
    if not args.wandb:
        return env

    env["COMET_WANDB_LIVE"] = "1"
    env["WANDB_PROJECT"] = args.wandb_project
    env["COMET_WANDB_RUN_NAME"] = name
    env["COMET_WANDB_RUN_ID"] = name
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_offline:
        env["WANDB_MODE"] = "offline"

    if not args.no_wandb_embedding_images:
        env["COMET_WANDB_EMBEDDING_IMAGES"] = "1"
        env["COMET_WANDB_EMBEDDING_INTERVAL"] = str(args.embedding_interval)
        env["COMET_WANDB_EMBEDDING_SUBSETS"] = args.embedding_subsets
        env["COMET_WANDB_EMBEDDING_COLOR_BY"] = args.embedding_color_by
        env["COMET_WANDB_EMBEDDING_CLASSES"] = ",".join(ORGAN_CLASSES)
        env["COMET_WANDB_EMBEDDING_OUT_DIR"] = f"infer_results/{args.name}/embedding_images"
        if args.embedding_single_organ_only:
            env["COMET_WANDB_EMBEDDING_SINGLE_ORGAN_ONLY"] = "1"
    return env


def build_train_cmd(args, fold, combo, save_dir, tmp_save_dir, log_dir):
    task_name = f"{TASK_ROOT}/fold_V{fold}"
    cmd = [
        sys.executable, "../unimol/train_np.py", "./",
        "--task-name", task_name,
        "--user-dir", "../unimol",
        "--train-subset", "train",
        "--valid-subset", "valid",
        "--conf-size", str(CONF_SIZE),
        "--num-workers", str(args.num_workers),
        "--ddp-backend=c10d",
        "--dict-name", DICT_NAME,
        "--task", "mol_np_finetune",
        "--loss", LOSS,
        "--arch", "np_unimol",
        "--classification-head-name", HEAD_NAME,
        "--num-classes", str(NUM_CLASSES),
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.99)",
        "--adam-eps", "1e-6",
        "--clip-norm", "1.0",
        "--lr-scheduler", "polynomial_decay",
        "--lr", str(combo["lr"]),
        "--warmup-ratio", str(args.warmup),
        "--max-epoch", str(args.epochs),
        "--batch-size", str(combo["batch_size"]),
        "--required-batch-size-multiple", "1",
        "--pooler-dropout", str(combo["dropout"]),
        "--seed", str(combo["seed"]),
        "--log-interval", "100",
        "--log-format", "simple",
        "--validate-interval", "1",
        "--keep-last-epochs", "5",
        "--finetune-from-model", PRETRAINED,
        "--best-checkpoint-metric", METRIC,
        "--patience", str(args.patience),
        "--maximize-best-checkpoint-metric",
        "--save-dir", rel_to_experiments(save_dir),
        "--tmp-save-dir", rel_to_experiments(tmp_save_dir),
        "--only-polar", str(ONLY_POLAR),
        "--tensorboard-logdir", rel_to_experiments(log_dir),
        "--full-dataset-task-schema-path", SCHEMA,
        "--freeze-molecule-encoder",
        "--epoch-to-freeze-molecule-encoder", "1000000",
        "--concat-datasets",
        "--lnp-encoder-layers", str(combo["lnp_layers"]),
        "--lnp-encoder-embed-dim", str(combo["lnp_embed"]),
        "--lnp-encoder-ffn-embed-dim", str(combo["lnp_ffn"]),
        "--lnp-encoder-attention-heads", str(combo["lnp_heads"]),
    ]
    if args.fp16:
        cmd.extend(["--fp16", "--fp16-init-scale", "4", "--fp16-scale-window", "256"])
    return cmd


def build_infer_cmd(args, fold, combo, save_dir, results_dir):
    task_name = f"{TASK_ROOT}/fold_V{fold}"
    return [
        sys.executable, "../unimol/infer_np.py",
        "--user-dir", "../unimol", "./",
        "--task-name", task_name,
        "--valid-subset", "test",
        "--num-workers", str(args.num_workers),
        "--ddp-backend=c10d",
        "--batch-size", str(args.eval_batch_size),
        "--required-batch-size-multiple", "1",
        "--task", "mol_np_finetune",
        "--loss", LOSS,
        "--arch", "np_unimol",
        "--classification-head-name", HEAD_NAME,
        "--num-classes", str(NUM_CLASSES),
        "--dict-name", DICT_NAME,
        "--conf-size", str(CONF_SIZE),
        "--only-polar", str(ONLY_POLAR),
        "--path", rel_to_experiments(save_dir / "checkpoint_best.pt"),
        "--log-interval", "50",
        "--log-format", "simple",
        "--results-path", rel_to_experiments(results_dir),
        "--lnp-encoder-layers", str(combo["lnp_layers"]),
        "--lnp-encoder-embed-dim", str(combo["lnp_embed"]),
        "--lnp-encoder-ffn-embed-dim", str(combo["lnp_ffn"]),
        "--lnp-encoder-attention-heads", str(combo["lnp_heads"]),
        "--full-dataset-task-schema-path", SCHEMA,
        "--load-full-np-model",
        "--concat-datasets",
        "--output-cls-rep",
        *(["--fp16", "--fp16-init-scale", "4", "--fp16-scale-window", "256"] if args.fp16 else []),
    ]


def load_test_metrics(results_dir, save_dir):
    stem = save_dir.name
    json_path = results_dir / f"{stem}_test.json"
    pkl_path = results_dir / f"{stem}_test.out.pkl"

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if pkl_path.exists():
        with pkl_path.open("rb") as handle:
            payload = pickle.load(handle)
        metrics = {}
        if "prob" in payload and "target" in payload:
            prob = payload["prob"]
            target = payload["target"]
            correct = (prob.argmax(-1) == target.argmax(-1)).sum().item()
            total = prob.shape[0]
            metrics[TEST_METRIC] = correct / total if total else None
        return metrics

    return {}


def maybe_copy_best(sweep_root, record, metric, minimize):
    if metric not in record.get("test_metrics", {}):
        return

    best_path = sweep_root / "best_model.json"
    current = None
    if best_path.exists():
        with best_path.open("r", encoding="utf-8") as handle:
            current = json.load(handle)

    score = record["test_metrics"][metric]
    if score is None:
        return
    current_score = None if current is None else current.get("score")
    is_better = (
        current_score is None
        or (score < current_score if minimize else score > current_score)
    )
    if not is_better:
        return

    best_dir = sweep_root / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(record["checkpoint"])
    if checkpoint.exists():
        shutil.copy2(checkpoint, best_dir / "checkpoint_best.pt")
    config = Path(record["config_path"])
    if config.exists():
        shutil.copy2(config, best_dir / "config.json")

    summary = {
        "metric": metric,
        "score": score,
        "record": record,
        "copied_checkpoint": str(best_dir / "checkpoint_best.pt"),
        "copied_config": str(best_dir / "config.json"),
    }
    save_json(best_path, summary)
    print(f"[sweep] new best {metric}={score:.6g}: {checkpoint}")


def combos(args):
    keys = ["lr", "batch_size", "dropout", "seed", "lnp_layers", "lnp_embed", "lnp_ffn", "lnp_heads"]
    values = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "seed": args.seed,
        "lnp_layers": args.lnp_layers,
        "lnp_embed": args.lnp_embed,
        "lnp_ffn": args.lnp_ffn,
        "lnp_heads": args.lnp_heads,
    }
    for raw in itertools.product(*(values[key] for key in keys)):
        yield dict(zip(keys, raw))


def clean_run_dirs(save_dir, tmp_save_dir, log_dir, results_dir):
    for path in (save_dir, tmp_save_dir, log_dir, results_dir):
        if path.exists():
            shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="organ_sweep")
    parser.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--lr", nargs="*", type=float, default=[1e-4])
    parser.add_argument("--batch-size", nargs="*", type=int, default=[8])
    parser.add_argument("--dropout", nargs="*", type=float, default=[0.1])
    parser.add_argument("--seed", nargs="*", type=int, default=[1])
    parser.add_argument("--lnp-layers", nargs="*", type=int, default=[DEFAULT_LNP_LAYERS])
    parser.add_argument("--lnp-embed", nargs="*", type=int, default=[DEFAULT_LNP_EMBED])
    parser.add_argument("--lnp-ffn", nargs="*", type=int, default=[DEFAULT_LNP_FFN])
    parser.add_argument("--lnp-heads", nargs="*", type=int, default=[DEFAULT_LNP_HEADS])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup", type=float, default=WARMUP)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--best-metric", default=TEST_METRIC)
    parser.add_argument("--minimize-best-metric", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="comet-lnpdb-organ")
    parser.add_argument("--wandb-run-name-prefix", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--no-wandb-embedding-images", action="store_true")
    parser.add_argument("--embedding-interval", type=int, default=1)
    parser.add_argument("--embedding-subsets", default="valid")
    parser.add_argument("--embedding-color-by", choices=["target", "pred"], default="target")
    parser.add_argument("--embedding-single-organ-only", action="store_true")
    args = parser.parse_args()

    if args.embedding_interval < 1:
        parser.error("--embedding-interval must be >= 1")

    sweep_root = REPO_ROOT / (args.output_dir or f"organ_finetune/sweeps/{args.name}")
    config_dir = sweep_root / "configs"
    save_root = EXPERIMENTS / "save_lnpdb_organ_sweeps" / args.name
    tmp_root = EXPERIMENTS / "tmp_save_lnpdb_organ_sweeps" / args.name
    log_root = EXPERIMENTS / "logs" / "tmp" / "organ_sweeps" / args.name
    infer_root = EXPERIMENTS / "infer_results" / "organ_sweeps" / args.name
    manifest_path = sweep_root / "manifest.json"
    manifest = {
        "name": args.name,
        "best_metric": args.best_metric,
        "minimize_best_metric": args.minimize_best_metric,
        "runs": [],
    }

    run_index = 0
    for combo in combos(args):
        for fold in args.folds:
            if args.max_runs is not None and run_index >= args.max_runs:
                save_json(manifest_path, manifest)
                print(f"[sweep] reached --max-runs={args.max_runs}")
                return

            name = exp_name(args, fold, combo)
            if args.wandb_run_name_prefix:
                name = f"{args.wandb_run_name_prefix}_{name}"
            save_dir = save_root / name
            tmp_save_dir = tmp_root / name
            log_dir = log_root / f"log_{name}"
            results_dir = infer_root / f"infer_{name}"
            config_path = config_dir / f"{name}.json"
            checkpoint = save_dir / "checkpoint_best.pt"
            record = {
                "index": run_index,
                "fold": fold,
                "name": name,
                "combo": combo,
                "save_dir": str(save_dir),
                "tmp_save_dir": str(tmp_save_dir),
                "log_dir": str(log_dir),
                "infer_dir": str(results_dir),
                "checkpoint": str(checkpoint),
                "config_path": str(config_path),
                "status": "pending",
            }
            manifest["runs"].append(record)
            save_json(config_path, record)
            save_json(manifest_path, manifest)

            if args.clean:
                clean_run_dirs(save_dir, tmp_save_dir, log_dir, results_dir)

            if args.skip_existing and checkpoint.exists() and results_dir.exists():
                print(f"[sweep] skipping existing run: {name}")
                record["status"] = "skipped_existing"
                record["test_metrics"] = load_test_metrics(results_dir, save_dir)
                maybe_copy_best(sweep_root, record, args.best_metric, args.minimize_best_metric)
                save_json(manifest_path, manifest)
                run_index += 1
                continue

            print(f"\n=== Organ sweep run {run_index}: {name} ===")
            env = build_env(args, name)
            train_rc = run_cmd(
                build_train_cmd(args, fold, combo, save_dir, tmp_save_dir, log_dir),
                env,
                args.dry_run,
            )
            if train_rc != 0:
                record["status"] = "train_failed"
                record["returncode"] = train_rc
                save_json(manifest_path, manifest)
                if not args.keep_going:
                    raise SystemExit(train_rc)
                run_index += 1
                continue

            if not args.dry_run and not checkpoint.exists():
                record["status"] = "missing_checkpoint"
                save_json(manifest_path, manifest)
                if not args.keep_going:
                    raise SystemExit(f"missing checkpoint: {checkpoint}")
                run_index += 1
                continue

            infer_rc = run_cmd(
                build_infer_cmd(args, fold, combo, save_dir, results_dir),
                env,
                args.dry_run,
            )
            if infer_rc != 0:
                record["status"] = "infer_failed"
                record["returncode"] = infer_rc
                save_json(manifest_path, manifest)
                if not args.keep_going:
                    raise SystemExit(infer_rc)
                run_index += 1
                continue

            record["status"] = "dry_run" if args.dry_run else "completed"
            record["test_metrics"] = {} if args.dry_run else load_test_metrics(results_dir, save_dir)
            maybe_copy_best(sweep_root, record, args.best_metric, args.minimize_best_metric)
            save_json(manifest_path, manifest)
            run_index += 1

    print(f"\n[sweep] complete: {manifest_path}")
    best_path = sweep_root / "best_model.json"
    if best_path.exists():
        print(f"[sweep] best model: {best_path}")


if __name__ == "__main__":
    main()
