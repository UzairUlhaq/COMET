"""Step 4 — fine-tune the organ classifier.

Loads the pretrained UniMol molecular encoder (ckp/mol_pre_no_h_220816.pt) and
trains the LNP encoder + a single 9-way softmax head on the organ distribution,
using the soft-target cross-entropy loss (np_finetune_soft_cross_entropy).

Mirrors the proven experiments/training_script_LNPDB_heartkidney.py invocation
(same task/arch/optimizer/encoder flags, run with cwd=experiments so ../unimol,
../ckp and task_schemas/ resolve), changing only:
  * --loss        -> np_finetune_soft_cross_entropy   (soft distribution target)
  * --num-classes -> 9                                 (one logit per organ)
  * --classification-head-name -> organ                (matches the schema task)
  * --best-checkpoint-metric   -> valid_top1_acc (maximize)
and dropping the contrastive-only flags.

Run from the repo root:  python organ_finetune/step4_train.py
Tip: do a SMOKE TEST first with --smoke (1 epoch) to validate the wiring.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"

# ---- config -----------------------------------------------------------------
FOLD = 0
TASK_NAME = f"processed_data_dirs/lnpdb_organ_gen/fold_V{FOLD}"
SCHEMA = "task_schemas/lnpdb_organ_schema.json"
HEAD_NAME = "organ"
NUM_CLASSES = 9                     # == len(ORGAN_CLASSES) in step2
LOSS = "np_finetune_soft_cross_entropy"
PRETRAINED = "../ckp/mol_pre_no_h_220816.pt"
DICT_NAME = "dict.txt"

LR = 1e-4
BATCH_SIZE = 8
WARMUP = 0.06
DROPOUT = 0.1
EPOCHS = 30
CONF_SIZE = 11
ONLY_POLAR = 0
SEED = 1
METRIC = "valid_top1_acc"

LNP_LAYERS, LNP_EMBED, LNP_FFN, LNP_HEADS = 8, 256, 256, 8

SAVE_DIR = "./save_lnpdb_organ"
TMP_SAVE_DIR = "./tmp_save_lnpdb_organ"
LOG_DIR = "./logs/tmp/log_lnpdb_organ"


def build_cmd(max_epoch):
    return (
        f"{sys.executable} ../unimol/train_np.py ./ "
        f"--task-name {TASK_NAME} --user-dir ../unimol "
        f"--train-subset train --valid-subset valid "
        f"--conf-size {CONF_SIZE} --num-workers 0 --ddp-backend=c10d "
        f"--dict-name {DICT_NAME} "
        f"--task mol_np_finetune --loss {LOSS} --arch np_unimol "
        f"--classification-head-name {HEAD_NAME} --num-classes {NUM_CLASSES} "
        f"--optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 "
        f"--lr-scheduler polynomial_decay --lr {LR} --warmup-ratio {WARMUP} "
        f"--max-epoch {max_epoch} --batch-size {BATCH_SIZE} --required-batch-size-multiple 1 "
        f"--pooler-dropout {DROPOUT} --seed {SEED} "
        f"--fp16 --fp16-init-scale 4 --fp16-scale-window 256 "
        f"--log-interval 100 --log-format simple --validate-interval 1 --keep-last-epochs 5 "
        f"--finetune-from-model {PRETRAINED} "
        f"--best-checkpoint-metric {METRIC} --patience 20 --maximize-best-checkpoint-metric "
        f"--save-dir {SAVE_DIR} --tmp-save-dir {TMP_SAVE_DIR} --only-polar {ONLY_POLAR} "
        f"--tensorboard-logdir {LOG_DIR} "
        f"--full-dataset-task-schema-path {SCHEMA} "
        f"--freeze-molecule-encoder --epoch-to-freeze-molecule-encoder 1000000 "
        f"--concat-datasets "
        f"--lnp-encoder-layers {LNP_LAYERS} --lnp-encoder-embed-dim {LNP_EMBED} "
        f"--lnp-encoder-ffn-embed-dim {LNP_FFN} --lnp-encoder-attention-heads {LNP_HEADS}"
    )


def build_wandb_env(args):
    """Return env vars that switch on the framework's built-in W&B logging.

    The training loop already pushes per-epoch train/valid stats (and per-interval
    train_inner stats) through unimol/core/logging/progress_bar.py, which mirrors
    every scalar to W&B when COMET_WANDB_LIVE is truthy. So enabling W&B is just a
    matter of exporting these vars to the subprocess — no change to the train loop.

    A stable run id (derived from the save dir / fold) means a resumed training run
    continues the SAME W&B run instead of starting a fresh chart each launch.
    """
    env = dict(os.environ)
    env["COMET_WANDB_LIVE"] = "1"
    env["WANDB_PROJECT"] = args.wandb_project
    env["COMET_WANDB_RUN_ID"] = args.wandb_run_id or f"lnpdb_organ_fold{FOLD}"
    run_name = args.wandb_run_name or (
        f"organ_fold{FOLD}_smoke" if args.smoke else f"organ_fold{FOLD}"
    )
    env["COMET_WANDB_RUN_NAME"] = run_name
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_offline:
        env["WANDB_MODE"] = "offline"
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("[step4] WARNING: --wandb set but the 'wandb' package is not "
              "installed; W&B logging will be skipped. Install with: pip install wandb")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1-epoch run to validate the wiring before a full train.")
    ap.add_argument("--wandb", action="store_true",
                    help="Log per-epoch (and per-interval) metrics to Weights & Biases.")
    ap.add_argument("--wandb-project", default="comet-lnpdb-organ",
                    help="W&B project name (default: comet-lnpdb-organ).")
    ap.add_argument("--wandb-run-name", default=None,
                    help="W&B run display name (default: organ_fold<FOLD>[_smoke]).")
    ap.add_argument("--wandb-run-id", default=None,
                    help="Stable W&B run id; resuming reuses it to continue the same "
                         "run (default: lnpdb_organ_fold<FOLD>).")
    ap.add_argument("--wandb-entity", default=None,
                    help="W&B entity (team/user). Defaults to your wandb login.")
    ap.add_argument("--wandb-offline", action="store_true",
                    help="Run W&B in offline mode (sync later with `wandb sync`).")
    args = ap.parse_args()

    max_epoch = 1 if args.smoke else EPOCHS
    cmd = build_cmd(max_epoch)
    env = build_wandb_env(args) if args.wandb else None
    if env is not None:
        print(f"[step4] W&B logging ON -> project={env['WANDB_PROJECT']} "
              f"run={env['COMET_WANDB_RUN_NAME']} id={env['COMET_WANDB_RUN_ID']}")
    print(f"[step4] cwd={EXPERIMENTS}\n[step4] {cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=EXPERIMENTS, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
