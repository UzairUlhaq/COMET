"""Step 5 — run inference with the fine-tuned organ classifier.

Loads checkpoint_best.pt and predicts on the held-out test split. The loss's
validation path produces softmax probabilities; with --output-cls-rep the
per-LNP [CLS] embedding is also dumped. Output: a .out.pkl under results-path.

Run from the repo root:  python organ_finetune/step5_infer.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"

FOLD = 0
TASK_NAME = f"processed_data_dirs/lnpdb_organ_gen/fold_V{FOLD}"
SCHEMA = "task_schemas/lnpdb_organ_schema.json"
HEAD_NAME = "organ"
NUM_CLASSES = 9
LOSS = "np_finetune_soft_cross_entropy"
DICT_NAME = "dict.txt"
CONF_SIZE = 11
ONLY_POLAR = 0
BATCH_SIZE = 8

WEIGHT_PATH = "./save_lnpdb_organ/checkpoint_best.pt"
RESULTS_PATH = "./infer_results/lnpdb_organ"

LNP_LAYERS, LNP_EMBED, LNP_FFN, LNP_HEADS = 8, 256, 256, 8


def main():
    cmd = (
        f"{sys.executable} ../unimol/infer_np.py --user-dir ../unimol ./ "
        f"--task-name {TASK_NAME} --valid-subset test "
        f"--num-workers 0 --ddp-backend=c10d --batch-size {BATCH_SIZE} --required-batch-size-multiple 1 "
        f"--task mol_np_finetune --loss {LOSS} --arch np_unimol "
        f"--classification-head-name {HEAD_NAME} --num-classes {NUM_CLASSES} "
        f"--dict-name {DICT_NAME} --conf-size {CONF_SIZE} --only-polar {ONLY_POLAR} "
        f"--path {WEIGHT_PATH} "
        f"--fp16 --fp16-init-scale 4 --fp16-scale-window 256 "
        f"--log-interval 50 --log-format simple "
        f"--results-path {RESULTS_PATH} "
        f"--lnp-encoder-layers {LNP_LAYERS} --lnp-encoder-embed-dim {LNP_EMBED} "
        f"--lnp-encoder-ffn-embed-dim {LNP_FFN} --lnp-encoder-attention-heads {LNP_HEADS} "
        f"--full-dataset-task-schema-path {SCHEMA} "
        f"--load-full-np-model --concat-datasets --output-cls-rep"
    )
    print(f"[step5] cwd={EXPERIMENTS}\n[step5] {cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=EXPERIMENTS)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
