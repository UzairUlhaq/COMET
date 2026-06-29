"""Visualize the organ-finetune model's LNP embeddings with UMAP.

What it does
------------
1. Runs `infer_np.py --output-cls-rep` on a trained organ checkpoint. With the
   `--output-cls-rep` flag the soft-cross-entropy loss surfaces the per-sample
   LNP [CLS] embedding (the `organ` head's <s> token rep, dim = 256) into the
   inference `.out.pkl`, alongside the predicted distribution (`prob`), the soft
   target (`target`) and `lnp_ids`. See:
     - unimol/losses/np_finetune_soft_cross_entropy.py  (logging_output keys)
     - unimol/models/unimol.py:694                       (cls_representations["organ"])
2. Loads that pkl, derives each sample's class as argmax(soft target), runs UMAP
   to 2D, and saves a scatter plot colored by organ class (+ the 2D coords CSV).

The embedding is the model's learned formulation representation right before the
classification head, so the scatter shows how the model separates organs in its
latent space.

Usage
-----
    # default: project the test split of the fold-V0 organ model
    python visualize_embeddings_organ.py

    # project all splits together (more points = clearer cluster structure)
    python visualize_embeddings_organ.py --subsets train,valid,test

    # re-plot from an inference pkl that already exists (skip the model run)
    python visualize_embeddings_organ.py --from-pkl infer_results/.../save_lnpdb_organ_test.out.pkl

Requires `umap-learn` (`pip install umap-learn`). If it isn't installed the
script falls back to PCA and tells you, so you still get a plot.
"""

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
os.chdir(EXPERIMENTS_DIR)

# Organ order == position in the softmax/target vector. Kept identical to
# dataset_preprocessing_organ.py:ORGAN_CLASSES so labels line up with `target`.
ORGAN_CLASSES = [
    "lung_epithelium", "liver", "muscle", "spleen", "bone_marrow",
    "heart", "lung", "kidney", "ear",
]

# Model architecture used to train save_lnpdb_organ/checkpoint_best.pt
# (read off the checkpoint's stored args; mirror of the heartkidney recipe).
LNP_ENCODER_LAYERS = 8
LNP_ENCODER_EMBED_DIM = 256
LNP_ENCODER_FFN_EMBED_DIM = 256
LNP_ENCODER_ATTENTION_HEADS = 8

DICT_NAME = "dict.txt"
CONF_SIZE = 11
ONLY_POLAR = 0
TASK_NUM = 9                       # softmax over the 9 organs
LOSS_FUNC = "np_finetune_soft_cross_entropy"
TASK_NAME = "processed_data_dirs/lnpdb_organ_gen/fold_V0"
SCHEMA_PATH = "task_schemas/lnpdb_organ_schema.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="UMAP visualization of the organ-finetune model's LNP embeddings."
    )
    p.add_argument(
        "--weight-path",
        default="save_lnpdb_organ/checkpoint_best.pt",
        help="trained organ checkpoint to embed with",
    )
    p.add_argument(
        "--task-name", default=TASK_NAME,
        help="data folder (fold) the splits live under",
    )
    p.add_argument(
        "--subsets", default="test",
        help="comma-separated splits to embed (train,valid,test)",
    )
    p.add_argument(
        "--results-path", default="./infer_results/organ_embeddings",
        help="where infer_np.py writes its .out.pkl",
    )
    p.add_argument(
        "--out-prefix", default="./infer_results/organ_embeddings/umap_organ",
        help="output path prefix for the .png and .csv",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--color-by", choices=["target", "pred"], default="target",
        help="color points by true organ (argmax target) or predicted (argmax prob)",
    )
    p.add_argument(
        "--single-organ-only", action="store_true",
        help="drop multi-organ (soft) samples for a cleaner, unambiguous plot",
    )
    p.add_argument(
        "--from-pkl", default=None,
        help="skip inference and load embeddings from this existing .out.pkl",
    )
    # UMAP knobs
    p.add_argument("--n-neighbors", type=int, default=60)
    p.add_argument("--min-dist", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def run_inference(args, subset):
    """Run infer_np.py with --output-cls-rep for one split; return the pkl path."""
    os.makedirs(args.results_path, exist_ok=True)
    # infer_np.py names the file as <checkpoint-parent-dir>_<subset>.out.pkl
    fname = args.weight_path.split("/")[-2]
    pkl_path = os.path.join(args.results_path, f"{fname}_{subset}.out.pkl")

    cmd = (
        f"python ../unimol/infer_np.py --user-dir ../unimol ./ "
        f"--task-name {args.task_name} --valid-subset {subset} "
        f"--num-workers {args.num_workers} --ddp-backend=c10d "
        f"--batch-size {args.batch_size} --required-batch-size-multiple 1 "
        f"--task mol_np_finetune --loss {LOSS_FUNC} --arch np_unimol "
        f"--classification-head-name organ --num-classes {TASK_NUM} "
        f"--dict-name {DICT_NAME} --conf-size {CONF_SIZE} --only-polar {ONLY_POLAR} "
        f"--path {args.weight_path} "
        f"--fp16 --fp16-init-scale 4 --fp16-scale-window 256 "
        f"--log-interval 50 --log-format simple "
        f"--results-path {args.results_path} "
        f"--lnp-encoder-layers {LNP_ENCODER_LAYERS} "
        f"--lnp-encoder-embed-dim {LNP_ENCODER_EMBED_DIM} "
        f"--lnp-encoder-ffn-embed-dim {LNP_ENCODER_FFN_EMBED_DIM} "
        f"--lnp-encoder-attention-heads {LNP_ENCODER_ATTENTION_HEADS} "
        f"--full-dataset-task-schema-path {SCHEMA_PATH} "
        f"--load-full-np-model --concat-datasets "
        f"--output-cls-rep"
    )
    print(f"\n[infer] {subset}:\n{cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 or not os.path.exists(pkl_path):
        raise RuntimeError(f"inference failed for subset '{subset}' (expected {pkl_path})")
    return pkl_path


def load_embeddings(pkl_path):
    """Pull (embeddings [N,D], target [N,9], prob [N,9], lnp_ids [N]) out of a pkl."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    # The CLS rep is keyed by task name -> "cls_organ"; fall back to any cls_* key.
    cls_keys = [k for k in d if k.startswith("cls_")]
    if not cls_keys:
        raise KeyError(
            f"No cls_* embeddings in {pkl_path}. "
            "Was inference run with --output-cls-rep?"
        )
    emb = np.asarray(d[cls_keys[0]], dtype=np.float32)          # [N, D]
    target = np.asarray(d.get("target"), dtype=np.float32)      # [N, 9]
    prob = np.asarray(d.get("prob"), dtype=np.float32)          # [N, 9]
    lnp_ids = list(d.get("lnp_ids", [None] * len(emb)))
    return emb, target, prob, lnp_ids


def reduce_2d(emb, args):
    """UMAP -> 2D, falling back to PCA if umap-learn isn't installed."""
    try:
        import umap  # umap-learn
        reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            n_components=2,
            random_state=args.seed,
            metric="euclidean",
        )
        coords = reducer.fit_transform(emb)
        return coords, "UMAP"
    except ImportError:
        print(
            "\n[warn] umap-learn not installed -> falling back to PCA.\n"
            "       For the UMAP plot you asked for: pip install umap-learn\n"
        )
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=args.seed).fit_transform(emb)
        return coords, "PCA"


def plot(coords, labels, method, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    present = [c for c in ORGAN_CLASSES if c in set(labels)]
    for i, organ in enumerate(present):
        m = labels == organ
        ax.scatter(
            coords[m, 0], coords[m, 1],
            s=18, alpha=0.7, color=cmap(i % 10),
            label=f"{organ} (n={int(m.sum())})", edgecolors="none",
        )
    ax.set_title(f"{method} of organ-model LNP embeddings (colored by organ class)")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    png_path = out_prefix + ".png"
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=200)
    print(f"[saved] plot  -> {png_path}")
    return png_path


def save_csv(coords, labels, lnp_ids, method, out_prefix):
    csv_path = out_prefix + ".csv"
    with open(csv_path, "w") as f:
        f.write(f"lnp_id,{method.lower()}_1,{method.lower()}_2,organ\n")
        for lid, (x, y), lab in zip(lnp_ids, coords, labels):
            f.write(f"{lid},{x:.6f},{y:.6f},{lab}\n")
    print(f"[saved] coords -> {csv_path}")


def main():
    args = parse_args()

    # 1) collect embeddings (run inference per subset unless reusing a pkl)
    emb_list, tgt_list, prob_list, id_list = [], [], [], []
    if args.from_pkl:
        pkls = [args.from_pkl]
    else:
        pkls = [run_inference(args, s.strip())
                for s in args.subsets.split(",") if s.strip()]

    for pkl_path in pkls:
        emb, target, prob, lnp_ids = load_embeddings(pkl_path)
        emb_list.append(emb)
        tgt_list.append(target)
        prob_list.append(prob)
        id_list.extend(lnp_ids)

    emb = np.concatenate(emb_list, axis=0)
    target = np.concatenate(tgt_list, axis=0)
    prob = np.concatenate(prob_list, axis=0)
    print(f"\n[info] {emb.shape[0]} samples, embedding dim {emb.shape[1]}")

    # 2) per-sample class label
    scores = target if args.color_by == "target" else prob
    class_idx = scores.argmax(axis=1)
    labels = np.array([ORGAN_CLASSES[i] for i in class_idx])

    # optionally keep only single-organ (one-hot target) samples
    if args.single_organ_only:
        is_single = (target > 0).sum(axis=1) == 1
        kept = int(is_single.sum())
        print(f"[info] single-organ-only: keeping {kept}/{len(is_single)} samples")
        emb, labels, target = emb[is_single], labels[is_single], target[is_single]
        id_list = [i for i, k in zip(id_list, is_single) if k]

    # 3) reduce + 4) plot
    coords, method = reduce_2d(emb, args)
    plot(coords, labels, method, args.out_prefix)
    save_csv(coords, labels, id_list, method, args.out_prefix)

    # class distribution summary
    print("\n[class counts]")
    uniq, cnts = np.unique(labels, return_counts=True)
    for u, c in sorted(zip(uniq, cnts), key=lambda x: -x[1]):
        print(f"  {u:<16} {c}")


if __name__ == "__main__":
    main()

