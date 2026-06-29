"""Compare organ LNP embeddings BEFORE vs AFTER organ finetuning, with UMAP.

Why this works the way it does
------------------------------
The organ recipe FREEZES the pretrained UniMol molecular encoder
(`--freeze-molecule-encoder`) and trains the 8-layer LNP cross-component
transformer (+ the `organ` CLS embedding and classification head) from scratch.
So "before vs after finetuning" compares the frozen molecular encoder's raw
formulation signal against the trained LNP-level representation:

    before : frozen pretrained mol encoder  (composition-weighted component CLS)
    after  : frozen pretrained mol encoder  +  TRAINED LNP transformer (checkpoint_best)

Both states embed the *same* organ samples (we don't shuffle val/test, so sample
order — and therefore the per-sample organ label — is identical across passes).
The "before" embedding is a composition-weighted mean of each component molecule's
pretrained UniMol CLS representation. The "after" embedding is the model's
`organ` CLS token output
(`cls_representations["organ"]`, dim 256; see unimol/models/unimol.py:694), the
formulation representation right before the classification head.

The "before" pass can't go through infer_np.py: the base checkpoint
(mol_pre_no_h_220816.pt) holds only mol-encoder weights whose keys lack the
`mol_model.` prefix, so it must be loaded into `model.mol_model` alone (mirroring
trainer.py:399). We therefore build the model in-process, load the mol encoder,
and pool component-level UniMol embeddings directly.

Usage
-----
    python visualize_embeddings_before_after_organ.py

    # all splits, drop multi-organ (soft) samples for unambiguous coloring
    python visualize_embeddings_before_after_organ.py \
        --valid-subset train,valid,test --single-organ-only

Requires `umap-learn` (`pip install umap-learn`); falls back to PCA otherwise.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pyprojroot import here as project_root

EXPERIMENTS_DIR = Path(__file__).resolve().parent
os.chdir(EXPERIMENTS_DIR)

sys.path.insert(0, str(project_root()))
importlib.import_module("unimol")

from unimol.core import checkpoint_utils, options, tasks, utils  # noqa: E402

# Organ order == position in the softmax/target vector (matches
# dataset_preprocessing_organ.py:ORGAN_CLASSES).
ORGAN_CLASSES = [
    "lung_epithelium", "liver", "muscle", "spleen", "bone_marrow",
    "heart", "lung", "kidney", "ear",
]

# Architecture of save_lnpdb_organ/checkpoint_best.pt (read off its stored args).
ARCH_FLAGS = [
    "--lnp-encoder-layers", "8",
    "--lnp-encoder-embed-dim", "256",
    "--lnp-encoder-ffn-embed-dim", "256",
    "--lnp-encoder-attention-heads", "8",
]


def build_args():
    """Construct the unimol args via an explicit input_args list, plus a few
    extra knobs for this script (registered on the same parser)."""
    parser = options.get_validation_parser()
    options.add_model_args(parser)

    g = parser.add_argument_group("before/after embedding viz")
    g.add_argument("--base-weight-path", default="../ckp/mol_pre_no_h_220816.pt",
                   help="pretrained mol-encoder checkpoint = the 'before' state")
    g.add_argument("--umap-out-prefix",
                   default="./infer_results/organ_embeddings/umap_before_after_organ",
                   help="output path prefix for the .png and .csv")
    g.add_argument("--single-organ-only", action="store_true",
                   help="drop multi-organ (soft) samples for cleaner coloring")
    g.add_argument("--include-organs", default=None,
                   help="comma-separated organ names to keep in the plot")
    g.add_argument("--n-neighbors", type=int, default=60)
    g.add_argument("--min-dist", type=float, default=0.3)
    # Fixed flags describing the organ task/model; user can still override
    # --path, --valid-subset, --batch-size, --seed, --base-weight-path on the CLI.
    input_args = [
        "./",
        "--task-name", "processed_data_dirs/lnpdb_organ_gen/fold_V0",
        "--valid-subset", "test",
        "--num-workers", "0", "--ddp-backend=c10d",
        "--batch-size", "4", "--required-batch-size-multiple", "1",
        "--task", "mol_np_finetune",
        "--loss", "np_finetune_soft_cross_entropy",
        "--arch", "np_unimol",
        "--classification-head-name", "organ", "--num-classes", "9",
        "--dict-name", "dict.txt", "--conf-size", "11", "--only-polar", "0",
        "--path", "save_lnpdb_organ_fold_V1/checkpoint_best.pt",
        "--results-path", "./infer_results/organ_embeddings",
        "--full-dataset-task-schema-path", "task_schemas/lnpdb_organ_schema.json",
        "--load-full-np-model", "--concat-datasets", "--output-cls-rep",
        *ARCH_FLAGS,
    ] + sys.argv[1:]  # CLI args override / extend the defaults above

    return options.parse_args_and_arch(parser, input_args=input_args)


def build_model_in_state(task, args, state):
    """Build a fresh np_unimol model and put it in 'before' or 'after' weights.

    before -> frozen pretrained mol encoder + random (seeded) LNP transformer
    after  -> full finetuned checkpoint (args.path)
    """
    # Seed so the 'before' random LNP transformer is reproducible.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = task.build_model(args)

    if state == "after":
        ckpt = checkpoint_utils.load_checkpoint_to_cpu(args.path)
        errors = model.load_state_dict(ckpt["model"], strict=False)
        loaded = "full finetuned checkpoint " + args.path
    elif state == "before":
        # Base checkpoint holds ONLY mol-encoder weights -> load into mol_model
        # alone; the LNP transformer stays at its seeded random init.
        ckpt = checkpoint_utils.load_checkpoint_to_cpu(args.base_weight_path)
        errors = model.mol_model.load_state_dict(ckpt["model"], strict=False)
        loaded = "pretrained mol encoder " + args.base_weight_path + " (LNP transformer random)"
    else:
        raise ValueError(state)

    if getattr(errors, "unexpected_keys", None):
        print(f"[{state}] unexpected keys (first 5): {errors.unexpected_keys[:5]}")
    print(f"[{state}] loaded {loaded}")
    return model


def _target_tensor(sample, head_name):
    target = sample["target"]["finetune_target"]
    if isinstance(target, dict):
        if head_name in target:
            return target[head_name]
        if len(target) == 1:
            return next(iter(target.values()))
        raise KeyError(f"target head '{head_name}' not in {list(target.keys())}")
    return target


def extract_unimol_component_embeddings(task, args, model, use_cuda):
    """Extract pure pretrained UniMol formulation embeddings.

    Each formulation has multiple component molecules. We take the pretrained
    UniMol CLS vector for each component and compute a composition-weighted mean
    using the component percentages. No LNP transformer or random task CLS is used.
    """
    embs, tgts, probs, ids = [], [], [], []
    for subset in args.valid_subset.split(","):
        subset = subset.strip()
        if not subset:
            continue
        if args.concat_datasets:
            task.load_concat_dataset(subset, combine=False, epoch=1)
        else:
            task.load_dataset(subset, combine=False, epoch=1)
        dataset = task.dataset(subset)

        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)

        for sample in itr:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            with torch.no_grad():
                encoder_rep, _ = model.mol_model(
                    **sample["mol_features"],
                    features_only=True,
                    output_rep_only=True,
                )
                mol_rep = encoder_rep[:, 0, :]
                mol_batch_ids = sample["mol_batch_ids"].clone()
                mol_batch_ids_shape = mol_batch_ids.shape
                flat_ids = mol_batch_ids.flatten()

                pad_rep = torch.zeros_like(mol_rep[0]).unsqueeze(0)
                mol_rep = torch.cat([mol_rep, pad_rep], dim=0)
                pad_idx = mol_rep.shape[0] - 1
                valid_components = flat_ids.ne(-1).view(mol_batch_ids_shape)
                flat_ids[flat_ids == -1] = pad_idx

                flat_component_rep = torch.index_select(mol_rep, 0, flat_ids)
                component_rep = torch.unflatten(flat_component_rep, 0, mol_batch_ids_shape)

                percents = sample["net_input"]["percents"].float()
                if percents.dim() == 3 and percents.size(-1) == 1:
                    percents = percents.squeeze(-1)
                weights = percents * valid_components.to(percents)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                formulation_rep = (component_rep * weights.unsqueeze(-1)).sum(dim=1)

                target = _target_tensor(sample, args.classification_head_name).float()

            embs.append(formulation_rep.float().cpu().numpy())
            tgts.append(target.float().cpu().numpy())
            # There is no pretrained-UniMol organ head, so keep a placeholder to
            # preserve the function's return shape.
            probs.append(torch.zeros_like(target).float().cpu().numpy())
            ids.extend(list(sample["net_input"].get("lnp_ids", [])))

    return (
        np.concatenate(embs),
        np.concatenate(tgts),
        np.concatenate(probs),
        ids,
    )

def extract_embeddings(task, args, model, loss, use_cuda):
    """Run the (unshuffled) splits through `model` and gather the per-sample
    `cls_organ` embedding, soft target, predicted prob, and lnp_ids."""
    embs, tgts, probs, ids = [], [], [], []
    for subset in args.valid_subset.split(","):
        subset = subset.strip()
        if not subset:
            continue
        if args.concat_datasets:
            task.load_concat_dataset(subset, combine=False, epoch=1)
        else:
            task.load_dataset(subset, combine=False, epoch=1)
        dataset = task.dataset(subset)

        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)

        log_outputs = []
        for sample in itr:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            _, _, log_output = task.valid_step(
                sample, model, loss, test=True, infer=False, output_cls_rep=True
            )
            log_outputs.append(log_output)

        reduced = task.reduce_metrics(log_outputs, loss, subset, infer=False)
        cls_key = next(k for k in reduced if k.startswith("cls_"))
        embs.append(np.asarray(reduced[cls_key], dtype=np.float32))
        tgts.append(np.asarray(reduced["target"], dtype=np.float32))
        probs.append(np.asarray(reduced["prob"], dtype=np.float32))
        ids.extend(list(reduced.get("lnp_ids", [])))

    return (np.concatenate(embs), np.concatenate(tgts),
            np.concatenate(probs), ids)


def reduce_2d(emb, args):
    """UMAP -> 2D, PCA fallback if umap-learn isn't installed."""
    try:
        import umap
        coords = umap.UMAP(
            n_neighbors=args.n_neighbors, min_dist=args.min_dist,
            n_components=2, random_state=args.seed, metric="euclidean",
        ).fit_transform(emb)
        return coords, "UMAP"
    except ImportError:
        print("\n[warn] umap-learn not installed -> PCA fallback. "
              "For UMAP: pip install umap-learn\n")
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=args.seed).fit_transform(emb), "PCA"

def plot_before_after(coords_before, coords_after, labels, method, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)
    cmap = plt.get_cmap("tab10")
    present = [c for c in ORGAN_CLASSES if c in set(labels)]
    color_of = {organ: cmap(i % 10) for i, organ in enumerate(present)}

    for ax, coords, title in (
        (axes[0], coords_before, "Before finetuning (pretrained UniMol component pool)"),
        (axes[1], coords_after, "After finetuning (checkpoint_best)"),
    ):
        for organ in present:
            m = labels == organ
            ax.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.7,
                       color=color_of[organ], edgecolors="none",
                       label=f"{organ} (n={int(m.sum())})")
        ax.set_title(title)
        ax.set_xlabel(f"{method}-1")
        ax.set_ylabel(f"{method}-2")
    axes[1].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{method} of organ LNP embeddings — before vs after finetuning",
                 fontsize=13)
    fig.tight_layout()

    png_path = out_prefix + ".png"
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=200)
    print(f"[saved] plot   -> {png_path}")


def save_csv(coords_before, coords_after, labels, ids, method, out_prefix):
    m = method.lower()
    csv_path = out_prefix + ".csv"
    with open(csv_path, "w") as f:
        f.write(f"lnp_id,organ,before_{m}_1,before_{m}_2,after_{m}_1,after_{m}_2\n")
        for lid, lab, (bx, by), (ax_, ay) in zip(ids, labels, coords_before, coords_after):
            f.write(f"{lid},{lab},{bx:.6f},{by:.6f},{ax_:.6f},{ay:.6f}\n")
    print(f"[saved] coords -> {csv_path}")


def filter_organs(args, emb_b, emb_a, labels, ids):
    if not args.include_organs:
        return emb_b, emb_a, labels, ids

    requested = [name.strip() for name in args.include_organs.split(",") if name.strip()]
    unknown = [name for name in requested if name not in ORGAN_CLASSES]
    if unknown:
        raise ValueError(
            f"Unknown organ(s): {unknown}. Valid names: {ORGAN_CLASSES}"
        )

    keep = np.isin(labels, requested)
    print(f"[info] include-organs={requested}: keeping {int(keep.sum())}/{len(keep)} samples")
    if not keep.any():
        raise ValueError(f"No samples matched --include-organs={args.include_organs}")
    return (
        emb_b[keep],
        emb_a[keep],
        labels[keep],
        [i for i, k in zip(ids, keep) if k],
    )


def main():
    args = build_args()
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)

    task = tasks.setup_task(args)
    loss = task.build_loss(args)
    loss.eval()

    results = {}

    model = build_model_in_state(task, args, "before")
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()
    model.eval()
    results["before"] = extract_unimol_component_embeddings(task, args, model, use_cuda)
    del model
    if use_cuda:
        torch.cuda.empty_cache()

    model = build_model_in_state(task, args, "after")
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()
    model.eval()
    results["after"] = extract_embeddings(task, args, model, loss, use_cuda)
    del model
    if use_cuda:
        torch.cuda.empty_cache()

    emb_b, target, prob, ids = results["before"]
    emb_a, _, _, _ = results["after"]
    print(f"\n[info] {emb_b.shape[0]} samples, embedding dim {emb_b.shape[1]}")

    labels = np.array([ORGAN_CLASSES[i] for i in target.argmax(axis=1)])

    if args.single_organ_only:
        keep = (target > 0).sum(axis=1) == 1
        print(f"[info] single-organ-only: keeping {int(keep.sum())}/{len(keep)}")
        emb_b, emb_a, labels = emb_b[keep], emb_a[keep], labels[keep]
        ids = [i for i, k in zip(ids, keep) if k]

    emb_b, emb_a, labels, ids = filter_organs(args, emb_b, emb_a, labels, ids)

    coords_b, method = reduce_2d(emb_b, args)
    coords_a, _ = reduce_2d(emb_a, args)

    plot_before_after(coords_b, coords_a, labels, method, args.umap_out_prefix)
    save_csv(coords_b, coords_a, labels, ids, method, args.umap_out_prefix)

    print("\n[class counts]")
    uniq, cnts = np.unique(labels, return_counts=True)
    for u, c in sorted(zip(uniq, cnts), key=lambda x: -x[1]):
        print(f"  {u:<16} {c}")


if __name__ == "__main__":
    main()
