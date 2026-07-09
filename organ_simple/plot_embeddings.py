import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from constants import COMPONENT_TYPES, ORGAN_CLASSES
from data import LNPEmbeddingDataset
from model import SimpleLNPTransformer


def class_indices_for(organ_classes):
    return [ORGAN_CLASSES.index(name) for name in organ_classes]


def load_checkpoint(checkpoint_path, device):
    return torch.load(checkpoint_path, map_location=device)


def load_model(checkpoint, dataset, device, organ_classes):
    saved_args = checkpoint.get("args", {})

    model = SimpleLNPTransformer(
        component_embedding_dim=dataset.component_embeddings.shape[-1],
        num_component_types=len(COMPONENT_TYPES),
        num_classes=len(organ_classes),
        embed_dim=saved_args.get("embed_dim", 256),
        layers=saved_args.get("layers", 2),
        heads=saved_args.get("heads", 4),
        dropout=saved_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def masked_mean(component_embeddings, mask):
    mask = mask.unsqueeze(-1).float()
    summed = (component_embeddings * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp_min(1.0)
    return summed / count


@torch.no_grad()
def collect_embeddings(model, loader, device):
    cls_embeddings = []
    mean_embeddings = []
    targets = []
    probs = []
    lnp_ids = []

    for batch in loader:
        lnp_ids.extend(batch["lnp_id"])
        batch = {k: v.to(device) for k, v in batch.items() if k != "lnp_id"}

        logits, cls_rep = model(
            batch["component_embeddings"],
            batch["percents"],
            batch["component_types"],
            batch["mask"],
        )

        cls_embeddings.append(cls_rep.cpu().numpy())
        mean_embeddings.append(
            masked_mean(batch["component_embeddings"], batch["mask"]).cpu().numpy()
        )
        targets.append(batch["target"].cpu().numpy())
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    return {
        "cls": np.concatenate(cls_embeddings, axis=0),
        "mean_components": np.concatenate(mean_embeddings, axis=0),
        "target": np.concatenate(targets, axis=0),
        "prob": np.concatenate(probs, axis=0),
        "lnp_ids": lnp_ids,
    }


def reduce_umap(embeddings, args):
    try:
        import umap
    except ImportError:
        print("[skip] UMAP needs: pip install umap-learn")
        return None

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric="euclidean",
        random_state=args.seed,
    )
    return reducer.fit_transform(embeddings)


def reduce_tsne(embeddings, args):
    from sklearn.manifold import TSNE

    perplexity = min(args.perplexity, max(1, embeddings.shape[0] - 1))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )
    return reducer.fit_transform(embeddings)


def reduce_phate(embeddings, args):
    try:
        import phate
    except ImportError:
        print("[skip] PHATE needs: pip install phate")
        return None

    knn = min(args.phate_knn, max(1, embeddings.shape[0] - 2))
    reducer = phate.PHATE(
        n_components=2,
        knn=knn,
        random_state=args.seed,
        verbose=False,
    )
    return reducer.fit_transform(embeddings)


def run_reducer(method, embeddings, args):
    if method == "umap":
        return reduce_umap(embeddings, args)
    if method == "tsne":
        return reduce_tsne(embeddings, args)
    if method == "phate":
        return reduce_phate(embeddings, args)
    raise ValueError(f"unknown method: {method}")


def class_labels(target, prob, color_by, organ_classes):
    scores = target if color_by == "target" else prob
    idx = scores.argmax(axis=1)
    return np.array([organ_classes[i] for i in idx]), idx


def filter_single_organ(payload):
    target = payload["target"]
    keep = (target > 0).sum(axis=1) == 1
    return {
        key: [value[i] for i, ok in enumerate(keep) if ok]
        if key == "lnp_ids"
        else value[keep]
        for key, value in payload.items()
    }


def plot_coords(coords, labels, title, out_path, organ_classes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    present = [organ for organ in organ_classes if organ in set(labels)]

    for i, organ in enumerate(present):
        mask = labels == organ
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=22,
            alpha=0.75,
            color=cmap(i % 10),
            label=f"{organ} (n={int(mask.sum())})",
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] plot  -> {out_path}")


def save_coords_csv(path, coords, labels, lnp_ids, target, prob, organ_classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lnp_id",
        "x",
        "y",
        "label",
        *[f"target_{organ}" for organ in organ_classes],
        *[f"prob_{organ}" for organ in organ_classes],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, lnp_id in enumerate(lnp_ids):
            row = {
                "lnp_id": lnp_id,
                "x": coords[i, 0],
                "y": coords[i, 1],
                "label": labels[i],
            }
            for organ, value in zip(organ_classes, target[i].tolist()):
                row[f"target_{organ}"] = value
            for organ, value in zip(organ_classes, prob[i].tolist()):
                row[f"prob_{organ}"] = value
            writer.writerow(row)
    print(f"[saved] coords -> {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--cache-dir", default="organ_simple/cache")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--embedding",
        choices=["cls", "mean_components", "both"],
        default="cls",
        help="cls is the learned LNP representation before the organ head",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["umap", "tsne", "phate"],
        choices=["umap", "tsne", "phate"],
    )
    parser.add_argument("--color-by", choices=["target", "pred"], default="target")
    parser.add_argument("--single-organ-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--phate-knn", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    fold_dir = Path(args.cache_dir) / f"fold_V{args.fold}"
    run_dir = Path(args.run_dir or f"organ_simple/runs/fold_V{args.fold}")
    checkpoint_path = Path(args.checkpoint or run_dir / "best.pt")
    output_dir = Path(args.output_dir or run_dir / "embedding_plots")

    checkpoint = load_checkpoint(checkpoint_path, device)
    saved_args = checkpoint.get("args", {})
    organ_classes = saved_args.get("organ_classes", ORGAN_CLASSES)
    class_indices = saved_args.get("class_indices", class_indices_for(organ_classes))
    single_organ_only = args.single_organ_only or saved_args.get(
        "single_organ_only", False
    )

    dataset = LNPEmbeddingDataset(
        fold_dir / f"{args.split}.npz",
        single_organ_only=single_organ_only,
        class_indices=class_indices,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = load_model(checkpoint, dataset, device, organ_classes)
    payload = collect_embeddings(model, loader, device)

    if single_organ_only:
        print(
            f"[info] single-organ-only filter removed {dataset.num_filtered} "
            f"{args.split} samples"
        )

    labels, _ = class_labels(
        payload["target"], payload["prob"], args.color_by, organ_classes
    )
    embeddings_to_plot = (
        ["cls", "mean_components"] if args.embedding == "both" else [args.embedding]
    )

    print(
        f"[info] split={args.split} samples={len(labels)} "
        f"classes={','.join(organ_classes)} "
        f"color_by={args.color_by} methods={','.join(args.methods)}"
    )

    for embedding_name in embeddings_to_plot:
        embeddings = payload[embedding_name]
        print(f"[info] {embedding_name}: shape={embeddings.shape}")

        for method in args.methods:
            coords = run_reducer(method, embeddings, args)
            if coords is None:
                continue

            stem = f"{args.split}_{embedding_name}_{method}_{args.color_by}"
            title = (
                f"{method.upper()} of {embedding_name} embeddings "
                f"(fold V{args.fold}, {args.split}, colored by {args.color_by})"
            )
            plot_coords(
                coords, labels, title, output_dir / f"{stem}.png", organ_classes
            )
            save_coords_csv(
                output_dir / f"{stem}.csv",
                coords,
                labels,
                payload["lnp_ids"],
                payload["target"],
                payload["prob"],
                organ_classes,
            )

    print("\n[class counts]")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in sorted(zip(unique, counts), key=lambda item: -item[1]):
        print(f"  {label:<16} {count}")


if __name__ == "__main__":
    main()
