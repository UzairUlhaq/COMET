import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
ORGAN_SIMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORGAN_SIMPLE))
sys.path.insert(0, str(REPO_ROOT))

from constants import ORGAN_CLASSES  # noqa: E402
from data import LNPEmbeddingDataset  # noqa: E402
from plot_embeddings import (  # noqa: E402
    class_indices_for,
    collect_embeddings,
    load_checkpoint,
    load_model,
)


def import_scanpy():
    try:
        import scanpy as sc
    except ImportError as exc:
        raise SystemExit(
            "Scanpy is not installed. Install it in this environment with:\n"
            "  pip install scanpy igraph leidenalg\n"
            "or with conda/mamba from conda-forge."
        ) from exc
    return sc


def target_pattern(target, organ_classes):
    active = [
        f"{organ}:{value:g}"
        for organ, value in zip(organ_classes, target)
        if value > 0
    ]
    return " + ".join(active) if active else "[none]"


def build_obs(payload, organ_classes):
    target = payload["target"]
    prob = payload["prob"]
    true_idx = target.argmax(axis=1)
    pred_idx = prob.argmax(axis=1)

    obs = pd.DataFrame(
        {
            "lnp_id": [str(value) for value in payload["lnp_ids"]],
            "target_organ": [organ_classes[int(i)] for i in true_idx],
            "pred_organ": [organ_classes[int(i)] for i in pred_idx],
            "correct": pred_idx == true_idx,
            "num_target_organs": (target > 0).sum(axis=1),
            "target_pattern": [target_pattern(row, organ_classes) for row in target],
            "max_target": target.max(axis=1),
            "max_prob": prob.max(axis=1),
        }
    )

    for organ, values in zip(organ_classes, target.T):
        obs[f"target_{organ}"] = values
    for organ, values in zip(organ_classes, prob.T):
        obs[f"prob_{organ}"] = values

    obs.index = [f"{lnp_id}_{i}" for i, lnp_id in enumerate(obs["lnp_id"])]
    return obs


def build_anndata(sc, payload, embedding_name, organ_classes):
    x = payload[embedding_name].astype(np.float32)
    obs = build_obs(payload, organ_classes)
    var = pd.DataFrame(index=[f"{embedding_name}_{i}" for i in range(x.shape[1])])
    adata = sc.AnnData(X=x, obs=obs, var=var)
    adata.uns["organ_classes"] = organ_classes
    adata.uns["embedding_name"] = embedding_name
    return adata


def add_scanpy_embeddings(sc, adata, args):
    n_obs, n_vars = adata.X.shape
    if n_obs < 3:
        raise ValueError("Need at least 3 samples for neighbor graph and UMAP.")

    n_neighbors = min(args.n_neighbors, n_obs - 1)
    n_pcs = min(args.n_pcs, n_obs - 1, n_vars)
    use_rep = None

    if args.pca and n_pcs >= 2 and n_pcs < n_vars:
        sc.pp.pca(adata, n_comps=n_pcs, svd_solver="arpack")
        use_rep = "X_pca"

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs if use_rep == "X_pca" else None,
        use_rep=use_rep,
        metric=args.metric,
        random_state=args.seed,
    )
    sc.tl.umap(
        adata,
        min_dist=args.min_dist,
        spread=args.spread,
        random_state=args.seed,
    )

    if not args.skip_leiden:
        try:
            sc.tl.leiden(
                adata,
                resolution=args.resolution,
                random_state=args.seed,
                flavor="igraph",
                n_iterations=2,
                directed=False,
            )
        except Exception as exc:
            print(f"[warn] Leiden clustering skipped: {exc}")


def save_umap_csv(path, adata):
    coords = adata.obsm["X_umap"]
    obs = adata.obs.copy()
    obs.insert(0, "umap_y", coords[:, 1])
    obs.insert(0, "umap_x", coords[:, 0])
    path.parent.mkdir(parents=True, exist_ok=True)
    obs.to_csv(path, index=False)
    print(f"[saved] UMAP table -> {path}")


def save_cluster_summary(path, adata):
    rows = []
    if "leiden" not in adata.obs:
        return

    for cluster, group in adata.obs.groupby("leiden", observed=True):
        target_counts = group["target_organ"].value_counts()
        pred_counts = group["pred_organ"].value_counts()
        rows.append(
            {
                "leiden": cluster,
                "count": len(group),
                "top_target_organ": target_counts.index[0],
                "top_target_count": int(target_counts.iloc[0]),
                "top_pred_organ": pred_counts.index[0],
                "top_pred_count": int(pred_counts.iloc[0]),
                "accuracy": float(group["correct"].mean()),
                "lnp_ids": "|".join(group["lnp_id"].astype(str).tolist()),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] cluster summary -> {path}")


def save_plots(sc, adata, output_dir, stem, colors):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for color in colors:
        if color not in adata.obs:
            continue
        ax = sc.pl.umap(adata, color=color, show=False)
        fig = ax.figure
        out_path = output_dir / f"{stem}_umap_{color}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] plot -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scanpy UMAP, neighbor graph, and Leiden analysis for organ_simple embeddings."
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--cache-dir", default="organ_simple/cache")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--embedding",
        choices=["cls", "mean_components"],
        default="cls",
        help="cls uses the trained transformer representation; mean_components uses frozen component embeddings",
    )
    parser.add_argument("--single-organ-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--no-pca", dest="pca", action="store_false")
    parser.set_defaults(pca=True)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--resolution", type=float, default=0.7)
    parser.add_argument("--skip-leiden", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sc = import_scanpy()
    sc.settings.verbosity = 2

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    fold_dir = Path(args.cache_dir) / f"fold_V{args.fold}"
    run_dir = Path(args.run_dir or f"organ_simple/runs/fold_V{args.fold}")
    checkpoint_path = Path(args.checkpoint or run_dir / "best.pt")
    output_dir = Path(args.output_dir or run_dir / "scanpy_umap")

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
    if single_organ_only:
        print(
            f"[info] single-organ-only filter removed {dataset.num_filtered} "
            f"{args.split} samples"
        )

    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = load_model(checkpoint, dataset, device, organ_classes)
    payload = collect_embeddings(model, loader, device)

    adata = build_anndata(sc, payload, args.embedding, organ_classes)
    add_scanpy_embeddings(sc, adata, args)

    stem = f"{args.split}_{args.embedding}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_umap_csv(output_dir / f"{stem}_umap.csv", adata)
    save_cluster_summary(output_dir / f"{stem}_leiden_summary.csv", adata)
    save_plots(
        sc,
        adata,
        output_dir,
        stem,
        ["target_organ", "pred_organ", "correct", "num_target_organs", "leiden"],
    )

    h5ad_path = output_dir / f"{stem}.h5ad"
    adata.write(h5ad_path, compression="gzip")
    print(f"[saved] AnnData -> {h5ad_path}")


if __name__ == "__main__":
    main()
