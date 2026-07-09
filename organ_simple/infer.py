import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from constants import COMPONENT_TYPES, ORGAN_CLASSES
from data import LNPEmbeddingDataset
from model import SimpleLNPTransformer, organ_loss


def class_indices_for(organ_classes):
    return [ORGAN_CLASSES.index(name) for name in organ_classes]


def top1_accuracy(logits, target):
    return (logits.argmax(dim=-1) == target.argmax(dim=-1)).float().mean().item()


def macro_f1_from_labels(pred, true, num_classes):
    scores = []
    for cls in range(num_classes):
        tp = ((pred == cls) & (true == cls)).sum().float()
        fp = ((pred == cls) & (true != cls)).sum().float()
        fn = ((pred != cls) & (true == cls)).sum().float()
        denom = (2 * tp) + fp + fn
        if denom > 0:
            scores.append(((2 * tp) / denom).item())
    return sum(scores) / len(scores) if scores else 0.0


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


@torch.no_grad()
def run_inference(model, loader, device, organ_classes, hard_labels=False):
    rows = []
    total_loss, total_top1, total_n = 0.0, 0.0, 0
    all_pred, all_true = [], []

    for batch in loader:
        lnp_ids = batch["lnp_id"]
        batch = {k: v.to(device) for k, v in batch.items() if k != "lnp_id"}

        logits, _ = model(
            batch["component_embeddings"],
            batch["percents"],
            batch["component_types"],
            batch["mask"],
        )
        probs = torch.softmax(logits, dim=-1)
        loss = organ_loss(logits, batch["target"], hard_labels=hard_labels)

        n = batch["target"].shape[0]
        total_loss += loss.item() * n
        total_top1 += top1_accuracy(logits, batch["target"]) * n
        total_n += n

        pred_idx = probs.argmax(dim=-1).cpu()
        true_idx = batch["target"].argmax(dim=-1).cpu()
        all_pred.append(pred_idx)
        all_true.append(true_idx)
        probs = probs.cpu()
        targets = batch["target"].cpu()

        for i, lnp_id in enumerate(lnp_ids):
            row = {
                "lnp_id": lnp_id,
                "true_organ": organ_classes[true_idx[i].item()],
                "pred_organ": organ_classes[pred_idx[i].item()],
                "correct": int(pred_idx[i].item() == true_idx[i].item()),
            }
            for organ, value in zip(organ_classes, targets[i].tolist()):
                row[f"target_{organ}"] = value
            for organ, value in zip(organ_classes, probs[i].tolist()):
                row[f"prob_{organ}"] = value
            rows.append(row)

    metrics = {
        "loss": total_loss / total_n,
        "top1": total_top1 / total_n,
        "macro_f1": macro_f1_from_labels(
            torch.cat(all_pred), torch.cat(all_true), len(organ_classes)
        ),
        "num_samples": total_n,
    }
    return rows, metrics


def write_csv(path, rows, organ_classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lnp_id",
        "true_organ",
        "pred_organ",
        "correct",
        *[f"target_{organ}" for organ in organ_classes],
        *[f"prob_{organ}" for organ in organ_classes],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--cache-dir", default="organ_simple/cache")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--single-organ-only",
        action="store_true",
        help="drop samples whose target distribution has more than one nonzero organ",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    fold_dir = Path(args.cache_dir) / f"fold_V{args.fold}"
    run_dir = Path(args.run_dir or f"organ_simple/runs/fold_V{args.fold}")
    checkpoint_path = Path(args.checkpoint or run_dir / "best.pt")
    output_path = Path(args.output or run_dir / f"{args.split}_predictions.csv")

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
            f"single-organ-only filter removed {dataset.num_filtered} "
            f"{args.split} samples"
        )
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = load_model(checkpoint, dataset, device, organ_classes)

    rows, metrics = run_inference(
        model, loader, device, organ_classes, hard_labels=single_organ_only
    )
    metrics["loss_name"] = (
        "cross_entropy" if single_organ_only else "soft_cross_entropy"
    )
    metrics["organ_classes"] = organ_classes
    write_csv(output_path, rows, organ_classes)

    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"wrote predictions: {output_path}")
    print(f"wrote metrics: {metrics_path}")
    print(
        f"{args.split}_loss={metrics['loss']:.4f} "
        f"{args.split}_top1={metrics['top1']:.4f} "
        f"{args.split}_macro_f1={metrics['macro_f1']:.4f} "
        f"num_samples={metrics['num_samples']}"
    )


if __name__ == "__main__":
    main()
