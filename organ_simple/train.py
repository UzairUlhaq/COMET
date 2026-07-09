import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from constants import COMPONENT_TYPES, ORGAN_CLASSES
from data import LNPEmbeddingDataset
from model import SimpleLNPTransformer, organ_loss


def parse_organ_classes(value):
    if value == "all":
        return ORGAN_CLASSES
    classes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in classes if item not in ORGAN_CLASSES]
    if unknown:
        raise ValueError(f"unknown organ classes: {', '.join(unknown)}")
    if not classes:
        raise ValueError("at least one organ class is required")
    return classes


def accuracy(logits, target):
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


def run_epoch(model, loader, optimizer, device, num_classes, hard_labels=False):
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    all_pred, all_true = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "lnp_id"}
        logits, _ = model(
            batch["component_embeddings"],
            batch["percents"],
            batch["component_types"],
            batch["mask"],
        )
        loss = organ_loss(logits, batch["target"], hard_labels=hard_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        n = batch["target"].shape[0]
        total_loss += loss.item() * n
        total_acc += accuracy(logits.detach(), batch["target"]) * n
        all_pred.append(logits.detach().argmax(dim=-1).cpu())
        all_true.append(batch["target"].argmax(dim=-1).cpu())
        total_n += n

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    return {
        "loss": total_loss / total_n,
        "top1": total_acc / total_n,
        "macro_f1": macro_f1_from_labels(pred, true, num_classes),
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes, hard_labels=False):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    all_pred, all_true = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "lnp_id"}
        logits, _ = model(
            batch["component_embeddings"],
            batch["percents"],
            batch["component_types"],
            batch["mask"],
        )
        loss = organ_loss(logits, batch["target"], hard_labels=hard_labels)

        n = batch["target"].shape[0]
        total_loss += loss.item() * n
        total_acc += accuracy(logits, batch["target"]) * n
        all_pred.append(logits.argmax(dim=-1).cpu())
        all_true.append(batch["target"].argmax(dim=-1).cpu())
        total_n += n

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    return {
        "loss": total_loss / total_n,
        "top1": total_acc / total_n,
        "macro_f1": macro_f1_from_labels(pred, true, num_classes),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cache-dir", default="organ_simple/cache")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--single-organ-only",
        action="store_true",
        help="drop samples whose target distribution has more than one nonzero organ",
    )
    parser.add_argument(
        "--organ-classes",
        default="all",
        help=(
            "comma-separated subset of organ classes to train on, e.g. "
            "lung_epithelium,liver,muscle"
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    organ_classes = parse_organ_classes(args.organ_classes)
    class_indices = [ORGAN_CLASSES.index(name) for name in organ_classes]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    fold_dir = Path(args.cache_dir) / f"fold_V{args.fold}"
    run_dir = Path(args.run_dir or f"organ_simple/runs/fold_V{args.fold}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set = LNPEmbeddingDataset(
        fold_dir / "train.npz",
        single_organ_only=args.single_organ_only,
        class_indices=class_indices,
    )
    valid_set = LNPEmbeddingDataset(
        fold_dir / "valid.npz",
        single_organ_only=args.single_organ_only,
        class_indices=class_indices,
    )
    test_set = LNPEmbeddingDataset(
        fold_dir / "test.npz",
        single_organ_only=args.single_organ_only,
        class_indices=class_indices,
    )
    print(f"organ classes: {', '.join(organ_classes)}")
    print(
        "dataset filter removed "
        f"train={train_set.num_filtered}, "
        f"valid={valid_set.num_filtered}, "
        f"test={test_set.num_filtered} samples"
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = SimpleLNPTransformer(
        component_embedding_dim=train_set.component_embeddings.shape[-1],
        num_component_types=len(COMPONENT_TYPES),
        num_classes=len(organ_classes),
        embed_dim=args.embed_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_valid, bad_epochs = -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            len(organ_classes),
            hard_labels=args.single_organ_only,
        )
        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            len(organ_classes),
            hard_labels=args.single_organ_only,
        )
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} valid_top1={valid_metrics['top1']:.4f} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}"
        )

        if valid_metrics["top1"] > best_valid:
            best_valid = valid_metrics["top1"]
            bad_epochs = 0
            saved_args = vars(args).copy()
            saved_args["organ_classes"] = organ_classes
            saved_args["class_indices"] = class_indices
            torch.save(
                {"model": model.state_dict(), "args": saved_args, "valid": valid_metrics},
                run_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        len(organ_classes),
        hard_labels=args.single_organ_only,
    )
    metrics = {
        "best_valid_top1": best_valid,
        "loss_name": "cross_entropy" if args.single_organ_only else "soft_cross_entropy",
        "organ_classes": organ_classes,
        "test": test_metrics,
        "history": history,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"best checkpoint: {run_dir / 'best.pt'}")
    print(
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_top1={test_metrics['top1']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
