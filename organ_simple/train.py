import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from constants import COMPONENT_TYPES, ORGAN_CLASSES
from data import LNPEmbeddingDataset
from model import SimpleLNPTransformer, soft_cross_entropy


def accuracy(logits, target):
    return (logits.argmax(dim=-1) == target.argmax(dim=-1)).float().mean().item()


def run_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "lnp_id"}
        logits, _ = model(
            batch["component_embeddings"],
            batch["percents"],
            batch["component_types"],
            batch["mask"],
        )
        loss = soft_cross_entropy(logits, batch["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        n = batch["target"].shape[0]
        total_loss += loss.item() * n
        total_acc += accuracy(logits.detach(), batch["target"]) * n
        total_n += n
    return {"loss": total_loss / total_n, "top1": total_acc / total_n}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "lnp_id"}
        logits, _ = model(
            batch["component_embeddings"],
            batch["percents"],
            batch["component_types"],
            batch["mask"],
        )
        loss = soft_cross_entropy(logits, batch["target"])

        n = batch["target"].shape[0]
        total_loss += loss.item() * n
        total_acc += accuracy(logits, batch["target"]) * n
        total_n += n
    return {"loss": total_loss / total_n, "top1": total_acc / total_n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--cache-dir", default="organ_simple/cache")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    fold_dir = Path(args.cache_dir) / f"fold_V{args.fold}"
    run_dir = Path(args.run_dir or f"organ_simple/runs/fold_V{args.fold}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set = LNPEmbeddingDataset(fold_dir / "train.npz")
    valid_set = LNPEmbeddingDataset(fold_dir / "valid.npz")
    test_set = LNPEmbeddingDataset(fold_dir / "test.npz")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = SimpleLNPTransformer(
        component_embedding_dim=train_set.component_embeddings.shape[-1],
        num_component_types=len(COMPONENT_TYPES),
        num_classes=len(ORGAN_CLASSES),
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
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        valid_metrics = evaluate(model, valid_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} valid_top1={valid_metrics['top1']:.4f}"
        )

        if valid_metrics["top1"] > best_valid:
            best_valid = valid_metrics["top1"]
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "valid": valid_metrics},
                run_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)
    metrics = {"best_valid_top1": best_valid, "test": test_metrics, "history": history}
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"best checkpoint: {run_dir / 'best.pt'}")
    print(f"test_loss={test_metrics['loss']:.4f} test_top1={test_metrics['top1']:.4f}")


if __name__ == "__main__":
    main()
