from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.stage3_temporal.bilstm import BiLSTMTemporalModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensors-dir", required=True, help="Directory with train/val feature and label .npy files.")
    parser.add_argument("--output", required=True, help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--class-balanced-loss", action="store_true", help="Use class weights from training labels.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="CrossEntropy label smoothing in [0,1).")
    args = parser.parse_args()

    tensors_dir = Path(args.tensors_dir)
    x_train = torch.from_numpy(np.load(tensors_dir / "train_features.npy")).float()
    y_train = torch.from_numpy(np.load(tensors_dir / "train_labels.npy")).long()
    x_val = torch.from_numpy(np.load(tensors_dir / "val_features.npy")).float()
    y_val = torch.from_numpy(np.load(tensors_dir / "val_labels.npy")).long()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = BiLSTMTemporalModel(input_dim=x_train.shape[-1]).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=max(1, args.epochs))
    loss_fn = _build_loss(y_train, args.class_balanced_loss, args.label_smoothing, device)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict] = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optim.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs["emotion_logits"], batch_y)
            loss.backward()
            optim.step()
            train_loss += float(loss.item())
            train_count += 1
        sched.step()

        val_loss = _evaluate(model, val_loader, loss_fn, device)
        train_avg = train_loss / max(1, train_count)
        history.append({"epoch": epoch + 1, "train_loss": train_avg, "val_loss": val_loss})
        print(f"epoch={epoch + 1} train_loss={train_avg:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = model.state_dict()
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state or model.state_dict(),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "history": history,
            "class_balanced_loss": bool(args.class_balanced_loss),
            "label_smoothing": float(args.label_smoothing),
        },
        args.output,
    )
    (Path(args.output).with_suffix(".metrics.json")).write_text(
        json.dumps(
            {
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "epochs_ran": len(history),
                "last": history[-1] if history else None,
                "class_balanced_loss": bool(args.class_balanced_loss),
                "label_smoothing": float(args.label_smoothing),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            total += float(loss_fn(outputs["emotion_logits"], batch_y).item())
            count += 1
    return total / max(1, count)


def _build_loss(y_train: torch.Tensor, class_balanced_loss: bool, label_smoothing: float, device: torch.device) -> nn.Module:
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("--label-smoothing must be in [0,1).")
    if not class_balanced_loss:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    labels = y_train.cpu().numpy().astype(np.int64)
    classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    inv = 1.0 / counts
    weights = inv / np.sum(inv) * classes
    weights_t = torch.from_numpy(weights).float().to(device)
    return nn.CrossEntropyLoss(weight=weights_t, label_smoothing=label_smoothing)


if __name__ == "__main__":
    main()
