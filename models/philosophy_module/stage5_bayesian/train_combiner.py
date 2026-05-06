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

from models.philosophy_module.stage5_bayesian.bayesian_combiner import BayesPhiloNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to train .npy file of shape [N,4].")
    parser.add_argument("--labels", required=True, help="Path to train .npy file of shape [N].")
    parser.add_argument("--val-features", default=None, help="Optional validation features .npy file.")
    parser.add_argument("--val-labels", default=None, help="Optional validation labels .npy file.")
    parser.add_argument("--output", required=True, help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    x = torch.from_numpy(np.load(args.features)).float()
    y = torch.from_numpy(np.load(args.labels)).float().unsqueeze(-1)
    dataset = TensorDataset(x, y)
    # Drop incomplete final training batch so BatchNorm never sees batch size 1.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_loader = None
    if args.val_features and args.val_labels:
        x_val = torch.from_numpy(np.load(args.val_features)).float()
        y_val = torch.from_numpy(np.load(args.val_labels)).float().unsqueeze(-1)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = BayesPhiloNet().to(device)
    optim = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs, 1))
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience = 15
    stale = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optim.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
        sched.step()
        train_loss = epoch_loss / max(len(loader), 1)
        score_loss = _evaluate(model, val_loader, loss_fn, device) if val_loader else train_loss
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": score_loss})
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={score_loss:.4f}")

        if score_loss + 1e-4 < best_loss:
            best_loss = score_loss
            best_state = model.state_dict()
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state or model.state_dict(),
            "best_val_loss": best_loss,
            "best_epoch": best_epoch,
            "history": history,
        },
        output_path,
    )
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(
            {
                "best_val_loss": best_loss,
                "best_epoch": best_epoch,
                "epochs_ran": len(history),
                "last": history[-1] if history else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _evaluate(model: nn.Module, loader: DataLoader | None, loss_fn: nn.Module, device: torch.device) -> float:
    if loader is None:
        return float("inf")
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            total += float(loss_fn(pred, batch_y).item())
            count += 1
    return total / max(count, 1)


if __name__ == "__main__":
    main()
