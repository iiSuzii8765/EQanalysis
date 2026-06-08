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


class BaselineRegressionHead(nn.Module):
    """Deterministic two-layer regression baseline (Baseline A).

    Used in the ablation study with ablation_mode='no_philosophy' where the
    input is the 12-dim raw Stage 3 feature vector rather than phi from Stage 4.
    No dropout — this is intentionally the non-Bayesian, uncertainty-free baseline.
    """

    def __init__(self, input_dim: int = 12) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    output_path: Path,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> dict:
    """Train BaselineRegressionHead and save the best checkpoint.

    Returns the metrics dict that was written to ``output_path.with_suffix('.metrics.json')``.
    """
    if device is None:
        device = torch.device("cpu")

    input_dim = x_train.shape[-1]
    model = BaselineRegressionHead(input_dim=input_dim).to(device)
    optim = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = CosineAnnealingLR(optim, T_max=max(epochs, 1))
    loss_fn = nn.MSELoss()

    x_t = torch.from_numpy(x_train).float()
    y_t = torch.from_numpy(y_train).float().unsqueeze(-1)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)

    val_loader: DataLoader | None = None
    if x_val is not None and y_val is not None:
        x_v = torch.from_numpy(x_val).float()
        y_v = torch.from_numpy(y_val).float().unsqueeze(-1)
        val_loader = DataLoader(TensorDataset(x_v, y_v), batch_size=batch_size, shuffle=False)

    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience = 15
    stale = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
        sched.step()

        train_loss = epoch_loss / max(len(loader), 1)
        score_loss = _val_loss(model, val_loader, loss_fn, device) if val_loader else train_loss
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state or model.state_dict(),
            "input_dim": input_dim,
            "best_val_loss": best_loss,
            "best_epoch": best_epoch,
            "history": history,
        },
        output_path,
    )
    metrics = {
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "last": history[-1] if history else None,
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(
    x_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device | None = None,
) -> dict:
    """Run deterministic inference and write evaluation report.

    No MC passes — BaselineRegressionHead is the deterministic Baseline A.
    """
    if device is None:
        device = torch.device("cpu")

    state = torch.load(checkpoint_path, map_location=device)
    input_dim = state.get("input_dim", x_test.shape[-1])
    model = BaselineRegressionHead(input_dim=input_dim).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()

    x_t = torch.from_numpy(x_test).float().to(device)
    with torch.no_grad():
        preds = model(x_t).squeeze(-1).cpu().numpy()

    mae = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    if np.std(preds) > 1e-8 and np.std(y_test) > 1e-8:
        pearson = float(np.corrcoef(preds, y_test)[0, 1])
    else:
        pearson = 0.0

    report = {
        "ablation": "baseline_a",
        "model": "BaselineRegressionHead",
        "samples": int(len(y_test)),
        "input_dim": int(input_dim),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(pearson, 4),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _val_loss(
    model: nn.Module,
    loader: DataLoader | None,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    if loader is None:
        return float("inf")
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            total += float(loss_fn(model(batch_x.to(device)), batch_y.to(device)).item())
            count += 1
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline A regression head for Stage 5 ablation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- train subcommand --
    tr = sub.add_parser("train", help="Train BaselineRegressionHead.")
    tr.add_argument("--features", required=True, help="Train features .npy [N, D].")
    tr.add_argument("--labels", required=True, help="Train labels .npy [N].")
    tr.add_argument("--val-features", default=None, help="Validation features .npy.")
    tr.add_argument("--val-labels", default=None, help="Validation labels .npy.")
    tr.add_argument("--output", required=True, help="Checkpoint output path (.pt).")
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--device", default="cpu")

    # -- eval subcommand --
    ev = sub.add_parser("eval", help="Evaluate BaselineRegressionHead.")
    ev.add_argument("--features", required=True, help="Test features .npy [N, D].")
    ev.add_argument("--labels", required=True, help="Test labels .npy [N].")
    ev.add_argument("--checkpoint", required=True, help="Checkpoint path (.pt).")
    ev.add_argument(
        "--output",
        default="artifacts/ablation/baseline_a_eval.json",
        help="JSON report output path.",
    )
    ev.add_argument("--device", default="cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    if args.cmd == "train":
        x_val = np.load(args.val_features).astype(np.float32) if args.val_features else None
        y_val = np.load(args.val_labels).astype(np.float32) if args.val_labels else None
        train_baseline(
            x_train=np.load(args.features).astype(np.float32),
            y_train=np.load(args.labels).astype(np.float32),
            x_val=x_val,
            y_val=y_val,
            output_path=Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        evaluate_baseline(
            x_test=np.load(args.features).astype(np.float32),
            y_test=np.load(args.labels).astype(np.float32),
            checkpoint_path=Path(args.checkpoint),
            output_path=Path(args.output),
            device=device,
        )


if __name__ == "__main__":
    main()
