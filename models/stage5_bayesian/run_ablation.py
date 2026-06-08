"""run_ablation.py — end-to-end Stage 5 ablation study across four conditions.

Run from the project root:
    python models/stage5_bayesian/run_ablation.py

Results are written to artifacts/ablation/ablation_results.json and a
comparison table is printed to stdout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from data.preprocessing.build_stage5_dataset import build_phi_and_targets
from models.philosophy_module.stage5_bayesian.bayesian_combiner import BayesPhiloNet
from models.stage5_bayesian.baseline_head import BaselineRegressionHead

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Training hyperparameters — identical to train_combiner.py ────────────────
EPOCHS = 200
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 15
MC_PASSES = 50

# ── Paths ─────────────────────────────────────────────────────────────────────
TENSORS_DIR = Path("artifacts/stage3_training/tensors")
OUTPUT_DIR = Path("artifacts/ablation")


class ConditionResult(TypedDict):
    pearson_r: float
    mae: float
    rmse: float


# ── Utilities ─────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    label: str = "",
) -> nn.Module:
    """Train model in-memory; returns model loaded with best-val-loss weights."""
    model = model.to(device)
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(optim, T_max=max(EPOCHS, 1))
    loss_fn = nn.MSELoss()

    x_t = torch.from_numpy(x_train).float()
    y_t = torch.from_numpy(y_train).float().unsqueeze(-1)
    # drop_last keeps BayesPhiloNet's BatchNorm from seeing a singleton batch
    loader = DataLoader(
        TensorDataset(x_t, y_t), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    x_v = torch.from_numpy(x_val).float().to(device)
    y_v = torch.from_numpy(y_val).float().unsqueeze(-1).to(device)

    best_loss = float("inf")
    best_state: dict | None = None
    stale = 0
    prefix = f"[{label}] " if label else ""

    for epoch in range(EPOCHS):
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

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_v), y_v).item())

        print(
            f"{prefix}epoch={epoch + 1:>3}/{EPOCHS}"
            f"  train={train_loss:.4f}  val={val_loss:.4f}"
        )

        if val_loss + 1e-4 < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"{prefix}early stop at epoch {epoch + 1} (patience={PATIENCE})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _eval_mc(
    model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device
) -> ConditionResult:
    """MC dropout inference — model.train() keeps dropout active."""
    x_t = torch.from_numpy(x).float().to(device)
    model.train()
    with torch.no_grad():
        passes = torch.stack(
            [model(x_t).squeeze(-1) for _ in range(MC_PASSES)], dim=0
        )
    preds = passes.mean(dim=0).cpu().numpy()
    return _compute_metrics(preds, y)


def _eval_deterministic(
    model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device
) -> ConditionResult:
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(x).float().to(device)).squeeze(-1).cpu().numpy()
    return _compute_metrics(preds, y)


def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> ConditionResult:
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    if np.std(preds) > 1e-8 and np.std(targets) > 1e-8:
        pearson = float(np.corrcoef(preds, targets)[0, 1])
    else:
        pearson = 0.0
    return {
        "pearson_r": round(pearson, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading Stage 3 tensors from {TENSORS_DIR} ...")
    train_seq = np.load(TENSORS_DIR / "train_features.npy").astype(np.float32)
    val_seq   = np.load(TENSORS_DIR / "val_features.npy").astype(np.float32)
    test_seq  = np.load(TENSORS_DIR / "test_features.npy").astype(np.float32)
    print(f"  train={train_seq.shape}  val={val_seq.shape}  test={test_seq.shape}")

    # Condition name → key used in the output JSON
    conditions: list[tuple[str, str]] = [
        ("full",         "full_model"),
        ("no_f2",        "no_f2"),
        ("linear_only",  "linear_only"),
        ("no_philosophy","no_philosophy"),
    ]

    output: dict[str, ConditionResult] = {}

    for ablation_mode, result_key in conditions:
        print(f"\n{'=' * 62}")
        print(f"  Condition: {result_key}")
        print(f"{'=' * 62}")

        # Reset seed before every condition so initialisation is comparable.
        _seed_everything(SEED)

        if ablation_mode == "linear_only":
            # No training — just evaluate the equal-weight linear combiner on
            # the test phi vector.  Phi is [F1, F2, (1-F3), F4]; sum * 0.25.
            x_test, y_test = build_phi_and_targets(test_seq, ablation_mode="linear_only")
            preds = 0.25 * x_test.sum(axis=1)
            metrics = _compute_metrics(preds, y_test)

        else:
            x_train, y_train = build_phi_and_targets(train_seq, ablation_mode=ablation_mode)
            x_val,   y_val   = build_phi_and_targets(val_seq,   ablation_mode=ablation_mode)
            x_test,  y_test  = build_phi_and_targets(test_seq,  ablation_mode=ablation_mode)
            print(
                f"  features: train={x_train.shape}  val={x_val.shape}  test={x_test.shape}"
            )

            if ablation_mode == "no_philosophy":
                model: nn.Module = BaselineRegressionHead(input_dim=x_train.shape[-1])
                model = _train(model, x_train, y_train, x_val, y_val, device, label=result_key)
                metrics = _eval_deterministic(model, x_test, y_test, device)
            else:
                # "full" or "no_f2" — BayesPhiloNet with MC dropout inference
                model = BayesPhiloNet()
                model = _train(model, x_train, y_train, x_val, y_val, device, label=result_key)
                metrics = _eval_mc(model, x_test, y_test, device)

        output[result_key] = metrics
        print(
            f"\n  -> Pearson r={metrics['pearson_r']:.4f}"
            f"  MAE={metrics['mae']:.4f}"
            f"  RMSE={metrics['rmse']:.4f}"
        )

    # ── Persist ───────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "ablation_results.json"
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_path}")

    # ── Comparison table ──────────────────────────────────────────────────────
    _print_table(output)


def _print_table(results: dict[str, ConditionResult]) -> None:
    full_r = results["full_model"]["pearson_r"]

    col_w = [20, 11, 9, 9, 13]
    sep = "-" * sum(col_w)

    def _row(cond: str, r: ConditionResult, delta: str) -> str:
        return (
            f"{cond:<{col_w[0]}}"
            f"{r['pearson_r']:>{col_w[1]}.4f}"
            f"{r['mae']:>{col_w[2]}.4f}"
            f"{r['rmse']:>{col_w[3]}.4f}"
            f"{delta:>{col_w[4]}}"
        )

    header = (
        f"{'Condition':<{col_w[0]}}"
        f"{'Pearson r':>{col_w[1]}}"
        f"{'MAE':>{col_w[2]}}"
        f"{'RMSE':>{col_w[3]}}"
        f"{'ΔPearson':>{col_w[4]}}"
    )

    print(f"\n{sep}")
    print("Stage 5 Ablation Study — Test-Set Results")
    print(sep)
    print(header)
    print(sep)

    display_order = [
        ("full_model",    "full_model"),
        ("no_f2",         "no_f2"),
        ("linear_only",   "linear_only"),
        ("no_philosophy", "no_philosophy"),
    ]
    for key, label in display_order:
        r = results[key]
        if key == "full_model":
            delta_str = "—"
        else:
            delta = r["pearson_r"] - full_r
            delta_str = f"{delta:+.4f}"
        print(_row(label, r, delta_str))

    print(sep)


if __name__ == "__main__":
    main()
