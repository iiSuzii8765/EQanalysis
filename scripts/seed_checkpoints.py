"""
Seed initial model checkpoints for Stage 2 (ResNetFER), Stage 3 (BiLSTM),
and Stage 5 (BayesPhiloNet).

These are NOT trained on real data. They provide meaningful starting points:

  Stage 2 — ImageNet-pretrained ResNet-50 backbone with zero-initialised task
             heads. The backbone already encodes strong visual features; only
             the emotion/VA/AU heads need task-specific fine-tuning.

  Stage 3 — BiLSTM with near-zero output-head initialisation. The LSTM cells
             use PyTorch's default orthogonal init; only the small output heads
             are zeroed so early training follows the heuristic fallback closely.

  Stage 5 — BayesPhiloNet trained on 10 000 synthetic (φ, ERS) pairs using the
             analytic formula:
                 ERS = 0.28·F1 + 0.34·F2 + 0.18·(1-F3) + 0.20·F4
             This ensures the network produces sensible ERS values from the first
             real inference, before any fine-tuning on labelled sessions.

Run once before deploying:
    python scripts/seed_checkpoints.py

Outputs:
    artifacts/checkpoints/stage2_resnet.pt
    artifacts/checkpoints/stage3_bilstm.pt
    artifacts/checkpoints/stage5_bayes.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.philosophy_module.stage5_bayesian.bayesian_combiner import BayesPhiloNet
from models.stage2_spatial.resnet_fer import ResNetFER
from models.stage3_temporal.bilstm import BiLSTMTemporalModel

CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def seed_stage2(path: Path) -> None:
    """
    Save Stage 2 checkpoint with ImageNet-pretrained ResNet-50 backbone.

    Task heads (emotion, VA, AU) are zero-initialised so the network starts
    predicting near-uniform distributions. The backbone's ImageNet features are
    preserved — they already encode edges, textures and object parts that
    transfer well to face analysis.
    """
    print("Seeding Stage 2 (ResNetFER, ImageNet backbone)…")
    model = ResNetFER(pretrained=True)
    for head in [model.emotion_head, model.va_head, model.au_head]:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    torch.save(model.state_dict(), path)
    size_kb = path.stat().st_size / 1024
    print(f"  saved  {path.name}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def seed_stage3(path: Path, input_dim: int = 35) -> None:
    """
    Save Stage 3 checkpoint with near-zero output heads.

    The LSTM cells are left at PyTorch default (orthogonal + zeros) which
    is already a good starting point for sequence modelling. Only the linear
    emotion and feature output heads are zeroed.
    """
    print("Seeding Stage 3 (BiLSTM)…")
    model = BiLSTMTemporalModel(input_dim=input_dim)
    for head in [model.emotion_head, model.feature_head]:
        nn.init.uniform_(head.weight, -0.005, 0.005)
        nn.init.zeros_(head.bias)
    torch.save(model.state_dict(), path)
    size_kb = path.stat().st_size / 1024
    print(f"  saved  {path.name}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------

def seed_stage5(
    path: Path,
    n_synthetic: int = 10_000,
    epochs: int = 400,
    lr: float = 1e-3,
) -> None:
    """
    Train Stage 5 on synthetic data so it reproduces the analytic ERS formula.

    Synthetic targets:
        ERS = clip(0.28·F1 + 0.34·F2 + 0.18·(1-F3) + 0.20·F4, 0, 1)

    The φ vector passed into the model during inference is:
        φ = [F1, F2, 1-F3, F4]
    so the formula simplifies to a weighted sum of φ components.

    Training on this synthetic distribution means the model will produce
    sensible ERS values from the first real session without needing any labelled
    data. Fine-tuning on real supervision will pull the weights toward empirical
    truth from this informed starting point.
    """
    print(f"Seeding Stage 5 (BayesPhiloNet) — {n_synthetic} synthetic samples, {epochs} epochs…")

    rng = np.random.default_rng(42)
    phi = rng.uniform(0.0, 1.0, (n_synthetic, 4)).astype(np.float32)
    # φ = [F1, F2, 1-F3, F4]  — matches how pipeline.py constructs the vector
    ers = np.clip(
        0.28 * phi[:, 0] + 0.34 * phi[:, 1] + 0.18 * phi[:, 2] + 0.20 * phi[:, 3],
        0.0, 1.0,
    ).astype(np.float32)

    X = torch.from_numpy(phi)
    y = torch.from_numpy(ers).unsqueeze(1)

    # Train without dropout so BatchNorm statistics settle correctly.
    model = BayesPhiloNet(dropout_p=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    model.train()
    batch_size = 256
    for epoch in range(epochs):
        perm = torch.randperm(n_synthetic)
        epoch_loss = 0.0
        for start in range(0, n_synthetic, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            pred = model(X[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1:>4}/{epochs}  loss={epoch_loss:.5f}")

    _verify_stage5(model, X, y)
    # Save with dropout_p restored so MC passes work during inference.
    final = BayesPhiloNet(dropout_p=0.3)
    final.load_state_dict(model.state_dict())
    torch.save(final.state_dict(), path)
    size_kb = path.stat().st_size / 1024
    print(f"  saved  {path.name}  ({size_kb:.0f} KB)")


def _verify_stage5(model: BayesPhiloNet, X: torch.Tensor, y: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        pred = model(X).squeeze(1).numpy()
    mae = float(np.mean(np.abs(pred - y.squeeze(1).numpy())))
    status = "PASS" if mae < 0.05 else "WARN — MAE higher than expected"
    print(f"  verification MAE on full synthetic set: {mae:.5f}  [{status}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    stage2_path = CHECKPOINT_DIR / "stage2_resnet.pt"
    stage3_path = CHECKPOINT_DIR / "stage3_bilstm.pt"
    stage5_path = CHECKPOINT_DIR / "stage5_bayes.pt"

    seed_stage2(stage2_path)
    seed_stage3(stage3_path)
    seed_stage5(stage5_path)

    print("\nCheckpoints written:")
    for p in [stage2_path, stage3_path, stage5_path]:
        print(f"  {p.relative_to(ROOT)}")
    print("\nNext step: fine-tune each stage on real labelled data.")
    print("  Stage 2: python -m models.stage2_spatial.run_stage2_pipeline")
    print("  Stage 3: python -m models.stage3_temporal.run_stage3_pipeline")
    print("  Stage 5: python -m models.philosophy_module.stage5_bayesian.run_stage5_pipeline")


if __name__ == "__main__":
    main()
