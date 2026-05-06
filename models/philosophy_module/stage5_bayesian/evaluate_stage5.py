from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.philosophy_module.stage5_bayesian.bayesian_combiner import BayesPhiloNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to test features .npy [N,4].")
    parser.add_argument("--labels", required=True, help="Path to test labels .npy [N].")
    parser.add_argument("--checkpoint", required=True, help="Path to stage5 checkpoint.")
    parser.add_argument("--output", required=True, help="Path to JSON report.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mc-passes", type=int, default=50)
    args = parser.parse_args()

    x = np.load(args.features).astype(np.float32)
    y = np.load(args.labels).astype(np.float32)
    if len(x) == 0:
        raise RuntimeError("No test samples found for Stage 5 evaluation.")

    device = torch.device(args.device)
    model = BayesPhiloNet().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    preds_mean, preds_std = mc_predict(model, x, device, args.mc_passes)
    mae = float(np.mean(np.abs(preds_mean - y)))
    rmse = float(np.sqrt(np.mean((preds_mean - y) ** 2)))
    pearson = float(np.corrcoef(preds_mean, y)[0, 1]) if np.std(preds_mean) > 1e-8 and np.std(y) > 1e-8 else 0.0
    covered = ((y >= preds_mean - 1.64 * preds_std) & (y <= preds_mean + 1.64 * preds_std)).astype(np.float32)

    report = {
        "samples": int(len(y)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(pearson, 4),
        "mean_uncertainty": round(float(np.mean(preds_std)), 4),
        "ci90_coverage": round(float(np.mean(covered)), 4),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def mc_predict(model: BayesPhiloNet, x: np.ndarray, device: torch.device, mc_passes: int) -> tuple[np.ndarray, np.ndarray]:
    x_t = torch.from_numpy(x).to(device)
    model.train()
    with torch.no_grad():
        samples = torch.stack([model(x_t).squeeze(-1) for _ in range(mc_passes)], dim=0).cpu().numpy()
    return samples.mean(axis=0), samples.std(axis=0)


if __name__ == "__main__":
    main()
