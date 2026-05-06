from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from models.philosophy_module.stage5_bayesian.bayesian_combiner import BayesPhiloNet


@dataclass
class Stage5InferenceConfig:
    checkpoint_path: str | None
    device: str = "cpu"
    mc_passes: int = 50


class Stage5InferenceRunner:
    def __init__(self, config: Stage5InferenceConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = BayesPhiloNet().to(self.device)
        self.checkpoint_loaded = self._maybe_load_checkpoint(config.checkpoint_path)

    def _maybe_load_checkpoint(self, checkpoint_path: str | None) -> bool:
        if not checkpoint_path:
            return False
        path = Path(checkpoint_path)
        if not path.exists():
            return False
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        return True

    def predict_with_uncertainty(self, phi: np.ndarray) -> dict[str, float]:
        phi_tensor = torch.from_numpy(phi.astype(np.float32)).reshape(1, -1).to(self.device)

        if not self.checkpoint_loaded:
            ers = float(np.clip(0.28 * phi[0] + 0.34 * phi[1] + 0.18 * phi[2] + 0.20 * phi[3], 0.0, 1.0))
            uncertainty = 0.12
            return {
                "ERS": round(ers, 3),
                "ERS_uncertainty": uncertainty,
                "ERS_ci_low": round(max(0.0, ers - 1.64 * uncertainty), 3),
                "ERS_ci_high": round(min(1.0, ers + 1.64 * uncertainty), 3),
                "confidence_flag": uncertainty < 0.15,
            }

        # Keep BatchNorm layers in eval mode for single-sample inference,
        # but activate dropout layers to sample MC uncertainty.
        self.model.eval()
        _enable_dropout_layers(self.model)
        with torch.no_grad():
            samples = torch.stack([self.model(phi_tensor).squeeze(-1) for _ in range(self.config.mc_passes)]).cpu().numpy()
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        return {
            "ERS": round(mean, 3),
            "ERS_uncertainty": round(std, 3),
            "ERS_ci_low": round(float(np.quantile(samples, 0.05)), 3),
            "ERS_ci_high": round(float(np.quantile(samples, 0.95)), 3),
            "confidence_flag": std < 0.15,
        }


def _enable_dropout_layers(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Dropout):
            submodule.train()
