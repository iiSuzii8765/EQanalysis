from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from models.stage3_temporal.bilstm import BiLSTMTemporalModel


@dataclass
class Stage3InferenceConfig:
    checkpoint_path: str | None
    device: str = "cpu"
    input_dim: int = 35


class Stage3InferenceRunner:
    def __init__(self, config: Stage3InferenceConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = BiLSTMTemporalModel(input_dim=config.input_dim).to(self.device)
        self.model.eval()
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

    def infer_window(
        self,
        stage2_features: np.ndarray,
        stage2_emotions: np.ndarray,
        au_matrix: np.ndarray,
    ) -> dict[str, np.ndarray | float | bool]:
        x = torch.from_numpy(stage2_features[None, ...]).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            emotion_probs = torch.softmax(outputs["emotion_logits"], dim=-1).cpu().numpy()[0]
            temporal_features = outputs["temporal_features"].cpu().numpy()[0]
            attention = outputs["attention"].cpu().numpy()[0]

        if not self.checkpoint_loaded:
            return _heuristic_temporal_outputs(stage2_emotions, au_matrix, attention)

        dominant_sequence = np.argmax(stage2_emotions, axis=1)
        transitions = float(np.sum(dominant_sequence[1:] != dominant_sequence[:-1]))
        transition_rate = transitions / max(len(dominant_sequence) / 25.0, 1e-3)
        suppression_idx = float(np.mean(np.mean(np.abs(np.diff(au_matrix, axis=0)), axis=1) < 0.05))
        microexpr = _has_microexpression(au_matrix)
        return {
            "sequence_emotion_probs": emotion_probs.astype(np.float32),
            "transition_rate": float(transition_rate),
            "suppression_idx": suppression_idx,
            "microexpression": microexpr,
            "attention_weights": attention.astype(np.float32),
            "valence_slope": float(temporal_features[0]),
            "arousal_mean": float(np.clip((temporal_features[1] + 1.0) / 2.0, 0.0, 1.0)),
            "au_freeze_ratio": float(np.clip((temporal_features[2] + 1.0) / 2.0, 0.0, 1.0)),
            "expr_variability": float(np.clip((temporal_features[3] + 1.0) / 2.0, 0.0, 1.0)),
            "checkpoint_loaded": True,
        }


def _heuristic_temporal_outputs(stage2_emotions: np.ndarray, au_matrix: np.ndarray, attention: np.ndarray) -> dict[str, np.ndarray | float | bool]:
    dominant_sequence = np.argmax(stage2_emotions, axis=1)
    transitions = float(np.sum(dominant_sequence[1:] != dominant_sequence[:-1]))
    transition_rate = transitions / max(len(dominant_sequence) / 25.0, 1e-3)
    au_delta = np.abs(np.diff(au_matrix, axis=0)) if len(au_matrix) > 1 else np.zeros_like(au_matrix)
    suppression_idx = float(np.mean(np.mean(au_delta, axis=1) < 0.05)) if len(au_delta) else 0.0
    variability = float(np.mean(np.std(stage2_emotions, axis=0)))
    arousal_cols = [idx for idx in [3, 4, 16] if idx < au_matrix.shape[1]]
    arousal_mean = float(np.clip(np.mean(au_matrix[:, arousal_cols]) / 2.0, 0.0, 1.0)) if arousal_cols else 0.0
    return {
        "sequence_emotion_probs": stage2_emotions.mean(axis=0).astype(np.float32),
        "transition_rate": float(transition_rate),
        "suppression_idx": float(suppression_idx),
        "microexpression": _has_microexpression(au_matrix),
        "attention_weights": attention.astype(np.float32),
        "valence_slope": float(np.polyfit(np.arange(len(stage2_emotions)), stage2_emotions[:, 3] - stage2_emotions[:, 4], 1)[0]) if len(stage2_emotions) > 1 else 0.0,
        "arousal_mean": arousal_mean,
        "au_freeze_ratio": float(suppression_idx),
        "expr_variability": variability,
        "checkpoint_loaded": False,
    }


def _has_microexpression(au_matrix: np.ndarray, fps: int = 25) -> bool:
    threshold = 0.8
    max_frames = int(0.5 * fps)
    for col in range(au_matrix.shape[1]):
        above = au_matrix[:, col] > threshold
        run = 0
        for flag in above:
            run = run + 1 if flag else 0
            if 0 < run <= max_frames:
                return True
    return False
