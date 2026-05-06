from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet50_Weights

from models.stage2_spatial.resnet_fer import ResNetFER

MODEL_AU_KEYS = [
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU24_r",
    "AU25_r",
    "AU26_r",
    "AU28_r",
    "AU43_r",
    "AU45_r",
    "AU11_r",
    "AU13_r",
    "AU16_r",
    "AU18_r",
    "AU22_r",
]


@dataclass
class Stage2InferenceConfig:
    checkpoint_path: str | None
    device: str = "cpu"
    batch_size: int = 8


class Stage2InferenceRunner:
    def __init__(self, config: Stage2InferenceConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = ResNetFER(pretrained=False).to(self.device)
        self.model.eval()
        self._weights = ResNet50_Weights.DEFAULT.transforms()
        self._checkpoint_loaded = self._maybe_load_checkpoint(config.checkpoint_path)
        # Public alias for compatibility with callers that check this directly.
        self.checkpoint_loaded = self._checkpoint_loaded

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

    def infer(
        self,
        frames: list[np.ndarray],
        openface_au_matrix: np.ndarray | None = None,
        openface_au_keys: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        if not frames:
            raise RuntimeError("Stage 2 inference received no frames.")

        tensors = [self._weights(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0) for frame in frames]
        emotion_probs: list[np.ndarray] = []
        va_outputs: list[np.ndarray] = []
        au_probs: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(tensors), self.config.batch_size):
                batch = torch.stack(tensors[start : start + self.config.batch_size]).to(self.device)
                outputs = self.model(batch)
                emotion_probs.extend(torch.softmax(outputs["emotion_logits"], dim=-1).cpu().numpy())
                va_outputs.extend(torch.tanh(outputs["va"]).cpu().numpy())
                au_probs.extend(torch.sigmoid(outputs["au_logits"]).cpu().numpy())
                embeddings.extend(outputs["embedding"].cpu().numpy())

        emotion_probs_arr = np.asarray(emotion_probs, dtype=np.float32)
        va_arr = np.asarray(va_outputs, dtype=np.float32)
        au_probs_arr = np.asarray(au_probs, dtype=np.float32)
        embed_arr = np.asarray(embeddings, dtype=np.float32)

        if not self._checkpoint_loaded and openface_au_matrix is not None and len(openface_au_matrix) == len(frames):
            # Fallback while training checkpoints are absent: fuse weak image model outputs with AU-derived priors.
            emotion_probs_arr = _blend_emotion_priors(emotion_probs_arr, openface_au_matrix, openface_au_keys)
            va_arr = _blend_valence_arousal(va_arr, openface_au_matrix, openface_au_keys)
            projected_openface = _project_openface_aus(
                openface_au_matrix=openface_au_matrix,
                openface_au_keys=openface_au_keys,
                output_dim=au_probs_arr.shape[1],
            )
            au_probs_arr = 0.5 * au_probs_arr + 0.5 * projected_openface

        return {
            "emotion_probs": emotion_probs_arr,
            "valence": va_arr[:, 0],
            "arousal": (va_arr[:, 1] + 1.0) / 2.0,
            "au_probs": au_probs_arr,
            "embeddings": embed_arr,
            "checkpoint_loaded": bool(self._checkpoint_loaded),
        }


def _normalize_au_matrix(au_matrix: np.ndarray) -> np.ndarray:
    au = np.asarray(au_matrix, dtype=np.float32)
    max_val = np.maximum(np.max(au, axis=0, keepdims=True), 1.0)
    return np.clip(au / max_val, 0.0, 1.0)


def _project_openface_aus(openface_au_matrix: np.ndarray, openface_au_keys: list[str] | None, output_dim: int) -> np.ndarray:
    au = _normalize_au_matrix(openface_au_matrix)
    if output_dim == au.shape[1]:
        return au

    projected = np.zeros((au.shape[0], output_dim), dtype=np.float32)
    if not openface_au_keys:
        width = min(output_dim, au.shape[1])
        projected[:, :width] = au[:, :width]
        return projected

    source_map = {key: idx for idx, key in enumerate(openface_au_keys)}
    for target_idx, target_key in enumerate(MODEL_AU_KEYS[:output_dim]):
        source_idx = source_map.get(target_key)
        if source_idx is not None:
            projected[:, target_idx] = au[:, source_idx]
    return projected


def _blend_emotion_priors(model_probs: np.ndarray, au_matrix: np.ndarray, au_keys: list[str] | None) -> np.ndarray:
    au = _normalize_au_matrix(au_matrix)
    au_map = _au_map(au, au_keys)
    scores = np.stack(
        [
            _au_col(au_map, "AU04_r") + _au_col(au_map, "AU23_r") + _au_col(au_map, "AU24_r"),
            _au_col(au_map, "AU09_r") + _au_col(au_map, "AU16_r"),
            _au_col(au_map, "AU01_r") + _au_col(au_map, "AU02_r") + _au_col(au_map, "AU05_r"),
            _au_col(au_map, "AU06_r") + _au_col(au_map, "AU12_r"),
            _au_col(au_map, "AU01_r") + _au_col(au_map, "AU04_r") + _au_col(au_map, "AU15_r"),
            _au_col(au_map, "AU01_r") + _au_col(au_map, "AU02_r") + _au_col(au_map, "AU26_r"),
            _au_col(au_map, "AU12_r") + _au_col(au_map, "AU14_r"),
            np.clip(1.0 - np.mean(au, axis=1), 0.0, 1.0),  # neutral-ish
        ],
        axis=1,
    )
    scores = scores / (np.sum(scores, axis=1, keepdims=True) + 1e-6)
    return 0.35 * model_probs + 0.65 * scores


def _blend_valence_arousal(va_arr: np.ndarray, au_matrix: np.ndarray, au_keys: list[str] | None) -> np.ndarray:
    au = _normalize_au_matrix(au_matrix)
    au_map = _au_map(au, au_keys)
    valence = np.clip(
        (_au_col(au_map, "AU06_r") + _au_col(au_map, "AU12_r") - _au_col(au_map, "AU09_r") - _au_col(au_map, "AU15_r") - _au_col(au_map, "AU23_r")) / 2.0,
        -1.0,
        1.0,
    )
    arousal = np.clip(
        (_au_col(au_map, "AU05_r") + _au_col(au_map, "AU26_r") + _au_col(au_map, "AU04_r")) / 2.0,
        0.0,
        1.0,
    )
    blended_valence = 0.3 * va_arr[:, 0] + 0.7 * valence
    blended_arousal = 0.3 * ((va_arr[:, 1] + 1.0) / 2.0) + 0.7 * arousal
    return np.stack([blended_valence, blended_arousal * 2.0 - 1.0], axis=1)


def _au_map(au_matrix: np.ndarray, au_keys: list[str] | None) -> dict[str, np.ndarray]:
    if au_keys is None:
        return {}
    return {key: au_matrix[:, idx] for idx, key in enumerate(au_keys)}


def _au_col(au_map: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key in au_map:
        return au_map[key]
    if au_map:
        length = len(next(iter(au_map.values())))
        return np.zeros(length, dtype=np.float32)
    return np.zeros(1, dtype=np.float32)
