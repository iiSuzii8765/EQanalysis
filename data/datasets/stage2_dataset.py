from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights


@dataclass
class Stage2Record:
    image_path: str
    emotion: int
    valence: float
    arousal: float
    aus: np.ndarray


def load_stage2_manifest(manifest_path: str) -> list[Stage2Record]:
    df = pd.read_csv(manifest_path)
    required_cols = {"image_path", "emotion", "valence", "arousal"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in manifest: {sorted(missing)}")

    au_cols = [c for c in df.columns if c.startswith("AU_")]
    if not au_cols:
        raise ValueError("Manifest must include AU columns named AU_00 ... AU_24")

    records: list[Stage2Record] = []
    for _, row in df.iterrows():
        records.append(
            Stage2Record(
                image_path=str(row["image_path"]),
                emotion=int(row["emotion"]),
                valence=float(row["valence"]),
                arousal=float(row["arousal"]),
                aus=np.array([float(row[c]) for c in au_cols], dtype=np.float32),
            )
        )
    return records


class Stage2ImageDataset(Dataset):
    def __init__(self, splits_dir: str, split: str, augment: bool = False) -> None:
        base = Path(splits_dir)
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        split_csv = base / f"{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        self.df = pd.read_csv(split_csv)
        required_cols = {"image_path", "emotion", "valence", "arousal"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Split CSV missing required columns: {sorted(missing)}")

        self.au_cols = [c for c in self.df.columns if c.startswith("AU_")]
        if len(self.au_cols) != 25:
            raise ValueError("Split CSV must include exactly 25 AU columns AU_00..AU_24.")
        self._transform = ResNet50_Weights.DEFAULT.transforms()
        self._augment = augment

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        image_path = Path(str(row["image_path"]))
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._augment:
            frame = _augment_frame(frame)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_tensor = self._transform(frame_tensor)

        emotion = torch.tensor(int(row["emotion"]), dtype=torch.long)
        va = torch.tensor([float(row["valence"]), float(row["arousal"])], dtype=torch.float32)
        au = torch.tensor([float(row[col]) for col in self.au_cols], dtype=torch.float32)
        return frame_tensor, emotion, va, au


def _augment_frame(frame_rgb: np.ndarray) -> np.ndarray:
    out = frame_rgb

    # Random horizontal flip.
    if np.random.rand() < 0.5:
        out = np.ascontiguousarray(out[:, ::-1, :])

    # Mild random crop + resize to inject translation/scale robustness.
    h, w, _ = out.shape
    crop_scale = np.random.uniform(0.90, 1.00)
    ch = max(32, int(h * crop_scale))
    cw = max(32, int(w * crop_scale))
    y0 = np.random.randint(0, max(1, h - ch + 1))
    x0 = np.random.randint(0, max(1, w - cw + 1))
    out = out[y0 : y0 + ch, x0 : x0 + cw]
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    # Brightness/contrast jitter.
    alpha = float(np.random.uniform(0.9, 1.1))  # contrast
    beta = float(np.random.uniform(-10.0, 10.0))  # brightness
    out = np.clip(out.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)
    return out
