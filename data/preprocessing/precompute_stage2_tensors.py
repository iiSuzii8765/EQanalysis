from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to stage2 manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory for .npy tensors.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    required = {"image_path", "emotion", "valence", "arousal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    au_cols = [c for c in df.columns if c.startswith("AU_")]
    if len(au_cols) != 25:
        raise ValueError("Manifest must include exactly 25 AU columns AU_00..AU_24.")

    frames, emotion, va, au = _load_arrays(df, au_cols, args.image_size)
    idx_train, idx_val, idx_test = _split_indices(len(frames), args.train_ratio, args.val_ratio, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_split(output_dir, "train", frames, emotion, va, au, idx_train)
    _save_split(output_dir, "val", frames, emotion, va, au, idx_val)
    _save_split(output_dir, "test", frames, emotion, va, au, idx_test)

    au_pos_weight = _compute_au_pos_weight(au[idx_train])
    np.save(output_dir / "au_pos_weight.npy", au_pos_weight.astype(np.float32))

    metadata = {
        "samples_total": int(len(frames)),
        "samples_train": int(len(idx_train)),
        "samples_val": int(len(idx_val)),
        "samples_test": int(len(idx_test)),
        "image_size": int(args.image_size),
        "au_columns": au_cols,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved tensors to: {output_dir}")


def _load_arrays(df: pd.DataFrame, au_cols: list[str], image_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_list: list[np.ndarray] = []
    emotion_list: list[int] = []
    va_list: list[list[float]] = []
    au_list: list[np.ndarray] = []

    for _, row in df.iterrows():
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))  # [3,H,W]
        frame_list.append(frame)

        emotion_list.append(int(row["emotion"]))
        va_list.append([float(row["valence"]), float(row["arousal"])])
        au_vec = np.array([float(row[c]) for c in au_cols], dtype=np.float32)
        au_list.append(np.clip(au_vec, 0.0, 5.0) / 5.0)

    if not frame_list:
        raise RuntimeError("No valid image samples could be loaded from manifest.")

    return (
        np.stack(frame_list).astype(np.float32),
        np.asarray(emotion_list, dtype=np.int64),
        np.asarray(va_list, dtype=np.float32),
        np.stack(au_list).astype(np.float32),
    )


def _split_indices(total: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if total < 10:
        raise RuntimeError(f"Need at least 10 samples for split, got {total}.")
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]
    return idx_train, idx_val, idx_test


def _save_split(
    output_dir: Path,
    split: str,
    frames: np.ndarray,
    emotion: np.ndarray,
    va: np.ndarray,
    au: np.ndarray,
    indices: np.ndarray,
) -> None:
    np.save(output_dir / f"{split}_frames.npy", frames[indices])
    np.save(output_dir / f"{split}_emotion.npy", emotion[indices])
    np.save(output_dir / f"{split}_va.npy", va[indices])
    np.save(output_dir / f"{split}_au.npy", au[indices])


def _compute_au_pos_weight(au_train: np.ndarray) -> np.ndarray:
    positive = np.sum(au_train > 0.2, axis=0).astype(np.float32)
    negative = np.maximum(1.0, au_train.shape[0] - positive)
    return np.clip(negative / np.maximum(1.0, positive), 0.5, 20.0)


if __name__ == "__main__":
    main()
