from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to stage2 manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory for train/val/test split CSVs.")
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
    if len(df) < 10:
        raise RuntimeError(f"Need at least 10 rows for split, got {len(df)}.")

    train_df, val_df, test_df = _stratified_split_by_emotion(
        df=df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    au_pos_weight = _compute_au_pos_weight(train_df[au_cols].to_numpy(dtype=np.float32))
    np.save(output_dir / "au_pos_weight.npy", au_pos_weight.astype(np.float32))

    metadata = {
        "samples_total": int(len(df)),
        "samples_train": int(len(train_df)),
        "samples_val": int(len(val_df)),
        "samples_test": int(len(test_df)),
        "au_columns": au_cols,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved split CSVs to: {output_dir}")


def _compute_au_pos_weight(au_train: np.ndarray) -> np.ndarray:
    positive = np.sum(au_train > 0.2, axis=0).astype(np.float32)
    negative = np.maximum(1.0, au_train.shape[0] - positive)
    return np.clip(negative / np.maximum(1.0, positive), 0.5, 20.0)


def _stratified_split_by_emotion(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train-ratio must be in (0,1).")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val-ratio must be in [0,1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be < 1.0.")

    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for emotion_id, group in df.groupby("emotion"):
        idx = np.arange(len(group))
        rng.shuffle(idx)
        g = group.iloc[idx].reset_index(drop=True)
        n = len(g)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, min(n_train, n - 1))
        n_val = max(1, min(n_val, n - n_train - 1)) if n - n_train >= 2 else max(0, n - n_train - 1)

        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train : n_train + n_val])
        test_parts.append(g.iloc[n_train + n_val :])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


if __name__ == "__main__":
    main()
