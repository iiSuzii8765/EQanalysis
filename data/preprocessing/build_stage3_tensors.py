from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from models.stage2_spatial.inference import MODEL_AU_KEYS, Stage2InferenceConfig, Stage2InferenceRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", required=True, help="Directory with train/val/test CSV splits from Stage 2.")
    parser.add_argument("--stage2-checkpoint", required=True, help="Path to trained stage2_resnet.pt checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Output directory for Stage 3 tensors.")
    parser.add_argument("--window-size", type=int, default=75)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256, help="Rows loaded into RAM per inference chunk.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples-per-split", type=int, default=0, help="Optional cap per split (0 = no cap).")
    parser.add_argument(
        "--sequence-mode",
        choices=["grouped_by_emotion", "sliding"],
        default="grouped_by_emotion",
        help="How to create Stage 3 sequences from frame-level Stage 2 features.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage2_runner = Stage2InferenceRunner(
        Stage2InferenceConfig(
            checkpoint_path=args.stage2_checkpoint,
            device=args.device,
            batch_size=args.batch_size,
        )
    )
    if not stage2_runner.checkpoint_loaded:
        raise RuntimeError(
            "Stage 2 checkpoint could not be loaded. "
            f"Expected a valid file at: {args.stage2_checkpoint}"
        )

    rng = np.random.default_rng(args.seed)
    metadata: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        split_csv = splits_dir / f"{split}.csv"
        if not split_csv.exists():
            continue

        df = pd.read_csv(split_csv)
        if args.max_samples_per_split > 0:
            df = df.iloc[: args.max_samples_per_split].copy()
        if df.empty:
            continue

        frame_features, emotions, loaded_frames = _build_frame_features(
            df=df,
            stage2_runner=stage2_runner,
            chunk_size=args.chunk_size,
        )
        if args.sequence_mode == "grouped_by_emotion":
            seq_features, seq_labels = _to_sequences_grouped_by_emotion(
                frame_features=frame_features,
                emotion_labels=emotions,
                window_size=args.window_size,
                stride=args.stride,
                rng=rng,
            )
        else:
            seq_features, seq_labels = _to_sequences_sliding(
                frame_features=frame_features,
                emotion_labels=emotions,
                window_size=args.window_size,
                stride=args.stride,
            )
        if len(seq_features) == 0:
            continue

        np.save(output_dir / f"{split}_features.npy", seq_features.astype(np.float32))
        np.save(output_dir / f"{split}_labels.npy", seq_labels.astype(np.int64))
        metadata[split] = {
            "frames": int(loaded_frames),
            "sequences": int(len(seq_features)),
            "feature_dim": int(seq_features.shape[-1]),
            "sequence_mode": args.sequence_mode,
        }

    if not metadata:
        raise RuntimeError("No stage3 tensors generated. Check split CSVs and image paths.")

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Stage 3 tensors to: {output_dir}")
    print(json.dumps(metadata, indent=2))


def _build_frame_features(
    df: pd.DataFrame,
    stage2_runner: Stage2InferenceRunner,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    required = {"image_path", "emotion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV missing required columns: {sorted(missing)}")

    au_cols = [c for c in df.columns if c.startswith("AU_")]
    if len(au_cols) != 25:
        raise ValueError("Split CSV must include 25 AU columns AU_00..AU_24.")

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    loaded_frames = 0
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        frames: list[np.ndarray] = []
        au_rows: list[np.ndarray] = []
        emotions: list[int] = []
        for _, row in chunk.iterrows():
            path = Path(str(row["image_path"]))
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            au_rows.append(np.array([float(row[c]) for c in au_cols], dtype=np.float32))
            emotions.append(int(row["emotion"]))
        if not frames:
            continue

        loaded_frames += len(frames)
        au_matrix = np.stack(au_rows).astype(np.float32)
        stage2_outputs = stage2_runner.infer(
            frames,
            openface_au_matrix=au_matrix,
            openface_au_keys=MODEL_AU_KEYS[: au_matrix.shape[1]],
        )
        frame_features = np.concatenate(
            [
                stage2_outputs["emotion_probs"],
                stage2_outputs["au_probs"],
                stage2_outputs["valence"][:, None],
                stage2_outputs["arousal"][:, None],
            ],
            axis=1,
        ).astype(np.float32)
        feature_chunks.append(frame_features)
        label_chunks.append(np.array(emotions, dtype=np.int64))

    if loaded_frames == 0:
        raise RuntimeError("No valid images loaded for Stage 3 tensor generation.")

    return np.concatenate(feature_chunks, axis=0), np.concatenate(label_chunks, axis=0), loaded_frames


def _to_sequences_sliding(
    frame_features: np.ndarray,
    emotion_labels: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    seq_features: list[np.ndarray] = []
    seq_labels: list[int] = []
    for start in range(0, len(frame_features) - window_size + 1, stride):
        end = start + window_size
        window = frame_features[start:end]
        labels = emotion_labels[start:end]
        values, counts = np.unique(labels, return_counts=True)
        seq_label = int(values[np.argmax(counts)])
        seq_features.append(window)
        seq_labels.append(seq_label)
    if not seq_features:
        return np.empty((0, window_size, frame_features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(seq_features).astype(np.float32), np.array(seq_labels, dtype=np.int64)


def _to_sequences_grouped_by_emotion(
    frame_features: np.ndarray,
    emotion_labels: np.ndarray,
    window_size: int,
    stride: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    seq_features: list[np.ndarray] = []
    seq_labels: list[int] = []
    for emotion_id in sorted(np.unique(emotion_labels).tolist()):
        idx = np.where(emotion_labels == emotion_id)[0]
        if len(idx) < window_size:
            continue
        class_features = frame_features[idx]
        # AffectNet frames are i.i.d.; shuffle within class to avoid path-order artifacts.
        class_features = class_features[rng.permutation(len(class_features))]
        for start in range(0, len(class_features) - window_size + 1, stride):
            end = start + window_size
            seq_features.append(class_features[start:end])
            seq_labels.append(int(emotion_id))

    if not seq_features:
        return np.empty((0, window_size, frame_features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    features = np.stack(seq_features).astype(np.float32)
    labels = np.array(seq_labels, dtype=np.int64)
    order = rng.permutation(len(features))
    return features[order], labels[order]


if __name__ == "__main__":
    main()
