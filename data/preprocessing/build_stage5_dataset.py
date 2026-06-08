from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from app.scoring import compute_f_scores
from models.stage2_spatial.inference import MODEL_AU_KEYS

EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "contempt", "neutral"]

AblationMode = Literal["full", "no_philosophy", "no_f2", "linear_only"]

# Dimension of raw Stage 3 aggregate features used by "no_philosophy" mode:
#   8 (emotion_probs mean) + 1 (valence_mean) + 1 (arousal_mean)
#   + 1 (suppression_index) + 1 (transition_rate) = 12
RAW_STAGE3_DIM = 12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-tensors-dir", required=True, help="Directory containing train/val/test stage3 tensors.")
    parser.add_argument("--output-dir", required=True, help="Output directory for stage5 features and labels.")
    parser.add_argument(
        "--ablation-mode",
        default="full",
        choices=["full", "no_philosophy", "no_f2", "linear_only"],
        help=(
            "Feature representation: "
            "'full' = phi=[F1,F2,1-F3,F4] (default); "
            "'no_philosophy' = raw 12-dim Stage 3 aggregates; "
            "'no_f2' = phi with F2 zeroed out; "
            "'linear_only' = phi with equal weights (BayesPhiloNet skipped at train time)."
        ),
    )
    args = parser.parse_args()

    stage3_dir = Path(args.stage3_tensors_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict[str, float | int | str]] = {}
    for split in ("train", "val", "test"):
        x_path = stage3_dir / f"{split}_features.npy"
        if not x_path.exists():
            continue
        features = np.load(x_path).astype(np.float32)  # [N, T, 35]
        X, ers = build_phi_and_targets(features, ablation_mode=args.ablation_mode)
        np.save(output_dir / f"{split}_features.npy", X.astype(np.float32))
        np.save(output_dir / f"{split}_labels.npy", ers.astype(np.float32))
        metadata[split] = {
            "samples": int(len(X)),
            "feature_dim": int(X.shape[-1]),
            "ablation_mode": args.ablation_mode,
            "ers_mean": float(np.mean(ers)),
            "ers_std": float(np.std(ers)),
        }

    if not metadata:
        raise RuntimeError("No split tensors found in Stage 3 tensors directory.")

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Stage 5 dataset to: {output_dir}")
    print(json.dumps(metadata, indent=2))


def build_phi_and_targets(
    seq_features: np.ndarray,
    ablation_mode: AblationMode = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Stage 3 temporal tensors to Stage 5 (feature, ERS-target) pairs.

    ERS proxy targets are always derived from the full phi formula regardless
    of ablation_mode, so targets are comparable across ablations.

    Args:
        seq_features: float32 array of shape [N, T, 35].
        ablation_mode: Controls the input feature representation.
            - "full":          phi = [F1, F2, 1-F3, F4]  (4-dim)
            - "no_philosophy": raw 12-dim Stage 3 aggregates
            - "no_f2":         phi with F2 forced to 0.0  (4-dim)
            - "linear_only":   phi = [F1, F2, 1-F3, F4]  (4-dim); caller
                               should apply a weighted sum (w=0.25 each)
                               instead of BayesPhiloNet.

    Returns:
        X: float32 feature matrix [N, D].
        ers: float32 ERS proxy targets [N].
    """
    if ablation_mode not in ("full", "no_philosophy", "no_f2", "linear_only"):
        raise ValueError(f"Unknown ablation_mode: {ablation_mode!r}")

    windows = [_sequence_to_window_dict(seq) for seq in seq_features]
    baseline = {
        "variability": float(np.mean([w["emotion_variability"] for w in windows])) + 1e-6,
        "transition_rate": float(np.mean([w["transition_rate"] for w in windows])) + 1e-6,
    }

    feature_rows: list[np.ndarray] = []
    targets: list[float] = []
    for w in windows:
        f = compute_f_scores(w, baseline=baseline)
        phi_full = np.array([f["F1"], f["F2"], 1.0 - f["F3"], f["F4"]], dtype=np.float32)
        # ERS proxy is always computed from full phi so targets stay consistent.
        target = float(np.clip(
            0.28 * phi_full[0] + 0.34 * phi_full[1] + 0.18 * phi_full[2] + 0.20 * phi_full[3],
            0.0, 1.0,
        ))

        if ablation_mode == "full":
            feature_vec = phi_full
        elif ablation_mode == "no_philosophy":
            feature_vec = _raw_stage3_features(w)
        elif ablation_mode == "no_f2":
            feature_vec = np.array([f["F1"], 0.0, 1.0 - f["F3"], f["F4"]], dtype=np.float32)
        else:  # "linear_only" — same phi, training uses a linear combiner
            feature_vec = phi_full

        feature_rows.append(feature_vec)
        targets.append(target)

    return np.stack(feature_rows).astype(np.float32), np.array(targets, dtype=np.float32)


def _raw_stage3_features(w: dict) -> np.ndarray:
    """12-dim raw Stage 3 aggregate: 8 emotion means + valence_mean + arousal_mean
    + suppression_index + transition_rate."""
    emotion_means = np.array(
        [w["emotion_probs"][label] for label in EMOTION_LABELS],
        dtype=np.float32,
    )
    scalars = np.array(
        [w["valence_mean"], w["arousal_mean"], w["suppression_idx"], w["transition_rate"]],
        dtype=np.float32,
    )
    return np.concatenate([emotion_means, scalars])


def _sequence_to_window_dict(seq: np.ndarray) -> dict:
    emotion_probs = seq[:, :8]
    au_matrix = seq[:, 8:33]
    valence = seq[:, 33]
    arousal = seq[:, 34]
    mean_au_vec = np.mean(au_matrix, axis=0)
    mean_au = {k: float(v) for k, v in zip(MODEL_AU_KEYS, mean_au_vec)}

    frame_emotions = np.argmax(emotion_probs, axis=1)
    transitions = float(np.sum(frame_emotions[1:] != frame_emotions[:-1])) if len(frame_emotions) > 1 else 0.0
    transition_rate = transitions / max(len(frame_emotions) / 25.0, 1e-3)

    au_delta = np.abs(np.diff(au_matrix, axis=0)) if len(au_matrix) > 1 else np.zeros_like(au_matrix)
    suppression_idx = float(np.mean(np.mean(au_delta, axis=1) < 0.05)) if len(au_delta) else 0.0

    return {
        "valence_delta": float(valence[-1] - valence[0]) if len(valence) > 1 else 0.0,
        "valence_mean": float(np.mean(valence)),
        "mean_au": mean_au,
        "arousal_mean": float(np.mean(arousal)),
        "au_freeze_ratio": suppression_idx,
        "suppression_idx": suppression_idx,
        "emotion_probs": {label: float(np.mean(emotion_probs[:, idx])) for idx, label in enumerate(EMOTION_LABELS)},
        "au_keys": MODEL_AU_KEYS,
        "transition_rate": float(transition_rate),
        "emotion_variability": float(np.mean(np.std(emotion_probs, axis=0))),
        "au_pairwise_dispersion": _pairwise_dispersion(au_matrix),
        "arc_error": _arc_error(au_matrix),
        "onset_offset_asymmetry": _onset_offset_asymmetry(au_matrix),
    }


def _pairwise_dispersion(au_matrix: np.ndarray) -> float:
    if au_matrix.shape[0] < 2:
        return 0.0
    std = np.std(au_matrix, axis=0)
    valid = std > 1e-6
    if np.sum(valid) < 2:
        return 0.5
    valid_data = au_matrix[:, valid].astype(np.float64)
    centered = valid_data - valid_data.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(centered**2, axis=0, keepdims=True))
    denom = np.where(denom < 1e-12, 1.0, denom)
    normalized = centered / denom
    corr = np.clip(normalized.T @ normalized, -1.0, 1.0)
    return float(np.mean(np.abs(corr - np.eye(corr.shape[0]))))


def _arc_error(au_matrix: np.ndarray) -> float:
    if au_matrix.shape[0] < 4:
        return 0.5
    errors: list[float] = []
    t = au_matrix.shape[0]
    for k in range(au_matrix.shape[1]):
        traj = au_matrix[:, k]
        above = np.where(traj > 0.5)[0]
        if len(above) < 2:
            continue
        onset = int(above[0])
        offset = int(above[-1])
        apex = int(np.argmax(traj))
        rise = max(0, apex - onset) / t
        hold = max(0, offset - apex) / t
        errors.append(abs(rise - 0.30) + abs(hold - 0.40))
    if not errors:
        return 0.5
    return float(np.clip(np.mean(errors), 0.0, 1.0))


def _onset_offset_asymmetry(au_matrix: np.ndarray) -> float:
    if au_matrix.shape[0] < 4:
        return 1.0
    ratios: list[float] = []
    for k in range(au_matrix.shape[1]):
        traj = au_matrix[:, k]
        above = np.where(traj > 0.5)[0]
        if len(above) < 2:
            continue
        onset = int(above[0])
        offset = int(above[-1])
        apex = int(np.argmax(traj))
        onset_speed = max(1.0, float(apex - onset))
        offset_speed = max(1.0, float(offset - apex))
        ratios.append(onset_speed / offset_speed)
    if not ratios:
        return 1.0
    return float(np.mean(ratios))


if __name__ == "__main__":
    main()
