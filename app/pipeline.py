from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.openface_extractor import run_openface
from app.scoring import classify_strategy, compute_f_scores
from data.preprocessing.extract_frames import load_video_frames, resize_frames
from data.preprocessing.segment_windows import segment_indices
from models.philosophy_module.stage5_bayesian.inference import Stage5InferenceConfig, Stage5InferenceRunner
from models.stage2_spatial.inference import Stage2InferenceConfig, Stage2InferenceRunner
from models.stage3_temporal.inference import Stage3InferenceConfig, Stage3InferenceRunner


def run_week3_pipeline(video_path: str, context: str, session_id: str) -> dict:
    csv_path = run_openface(video_path=video_path, session_id=session_id)
    features_df = _load_openface_csv(csv_path)
    frames, _ = load_video_frames(video_path, target_fps=settings.target_fps)
    frames = resize_frames(frames, size=(224, 224))
    aligned = _align_modalities(features_df, frames)

    stage2_runner = Stage2InferenceRunner(
        Stage2InferenceConfig(
            checkpoint_path=settings.stage2_checkpoint_path,
            device=settings.inference_device,
            batch_size=settings.inference_batch_size,
            au_blend_alpha=settings.stage2_au_blend_alpha,
        )
    )
    stage2_outputs = stage2_runner.infer(
        aligned["frames"],
        openface_au_matrix=aligned["au_matrix"],
        openface_au_keys=aligned["au_cols"],
    )

    stage3_runner = Stage3InferenceRunner(
        Stage3InferenceConfig(
            checkpoint_path=settings.stage3_checkpoint_path,
            device=settings.inference_device,
            input_dim=stage2_outputs["emotion_probs"].shape[1] + stage2_outputs["au_probs"].shape[1] + 2,
        )
    )
    stage5_runner = Stage5InferenceRunner(
        Stage5InferenceConfig(
            checkpoint_path=settings.stage5_checkpoint_path,
            device=settings.inference_device,
            mc_passes=settings.stage5_mc_passes,
        )
    )

    windows = _build_windows(aligned, stage2_outputs, stage3_runner)
    if not windows:
        raise RuntimeError("No valid windows were generated from OpenFace output.")

    baseline = _compute_baseline(windows)
    output_windows: list[dict] = []
    for window in windows:
        f_scores = compute_f_scores(window, baseline=baseline)
        phi = np.array([f_scores["F1"], f_scores["F2"], 1.0 - f_scores["F3"], f_scores["F4"]], dtype=np.float32)
        ers = stage5_runner.predict_with_uncertainty(phi)
        strategy = classify_strategy(
            f_scores["F1"],
            f_scores["F2"],
            f_scores["F3"],
            f_scores["F4"],
            transition_rate=window["transition_rate"],
            baseline_rate=baseline["transition_rate"],
        )
        dominant_emotion = max(window["emotion_probs"], key=window["emotion_probs"].get)
        output_windows.append(
            {
                "timestamp_start": round(float(window["timestamp_start"]), 3),
                "timestamp_end": round(float(window["timestamp_end"]), 3),
                "ERS": ers["ERS"],
                "ERS_uncertainty": ers["ERS_uncertainty"],
                "ERS_ci_low": ers["ERS_ci_low"],
                "ERS_ci_high": ers["ERS_ci_high"],
                "philosophy_scores": {
                    "F1_appraisal": round(f_scores["F1"], 3),
                    "F2_somatic": round(f_scores["F2"], 3),
                    "F3_coherence": round(f_scores["F3"], 3),
                    "F4_cognitive": round(f_scores["F4"], 3),
                },
                "shap_attributions": [round(float(v), 3) for v in phi.tolist()],
                "regulation_strategy": strategy,
                "confidence_flag": bool(ers["confidence_flag"]),
                "dominant_emotion": dominant_emotion,
                "microexpression": bool(window["microexpression"]),
                "context": context,
                "deployment_mode": "async",
            }
        )

    return {
        "session_processed_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "week3-stage2-stage3-stage5",
        "session_id": session_id,
        "video_path": str(Path(video_path)),
        "openface_csv_path": str(csv_path),
        "context": context,
        "windows_count": len(output_windows),
        "model_status": {
            "stage2_checkpoint_loaded": bool(stage2_outputs.get("checkpoint_loaded", False)),
            "stage3_checkpoint_loaded": bool(stage3_runner.checkpoint_loaded),
            "stage5_checkpoint_loaded": bool(stage5_runner.checkpoint_loaded),
        },
        "session_summary": _compute_session_summary(output_windows),
        "windows": output_windows,
    }


def _load_openface_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "success" in df.columns:
        df = df[df["success"] == 1]
    if df.empty:
        raise RuntimeError("OpenFace CSV has no successful face tracking frames.")

    au_cols = sorted([c for c in df.columns if c.startswith("AU") and c.endswith("_r")])
    if not au_cols:
        raise RuntimeError("OpenFace CSV has no AU intensity columns (AU*_r).")

    keep_cols = ["frame", "timestamp", "confidence", *au_cols]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    out = out.interpolate(limit_direction="both").fillna(0.0)
    out["timestamp"] = out["timestamp"] - float(out["timestamp"].iloc[0])
    return out


def _align_modalities(df: pd.DataFrame, frames: list[np.ndarray]) -> dict:
    timestamps = df["timestamp"].to_numpy()
    fps_estimate = _estimate_fps(timestamps)
    sampled_df = _resample_df(df, source_fps=fps_estimate, target_fps=settings.target_fps)
    length = min(len(sampled_df), len(frames))
    if length < settings.window_size_frames:
        raise RuntimeError(
            f"Not enough aligned frames for one window. need={settings.window_size_frames} actual={length}"
        )
    sampled_df = sampled_df.iloc[:length].reset_index(drop=True)
    return {
        "df": sampled_df,
        "frames": frames[:length],
        "au_cols": [c for c in sampled_df.columns if c.startswith("AU") and c.endswith("_r")],
        "au_matrix": sampled_df[[c for c in sampled_df.columns if c.startswith("AU") and c.endswith("_r")]].to_numpy(
            dtype=np.float32
        ),
    }


def _build_windows(aligned: dict, stage2_outputs: dict, stage3_runner: Stage3InferenceRunner) -> list[dict]:
    au_cols = aligned["au_cols"]
    df = aligned["df"]
    au_matrix = aligned["au_matrix"]
    stage2_features = np.concatenate(
        [
            stage2_outputs["emotion_probs"],
            stage2_outputs["au_probs"],
            stage2_outputs["valence"][:, None],
            stage2_outputs["arousal"][:, None],
        ],
        axis=1,
    ).astype(np.float32)
    windows: list[dict] = []
    for start, end in segment_indices(len(df), settings.window_size_frames, settings.window_stride_frames):
        segment = df.iloc[start:end]
        window_au = au_matrix[start:end]
        window_stage2 = stage2_features[start:end]
        window_emotions = stage2_outputs["emotion_probs"][start:end]
        temporal = stage3_runner.infer_window(window_stage2, window_emotions, window_au)
        windows.append(_compute_window_features(segment, au_cols, window_au, stage2_outputs, start, end, temporal))
    return windows


def _estimate_fps(timestamps: np.ndarray) -> float:
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return float(settings.target_fps)
    return float(1.0 / np.median(diffs))


def _resample_df(df: pd.DataFrame, source_fps: float, target_fps: int) -> pd.DataFrame:
    if source_fps <= 0:
        return df
    step = max(1, int(round(source_fps / target_fps)))
    sampled = df.iloc[::step].reset_index(drop=True)
    if sampled.empty:
        sampled = df.copy().reset_index(drop=True)
    return sampled


def _compute_window_features(
    segment: pd.DataFrame,
    au_cols: list[str],
    au_matrix: np.ndarray,
    stage2_outputs: dict,
    start: int,
    end: int,
    temporal: dict,
) -> dict:
    conf = float(segment["confidence"].mean()) if "confidence" in segment.columns else 0.8
    conf = max(0.0, min(1.0, conf))
    au_mean_vec = au_matrix.mean(axis=0)
    au_delta = np.diff(au_matrix, axis=0) if len(au_matrix) > 1 else np.zeros_like(au_matrix)
    au_delta_abs = np.abs(au_delta)

    au_intensity_mean = float(np.mean(au_mean_vec))
    valence_curve = stage2_outputs["valence"][start:end]
    arousal_curve = stage2_outputs["arousal"][start:end]
    valence_delta = float(valence_curve[-1] - valence_curve[0]) if len(valence_curve) > 1 else 0.0
    frame_emotions = np.argmax(stage2_outputs["emotion_probs"][start:end], axis=1)
    emotion_probs_window = stage2_outputs["emotion_probs"][start:end]
    mean_emotion_probs = emotion_probs_window.mean(axis=0)
    emotion_prob_dict = {label: float(mean_emotion_probs[idx]) for idx, label in enumerate(_emotion_labels())}

    pairwise_dispersion = _pairwise_dispersion(au_matrix)
    arc_error = _arc_error(au_matrix)
    onset_offset_asym = _onset_offset_asymmetry(au_matrix)
    au_noise = float(np.std(au_delta_abs)) if len(au_delta_abs) else 0.0

    return {
        "timestamp_start": float(segment["timestamp"].iloc[0]),
        "timestamp_end": float(segment["timestamp"].iloc[-1]),
        "au_keys": au_cols,
        "mean_au": {k: float(v) for k, v in zip(au_cols, au_mean_vec)},
        "au_intensity_mean": au_intensity_mean,
        "valence_mean": float(np.mean(valence_curve)),
        "valence_delta": valence_delta,
        "arousal_mean": float(temporal["arousal_mean"]),
        "transition_rate": float(temporal["transition_rate"]),
        "suppression_idx": float(temporal["suppression_idx"]),
        "microexpression": bool(temporal["microexpression"]),
        "au_freeze_ratio": float(temporal["au_freeze_ratio"]),
        "emotion_variability": float(temporal["expr_variability"]),
        "au_pairwise_dispersion": pairwise_dispersion,
        "arc_error": arc_error,
        "onset_offset_asymmetry": onset_offset_asym,
        "au_noise": au_noise,
        "face_confidence": conf,
        "emotion_probs": emotion_prob_dict,
        "attention_weights": temporal["attention_weights"],
        "frame_emotions": frame_emotions,
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
    corr = normalized.T @ normalized
    corr = np.clip(corr, -1.0, 1.0)

    return float(np.mean(np.abs(corr - np.eye(corr.shape[0]))))


def _arc_error(au_matrix: np.ndarray) -> float:
    if au_matrix.shape[0] < 4:
        return 0.5
    errors = []
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
        err = abs(rise - 0.30) + abs(hold - 0.40)
        errors.append(err)
    if not errors:
        return 0.5
    return float(np.clip(np.mean(errors), 0.0, 1.0))


def _onset_offset_asymmetry(au_matrix: np.ndarray) -> float:
    if au_matrix.shape[0] < 4:
        return 1.0
    ratios = []
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


def _compute_baseline(windows: list[dict]) -> dict[str, float]:
    return {
        "variability": float(np.mean([w["emotion_variability"] for w in windows])) + 1e-6,
        "transition_rate": float(np.mean([w["transition_rate"] for w in windows])) + 1e-6,
    }


def _compute_session_summary(output_windows: list[dict]) -> dict:
    ers_values = [w["ERS"] for w in output_windows]
    strategies = [w["regulation_strategy"] for w in output_windows]
    suppression_ratio = sum(1 for s in strategies if s == "suppression") / len(strategies)
    reappraisal_ratio = sum(1 for s in strategies if s == "reappraisal") / len(strategies)
    dysreg_ratio = sum(1 for s in strategies if s == "dysregulation") / len(strategies)
    return {
        "ERS_mean": round(float(np.mean(ers_values)), 3),
        "ERS_peak": round(float(np.max(ers_values)), 3),
        "ERS_variability": round(float(np.std(ers_values)), 3),
        "suppression_ratio": round(float(suppression_ratio), 3),
        "reappraisal_ratio": round(float(reappraisal_ratio), 3),
        "dysregulation_ratio": round(float(dysreg_ratio), 3),
        "mean_uncertainty": round(float(np.mean([w["ERS_uncertainty"] for w in output_windows])), 3),
    }


def _emotion_labels() -> list[str]:
    return ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "contempt", "neutral"]
