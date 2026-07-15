"""
Stage 1: MediaPipe Face Mesh → Action Unit (AU) intensity estimator.

Computes per-frame AU intensity approximations from 468 face landmarks,
returning a DataFrame that matches the OpenFace _r column schema used
throughout the rest of the pipeline.

All distances are normalized by inter-ocular distance (IOD) so the
output is scale-invariant regardless of face size or camera distance.
Values are clipped to [0, 5] to match OpenFace's _r intensity range.

This is the primary Stage 1 path when the OpenFace binary is not
available (dev environments, Windows, Docker builds without OpenFace).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices (478-point model)
# ---------------------------------------------------------------------------
_L_EYE_TOP = 159
_L_EYE_BOT = 145
_L_EYE_IN  = 133
_L_EYE_OUT = 33

_R_EYE_TOP = 386
_R_EYE_BOT = 374
_R_EYE_IN  = 362
_R_EYE_OUT = 263

_L_BROW_IN  = 107
_L_BROW_MID = 105
_L_BROW_OUT = 70

_R_BROW_IN  = 336
_R_BROW_MID = 334
_R_BROW_OUT = 300

_NOSE_TIP    = 4
_NOSE_BRIDGE = 168
_NOSE_L      = 98
_NOSE_R      = 327

_MOUTH_L        = 61
_MOUTH_R        = 291
_UPPER_LIP_IN   = 13   # upper inner lip (just inside the opening)
_LOWER_LIP_IN   = 14   # lower inner lip
_UPPER_LIP_OUT  = 0    # cupid's bow top / outer upper lip
_LOWER_LIP_OUT  = 17   # outer bottom of lower lip

_CHIN     = 152
_L_CHEEK  = 234
_R_CHEEK  = 454

# AU output columns — identical ordering to Stage 2's MODEL_AU_KEYS.
AU_KEYS: list[str] = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r",
    "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
    "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU24_r",
    "AU25_r", "AU26_r", "AU28_r", "AU43_r", "AU45_r",
    "AU11_r", "AU13_r", "AU16_r", "AU18_r", "AU22_r",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _pt(lm: object, idx: int) -> np.ndarray:
    """Return (x, y) of a MediaPipe NormalizedLandmark in image-fraction coords."""
    p = lm[idx]  # type: ignore[index]
    return np.array([p.x, p.y], dtype=np.float64)


def _dist(lm: object, a: int, b: int) -> float:
    return float(np.linalg.norm(_pt(lm, a) - _pt(lm, b)))


# ---------------------------------------------------------------------------
# Per-frame AU computation
# ---------------------------------------------------------------------------

def _compute_aus(lm: object, iod: float) -> dict[str, float]:
    """
    Compute AU intensity estimates from a single frame's landmarks.

    iod: inter-ocular distance (normalization reference, same units as landmark coords).
    """
    eps = max(iod, 1e-6)

    # ---- eye geometry ----
    l_eye_h = _dist(lm, _L_EYE_TOP, _L_EYE_BOT) / eps
    r_eye_h = _dist(lm, _R_EYE_TOP, _R_EYE_BOT) / eps
    mean_eye_h = (l_eye_h + r_eye_h) / 2.0

    # ---- brow elevation (y distance from brow to eye corner, positive = brow raised) ----
    # In normalized image coords, y increases downward, so brow ABOVE eye → brow_y < eye_y.
    l_brow_in_elev  = (_pt(lm, _L_EYE_IN)[1]  - _pt(lm, _L_BROW_IN)[1])  / eps
    r_brow_in_elev  = (_pt(lm, _R_EYE_IN)[1]  - _pt(lm, _R_BROW_IN)[1])  / eps
    l_brow_out_elev = (_pt(lm, _L_EYE_OUT)[1] - _pt(lm, _L_BROW_OUT)[1]) / eps
    r_brow_out_elev = (_pt(lm, _R_EYE_OUT)[1] - _pt(lm, _R_BROW_OUT)[1]) / eps
    inner_brow_mean = (l_brow_in_elev  + r_brow_in_elev)  / 2.0
    outer_brow_mean = (l_brow_out_elev + r_brow_out_elev) / 2.0

    # brow lowerer: inner brows moving toward each other (distance shrinks)
    inner_brow_horiz = _dist(lm, _L_BROW_IN, _R_BROW_IN) / eps
    brow_lower_signal = max(0.0, 0.5 - inner_brow_horiz)  # neutral spacing ≈ 0.5 IOD

    # ---- nose ----
    nose_w = _dist(lm, _NOSE_L, _NOSE_R) / eps  # smaller → wrinkled

    # ---- lip geometry ----
    mouth_w   = _dist(lm, _MOUTH_L, _MOUTH_R)  / eps
    lip_gap   = _dist(lm, _UPPER_LIP_IN, _LOWER_LIP_IN) / eps
    outer_gap = _dist(lm, _UPPER_LIP_OUT, _LOWER_LIP_OUT) / eps

    # lip corner vertical offset relative to mid-lip y
    lip_mid_y     = (_pt(lm, _UPPER_LIP_IN)[1] + _pt(lm, _LOWER_LIP_IN)[1]) / 2.0
    corner_mean_y = (_pt(lm, _MOUTH_L)[1] + _pt(lm, _MOUTH_R)[1]) / 2.0
    # smile (AU12): corners above lip midline → corner_mean_y < lip_mid_y
    corner_up   = max(0.0, (lip_mid_y - corner_mean_y) / eps)
    # sad (AU15): corners below lip midline
    corner_down = max(0.0, (corner_mean_y - lip_mid_y) / eps)

    # chin relative to lower lip
    lower_lip_y = _pt(lm, _LOWER_LIP_OUT)[1]
    chin_y      = _pt(lm, _CHIN)[1]
    chin_raise  = max(0.0, (lower_lip_y - chin_y) / eps)   # chin moves up → gap shrinks

    # jaw drop (nose-tip to chin distance; longer = dropped jaw)
    jaw_dist        = _dist(lm, _NOSE_TIP, _CHIN) / eps
    jaw_drop_signal = max(0.0, jaw_dist - 1.8)             # ~1.8 IOD is neutral

    # cheek elevation proxy (cheek above nose tip → cheek_y < nose_y in image coords)
    cheek_mean_y = (_pt(lm, _L_CHEEK)[1] + _pt(lm, _R_CHEEK)[1]) / 2.0
    nose_y       = _pt(lm, _NOSE_TIP)[1]
    cheek_elev   = max(0.0, (nose_y - cheek_mean_y) / eps)

    def _s(x: float) -> float:
        """Scale to [0, 5] range."""
        return float(np.clip(x * 5.0, 0.0, 5.0))

    return {
        # AU01 inner brow raise: inner brow elevated above neutral
        "AU01_r": _s(inner_brow_mean),
        # AU02 outer brow raise
        "AU02_r": _s(outer_brow_mean),
        # AU04 brow lowerer: inner brows pulled together
        "AU04_r": _s(brow_lower_signal * 4.0),
        # AU05 upper lid raiser: eye opens wider (mean_eye_h > baseline ≈ 0.18)
        "AU05_r": _s(max(0.0, mean_eye_h - 0.18) * 12.0),
        # AU06 cheek raiser: cheeks push up (smiling)
        "AU06_r": _s(cheek_elev * 2.0),
        # AU07 lid tightener: eye narrows (mean_eye_h below baseline)
        "AU07_r": _s(max(0.0, 0.18 - mean_eye_h) * 12.0),
        # AU09 nose wrinkler: nasal wings compress
        "AU09_r": _s(max(0.0, 0.35 - nose_w) * 8.0),
        # AU10 upper lip raiser: upper lip moves toward nose
        "AU10_r": _s(max(0.0, (_pt(lm, _NOSE_TIP)[1] - _pt(lm, _UPPER_LIP_OUT)[1]) / eps - 0.5) * 4.0),
        # AU12 lip corner puller: zygomaticus — corners pulled up and out
        "AU12_r": _s(corner_up * 4.0),
        # AU14 dimpler: mouth width noticeably wider than neutral
        "AU14_r": _s(max(0.0, mouth_w - 0.9) * 3.0),
        # AU15 lip corner depressor: depressor anguli oris
        "AU15_r": _s(corner_down * 4.0),
        # AU17 chin raiser: mentalis — chin bunching up toward lower lip
        "AU17_r": _s(chin_raise * 3.0),
        # AU20 lip stretcher: risorius — horizontal lip stretch
        "AU20_r": _s(max(0.0, mouth_w - 0.8) * 3.0),
        # AU23 lip tightener: orbicularis oris — lip gap compresses
        "AU23_r": _s(max(0.0, 0.06 - lip_gap) * 30.0),
        # AU24 lip pressor: lips pressed flat
        "AU24_r": _s(max(0.0, 0.04 - outer_gap) * 40.0),
        # AU25 lips part: inter-lip gap
        "AU25_r": _s(lip_gap * 8.0),
        # AU26 jaw drop: increased nose-chin distance
        "AU26_r": _s(jaw_drop_signal * 2.0),
        # AU28 lip suck: inward lip compression (mirrors AU24)
        "AU28_r": _s(max(0.0, 0.04 - outer_gap) * 50.0),
        # AU43 eyes closed: very small eye opening
        "AU43_r": _s(max(0.0, 0.08 - mean_eye_h) * 30.0),
        # AU45 blink: filled in post-processing from temporal pattern
        "AU45_r": 0.0,
        # AUs with no reliable 2-D landmark proxy — held at 0 until depth data available
        "AU11_r": 0.0,   # nasolabial deepener
        "AU13_r": 0.0,   # cheek puffer
        "AU16_r": _s(lip_gap * 4.0),   # lower lip depressor (partial proxy via gap)
        "AU18_r": 0.0,   # lip pucker (requires 3-D depth)
        "AU22_r": _s(max(0.0, 0.7 - mouth_w) * 3.0),  # lip funneler: mouth compressed
    }


# ---------------------------------------------------------------------------
# Frame-level extraction entry point
# ---------------------------------------------------------------------------

def extract_aus_from_frames(
    frames: list[np.ndarray],
    target_fps: int = 25,
) -> pd.DataFrame:
    """
    Extract AU intensities from a list of BGR frames using MediaPipe Face Mesh.

    Returns a DataFrame with columns:
        frame, timestamp, confidence, AU01_r … AU22_r
    — identical schema to the CSV produced by OpenFace FeatureExtraction.

    Frames with no detected face receive confidence=0.0 and all AUs at 0.
    """
    if not _MP_AVAILABLE:
        raise ImportError(
            "mediapipe is required for Stage 1 extraction. "
            "Install it with:  pip install 'mediapipe>=0.10'"
        )

    face_mesh_cls = mp.solutions.face_mesh.FaceMesh  # type: ignore[attr-defined]
    records: list[dict] = []

    with face_mesh_cls(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for frame_idx, frame in enumerate(frames):
            timestamp = frame_idx / max(target_fps, 1)
            # MediaPipe expects RGB; OpenCV loads BGR.
            rgb = frame[:, :, ::-1].copy() if frame.ndim == 3 and frame.shape[2] == 3 else frame
            result = face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                records.append(_null_row(frame_idx, timestamp))
                continue

            lm = result.multi_face_landmarks[0].landmark
            l_eye_c = (_pt(lm, _L_EYE_IN) + _pt(lm, _L_EYE_OUT)) / 2.0
            r_eye_c = (_pt(lm, _R_EYE_IN) + _pt(lm, _R_EYE_OUT)) / 2.0
            iod = float(np.linalg.norm(l_eye_c - r_eye_c))

            aus = _compute_aus(lm, iod)
            records.append({"frame": frame_idx, "timestamp": timestamp, "confidence": 0.85, **aus})

    if not records:
        return _empty_df()

    df = pd.DataFrame(records)
    _postprocess_blink(df)
    return df


def _postprocess_blink(df: pd.DataFrame) -> None:
    """Detect rapid single-frame eye closures and mark them as blinks (AU45) in-place."""
    if "AU43_r" not in df.columns or len(df) < 3:
        return
    closed = (df["AU43_r"] > 1.0).to_numpy()
    for i in range(1, len(closed) - 1):
        # Blink: eye closed this frame but open in frames on either side.
        if closed[i] and not closed[i - 1] and not closed[i + 1]:
            df.at[i, "AU45_r"] = 5.0


def _null_row(frame_idx: int, timestamp: float) -> dict:
    row: dict = {"frame": frame_idx, "timestamp": timestamp, "confidence": 0.0}
    row.update({k: 0.0 for k in AU_KEYS})
    return row


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["frame", "timestamp", "confidence", *AU_KEYS])
