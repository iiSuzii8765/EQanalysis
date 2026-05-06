from __future__ import annotations

import numpy as np

PROTOTYPE_AUS: dict[str, list[str]] = {
    "happiness": ["AU06_r", "AU12_r"],
    "sadness": ["AU01_r", "AU04_r", "AU15_r", "AU17_r"],
    "anger": ["AU04_r", "AU05_r", "AU07_r", "AU23_r", "AU24_r"],
    "fear": ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU07_r", "AU20_r", "AU26_r"],
    "disgust": ["AU09_r", "AU15_r", "AU16_r", "AU25_r", "AU26_r"],
    "surprise": ["AU01_r", "AU02_r", "AU05_r", "AU26_r"],
    "contempt": ["AU12_r", "AU14_r"],
    "neutral": [],
}


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_f2_somatic(window: dict) -> float:
    mean_au = window["mean_au"]
    emotion_probs = window["emotion_probs"]
    dominant_emotion = max(emotion_probs, key=emotion_probs.get)
    expected = np.zeros(len(window["au_keys"]), dtype=np.float32)
    for idx, key in enumerate(window["au_keys"]):
        if key in PROTOTYPE_AUS.get(dominant_emotion, []):
            expected[idx] = 1.0
    observed = np.array([mean_au[k] for k in window["au_keys"]], dtype=np.float32)
    exp_norm = float(np.linalg.norm(expected))
    obs_norm = float(np.linalg.norm(observed))
    cosine_sim = 0.0 if exp_norm == 0 or obs_norm == 0 else float(np.dot(expected, observed) / (exp_norm * obs_norm))
    scdi = 1.0 - clip01((cosine_sim + 1.0) / 2.0)
    return clip01(0.7 * scdi + 0.3 * window["suppression_idx"])
