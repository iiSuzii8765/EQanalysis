from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Week3Metrics:
    ers_mean: float
    ers_std: float
    mean_uncertainty: float
    high_confidence_ratio: float


def summarize_result(result: dict) -> Week3Metrics:
    windows = result.get("windows", [])
    if not windows:
        raise ValueError("No windows found in result payload.")

    ers = np.array([w["ERS"] for w in windows], dtype=np.float32)
    uncertainty = np.array([w["ERS_uncertainty"] for w in windows], dtype=np.float32)
    confidence = np.array([1.0 if w["confidence_flag"] else 0.0 for w in windows], dtype=np.float32)

    return Week3Metrics(
        ers_mean=float(np.mean(ers)),
        ers_std=float(np.std(ers)),
        mean_uncertainty=float(np.mean(uncertainty)),
        high_confidence_ratio=float(np.mean(confidence)),
    )
