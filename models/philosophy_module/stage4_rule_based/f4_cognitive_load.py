from __future__ import annotations


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sigmoid(x: float) -> float:
    return 1 / (1 + pow(2.718281828, -x))


def compute_f4_cognitive_load(window: dict, baseline: dict[str, float]) -> float:
    variability = window["emotion_variability"]
    baseline_var = max(0.05, baseline["variability"])
    f4a = clip01(1.0 - (variability / baseline_var))
    baseline_rate = max(0.2, baseline["transition_rate"])
    deviation = abs(window["transition_rate"] - baseline_rate) / baseline_rate
    f4b = clip01(sigmoid(deviation * 3))
    f4c = clip01(sigmoid(window["onset_offset_asymmetry"] - 1.0))
    return clip01(0.40 * f4a + 0.35 * f4b + 0.25 * f4c)
