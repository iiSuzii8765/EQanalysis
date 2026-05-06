from __future__ import annotations


def compute_f1_appraisal(window: dict) -> float:
    delta_valence = window["valence_delta"]
    mean_au = window["mean_au"]
    au_surprise = mean_au.get("AU01_r", 0.0) + mean_au.get("AU02_r", 0.0) + mean_au.get("AU05_r", 0.0)
    novelty = 1 / (1 + pow(2.718281828, -(delta_valence + 0.5 * au_surprise)))
    goal_relevance = max(0.0, min(1.0, window["arousal_mean"]))
    positive_au = mean_au.get("AU06_r", 0.0) + mean_au.get("AU12_r", 0.0)
    suppression_au = mean_au.get("AU23_r", 0.0) + mean_au.get("AU24_r", 0.0) + window["au_freeze_ratio"]
    coping = positive_au / (positive_au + suppression_au + 1e-6)
    norm_compat = 1.0 - (1 / (1 + pow(2.718281828, -(mean_au.get("AU14_r", 0.0) + mean_au.get("AU09_r", 0.0)))))
    return max(0.0, min(1.0, goal_relevance * (1 - coping) + 0.2 * (1 - norm_compat) + 0.15 * novelty))
