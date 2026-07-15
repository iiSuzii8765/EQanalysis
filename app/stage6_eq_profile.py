from __future__ import annotations

import numpy as np


def compute_goleman_profile(output_windows: list[dict]) -> dict:
    """Aggregate Stage 5 per-window outputs into a Goleman Five-Domain EQ profile."""
    if not output_windows:
        return _empty_profile()

    n = len(output_windows)

    ers_values = np.array([w["ERS"] for w in output_windows], dtype=np.float32)
    ers_uncertainty = np.array([w["ERS_uncertainty"] for w in output_windows], dtype=np.float32)
    strategies = [w["regulation_strategy"] for w in output_windows]

    f1_values = np.array([w["philosophy_scores"]["F1_appraisal"] for w in output_windows], dtype=np.float32)
    f2_values = np.array([w["philosophy_scores"]["F2_somatic"] for w in output_windows], dtype=np.float32)
    f3_values = np.array([w["philosophy_scores"]["F3_coherence"] for w in output_windows], dtype=np.float32)
    f4_values = np.array([w["philosophy_scores"]["F4_cognitive"] for w in output_windows], dtype=np.float32)

    ers_mean = float(np.mean(ers_values))
    ers_peak = float(np.max(ers_values))
    ers_variability = float(np.std(ers_values))

    suppression_ratio = sum(1 for s in strategies if s == "suppression") / n
    reappraisal_ratio = sum(1 for s in strategies if s == "reappraisal") / n
    dysreg_rate = sum(1 for s in strategies if s == "dysregulation") / n

    mean_f2 = float(np.mean(f2_values))
    mean_f3 = float(np.mean(f3_values))
    mean_f4 = float(np.mean(f4_values))
    mean_uncertainty = float(np.mean(ers_uncertainty))

    # Motivation uses only high-arousal windows (ERS >= median as proxy for high arousal)
    ers_median = float(np.median(ers_values))
    high_arousal_mask = ers_values >= ers_median
    motivation_f1 = float(np.mean(f1_values[high_arousal_mask])) if high_arousal_mask.any() else float(np.mean(f1_values))

    self_awareness = _clamp(100.0 * (1.0 - mean_f2) * mean_f3)
    self_regulation = _clamp(100.0 * (1.0 - suppression_ratio) * (1.0 - ers_variability))
    motivation = _clamp(100.0 * motivation_f1)
    empathy = _clamp(100.0 * mean_f3 * (1.0 - mean_f2))
    social_skills = _clamp(100.0 * reappraisal_ratio * (1.0 - mean_f4))

    overall_eq_score = round((self_awareness + self_regulation + motivation + empathy + social_skills) / 5.0)

    strategy_counts = {
        "suppression": suppression_ratio,
        "reappraisal": reappraisal_ratio,
        "dysregulation": dysreg_rate,
        "neutral": sum(1 for s in strategies if s == "neutral") / n,
        "expression": sum(1 for s in strategies if s == "expression") / n,
    }
    dominant_strategy = max(strategy_counts, key=strategy_counts.get)  # type: ignore[arg-type]

    return {
        "goleman_eq_profile": {
            "self_awareness": round(self_awareness),
            "self_regulation": round(self_regulation),
            "motivation": round(motivation),
            "empathy": round(empathy),
            "social_skills": round(social_skills),
        },
        "overall_eq_score": overall_eq_score,
        "session_summary": {
            "ERS_mean": round(ers_mean, 3),
            "ERS_peak": round(ers_peak, 3),
            "ERS_variability": round(ers_variability, 3),
            "suppression_ratio": round(suppression_ratio, 3),
            "reappraisal_ratio": round(reappraisal_ratio, 3),
            "dysregulation_ratio": round(dysreg_rate, 3),
            "mean_uncertainty": round(mean_uncertainty, 3),
            "dominant_strategy": dominant_strategy,
        },
        "top_insights": _extract_top_insights(output_windows, f2_values, ers_values),
        "wellness_flag": _wellness_flag(strategies, ers_variability),
    }


def _extract_top_insights(
    output_windows: list[dict],
    f2_values: np.ndarray,
    ers_values: np.ndarray,
    top_n: int = 3,
) -> list[dict]:
    suppression_indices = [
        i for i, w in enumerate(output_windows) if w["regulation_strategy"] == "suppression"
    ]
    if not suppression_indices:
        return []

    suppression_indices.sort(key=lambda i: f2_values[i], reverse=True)

    insights = []
    for i in suppression_indices[:top_n]:
        window = output_windows[i]
        f2 = float(f2_values[i])
        dominant = window.get("dominant_emotion", "neutral")
        context = window.get("context", "general")
        ts = _fmt_ts(window["timestamp_start"])
        insights.append(
            {
                "timestamp": ts,
                "signal": f"Suppression spike — F2: {f2:.2f}, dominant emotion: {dominant}",
                "suggestion": _coaching_suggestion(context, f2),
            }
        )
    return insights


def _fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _coaching_suggestion(context: str, f2: float) -> str:
    if context == "pitch":
        if f2 > 0.75:
            return "High somatic suppression detected. Practice reappraisal framing before valuation and competitive moat questions."
        return "Mild suppression. Rehearse open body language and controlled breathing ahead of investor Q&A."
    if context == "team":
        return "Suppression visible to your team. Briefly naming the emotion ('I want to think carefully about this') preserves psychological safety."
    if context == "manda":
        return "Follow-up diligence question warranted — somatic signal diverged from verbal messaging at this moment."
    return "Consider a reappraisal technique: reframe the triggering question as an opportunity rather than a threat."


def _wellness_flag(strategies: list[str], ers_variability: float) -> bool:
    """Single-session proxy for the multi-session DERS wellness signal.

    Full flag requires 3+ sessions of sustained suppression; here we trigger
    if >50 % of windows are suppression AND within-session ERS variability is
    below 0.15 (flat, chronically suppressed pattern).
    """
    if len(strategies) < 5:
        return False
    ratio = sum(1 for s in strategies if s == "suppression") / len(strategies)
    return ratio > 0.5 and ers_variability < 0.15


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _empty_profile() -> dict:
    return {
        "goleman_eq_profile": {
            "self_awareness": 0,
            "self_regulation": 0,
            "motivation": 0,
            "empathy": 0,
            "social_skills": 0,
        },
        "overall_eq_score": 0,
        "session_summary": {},
        "top_insights": [],
        "wellness_flag": False,
    }
