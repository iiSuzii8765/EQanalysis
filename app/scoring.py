from __future__ import annotations

from typing import Any

from models.philosophy_module.stage4_rule_based.f1_appraisal import compute_f1_appraisal
from models.philosophy_module.stage4_rule_based.f2_somatic import compute_f2_somatic
from models.philosophy_module.stage4_rule_based.f3_phenomenology import compute_f3_phenomenology
from models.philosophy_module.stage4_rule_based.f4_cognitive_load import compute_f4_cognitive_load


def classify_strategy(f1: float, f2: float, f3: float, f4: float, transition_rate: float, baseline_rate: float) -> str:
    if f2 > 0.6 and f4 > 0.5 and (1 - f3) > 0.5:
        return "suppression"
    if f1 < 0.5 and f2 < 0.4 and f3 > 0.6:
        return "reappraisal"
    if transition_rate > 2 * max(0.2, baseline_rate) and f4 > 0.7:
        return "dysregulation"
    if all(v < 0.3 for v in [f1, f2, f4]) and f3 > 0.6:
        return "neutral"
    return "expression"


def compute_f_scores(window: dict[str, Any], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "F1": compute_f1_appraisal(window),
        "F2": compute_f2_somatic(window),
        "F3": compute_f3_phenomenology(window),
        "F4": compute_f4_cognitive_load(window, baseline),
    }
