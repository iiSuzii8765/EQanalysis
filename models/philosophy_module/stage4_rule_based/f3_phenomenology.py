from __future__ import annotations


def compute_f3_phenomenology(window: dict) -> float:
    spatial = max(0.0, min(1.0, 1.0 - window["au_pairwise_dispersion"]))
    temporal = max(0.0, min(1.0, 1.0 - window["arc_error"]))
    return max(0.0, min(1.0, 0.5 * spatial + 0.5 * temporal))
