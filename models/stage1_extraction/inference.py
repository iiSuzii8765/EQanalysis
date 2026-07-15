"""
Stage 1 inference runner: video frames → AU DataFrame.

Wraps the MediaPipe Face Mesh extractor and returns a DataFrame in the
same format as OpenFace FeatureExtraction CSV output, so all downstream
stages are agnostic of how facial features were extracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.stage1_extraction.mediapipe_au import AU_KEYS, extract_aus_from_frames


@dataclass
class Stage1ExtractionConfig:
    target_fps: int = 25
    # Frames where MediaPipe confidence is below this threshold are retained but
    # flagged; the pipeline's own confidence column filtering handles the rest.
    min_face_confidence: float = 0.4
    # Minimum number of frames required after filtering for the runner to succeed.
    min_frames: int = 10


class Stage1ExtractionRunner:
    """
    Stage 1 extraction runner.

    Usage:
        runner = Stage1ExtractionRunner()
        df = runner.extract(frames)   # → DataFrame with AU*_r columns

    The returned DataFrame matches the OpenFace CSV schema so it can be
    passed directly to _load_openface_csv() in the pipeline.
    """

    def __init__(self, config: Stage1ExtractionConfig | None = None) -> None:
        self.config = config or Stage1ExtractionConfig()

    def extract(self, frames: list[np.ndarray]) -> pd.DataFrame:
        df = extract_aus_from_frames(frames, target_fps=self.config.target_fps)
        if df.empty:
            raise RuntimeError(
                "Stage 1 (MediaPipe) extraction produced no output. "
                "Ensure the video contains a visible frontal face."
            )

        # Prefer high-confidence frames; fall back to full DataFrame if too few survive.
        high_conf = df[df["confidence"] >= self.config.min_face_confidence]
        if len(high_conf) >= self.config.min_frames:
            return high_conf.reset_index(drop=True)
        return df.reset_index(drop=True)

    @property
    def au_keys(self) -> list[str]:
        return list(AU_KEYS)
