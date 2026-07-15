from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.config import settings


class OpenFaceExtractionError(RuntimeError):
    """Raised when face feature extraction fails via both OpenFace and the Stage 1 fallback."""


def run_openface(video_path: str, session_id: str) -> Path:
    """
    Extract facial features from a video and return the generated CSV path.

    Primary path: OpenFace FeatureExtraction binary (writes CSV directly).
    Fallback path: Stage 1 MediaPipe extractor (when binary is unavailable).
    Both paths produce a CSV with the same schema so downstream stages are
    unaffected by which extractor ran.
    """
    binary_path = Path(settings.openface_binary)
    if not binary_path.exists():
        if settings.stage1_use_mediapipe_fallback:
            return _run_stage1_fallback(video_path, session_id)
        raise OpenFaceExtractionError(
            f"OpenFace binary not found at '{binary_path}' and Stage 1 fallback is disabled. "
            "Set STAGE1_USE_MEDIAPIPE_FALLBACK=true or install OpenFace."
        )
    return _run_openface_binary(video_path, session_id, binary_path)


# ---------------------------------------------------------------------------
# Primary: OpenFace binary
# ---------------------------------------------------------------------------

def _run_openface_binary(video_path: str, session_id: str, binary_path: Path) -> Path:
    output_dir = settings.openface_output_path / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(video_path)
    csv_path = output_dir / f"{input_path.stem}.csv"

    command = [
        str(binary_path),
        "-f", str(input_path),
        "-out_dir", str(output_dir),
        "-aus",
        "-pose",
        "-2Dfp",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise OpenFaceExtractionError(
            "OpenFace FeatureExtraction failed. "
            f"stdout='{result.stdout[-1000:]}' stderr='{result.stderr[-1000:]}'"
        )

    if not csv_path.exists():
        csv_candidates = list(output_dir.glob("*.csv"))
        if not csv_candidates:
            raise OpenFaceExtractionError(
                f"OpenFace completed but no CSV output found in '{output_dir}'."
            )
        csv_path = csv_candidates[0]

    for tmp in output_dir.glob("*.avi"):
        tmp.unlink(missing_ok=True)
    for tmp in output_dir.glob("*.hog"):
        tmp.unlink(missing_ok=True)
    for tmp in output_dir.glob("*.txt"):
        tmp.unlink(missing_ok=True)
    for tmp in output_dir.glob("*.bmp"):
        tmp.unlink(missing_ok=True)
    for tmp in output_dir.glob("*.png"):
        tmp.unlink(missing_ok=True)
    shutil.rmtree(output_dir / "aligned", ignore_errors=True)

    return csv_path


# ---------------------------------------------------------------------------
# Fallback: Stage 1 MediaPipe extractor
# ---------------------------------------------------------------------------

def _run_stage1_fallback(video_path: str, session_id: str) -> Path:
    """
    Run the Stage 1 MediaPipe extractor and write its output as a CSV so the
    rest of the pipeline (which expects a file path) is unaffected.
    """
    from data.preprocessing.extract_frames import load_video_frames, resize_frames
    from models.stage1_extraction.inference import Stage1ExtractionConfig, Stage1ExtractionRunner

    frames, _ = load_video_frames(video_path, target_fps=settings.target_fps)
    frames = resize_frames(frames, size=(224, 224))

    runner = Stage1ExtractionRunner(
        Stage1ExtractionConfig(target_fps=settings.target_fps)
    )
    df = runner.extract(frames)

    output_dir = settings.openface_output_path / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{Path(video_path).stem}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
