from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.config import settings


class OpenFaceExtractionError(RuntimeError):
    """Raised when OpenFace feature extraction fails."""


def run_openface(video_path: str, session_id: str) -> Path:
    """
    Run OpenFace FeatureExtraction on a video and return the generated CSV path.
    """
    binary_path = Path(settings.openface_binary)
    if not binary_path.exists():
        raise OpenFaceExtractionError(
            f"OpenFace binary not found at '{binary_path}'. "
            "Install OpenFace in the worker image or update OPENFACE_BINARY."
        )

    output_dir = settings.openface_output_path / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # OpenFace writes CSV named after the input file stem.
    input_path = Path(video_path)
    csv_path = output_dir / f"{input_path.stem}.csv"

    command = [
        str(binary_path),
        "-f",
        str(input_path),
        "-out_dir",
        str(output_dir),
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

    # Keep outputs deterministic by removing intermediate tracking videos.
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
