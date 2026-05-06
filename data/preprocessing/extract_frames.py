from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_video_frames(video_path: str, target_fps: int) -> tuple[list[np.ndarray], float]:
    """
    Decode a video and subsample frames to approximately target_fps.
    Returns RGB frames and the source fps estimate.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = float(target_fps)

    stride = max(1, int(round(source_fps / target_fps)))
    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if index % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        index += 1

    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return frames, source_fps


def resize_frames(frames: list[np.ndarray], size: tuple[int, int] = (224, 224)) -> list[np.ndarray]:
    return [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames]
