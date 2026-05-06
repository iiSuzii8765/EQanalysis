from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def segment_indices(total_frames: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if total_frames <= 0 or window_size <= 0 or stride <= 0:
        return []
    return [(start, start + window_size) for start in range(0, total_frames - window_size + 1, stride)]


def segment_sequence(sequence: Sequence[T], window_size: int, stride: int) -> list[Sequence[T]]:
    return [sequence[start:end] for start, end in segment_indices(len(sequence), window_size, stride)]
