from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Canonical Stage 2 mapping used across this repo.
EMOTION_TO_ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "happiness": 3,
    "sad": 4,
    "sadness": 4,
    "surprise": 5,
    "contempt": 6,
    "neutral": 7,
}

# Bootstrap VA targets used elsewhere in this codebase.
VA_MAP = {
    0: (-0.7, 0.7),  # anger
    1: (-0.6, 0.6),  # disgust
    2: (-0.7, 0.8),  # fear
    3: (0.7, 0.6),  # happy
    4: (-0.7, 0.3),  # sad
    5: (0.2, 0.8),  # surprise
    6: (-0.3, 0.4),  # contempt
    7: (0.0, 0.2),  # neutral
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Root with class folders, e.g. images/happy/*.jpg")
    parser.add_argument("--output", required=True, help="Output CSV path with schema image,emotion,valence,arousal")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir does not exist: {images_dir}")

    rows: list[dict[str, str | int | float]] = []
    counts: Counter[int] = Counter()
    skipped_unknown = 0

    for path in images_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(images_dir).as_posix()
        parts = rel.split("/")
        if not parts:
            continue
        label_name = parts[0].strip().lower()
        if label_name not in EMOTION_TO_ID:
            skipped_unknown += 1
            continue

        emotion = EMOTION_TO_ID[label_name]
        valence, arousal = VA_MAP[emotion]
        rows.append(
            {
                "image": rel,
                "emotion": emotion,
                "valence": valence,
                "arousal": arousal,
            }
        )
        counts[emotion] += 1

    if not rows:
        raise RuntimeError("No valid labeled images found from folder names.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "emotion", "valence", "arousal"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows -> {output_path}")
    print("Class counts:", dict(sorted(counts.items())))
    if skipped_unknown:
        print(f"Skipped files in unknown label folders: {skipped_unknown}")


if __name__ == "__main__":
    main()
