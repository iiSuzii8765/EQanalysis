from __future__ import annotations

import pandas as pd


def main() -> None:
    src = pd.read_csv("data/datasets/affectnet/labels.csv")

    label_to_emotion = {
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

    # Placeholder valence/arousal mapping for bootstrap training.
    va_map = {
        0: (-0.7, 0.7),  # anger
        1: (-0.6, 0.6),  # disgust
        2: (-0.7, 0.8),  # fear
        3: (0.7, 0.6),  # happy
        4: (-0.7, 0.3),  # sad
        5: (0.2, 0.8),  # surprise
        6: (-0.3, 0.4),  # contempt
        7: (0.0, 0.2),  # neutral
    }

    df = src.copy()
    df["label_norm"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label_norm"].isin(label_to_emotion)].copy()

    # Convert to Stage 2 expected schema.
    df["image"] = df["pth"].astype(str)
    df["emotion"] = df["label_norm"].map(label_to_emotion).astype(int)
    df["valence"] = df["emotion"].map(lambda e: va_map[e][0]).astype(float)
    df["arousal"] = df["emotion"].map(lambda e: va_map[e][1]).astype(float)

    out = df[["image", "emotion", "valence", "arousal"]]
    out_path = "data/datasets/affectnet/labels_stage2.csv"
    out.to_csv(out_path, index=False)

    print(f"saved {len(out)} rows -> {out_path}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
