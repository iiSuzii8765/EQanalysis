from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Root directory containing face images.")
    parser.add_argument("--labels-csv", required=True, help="CSV with columns: image,emotion,valence,arousal")
    parser.add_argument(
        "--openface-csv",
        default=None,
        help="Optional OpenFace frame-level CSV with AU columns to derive AU targets by image key.",
    )
    parser.add_argument("--output", required=True, help="Output manifest CSV path.")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_df = pd.read_csv(args.labels_csv)
    required = {"image", "emotion", "valence", "arousal"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels-csv missing columns: {sorted(missing)}")

    labels_df["image_path"] = labels_df["image"].map(lambda x: str((images_dir / str(x)).resolve()))
    labels_df = labels_df[labels_df["image_path"].map(lambda p: Path(p).exists())].copy()
    if labels_df.empty:
        raise RuntimeError("No valid image paths found from labels CSV.")

    if args.openface_csv:
        openface_df = pd.read_csv(args.openface_csv)
        au_r_cols = sorted([c for c in openface_df.columns if c.startswith("AU") and c.endswith("_r")])
        if len(au_r_cols) == 0:
            raise ValueError("OpenFace CSV missing AU intensity columns (AU*_r).")
        if "frame_name" not in openface_df.columns:
            raise ValueError("OpenFace CSV must include frame_name column to join by image basename.")

        openface_df["image"] = openface_df["frame_name"].map(lambda x: Path(str(x)).name)
        au_group = openface_df.groupby("image")[au_r_cols].mean().reset_index()
        labels_df["image"] = labels_df["image"].map(lambda x: Path(str(x)).name)
        merged = labels_df.merge(au_group, on="image", how="left")
    else:
        merged = labels_df.copy()
        # If AU labels are not provided, initialize zeros; training can still proceed.
        for idx in range(25):
            merged[f"AU{idx:02d}_r"] = 0.0

    # Map available OpenFace AU columns into fixed 25 slots AU_00..AU_24.
    mapped = pd.DataFrame()
    mapped["image_path"] = merged["image_path"]
    mapped["emotion"] = merged["emotion"].astype(int)
    mapped["valence"] = merged["valence"].astype(float)
    mapped["arousal"] = merged["arousal"].astype(float)

    source_aus = {c: merged[c].fillna(0.0).astype(float).to_numpy() for c in merged.columns if c.startswith("AU") and c.endswith("_r")}
    canonical_keys = [
        "AU01_r",
        "AU02_r",
        "AU04_r",
        "AU05_r",
        "AU06_r",
        "AU07_r",
        "AU09_r",
        "AU10_r",
        "AU12_r",
        "AU14_r",
        "AU15_r",
        "AU17_r",
        "AU20_r",
        "AU23_r",
        "AU24_r",
        "AU25_r",
        "AU26_r",
        "AU28_r",
        "AU43_r",
        "AU45_r",
        "AU11_r",
        "AU13_r",
        "AU16_r",
        "AU18_r",
        "AU22_r",
    ]
    row_count = len(mapped)
    for idx, key in enumerate(canonical_keys):
        mapped[f"AU_{idx:02d}"] = source_aus.get(key, np.zeros(row_count, dtype=np.float32))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(output_path, index=False)
    print(f"Saved manifest: {output_path} ({len(mapped)} rows)")


if __name__ == "__main__":
    main()
