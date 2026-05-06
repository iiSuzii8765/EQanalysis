from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-tensors-dir", required=True, help="Path to Stage 3 tensors directory.")
    parser.add_argument("--output-root", default="artifacts/stage5_training")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mc-passes", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.output_root)
    dataset_dir = root / "dataset"
    checkpoint_path = root / "stage5_bayes.pt"
    eval_path = root / "stage5_eval.json"
    root.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "data/preprocessing/build_stage5_dataset.py",
            "--stage3-tensors-dir",
            args.stage3_tensors_dir,
            "--output-dir",
            str(dataset_dir),
        ]
    )
    run(
        [
            sys.executable,
            "models/philosophy_module/stage5_bayesian/train_combiner.py",
            "--features",
            str(dataset_dir / "train_features.npy"),
            "--labels",
            str(dataset_dir / "train_labels.npy"),
            "--val-features",
            str(dataset_dir / "val_features.npy"),
            "--val-labels",
            str(dataset_dir / "val_labels.npy"),
            "--output",
            str(checkpoint_path),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
        ]
    )
    run(
        [
            sys.executable,
            "models/philosophy_module/stage5_bayesian/evaluate_stage5.py",
            "--features",
            str(dataset_dir / "test_features.npy"),
            "--labels",
            str(dataset_dir / "test_labels.npy"),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(eval_path),
            "--device",
            args.device,
            "--mc-passes",
            str(args.mc_passes),
        ]
    )
    print(f"Stage5 checkpoint generated: {checkpoint_path}")


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


if __name__ == "__main__":
    main()
