from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output-root", default="artifacts/stage2_training")
    parser.add_argument("--openface-csv", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--class-balanced-loss", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lambda-emotion", type=float, default=1.0)
    parser.add_argument("--lambda-va", type=float, default=0.5)
    parser.add_argument("--lambda-au", type=float, default=0.4)
    args = parser.parse_args()

    root = Path(args.output_root)
    manifest_path = root / "manifest.csv"
    splits_dir = root / "splits"
    checkpoint_path = root / "stage2_resnet.pt"
    root.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "data/preprocessing/build_stage2_manifest.py",
            "--images-dir",
            args.images_dir,
            "--labels-csv",
            args.labels_csv,
            "--output",
            str(manifest_path),
            *([] if args.openface_csv is None else ["--openface-csv", args.openface_csv]),
        ]
    )
    run(
        [
            sys.executable,
            "data/preprocessing/split_stage2_manifest.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(splits_dir),
        ]
    )
    run(
        [
            sys.executable,
            "models/stage2_spatial/train_stage2.py",
            "--splits-dir",
            str(splits_dir),
            "--output",
            str(checkpoint_path),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--patience",
            str(args.patience),
            "--label-smoothing",
            str(args.label_smoothing),
            "--lambda-emotion",
            str(args.lambda_emotion),
            "--lambda-va",
            str(args.lambda_va),
            "--lambda-au",
            str(args.lambda_au),
            *(["--class-balanced-loss"] if args.class_balanced_loss else []),
            *(["--weighted-sampler"] if args.weighted_sampler else []),
        ]
    )
    print(f"Stage2 checkpoint generated: {checkpoint_path}")


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


if __name__ == "__main__":
    main()
