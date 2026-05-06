from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", required=True, help="Stage 2 split CSV directory.")
    parser.add_argument("--stage2-checkpoint", required=True, help="Path to stage2_resnet.pt")
    parser.add_argument("--output-root", default="artifacts/stage3_training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=75)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument(
        "--sequence-mode",
        choices=["grouped_by_emotion", "sliding"],
        default="grouped_by_emotion",
        help="Sequence construction strategy for Stage 3 tensors.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-balanced-loss", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    tensors_dir = root / "tensors"
    checkpoint_path = root / "stage3_bilstm.pt"
    eval_report_path = root / "stage3_eval.json"
    root.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "data/preprocessing/build_stage3_tensors.py",
            "--splits-dir",
            args.splits_dir,
            "--stage2-checkpoint",
            args.stage2_checkpoint,
            "--output-dir",
            str(tensors_dir),
            "--window-size",
            str(args.window_size),
            "--stride",
            str(args.stride),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--max-samples-per-split",
            str(args.max_samples_per_split),
            "--sequence-mode",
            args.sequence_mode,
            "--seed",
            str(args.seed),
        ]
    )
    run(
        [
            sys.executable,
            "models/stage3_temporal/train_stage3.py",
            "--tensors-dir",
            str(tensors_dir),
            "--output",
            str(checkpoint_path),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--label-smoothing",
            str(args.label_smoothing),
            *(["--class-balanced-loss"] if args.class_balanced_loss else []),
        ]
    )
    if not args.skip_eval:
        run(
            [
                sys.executable,
                "models/stage3_temporal/evaluate_stage3.py",
                "--tensors-dir",
                str(tensors_dir),
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(eval_report_path),
                "--device",
                args.device,
            ]
        )
    print(f"Stage3 checkpoint generated: {checkpoint_path}")


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


if __name__ == "__main__":
    main()
