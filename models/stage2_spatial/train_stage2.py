from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.datasets.stage2_dataset import Stage2ImageDataset
from models.stage2_spatial.loss import Stage2CombinedLoss
from models.stage2_spatial.resnet_fer import ResNetFER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", required=True, help="Directory produced by split_stage2_manifest.py")
    parser.add_argument("--output", required=True, help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--class-balanced-loss", action="store_true", help="Use inverse-frequency class weights.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="CrossEntropy label smoothing in [0,1).")
    parser.add_argument("--weighted-sampler", action="store_true", help="Use weighted random sampling for train batches.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lambda-emotion", type=float, default=1.0)
    parser.add_argument("--lambda-va", type=float, default=0.5)
    parser.add_argument("--lambda-au", type=float, default=0.4)
    args = parser.parse_args()
    if not (0.0 <= args.label_smoothing < 1.0):
        raise ValueError("--label-smoothing must be in [0,1).")

    device = torch.device(args.device)
    train_dataset = Stage2ImageDataset(args.splits_dir, split="train", augment=True)
    val_dataset = Stage2ImageDataset(args.splits_dir, split="val", augment=False)
    train_loader = _build_train_loader(train_dataset, args.batch_size, args.weighted_sampler, args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ResNetFER(pretrained=True).to(device)
    emotion_weight = _compute_emotion_weight(train_dataset, device) if args.class_balanced_loss else None
    criterion = Stage2CombinedLoss(
        lambda_emotion=args.lambda_emotion,
        lambda_va=args.lambda_va,
        lambda_au=args.lambda_au,
        emotion_weight=emotion_weight,
        label_smoothing=args.label_smoothing,
    )
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=max(1, args.epochs))

    pos_weight_path = Path(args.splits_dir) / "au_pos_weight.npy"
    au_pos_weight = torch.from_numpy(np.load(pos_weight_path)).float().to(device) if pos_weight_path.exists() else None

    best_state = None
    best_val = float("inf")
    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_confusion_matrix: list[list[int]] | None = None
    stale = 0
    train_history: list[dict] = []

    for epoch in range(args.epochs):
        model.train()
        epoch_train = {"total": 0.0, "emotion": 0.0, "va": 0.0, "au": 0.0, "count": 0}
        for batch_x, batch_emotion, batch_va, batch_au in train_loader:
            batch_x = batch_x.to(device)
            batch_emotion = batch_emotion.to(device)
            batch_va = batch_va.to(device)
            batch_au = batch_au.to(device)
            optim.zero_grad()
            outputs = model(batch_x)
            losses = criterion(outputs, batch_emotion, batch_va, batch_au, au_weight=au_pos_weight)
            losses["total"].backward()
            optim.step()
            epoch_train["total"] += float(losses["total"].item())
            epoch_train["emotion"] += float(losses["emotion"].item())
            epoch_train["va"] += float(losses["va"].item())
            epoch_train["au"] += float(losses["au"].item())
            epoch_train["count"] += 1

        sched.step()
        train_avg = _avg_losses(epoch_train)
        val_avg = _evaluate(model, val_loader, criterion, device, au_pos_weight)
        train_history.append({"epoch": epoch + 1, "train": train_avg, "val": val_avg})
        print(
            f"epoch={epoch + 1} "
            f"train_total={train_avg['total']:.4f} train_emotion={train_avg['emotion']:.4f} "
            f"val_total={val_avg['total']:.4f} val_emotion={val_avg['emotion']:.4f} "
            f"val_acc={val_avg['accuracy']:.4f} val_f1_macro={val_avg['f1_macro']:.4f}"
        )

        is_better_f1 = val_avg["f1_macro"] > best_val_macro_f1 + 1e-5
        is_tie_better_loss = abs(val_avg["f1_macro"] - best_val_macro_f1) <= 1e-5 and val_avg["total"] < best_val - 1e-4
        if is_better_f1 or is_tie_better_loss:
            best_val = val_avg["total"]
            best_val_macro_f1 = val_avg["f1_macro"]
            best_epoch = epoch + 1
            best_state = model.state_dict()
            best_confusion_matrix = val_avg["confusion_matrix"]
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state or model.state_dict(),
            "best_val_total": best_val,
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": best_epoch,
            "best_confusion_matrix": best_confusion_matrix,
            "history": train_history,
            "class_balanced_loss": bool(args.class_balanced_loss),
            "label_smoothing": float(args.label_smoothing),
            "weighted_sampler": bool(args.weighted_sampler),
            "lambda_emotion": float(args.lambda_emotion),
            "lambda_va": float(args.lambda_va),
            "lambda_au": float(args.lambda_au),
        },
        args.output,
    )
    (Path(args.output).with_suffix(".metrics.json")).write_text(
        json.dumps(
            {
                "best_val_total": best_val,
                "best_val_macro_f1": best_val_macro_f1,
                "best_epoch": best_epoch,
                "epochs_ran": len(train_history),
                "last": train_history[-1] if train_history else None,
                "class_balanced_loss": bool(args.class_balanced_loss),
                "label_smoothing": float(args.label_smoothing),
                "weighted_sampler": bool(args.weighted_sampler),
                "lambda_emotion": float(args.lambda_emotion),
                "lambda_va": float(args.lambda_va),
                "lambda_au": float(args.lambda_au),
                "best_confusion_matrix": best_confusion_matrix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Stage2CombinedLoss,
    device: torch.device,
    au_pos_weight: torch.Tensor | None,
) -> dict[str, float]:
    model.eval()
    running = {"total": 0.0, "emotion": 0.0, "va": 0.0, "au": 0.0, "count": 0}
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_emotion, batch_va, batch_au in loader:
            batch_x = batch_x.to(device)
            batch_emotion = batch_emotion.to(device)
            batch_va = batch_va.to(device)
            batch_au = batch_au.to(device)
            outputs = model(batch_x)
            losses = criterion(outputs, batch_emotion, batch_va, batch_au, au_weight=au_pos_weight)
            running["total"] += float(losses["total"].item())
            running["emotion"] += float(losses["emotion"].item())
            running["va"] += float(losses["va"].item())
            running["au"] += float(losses["au"].item())
            running["count"] += 1
            pred = torch.argmax(outputs["emotion_logits"], dim=1)
            all_true.append(batch_emotion.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    metrics = _avg_losses(running)
    if all_true:
        y_true = np.concatenate(all_true).astype(np.int64)
        y_pred = np.concatenate(all_pred).astype(np.int64)
        accuracy = float(np.mean(y_true == y_pred))
        conf = _confusion_matrix(y_true, y_pred, num_classes=8)
        f1_macro, _ = _f1_macro(conf)
    else:
        accuracy = 0.0
        conf = np.zeros((8, 8), dtype=np.int64)
        f1_macro = 0.0
    metrics["accuracy"] = accuracy
    metrics["f1_macro"] = f1_macro
    metrics["confusion_matrix"] = conf.tolist()
    return metrics


def _avg_losses(loss_dict: dict[str, float]) -> dict[str, float]:
    count = max(1, int(loss_dict["count"]))
    return {
        "total": loss_dict["total"] / count,
        "emotion": loss_dict["emotion"] / count,
        "va": loss_dict["va"] / count,
        "au": loss_dict["au"] / count,
    }


def _compute_emotion_weight(dataset: Stage2ImageDataset, device: torch.device) -> torch.Tensor:
    labels = dataset.df["emotion"].astype(int).to_numpy()
    num_classes = max(int(labels.max()) + 1, 8)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    inv = 1.0 / counts
    weights = inv / np.sum(inv) * num_classes
    return torch.from_numpy(weights).float().to(device)


def _build_train_loader(dataset: Stage2ImageDataset, batch_size: int, weighted_sampler: bool, num_workers: int) -> DataLoader:
    if not weighted_sampler:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    labels = dataset.df["emotion"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=8).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    sample_weights = (1.0 / counts[labels]).astype(np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=num_workers)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    return conf


def _f1_macro(conf: np.ndarray) -> tuple[float, list[float]]:
    per_class: list[float] = []
    for c in range(conf.shape[0]):
        tp = float(conf[c, c])
        fp = float(np.sum(conf[:, c]) - tp)
        fn = float(np.sum(conf[c, :]) - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if precision + recall == 0.0:
            per_class.append(0.0)
        else:
            per_class.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(per_class)), per_class


if __name__ == "__main__":
    main()
