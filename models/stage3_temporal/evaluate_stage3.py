from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.stage3_temporal.bilstm import BiLSTMTemporalModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensors-dir", required=True, help="Directory with test_features.npy and test_labels.npy")
    parser.add_argument("--checkpoint", required=True, help="Path to stage3 checkpoint.")
    parser.add_argument("--output", required=True, help="Path to write evaluation JSON.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    tensors_dir = Path(args.tensors_dir)
    x_test = torch.from_numpy(np.load(tensors_dir / "test_features.npy")).float()
    y_test = torch.from_numpy(np.load(tensors_dir / "test_labels.npy")).long()
    loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = BiLSTMTemporalModel(input_dim=x_test.shape[-1]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))["emotion_logits"]
            pred = torch.argmax(logits, dim=-1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    num_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
    conf = confusion_matrix(y_true, y_pred, num_classes)
    accuracy = float(np.mean(y_true == y_pred))
    macro_f1 = float(np.mean([f1_for_class(conf, c) for c in range(num_classes)]))

    report = {
        "samples": int(len(y_true)),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(macro_f1, 4),
        "f1_per_class": {str(c): round(float(f1_for_class(conf, c)), 4) for c in range(num_classes)},
        "confusion_matrix": conf.tolist(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    return conf


def f1_for_class(conf: np.ndarray, cls: int) -> float:
    tp = float(conf[cls, cls])
    fp = float(np.sum(conf[:, cls]) - tp)
    fn = float(np.sum(conf[cls, :]) - tp)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


if __name__ == "__main__":
    main()
