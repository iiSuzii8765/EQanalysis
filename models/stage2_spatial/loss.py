from __future__ import annotations

import torch
from torch import nn


class Stage2CombinedLoss(nn.Module):
    def __init__(
        self,
        lambda_emotion: float = 1.0,
        lambda_va: float = 0.5,
        lambda_au: float = 0.8,
        emotion_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_emotion = lambda_emotion
        self.lambda_va = lambda_va
        self.lambda_au = lambda_au
        self.emotion_loss = nn.CrossEntropyLoss(weight=emotion_weight, label_smoothing=label_smoothing)
        self.va_loss = nn.MSELoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        emotion_target: torch.Tensor,
        va_target: torch.Tensor,
        au_target: torch.Tensor,
        au_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        emotion = self.emotion_loss(outputs["emotion_logits"], emotion_target)
        va = self.va_loss(outputs["va"], va_target)
        au = nn.functional.binary_cross_entropy_with_logits(
            outputs["au_logits"],
            au_target,
            pos_weight=au_weight,
        )
        total = self.lambda_emotion * emotion + self.lambda_va * va + self.lambda_au * au
        return {"total": total, "emotion": emotion, "va": va, "au": au}
