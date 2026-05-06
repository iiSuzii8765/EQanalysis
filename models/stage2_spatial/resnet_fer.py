from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetFER(nn.Module):
    def __init__(self, num_aus: int = 25, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.emotion_head = nn.Linear(in_features, 8)
        self.va_head = nn.Linear(in_features, 2)
        self.au_head = nn.Linear(in_features, num_aus)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "embedding": features,
            "emotion_logits": self.emotion_head(features),
            "va": self.va_head(features),
            "au_logits": self.au_head(features),
        }
