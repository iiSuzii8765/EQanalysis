from __future__ import annotations

import torch
from torch import nn


class BiLSTMTemporalModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, num_emotions: int = 8) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.emotion_head = nn.Linear(hidden_dim * 2, num_emotions)
        self.feature_head = nn.Linear(hidden_dim * 2, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs, _ = self.lstm(x)
        attn_logits = self.attention(outputs).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        context = torch.sum(outputs * attn.unsqueeze(-1), dim=1)
        features = self.feature_head(context)
        return {
            "context": context,
            "attention": attn,
            "emotion_logits": self.emotion_head(context),
            "temporal_features": features,
        }
