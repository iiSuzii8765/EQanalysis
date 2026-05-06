from __future__ import annotations

import torch
from torch import nn


class BayesPhiloNet(nn.Module):
    def __init__(self, dropout_p: float = 0.3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.drop(torch.relu(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))
