from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, *, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels=3, out_channels=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._block(in_channels=32, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._block(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    @staticmethod
    def _block(*, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)
